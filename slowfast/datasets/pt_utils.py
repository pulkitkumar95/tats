#!/usr/bin/env python3

import logging
import numpy as np
import os
import random
import time
from collections import defaultdict
import cv2
import torch
from iopath.common.file_io import g_pathmgr
from torch.utils.data.distributed import DistributedSampler

from . import transform as transform
from itertools import combinations
from einops import rearrange
import torch.nn.functional as F

logger = logging.getLogger(__name__)
from PIL import Image

def overlay_patches_and_save_gif(images, patch_numbers, grid_size, output_filename, duration=10):
    result_images = []
    
    for img, patch_num in zip(images, patch_numbers):
        # Convert image to BGR if it's grayscale
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        height, width = img.shape[:2]
        rows, cols = grid_size
        
        # Calculate patch dimensions
        patch_height = height // rows
        patch_width = width // cols
        
        # Calculate row and column of the patch
        patch_row = patch_num // cols
        patch_col = patch_num % cols
        
        # Calculate patch coordinates
        start_x = int(patch_col * patch_width)
        start_y = int(patch_row * patch_height)
        end_x = int(start_x + patch_width)
        end_y = int(start_y + patch_height)
        
        # Create a copy of the image
        overlay = img.copy()
        
        # Draw filled red rectangle with alpha=0.5
        cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
        
        # Draw red border
        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
        
        # Convert from BGR to RGB for PIL
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result_images.append(Image.fromarray(img_rgb))
    
    # Save as GIF
    result_images[0].save(output_filename, save_all=True, append_images=result_images[1:], duration=duration, loop=0)
    breakpoint()
    
    return result_images

def create_temporal_mask(cfg, points, pt_visibility_mask):
    # assert points.shape[0] == cfg.DATA.NUM_FRAMES
    temporal_length = points.shape[0] # not using frames due to downsample
    num_patches = cfg.num_patches
    total_st_tokens = num_patches * temporal_length
    mask = torch.zeros(total_st_tokens, total_st_tokens, dtype=torch.bool)
    grid_size = int(num_patches ** 0.5)
    pt_idx_grid = torch.arange(num_patches).view( 1, 1, grid_size, grid_size).float()

    # masking the points which go beyond the image, could be used with query and visibitu too

    points = rearrange(points, 't p d -> p 1 t d').float()
    pt_visibility_mask = rearrange(pt_visibility_mask, 't p -> p t')
    points_in_vis_mask = ((points>-1) & (points<1)).all(dim=-1).squeeze()
    points_in_vis_mask = points_in_vis_mask & pt_visibility_mask
    num_points = points.shape[0]
    pt_idx_grid = pt_idx_grid.repeat(num_points, 1, 1, 1)
    sampled_grid = F.grid_sample(pt_idx_grid, points, align_corners=True, mode='nearest')
    sampled_grid = sampled_grid.squeeze()
    base_sample_grid = sampled_grid.clone()
    base_points = points.clone().squeeze(1)
    add_points = torch.arange(temporal_length).reshape(1, -1) * num_patches
    sampled_grid += add_points
    sampled_grid = sampled_grid.long()
    
    max_unique = 0
    use_pt_idx = -1
    patches_to_take = 0

    for pt_idx in range(num_points):
        point_ids_to_consider = sampled_grid[pt_idx][points_in_vis_mask[pt_idx]]
        unique_patches = torch.unique(point_ids_to_consider%num_patches)
        if len(unique_patches) > max_unique:
            max_unique = len(unique_patches)
            use_pt_idx = pt_idx
            patches_to_take = sampled_grid[pt_idx]

        for i, j in combinations(point_ids_to_consider, 2):
            mask[i, j] = True
            mask[j, i] = True
    # set diagnal to 1
    mask = mask + torch.eye(total_st_tokens, dtype=torch.bool)
    return mask, use_pt_idx, patches_to_take % num_patches, base_sample_grid, points_in_vis_mask, base_points


def get_valid_points(pred_tracks, per_point_queries, pred_visibility, cropped_coords):
    min_x, max_x, min_y, max_y = cropped_coords
    all_point_indices = np.arange(pred_tracks.shape[1]).astype(np.int16)
    init_point_each_point = pred_tracks[per_point_queries, all_point_indices]
    valid_x_points =  (init_point_each_point[:, 0] > min_x) & (init_point_each_point[:, 0] < max_x)
    valid_y_points = (init_point_each_point[:, 1] > min_y) & (init_point_each_point[:, 1] < max_y)
    valid_points = valid_x_points & valid_y_points
    valid_points = valid_points.numpy()
    return valid_points

def process_points(pred_tracks_info, scale_factor, cropped_coords):
    
    pred_tracks = pred_tracks_info['pred_tracks'].clone()
    pred_visibility = pred_tracks_info['pred_visibility'].clone()
    per_point_queries = np.argmax(pred_visibility.numpy(), axis=0)
    pred_tracks = pred_tracks * scale_factor
    points_in_cropped = get_valid_points(pred_tracks, per_point_queries,pred_visibility, cropped_coords)
    new_pred_tracks = pred_tracks[:,points_in_cropped]
    new_pred_visibility = pred_visibility[:,points_in_cropped]

    x_min, x_max, y_min, y_max = cropped_coords
    crop_width = x_max - x_min
    crop_height = y_max - y_min
    crop_scale = np.array([crop_width, crop_height]).reshape(1, 1, 2)
    #crop_size is the final crop size
    move_origin = torch.Tensor([x_min, y_min]).view(1, 1, 2)
    new_pred_tracks = new_pred_tracks - move_origin
    new_pred_tracks = (new_pred_tracks / crop_scale)
    # -1 to 1 normalization
    new_pred_tracks = new_pred_tracks * 2 - 1


    new_pred_tracks_info = {'pred_tracks': new_pred_tracks, 
                            'per_point_queries': per_point_queries, 
                            'pred_visibility': new_pred_visibility}
    return new_pred_tracks_info


def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
    pred_tracks_info=None
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    if pred_tracks_info is not None:
        #TODO(pulkit): Implement points for these two as well 
        assert motion_shift == False
        assert random_horizontal_flip == False
    assert spatial_idx in [-1, 0, 1, 2]
    
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
    
            frames, _, factor_of_points = transform.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
                return_factor=True
            )
            frames, _, cropped_coords = transform.random_crop(frames, crop_size,
                                                              return_coords=True)
        else:
            factor_of_points = 1
            transform_func = (
                transform.random_resized_crop_with_shift
                if motion_shift
                else transform.random_resized_crop
            )
            frames, cropped_coords = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
                return_coords=True,
            )
        if random_horizontal_flip:
            frames, _ = transform.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        # assert len({min_scale, max_scale, crop_size}) == 1
        frames, _, factor_of_points = transform.random_short_side_scale_jitter(
            frames, crop_size, crop_size, return_factor=True
        )
        frames, _, cropped_coords = transform.uniform_crop(frames, 
                                                           crop_size, 
                                                           spatial_idx,
                                                           return_coords=True)
    if pred_tracks_info is not None:
        new_pred_tracks_info = process_points(pred_tracks_info, 
                                              factor_of_points, 
                                              cropped_coords)
        return frames, new_pred_tracks_info

    return frames