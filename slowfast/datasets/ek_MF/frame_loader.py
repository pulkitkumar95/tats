#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import torch
import numpy as np
from .. import utils as utils
from ..decoder import get_start_end_idx

def sample_test_frames_indices(total_frames, num_frmaes_per_clip, num_clips_to_take=10):
    init_indices = np.linspace(0, total_frames - 1, num_clips_to_take+1).astype(int)
    frame_index_list = []
    for idx in range(num_clips_to_take):
        start_idx = init_indices[idx]
        end_idx = init_indices[idx+1]
        frame_indices = np.linspace(start_idx, end_idx-1, num_frmaes_per_clip).astype(int)
        frame_index_list.append(frame_indices)
    frame_index_list = np.hstack(frame_index_list)
    return torch.from_numpy(frame_index_list).long()



def temporal_sampling(
    num_frames, start_idx, end_idx, num_samples, start_frame=0
):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        num_frames (int): number of frames of the trimmed action clip
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
        start_frame (int): starting frame of the action clip in the untrimmed video
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, num_frames - 1).long()
    return start_frame + index


def pack_frames_to_video_clip(
    cfg, video_record, temporal_sample_index, target_fps=60, ret_seq = False,
):
    # Load video by loading its extracted frames
    path_to_video = '{}/{}/rgb_frames/{}'.format(
        cfg.EPICKITCHENS.VISUAL_DATA_DIR,
        video_record.participant,
        video_record.untrimmed_video_name
    
    )
    img_tmpl = "frame_{:010d}.jpg"
    fps = video_record.fps
    sampling_rate = cfg.DATA.SAMPLING_RATE
    num_samples = cfg.DATA.NUM_FRAMES
    start_idx, end_idx = get_start_end_idx(
        video_record.num_frames,
        num_samples * sampling_rate * fps / target_fps,
        temporal_sample_index,
        cfg.TEST.NUM_ENSEMBLE_VIEWS,
    )
    start_idx, end_idx = start_idx + 1, end_idx + 1
    frame_idx = temporal_sampling(
        video_record.num_frames,
        start_idx, end_idx, num_samples,
        start_frame=video_record.start_frame
    )
    img_paths = [
        os.path.join(
            path_to_video, 
            img_tmpl.format(idx.item()
        )) for idx in frame_idx]
    frames = utils.retry_load_images(img_paths)
    if ret_seq:
        return frames, frame_idx
    return frames

def get_frames(cfg, video_record, target_fps=10, mode='train'):
    # Load video by loading its extracted frames
    path_to_video = '{}/{}'.format(
        cfg.EPICKITCHENS.VISUAL_DATA_DIR,
        video_record.untrimmed_video_name
    
    )
    img_tmpl = "frame_{:010d}.jpg"
    vid_fps = video_record.fps
    step = int(vid_fps / target_fps)
    # + 1 below for start and end conforming to previous functionality
    all_clip_frame_indices = np.array(range(video_record.start_frame + 1, video_record.end_frame + 1, step))
    num_frames = len(all_clip_frame_indices)
    if mode == 'val':
        index_select = sample_test_frames_indices(num_frames, 
                                cfg.DATA.NUM_FRAMES, 
                                cfg.DATA.NUM_TEST_CLIPS)
    else:
        index_select = torch.linspace(0, num_frames - 1, 
                                cfg.DATA.NUM_FRAMES).long()
    index_select = index_select.numpy()
    all_clip_frame_indices = all_clip_frame_indices[index_select]


    img_paths = [
        os.path.join(
            path_to_video, 
            img_tmpl.format(idx.item()
        )) for idx in all_clip_frame_indices]
    frames = utils.retry_load_images(img_paths)
    return frames, index_select