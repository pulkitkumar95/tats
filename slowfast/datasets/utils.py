#!/usr/bin/env python3

import logging
import numpy as np
import os
import json
import random
import time
from collections import defaultdict
import cv2
import torch
from torch.utils.data.distributed import DistributedSampler

from slowfast.utils.env import pathmgr
import glob

from . import transform as transform
try: 
    from decord import VideoReader
    DECORD_AVAILABLE = True
except:
    DECORD_AVAILABLE = False

from PIL import Image
from einops import rearrange
import torch.nn.functional as F
from itertools import combinations

logger = logging.getLogger(__name__)

def read_k400_video(video_path, max_fps=10, indices_to_take=None):
   
    vr = VideoReader(video_path)
    fps = np.around(vr.get_avg_fps())
    num_frames = len(vr)
    duration = vr.get_frame_timestamp(num_frames-1)[-1]
    frame_start = 0 
    frame_end = len(vr) - 1
    frames_in_clip = int(duration * max_fps)
    
    frames_to_take = np.linspace(frame_start, frame_end, frames_in_clip).astype(int)
    if indices_to_take is not None:
        frames_to_take = frames_to_take[indices_to_take]
    frames = [vr[i].asnumpy()[None] for i in frames_to_take]
    frames = np.concatenate(frames, axis=0)
    return frames


def read_video(video_path, total_frames, indices_to_take=None):
    global DECORD_AVAILABLE
    duration = -1
    if DECORD_AVAILABLE:
        vr = VideoReader(video_path, num_threads=1)
        fps = np.around(vr.get_avg_fps())
        num_frames = len(vr)
        duration = vr.get_frame_timestamp(num_frames-1)[-1]
        use_cv2 = False
    if not DECORD_AVAILABLE or duration < 0:
        use_cv2 = True
        vr, duration = read_webm_video(video_path, return_duration=True)
    frame_start = 0 
    frame_end = len(vr) - 1
    frames_to_take = np.linspace(frame_start, frame_end, total_frames).astype(int)

    if indices_to_take is not None:
        frames_to_take = frames_to_take[indices_to_take]
    if not use_cv2:
        frames = [vr[i].asnumpy()[None] for i in frames_to_take]
    else:
        frames = [vr[i][None] for i in frames_to_take]
    frames = np.concatenate(frames, axis=0)
    return frames


def retry_load_images(image_paths, retry=10, backend="pytorch"):
    """
    This function is to load images with support of retrying for failed load.

    Args:
        image_paths (list): paths of images needed to be loaded.
        retry (int, optional): maximum time of loading retrying. Defaults to 10.
        backend (str): `pytorch` or `cv2`.

    Returns:
        imgs (list): list of loaded images.
    """
    for i in range(retry):
        imgs = []
        for image_path in image_paths:
            with pathmgr.open(image_path, "rb") as f:
                img_str = np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(img_str, flags=cv2.IMREAD_COLOR)
            imgs.append(img)

        if all(img is not None for img in imgs):
            if backend == "pytorch":
                imgs = torch.as_tensor(np.stack(imgs))
            return imgs
        else:
            logger.warn("Reading failed. Will retry.")
            time.sleep(1.0)
        if i == retry - 1:
            raise Exception("Failed to load images {}".format(image_paths))


def get_sequence(center_idx, half_len, sample_rate, num_frames):
    """
    Sample frames among the corresponding clip.

    Args:
        center_idx (int): center frame idx for current clip
        half_len (int): half of the clip length
        sample_rate (int): sampling rate for sampling frames inside of the clip
        num_frames (int): number of expected sampled frames

    Returns:
        seq (list): list of indexes of sampled frames in this clip.
    """
    seq = list(range(center_idx - half_len, center_idx + half_len, sample_rate))

    for seq_idx in range(len(seq)):
        if seq[seq_idx] < 0:
            seq[seq_idx] = 0
        elif seq[seq_idx] >= num_frames:
            seq[seq_idx] = num_frames - 1
    return seq


def pack_pathway_output(cfg, frames):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
    """
    if cfg.DATA.REVERSE_INPUT_CHANNEL:
        frames = frames[[2, 1, 0], :, :, :]
    if cfg.MODEL.ARCH in cfg.MODEL.SINGLE_PATHWAY_ARCH:
        frame_list = [frames]
    elif cfg.MODEL.ARCH in cfg.MODEL.MULTI_PATHWAY_ARCH:
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // cfg.SLOWFAST.ALPHA
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
    else:
        raise NotImplementedError(
            "Model arch {} is not in {}".format(
                cfg.MODEL.ARCH,
                cfg.MODEL.SINGLE_PATHWAY_ARCH + cfg.MODEL.MULTI_PATHWAY_ARCH,
            )
        )
    return frame_list


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
    boxes=None,
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
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, boxes = transform.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
                boxes=boxes,
            )
            frames, boxes = transform.random_crop(frames, crop_size, boxes=boxes)
        else:
            transform_func = (
                transform.random_resized_crop_with_shift
                if motion_shift
                else transform.random_resized_crop
            )
            if boxes is not None: assert not motion_shift
            frames = transform_func(
                images=frames, boxes=boxes,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
            if boxes is not None: frames, boxes = frames
        if random_horizontal_flip:
            frames, boxes = transform.horizontal_flip(0.5, frames, boxes=boxes)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale}) == 1
        frames, boxes = transform.random_short_side_scale_jitter(
            frames, min_scale, max_scale, boxes=boxes,
        )
        frames, boxes = transform.uniform_crop(frames, crop_size, spatial_idx, boxes=boxes)
    if boxes is not None: return frames, boxes
    return frames


def as_binary_vector(labels, num_classes):
    """
    Construct binary label vector given a list of label indices.
    Args:
        labels (list): The input label list.
        num_classes (int): Number of classes of the label vector.
    Returns:
        labels (numpy array): the resulting binary vector.
    """
    label_arr = np.zeros((num_classes,))

    for lbl in set(labels):
        label_arr[lbl] = 1.0
    return label_arr


def aggregate_labels(label_list):
    """
    Join a list of label list.
    Args:
        labels (list): The input label list.
    Returns:
        labels (list): The joint list of all lists in input.
    """
    all_labels = []
    for labels in label_list:
        for l in labels:
            all_labels.append(l)
    return list(set(all_labels))


def convert_to_video_level_labels(labels):
    """
    Aggregate annotations from all frames of a video to form video-level labels.
    Args:
        labels (list): The input label list.
    Returns:
        labels (list): Same as input, but with each label replaced by
        a video-level one.
    """
    for video_id in range(len(labels)):
        video_level_labels = aggregate_labels(labels[video_id])
        for i in range(len(labels[video_id])):
            labels[video_id][i] = video_level_labels
    return labels


def load_image_lists(frame_list_file, prefix="", return_list=False):
    """
    Load image paths and labels from a "frame list".
    Each line of the frame list contains:
    `original_vido_id video_id frame_id path labels`
    Args:
        frame_list_file (string): path to the frame list.
        prefix (str): the prefix for the path.
        return_list (bool): if True, return a list. If False, return a dict.
    Returns:
        image_paths (list or dict): list of list containing path to eacwhah frame.
            If return_list is False, then return in a dict form.
        labels (list or dict): list of list containing label of each frame.
            If return_list is False, then return in a dict form.
    """
    image_paths = defaultdict(list)
    labels = defaultdict(list)
    with pathmgr.open(frame_list_file, "r") as f:
        assert f.readline().startswith("original_vido_id")
        for line in f:
            row = line.split()
            # original_vido_id video_id frame_id path labels
            assert len(row) == 5
            video_name = row[0]
            if prefix == "":
                path = row[3]
            else:
                path = os.path.join(prefix, row[3])
            image_paths[video_name].append(path)
            frame_labels = row[-1].replace('"', "")
            if frame_labels != "":
                labels[video_name].append(
                    [int(x) for x in frame_labels.split(",")]
                )
            else:
                labels[video_name].append([])

    if return_list:
        keys = image_paths.keys()
        image_paths = [image_paths[key] for key in keys]
        labels = [labels[key] for key in keys]
        return image_paths, labels
    return dict(image_paths), dict(labels)


def ssv2_load_image_lists(frame_list_file, sort_out=[], prefix="", return_list=False):
    """
    Load image paths and labels from a "frame list".
    Each line of the frame list contains:
    `original_vido_id video_id frame_id path labels`
    Args:
        frame_list_file (string): path to the frame list.
        prefix (str): the prefix for the path.
        return_list (bool): if True, return a list. If False, return a dict.
    Returns:
        image_paths (list or dict): list of list containing path to each frame.
            If return_list is False, then return in a dict form.
        labels (list or dict): list of list containing label of each frame.
            If return_list is False, then return in a dict form.
    """
    image_paths = defaultdict(list)
    labels = defaultdict(list)
    with open(frame_list_file, 'rt') as f:
        djson = json.load(f)

    for entry in djson:
        video_name = str(entry['id'])
        if video_name in sort_out: continue
        path = os.path.join(prefix, video_name)
        image_paths[video_name].append(path)
        frame_labels = ""
        labels[video_name].append([])

    if return_list:
        keys = image_paths.keys()
        image_paths = [image_paths[key] for key in keys]
        labels = [labels[key] for key in keys]
        return image_paths, labels
    return dict(image_paths), dict(labels)



def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor

def get_random_sampling_rate(long_cycle_sampling_rate, sampling_rate):
    """
    When multigrid training uses a fewer number of frames, we randomly
    increase the sampling rate so that some clips cover the original span.
    """
    if long_cycle_sampling_rate > 0:
        assert long_cycle_sampling_rate >= sampling_rate
        return random.randint(sampling_rate, long_cycle_sampling_rate)
    else:
        return sampling_rate


def revert_tensor_normalize(tensor, mean, std):
    """
    Revert normalization for a given tensor by multiplying by the std and adding the mean.
    Args:
        tensor (tensor): tensor to revert normalization.
        mean (tensor or list): mean value to add.
        std (tensor or list): std to multiply.
    """
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor * std
    tensor = tensor + mean
    return tensor


def create_sampler(dataset, shuffle, cfg):
    """
    Create sampler for the given dataset.
    Args:
        dataset (torch.utils.data.Dataset): the given dataset.
        shuffle (bool): set to ``True`` to have the data reshuffled
            at every epoch.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        sampler (Sampler): the created sampler.
    """
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None

    return sampler


def loader_worker_init_fn(dataset):
    """
    Create init function passed to pytorch data loader.
    Args:
        dataset (torch.utils.data.Dataset): the given dataset.
    """
    return None

## get edegs
import cv2
import numpy as np
def get_vid_edges(vid):
    # vid: torch.Tensor([T H W C])
    vid = vid.numpy() # T H W C
    if vid.dtype != np.uint8:
        vid = (vid * 255.0).astype(np.uint8)
    edges = np.stack([get_img_edges(img) for img in vid], axis = 0) # T H W
    edges = torch.from_numpy(edges)
    return edges

def get_img_edges(img):
    # img: np.array([H, W, 3])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img,100,200)
    return edges


def read_webm_video(webm_file_path, return_duration=False):
    cap = cv2.VideoCapture(webm_file_path)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Could not open the video file.")
        exit()

    frames = []

    # Read and store each frame in a list
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        #convert to rgb
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame[None])

    # Release the VideoCapture object
    cap.release()

    # Convert the list of frames to a NumPy array
    video_array = np.concatenate(frames, axis=0)
    if return_duration:
        return video_array, duration

    return video_array


def get_seq_frames(video_length, num_frames, mode='train'):
    """
    Given the video index, return the list of sampled frame indexes.
    Args:
        index (int): the video index.
    Returns:
        seq (list): the indexes of frames of sampled from the video.
    """
    
    seg_size = float(video_length) / num_frames
    seq = []
    for i in range(num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1))) - 1
        if start >= end:
            end = start + 1

        if mode == "train":
            seq.append(min(random.randint(start, end), video_length - 1))
        else:
            seq.append((start + end) // 2)
    return seq

def read_ssv2_frames(old_file_path, index_select=None, cfg=None, 
                    num_frames_in_feat=None, mode=None,
                    base_frame_path=None):
    
    vid_name = old_file_path.split('/')[-1]
    frame_path = os.path.join(base_frame_path, vid_name)
    frame_list = np.array(sorted(glob.glob(frame_path + '/*')))

    frame_extraction_fps = 30
    feat_fps = 12
    frame_indices = np.linspace(0, len(frame_list), num_frames_in_feat, endpoint=False).astype(int)
    frame_list = frame_list[frame_indices]
    if index_select is None:
        if mode == 'train' and cfg.DATA.USE_RAND_FRAMES:
            index_select = get_seq_frames(len(frame_list), cfg.DATA.NUM_FRAMES)
        else:
            index_select = np.linspace(0, len(frame_list) - 1, cfg.DATA.NUM_FRAMES).astype(int)
    frame_list = frame_list[index_select]
    frames = []
    for frame in frame_list:
        img = Image.open(frame).convert('RGB')
        img = np.array(img)
        frames.append(img[None])
    frames = np.concatenate(frames, axis=0)
    return frames, torch.Tensor(index_select).long()


def create_temporal_mask(cfg, points):
    assert points.shape[0] == cfg.DATA.NUM_FRAMES
    total_st_tokens = cfg.num_patches * cfg.DATA.NUM_FRAMES
    mask = torch.zeros(total_st_tokens, total_st_tokens, dtype=torch.bool)
    grid_size = int(cfg.num_patches ** 0.5)
    pt_idx_grid = torch.arange(cfg.num_patches).view( 1, 1, grid_size, grid_size).float()

    # masking the points which go beyond the image, could be used with query and visibitu too
    if cfg.MF.USE_POINTS:
        points = rearrange(points, 't p d -> p 1 t d')
        points_in_vis_mask = ((points>=-1) & (points<=1)).all(dim=-1).squeeze()
        num_points = points.shape[0]
        pt_idx_grid = pt_idx_grid.repeat(num_points, 1, 1, 1)
        sampled_grid = F.grid_sample(pt_idx_grid, points, align_corners=True, mode='nearest')
        sampled_grid = sampled_grid.squeeze()
        add_points = torch.arange(cfg.DATA.NUM_FRAMES).reshape(1, -1) * cfg.num_patches
        sampled_grid += add_points
        sampled_grid = sampled_grid.long()
    else:
        
        return 0 # no masking needed

    for pt_idx in range(num_points):
        point_ids_to_consider = sampled_grid[pt_idx][points_in_vis_mask[pt_idx]]
        for i, j in combinations(point_ids_to_consider, 2):
            mask[i, j] = True
            mask[j, i] = True
    # set diagnal to 1
    mask = mask + torch.eye(total_st_tokens, dtype=torch.bool)
    return mask


    





