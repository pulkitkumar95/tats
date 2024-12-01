#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import json
import pandas as pd
import numpy as np
import os
import random
import glob
from itertools import chain as chain
import torch
import torch.utils.data

import slowfast.utils.logging as logging

from .build import DATASET_REGISTRY

from . import utils as utils
import pickle
from torchvision import transforms as pth_transforms
from .point_sampler import point_sampler


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

    



logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Ucf101(torch.utils.data.Dataset):
    """
    UCF 1-1 video loader. Construct the 101 video loader,
    then sample clips from the videos. For training and validation, a single
    clip is randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10, sure=False):
        """
        Load UCF data (frame paths, labels, etc. ) to a given
        Dataset object. The dataset could be downloaded from Something-Something
        official website (https://20bn.com/datasets/something-something).
        Please see datasets/DATASET.md for more information about the data format.
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries for reading frames from disk.
        """


        
        
        self.data_root = dataroot = cfg.DATA.PATH_TO_DATA_DIR
        csv_name = 'few_shot_main_data_csvs/ucf_few_shot.csv' if cfg.TASK == 'few_shot' else  cfg.DATA.DATA_CSV_NAME 
        self.data_df = pd.read_csv(os.path.join(dataroot, csv_name))
        self.video_root = os.path.join(dataroot, 'videos')
        self.base_feature_path = os.path.join(dataroot, cfg.MF.POINT_INFO_NAME)
        print(dataroot)
        self.features_present = glob.glob(os.path.join(self.base_feature_path, '*.pkl'))
        self.feat_vid_ids = [os.path.basename(f).split('.')[0] for f in self.features_present]
        self.feat_vid_ids = set(self.feat_vid_ids)
        
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for UCF".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing UCF {}...".format(mode))
        self._construct_loader()

        self.aug = False
        self.rand_erase = False
        if self.mode == "train" and self.cfg.AUG.ENABLE:
            self.aug = True
            if self.cfg.AUG.RE_PROB > 0:
                self.rand_erase = True
        if self.cfg.MODEL.FEAT_EXTRACTOR in ['dino', 'resnet', 'vit']:
            self.data_transform = pth_transforms.Compose([
                pth_transforms.Resize((224, 224)),
                pth_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225])
            ])
        elif self.cfg.MODEL.FEAT_EXTRACTOR == 'slow_r50':
            self.data_transform = Res3DTransform(cfg)
        else:
            raise NotImplementedError('Feat extractor not supported')

    def _construct_loader(self):
        """
        Construct the video loader.
        """

        # Loading labels.
        data_split = self.mode
        if data_split == "val":
            data_split = "test"
        
        self.split_df = self.data_df[self.data_df['split'] == data_split]
        self.split_df = self.split_df.reset_index(drop=True)

        #TODO: For semisupervised case, add an extra label for supervised or not 
        if data_split == 'train':
            if self.cfg.DATA.SAMPLE_RATIO < 1.0:
                random_state_seed = self.cfg.DATA.SAMPLE_RANDOM_STATE
                self.split_df = self.split_df.sample(
                                            frac=self.cfg.DATA.SAMPLE_RATIO,
                                            random_state=random_state_seed)
                self.split_df = self.split_df.reset_index(drop=True)

        
        
        self.split_df['feat_path'] = self.split_df['video_name'].apply(
                                lambda x: os.path.join(self.base_feature_path, 
                                                                    f'{x}.pkl'))
        self.split_df['feat_exists'] = self.split_df['feat_path'].apply(
                                                    lambda x: os.path.exists(x))

     
        assert len(self.split_df) > 0, f'No videos found for split {data_split}'
        print(f'Found {len(self.split_df)} videos for split {data_split},' 
                            f'{self.split_df.feat_exists.sum()} have features') 
        self.split_df = self.split_df[self.split_df['feat_exists']].reset_index(
                                                                    drop=True)
        self.split_df['video_path'] = self.split_df['vid_base_name'].apply(
                                    lambda x: os.path.join(self.data_root, x))


        self._video_names = self.split_df['video_name'].tolist()
        self._labels =  self.split_df['label'].tolist()
        self._path_to_videos = self.split_df['video_path'].tolist()
        self._feat_paths = self.split_df['feat_path'].tolist()

    
        
        logger.info("UCF dataloader constructed ")

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._video_names)
        


    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video frames can be fetched.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): the index of the video.
        """
        if self.cfg.TASK == 'few_shot':
            index, batch_label, sample_type = index
        else:
            sample_type = ''
            batch_label = 0

        feat_path = self._feat_paths[index]
        pt_dict = pickle.load(open(feat_path, 'rb'))
        pred_tracks = pt_dict['pred_tracks'].squeeze(0)
        pred_visibility = pt_dict['pred_visibility'].squeeze(0)
        index_select = torch.linspace(0, pred_tracks.shape[0] - 1, 
                                    self.cfg.DATA.NUM_FRAMES).long()
        


        metadata = {}
        short_cycle_idx = None
        #Videos are being loaded at 10 fps on default
        video = utils.read_video(self._path_to_videos[index], 
                                      indices_to_take=index_select.numpy())
        video = torch.from_numpy(video)
        video = video.permute(0, 3, 1, 2) / 255 # [T, C, H, W]
        # Sampling frames from the video at regular interval
        
        max_y, max_x = video.shape[-2:]
        # Transforming input video frames with dino transformations
        video = self.data_transform(video)
        # Loading points 

        num_points = pred_tracks.shape[1]
        min_dim = min(max_x, max_y)
        if num_points > self.cfg.MF.POINTS_TO_SAMPLE:
            assert self.cfg.MF.POINT_SAMPLING != 'None'
            per_point_queries = pt_dict['per_point_queries']
            try: 
                filtered_points, _ = point_sampler(self.cfg, pt_dict, pred_tracks.clone(), 
                                                pred_visibility.clone(), 
                                                per_point_queries=per_point_queries, 
                                                points_to_sample=self.cfg.MF.POINTS_TO_SAMPLE, 
                                                sampling_type=self.cfg.MF.POINT_SAMPLING,
                                                index_select=index_select,
                                                split=self.mode)
            except:
                print('Error in point sampling')
                print('Video name: ', self._video_names[index])
                print(index)
                assert 1==2
            pred_tracks_to_take = pred_tracks[:, filtered_points]
            pred_visibility_to_take = pred_visibility[:, filtered_points]
            if pred_tracks_to_take.shape[1] == self.cfg.FEW_SHOT.PT_FIX_SAMPLING_NUM_POINTS:
                pred_tracks = pred_tracks_to_take
                pred_visibility = pred_visibility_to_take
            else:
                print('Error in point sampling, hacking it')
                print('Video name: ', self._video_names[index])
                all_indices = np.argwhere(filtered_points)[:,0]
                points_missing = self.cfg.FEW_SHOT.PT_FIX_SAMPLING_NUM_POINTS - pred_tracks_to_take.shape[1]
                # randomly sample points from the original points
                random_indices = np.random.choice(all_indices, points_missing, replace=False)
                pred_tracks = torch.cat([pred_tracks_to_take, pred_tracks[:, random_indices]], dim=1)
                pred_visibility = torch.cat([pred_visibility_to_take, pred_visibility[:, random_indices]], dim=1)
        
        if self.cfg.MF.FWD_BWD:
            rev_feat_path = os.path.join(self.rev_base_feature_path, 
                                            f'{self._video_names[index]}.pkl')
            rev_pt_dict = pickle.load(open(rev_feat_path, 'rb'))
            rev_pred_tracks = rev_pt_dict['pred_tracks'].squeeze(0)
            rev_pred_visibility = rev_pt_dict['pred_visibility'].squeeze(0)
            pred_tracks = torch.cat([pred_tracks, rev_pred_tracks], dim=1)
            pred_visibility = torch.cat([pred_visibility, rev_pred_visibility], dim=1)

        # Normalizing the points between -1 and 1
        div_factor = torch.tensor([max_x, max_y]).view(1, 1, 2)
        pred_tracks = pred_tracks / div_factor
        pred_tracks = (pred_tracks - 0.5)/ 0.5
        metadata['pred_tracks'] = pred_tracks[index_select].float()
        metadata['pred_visibility'] = pred_visibility[index_select]
        metadata['video_name'] = self._video_names[index]
        if self.cfg.DATA.BOTH_DIRECTION:
            if 'reverse' in self._feat_paths[index]:
                metadata['reverse'] = True
            else:
                metadata['reverse'] = False
        metadata['batch_label'] = batch_label
        metadata['sample_type'] = sample_type
        # Sample only num_frames frames from the video.
        label = self._labels[index]
        return video, label, index, metadata
