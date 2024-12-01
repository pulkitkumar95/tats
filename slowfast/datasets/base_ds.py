#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import json
import numpy as np
import os
import random
import glob
from itertools import chain as chain
import torch
import torch.utils.data
from torchvision import transforms
from iopath.common.file_io import g_pathmgr

import slowfast.utils.logging as logging

from .build import DATASET_REGISTRY

from .random_erasing import RandomErasing
from . import utils as utils
from . import pt_utils as pt_utils

import pickle
from torchvision import transforms as pth_transforms
from .point_sampler import point_sampler, get_point_query_mask, temp_seed
from . import autoaugment as autoaugment
import pandas as pd


logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class BaseDataset(torch.utils.data.Dataset):
    """
    Something-Something v2 (SSV2) video loader. Construct the SSV2 video loader,
    then sample clips from the videos. For training and validation, a single
    clip is randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10, sure=False):
        """
        Load Something-Something V2 data (frame paths, labels, etc. ) to a given
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

        self.cfg = cfg
        self.mode = mode
        

        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        self._num_clips = 1
       
        logger.info("Constructing {} dataset...".format(mode))
        self.sure = sure
        self._construct_loader()

        self.aug = False
        self.rand_erase = False
        if self.mode == "train" and self.cfg.AUG.ENABLE:
            self.aug = True
            if self.cfg.AUG.RE_PROB > 0:
                self.rand_erase = True
        if self.cfg.MODEL.FEAT_EXTRACTOR in ['dino', 'resnet', 'vit', 'conv']:

            self.data_mean = [ 0.48145466, 0.4578275, 0.40821073]
            self.data_std = [0.26862954, 0.26130258, 0.27577711]

           
            self.data_transform = pth_transforms.Compose([
                    pth_transforms.Resize((224, 224)),
                        pth_transforms.Normalize(
                    mean=self.data_mean, std=self.data_std) ])

        
        else:
            raise NotImplementedError('Model not supported')

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        raise NotImplementedError('BaseDataset is not implemented')


    def get_seq_frames(self, video_length):
        """
        Given the video index, return the list of sampled frame indexes.
        Args:
            index (int): the video index.
        Returns:
            seq (list): the indexes of frames of sampled from the video.
        """
        num_frames = self.cfg.DATA.NUM_FRAMES

        seg_size = float(video_length) / num_frames
        seq = []
        for i in range(num_frames):
            start = int(np.round(seg_size * i))
            end = int(np.round(seg_size * (i + 1))) - 1
            if start >= end:
                end = start + 1
            if self.mode == "train":
                sampled_frame = random.randint(start, end)
                sampled_frame = min(sampled_frame, video_length - 1)
                seq.append(sampled_frame)
            else:
                seq.append((start + end) // 2)

        return seq

   
    def __getitem__(self, index, unsure=True):
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
        vid_id = index // self._num_clips
        feat_path = self._feat_paths[index]
        pt_dict = pickle.load(open(feat_path, 'rb'))
        pred_tracks = pt_dict['pred_tracks'].squeeze(0)
        pred_visibility = pt_dict['pred_visibility'].squeeze(0)

        per_point_queries = pt_dict['per_point_queries']
        
        # Selecting the frames for the current clip, currently only supports uniform sampling
        index_select = np.linspace(0, pred_tracks.shape[0] - 1, 
                                    self.cfg.DATA.NUM_FRAMES).astype(int)
            
        
        metadata = {}

        video = utils.read_video(self._path_to_videos[index], 
                                    total_frames=pred_tracks.shape[0],
                                    indices_to_take=index_select)
       
                

        if self.cfg.DATA.USE_RAND_AUGMENT and self.mode in ["train"]:
            # Transform to PIL Image
            frames = [transforms.ToPILImage()(frame) for frame in video]
            # Perform RandAugment
            auto_augment_desc = "rand-m20-mstd0.5-inc1"
            aa_params = dict(
                img_mean=tuple([min(255, round(255 * x)) for x in self.data_mean]),
            )
            seed = random.randint(0, 100000000)

            
            frames = [ autoaugment.rand_augment_transform(
                auto_augment_desc, aa_params, seed)(frame) for frame in frames]
            # To Tensor: T H W C
            frames = [np.array(frame) for frame in frames]
            video = np.stack(frames)
    
        video = torch.from_numpy(video)
        video = video.permute(0, 3, 1, 2) / 255 # [T, C, H, W]
        
        max_y, max_x = video.shape[-2:]
        min_dim = min(max_x, max_y)
        # Transforming input video frames with corresponding transformations
        video = self.data_transform(video)
        

        num_points = pred_tracks.shape[1]
        if ((num_points > self.cfg.POINT_INFO.NUM_POINTS_TO_SAMPLE) or 
            (self.cfg.POINT_INFO.PT_FIX_SAMPLING_TRAIN) or
            (self.cfg.POINT_INFO.PT_FIX_SAMPLING_TEST)):
            
            assert self.cfg.POINT_INFO.SAMPLING_TYPE != 'None'
            per_point_queries = pt_dict['per_point_queries']
            filtered_points, _ = point_sampler(self.cfg, pt_dict, pred_tracks.clone(), 
                                                pred_visibility.clone(), 
                                                per_point_queries=per_point_queries, 
                                                points_to_sample=self.cfg.POINT_INFO.NUM_POINTS_TO_SAMPLE, 
                                                sampling_type=self.cfg.POINT_INFO.SAMPLING_TYPE,
                                                index_select=index_select,
                                                split=self.mode)
            
            pred_tracks_to_take = pred_tracks[:, filtered_points]
            pred_visibility_to_take = pred_visibility[:, filtered_points]
            per_point_queries_to_take = per_point_queries[filtered_points]
        else:
            pred_tracks_to_take = pred_tracks
            pred_visibility_to_take = pred_visibility
            per_point_queries_to_take = per_point_queries
            filtered_points = np.ones(num_points, dtype=bool)
            
        if pred_tracks_to_take.shape[1] == self.cfg.POINT_INFO.NUM_POINTS_TO_SAMPLE:
            pred_tracks = pred_tracks_to_take
            pred_visibility = pred_visibility_to_take
            per_point_queries = per_point_queries_to_take
        

        else:
            # Randomly sample points from the original points if not enough points are present
            all_indices = np.argwhere(filtered_points)[:,0]
            points_missing = self.cfg.POINT_INFO.NUM_POINTS_TO_SAMPLE - pred_tracks_to_take.shape[1]
            # randomly sample points from the original points
            set_seed = vid_id if self.mode == 'test' else None
            with temp_seed(set_seed):
                try:
                    random_indices = np.random.choice(all_indices, points_missing, replace=False)
                except:
                    random_indices = np.random.choice(all_indices, points_missing, replace=True)
            pred_tracks = torch.cat([pred_tracks_to_take, pred_tracks[:, random_indices]], dim=1)
            pred_visibility = torch.cat([pred_visibility_to_take, pred_visibility[:, random_indices]], dim=1)
            per_point_queries = np.concatenate([per_point_queries_to_take, per_point_queries[random_indices]])
        
        pt_query_mask = torch.ones_like(pred_visibility, dtype=torch.bool)

        if self.cfg.POINT_INFO.USE_PT_QUERY_MASK:
            pt_query_mask = get_point_query_mask(per_point_queries, pt_query_mask)

        # Normalizing the points between -1 and 1
        
        div_factor = torch.tensor([max_x, max_y]).view(1, 1, 2)
        pred_tracks = pred_tracks / div_factor
        pred_tracks = (pred_tracks - 0.5)/ 0.5

        # Selecting the points for the current clip
        pt_to_take = pred_tracks[index_select].float()
        pred_visibility = pred_visibility[index_select]
        pt_query_mask = pt_query_mask[index_select]

        metadata['pred_tracks'] = pt_to_take
        metadata['pred_visibility'] = pred_visibility
        metadata['pred_query_mask'] = pt_query_mask

        metadata['video_name'] = self._video_names[index]
        metadata['batch_label'] = batch_label
        metadata['sample_type'] = sample_type
      

        if self.cfg.DATA.BOTH_DIRECTION:
            if 'reverse' in self._feat_paths[index]:
                metadata['reverse'] = True
            else:
                metadata['reverse'] = False

        label = self._labels[index]
        vid_id = self.dict_vid_id[index // self._num_clips]

        return video, label, vid_id, metadata
        
    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos


    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)

    def get_frame_path(self, vid_name, frame_idx):
        """
        Load frame
        :param vid_name: video name
        :param frame_idx: index
        :return:
        """
        ppath = os.path.join(os.path.dirname(self.data_root), 'frames', vid_name, '%04d.jpg' % (frame_idx + 1))
        ppath = os.path.join(self.data_root, 'frames', vid_name, '%04d.jpg' % (frame_idx + 1))
        return ppath

    def _frame_to_list_img(self, frames):
        img_list = [
            transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))
        ]
        return img_list

    def _list_img_to_frames(self, img_list):
        img_list = [transforms.ToTensor()(img) for img in img_list]
        return torch.stack(img_list)
    
    def _make_final_lists(self):
        self._path_to_videos_singles = self.split_df['video_path'].tolist()
        self._path_to_videos = []
        self._video_names_singles = self.split_df['vid_id'].tolist()
        self._video_names = []
    
        self._labels_singles = self.split_df['label_id'].tolist()
        self._labels = []
        self._feat_paths_singles = self.split_df['feat_path'].tolist()
        self._feat_paths = []
        self.dict_vid_id_singles = self.split_df['vid_id'].tolist()
        self.dict_vid_id = []
        self.spatial_temporal_index = []
        self._vid_id_to_name = {}
        for index in range(len(self._path_to_videos_singles)):
            for clip_idx in range(self._num_clips):
                self._path_to_videos.append(self._path_to_videos_singles[index])
                self._video_names.append(self._video_names_singles[index])
                self._labels.append(self._labels_singles[index])
                self._feat_paths.append(self._feat_paths_singles[index])
                self.spatial_temporal_index.append(clip_idx)
            self.dict_vid_id.append(index)
            self._vid_id_to_name[index] = self._video_names_singles[index]


@DATASET_REGISTRY.register()
class Finegym(BaseDataset):
    def __init__(self, cfg, mode):
        super(Finegym, self).__init__(cfg, mode)
    def _construct_loader(self):
        """
        Construct the video loader.
        """
        mode_to_take = self.mode

        self.data_root = dataroot = self.cfg.DATA.PATH_TO_DATA_DIR
        if self.cfg.TASK == 'semi_sup':
            csv_extra_str = f'_{self.cfg.DATA.SEMI_SUP_PERC}'
        else:
            csv_extra_str = ''
        self.data_csv_path = os.path.join(dataroot,'data_to_transfer', f'{self.cfg.DATA.SPLIT}{csv_extra_str}.csv')
        self.dataset_df = pd.read_csv(self.data_csv_path)
        self.dataset_df['vid_id'] = self.dataset_df['video_path'].apply(
            lambda x: os.path.basename(x).split('.')[0]
        )
        ds_vid_ids_present = set(self.dataset_df['vid_id'].tolist())
        
        self.base_feature_path = os.path.join(dataroot, self.cfg.POINT_INFO.NAME)
        self.features_present = glob.glob(os.path.join(self.base_feature_path, '*.pkl'))
        self.feat_vid_ids = [os.path.basename(f).split('.')[0] for f in self.features_present]
        self.feat_vid_ids = set(self.feat_vid_ids)
        print('Total dataset videos: ', len(ds_vid_ids_present))
        print('Total feature videos: ', len(self.feat_vid_ids))
        self.dataset_df['feat_exists'] = self.dataset_df['vid_id'].apply(
            lambda x: x in self.feat_vid_ids
        )

    
        if self.cfg.TEST.USE_TRAIN:
            mode_to_take = 'train'
        if self.mode == 'test':
            mode_to_take = 'val'
            
        self.split_df = self.dataset_df[self.dataset_df['split'] == mode_to_take].reset_index(drop=True)
        if mode_to_take == 'train' and self.cfg.TASK == 'semi_sup':
            self.split_df = self.split_df[self.split_df['semi_sup_split']=='labelled']
        self._path_to_videos = []
        self.split_df['feat_path'] = self.split_df['feature_name'].apply(
                                lambda x: os.path.join(self.base_feature_path, x))
        print('Feats stats: ', self.split_df['feat_exists'].value_counts())
        self.split_df = self.split_df[self.split_df['feat_exists'] == True]
        self._make_final_lists()


        
@DATASET_REGISTRY.register()
class Haa500(BaseDataset):
    def __init__(self, cfg, mode):
        super(Haa500, self).__init__(cfg, mode)
    def _construct_loader(self):
        """
        Construct the video loader.
        """
        mode_to_take = self.mode

        self.data_root = dataroot = self.cfg.DATA.PATH_TO_DATA_DIR
        self.data_csv_path = os.path.join(dataroot,'data_to_transfer', f'haa500.csv')
        self.dataset_df = pd.read_csv(self.data_csv_path)
        self.dataset_df['vid_id'] = self.dataset_df['video_path'].apply(
            lambda x: os.path.basename(x).split('.')[0]
        )
        ds_vid_ids_present = set(self.dataset_df['vid_id'].tolist())
        
        self.base_feature_path = os.path.join(dataroot, self.cfg.POINT)
        self.features_present = glob.glob(os.path.join(self.base_feature_path, '*.pkl'))
        self.feat_vid_ids = [os.path.basename(f).split('.')[0] for f in self.features_present]
        self.feat_vid_ids = set(self.feat_vid_ids)
        print('Total dataset videos: ', len(ds_vid_ids_present))
        print('Total feature videos: ', len(self.feat_vid_ids))
        self.dataset_df['feat_exists'] = self.dataset_df['vid_id'].apply(
            lambda x: x in self.feat_vid_ids
        )

    
        if self.cfg.TEST.USE_TRAIN:
            mode_to_take = 'train'
        if self.mode == 'test':
            mode_to_take = 'val'
            
        self.split_df = self.dataset_df[self.dataset_df['split'] == mode_to_take].reset_index(drop=True)
        self._path_to_videos = []
        self.split_df['feat_path'] = self.split_df['feature_name'].apply(
                                lambda x: os.path.join(self.base_feature_path, x))
        print('Feats stats: ', self.split_df['feat_exists'].value_counts())
        self.split_df = self.split_df[self.split_df['feat_exists'] == True]
        self._make_final_lists()


@DATASET_REGISTRY.register()
class Diving48(BaseDataset):
    def __init__(self, cfg, mode):
        super(Diving48, self).__init__(cfg, mode)
    def _construct_loader(self):
        """
        Construct the video loader.
        """
        mode_to_take = self.mode

        self.data_root = dataroot = self.cfg.DATA.PATH_TO_DATA_DIR
        if self.cfg.TASK == 'semi_sup':
            csv_name = 'diving48_{cfg.DATA.SEMI_SUP_PERC}.csv'
        else:
            csv_name = 'diving48.csv'
        
        self.data_csv_path = os.path.join(dataroot,'data_to_transfer', csv_name)
        self.dataset_df = pd.read_csv(self.data_csv_path)
        self.dataset_df['vid_id'] = self.dataset_df['video_path'].apply(
            lambda x: os.path.basename(x).split('.')[0]
        )
        ds_vid_ids_present = set(self.dataset_df['vid_id'].tolist())
        
        self.base_feature_path = os.path.join(dataroot, self.cfg.POINT_INFO.NAME)
        self.features_present = glob.glob(os.path.join(self.base_feature_path, '*.pkl'))
        self.feat_vid_ids = [os.path.basename(f).split('.')[0] for f in self.features_present]
        self.feat_vid_ids = set(self.feat_vid_ids)
        print('Total dataset videos: ', len(ds_vid_ids_present))
        print('Total feature videos: ', len(self.feat_vid_ids))
        self.dataset_df['feat_exists'] = self.dataset_df['vid_id'].apply(
            lambda x: x in self.feat_vid_ids
        )

    
        if self.cfg.TEST.USE_TRAIN:
            mode_to_take = 'train'
        if self.mode == 'test':
            mode_to_take = 'val'
            
        self.split_df = self.dataset_df[self.dataset_df['split'] == mode_to_take].reset_index(drop=True)
        if mode_to_take == 'train' and self.cfg.TASK == 'semi_sup':
            self.split_df = self.split_df[self.split_df['semi_sup_split']=='labelled']

        self._path_to_videos = []
        self.split_df['feat_path'] = self.split_df['feature_name'].apply(
                                lambda x: os.path.join(self.base_feature_path, x))
        print('Feats stats: ', self.split_df['feat_exists'].value_counts())
        self.split_df = self.split_df[self.split_df['feat_exists'] == True]
        self._make_final_lists()
        
        