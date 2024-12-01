#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import traceback
import numpy as np
import random
import pickle
import os
import pandas as pd
import torch
import torch.utils.data
from torchvision import transforms

import slowfast.utils.logging as logging
from slowfast.utils.box_ops import box_xyxy_to_cxcywh, zero_empty_boxes
from slowfast.utils import box_ops

from .build import DATASET_REGISTRY
from .ek_MF.epickitchens_record import EpicKitchensVideoRecord, sample_portion_from_data

# from . import autoaugment as autoaugment
from . import transform as transform
from . import utils as utils
from .ek_MF.frame_loader import pack_frames_to_video_clip, get_frames

from .transform import create_random_augment
from .random_erasing import RandomErasing
from torchvision import transforms as pth_transforms
from .base_transforms import Res3DTransform
from .point_sampler import point_sampler


logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
def Epickitchens(cfg, mode):
    if cfg.ORVIT.ENABLE:
        from slowfast.utils.LinkBoxes.epickitchens import get_ek_boxes
        dboxes = get_ek_boxes(cfg.EPICKITCHENS.VISUAL_DATA_DIR, verbose=True, h5=True)
    else:
        dboxes = None
    return Epickitchens_dataset(cfg, mode, orvit_boxes=dboxes)


class Epickitchens_dataset(torch.utils.data.Dataset):

    def __init__(self, cfg, mode, orvit_boxes=None):

        assert mode in [
            "train",
            "val",
            "test",
            "train+val"
        ], "Split '{}' not supported for EPIC-KITCHENS".format(mode)
        self.cfg = cfg
        self.data_root = dataroot = cfg.DATA.PATH_TO_DATA_DIR
        self.mode = mode
        self.target_fps = cfg.DATA.TARGET_FPS
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val", "train+val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = 1
            # self._num_clips = (
            #         cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            # )

        self.get_orvit_boxes = self.cfg.ORVIT.ENABLE

        self.base_feature_path = os.path.join(dataroot, cfg.MF.POINT_INFO_NAME)

        if getattr(self, 'get_orvit_boxes', False):
            from .ek_MF.epickitchens_record import EKBoxes
            self.ek_boxes = EKBoxes(cfg, boxes=orvit_boxes)
        logger.info("Constructing EPIC-KITCHENS {}...".format(mode))
        self._construct_loader()

        self.aug = False
        self.rand_erase = False
        if self.mode == "train" and self.cfg.AUG.ENABLE:
            self.aug = True
            if self.cfg.AUG.RE_PROB > 0:
                self.rand_erase = True

        if self.cfg.MODEL.FEAT_EXTRACTOR in ['dino', 'resnet','vit']:
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
        train_filename = self.cfg.EPICKITCHENS.TRAIN_LIST
        val_filename = self.cfg.EPICKITCHENS.VAL_LIST

        if self.mode == "train":
            path_annotations_pickle = [
                os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, train_filename)]
        elif self.mode == "val":
            path_annotations_pickle = [
                os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, val_filename)]
        elif self.mode == "test":
            path_annotations_pickle = [
                os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, self.cfg.EPICKITCHENS.TEST_LIST)]
        else:
            path_annotations_pickle = [
                os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, file)
                    for file in [train_filename, val_filename]]

        for file in path_annotations_pickle:
            assert os.path.exists(file), "{} dir not found".format(
                file
            )
        
        self._video_records = []
        self._spatial_temporal_idx = []
        self._feat_paths = []
        self._vid_id_to_name = []
        self.dict_vid_id = []
        self._labels = []
        missing = 0
        take_idx = -1
        curr_idx = 0
        for file in path_annotations_pickle:
            for tup in pd.read_pickle(file).iterrows():
                for idx in range(self._num_clips):
                    rec = EpicKitchensVideoRecord(tup)
                    self._video_records.append(rec)
                    self._spatial_temporal_idx.append(idx)
                    feat_path = os.path.join(self.base_feature_path,
                                                f'{rec.metadata["narration_id"]}.pkl')
                    if rec.metadata["narration_id"] == 'P18_09_0':
                        take_idx = len(self._video_records) - 1
                    self._feat_paths.append(feat_path)
                    self._vid_id_to_name.append(rec.metadata["narration_id"])
                    self.dict_vid_id.append(curr_idx)
                    if self.cfg.TASK == 'few_shot':
                        self._labels.append(rec.few_shot_label)
                    curr_idx += 1

                    
        assert (
                len(self._video_records) > 0
        ), "Failed to load EPIC-KITCHENS split {} from {}".format(
            self.mode, path_annotations_pickle
        )
        logger.info(
            "Constructing epickitchens dataloader (size: {}) from {}".format(
                len(self._video_records), path_annotations_pickle
            )
        )
        self.model_name = self.cfg.MODEL.MODEL_NAME
        

    def __getitem__(self, index):
        if self.model_name == 'Pointformer':
            return self.getitem_pointformer(index)
        else:
            while True:
                try:
                    return self.getitem(index)
                except Exception as e:
                    logger.warn(f"EK dataloader: {traceback.format_exc()}")
                index = (index+1)%self.__len__()
    def getitem_pointformer(self, index):
        if self.cfg.TASK == 'few_shot':
            index, batch_label, sample_type = index
        else:
            sample_type = ''
            batch_label = 0
        
        vid_record = self._video_records[index]
        metadata = {}
        narration_id = vid_record.metadata['narration_id']
        
        # For epic_kichens, indices are coming from function itself.
        video, index_select = get_frames(self.cfg,
                            vid_record, 
                            target_fps=self.target_fps,
                            mode=self.mode
                            )
        video = video.permute(0, 3, 1, 2) / 255 # [T, C, H, W]
        
        max_y, max_x = video.shape[-2:]
        video = self.data_transform(video)
        pt_dict = pickle.load(open(self._feat_paths[index], 'rb'))
        pred_tracks = pt_dict['pred_tracks'].squeeze(0)
        pred_visibility = pt_dict['pred_visibility'].squeeze(0)
        min_dim = min(max_y, max_x)
        num_points = pred_tracks.shape[1]

        #TODO: hack right now, need to figure out why this is happening
        if (index_select >= pred_tracks.shape[0]).any():
            print(f'Issue with {narration_id}')
            print(index)
            index_select[index_select == pred_tracks.shape[0]] -=1


        if num_points > self.cfg.MF.POINTS_TO_SAMPLE:
            assert self.cfg.MF.POINT_SAMPLING != 'None'
            per_point_queries = pt_dict['per_point_queries']
            filtered_points, _ = point_sampler(self.cfg, pt_dict, pred_tracks.clone(), 
                                                pred_visibility.clone(), 
                                                per_point_queries=per_point_queries, 
                                                points_to_sample=self.cfg.MF.POINTS_TO_SAMPLE, 
                                                sampling_type=self.cfg.MF.POINT_SAMPLING,
                                                index_select=index_select,
                                                split=self.mode)
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

        div_factor = torch.tensor([max_x, max_y]).view(1, 1, 2)
        pred_tracks = pred_tracks / div_factor
        pred_tracks = (pred_tracks - 0.5)/ 0.5
        




        metadata['pred_tracks'] = pred_tracks[index_select].float()
        metadata['pred_visibility'] = pred_visibility[index_select]
        metadata['batch_label'] = batch_label
        metadata['sample_type'] = sample_type
        # Sample only num_frames frames from the video.
        if self.cfg.TASK == 'few_shot':    
            label = vid_record.few_shot_label
        else:
            label = vid_record.label
        extra_metadata = vid_record.metadata
        metadata.update(extra_metadata)
        vid_id = self.dict_vid_id[index]


        return video, label, vid_id, metadata


    
    def getitem(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        if self.mode in ["train", "val", "train+val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            if self.cfg.TEST.NUM_SPATIAL_CROPS == 3:
                spatial_sample_index = (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
                )
            elif self.cfg.TEST.NUM_SPATIAL_CROPS == 1:
                spatial_sample_index = 1
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )
        frames, seq = pack_frames_to_video_clip(self.cfg, self._video_records[index], temporal_sample_index, target_fps=self.target_fps, ret_seq = True)
        frames = torch.as_tensor(frames) # [T, H, W, C]
        nid = self._video_records[index].metadata['narration_id']
        if getattr(self, 'get_orvit_boxes', False):
            boxes = self.ek_boxes.get_boxes(self._video_records[index].untrimmed_video_name, seq.tolist(), nid = nid)
        else:
            boxes = None


        # Augmentations
        if self.aug:
            frames = self._aug_frame(
                frames,
                spatial_sample_index,
                min_scale,
                max_scale,
                crop_size,
                boxes = boxes,
            ) # [C, T, H, W] 
            if boxes is not None: frames, boxes = frames
        else:
            # Perform color normalization.
            frames = utils.tensor_normalize(
                frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
            )
            # T H W C -> C T H W.
            frames = frames.permute(3, 0, 1, 2)
            # Perform data augmentation.
            frames = utils.spatial_sampling(
                frames, boxes=boxes,
                spatial_idx=spatial_sample_index,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
                random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            )
            if boxes is not None: frames, boxes = frames
        

        label = self._video_records[index].label
        frames = utils.pack_pathway_output(self.cfg, frames)
        metadata = self._video_records[index].metadata
        if boxes is not None:
            # boxes[boxes < 0] = 0
            # boxes[boxes > 1] = 1
            # boxes = box_xyxy_to_cxcywh(boxes.transpose([1,0,2])) # T, O, 4
            # boxes = zero_empty_boxes(boxes, mode='cxcywh', eps = 0.05)
            boxes = torch.from_numpy(boxes)
            boxes = self.ek_boxes.prepare_boxes(boxes, nid = nid)
            metadata['orvit_bboxes'] = torch.tensor(boxes)

        return frames, label, index, metadata




    def __len__(self):
        return len(self._video_records)

    def spatial_sampling(
            self,
            frames,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=224,
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
        Returns:
            frames (tensor): spatially sampled frames.
        """
        assert spatial_idx in [-1, 0, 1, 2]
        if spatial_idx == -1:
            frames, _ = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames, _ = transform.random_crop(frames, crop_size)
            frames, _ = transform.horizontal_flip(0.5, frames)
        else:
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
            frames, _ = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames, _ = transform.uniform_crop(frames, crop_size, spatial_idx)
        return frames


    def _aug_frame(
        self,
        frames,
        spatial_sample_index,
        min_scale,
        max_scale,
        crop_size,
        boxes=None,
    ):

        aug_transform = create_random_augment(
            input_size=(frames.size(1), frames.size(2)),
            auto_augment=self.cfg.AUG.AA_TYPE,
            interpolation=self.cfg.AUG.INTERPOLATION,
            with_boxes = boxes is not None,
        )
        # T H W C -> T C H W.
        frames = frames.permute(0, 3, 1, 2) # [T, C, H, W]
        list_img = self._frame_to_list_img(frames)
        list_img = aug_transform(list_img, boxes=boxes)
        if boxes is not None:
            list_img, boxes = list_img
        frames = self._list_img_to_frames(list_img)
        frames = frames.permute(0, 2, 3, 1) # [T, H, W, C]

        frames = utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE,
            self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE,
        )
        relative_scales = (
            None if (self.mode not in ["train"] or len(scl) == 0) else scl
        )
        relative_aspect = (
            None if (self.mode not in ["train"] or len(asp) == 0) else asp
        )

        if boxes is not None:
            orig_shape = boxes.shape
            boxes = boxes.reshape([-1, 4])

        frames = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            aspect_ratio=relative_aspect,
            scale=relative_scales,
            motion_shift=self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT
            if self.mode in ["train"]
            else False,
            boxes=boxes,
        )
        
        if boxes is not None:
            frames, boxes = frames
            boxes = boxes.reshape(orig_shape)
        if self.rand_erase:
            erase_transform = RandomErasing(
                self.cfg.AUG.RE_PROB,
                mode=self.cfg.AUG.RE_MODE,
                max_count=self.cfg.AUG.RE_COUNT,
                num_splits=self.cfg.AUG.RE_COUNT,
                device="cpu",
            )
            frames = frames.permute(1, 0, 2, 3)
            frames = erase_transform(frames)
            frames = frames.permute(1, 0, 2, 3)
        if boxes is not None: return frames, boxes
        return frames

    def _frame_to_list_img(self, frames):
        img_list = [
            transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))
        ]
        return img_list

    def _list_img_to_frames(self, img_list):
        img_list = [transforms.ToTensor()(img) for img in img_list]
        return torch.stack(img_list)

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._video_records)