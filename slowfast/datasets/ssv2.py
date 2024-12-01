#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import json
import glob
import pandas as pd
from . import utils as utils
from .base_ds import BaseDataset
from .build import DATASET_REGISTRY
import slowfast.utils.logging as logging
from iopath.common.file_io import g_pathmgr


logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Ssv2(BaseDataset):
    def __init__(self, cfg, mode):
        super(Ssv2, self).__init__(cfg, mode)
    def _construct_loader(self):
        """
        Construct the video loader.
        """
        
        self.data_root = dataroot = self.cfg.DATA.PATH_TO_DATA_DIR
        self.video_root = os.path.join(dataroot, '20bn-something-something-v2')
        if os.path.exists(os.path.join(dataroot, 'frames')):
            self.base_frame_path = os.path.join(dataroot, 'frames')
        else:
            self.base_frame_path = '/fs/vulcan-datasets/SomethingV2/frames/'
        
        self.base_feature_path = os.path.join(dataroot, self.cfg.POINT_INFO.NAME)
        self.splits_root = os.path.join(dataroot, 'ssv2_splits')
        self.features_present = glob.glob(os.path.join(self.base_feature_path, '*.pkl'))
        self.feat_vid_ids = [os.path.basename(f).split('.')[0] for f in self.features_present]
        self.feat_vid_ids = set(self.feat_vid_ids)

        # Loading labels.
        data_split = self.mode
        split = self.cfg.SSV2.SPLIT
       
        label_file = os.path.join(self.splits_root, 
                    f'dataset_splits/{split}/{data_split}_few_shot.json')
        with g_pathmgr.open(label_file, "r") as f:
            label_json = json.load(f)
        
        split_df_list = []
    
        for _, video in enumerate(label_json):
            video_name = video["id"]
            label = int(video['label'])
         
            feat_path = os.path.join(self.base_feature_path, f'{video_name}.pkl')
            video_path = os.path.join(self.video_root, f'{video_name}.webm')
            if not (video_name in self.feat_vid_ids) : 
                print(f'{video_name} not in feat_vid_ids')
                continue
            split_df_list.append({'vid_id': video_name, 'video_path': video_path, 
                                  'label_id': label, 'feat_path': feat_path})

        self.split_df = pd.DataFrame(split_df_list)
        self._make_final_lists()
        