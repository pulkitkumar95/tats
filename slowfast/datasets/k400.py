#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import pandas as pd
from . import utils as utils
from .base_ds import BaseDataset
from .build import DATASET_REGISTRY
import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class K400_cleaned(BaseDataset):
    def __init__(self, cfg, mode):
        super(K400_cleaned, self).__init__(cfg, mode)
    def _construct_loader(self):
        """
        Construct the video loader.
        """
        
        self.data_root = dataroot = self.cfg.DATA.PATH_TO_DATA_DIR
        csv_name_to_use = 'data_csvs/kinetics100.csv' if self.cfg.TASK == 'few_shot' else 'kinetics400_all.csv'
        self.dataset_csv_path = os.path.join(dataroot, csv_name_to_use)
        self.dataset_df = pd.read_csv(self.dataset_csv_path)
        self.base_feature_path = os.path.join(dataroot, self.cfg.POINT_INFO.NAME)
        if self.cfg.TASK == 'few_shot':
            self.dataset_df['video_path'] = self.dataset_df['vid_base_path'].apply(
                                    lambda x: os.path.join(self.data_root, x))
            #MOLO directly evaluated on test, keeping that for now. Will change later
            if self.mode == 'val':
                self.mode == 'test'

        self.split_df = self.dataset_df[self.dataset_df['split'] == self.mode].reset_index(drop=True)
        self._path_to_videos = []
        self.split_df['feat_path'] = self.split_df['feat_base_name'].apply(
                                lambda x: os.path.join(self.base_feature_path, x))
        self._make_final_lists()