#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import sys
sys.path = [x for x  in sys.path if not (os.path.isdir(x) and 'slowfast' in os.listdir(x))]
sys.path.append(os.getcwd())

import slowfast
assert slowfast.__file__.startswith(os.getcwd()), f"sys.path: {sys.path}, slowfast.__file__: {slowfast.__file__}"

"""Wrapper to train and test a video classification model."""
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args

from test_net import test
from train_net import train
from train_few_shot import train_few_shot
from test_few_shot import test_few_shot
import wandb
from dist_utils import init_distributed_mode, is_dist_avail_and_initialized



def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    # 
    os.environ["WANDB_RUN_GROUP"] = cfg.WANDB_STUFF.WANDB_ID
    os.environ["WANDB_RESUME"] = "allow"
    os.environ["WANDB_RUN_ID"] = cfg.WANDB_STUFF.WANDB_ID

    if args.new_dist_init:
        args = init_distributed_mode(args)
    else:
        os.environ["MASTER_PORT"] = str(cfg.MASTER_PORT)


    if cfg.DEBUG:
        os.environ["WANDB_MODE"] = "offline"
        os.environ['DEBUG'] = 'True' 
    else:
        os.environ['DEBUG'] = 'False'

    if '$SCRATCH_DIR' in  cfg.DATA.PATH_TO_DATA_DIR:
        cfg.DATA.PATH_TO_DATA_DIR = cfg.DATA.PATH_TO_DATA_DIR.replace('$SCRATCH_DIR', os.environ['SCRATCH_DIR'])
    if cfg.TRAIN.DATASET == 'epickitchens':
        cfg.EPICKITCHENS.VISUAL_DATA_DIR = cfg.EPICKITCHENS.VISUAL_DATA_DIR.replace('$SCRATCH_DIR', os.environ['SCRATCH_DIR'])
        
    if cfg.CUDA_VISIBLE_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CUDA_VISIBLE_DEVICES
    if cfg.TASK == 'classification':
    # Perform training.
        if cfg.TRAIN.ENABLE:
            launch_job(cfg=cfg, init_method=args.init_method, func=train, args=args)

        # Perform multi-clip testing.
        if cfg.TEST.ENABLE:
            launch_job(cfg=cfg, init_method=args.init_method, func=test, args=args)


    elif cfg.TASK == 'few_shot':
        wandb_run = None
        if cfg.TRAIN.ENABLE:
            wandb_run = launch_job(cfg=cfg, init_method=args.init_method, func=train_few_shot, args=args)
        if cfg.TEST.ENABLE:
            launch_job(cfg=cfg, init_method=args.init_method, func=test_few_shot, args=args, wandb_run=wandb_run)



if __name__ == "__main__":
    main()
