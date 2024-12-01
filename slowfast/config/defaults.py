#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Configs."""
from fvcore.common.config import CfgNode

from . import custom_config



# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()
_C.TASK = 'classification'


# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# If True Train the model, else skip training.
_C.TRAIN.ENABLE = True

# Dataset.
_C.TRAIN.DATASET = "kinetics"

# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 64

# Evaluate model on test data every eval period epochs.
_C.TRAIN.EVAL_PERIOD = 10

# Save model checkpoint every checkpoint period epochs.
_C.TRAIN.CHECKPOINT_PERIOD = 10

# Resume training from the latest checkpoint in the output directory.
_C.TRAIN.AUTO_RESUME = True

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

# Checkpoint types include `caffe2` or `pytorch`.
_C.TRAIN.CHECKPOINT_TYPE = "pytorch"

# If True, perform inflation when loading checkpoint.
_C.TRAIN.CHECKPOINT_INFLATE = False

# If True, reset epochs when loading checkpoint.
_C.TRAIN.CHECKPOINT_EPOCH_RESET = False

# If set, clear all layer names according to the pattern provided.
_C.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN = ()  # ("backbone.",)

# If set, replace all layer names according to the pattern provided.
_C.TRAIN.CHECKPOINT_REPLACE_NAME_PATTERN = []

# If True, use FP16 for activations
_C.TRAIN.MIXED_PRECISION = False


_C.TRAIN.VAL_ONLY = False
_C.TRAIN.NO_FWD_PASS = False
_C.TRAIN.NEW_TRAIN = True
# ---------------------------------------------------------------------------- #
# Augmentation options.
# ---------------------------------------------------------------------------- #
_C.AUG = CfgNode()

# Whether to enable randaug.
_C.AUG.ENABLE = False

# Number of repeated augmentations to used during training.
# If this is greater than 1, then the actual batch size is
# TRAIN.BATCH_SIZE * AUG.NUM_SAMPLE.
_C.AUG.NUM_SAMPLE = 1

# Not used if using randaug.
_C.AUG.COLOR_JITTER = 0.4

# RandAug parameters.
_C.AUG.AA_TYPE = "rand-m9-mstd0.5-inc1"

# Sample different augmentations for each frame
_C.AUG.DIFFERENT_AUG_PER_FRAME = False

# Interpolation method.
_C.AUG.INTERPOLATION = "bicubic"

# Probability of random erasing.
_C.AUG.RE_PROB = 0.25

# Random erasing mode.
_C.AUG.RE_MODE = "pixel"

# Random erase count.
_C.AUG.RE_COUNT = 1

# Do not random erase first (clean) augmentation split.
_C.AUG.RE_SPLIT = False

# ---------------------------------------------------------------------------- #
# MipUp options.
# ---------------------------------------------------------------------------- #
_C.MIXUP = CfgNode()

# Whether to use mixup.
_C.MIXUP.ENABLE = False

# Mixup alpha.
_C.MIXUP.ALPHA = 0.8

# Cutmix alpha.
_C.MIXUP.CUTMIX_ALPHA = 1.0

# Probability of performing mixup or cutmix when either/both is enabled.
_C.MIXUP.PROB = 1.0

# Probability of switching to cutmix when both mixup and cutmix enabled.
_C.MIXUP.SWITCH_PROB = 0.5

# Label smoothing.
_C.MIXUP.LABEL_SMOOTH_VALUE = 0.1

# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

# If True test the model, else skip the testing.
_C.TEST.ENABLE = True

# Dataset for testing.
_C.TEST.DATASET = "kinetics"

# Total mini-batch size
_C.TEST.BATCH_SIZE = 8

# Path to the checkpoint to load the initial weight.
_C.TEST.CHECKPOINT_FILE_PATH = ""


# Checkpoint types include `caffe2` or `pytorch`.
_C.TEST.CHECKPOINT_TYPE = "pytorch"
# Path to saving prediction results file.
_C.TEST.SAVE_RESULTS_PATH = ""

_C.TEST.TEST_EPOCH_NUM = -1

_C.MODEL = CfgNode()

# Model architecture.
_C.MODEL.ARCH = "slowfast"

# Model name
_C.MODEL.MODEL_NAME = "SlowFast"

# The number of classes to predict for the model.
_C.MODEL.NUM_CLASSES = 400

# Loss function.
_C.MODEL.LOSS_FUNC = "cross_entropy"

# Model architectures that has one single pathway.
_C.MODEL.SINGLE_PATHWAY_ARCH = ["2d", "c2d", "i3d", "slow", "x3d", "mvit"]

# Model architectures that has multiple pathways.
_C.MODEL.MULTI_PATHWAY_ARCH = ["slowfast"]

# Dropout rate before final projection in the backbone.
_C.MODEL.DROPOUT_RATE = 0.5

# Randomly drop rate for Res-blocks, linearly increase from res2 to res5
_C.MODEL.DROPCONNECT_RATE = 0.0

# The std to initialize the fc layer(s).
_C.MODEL.FC_INIT_STD = 0.01

# Activation layer for the output head.
_C.MODEL.HEAD_ACT = "softmax"

_C.MODEL.LOAD_IN_PRETRAIN = ""

_C.MODEL.FEAT_EXTRACTOR = 'dino'
_C.MODEL.DINO_CONFIG = 'dinov2_vits14'

_C.MODEL.EXTRACTOR_LAYER = ''
_C.MODEL.TRAIN_BACKBONE = False
_C.MODEL.FEAT_EXTRACT_MODE = 'nearest'
_C.MODEL.RESNET_TYPE = 'imagenet'
_C.MODEL.VIT_TYPE = 'in_1k'
_C.MODEL.USE_EXTRA_ENCODER = False
_C.MODEL.EXTRA_ENCODER_DEPTH = 1
_C.MODEL.RESNET_REDUCE_DIM = False




# ---------------------------------------------------------------------------- #
# MF options
# ---------------------------------------------------------------------------- #
_C.MF = CfgNode()
_C.MF.USE_BASE_POS_EMBED = True
_C.MF.USE_PT_SPACE_POS_EMBED = True
_C.MF.USE_POINTS = True
_C.MF.POINT_INFO_NAME = 'cotrack_fps_10_grid_16_tskip_None'
_C.MF.POINT_GRID_SIZE = 16
# Patch-size spatial to tokenize input
_C.MF.PATCH_SIZE = 16

# Patch-size temporal to tokenize input
_C.MF.PATCH_SIZE_TEMP = 2

# Number of input channels
_C.MF.CHANNELS = 3

# Embedding dimension
_C.MF.EMBED_DIM = 768

# Depth of transformer: number of layers
_C.MF.DEPTH = 12

# number of attention heads
_C.MF.NUM_HEADS = 12

# expansion ratio for MLP
_C.MF.MLP_RATIO = 4

# add bias to QKV projection layer
_C.MF.QKV_BIAS = True

# video input
_C.MF.VIDEO_INPUT = True

# temporal resolution i.e. number of frames
_C.MF.TEMPORAL_RESOLUTION = 8

# use MLP classification head
_C.MF.USE_MLP = False

# Dropout rate for
_C.MF.DROP = 0.0

# Stochastic drop rate
_C.MF.DROP_PATH = 0.0

# Dropout rate for MLP head
_C.MF.HEAD_DROPOUT = 0.0

# Dropout rate for positional embeddings
_C.MF.POS_DROPOUT = 0.0

# Dropout rate 
_C.MF.ATTN_DROPOUT = 0.0

# Activation for head
_C.MF.HEAD_ACT = "tanh"

# Use IM pretrained weights
_C.MF.IM_PRETRAINED = True

# Pretrained weights type
_C.MF.PRETRAINED_WEIGHTS = "vit_1k"

# Type of position embedding
_C.MF.POS_EMBED = "separate"

# Self-Attention layer
_C.MF.ATTN_LAYER = "trajectory"

# Approximation type
_C.MF.APPROX_ATTN_TYPE = "none"

# Approximation Dimension
_C.MF.APPROX_ATTN_DIM = 128
_C.MF.FWD_BWD = False
_C.MF.PT_ATTENTION = 'divided_space_time'
_C.MF.USE_PT_VISIBILITY = False
_C.MF.POINT_SAMPLING = "None"
_C.MF.POINTS_TO_SAMPLE = 256
_C.MF.RANDOM_POINTS = False




# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# The path to the data directory.
_C.DATA.SAMPLE_RATIO = 1.0
_C.DATA.SAMPLE_RANDOM_STATE = 42
_C.DATA.NUM_TEST_CLIPS = 1

_C.DATA.DATA_CSV_NAME = ""
_C.DATA.PATH_TO_DATA_DIR = ""

# The separator used between path and label.
_C.DATA.PATH_LABEL_SEPARATOR = " "

# Video path prefix if any.
_C.DATA.PATH_PREFIX = ""

# The number of frames of the input clip.
_C.DATA.NUM_FRAMES = 8
_C.DATA.BOTH_DIRECTION = False

# The video sampling rate of the input clip.
_C.DATA.SAMPLING_RATE = 8

# Eigenvalues for PCA jittering. Note PCA is RGB based.
_C.DATA.TRAIN_PCA_EIGVAL = [0.225, 0.224, 0.229]

# Eigenvectors for PCA jittering.
_C.DATA.TRAIN_PCA_EIGVEC = [
    [-0.5675, 0.7192, 0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948, 0.4203],
]

# If a imdb have been dumpped to a local file with the following format:
# `{"im_path": im_path, "class": cont_id}`
# then we can skip the construction of imdb and load it from the local file.
_C.DATA.PATH_TO_PRELOAD_IMDB = ""

# The mean value of the video raw pixels across the R G B channels.
_C.DATA.MEAN = [0.45, 0.45, 0.45]
# List of input frame channel dimensions.

_C.DATA.INPUT_CHANNEL_NUM = [3, 3]

# The std value of the video raw pixels across the R G B channels.
_C.DATA.STD = [0.225, 0.225, 0.225]

# The spatial augmentation jitter scales for training.
_C.DATA.TRAIN_JITTER_SCALES = [256, 320]

# The relative scale range of Inception-style area based random resizing augmentation.
# If this is provided, DATA.TRAIN_JITTER_SCALES above is ignored.
_C.DATA.TRAIN_JITTER_SCALES_RELATIVE = []

# The relative aspect ratio range of Inception-style area based random resizing
# augmentation.
_C.DATA.TRAIN_JITTER_ASPECT_RELATIVE = []

# If True, perform stride length uniform temporal sampling.
_C.DATA.USE_OFFSET_SAMPLING = False

# Whether to apply motion shift for augmentation.
_C.DATA.TRAIN_JITTER_MOTION_SHIFT = False

# The spatial crop size for training.
_C.DATA.TRAIN_CROP_SIZE = 224

# The spatial crop size for testing.
_C.DATA.TEST_CROP_SIZE = 256

# Input videos may has different fps, convert it to the target video fps before
# frame sampling.
_C.DATA.TARGET_FPS = 30

# Decoding backend, options include `pyav` or `torchvision`
_C.DATA.DECODING_BACKEND = "pyav"

# if True, sample uniformly in [1 / max_scale, 1 / min_scale] and take a
# reciprocal to get the scale. If False, take a uniform sample from
# [min_scale, max_scale].
_C.DATA.INV_UNIFORM_SAMPLE = False

# If True, perform random horizontal flip on the video frames during training.
_C.DATA.RANDOM_FLIP = True

# If True, calculdate the map as metric.
_C.DATA.MULTI_LABEL = False

# Method to perform the ensemble, options include "sum" and "max".
_C.DATA.ENSEMBLE_METHOD = "sum"

# If True, revert the default input channel (RBG <-> BGR).
_C.DATA.REVERSE_INPUT_CHANNEL = False
_C.DATA.USE_RAND_AUGMENT = False
_C.DATA.USE_RAND_FRAMES = False


# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# Base learning rate.
_C.SOLVER.BASE_LR = 0.1

# If positive use different lr for ORViT
_C.SOLVER.ORVIT_BASE_LR = -1.0

# Learning rate policy (see utils/lr_policy.py for options and examples).
_C.SOLVER.LR_POLICY = "cosine"

# Final learning rates for 'cosine' policy.
_C.SOLVER.COSINE_END_LR = 0.0

# Exponential decay factor.
_C.SOLVER.GAMMA = 0.1

# Step size for 'exp' and 'cos' policies (in epochs).
_C.SOLVER.STEP_SIZE = 1

# Steps for 'steps_' policies (in epochs).
_C.SOLVER.STEPS = []

# Learning rates for 'steps_' policies.
_C.SOLVER.LRS = []

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 300

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.SOLVER.NESTEROV = True

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARMUP_FACTOR.
_C.SOLVER.WARMUP_FACTOR = 0.1

# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 0

# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 0.01

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "sgd"

# Base learning rate is linearly scaled with NUM_SHARDS.
_C.SOLVER.BASE_LR_SCALE_NUM_SHARDS = False

# If True, start from the peak cosine learning rate after warm up.
_C.SOLVER.COSINE_AFTER_WARMUP = False

# If True, perform no weight decay on parameter with one dimension (bias term, etc).
_C.SOLVER.ZERO_WD_1D_PARAM = False

# Clip gradient at this value before optimizer update
_C.SOLVER.CLIP_GRAD_VAL = None

# Clip gradient at this norm before optimizer update
_C.SOLVER.CLIP_GRAD_L2NORM = None
_C.SOLVER.EXTRA_ENCODER_LR = 0.0
_C.SOLVER.TEMPRATURE = 1.0

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use (applies to both training and testing).
_C.NUM_GPUS = 1

_C.CUDA_VISIBLE_DEVICES = ''

# Number of machine to use for the job.
_C.NUM_SHARDS = 1

# The index of the current machine.
_C.SHARD_ID = 0

# Output basedir.
_C.OUTPUT_DIR = "./tmp"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 1

# Log period in iters.
_C.LOG_PERIOD = 10

# If True, log the model info.
_C.LOG_MODEL_INFO = False

# Distributed backend.
_C.DIST_BACKEND = "nccl"

_C.SPLIT_QKV_CHECKPOINT = False



# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True

# Enable multi thread decoding.
_C.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE = False




# -----------------------------------------------------------------------------
# SSv2 Dataset options
# -----------------------------------------------------------------------------
_C.SSV2 = CfgNode()

# 
_C.SSV2.DATA_ROOT = '/home/gamir/DER-Roei/datasets/smthsmth'
_C.SSV2.SPLITS_ROOT = '../ORViT/data/ssv2' # Path to the folder that contains the "dataset_splits" dir provided in "Something-Else"
_C.SSV2.SPLIT = 'standard'

# Boxes format: detectron2/annotated
_C.SSV2.BOXES_FORMAT = 'detectron2'
_C.SSV2.READ_VID = 'frames'


# -----------------------------------------------------------------------------
# EPIC-KITCHENS Dataset options
# -----------------------------------------------------------------------------
_C.EPICKITCHENS = CfgNode()

# Path to Epic-Kitchens RGB data directory
_C.EPICKITCHENS.VISUAL_DATA_DIR = "/home/gamir/datasets/datasets/EPIC-KITCHENS/epic-kitchens-download-scripts-master/EPIC-KITCHENS"

# Path to Epic-Kitchens Annotation directory
_C.EPICKITCHENS.ANNOTATIONS_DIR = "/home/gamir/datasets/datasets/EPIC-KITCHENS"

# List of EPIC-100 TRAIN files
_C.EPICKITCHENS.TRAIN_LIST = "EPIC_100_train.pkl"

# List of EPIC-100 VAL files
_C.EPICKITCHENS.VAL_LIST = "EPIC_100_validation.pkl"

# List of EPIC-100 TEST files
_C.EPICKITCHENS.TEST_LIST = "EPIC_100_validation.pkl"

# Testing split
_C.EPICKITCHENS.TEST_SPLIT = "validation"

# Use Train + Val
_C.EPICKITCHENS.TRAIN_PLUS_VAL = False


# ---------------------------------------------------------------------------- #
# Multigrid training options
# See https://arxiv.org/abs/1912.00998 for details about multigrid training.
# ---------------------------------------------------------------------------- #
_C.MULTIGRID = CfgNode()

# Multigrid training allows us to train for more epochs with fewer iterations.
# This hyperparameter specifies how many times more epochs to train.
# The default setting in paper trains for 1.5x more epochs than baseline.
_C.MULTIGRID.EPOCH_FACTOR = 1.5

# Enable short cycles.
_C.MULTIGRID.SHORT_CYCLE = False
# Short cycle additional spatial dimensions relative to the default crop size.
_C.MULTIGRID.SHORT_CYCLE_FACTORS = [0.5, 0.5 ** 0.5]

_C.MULTIGRID.LONG_CYCLE = False
# (Temporal, Spatial) dimensions relative to the default shape.
_C.MULTIGRID.LONG_CYCLE_FACTORS = [
    (0.25, 0.5 ** 0.5),
    (0.5, 0.5 ** 0.5),
    (0.5, 1),
    (1, 1),
]

# While a standard BN computes stats across all examples in a GPU,
# for multigrid training we fix the number of clips to compute BN stats on.
# See https://arxiv.org/abs/1912.00998 for details.
_C.MULTIGRID.BN_BASE_SIZE = 8

# Multigrid training epochs are not proportional to actual training time or
# computations, so _C.TRAIN.EVAL_PERIOD leads to too frequent or rare
# evaluation. We use a multigrid-specific rule to determine when to evaluate:
# This hyperparameter defines how many times to evaluate a model per long
# cycle shape.
_C.MULTIGRID.EVAL_FREQ = 3

# No need to specify; Set automatically and used as global variables.
_C.MULTIGRID.LONG_CYCLE_SAMPLING_RATE = 0
_C.MULTIGRID.DEFAULT_B = 0
_C.MULTIGRID.DEFAULT_T = 0
_C.MULTIGRID.DEFAULT_S = 0

# -----------------------------------------------------------------------------
# Tensorboard Visualization Options

# ---------------------------------------------------------------------------- #
# Few shot options
# ---------------------------------------------------------------------------- #
_C.FEW_SHOT = CfgNode()
_C.FEW_SHOT.N_WAY = 5
_C.FEW_SHOT.K_SHOT = 1

_C.FEW_SHOT.TRAIN_QUERY_PER_CLASS = 6
_C.FEW_SHOT.TEST_QUERY_PER_CLASS = 1
_C.FEW_SHOT.TRAIN_EPISODES = 100000
_C.FEW_SHOT.TEST_EPISODES = 10000
_C.FEW_SHOT.PATCH_TOKENS_AGG = 'spatial'
_C.FEW_SHOT.USE_MODEL = True
_C.FEW_SHOT.DIST_NORM = 'none'
_C.FEW_SHOT.TRAIN_OG_EPISODES = False
_C.FEW_SHOT.CLASS_LOSS_LAMBDA = 1.0
_C.FEW_SHOT.Q2S_LOSS_LAMBDA = 1.0







_C.DEBUG = False
_C.LOCAL_RANK = -1
_C.MASTER_PORT = 8888

_C.WANDB_STUFF = CfgNode()
_C.WANDB_STUFF.WANDB_ID = 'abcdff'
_C.WANDB_STUFF.EXP_NAME = "default"
_C.WANDB_STUFF.WANDB_ID_OLD = ''

_C.POINT_INFO = CfgNode()
_C.POINT_INFO.ENABLE = True
_C.POINT_INFO.GRID_SIZE = 16
_C.POINT_INFO.NAME = ''
_C.POINT_INFO.NUM_POINTS_TO_SAMPLE = 256
_C.POINT_INFO.SAMPLING_TYPE = 'random'
_C.POINT_INFO.PT_FIX_SAMPLING_TRAIN = False
_C.POINT_INFO.PT_FIX_SAMPLING_TEST = False
_C.POINT_INFO.USE_PT_QUERY_MASK = False


# Add custom config with default values.
custom_config.add_custom_config(_C)


def assert_and_infer_cfg(cfg):
    # BN assertions.
    # TRAIN assertions.
    assert cfg.TRAIN.CHECKPOINT_TYPE in ["pytorch", "caffe2"]
    assert cfg.NUM_GPUS == 0 or cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0

    # TEST assertions.
    assert cfg.TEST.CHECKPOINT_TYPE in ["pytorch", "caffe2"]
    assert cfg.NUM_GPUS == 0 or cfg.TEST.BATCH_SIZE % cfg.NUM_GPUS == 0

    # Execute LR scaling by num_shards.
    if cfg.SOLVER.BASE_LR_SCALE_NUM_SHARDS:
        cfg.SOLVER.BASE_LR *= cfg.NUM_SHARDS
        cfg.SOLVER.WARMUP_START_LR *= cfg.NUM_SHARDS
        cfg.SOLVER.COSINE_END_LR *= cfg.NUM_SHARDS

    # General assertions.
    assert cfg.SHARD_ID < cfg.NUM_SHARDS
    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()


if __name__ == "__main__":
    cfg = get_cfg()
    print(cfg)
    cfg_dict = convert_to_dict(cfg)
    print(cfg_dict)
