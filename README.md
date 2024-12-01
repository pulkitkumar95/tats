<div align="center">
<h1>Trajectory-aligned Space-time Tokens for Few-shot Action Recognition</h1>

[**Pulkit Kumar**](https://www.cs.umd.edu/~pulkit/)<sup>1</sup> · [**Namitha Padmanabhan**](https://namithap10.github.io/)<sup>1</sup> · [**Luke Luo**](https://github.com/lluo02)<sup>1</sup> · [**Sai Saketh Rambhatla**](https://rssaketh.github.io)<sup>1,2</sup> · [**Abhinav Shrivastava**](http://www.cs.umd.edu/~abhinav/)<sup>1</sup>

<sup>1</sup>University of Maryland, College Park&emsp;&emsp;&emsp;&emsp;<sup>2</sup>GenAI, Meta

**ECCV 2024**

<a href="https://arxiv.org/abs/2407.18249"><img src='https://img.shields.io/badge/arXiv-TATs-red' alt='Paper PDF'></a>
<a href='https://www.cs.umd.edu/~pulkit/tats/'><img src='https://img.shields.io/badge/Project_Page-TATs-green' alt='Project Page'></a>

<!-- <p float='center'><img src="assets/teaser.png" width="80%" /></p>
<span style="color: green; font-size: 1.3em; font-weight: bold;">LocoTrack is an incredibly efficient model,</span> enabling near-dense point tracking in real-time. It is <span style="color: red; font-size: 1.3em; font-weight: bold;">6x faster</span> than the previous state-of-the-art models. -->
</div>

## Code Release
This repository contains the code for our paper "Trajectory-aligned Space-time Tokens for Few-shot Action Recognition". The code will be released by 8/15/24.

## Installation

1. Create a conda environment using the provided environment file:
```bash
conda env create -f environment.yml
```

2. Activate the environment:
```bash
conda activate tats
```

## Dataset Preparation

Please refer to:
- `slowfast/datasets/DATASET.md` for dataset preparation instructions
- `point_tracking/README.md` for point extraction details

## Training and Testing

Before running the training, set the following environment variables:
```bash
# Path to store PyTorch models and weights
export TORCH_HOME=/path/to/torch/home

# Path to dataset directory containing:
# - Videos
# - Dataset splits
# - Point tracking data
export DATA_DIR=/path/to/data
```

To train the model on SSv2 small, you can use the following command:

```bash
torchrun --nproc_per_node=$NUM_GPUS --master_port=$PORT tools/run_net.py \
    --init_method env:// \
    --new_dist_init \
    --cfg configs/TaTs/ssv2_longer_steps.yaml \
    OUTPUT_DIR $OUTPUT_DIR \
    NUM_GPUS $NUM_GPUS \
    DATA_LOADER.NUM_WORKERS $NUM_WORKERS \
    DATA.PATH_TO_DATA_DIR $DATA_DIR \
    DATA.USE_RAND_AUGMENT True \
    POINT_INFO.NAME cotracker2_16_uniform_8_corrected \
    MODEL.FEAT_EXTRACTOR dino \
    MODEL.DINO_CONFIG dinov2_vitb14 \
    FEW_SHOT.TRAIN_EPISODES $TRAIN_EPISODES \
    FEW_SHOT.K_SHOT $K_SHOT \
    FEW_SHOT.TRAIN_QUERY_PER_CLASS $TRAIN_QUERY_PER_CLASS \
    FEW_SHOT.N_WAY $N_WAY \
    WANDB_STUFF.WANDB_ID $WANDB_ID \
    WANDB_STUFF.EXP_NAME $EXP_NAME \
    SSV2.SPLIT ssv2_small_molo
```

Key parameters:
- `NUM_GPUS`: Number of GPUs to use (e.g., 4)
- `NUM_WORKERS`: Number of data loader workers (e.g., 16)
- `K_SHOT`: Number of support examples per class (e.g., 1)
- `N_WAY`: Number of classes per episode (e.g., 5)
- `TRAIN_EPISODES`: Number of training episodes (e.g., 400)
- `TRAIN_QUERY_PER_CLASS`: Number of query examples per class (e.g., 6)

## Citation

If you find this code and out paper useful for your research, please cite our paper:

```bibtex
@inproceedings{kumar2024trajectory,
  title={Trajectory-aligned Space-time Tokens for Few-shot Action Recognition},
  author={Kumar, Pulkit and Padmanabhan, Namitha and Luo, Luke and Rambhatla, Sai Saketh and Shrivastava, Abhinav},
  booktitle={European Conference on Computer Vision},
  pages={474--493},
  year={2024},
  organization={Springer}
}