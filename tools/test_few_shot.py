#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pprint
import torch
from einops import rearrange
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
from slowfast.datasets import loader
from slowfast.datasets.mixup import MixUp
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, EpochTimer, TrainMeter, ValMeter, eval_extra_metrics,  EPICTrainMeter, EPICValMeter
from slowfast.utils.multigrid import MultigridSchedule
import os
import wandb
import pandas as pd
from fvcore.common.config import CfgNode

def wandb_init_dict(cfg_node, key_list=[]):
    """
    Convert a config node to dictionary.
    """
    if not isinstance(cfg_node, CfgNode):
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = wandb_init_dict(v)
        return cfg_dict

def process_patch_tokens(cfg, support_tokens, query_tokens):
    """
    Process the patch tokens for few shot learning. 
    Ref: https://github.com/alibaba-mmai-research/MoLo/blob/f7f73b6dd8cba446b414b1c47652ab26033bc88e/models/base/few_shot.py#L2552
    args:
        cfg: config
        support_tokens: (num_support, temp_len, num_patches, embed_dim)
        query_tokens: (num_query, temp_len, num_patches, embed_dim)
    """
    #Putting an activation here, may be not needed
    support_tokens = F.relu(support_tokens)
    query_tokens = F.relu(query_tokens)
    
    num_supports = support_tokens.shape[0]
    num_querries = query_tokens.shape[0]
    if not cfg.MODEL.USE_EXTRA_ENCODER:
        if cfg.FEW_SHOT.PATCH_TOKENS_AGG == 'temporal':
            support_tokens = support_tokens.mean(dim=1)
            query_tokens = query_tokens.mean(dim=1)
        elif cfg.FEW_SHOT.PATCH_TOKENS_AGG == 'spatial':
            support_tokens = support_tokens.mean(dim=2)
            query_tokens = query_tokens.mean(dim=2)
        elif cfg.FEW_SHOT.PATCH_TOKENS_AGG == 'no_agg':
            support_tokens = rearrange(support_tokens, 'b t p e -> b (t p) e')
            query_tokens = rearrange(query_tokens, 'b t p e -> b (t p) e')
        else:
            raise NotImplementedError(
                f"Aggregation method {cfg.FEW_SHOT.PATCH_TOKENS_AGG} not implemented")
    
    support_tokens = rearrange(support_tokens, 'b p e -> (b p) e')
    query_tokens = rearrange(query_tokens, 'b p e -> (b p) e')
    sim_matrix = cos_sim(query_tokens, support_tokens)
    dist_matrix = 1 - sim_matrix

    dist_rearranged = rearrange(dist_matrix, '(q qt) (s st) -> q s qt st',q=num_querries, s=num_supports)
    # Take the minimum distance for each query token
    dist_logits = dist_rearranged.min(3)[0].sum(2) + dist_rearranged.min(2)[0].sum(2)
    if cfg.FEW_SHOT.DIST_NORM == 'max_div':
        max_dist = dist_logits.max(dim=1, keepdim=True)[0]
        dist_logits = dist_logits / max_dist
    elif cfg.FEW_SHOT.DIST_NORM == 'max_sub':
        max_dist = dist_logits.max(dim=1, keepdim=True)[0]
        dist_logits = max_dist - dist_logits



    
    return - dist_logits

def cos_sim(x, y, epsilon=0.01):
    """
    Calculates the cosine similarity between the last dimension of two tensors.
    """
    numerator = torch.matmul(x, y.transpose(-1,-2))
    xnorm = torch.norm(x, dim=-1).unsqueeze(-1)
    ynorm = torch.norm(y, dim=-1).unsqueeze(-1)
    denominator = torch.matmul(xnorm, ynorm.transpose(-1,-2)) + epsilon
    dists = torch.div(numerator, denominator)
    return dists


def support_query_split(preds, labels, metadata):
    device = preds.device
    sample_info = np.array(metadata['sample_type'])
    batch_labels = metadata['batch_label']
    support_condition = sample_info=='support'
    support_labels = labels[support_condition]
    support_preds = preds[support_condition]
    support_batch_labels = batch_labels[support_condition]

    # average the support preds for each class
    support_to_take = []
    support_main_label_to_take = []
    support_batch_label_to_take = []
    for label in np.unique(support_batch_labels.cpu().numpy()):
        label_condition = support_batch_labels==label
        label_mean_support = support_preds[label_condition].mean(dim=0, 
                                                                keepdims=True)
        support_main_label = support_labels[label_condition][0]
        support_main_label_to_take.append(support_main_label)
        support_batch_label_to_take.append(label)
        support_to_take.append(label_mean_support)
    
    support_labels = torch.tensor(support_main_label_to_take, device=device)
    support_batch_labels = torch.tensor(support_batch_label_to_take, device=device)
    support_preds = torch.cat(support_to_take, dim=0)


    query_labels = labels[~support_condition]
    query_preds = preds[~support_condition]
    query_batch_labels = batch_labels[~support_condition]
    return_dict = {
        'query_labels':query_labels,
        'query_batch_labels':query_batch_labels,
        'support_labels':support_labels,
        'support_batch_labels':support_batch_labels,
        'support_preds':support_preds,
        'query_preds':query_preds
    }
    return return_dict
    
    to_log = {'TRAIN':[
        'BATCH_SIZE',
        'DATASET'],
        'SSV2':['SPLIT', 'READ_VID'],
        'SOLVER':
        ['BASE_LR',
        'LR_POLICY',
         'MAX_EPOCH',
         'WARMUP_EPOCHS',
         'WARMUP_START_LR',
         'WEIGHT_DECAY', 
         'OPTIMIZING_METHOD',
         'TEMPRATURE'],
         'MF': ['DEPTH','NUM_HEADS', 'HEAD_DROPOUT', 'DROP', 
                'USE_PT_SPACE_POS_EMBED', 'USE_POINTS', 'POINT_INFO_NAME', 'POINT_GRID_SIZE',
                'FWD_BWD', 'PT_ATTENTION', 'POINT_SAMPLING', 'POINTS_TO_SAMPLE', 'USE_PT_VISIBILITY'],
         'WANDB_STUFF':['WANDB_ID', 'EXP_NAME', 'OLD_WANDB_ID'],
         'DATA': ['SAMPLE_RATIO', 'NUM_FRAMES', 'BOTH_DIRECTION','USE_RAND_AUGMENT',
         'USE_RAND_FRAMES'],
         'MODEL': ['FEAT_EXTRACTOR', 'EXTRACTOR_LAYER', 'LOSS_FUNC', 'DINO_CONFIG',
                   'TRAIN_BACKBONE', 'FEAT_EXTRACT_MODE', 'RESNET_TYPE', 'USE_EXTRA_ENCODER',
                   'EXTRA_ENCODER_DEPTH', 'VIT_TYPE', 'RESNET_REDUCE_DIM'],
         'RNG_SEED':[],
         'FEW_SHOT':['N_WAY', 'K_SHOT', 'TRAIN_QUERY_PER_CLASS', 
                    'TEST_QUERY_PER_CLASS', 'TRAIN_EPISODES', 'TEST_EPISODES', 
                    'K_SHOT_TEST', 'PATCH_TOKENS_AGG', 'USE_MODEl', 'DIST_NORM',
                      'TRAIN_OG_EPISODES', 'CLASS_LOSS_LAMBDA',
                      'PT_FIX_SAMPLING_TRAIN', 'PT_FIX_SAMPLING_TEST',
                      'PT_FIX_SAMPLING_STRATEGY', 'PT_FIX_SAMPLING_NUM_POINTS',
                      'Q2S_LOSS_LAMBDA', 'USE_PT_QUERY_MASK'],

           }
    wandb_dict = {}
    for k,v in cfg.items():
        if k in to_log:
            if k == 'RNG_SEED':
                wandb_dict[k] = v
                continue
            for kk,vv in v.items():
                if kk in to_log[k]:
                    wandb_dict[k+'_'+kk] = vv
    return wandb_dict

logger = logging.get_logger(__name__)

def conv_fp16(input):
    return np.float16(np.around(input, 4))






@torch.no_grad()
def test_epoch(val_loader, model, val_meter, cur_epoch, cfg, args, writer=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()
    epoch_top_1_acc_few_shot = []
    epoch_q2s_loss = []
    num_test_classes = len(val_loader.batch_sampler.class_ids)
    confusion_matrix = np.zeros((num_test_classes, num_test_classes))
    all_df = []

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        if cur_iter > len(val_loader):
            break
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            inputs, labels, meta = misc.iter_to_cuda([inputs, labels, meta])

        val_meter.data_toc()

        # for few shot, patch tokens are also returning
        preds, patch_tokens = model(inputs, meta)
        
        patch_support_query_dict = support_query_split(patch_tokens, labels, meta)
        patch_q2s_logits = process_patch_tokens(
                                    cfg, 
                                    patch_support_query_dict['support_preds'], 
                                    patch_support_query_dict['query_preds'])
        q2s_labels = patch_support_query_dict['query_batch_labels']
        q2s_loss = F.cross_entropy(patch_q2s_logits, q2s_labels)
        
            
        # Explicitly declare reduction to mean.
        loss_fun = losses.get_loss_func(cfg)(
            reduction="mean"
        )
        few_shotk_correct = metrics.topks_correct(patch_q2s_logits, 
                                                    q2s_labels, (1, 5))
        few_shot_top1_acc, _ = [
            (x / patch_q2s_logits.size(0)) * 100.0 for x in few_shotk_correct
        ]
        cfg['wandb'].log({
            'iteration': cur_iter,
            'iter_top_1_acc': few_shot_top1_acc.item(),
        })

        if cfg.NUM_GPUS > 1:
            few_shot_top1_acc, q2s_loss = du.all_reduce([few_shot_top1_acc, q2s_loss])
            q2s_loss = du.all_reduce([q2s_loss])[0]

        # Copy the errors from GPU to CPU (sync point).
        few_shot_top1_acc = few_shot_top1_acc.item()
        q2s_loss = q2s_loss.item()
        epoch_q2s_loss.append(q2s_loss)
        epoch_top_1_acc_few_shot.append(few_shot_top1_acc)

        support_labels = patch_support_query_dict['support_labels']
        query_labels = patch_support_query_dict['query_labels']

        if cfg.NUM_GPUS > 1:
            patch_q2s_logits, support_labels, query_labels = du.all_gather(
                [patch_q2s_logits, support_labels, query_labels]
            )
        patch_q2s_logits = patch_q2s_logits.cpu().numpy()
        support_labels = support_labels.cpu().numpy()
        query_labels = query_labels.cpu().numpy()
        pred_query_batch_labels = patch_q2s_logits.argmax(axis=1)
        pred_query_labels = support_labels[pred_query_batch_labels]
        confusion_matrix[query_labels, pred_query_labels] += 1
        batch_df = pd.DataFrame({'y_true':query_labels, 'y_preds':pred_query_labels})
        all_df.append(batch_df)

        val_meter.iter_toc()
        # Update and log stats.
        val_meter.update_stats(
            q2s_loss,
            few_shot_top1_acc,
            inputs[0].size(0)
            * max(
                cfg.NUM_GPUS, 1
            ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
        )

            
        val_meter.update_predictions(preds, labels)
        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    log_dict = {
        'test_q2s_loss': np.mean(epoch_q2s_loss),
        'test_top1_acc_few_shot': np.mean(epoch_top_1_acc_few_shot),
        'epoch': cur_epoch}
    if cfg['wandb']:
        cfg['wandb'].log(log_dict)
    all_df = pd.concat(all_df)
    all_df.to_csv(os.path.join(cfg.OUTPUT_DIR,cfg['csv_dump_name']))
  
    val_meter.reset()


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def build_trainer(cfg):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        model (nn.Module): training model.
        optimizer (Optimizer): optimizer.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validatoin data loader.
        precise_bn_loader (DataLoader): training data loader for computing
            precise BN.
        train_meter (TrainMeter): tool for measuring training stats.
        val_meter (ValMeter): tool for measuring validation stats.
    """
    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = loader.construct_loader(
        cfg, "train", is_precise_bn=True
    )
    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    return (
        model,
        optimizer,
        train_loader,
        val_loader,
        precise_bn_loader,
        train_meter,
        val_meter,
    )




def test_few_shot(cfg, args, local_rank=-1, wandb_run=None):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    if not args.new_dist_init:
        du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed_all(cfg.RNG_SEED)
    torch.backends.cudnn.deterministic = True

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    if wandb_run is not None:
        wandb_instance = wandb_run  
        wandb_instance.define_metric("test*", step_metric="epoch")
        wandb_instance.define_metric("test_top1_acc_few_shot", summary="max")
    else:
        if du.get_rank() == 0:
            wandb_config_dict = wandb_init_dict(cfg)
            wandb_instance = wandb.init(project='TaTs_test_only',config=wandb_config_dict, 
                                        entity="act_seg_pi_umd")
            wandb_instance.define_metric("epoch")
            wandb_instance.define_metric("iteration")

            wandb_instance.define_metric("iter*", step_metric="iteration")

            wandb_instance.define_metric("train*", step_metric="epoch")
            wandb_instance.define_metric("val*", step_metric="epoch")
            wandb_instance.define_metric("test*", step_metric="epoch")

            wandb_instance.define_metric("train_loss", summary="min")
            wandb_instance.define_metric("val_loss", summary="min")
            wandb_instance.define_metric("test_loss", summary="min")
            wandb_instance.define_metric("val_top5_acc", summary="max")
            wandb_instance.define_metric("val_top1_acc", summary="max")
            wandb_instance.define_metric("test_top1_acc_few_shot", summary="max")
        else:
            wandb_instance = None
    cfg['wandb'] = wandb_instance
    cfg['csv_dump_name'] = 'confusion_matrix.csv'
   

    # Init multigrid.
    logger.info("Test with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)


    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)
    cur_epoch = cu.load_test_checkpoint(cfg, model)
    val_loader = loader.construct_loader(cfg, "test") # MOLO uses test set for validation
    val_meter = ValMeter(len(val_loader), cfg)
            

    test_epoch(val_loader, model, val_meter, cur_epoch, cfg, args, None)

   