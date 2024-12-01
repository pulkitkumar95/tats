#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

import numpy as np
import pprint
import torch
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
import time
import sys
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

logger = logging.get_logger(__name__)

def conv_fp16(input):
    return np.float16(np.around(input, 4))



def train_epoch(
    train_loader,
    model,
    optimizer,
    scaler,
    train_meter,
    cur_epoch,
    cfg,
    args,
    writer=None,
    wandb_run=None,
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)
    if cfg.DEBUG:
        data_size = 30

    epoch_loss = []
    epoch_top_1_err = []
    epoch_top_5_err = []
    epoch_noun_loss = []
    epoch_verb_loss = []
    epoch_noun_top_1_acc = []
    epoch_noun_top_5_acc = []
    epoch_verb_top_1_acc = []
    epoch_verb_top_5_acc = []
    epoch_top_1_acc = []
    epoch_top_5_acc = []
    wandb_log = wandb_run is not None


    if cfg.MIXUP.ENABLE:
        mixup_fn = MixUp(
            mixup_alpha=cfg.MIXUP.ALPHA,
            cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA,
            mix_prob=cfg.MIXUP.PROB,
            switch_prob=cfg.MIXUP.SWITCH_PROB,
            label_smoothing=cfg.MIXUP.LABEL_SMOOTH_VALUE,
            num_classes=misc.get_num_classes(cfg) #{'noun': 300, 'verb': 97} if cfg.TRAIN.DATASET == 'epickitchens' else cfg.MODEL.NUM_CLASSES,
        )
    lr = optim.get_epoch_lr(cur_epoch, cfg)
    optim.set_lr(optimizer, lr, log=True)
    start_time = time.time()
    num_bathes = len(train_loader)
    for cur_iter, (inputs, labels, _vid_idx, meta) in enumerate(train_loader):

        if cfg.TRAIN.NO_FWD_PASS:
            if cur_iter %50 == 0:
                print(f'{cur_iter}/ {num_bathes}')
            continue
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            inputs, labels, meta = misc.iter_to_cuda([inputs, labels, meta])
        # Update the learning rate.
        

        train_meter.data_toc()
        if cfg.MIXUP.ENABLE:
            samples, labels = mixup_fn(inputs[0], labels)
            inputs[0] = samples
        

        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            if cfg.DETECTION.ENABLE:
                preds = model(inputs, meta, bboxes=meta["boxes"])
            else:
                preds = model(inputs, meta)
            preds = preds / cfg.SOLVER.TEMPRATURE
            
            if isinstance(preds, tuple): preds, extra_preds = preds
            else: extra_preds = None
                
            # Explicitly declare reduction to mean.
            loss_fun = losses.get_loss_func(cfg)(
                reduction="mean"
            )
            # Compute the loss.
            if cfg.TRAIN.DATASET == 'epickitchens':
                loss_dict = loss_fun(extra_preds, labels)
                loss = loss_dict['verb_loss'] + loss_dict['noun_loss']
            else:
                loss = loss_fun(preds, labels)
                loss_dict = {'loss':loss}
        # check Nan Loss.
        misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)
        # Clip gradients if necessary
        if cfg.SOLVER.CLIP_GRAD_VAL:
            torch.nn.utils.clip_grad_value_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_VAL
            )
        elif cfg.SOLVER.CLIP_GRAD_L2NORM:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM
            )

        # Update the parameters.
        scaler.step(optimizer)
        scaler.update()

        if cfg.MIXUP.ENABLE:
            is_dict_label = isinstance(labels, dict)
            if is_dict_label:
                dict_labels = labels
                dict_preds = extra_preds
            else:
                dict_labels = {"class":labels}
                dict_preds = {"class": preds}
            for k,labels in dict_labels.items():
                preds = dict_preds[k]
                _top_max_k_vals, top_max_k_inds = torch.topk(
                    labels, 2, dim=1, largest=True, sorted=True
                )
                idx_top1 = torch.arange(labels.shape[0]), top_max_k_inds[:, 0]
                idx_top2 = torch.arange(labels.shape[0]), top_max_k_inds[:, 1]
                preds = preds.detach()
                preds[idx_top1] += preds[idx_top2]
                preds[idx_top2] = 0.0
                labels = top_max_k_inds[:, 0]
                dict_preds[k] = preds
                dict_labels[k] = labels
            if is_dict_label:
                labels = dict_labels
                extra_preds = dict_preds
            else:
                labels = dict_labels['class']
                preds = dict_preds['class']

        if cfg.DETECTION.ENABLE:
            if cfg.NUM_GPUS > 1:
                loss = du.all_reduce([loss])[0]
            loss = loss.item()

            # Update and log stats.
            train_meter.update_stats(None, None, None, loss, lr)
            # write to tensorboard format if available.
            if writer is not None:
                _add = {"Train/loss": loss}
                _add.update({f"Train/{k}":v for k,v in lr.items()})
                writer.add_scalars(
                    _add,
                    global_step=data_size * cur_epoch + cur_iter,
                )

        else:
            top1_err, top5_err = None, None
            top1_err, top5_err = None, None

            if isinstance(labels, (dict,)) and cfg.TRAIN.DATASET == "epickitchens":
                # Compute the verb accuracies.
                verb_top1_acc, verb_top5_acc = metrics.topk_accuracies(
                    extra_preds['verb'], labels['verb'], (1, 5))

                loss_verb, loss_noun = loss_dict['verb_loss'], loss_dict['noun_loss']
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss_verb, verb_top1_acc, verb_top5_acc = du.all_reduce(
                        [loss_verb, verb_top1_acc, verb_top5_acc]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss_verb, verb_top1_acc, verb_top5_acc = (
                    loss_verb.item(),
                    verb_top1_acc.item(),
                    verb_top5_acc.item(),
                )

                # Compute the noun accuracies.
                noun_top1_acc, noun_top5_acc = metrics.topk_accuracies(
                    extra_preds['noun'], labels['noun'], (1, 5))

                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss_noun, noun_top1_acc, noun_top5_acc = du.all_reduce(
                        [loss_noun, noun_top1_acc, noun_top5_acc]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss_noun, noun_top1_acc, noun_top5_acc = (
                    loss_noun.item(),
                    noun_top1_acc.item(),
                    noun_top5_acc.item(),
                )

                # Compute the action accuracies.
                action_top1_acc, action_top5_acc = metrics.multitask_topk_accuracies(
                    (extra_preds['verb'], extra_preds['noun']),
                    (labels['verb'], labels['noun']),
                    (1, 5))

                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, action_top1_acc, action_top5_acc = du.all_reduce(
                        [loss, action_top1_acc, action_top5_acc]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss, action_top1_acc, action_top5_acc = (
                    loss.item(),
                    action_top1_acc.item(),
                    action_top5_acc.item(),
                )
                global_iter = data_size * cur_epoch + cur_iter
                wandb_iter_dict = {'iter_loss':loss,
                                #    'iter_verb_loss':loss_verb,
                                #       'iter_noun_loss':loss_noun,
                                    'iteration':global_iter}
                if wandb_log:
                    wandb_run.log(wandb_iter_dict)

                epoch_loss.append(conv_fp16(loss))
                epoch_verb_loss.append(conv_fp16(loss_verb))
                epoch_noun_loss.append(conv_fp16(loss_noun))
                epoch_verb_top_1_acc.append(conv_fp16(verb_top1_acc))
                epoch_verb_top_5_acc.append(conv_fp16(verb_top5_acc))
                epoch_noun_top_1_acc.append(conv_fp16(noun_top1_acc))
                epoch_noun_top_5_acc.append(conv_fp16(noun_top5_acc))
                epoch_top_1_acc.append(conv_fp16(action_top1_acc))
                epoch_top_5_acc.append(conv_fp16(action_top5_acc))
                

                # Update and log stats.
                train_meter.update_stats(
                    (verb_top1_acc, noun_top1_acc, action_top1_acc),
                    (verb_top5_acc, noun_top5_acc, action_top5_acc),
                    (loss_verb, loss_noun, loss),
                    lr, inputs[0].size(0) * cfg.NUM_GPUS
                )
            else:
                if cfg.DATA.MULTI_LABEL:
                    # Gather all the predictions across all the devices.
                    if cfg.NUM_GPUS > 1:
                        [loss] = du.all_reduce([loss])
                    loss = loss.item()

                else:
                    # Compute the errors.
                    num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
                    top1_err, top5_err = [
                        (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                    ]
                    # Gather all the predictions across all the devices.
                    if cfg.NUM_GPUS > 1:
                        loss, top1_err, top5_err = du.all_reduce(
                            [loss, top1_err, top5_err]
                        )

                    # Copy the stats from GPU to CPU (sync point).
                    loss, top1_err, top5_err = (
                        loss.item(),
                        top1_err.item(),
                        top5_err.item(),
                    )
                    epoch_loss.append(loss)
                    epoch_top_1_err.append(top1_err)
                    epoch_top_5_err.append(top5_err)
                    global_iter = data_size * cur_epoch + cur_iter
                    wandb_iter_dict = {'iter_loss':loss, 
                                       'iter_top1_err':top1_err, 
                                       'iter_top5_err':top5_err,
                                    'iteration':global_iter}
                    if wandb_log:
                        wandb_run.log(wandb_iter_dict)

                # Update and log stats.
                train_meter.update_stats(
                    top1_err,
                    top5_err,
                    loss_dict,
                    lr,
                    inputs[0].size(0)
                    * max(
                        cfg.NUM_GPUS, 1
                    ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                )

                    


        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
    if cfg.TRAIN.NO_FWD_PASS:
        end_time = time.time()
        print(f"Time taken for one epoch: {end_time-start_time}s")
        sys.exit(0)

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()
    if cfg.TRAIN.DATASET == 'epickitchens':
        wandb_iter_dict = {'train_loss':np.mean(epoch_loss),
                            'train_verb_loss':np.mean(epoch_verb_loss),
                            'train_noun_loss':np.mean(epoch_noun_loss),
                            'train_verb_top1_acc':np.mean(epoch_verb_top_1_acc),
                            'train_verb_top5_acc':np.mean(epoch_verb_top_5_acc),
                            'train_noun_top1_acc':np.mean(epoch_noun_top_1_acc),
                            'train_noun_top5_acc':np.mean(epoch_noun_top_5_acc),
                            'train_top1_acc':np.mean(epoch_top_1_acc),
                            'train_top5_acc':np.mean(epoch_top_5_acc),
                            'epoch':cur_epoch}
    else:
        wandb_iter_dict = {'train_loss':np.mean(epoch_loss),
                       'train_top1_err':np.mean(epoch_top_1_err), 
                        'train_top5_err':np.mean(epoch_top_5_err),
                        'train_top5_acc':100-np.mean(epoch_top_5_err),
                        'train_top1_acc':100-np.mean(epoch_top_1_err),
                        'epoch':cur_epoch}
    if wandb_log:
        wandb_run.log(wandb_iter_dict)


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, args, writer=None,
               wandb_run=None):
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
    epoch_loss = []
    epoch_top_1_err = []
    epoch_top_5_err = []
    epoch_noun_loss = []
    epoch_verb_loss = []
    epoch_noun_top_1_acc = []
    epoch_noun_top_5_acc = []
    epoch_verb_top_1_acc = []
    epoch_verb_top_5_acc = []
    epoch_top_1_acc = []
    epoch_top_5_acc = []
    wandb_log = wandb_run is not None

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            inputs, labels, meta = misc.iter_to_cuda([inputs, labels, meta])

        val_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta, bboxes=meta["boxes"])
            if isinstance(preds, tuple): preds, extra_preds = preds
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            if cfg.NUM_GPUS:
                preds = misc.iter_to_cpu(preds)
                ori_boxes = ori_boxes.cpu()
                metadata = metadata.cpu()

            if cfg.NUM_GPUS > 1:
                if isinstance(preds, dict):
                    preds = du.all_gather_unaligned(preds)
                    preds = {k:torch.cat([p[k] for p in preds], dim=0) for k in preds[0].keys()}
                    ori_boxes = du.all_gather_unaligned(ori_boxes)
                    # fix batch index
                    _max = 0
                    for iboxes in ori_boxes:
                        iboxes[:, 0] = iboxes[:, 0] + _max
                        _max = max(iboxes[:, 0]) + 1
                    ori_boxes = torch.cat(ori_boxes, dim=0)
                else:
                    preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                    ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(preds, ori_boxes, metadata)

        else:
            preds = model(inputs, meta)
            preds = preds / cfg.SOLVER.TEMPRATURE
            if isinstance(preds, tuple): preds, extra_preds = preds
            else: extra_preds = None
                
            # Explicitly declare reduction to mean.
            loss_fun = losses.get_loss_func(cfg)(
                reduction="mean"
            )
            # Compute the loss.
            if cfg.TRAIN.DATASET == 'epickitchens':
                loss_dict = loss_fun(extra_preds, labels)
                loss = loss_dict['verb_loss'] + loss_dict['noun_loss']
            else:
                loss = loss_fun(preds, labels)
                loss_dict = {'loss':loss}

                

            if isinstance(labels, (dict,)) and cfg.TRAIN.DATASET == "epickitchens":
                # Compute the verb accuracies.
                verb_top1_acc, verb_top5_acc = metrics.topk_accuracies(
                    extra_preds['verb'], labels['verb'], (1, 5))
                loss_verb, loss_noun = loss_dict['verb_loss'], loss_dict['noun_loss']

                # Combine the errors across the GPUs.
                if cfg.NUM_GPUS > 1:
                    verb_top1_acc, verb_top5_acc = du.all_reduce(
                        [verb_top1_acc, verb_top5_acc])
                    loss, loss_verb, loss_noun = du.all_reduce(
                        [loss, loss_verb, loss_noun])
                loss, loss_verb, loss_noun = loss.item(), loss_verb.item(), loss_noun.item()
                    

                # Copy the errors from GPU to CPU (sync point).
                verb_top1_acc, verb_top5_acc = verb_top1_acc.item(), verb_top5_acc.item()

                # Compute the noun accuracies.
                noun_top1_acc, noun_top5_acc = metrics.topk_accuracies(
                    extra_preds['noun'], labels['noun'], (1, 5))

                # Combine the errors across the GPUs.
                if cfg.NUM_GPUS > 1:
                    noun_top1_acc, noun_top5_acc = du.all_reduce(
                        [noun_top1_acc, noun_top5_acc])

                # Copy the errors from GPU to CPU (sync point).
                noun_top1_acc, noun_top5_acc = noun_top1_acc.item(), noun_top5_acc.item()

                # Compute the action accuracies.
                action_top1_acc, action_top5_acc = metrics.multitask_topk_accuracies(
                    (extra_preds['verb'], extra_preds['noun']),
                    (labels['verb'], labels['noun']),
                    (1, 5))

                # Combine the errors across the GPUs.
                if cfg.NUM_GPUS > 1:
                    action_top1_acc, action_top5_acc = du.all_reduce([action_top1_acc, action_top5_acc])

                # Copy the errors from GPU to CPU (sync point).
                action_top1_acc, action_top5_acc = action_top1_acc.item(), action_top5_acc.item()
                epoch_loss.append(loss)
                epoch_verb_loss.append(loss_verb)
                epoch_noun_loss.append(loss_noun)
                epoch_verb_top_1_acc.append(verb_top1_acc)
                epoch_verb_top_5_acc.append(verb_top5_acc)
                epoch_noun_top_1_acc.append(noun_top1_acc)
                epoch_noun_top_5_acc.append(noun_top5_acc)
                epoch_top_1_acc.append(action_top1_acc)
                epoch_top_5_acc.append(action_top5_acc)

                val_meter.iter_toc()
                
                # Update and log stats.
                val_meter.update_stats(
                    (verb_top1_acc, noun_top1_acc, action_top1_acc),
                    (verb_top5_acc, noun_top5_acc, action_top5_acc),
                    inputs[0].size(0) * cfg.NUM_GPUS
                )
                
                # write to tensorboard format if available.
                if writer is not None:
                    writer.add_scalars(
                        {
                            "Val/verb_top1_acc": verb_top1_acc,
                            "Val/verb_top5_acc": verb_top5_acc,
                            "Val/noun_top1_acc": noun_top1_acc,
                            "Val/noun_top5_acc": noun_top5_acc,
                            "Val/action_top1_acc": action_top1_acc,
                            "Val/action_top5_acc": action_top5_acc,
                        },
                        global_step=len(val_loader) * cur_epoch + cur_iter,
                    )
            else:
                if cfg.DATA.MULTI_LABEL:
                    if cfg.NUM_GPUS > 1:
                        preds, labels = du.all_gather([preds, labels])
                else:
                    loss_fun = losses.get_loss_func(cfg)(
                            reduction="mean"
                        )
                    # Compute the loss.
                    
                    loss = loss_fun(preds, labels)
                    # Compute the errors.
                    num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))

                    # Combine the errors across the GPUs.
                    top1_err, top5_err = [
                        (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                    ]
                    if cfg.NUM_GPUS > 1:
                        top1_err, top5_err = du.all_reduce([top1_err, top5_err])

                    # Copy the errors from GPU to CPU (sync point).
                    top1_err, top5_err = top1_err.item(), top5_err.item()
                    epoch_loss.append(loss.item())
                    epoch_top_1_err.append(top1_err)
                    epoch_top_5_err.append(top5_err)


                    # additional meterices
                    extra_metrics = eval_extra_metrics(cfg, preds, extra_preds, labels, meta)
                    if cfg.NUM_GPUS > 1:
                        all_keys, all_values = zip(*list(extra_metrics.items()))
                        all_values = du.all_reduce(all_values)
                        extra_metrics = dict(zip(all_keys, all_values))

                    val_meter.iter_toc()
                    # Update and log stats.
                    val_meter.update_stats(
                        top1_err,
                        top5_err,
                        inputs[0].size(0)
                        * max(
                            cfg.NUM_GPUS, 1
                        ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                        extra_metrices = extra_metrics,
                    )
                    # write to tensorboard format if available.
                   

            val_meter.update_predictions(preds, labels)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    if cfg.TRAIN.DATASET == 'epickitchens':
        log_dict = {'val_loss': np.mean(epoch_loss),
                    'val_verb_loss': np.mean(epoch_verb_loss),
                    'val_noun_loss': np.mean(epoch_noun_loss),
                    'val_verb_top1_acc': np.mean(epoch_verb_top_1_acc),
                    'val_verb_top5_acc': np.mean(epoch_verb_top_5_acc),
                    'val_noun_top1_acc': np.mean(epoch_noun_top_1_acc),
                    'val_noun_top5_acc': np.mean(epoch_noun_top_5_acc),
                    'val_top1_acc': np.mean(epoch_top_1_acc),
                    'val_top5_acc': np.mean(epoch_top_5_acc),
                    'epoch': cur_epoch}
    else:
        log_dict = {'val_loss': np.mean(epoch_loss),
                    'val_top1_err': np.mean(epoch_top_1_err),
                    'val_top5_err': np.mean(epoch_top_5_err),
                    'val_top5_acc': 100-np.mean(epoch_top_5_err),
                    'val_top1_acc': 100-np.mean(epoch_top_1_err),
                    'epoch': cur_epoch}
    if wandb_log:
        wandb_run.log(log_dict)
    # write to tensorboard format if available.
    if writer is not None:
        if cfg.DETECTION.ENABLE:
            writer.add_scalars(
                {"Val/mAP": val_meter.full_map}, global_step=cur_epoch
            )
        else:
            all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
            all_labels = [
                label.clone().detach() for label in val_meter.all_labels
            ]
            if cfg.NUM_GPUS:
                all_preds = [pred.cpu() for pred in all_preds]
                all_labels = [label.cpu() for label in all_labels]
            writer.plot_eval(
                preds=all_preds, labels=all_labels, global_step=cur_epoch
            )

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




def train(cfg, args, local_rank=-1):
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

    if args.rank == 0:    
        wandb_config_dict = wandb_init_dict(cfg)
        wandb_config_dict['slurm_id'] = os.environ.get('SLURM_JOB_ID')
        if "SLURM_ARRAY_TASK_ID" in os.environ:
            wandb_config_dict['slurm_id'] += "_" + os.environ.get(
                                                            'SLURM_ARRAY_TASK_ID')
        wandb_run = wandb.init(project='pt_orvit_post_eccv',config=wandb_config_dict, 
                                    entity="act_seg_pi_umd")
        wandb_run.define_metric("epoch")
        wandb_run.define_metric("iteration")

        wandb_run.define_metric("iter*", step_metric="iteration")

        wandb_run.define_metric("train*", step_metric="epoch")
        wandb_run.define_metric("val*", step_metric="epoch")
        wandb_run.define_metric("train_loss", summary="min")
        wandb_run.define_metric("val_loss", summary="min")
        wandb_run.define_metric("val_top5_acc", summary="max")
        wandb_run.define_metric("val_top1_acc", summary="max")
    else:
        wandb_run = None
   

    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if cfg.NUM_GPUS>1:
        cfg['num_patches'] = model.module.num_patches
    else:
        cfg['num_patches'] = model.num_patches


    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(
        cfg, model, optimizer, scaler if cfg.TRAIN.MIXED_PRECISION else None
    )

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = (
        loader.construct_loader(cfg, "train", is_precise_bn=True)
        if cfg.BN.USE_PRECISE_STATS
        else None
    )

    # Create meters.
    if cfg.DETECTION.ENABLE:
        train_meter = AVAMeter(len(train_loader), cfg, mode="train")
        val_meter = AVAMeter(len(val_loader), cfg, mode="val")
    else:
        if cfg.TRAIN.DATASET == 'epickitchens':
            train_meter = EPICTrainMeter(len(train_loader), cfg)
            val_meter = EPICValMeter(len(val_loader), cfg)
        else:
            train_meter = TrainMeter(len(train_loader), cfg)
            val_meter = ValMeter(len(val_loader), cfg)

    
    writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    epoch_timer = EpochTimer()
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, changed = multigrid.update_long_cycle(cfg, cur_epoch)
            if changed:
                (
                    model,
                    optimizer,
                    train_loader,
                    val_loader,
                    precise_bn_loader,
                    train_meter,
                    val_meter,
                ) = build_trainer(cfg)

                # Load checkpoint.
                if cu.has_checkpoint(cfg.OUTPUT_DIR):
                    last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
                    assert "{:05d}.pyth".format(cur_epoch) in last_checkpoint
                else:
                    last_checkpoint = cfg.TRAIN.CHECKPOINT_FILE_PATH
                logger.info("Load from {}".format(last_checkpoint))
                cu.load_checkpoint(
                    last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
                )

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # Train for one epoch.
        epoch_timer.epoch_tic()
        if not cfg.TRAIN.VAL_ONLY:
            train_epoch(
                train_loader,
                model,
                optimizer,
                scaler,
                train_meter,
                cur_epoch,
                cfg,
                args,
                writer,
                wandb_run
            )
        epoch_timer.epoch_toc()
        logger.info(
            f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
            f"from {start_epoch} to {cur_epoch} take "
            f"{epoch_timer.avg_epoch_time():.2f}s in average and "
            f"{epoch_timer.median_epoch_time():.2f}s in median."
        )
        logger.info(
            f"For epoch {cur_epoch}, each iteraction takes "
            f"{epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average. "
            f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
            f"{epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average."
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
            None if multigrid is None else multigrid.schedule,
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg, cur_epoch, None if multigrid is None else multigrid.schedule
        )

        # Compute precise BN stats.
        if (
            (is_checkp_epoch or is_eval_epoch)
            and cfg.BN.USE_PRECISE_STATS
            and len(get_bn_modules(model)) > 0
        ):
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        cfg_to_save = cfg.clone()
        if is_checkp_epoch:
            cu.save_checkpoint(
                cfg.OUTPUT_DIR,
                model,
                optimizer,
                cur_epoch,
                cfg_to_save,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
        # Evaluate the model on validation set.1
        if is_eval_epoch:
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, args, writer, 
                       wandb_run)
        if cfg.TRAIN.VAL_ONLY:
            break

    if writer is not None:
        writer.close()
