#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
import torch
import slowfast.models.losses as losses
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.env import pathmgr
from slowfast.utils.meters import AVAMeter, TestMeter, EPICTestMeter
import slowfast.utils.metrics as metrics
import pandas as pd
from tqdm import tqdm


logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
    
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()
    full_df = []
    epoch_top_1_err = []
    epoch_top_5_err = []

    #Setting for compact dumping of results
    np.set_printoptions(suppress=True, precision=6)
    vid_id_to_name_dict = test_loader.dataset._vid_id_to_name
    loss_fun = losses.get_loss_func(cfg)(
                reduction="none"
            )
    

    for cur_iter, (inputs, labels, video_idx, meta) in enumerate(tqdm(test_loader)):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            inputs, labels, video_idx, meta = misc.iter_to_cuda([inputs, labels, video_idx, meta])

        test_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta, bboxes=meta["boxes"])
            if isinstance(preds, tuple): preds, extra_preds = preds 
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
            ori_boxes = (
                ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
            )
            metadata = (
                metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()
            )

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(preds, ori_boxes, metadata)
            test_meter.log_iter_stats(None, cur_iter)
        else:
            # Perform the forward pass.
            preds = model(inputs, meta)
            if isinstance(preds, tuple):
                preds, extra_preds = preds
    
            if isinstance(labels, (dict,)):
                # Gather all the predictions across all the devices to perform ensemble.
                if cfg.NUM_GPUS > 1:
                    verb_preds, verb_labels, video_idx = du.all_gather(
                        [extra_preds['verb'], labels['verb'], video_idx]
                    )

                    noun_preds, noun_labels, video_idx = du.all_gather(
                        [extra_preds['noun'], labels['noun'], video_idx]
                    )
                    nid = du.all_gather_unaligned(meta['narration_id'])
                    metadata = {'narration_id': []}
                    for i in range(len(nid)):
                        metadata['narration_id'].extend(nid[i])
                else:
                    metadata = meta
                    verb_preds, verb_labels, video_idx = extra_preds['verb'], labels['verb'], video_idx
                    noun_preds, noun_labels, video_idx = extra_preds['noun'], labels['noun'], video_idx
                test_meter.iter_toc()
                # Update and log stats.
                test_meter.update_stats(
                    (verb_preds.detach().cpu(), noun_preds.detach().cpu()),
                    (verb_labels.detach().cpu(), noun_labels.detach().cpu()),
                    metadata,
                    video_idx.detach().cpu(),
                )
                test_meter.log_iter_stats(cur_iter)
            else:
                # Gather all the predictions across all the devices to perform ensemble.
               
                # Gather all the predictions across all the devices.
                

                # Copy the stats from GPU to CPU (sync point).

                loss = loss_fun(preds, labels) # reduction is none so that we can get per sample loss
                 
                if cfg.NUM_GPUS > 1:
                    preds, labels, video_idx, loss = du.all_gather(
                        [preds, labels, video_idx, loss]
                    )
                video_idx = video_idx.cpu().numpy().astype(int)

                video_names = [vid_id_to_name_dict[video_idx[idx]] for idx in range(len(video_idx))]
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                # test_meter.iter_toc()
                # # Update and log stats.
                # test_meter.update_stats(
                #     preds.detach(), labels.detach(), video_idx.detach()
                # )
                if cfg.NUM_GPUS:
                    preds = preds.cpu()
                    labels = labels.cpu()
                    top1_err = top1_err.cpu()
                    top5_err = top5_err.cpu()
                    loss = loss.cpu()
                epoch_top_1_err.append(top1_err)
                epoch_top_5_err.append(top5_err)
                preds = np.round(preds.numpy(), 6)
                labels = labels.numpy()
                loss = loss.numpy()
                batch_df = pd.DataFrame(preds, columns=[f'label_{i}' for i in range(preds.shape[1])])
                batch_df['gt_label'] = labels
                batch_df['video_name'] = video_names
                batch_df['loss'] = loss
                full_df.append(batch_df.copy())
                
                
                

                
            test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    # Log epoch stats and print the final testing results.
    if not cfg.DETECTION.ENABLE:
        if cfg.TEST.DATASET == 'epickitchens':
            pass
        else:
            all_preds = test_meter.video_preds.clone().detach()
            all_labels = test_meter.video_labels
            if cfg.NUM_GPUS:
                all_preds = all_preds.cpu()
                all_labels = all_labels.cpu()
            if writer is not None:
                writer.plot_eval(preds=all_preds, labels=all_labels)

            if cfg.TEST.SAVE_RESULTS_PATH != "":
                save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

                if du.is_root_proc():
                    with pathmgr.open(save_path, "wb") as f:
                        pickle.dump([all_preds, all_labels], f)

                logger.info(
                    "Successfully saved prediction results to {}".format(save_path)
                )
    full_df = pd.concat(full_df).reset_index(drop=True)
    #TODO: Change this to a more generic name, and better dumping
    num_frames = cfg.DATA.NUM_FRAMES
    full_df.to_csv(os.path.join(cfg.OUTPUT_DIR, f'test_results_{num_frames}.csv'), index=False)
    print(os.path.join(cfg.OUTPUT_DIR, 'test_results.csv'))
    
    print('Top 1 Error: ', np.mean(epoch_top_1_err))
    print('Top 5 Error: ', np.mean(epoch_top_5_err))


def test(cfg, args):
    """
    Perform multi-view testing on the pretrained video model.
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

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)

    if cfg.NUM_GPUS>1:
        cfg['num_patches'] = model.module.num_patches
    else:
        cfg['num_patches'] = model.num_patches
    
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)
    

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "val")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    if cfg.DETECTION.ENABLE:
        assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE or cfg.NUM_GPUS == 0
        test_meter = AVAMeter(len(test_loader), cfg, mode="test")
    else:
        assert (
            test_loader.dataset.num_videos
            % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
            == 0
        )

        # Create meters for multi-view testing.
        if cfg.TEST.DATASET == 'epickitchens':
            test_meter = EPICTestMeter(
                len(test_loader.dataset)
                // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
                [97, 300],
                len(test_loader),
            )
        else:
            test_meter = TestMeter(
                test_loader.dataset.num_videos
                // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
                cfg.MODEL.NUM_CLASSES,
                len(test_loader),
                cfg.DATA.MULTI_LABEL,
                cfg.DATA.ENSEMBLE_METHOD,
            )

    # Set up writer for logging to Tensorboard format.
    # if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
    #     cfg.NUM_GPUS * cfg.NUM_SHARDS
    # ):
    #     writer = tb.TensorboardWriter(cfg)
    # else:
    #     writer = None
    writer = None
    # # Perform multi-view test on the entire dataset.
    test_meter = perform_test(test_loader, model, test_meter, cfg, writer)
    if writer is not None:
        writer.close()
