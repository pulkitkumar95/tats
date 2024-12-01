#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Optimizer."""

import torch

import slowfast.utils.lr_policy as lr_policy

import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)


def construct_optimizer(model, cfg):
    """
    Construct a stochastic gradient descent or ADAM optimizer with momentum.
    Details can be found in:
    Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
    and
    Diederik P.Kingma, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."

    Args:
        model (model): model to perform stochastic gradient descent
        optimization or ADAM optimization.
        cfg (config): configs of hyper-parameters of SGD or ADAM, includes base
        learning rate,  momentum, weight_decay, dampening, and etc.
    """
    bn_parameters = []
    non_bn_parameters = []
    zero_parameters = []
    no_grad_parameters = []

    orvit_bn_parameters = []
    orvit_non_bn_parameters = []
    orvit_zero_parameters = []
    orvit_no_grad_parameters = []
    extra_encoder_parameters = []

    skip = {}
    if hasattr(model, "no_weight_decay"):
        skip = model.no_weight_decay()

    for name, p in model.named_parameters():
        is_bn = isinstance(p, torch.nn.modules.batchnorm._NormBase)

        if not p.requires_grad:
            no_grad_parameters.append(p)
        elif is_bn:
            bn_parameters.append(p)
        elif name in skip or (
            (len(p.shape) == 1 or name.endswith(".bias"))
            and cfg.SOLVER.ZERO_WD_1D_PARAM
        ):
            zero_parameters.append(p)

        else:
            non_bn_parameters.append(p)

   
    optim_params = [
        {"params": non_bn_parameters + orvit_non_bn_parameters, "weight_decay": cfg.SOLVER.WEIGHT_DECAY},
        {"params": zero_parameters + orvit_zero_parameters, "weight_decay": 0.0},
        {"params": extra_encoder_parameters, "lr": cfg.SOLVER.EXTRA_ENCODER_LR, 'extra_encoder': True},
    ]

    optim_params = [x for x in optim_params if len(x["params"])]

    # Check all parameters will be passed into optimizer.
    assert len(list(model.parameters())) == sum([
        len(non_bn_parameters),
        len(bn_parameters),
        len(zero_parameters),
        len(no_grad_parameters),
        len(orvit_non_bn_parameters),
        len(orvit_bn_parameters),
        len(orvit_zero_parameters),
        len(orvit_no_grad_parameters),
        len(extra_encoder_parameters)]), "parameter size does not match: {} + {} + {} + {} != {}".format(
        len(non_bn_parameters),
        len(bn_parameters),
        len(zero_parameters),
        len(no_grad_parameters),
        len(orvit_non_bn_parameters),
        len(orvit_bn_parameters),
        len(orvit_zero_parameters),
        len(orvit_no_grad_parameters),
        len(list(model.parameters())),
    )
    print(
        "bn {}, non bn {}, zero {} no grad {}".format(
            len(bn_parameters),
            len(non_bn_parameters),
            len(zero_parameters),
            len(no_grad_parameters),
        )
    )

    if cfg.SOLVER.OPTIMIZING_METHOD == "sgd":
        return torch.optim.SGD(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            dampening=cfg.SOLVER.DAMPENING,
            nesterov=cfg.SOLVER.NESTEROV,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adam":
        return torch.optim.Adam(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adamw":
        return torch.optim.AdamW(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            eps=1e-08,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg.SOLVER.OPTIMIZING_METHOD)
        )


def get_epoch_lr(cur_epoch, cfg):
    """
    Retrieves the lr for the given epoch (as specified by the lr policy).
    Args:
        cfg (config): configs of hyper-parameters of ADAM, includes base
        learning rate, betas, and weight decay.
        cur_epoch (float): the number of epoch of the current training stage.
    """
    return lr_policy.get_lr_at_epoch(cfg, cur_epoch)


def set_lr(optimizer, new_lr, log = False):
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    if 'orvit_lr' in new_lr:
        n_orvit = 0
        n_p_orvit = 0
        n_p = 0
        n = len(optimizer.param_groups)
        for param_group in optimizer.param_groups:
            if param_group['is_orvit']:
                n_orvit += 1
                n_p_orvit += len(param_group['params'])
            else:
                n_p += len(param_group['params'])
            _new_lr = new_lr['orvit_lr'] if param_group['is_orvit'] else new_lr['lr']
            param_group["lr"] = _new_lr
        if log: logger.info(f"Set lr {new_lr['lr']} for {n - n_orvit} groups parameters with {n_p} parameters and {new_lr['orvit_lr']} for {n_orvit} orvit group parameters with {n_p_orvit} parameters")
    else:
        if log: logger.info(f"Set lr {new_lr['lr']} for all paramters groups")
        for param_group in optimizer.param_groups:
            if 'extra_encoder' not in param_group:
                param_group["lr"] = new_lr['lr']
