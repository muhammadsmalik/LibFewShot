# -*- coding: utf-8 -*-
import datetime
import logging
import os
import builtins
from logging import getLogger
from time import time

import torch
import yaml
from torch import nn
import torch.distributed as dist
import torch_xla
import torch_xla.core.xla_model as xm

import torch_xla.distributed.xla_multiprocessing as xmp


from queue import Queue
import core.model as arch
from core.data import get_dataloader
from core.utils import (
    AverageMeter,
    ModelType,
    SaveType,
    TensorboardWriter,
    count_parameters,
    create_dirs,
    get_local_time,
    init_logger_config,
    init_seed,
    prepare_device,
    save_model,
    get_instance,
    data_prefetcher,
    GradualWarmupScheduler,
)


class Trainer(object):
    """
    The trainer.

    Build a trainer from config dict, set up optimizer, model, etc. Train/test/val and log.
    """

    def __init__(self, rank, config, result_dir, f, name):
        print("Before initializing device")
    device = xm.xla_device()
    print("After initializing device", device)
