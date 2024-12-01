#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .build import DATASET_REGISTRY, build_dataset  # noqa
from .ucf import Ucf101  # noqa
from .hmdb import Hmdb51  # noqa
# from .diving48 import Diving48  # noqa
from .base_ds import BaseDataset # noqa
from .ssv2 import Ssv2 # noqa
from .k400 import K400 # noqa

