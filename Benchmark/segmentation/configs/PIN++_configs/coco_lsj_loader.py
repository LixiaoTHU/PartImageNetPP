# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator
from omegaconf import OmegaConf

from paco.data.dataset_mapper import PACODatasetMapper


image_size = 224
min_size = 375
max_size = 500
dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="coco_2017_train"),
    mapper=L(PACODatasetMapper)(
        is_train=True,
        augmentations=[
            L(T.RandomFlip)(horizontal=True),  # flip first
            L(T.ResizeShortestEdge)(
                short_edge_length=min_size, 
                max_size=max_size),
        ],
        image_format="RGB",
        use_instance_mask=True,
        recompute_boxes=True,
    ),
    total_batch_size=64,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="coco_2017_val", filter_empty=False),
    mapper=L(PACODatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=min_size, max_size=max_size),
        ],
        image_format="${...train.mapper.image_format}",
    ),
    num_workers=4,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)
