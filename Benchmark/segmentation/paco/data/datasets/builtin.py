# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

from .paco import get_instances_meta, register_instances

# ==== Root directories ====
_PACO_ANNOTATION_ROOT = os.environ.get(
    "PACO_ANNOTATION_ROOT", "datasets/paco/annotations"
)
_PACO_IMAGE_ROOT = os.environ.get("PACO_IMAGE_ROOT", "datasets/paco/images")
_COCO_IMAGE_ROOT = os.environ.get("COCO_IMAGE_ROOT", "datasets/coco")

_PIN_ANNOTATION_ROOT = os.environ.get(
     "PIN++_ANNOTATION_ROOT",
     "/home/liuyining/lizhuhong/paco-main/PIN++"
     )
_PIN_IMAGE_ROOT = os.environ.get(
    "PIN++_IMAGE_ROOT",
    "/home/liuyining/data2/PIN++/images"
    )

# ==== Predefined datasets and splits for PACO ==========
"""
    "paco_lvis_v1_train": (
        os.path.join(_PACO_ANNOTATION_ROOT, "paco_lvis_v1_train.json"),
        _COCO_IMAGE_ROOT,
    ),
    "paco_lvis_v1_val": (
        os.path.join(_PACO_ANNOTATION_ROOT, "paco_lvis_v1_val.json"),
        _COCO_IMAGE_ROOT,
    ),
    "paco_lvis_v1_test": (
        os.path.join(_PACO_ANNOTATION_ROOT, "paco_lvis_v1_test.json"),
        _COCO_IMAGE_ROOT,
    ),
    
    "paco_joint_v1_train": (
        os.path.join(_PACO_ANNOTATION_ROOT, "paco_lvis_v1_train.json"),
        _COCO_IMAGE_ROOT,
    ),
    "paco_ego4d_v1_train": (
        os.path.join(_PACO_ANNOTATION_ROOT, "paco_ego4d_v1_train.json"),
        _PACO_IMAGE_ROOT,
    ),
    "paco_ego4d_v1_val": (
        os.path.join(_PACO_ANNOTATION_ROOT, "paco_ego4d_v1_val.json"),
        _PACO_IMAGE_ROOT,
    ),
    "paco_ego4d_v1_test": (
        os.path.join(_PACO_ANNOTATION_ROOT, "paco_ego4d_v1_test.json"),
        _PACO_IMAGE_ROOT,
    ),
"""
_PREDEFINED_PACO = {
    "PIN++_train": (
        os.path.join(_PIN_ANNOTATION_ROOT, "Pin++_train.json"),
        _PIN_IMAGE_ROOT,
    ),
    "PIN++_val": (
        os.path.join(_PIN_ANNOTATION_ROOT, "Pin++_val.json"),
        _PIN_IMAGE_ROOT,
    ),
    "PIN++_test": (
        os.path.join(_PIN_ANNOTATION_ROOT, "Pin++_test.json"),
        _PIN_IMAGE_ROOT,
    ),
    "PIN++Part_train": (
        os.path.join(_PIN_ANNOTATION_ROOT, "Pin++Part_train.json"),
        _PIN_IMAGE_ROOT,
    ),
    "PIN++Part_val": (
        os.path.join(_PIN_ANNOTATION_ROOT, "Pin++Part_val.json"),
        _PIN_IMAGE_ROOT,
    ),
    "PIN++Part_test": (
        os.path.join(_PIN_ANNOTATION_ROOT, "Pin++Part_test.json"),
        _PIN_IMAGE_ROOT,
    ),
    "PIN++Object_train": (
        os.path.join(_PIN_ANNOTATION_ROOT, "Pin++Object_train.json"),
        _PIN_IMAGE_ROOT,
    ),
    "PIN++Object_val": (
        os.path.join(_PIN_ANNOTATION_ROOT, "Pin++Object_val.json"),
        _PIN_IMAGE_ROOT,
    ),
    "PIN++Object_test": (
        os.path.join(_PIN_ANNOTATION_ROOT, "Pin++Object_test.json"),
        _PIN_IMAGE_ROOT,
    ),
    "PIN++5way5shot_train": (
        os.path.join(_PIN_ANNOTATION_ROOT, "Pin++5way5shot_train.json"),
        _PIN_IMAGE_ROOT,
    ),
    "PIN++5way5shot_val": (
        os.path.join(_PIN_ANNOTATION_ROOT, "Pin++5way5shot_val.json"),
        _PIN_IMAGE_ROOT,
    ),
    "PIN++5way5shot_test": (
        os.path.join(_PIN_ANNOTATION_ROOT, "Pin++5way5shot_test.json"),
        _PIN_IMAGE_ROOT,
    ),
}


def register_all_paco():
    for dataset_name, (annotation_path, image_root) in _PREDEFINED_PACO.items():
        register_instances(
            dataset_name,
            get_instances_meta(dataset_name),
            annotation_path,
            image_root,
        )
"""
def register_all_paco():
    for dataset_name, (annotation_path, image_root) in _PREDEFINED_PACO.items():
        register_instances(
            dataset_name,
            annotation_path,
            image_root,
        )
"""

if __name__.endswith(".builtin"):
    register_all_paco()
