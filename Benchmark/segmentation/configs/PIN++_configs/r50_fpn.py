from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.data.detection_utils import get_fed_loss_cls_weights
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.layers import ShapeSpec
from detectron2.layers.batch_norm import NaiveSyncBatchNorm
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import StandardROIHeads
from detectron2.modeling.roi_heads import (
    FastRCNNConvFCHead,
    FastRCNNOutputLayers,
    MaskRCNNConvUpsampleHead,
)
from paco.evaluation.paco_evaluation import PACOEvaluator
from detectron2.evaluation import COCOEvaluator
from paco.models.roi_heads import PACOROIHeads
import os
from .coco_lsj_loader import dataloader

num_classes = 4308
dataloader.train.total_batch_size = 8
dataloader.train.dataset.names = "PIN++_train"
dataloader.test.dataset.names = "PIN++_test"
dataloader.train.mapper.instance_mask_format = "bitmask"
dataloader.train.mapper.image_format = "RGB"


model = model_zoo.get_config("new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py").model
optimizer = model_zoo.get_config(
    "new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py"
).optimizer
lr_multiplier = model_zoo.get_config(
    "new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py"
).lr_multiplier
train = model_zoo.get_config("new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py").train

model.backbone.bottom_up.stem.norm = (
    model.backbone.bottom_up.stages.norm
) = model.backbone.norm = "SyncBN"

model.roi_heads.update(
    _target_=StandardROIHeads,
    num_classes=num_classes,
    batch_size_per_image=128,
    positive_fraction=0.25,
    proposal_matcher=L(Matcher)(
        thresholds=[0.5], labels=[0, 1], allow_low_quality_matches=False
    ),
    box_in_features=["p2", "p3", "p4", "p5"],
    box_pooler=L(ROIPooler)(
        output_size=7,
        scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
        sampling_ratio=0,
        pooler_type="ROIAlignV2",
    ),
    box_head=L(FastRCNNConvFCHead)(
        input_shape=ShapeSpec(channels=256, height=7, width=7),
        conv_dims=[256, 256, 256, 256],
        fc_dims=[1024],
        conv_norm=lambda c: NaiveSyncBatchNorm(c, stats_mode="N"),
    ),
    box_predictor=L(FastRCNNOutputLayers)(
        input_shape=ShapeSpec(channels=1024),
        test_score_thresh=0.001,
        box2box_transform=L(Box2BoxTransform)(weights=(10, 10, 5, 5)),
        num_classes="${..num_classes}",
        test_topk_per_image=300,
        use_sigmoid_ce=True,
        use_fed_loss=True,
        get_fed_loss_cls_weights=lambda: get_fed_loss_cls_weights(
            dataloader.train.dataset.names, 0.5
        ),
    ),
    mask_in_features=["p2", "p3", "p4", "p5"],
    mask_pooler=L(ROIPooler)(
        output_size=14,
        scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
        sampling_ratio=0,
        pooler_type="ROIAlignV2",
    ),
    mask_head=L(MaskRCNNConvUpsampleHead)(
        input_shape=ShapeSpec(channels=256, width=14, height=14),
        num_classes="${..num_classes}",
        conv_dims=[256, 256, 256, 256, 256],
        conv_norm=lambda c: NaiveSyncBatchNorm(c, stats_mode="N"),
    ),
)

model.proposal_generator.head.conv_dims = [-1, -1]

dataloader.train.sampler = L(RepeatFactorTrainingSampler)(
    repeat_factors=L(
        RepeatFactorTrainingSampler.repeat_factors_from_category_frequency
    )(dataset_dicts="${dataloader.train.dataset}", repeat_thresh=0.001)
)



dataloader.evaluator = L(PACOEvaluator)(
    dataset_name="${..test.dataset.names}",
    max_dets_per_image=300,
)


# Schedule
optimizer.lr = 0.03
train.max_iter = 62500
train.eval_period = 2500000
train.log_period = 100
train.output_dir = "./checkpoint/resnet50_dataset_op_lr_" + str(optimizer.lr)
train.checkpointer.period = 50

if not os.path.exists(train.output_dir):  
    os.makedirs(train.output_dir)  

lr_multiplier.scheduler.milestones = [round(train.max_iter * 0.5), round(train.max_iter * 0.75)]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 250 / train.max_iter
