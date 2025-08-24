


from .r50_fpn import (  # noqa
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)
train.output_dir = "./checkpoint/resnet101_dataset_op_lr_" + str(optimizer.lr)
model.backbone.bottom_up.stages.depth = 101