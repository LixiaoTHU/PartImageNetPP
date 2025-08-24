from functools import partial

from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate

from .vit_b_fpn import (  # noqa
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
)


model.backbone.net.embed_dim = 384
model.backbone.net.depth = 12
model.backbone.net.num_heads = 6
model.backbone.net.drop_path_rate = 0.4
# 5, 11, 17, 23 for global attention

optimizer.params.lr_factor_func = partial(
    get_vit_lr_decay_rate, lr_decay_rate=0.8, num_layers=12
)

train.output_dir = "./checkpoint/vits_dataset_op_lr_" + str(optimizer.lr)
