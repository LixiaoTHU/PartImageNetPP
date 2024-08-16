import torch
import torch.nn as nn
from timm.models import create_model, safe_model_name
from .partmodel import *

def build_model(args, _logger):
    # creating model
    _logger.info(f"Creating model: {args.model}")
    model_kwargs=dict({
        'num_classes': args.num_classes,
        'drop_path_rate': args.drop_path_rate,
        'drop_rate': args.drop,
        'global_pool': args.global_pool,
        'bn_momentum': args.bn_momentum,
        'bn_eps': args.bn_eps,
    })
    model = create_model(args.model, pretrained=False, **model_kwargs)


    def replace_relu_with_gelu(module):
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                module.__setattr__(name, nn.GELU())
            else:
                replace_relu_with_gelu(child)
    replace_relu_with_gelu(model)

    
    _logger.info(f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')


    model.cuda()
    if args.channels_last:  # for acceleration on CPU.
        model = model.to(memory_format=torch.channels_last)

    return model


def build_part_model(args, _logger):
    # creating model
    _logger.info(f"Creating model: {args.model}")
    model_kwargs=dict({
        'num_classes': args.num_classes,
        'drop_path_rate': args.drop_path_rate,
        'drop_rate': args.drop,
        'global_pool': args.global_pool,
        'bn_momentum': args.bn_momentum,
        'bn_eps': args.bn_eps,
    })
    if "resnet" in args.model or "resnext" in args.model:
        dtype = "resnet"
    else:
        dtype = None
    segmenter = create_model(args.model, pretrained=False, **model_kwargs) # [N, C, 7, 7]

    if args.partmodel_type == "twostage":
        model = TwostagePartModel(segmenter, args.seg_classes, args.num_classes)
 
    elif args.partmodel_type == "MPM":
        model = MPM(segmenter, args.seg_classes, args.num_classes, dtype=dtype, level=args.level, down=args.down)
    
    def replace_relu_with_gelu(module):
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                module.__setattr__(name, nn.GELU())
            else:
                replace_relu_with_gelu(child)
    replace_relu_with_gelu(model)
    
    _logger.info(f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')


    model.cuda()
    if args.channels_last:  # for acceleration on CPU.
        model = model.to(memory_format=torch.channels_last)

    return model