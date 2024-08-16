import os
import csv
import yaml
import argparse, logging
import time
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import matplotlib.pyplot as plt
import numpy as np

# timm func
from timm.utils import AverageMeter, reduce_tensor, accuracy
from data.timm_local.transforms_factory import create_transform

from utils import random_seed, NormalizeByChannelMeanStd, create_logger
from data import ImageNet, visualizations, partdataset
from model import build_model, build_part_model, ti_wideresnetwithswish



def get_args_parser():
    parser = argparse.ArgumentParser('Robust evaluation script', add_help=False)
    parser.add_argument('--configs', default='', type=str)
    parser.add_argument('--checkpoint', default='', type=str, metavar='PATH', help='path to the evaluated checkpoint')
    parser.add_argument('--rank', default=0, type=int, help='rank')

    # Model parameters
    parser.add_argument('--model-name', default='convnext_tiny', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--partmodel-type', type=str, default='default')
    parser.add_argument('--output-size', type=list, default=[7, 7])
    parser.add_argument('--num-classes', default=90, type=int, help='number of classes')
    parser.add_argument('--test-dir', default="", type=str, help='test evluation dir for autoattack')
    parser.add_argument('--test-file', default="", type=str, help='test evluation file for autoattack')

    parser.add_argument('--global-pool', default=None, type=str, metavar='POOL',
                        help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None. (opt)')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path-rate', type=float, default=0.0, metavar='PCT', help='Drop path rate (default: 0.)')

    # data parameters
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--crop-pct', default=0.875, type=float, metavar='N', help='Input image center crop percent (for validation only)')

    # misc
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--pin_mem', default=True)
    parser.add_argument('--output_dir', default='',
                        help='path to save accuary of each category')
    parser.add_argument('--per-class-acc', default=False, help='Whether record accuarcy of each category while eval')
    parser.add_argument('--pred-vis', default=False, help='Whether show pred mask while eval')

    # advprop
    parser.add_argument('--advprop', default=False, help='if use advprop')
    parser.add_argument('--channels-last', action='store_true', default=False,
                        help='Use channels_last memory layout (opt)')

    # attack info
    parser.add_argument('--norm', type=str, default='Linf', help='You can choose norm for aa attack', choices=['Linf', 'L2', 'L1'])
    parser.add_argument('--eval-eps', type=float, default=-1, help='attack epsilon.')
    parser.add_argument('--attack_types', type=str, nargs='*', default=('autoattack',), help='autoattack')
    parser.add_argument('--level', type=int, default=3, help='part level')
    parser.add_argument('--down', type=bool, default=True, help='downsample')

    # bn
    parser.add_argument('--bn-momentum', type=float, default=None, help='BatchNorm momentum override (if not None)')
    parser.add_argument('--bn-eps', type=float, default=None, help='BatchNorm epsilon override (if not None)')

    return parser

def build_transform(args):
    t = []
    if args.input_size == 224:
        size = int(args.input_size/args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
        )
        t.append(transforms.CenterCrop(args.input_size))
    else:       # for tiny
        t.append(
            transforms.Resize(args.input_size, interpolation=InterpolationMode.BICUBIC),
        )
    t.append(transforms.ToTensor())
    test_transform = transforms.Compose(t)
    return test_transform

def main(args):

    # fix the seed for reproducibility
    random_seed(args.seed, args.rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # create transform
    test_transform = build_transform(args)

    # create logger without saving log
    _logger = create_logger(args.output_dir, dist_rank=args.rank, name='main_test', default_level=logging.INFO, save_log=False)
    
    # get model
    if args.model_type == "original":

        model = build_model(args, _logger)
        if args.pretrain:
            model.load_state_dict(torch.load(args.pretrain, map_location='cpu'))
    elif args.model_type == "partmodel":
        model = build_part_model(args, _logger)
    else:
        print("Not Implemented!")
        exit(0)

    ckpt_dir = args.checkpoint
    ckpt = torch.load(ckpt_dir, map_location='cpu')
    ckpt = ckpt['state_dict_ema']
    model.load_state_dict(ckpt)

    # add normalize layer for evalation
    normalize = NormalizeByChannelMeanStd(mean=args.mean, std=args.std)
    class eval_model(nn.Module):
        def __init__(self, args, norm, model):
            super().__init__()
            self.norm = norm
            self.model = model
            self.return_seg = args.pred_vis

        def forward(self, img):
            img = self.norm(img)
            if self.return_seg:
                output, seg_result = self.model(img, return_seg = True)
                return output, seg_result

            else:
                output = self.model(img)
                return output
    model = eval_model(args, normalize, model)

    model = nn.DataParallel(model).cuda()


    # set dataloader
    if args.pred_vis:   
        dataset_eval = partdataset.PartImageNetSegDataset(args.eval_dir,transform=None,seg_fraction=1.0)
        dataset_eval.transform = create_transform(
                                        args.input_size,
                                        is_training=False,
                                        use_prefetcher=False,
                                        interpolation=args.interpolation,
                                        mean=args.mean,
                                        std=args.std,
                                        crop_pct=args.crop_pct
                                    )
    else:
        dataset_eval=ImageNet(root=args.test_dir, meta_file=args.test_file, transform=test_transform)

    sampler_eval=None
    n_gpu = torch.cuda.device_count()
    dataloader_eval = torch.utils.data.DataLoader(
        dataset=dataset_eval,
        batch_size=args.batch_size * n_gpu * 2,
        shuffle=True,
        num_workers=args.num_workers,
        sampler=sampler_eval,
        collate_fn=None,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    clean_acc=validate(model, dataloader_eval, args)
    print('*******Final Top1 acc of clean images is: {0:>7.4f}'.format(clean_acc['top1']))
    # for eps_int in [4]:
    if args.eval_eps < 0:
        if args.norm == "Linf":
            eps = 4 / 255
        elif args.norm == "L1":
            eps = 75
        elif args.norm == "L2":
            eps = 2
    else:
        if args.norm == "Linf":
            eps = args.eval_eps / 255
        else:
            eps = args.eval_eps
    robust_acc=adv_validate(model, dataloader_eval, args, eps)
    print('*******Final Top1 acc of eps {0} is: {1:>7.4f}'.format(eps, robust_acc['top1']))

def validate(model, loader, args, log_suffix='clean acc'):
    batch_time_m = AverageMeter()
    top1_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1

    N_CLASSES = args.num_classes
    class_correct = list(0. for i in range(N_CLASSES))
    class_total = list(0. for i in range(N_CLASSES))

    for batch_idx, batchinfo in enumerate(loader):
        if args.model_type == "original":
            input, target = batchinfo
        elif args.model_type == "partmodel":
            if args.pred_vis:
                input, target, seg, filenames = batchinfo
                seg = seg.float()
                seg = seg.cuda()
            else:
                input, target = batchinfo

        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            if args.pred_vis:
                output, segment_result = model(input)
            else:
                output = model(input)

        acc1, _ = accuracy(output, target, topk=(1, 5))

        top1_m.update(acc1.item(), input.size(0))

        batch_time_m.update(time.time() - end)
        end = time.time()

        log_name = 'Test ' + log_suffix
        print(
            '{0}: [{1:>4d}/{2}]  '
            'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
            'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '.format(
                log_name, batch_idx, last_idx, batch_time=batch_time_m, top1=top1_m))


    metrics = OrderedDict([('top1', top1_m.avg)])

    return metrics

def adv_validate(model, loader, args, eps, log_suffix='robust acc'):
    batch_time_m = AverageMeter()
    top1_m = AverageMeter()
    model.eval()

    # set attackers
    attackers={}
    for attack_type in args.attack_types:
        if attack_type == 'autoattack':
            from autoattack import AutoAttack
            adversary = AutoAttack(model, norm=args.norm, eps=eps, version='standard')
            attackers[attack_type]=adversary
        elif attack_type == 'autoattack-simple':
            from autoattack import AutoAttack
            adversary = AutoAttack(model, norm=args.norm, eps=eps, version='standard')
            adversary.attacks_to_run = ['apgd-ce']
            attackers[attack_type]=adversary

    end = time.time()
    last_idx = len(loader) - 1

    N_CLASSES = args.num_classes
    class_correct = list(0. for i in range(N_CLASSES))
    class_total = list(0. for i in range(N_CLASSES))

    for batch_idx, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        batch_size=target.size(0)
        robust_flag=torch.ones_like(target).cuda()

        # attack
        for attack_type in args.attack_types:
            if attack_type == 'autoattack':
                x_adv = attackers[attack_type].run_standard_evaluation(input, target, bs=target.size(0)) 
            

            with torch.no_grad():
                output = model(x_adv.detach())
                _, label=torch.max(output, dim=1)
                robust_label = label == target
                robust_flag = torch.logical_and(robust_flag, robust_label)

                if args.per_class_acc:
                    c = torch.eq(label, target).squeeze()
                    for i in range(input.shape[0]):
                        class_label = target[i].item()
                        class_correct[class_label] += c[i].item()
                        class_total[class_label] += 1
        
        
        acc1 = robust_flag.float().sum(0) * 100. / batch_size

        top1_m.update(acc1.item(), output.size(0))
        batch_time_m.update(time.time() - end)
        end = time.time()

        log_name = 'Test ' + log_suffix + ' of eps ' + str(eps)
        print(
            '{0}: [{1:>4d}/{2}]  '
            'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
            'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '.format(
                log_name, batch_idx, last_idx, batch_time=batch_time_m, top1=top1_m))



    metrics = OrderedDict([('top1', top1_m.avg)])

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Robust evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    opt = vars(args)
    if args.configs:
        opt.update(yaml.load(open(args.configs), Loader=yaml.FullLoader))
    
    args = argparse.Namespace(**opt)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    print("*" * 30)
    print(args_text)
    print("*" * 30)

    if args.num_classes == 1000 and args.per_class_acc:
        from data.classinfo_1k_pp import CLASSES
        classes = list(CLASSES.keys())
    main(args)