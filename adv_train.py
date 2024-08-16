# basic modules
import warnings
warnings.filterwarnings("ignore")
import argparse
import time
import yaml, os
import logging
from collections import OrderedDict
import csv

# torch related modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as NativeDDP

# timm functions
from timm.models import resume_checkpoint, load_checkpoint, model_parameters
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.utils import ModelEmaV2, distribute_bn, AverageMeter, reduce_tensor, dispatch_clip_grad, accuracy, get_outdir, CheckpointSaver, update_summary



# in functions
from utils import distributed_init, random_seed, create_logger, compute_seg_acc, unwrap_checkpoint
from model import *
from data import build_dataset, build_part_dataset
from adv_utils import adv_generator

from torch.cuda.amp import autocast, GradScaler

def get_args_parser():
    parser = argparse.ArgumentParser('Robust training script', add_help=False)
    parser.add_argument('--configs', default='', type=str)
    parser.add_argument('--model-type', type=str, default='original', choices=['original', 'partmodel'],)
    parser.add_argument('--partmodel-type', type=str, default='default')
    parser.add_argument('--output-size', type=list, default=[[7, 7]])
    parser.add_argument('--lambdas', type=float, default=1.0, help='seg and cls weight')
    parser.add_argument('--dataset-version', default="single", type=str, choices=["single", "multi"])
    parser.add_argument('--freeze-epoch', type=float, default=0, help='freeze segmenter to finetune head')
    parser.add_argument('--evaluate', default=False, help='evaluate only')

    #* distributed setting
    parser.add_argument('--distributed', default=True)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--rank', default=-1, type=int, help='rank')
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-backend', default='nccl', help='backend used to set up distributed training')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--dist-bn', type=str, default='reduce', help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
    
    #* model parameters
    parser.add_argument('--model', default='resnet50', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--num-classes', default=1000, type=int, help='number of classes')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain', default='', help='pretrain from checkpoint')
    parser.add_argument('--warmup', default=False, help='freeze segmenter parameters ')
    parser.add_argument('--full-pretrain', default=False, help='Pre-trained parameters include the entire model')
    parser.add_argument('--channels-last', action='store_true', default=False,
                        help='Use channels_last memory layout (opt)')
    
    parser.add_argument('--global-pool', default=None, type=str, metavar='POOL',
                        help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None. (opt)')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path-rate', type=float, default=0.0, metavar='PCT', help='Drop path rate (default: 0.)')


    #* Optimizer parameters
    parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=2e-5,
                        help='weight decay (default: 0.0001)')

    #* Learning rate schedule parameters
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--sched-on-updates', default=True, type=bool, metavar='SCHED_ON_UPDATES')
    parser.add_argument('--lrb', type=float, default=None, metavar='LR',
                        help='base learning rate (default: 5e-4)')
    parser.add_argument('--lr', type=float, default=None, help='actual learning rate after linear scaling')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                        help='learning rate cycle len multiplier (default: 1.0)')
    parser.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                        help='amount to decay each learning rate cycle (default: 0.5)')
    parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                        help='learning rate cycle limit, cycles enabled if > 1')
    parser.add_argument('--lr-k-decay', type=float, default=1.0,
                        help='learning rate k-decay for cosine/poly (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                        help='warmup learning rate (default: 0.0001)')
    parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                        help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=2, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    #* dataset parameters
    parser.add_argument('--batch-size', default=64, type=int)    # batch size per gpu
    parser.add_argument('--train-dir', default='', type=str, help='train dataset path')
    parser.add_argument('--eval-dir', default='', type=str, help='validation dataset path')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--crop-pct', default=0.875, type=float,
                        metavar='N', help='Input image center crop percent (for validation only)')
    parser.add_argument('--interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--mean', type=float, nargs='+', default=(0.485, 0.456, 0.406), metavar='MEAN',
                        help='Override mean pixel value of dataset')
    parser.add_argument('--std', type=float, nargs='+', default=(0.229, 0.224, 0.225), metavar='STD',
                        help='Override std deviation of dataset')
    
    #* Augmentation & regularization parameters
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--hflip', type=float, default=0.5,
                        help='Horizontal flip training aug probability')
    parser.add_argument('--vflip', type=float, default=0.,
                        help='Vertical flip training aug probability')
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". (default: None)'),
    parser.add_argument('--aug-splits', type=int, default=0,
                        help='Number of augmentation splits for split bn (default: 0, valid: 0 or >=2)')

    # random erase
    parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                        help='Random erase prob (default: 0.)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                        help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='random',
                        help='Training interpolation (random, bilinear, bicubic default: "random")')

    #* ema
    parser.add_argument('--model-ema', action='store_true', default=False,
                        help='Enable tracking moving average of model weights')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                        help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
    parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                        help='decay factor for model weights moving average (default: 0.9998)')

    # misc
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
    parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
    parser.add_argument('--max-history', type=int, default=5, help='how many recovery checkpoints')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='how many training processes to use (default: 4)')
    parser.add_argument('--output-dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--eval-metric', default='advtop1', type=str, metavar='EVAL_METRIC',
                        help='Best metric (default: "top1")')
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # bn
    parser.add_argument('--bn-momentum', type=float, default=None, help='BatchNorm momentum override (if not None)')
    parser.add_argument('--bn-eps', type=float, default=None, help='BatchNorm epsilon override (if not None)')

    # advtrain
    parser.add_argument('--advtrain', default=False, help='if use advtrain')
    parser.add_argument('--attack-criterion', type=str, default='regular', choices=['regular', 'smooth', 'mixup'], help='default args for: adversarial training')
    parser.add_argument('--attack-eps', type=float, default=4.0/255, help='attack epsilon.')
    parser.add_argument('--attack-step', type=float, default=8.0/255/3, help='attack epsilon.')
    parser.add_argument('--attack-it', type=int, default=3, help='attack iteration')

    parser.add_argument('--amp', type=int, default=False, help='automix precision training')
    parser.add_argument('--use-object-seg', default=False, help='if segmask use object label')
    parser.add_argument('--use-whether-pseudo', default=False, help='if use pseudo label')
    parser.add_argument('--ignore-blank', default=False, help='if ignore blank')
    parser.add_argument('--pseudo-weight', type=float, default=0.5, help='pseudo label weight')
    parser.add_argument('--level', type=int, default=3, help='part level')
    parser.add_argument('--down', type=bool, default=True, help='downsample')

    # advprop
    parser.add_argument('--advprop', default=False, help='if use advprop')
    
    return parser


def main(args, args_text):
    # distributed settings and logger
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1
    distributed_init(args)
    output_dir = args.output_dir
    if args.rank == 0:
        output_dir = get_outdir(args.output_dir, inc=True)

    _logger = create_logger(output_dir, dist_rank=args.rank, name='main_train', default_level=logging.INFO)

    # fix the seed for reproducibility
    random_seed(args.seed, args.rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # build model
    if args.model_type == "original":
        model = build_model(args, _logger)
        if args.pretrain:
            checkpoint = torch.load(args.pretrain, map_location='cpu')
            # load_checkpoint(model, checkpoint_path=args.pretrain, filter_fn=timm.models.swin_transformer.checkpoint_filter_fn)
            # torch.save(model.state_dict(), os.path.join(output_dir, 'pretrain.pth'))
            checkpoint = unwrap_checkpoint(checkpoint) #
            model.load_state_dict(checkpoint)
    elif args.model_type == "partmodel":
        
        model = build_part_model(args, _logger)
        if args.pretrain:
            if args.full_pretrain or args.evaluate:
                ckpt_dir = args.pretrain
                ckpt = torch.load(ckpt_dir, map_location='cpu')['state_dict']
                model.load_state_dict(ckpt)
            else:
                def remove_keys_with_head(od):
                    keys_to_remove = [key for key in od if 'head' in key or 'fc.weight' in key or 'fc.bias' in key]
                    for key in keys_to_remove:
                        # if args.num_classes < 1000:
                        del od[key]
                    return od
                checkpoint = torch.load(args.pretrain, map_location='cpu')
                checkpoint = unwrap_checkpoint(checkpoint)
                checkpoint = remove_keys_with_head(checkpoint)
                model.segmenter.load_state_dict(checkpoint)
    else:
        print("Not Implemented!")
        exit(0)

    # create optimizer
    optimizer=None
    if args.lr is None:
        args.lr = args.lrb * args.batch_size * args.world_size / 512
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None

    # resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=optimizer,
            loss_scaler=None,
            log_info = args.rank == 0)

    # setup ema
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)


    # setup distributed training
    if args.distributed:
        _logger.info("Using native Torch DistributedDataParallel.")
        if args.warmup:
            model = NativeDDP(model, device_ids=[args.device_id], find_unused_parameters=True)
        else:
            model = NativeDDP(model, device_ids=[args.device_id], find_unused_parameters=True)



    # create the train and eval dataloaders
    if args.model_type == "original":
        loader_train, loader_eval, mixup_fn = build_dataset(args)
    elif args.model_type == "partmodel":
        loader_train, loader_eval, mixup_fn = build_part_dataset(args)

    # setup learning rate schedule and starting epoch
    # lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    updates_per_epoch = len(loader_train)
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(args),
        updates_per_epoch=updates_per_epoch,
    )

    start_epoch = 0 if resume_epoch is None else resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        if args.sched_on_updates:
            lr_scheduler.step_update(start_epoch * updates_per_epoch)
        else:
            lr_scheduler.step(start_epoch)
    _logger.info('Scheduled epochs: {}'.format(num_epochs))



    # setup loss function
    train_loss_fn, validate_loss_fn = build_loss(args, mixup_fn)
    if args.model_type == "partmodel":
        train_seg_loss_fn, validate_seg_loss_fn = build_seg_loss(args, mixup_fn)
    else:
        train_seg_loss_fn, validate_seg_loss_fn = None, None

    # saver
    eval_metric = args.eval_metric
    saver = None
    best_metric = None
    best_epoch = None
    if args.rank == 0:
        decreasing=True if eval_metric=='loss' else False
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=args.max_history)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    if not args.evaluate:
        # start training
        _logger.info(f"Start training for {args.epochs} epochs")
        for epoch in range(start_epoch, args.epochs):
            if hasattr(loader_train, 'sampler'):
                loader_train.sampler.set_epoch(epoch)
            # one epoch training
            train_metrics = train_one_epoch(
                    epoch, model, loader_train, optimizer, train_loss_fn, args, seg_loss_fn=train_seg_loss_fn,
                    lr_scheduler=lr_scheduler, saver=saver, model_ema=model_ema, mixup_fn=mixup_fn, scaler=scaler, _logger=_logger)
            # train_metrics = None

            # distributed bn sync
            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                _logger.info("Distributing BatchNorm running means and vars")
                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            # calculate evaluation metric
            if model_ema is None or epoch % 5 == 0:
                eval_metrics = validate(model, loader_eval, validate_loss_fn, validate_seg_loss_fn, args, _logger=_logger)

            # model ema update
            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                ema_eval_metrics = validate(model_ema.module, loader_eval, validate_loss_fn, validate_seg_loss_fn, args, log_suffix=' (EMA)', _logger=_logger)
                eval_metrics = ema_eval_metrics


            # lr_scheduler update
            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            # output summary.csv
            if output_dir is not None and args.rank == 0:
                update_summary(
                    epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                    write_header=best_metric is None)

            # save checkpoint, print best metric
            if saver is not None:
                best_metric, best_epoch = saver.save_checkpoint(epoch, eval_metrics[eval_metric])
            torch.distributed.barrier()
        if best_metric is not None:
            _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))
            
    else:
        eval_metrics = validate(model, loader_eval, validate_loss_fn, validate_seg_loss_fn, args, _logger=_logger)
        rowd = OrderedDict([('eval_' + k, v) for k, v in eval_metrics.items()])
        with open(os.path.join(output_dir, 'eval_summary.csv'), mode='a') as cf:
            dw = csv.DictWriter(cf, fieldnames=rowd.keys())
            dw.writeheader()
            dw.writerow(rowd)


def train_one_epoch(epoch, model, loader, optimizer, loss_fn, args, seg_loss_fn = None, 
        lr_scheduler=None, saver=None, model_ema=None, mixup_fn=None, scaler=None, _logger=None):
    # mixup setting
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if mixup_fn is not None:
            mixup_fn.mixup_enabled = False
    

    if args.warmup:
        for param in model.module.segmenter.parameters():
            param.requires_grad = False

    # statistical variables
    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    num_epochs = args.epochs + args.cooldown_epochs

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)

    # warm up attack step
    args.attack_step = args.attack_eps * 2 / args.attack_it
    att_step = args.attack_step * min(epoch, args.warmup_epochs) / (args.warmup_epochs + 1e-10)
    att_eps=args.attack_eps
    att_it=args.attack_it

    cnt = 0
    for batch_idx, batchinfo in enumerate(loader):
        last_batch = batch_idx == last_idx
        if args.model_type == "original":
            input, target = batchinfo
        elif args.model_type == "partmodel":
            if args.use_whether_pseudo or args.ignore_blank:
                input, target, seg, filenames, pseudo_weight = batchinfo
                pseudo_weight = pseudo_weight.cuda(non_blocking=True)

                if args.mixup_prob > 0:
                    r = pseudo_weight.flip(0)
                    new_pseudo_weight = (r + pseudo_weight) / 2
                    new_pseudo_weight[r < 1e-4] = 0
                    new_pseudo_weight[pseudo_weight < 1e-4] = 0

                    pseudo_weight = new_pseudo_weight

                # cnt += torch.sum(pseudo_weight).item()
                # if batch_idx % 50 == 0:
                #     print("pseudo_weight: ", cnt)
            else:
                input, target, seg, filenames = batchinfo
                pseudo_weight = None
            seg = seg.float()
            seg = seg.cuda(non_blocking=True)
            
        input, target = input.cuda(non_blocking=True), target.cuda(non_blocking=True)



        if args.model_type == "original":
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        elif args.model_type == "partmodel":
            if mixup_fn is not None:
                input, target, seg = mixup_fn(input, target, seg)
                if not isinstance(seg, list):
                    seg = [seg]
            else:
                target = target.long().view(-1, 1)
                target = torch.full((target.shape[0], args.num_classes), 0, device=target.device).scatter_(1, target, 1)
                seg = downsample_seglabel(seg, args)
        

        if args.channels_last:
            input=input.contiguous(memory_format=torch.channels_last)
        
        data_time_m.update(time.time() - end)

        
        # generate adv input
        if args.advtrain:
            input_advtrain = adv_generator(args, input, target, model, att_eps, att_it, att_step, 
                                            random_start=False, attack_criterion=args.attack_criterion, scaler=scaler, _logger=_logger)
            

        # forward
        with autocast(enabled=args.amp):
            if args.advtrain:
                if args.model_type == "original":
                    output = model(input_advtrain)
                    loss = loss_fn(output, target)
                elif args.model_type == "partmodel":
                    output, segment_result = model(input_advtrain, return_seg=True)
                    loss = loss_fn(output, target)
                    loss_seg = seg_loss_fn(segment_result, seg, weight=pseudo_weight)
                    loss = loss + args.lambdas * loss_seg
            else:
                if args.model_type == "original":
                    output = model(input)
                    loss = loss_fn(output, target)
                elif args.model_type == "partmodel":
                    output, segment_result = model(input, return_seg=True)
                    loss = loss_fn(output, target)
                    loss_seg = seg_loss_fn(segment_result, seg, weight=pseudo_weight)
                    loss = loss + args.lambdas * loss_seg
        
        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        if args.amp:
            scaler.scale(loss).backward(create_graph=second_order)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward(create_graph=second_order)
            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))

            _logger.info(
            'Train: [{}/{}] [{:>4d}/{} ({:>3.0f}%)]  '
            'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
            'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
            '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
            'LR: {lr:.3e}  '
            'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                epoch, num_epochs,
                batch_idx, len(loader),
                100. * batch_idx / last_idx,
                loss=losses_m,
                batch_time=batch_time_m,
                rate=input.size(0) * args.world_size / batch_time_m.val,
                rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                lr=lr,
                data_time=data_time_m))

        # save checkpoint
        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        # update lr scheduler
        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])

def validate(model, loader, loss_fn, seg_loss_fn, args, log_suffix='', _logger=None):
    batch_time_m = AverageMeter()
    clf_losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    clf_adv_losses_m = AverageMeter()
    adv_top1_m = AverageMeter()
    adv_top5_m = AverageMeter()

    if args.model_type == "partmodel":
        seg_losses_m = AverageMeter()
        seg_adv_losses_m = AverageMeter()
        pixel_acc_m = AverageMeter()
        adv_pixel_acc_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1

    for batch_idx, batchinfo in enumerate(loader):
        last_batch = batch_idx == last_idx
        if args.model_type == "original":
            input, target = batchinfo
        elif args.model_type == "partmodel":
            input, target, seg, filenames = batchinfo
            seg = seg.float()
            seg = seg.cuda(non_blocking=True)
            seg, pixel_seg = downsample_seglabel(seg, args, return_pixel_seg=True)

        input, target = input.cuda(non_blocking=True), target.cuda(non_blocking=True)

            
        if args.channels_last:
            input=input.contiguous(memory_format=torch.channels_last)

        # normal eval process
        with torch.no_grad():
            if args.model_type == "original":
                output = model(input)
            elif args.model_type == "partmodel":    
                output, segment_result = model(input, return_seg=True)
            if isinstance(output, (tuple, list)):
                output = output[0]
            
            clf_loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.model_type == "partmodel":
                seg_loss = seg_loss_fn(segment_result, seg)

                # compute the top k predicted classes, per pixel, mask out the background class:
                top_k_acc = compute_seg_acc(segment_result, pixel_seg, args=args)

            if args.distributed:
                clf_reduced_loss = reduce_tensor(clf_loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)

                if args.model_type == "partmodel":
                    seg_reduced_loss = reduce_tensor(seg_loss.data, args.world_size)
                    pixel_acc = reduce_tensor(top_k_acc.data, args.world_size)
            else:
                clf_reduced_loss = clf_loss.data
                if args.model_type == "partmodel":
                    seg_reduced_loss = seg_loss.data
                    pixel_acc = top_k_acc.data

            torch.cuda.synchronize()

            # record normal results
            clf_losses_m.update(clf_reduced_loss.item(), input.size(0))
            if args.model_type == "partmodel":
                seg_losses_m.update(seg_reduced_loss.item(), input.size(0))
                pixel_acc_m.update(pixel_acc.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

        # adv eval process
        if args.advtrain:
            adv_input = adv_generator(args, input, target, model, args.attack_eps, 10, args.attack_eps/4, random_start=True, use_best=False, attack_criterion='regular')
            with torch.no_grad():
                if args.model_type == "original":
                    adv_output = model(adv_input)
                elif args.model_type == "partmodel":    
                    adv_output, adv_segment_result = model(adv_input, return_seg=True)
                if isinstance(adv_output, (tuple, list)):
                    adv_output = adv_output[0]
                
                clf_adv_loss = loss_fn(adv_output, target)
                adv_acc1, adv_acc5 = accuracy(adv_output, target, topk=(1, 5))

                if args.model_type == "partmodel": 
                    seg_adv_loss = seg_loss_fn(adv_segment_result, seg)
                    adv_top_k_acc = compute_seg_acc(adv_segment_result, pixel_seg, args=args)

                if args.distributed:
                    clf_adv_reduced_loss = reduce_tensor(clf_adv_loss.data, args.world_size)
                    adv_acc1 = reduce_tensor(adv_acc1, args.world_size)
                    adv_acc5 = reduce_tensor(adv_acc5, args.world_size)
                    if args.model_type == "partmodel": 
                        seg_adv_reduced_loss = reduce_tensor(seg_adv_loss.data, args.world_size)
                        adv_pixel_acc = reduce_tensor(adv_top_k_acc.data, args.world_size)
                else:
                    clf_adv_reduced_loss = clf_adv_loss.data
                    if args.model_type == "partmodel": 
                        seg_adv_reduced_loss = seg_adv_loss.data
                        adv_pixel_acc = adv_top_k_acc.data

                torch.cuda.synchronize()

                # record adv results
                clf_adv_losses_m.update(clf_adv_reduced_loss.item(), adv_input.size(0))
                if args.model_type == "partmodel": 
                    seg_adv_losses_m.update(seg_adv_reduced_loss.item(), adv_input.size(0))
                    adv_pixel_acc_m.update(adv_pixel_acc.item(), adv_input.size(0))
                adv_top1_m.update(adv_acc1.item(), adv_output.size(0))
                adv_top5_m.update(adv_acc5.item(), adv_output.size(0))


        batch_time_m.update(time.time() - end)
        end = time.time()

        if last_batch or batch_idx % args.log_interval == 0:
            log_name = 'Test' + log_suffix
            if args.model_type == "original":
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Classify_Loss: {clf_loss.val:>7.4f} ({clf_loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})  '
                    'Adv_Classify_Loss: {clf_adv_loss.val:>7.4f} ({clf_adv_loss.avg:>6.4f})  '
                    'AdvAcc@1: {adv_top1.val:>7.4f} ({adv_top1.avg:>7.4f})  '
                    'AdvAcc@5: {adv_top5.val:>7.4f} ({adv_top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        clf_loss=clf_losses_m, top1=top1_m, top5=top5_m, 
                        clf_adv_loss=clf_adv_losses_m, adv_top1=adv_top1_m, adv_top5=adv_top5_m))
            elif args.model_type == "partmodel":
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Classify_Loss: {clf_loss.val:>7.4f} ({clf_loss.avg:>6.4f})  '
                    'Seg_Loss: {seg_loss.val:>7.4f} ({seg_loss.avg:>6.4f})  '
                    'Adv_Classify_Loss: {clf_adv_loss.val:>7.4f} ({clf_adv_loss.avg:>6.4f})  '
                    'Adv_Seg_Loss: {seg_adv_loss.val:>7.4f} ({seg_adv_loss.avg:>6.4f})  '
                    'Pixel_Acc: {pixel_acc1.val:>7.4f} ({pixel_acc1.avg:>7.4f})  '
                    'Adv_Pixel_Acc: {adv_pixel_acc1.val:>7.4f} ({adv_pixel_acc1.avg:>7.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})  '
                    'AdvAcc@1: {adv_top1.val:>7.4f} ({adv_top1.avg:>7.4f})  '
                    'AdvAcc@5: {adv_top5.val:>7.4f} ({adv_top5.avg:>7.4f})  '.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        clf_loss=clf_losses_m, seg_loss=seg_losses_m, clf_adv_loss=clf_adv_losses_m, seg_adv_loss=seg_adv_losses_m,
                        pixel_acc1=pixel_acc_m, adv_pixel_acc1=adv_pixel_acc_m,
                        top1=top1_m, top5=top5_m, adv_top1=adv_top1_m, adv_top5=adv_top5_m))
    if args.model_type == "original":
        metrics = OrderedDict([('loss', clf_losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg), ('advloss', clf_adv_losses_m.avg), ('advtop1', adv_top1_m.avg), ('advtop5', adv_top5_m.avg)])
    elif args.model_type == "partmodel":
        metrics = OrderedDict([('clf_loss', clf_losses_m.avg), ('seg_loss', seg_losses_m.avg), 
                                ('clf_advloss', clf_adv_losses_m.avg), ('seg_advloss', seg_adv_losses_m.avg), ('pixel_acc', pixel_acc_m.avg), ('adv_pixel_acc', adv_pixel_acc_m.avg), 
                                ('top1', top1_m.avg), ('top5', top5_m.avg), ('advtop1', adv_top1_m.avg), ('advtop5', adv_top5_m.avg)])

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Robust training script', parents=[get_args_parser()])
    args = parser.parse_args()
    opt = vars(args)
    if args.configs:
        opt.update(yaml.load(open(args.configs), Loader=yaml.FullLoader))
    
    args = argparse.Namespace(**opt)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    main(args, args_text)