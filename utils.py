import os
import torch
import torch.nn as nn
import numpy as np
import random
import logging
import functools
import sys


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print



def distributed_init(args):
    if args.distributed:
        if args.local_rank != -1: # for distributed launch
            args.rank=args.local_rank
            args.device_id=args.local_rank
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            args.rank=int(os.environ['SLURM_PROCID'])
            args.device_id=args.rank % torch.cuda.device_count()
        torch.cuda.set_device(args.device_id)
        torch.distributed.init_process_group(backend=args.dist_backend,init_method=args.dist_url,world_size=args.world_size,rank=args.rank)
        setup_for_distributed(args.rank==0)
    else:
        args.local_rank=0
        args.world_size=1
        args.rank=0
        args.device_id=0
        torch.cuda.set_device(args.device_id)

def random_seed(seed=0, rank=0):
    seed = seed + rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_random_states():
    state = np.random.get_state()
    torch_state = torch.get_rng_state()
    rstate = random.getstate()
    return state, torch_state, rstate

def set_random_states(states):
    state, torch_state, rstate = states
    np.random.set_state(state)
    torch.set_rng_state(torch_state)
    random.setstate(rstate)

@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name='', default_level=logging.INFO, save_log = True):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(default_level)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(default_level)
        console_handler.setFormatter(
            logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    # create file handlers
    if save_log:
        torch.distributed.barrier()
        file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}.txt'), mode='a')
        file_handler.setLevel(default_level)
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(file_handler)

    return logger


class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        def normalize_fn(tensor, mean, std):
            """Differentiable version of torchvision.functional.normalize"""
            # here we assume the color channel is in at dim=1
            mean = mean[None, :, None, None]
            std = std[None, :, None, None]
            return tensor.sub(mean).div(std)

        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)

def compute_seg_acc(segment_result, pixel_seg, k=1, args=None):
    if isinstance(segment_result, list):
        segment_result = segment_result[0]
        pixel_seg = pixel_seg[0]
    
    # if args is not None and args.dataset_version == "multi":
    #     segment_result = segment_result[:, 1:]
    #     pixel_seg = pixel_seg - 1
        
        
    _, tk = torch.topk(segment_result, k, dim=1)
    pixel_seg[pixel_seg==0] = -1
    correct_pixels = torch.eq(pixel_seg[:, None, ...], tk).any(dim=1)
    cp = correct_pixels.float()
    top_k_acc = cp.sum() / ((pixel_seg>0).sum() + 1e-5) * 100
    return top_k_acc

def unwrap_checkpoint(checkpoint):
    if 'state_dict_ema' in checkpoint:
        checkpoint = checkpoint['state_dict_ema']
    elif 'model' in checkpoint:
        checkpoint = checkpoint['model']
    else:
        checkpoint = checkpoint
    return checkpoint