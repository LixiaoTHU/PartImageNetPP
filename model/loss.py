import torch.nn as nn
import torch
import torch.nn.functional as F
from timm.loss import JsdCrossEntropy, BinaryCrossEntropy, LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import NativeScaler

from contextlib import suppress

class SoftFocalLoss(nn.Module):
    def __init__(self, gamma=0):
        super(SoftFocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = torch.nn.CrossEntropyLoss(reduction="none")
    
    def forward_single(self, pred, gold, weight=None):
        B, C, H, W = pred.shape
        pred = pred.permute(0,2,3,1)
        gold = gold.permute(0,2,3,1)
        pred = pred.reshape(B*H*W, C)
        gold = gold.reshape(B*H*W, C)

        logp = self.ce(pred, gold)
        p = torch.exp(-logp)
        loss = ((1 - p) ** self.gamma) * logp
        loss = loss.reshape(B, H, W)
        if weight is not None:
            loss = loss * weight.unsqueeze(1).unsqueeze(2)
        loss = loss.mean()
        return loss

    def forward(self, pred, gold, weight=None):
        if isinstance(pred, list):
            losses = []
            for i in range(len(pred)):
                subpred = pred[i]
                subgold = gold[i]
                loss = self.forward_single(subpred, subgold, weight=weight)
                losses.append(loss)
            loss = sum(losses) / len(losses)
        else:
            print("not implemented yet!")
            exit(0)
        return loss


class SoftSigmoidLoss(nn.Module):
    def __init__(self, gamma=0):
        super(SoftSigmoidLoss, self).__init__()
        self.ce = torch.nn.BCEWithLogitsLoss(reduction="none")
    
    def forward_single(self, pred, gold, weight=None):
        B, C, H, W = pred.shape
        # pred = pred.permute(0,2,3,1)
        # gold = gold.permute(0,2,3,1)
        # pred = pred.reshape(B*H*W, C)
        # gold = gold.reshape(B*H*W, C)


        loss = self.ce(pred[:,1:], gold[:,1:]) # ignore background
        loss = torch.sum(loss, dim=1)
        if weight is not None:
            loss = loss * weight.unsqueeze(1).unsqueeze(2)
        loss = loss.mean()
        return loss

    def forward(self, pred, gold, weight=None):
        if isinstance(pred, list):
            losses = []
            for i in range(len(pred)):
                subpred = pred[i]
                subgold = gold[i]
                loss = self.forward_single(subpred, subgold, weight=weight)
                losses.append(loss)
            loss = sum(losses) / len(losses)
        else:
            print("not implemented yet!")
            exit(0)
        return loss


def build_loss(args, mixup_fn):
    if mixup_fn is not None:
        # smoothing and downsampling are handled with mixup target transform which outputs sparse, soft targets
        train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()
    
    return train_loss_fn, validate_loss_fn

def build_seg_loss(args, mixup_fn):
    if mixup_fn is not None:
        # smoothing and downsampling are handled with mixup target transform which outputs sparse, soft targets
        if args.dataset_version == "single":
            train_loss_fn = SoftFocalLoss(2)
            validate_loss_fn = SoftFocalLoss(2)
        elif args.dataset_version == "multi":
            # train_loss_fn = SoftSigmoidLoss()
            # validate_loss_fn = SoftSigmoidLoss()
            train_loss_fn = SoftFocalLoss(2)
            validate_loss_fn = SoftFocalLoss(2)
    else:
        print("Not Implemented yet!")
        exit(0)
    # elif args.smoothing:
    #      train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    # else:
    #     train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.cuda()
    validate_loss_fn = validate_loss_fn.cuda()
    
    return train_loss_fn, validate_loss_fn
