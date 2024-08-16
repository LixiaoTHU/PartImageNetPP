import torch
import torch.nn.functional as F

def downsample_seglabel(ori_seg, args, return_pixel_seg=True):

    if ori_seg.shape[1] > 1: # could be improved
        ori_seg[ori_seg == args.seg_classes] = -1
        ori_seg = torch.max(ori_seg, dim=1, keepdim=True)[0]


    seglist, pixel_seg_list = [], []

    seg = F.interpolate(ori_seg, size=args.output_size[0], mode="nearest")
    pixel_seg = seg.clone()  
    N, _, H, W = seg.shape
    seg = seg.long().view(-1, 1)
    seg = torch.full((seg.shape[0], args.seg_classes), 0, device=seg.device).scatter_(1, seg, 1).float()
    seg = seg.view(N, H, W, args.seg_classes).permute(0, 3, 1, 2)
    seglist.append(seg)
    pixel_seg_list.append(pixel_seg.long().squeeze())

    if len(args.output_size) > 1:
        for item in args.output_size[1:]:
            down_seg = F.interpolate(seg, size=item, mode="bilinear", align_corners=False)
            pixel_seg = F.interpolate(pixel_seg, size=item, mode="nearest")  

            seglist.append(down_seg)
            pixel_seg_list.append(pixel_seg.long().squeeze())


    if return_pixel_seg:
        return seglist, pixel_seg_list
    else:
        return seglist