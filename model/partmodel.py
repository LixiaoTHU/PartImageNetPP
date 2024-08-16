import torch
import torch.nn as nn
import torch.nn.functional as F



class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x



class PoolingFeatureExtractor(nn.Module):
    """Feature extraction layer for Downsampled part model."""

    def __init__(self, no_bg: bool, from_logits=True) -> None:
        super().__init__()
        self._no_bg: bool = no_bg
        self.from_logits = from_logits

    def forward(self, logits_masks: torch.Tensor) -> torch.Tensor:
        # masks: [N, num_segs (including background), H, W]
        if self.from_logits:
            masks = F.softmax(logits_masks, dim=1)
        else:
            masks = logits_masks
        # Remove background
        if self._no_bg:
            masks = masks[:, 1:]
        return masks




class TwostagePartModel(nn.Module):
    def __init__(self, segmenter, seg_classes, num_classes, from_logits=True) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.seg_classes = seg_classes

        self.segmenter = segmenter
        del self.segmenter.fc
        self.segmenter_head = nn.Sequential( # to design the head of the segmenter
                    nn.GELU(),
                    nn.BatchNorm2d(segmenter.num_features),
                    SeparableConv2d(segmenter.num_features, seg_classes, kernel_size=3, stride=1, padding=1, bias = False)
                                )
        
        no_bg = False
        segment_size, downsample_size = 7, 4
        self.pooling_feature_extractor = PoolingFeatureExtractor(no_bg=no_bg, from_logits=from_logits)
        input_dim = seg_classes - (1 if no_bg else 0)
        self.classifer = nn.Sequential( # Carlini version (adjustment)
            nn.AdaptiveAvgPool2d((downsample_size, downsample_size)),
            nn.Conv2d(input_dim, segmenter.num_features // 4, kernel_size=(downsample_size, downsample_size)),
            nn.BatchNorm2d(segmenter.num_features // 4),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(segmenter.num_features // 4, segmenter.num_features),
            nn.BatchNorm1d(segmenter.num_features),
            nn.GELU(),
            nn.Linear(segmenter.num_features, num_classes),
        )


    def forward(self, x, return_seg=False):
        x = self.segmenter.forward_features(x)
        segment_result = self.segmenter_head(x)
        classfier_input = self.pooling_feature_extractor(segment_result)
        cls_result = self.classifer(classfier_input)
        if return_seg:
            return cls_result, [segment_result]
        else:
            return cls_result









class MPM(nn.Module):
    def __init__(self, segmenter, seg_classes, num_classes, dtype = "resnet", level = 3, down = True) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.seg_classes = seg_classes
        self.dtype = dtype
        self.down = down

        self.segmenter = segmenter
        self.level = level
        if hasattr(self.segmenter, 'head'):
            del self.segmenter.head
        elif hasattr(self.segmenter, "fc"):
            del self.segmenter.fc

        if self.dtype == "resnet":
            compact_ratio = 4
            self.num_features = segmenter.num_features // compact_ratio
            self.build_compact(compact_ratio=compact_ratio)

        self.build_fpn()
        
        if  self.dtype == "resnet":
            self.classifer = nn.Sequential( 
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(segmenter.num_features, num_classes),
                )

    def build_fpn(self):

        self.segmenter_head1 = nn.Sequential(
                        nn.GELU(),
                        SeparableConv2d(self.num_features, self.seg_classes, kernel_size=3, stride=1, padding=1, bias = False)
                                    )
        

        self.up_layer1 = nn.Sequential(
                            nn.GELU(),
                            SeparableConv2d(self.num_features, self.num_features // 2, kernel_size=3, stride=1, padding=1, bias = False)
                                        )

    

        self.segmenter_head2 = nn.Sequential(
                        nn.GELU(),
                        SeparableConv2d(self.num_features // 2, self.seg_classes, kernel_size=3, stride=1, padding=1, bias = False)
                                    )

    


        self.up_layer2 = nn.Sequential(
                        nn.GELU(),
                        SeparableConv2d(self.num_features // 2, self.num_features // 4, kernel_size=3, stride=1, padding=1, bias = False)
                                    )

        

        self.segmenter_head3 = nn.Sequential(
                        nn.GELU(),
                        SeparableConv2d(self.num_features // 4, self.seg_classes, kernel_size=3, stride=1, padding=1, bias = False)
                                    )
    
    def build_compact(self, compact_ratio):
        ori_features = self.num_features * compact_ratio
        self.cp1 = nn.Sequential(
                            nn.Conv2d(ori_features, self.num_features, kernel_size=1, stride=1, padding=0, bias = False),
                            nn.BatchNorm2d(self.num_features)
        )
        self.cp2 = nn.Sequential(
                            nn.Conv2d(ori_features // 2, self.num_features // 2, kernel_size=1, stride=1, padding=0, bias = False),
                            nn.BatchNorm2d(self.num_features // 2)
        )
        self.cp3 = nn.Sequential(
                            nn.Conv2d(ori_features // 4, self.num_features // 4, kernel_size=1, stride=1, padding=0, bias = False),
                            nn.BatchNorm2d(self.num_features // 4)
        )

    
    
    def _upsample_add(self, x, y):
        _, _, H, W = y.shape  # b c h w
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward_features(self, x, return_features=False):
        feature_list = []
        x = self.segmenter.conv1(x)
        x = self.segmenter.bn1(x)
        x = self.segmenter.act1(x)
        x = self.segmenter.maxpool(x)
        x = self.segmenter.layer1(x)
        x = self.segmenter.layer2(x)
        if return_features:
            feature_list.append(self.cp3(x))
        x = self.segmenter.layer3(x)
        if return_features:
            feature_list.append(self.cp2(x))
        x = self.segmenter.layer4(x)
        if return_features:
            feature_list.append(self.cp1(x))
        
        if return_features:
            return x, feature_list
        else:
            return x

    def forward(self, x, return_seg=False):
        x, feature_list = self.forward_features(x, return_features=True)

        
        cls_result = self.classifer(x)


        if return_seg:
            if self.dtype == "resnet":
                x = feature_list[-1]
            sr1 = self.segmenter_head1(x)
            if self.down:
                mr2 = self._upsample_add(self.up_layer1(x), feature_list[-2])
                sr2 = self.segmenter_head2(mr2)
                mr3 = self._upsample_add(self.up_layer2(mr2), feature_list[-3])
                sr3 = self.segmenter_head3(mr3)
            else:
                sr2 = self.segmenter_head2(feature_list[-2])
                sr3 = self.segmenter_head3(feature_list[-3])

            if self.level == 3:
                return cls_result, [sr3, sr2, sr1]
            elif self.level == 2:
                return cls_result, [sr2, sr1]
            elif self.level == 1:
                return cls_result, [sr1]
        else:
            return cls_result