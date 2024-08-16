# PartImageNetPP

The official implementation of the paper "PartImageNet++ Dataset: Scaling up Part-based Models for Robust Recognition" (ECCV 2024).

The dataset is currently available at https://huggingface.co/datasets/lixiao20/PartImageNetPP

## Usage

Train a model, e.g., an MPM with ResNet-50-GELU as the backbone by AT:

```bash
python -m torch.distributed.launch --nproc_per_node=8 adv_train.py --configs=./train_configs/1k_pp/resnet50_part_pp_init.yaml

python -m torch.distributed.launch --nproc_per_node=8 adv_train.py --configs=./train_configs/1k_pp/resnet50_part_pp_at.yaml
```


The needed ResNet-50 checkpoint can be downloaded from https://download.pytorch.org/models/resnet50-11ad3fa6.pth. The required dataset can be downloaded from huggingface and processed with the method described in the paper.


Evaluate the trained models with Autoattack under different attack threats, e.g., the MPM model under L2 attack with $\epsilon = 2$ :

```bash
python eval.py \
    --configs=./train_configs/1k_pp/resnet50_part_pp_at.yaml \
    --checkpoint=/home/user/ssd/*/checkpoint.pth \
    --attack_types="autoattack" \
    --norm "L2" \
    --eval-eps 2
```

Please refer to the `train_configs` folder for more training configurations.

## Requisite

This code is implemented based on timm, and we have tested the code under the following environment settings (see https://github.com/huggingface/pytorch-image-models to install timm):


- timm == 0.9.7
- pytorch == 1.12.1
- torchvision == 0.13.1


## The released checkpoints

https://drive.google.com/drive/folders/1chr4IKr7JqnS02pVwZV6yNAgxAP58gg9?usp=sharing

If you find that our work is helpful to you, please star this project and consider cite:
```
@inproceedings{li2024pinpp,
  author = {Li, Xiao and Liu, Yining and Dong, Na and Qin, Sitian and Hu, Xiaolin},
  title = {PartImageNet++ Dataset: Scaling up Part-based Models for Robust Recognition},
  booktitle={European conference on computer vision},
  year = {2024},
  organization={Springer}
}
```