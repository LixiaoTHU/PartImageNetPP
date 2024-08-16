import torch
import os
from timm.data.distributed_sampler import OrderedDistributedSampler, RepeatAugSampler
from torchvision import datasets
from PIL import Image
import numpy as np
import random, copy
from torchvision import transforms
import torch.nn.functional as F

# for alignment bwtween images and part labels
from .timm_local.transforms_factory import create_transform
from .timm_local.mixup import Mixup
from timm.data.transforms import RandomResizedCropAndInterpolation, InterpolationMode
from .timm_local.auto_augment import RandAugment
from utils import set_random_states, get_random_states

class PartImageNetSegDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        transform=None,
        seg_fraction=1.0,
        seed=0,
        use_whether_pseudo=False,
        pseudo_weight=1.0,
        use_object_seg=False,
        ignore_blank=False
    ):
        """Load the processed PartImageNet dataset

        Args:
            root (str): Path to root directory
            split (str, optional): Data split to load. Defaults to 'train'.
            transform (optional): Transformations to apply to the images (and
                the segmentation masks if applicable). Defaults to None.
            use_label (bool, optional): Whether to yield class label. Defaults to False.
            seg_fraction (float, optional): Fraction of segmentation mask to
                provide. The dropped masks are set to all -1. Defaults to 1.
            seed (int, optional): Random seed. Defaults to 0.
        """
        self.root = root
        self.img_path = os.path.join(self.root, "img")
        self.seg_path = os.path.join(self.root, "seg")
        self.pseudo_path = os.path.join(self.root, "whether_pseudo")
        
        self.use_whether_pseudo = use_whether_pseudo
        self.pseudo_weight = pseudo_weight
        self.use_object_seg = use_object_seg
        self.ignore_blank = ignore_blank
        

        self.seed = seed
        self.transform = transform

        self.classes = self._list_classes(self.img_path)

        self.num_classes = len(self.classes)

        if self.num_classes == 1000:
            if "partimagenetpp" in self.root:
                from .classinfo_1k_pp import CLASSES

        self.classes_list = list(CLASSES.keys())
            
        self.num_seg_labels = sum([CLASSES[c] for c in self.classes])

        self.images, self.labels, self.masks, self.pseudos = self._get_data()


        # use only a fraction of the segmentation masks
        idx = np.arange(len(self.images))
        state = np.random.get_state()
        np.random.seed(self.seed)
        np.random.shuffle(idx)
        np.random.set_state(state)
        self.seg_drop_idx = idx[: int((1 - seg_fraction) * len(self.images))]


    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert("RGB")
        _mask = Image.open(self.masks[index])
        _label = self.labels[index]
        _filename = [self.images[index], self.masks[index]]

        if self.transform is not None:
            # ****************************************************
            # All now added data augmentation need to be rewrited
            # ****************************************************

            _img, _mask = self._augmentation(_img, _mask)
        

        if self.use_object_seg:
            # map _mask to object mask
            object_num = self.classes_list.index(self.images[index].rsplit('/')[-2]) + 1
            _mask = torch.where(_mask>0, object_num, 0)




        if self.ignore_blank or self.use_whether_pseudo:
            if _mask.sum() < 1e-5:
                _pseudo_weight = 0.0
            else:
                _pseudo_weight = 1.0

            if self.use_whether_pseudo:
                _pseudo = self.pseudos[index]
                if _pseudo == 1 and _pseudo_weight == 1.0:
                    _pseudo_weight = torch.tensor(self.pseudo_weight, dtype=torch.float32)

            
            _pseudo_weight = torch.tensor(_pseudo_weight, dtype=torch.float32)
            return _img, _label, _mask, _filename, _pseudo_weight

        else:
            return _img, _label, _mask, _filename

    def _augmentation(self, _img, _mask):


        aligned_transforms = [RandomResizedCropAndInterpolation, 
                                transforms.RandomHorizontalFlip,
                                transforms.Resize,
                                transforms.ToTensor,
                                transforms.CenterCrop
                            ]

        for t in self.transform.transforms:
            if isinstance(t, RandAugment): # modified
                _img, _mask = t(_img, _mask)
            else:
                states = get_random_states()
                _img = t(_img)
                set_random_states(states)
                if type(t) in aligned_transforms:
                    if isinstance(t, RandomResizedCropAndInterpolation) or isinstance(t, transforms.Resize):
                        interpolation = copy.copy(t.interpolation)
                        t.interpolation = InterpolationMode.NEAREST
                        _mask = t(_mask)
                        t.interpolation = interpolation
                    else:
                        _mask = t(_mask)

        return _img, _mask


    def _get_data(self):
        images, labels, masks, pseudos = [], [], [], []
        for l, label in enumerate(self.classes):
            c_img_path = os.path.join(self.img_path, label)
            c_part_path = os.path.join(self.seg_path, label)

            if self.use_whether_pseudo:
                c_pseudo_txt = os.path.join(self.pseudo_path, label + ".txt")
                pseudo_dict = self._load_pseudo_dict(c_pseudo_txt)


            imglist = os.listdir(c_img_path)
            images.extend([os.path.join(c_img_path, name) for name in imglist])
            masks.extend([os.path.join(c_part_path, name.split(".")[0] + ".png") for name in imglist])
            if self.use_whether_pseudo:
                pseudos.extend([pseudo_dict[name] for name in imglist])

            labels.extend([l] * len(imglist))

        labels = torch.tensor(labels, dtype=torch.long)
        return images, labels, masks, pseudos
    
    def _load_pseudo_dict(self, path):
        f = open(path, "r")
        lines = f.readlines()
        pseudo_dict = {}
        for line in lines:
            line = line.strip()
            name, label = line.split(" ")
            name = name.split("/")[-1]
            pseudo_dict[name] = int(label)
        return pseudo_dict

    def _list_classes(self, directory=None):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        return classes

    def __len__(self):
        return len(self.images)












def build_part_dataset(args):

    if args.dataset_version == "single":
        dataset_train = PartImageNetSegDataset(
            args.train_dir,
            transform=None,
            seg_fraction=1.0,
            use_whether_pseudo=args.use_whether_pseudo,
            pseudo_weight=args.pseudo_weight,
            use_object_seg = args.use_object_seg,
            ignore_blank=args.ignore_blank
        )
        dataset_eval = PartImageNetSegDataset(
            args.eval_dir,
            transform=None,
            seg_fraction=1.0,
            use_object_seg = args.use_object_seg
        )
    # dataset_train = datasets.ImageFolder(root=args.train_dir, transform=None)
    # dataset_eval = datasets.ImageFolder(root=args.eval_dir, transform=None)

    # build transform
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = args.interpolation


    re_num_splits = 0
    dataset_train.transform = create_transform(
        args.input_size,
        is_training=True,
        use_prefetcher=False,
        no_aug=args.no_aug,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,  #
        auto_augment=args.aa,   # rand-m9-mstd0.5-inc1
        interpolation=train_interpolation,
        mean=args.mean,
        std=args.std,
        crop_pct=args.crop_pct,
        tf_preprocessing=False,
        re_prob=args.reprob, 
        re_mode=args.remode,
        re_count=args.recount,
        re_num_splits=re_num_splits,
        separate=False
    )

    dataset_eval.transform = create_transform(
        args.input_size,
        is_training=False,
        use_prefetcher=False,
        interpolation=args.interpolation,
        mean=args.mean,
        std=args.std,
        crop_pct=args.crop_pct
    )

    # create sampler
    sampler_train = None
    sampler_eval = None
    if args.distributed and not isinstance(dataset_train, torch.utils.data.IterableDataset):
        sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train)
    sampler_eval = OrderedDistributedSampler(dataset_eval)

    # create dataloader
    dataloader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=sampler_train,
        collate_fn=None,
        pin_memory=args.pin_mem,
        drop_last=True
    )
    dataloader_eval = torch.utils.data.DataLoader(
        dataset=dataset_eval,
        batch_size=args.batch_size // 2,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=sampler_eval,
        collate_fn=None,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes, seg_classes=args.seg_classes, 
            outputsize=args.output_size)
        mixup_fn = Mixup(**mixup_args)

    return dataloader_train, dataloader_eval, mixup_fn