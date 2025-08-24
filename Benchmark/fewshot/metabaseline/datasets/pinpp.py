import os
import pickle
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .datasets import register


@register('pinpp')
class pinpp(Dataset):

    def __init__(self, root_path, split='train', **kwargs):
        split_tag = split
        if split == 'train':
            split_tag = 'train_phase_train'
        split_file = 'pinpp_category_split_{}.pickle'.format(split_tag)
        with open(os.path.join(root_path, split_file), 'rb') as f:
            pack = pickle.load(f, encoding='latin1')
        data = pack['data']
        label = pack['labels']
        bbox = pack['bbox']

        image_size = 80
        # print(len(data[0][0][0]))
        data = [Image.fromarray(x) for x in data]

        min_label = min(label)
        label = [x - min_label for x in label]
        
        self.data = data
        self.label = label
        self.n_classes = max(self.label) + 1
        self.bbox = bbox

        norm_params = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
        normalize = transforms.Normalize(**norm_params)
        self.default_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,  
        ])
        augment = kwargs.get('augment')
        if augment == 'resize':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif augment == 'crop':
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomCrop(image_size, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif augment is None:
            self.transform = self.default_transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return torch.stack((self.transform(self.data[i].crop((self.bbox[i][0], self.bbox[i][1], self.bbox[i][0] + self.bbox[i][2], self.bbox[i][1] + self.bbox[i][3]))), self.transform(self.data[i]))), self.label[i]

