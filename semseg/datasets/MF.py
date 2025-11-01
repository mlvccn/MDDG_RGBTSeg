import os
import torch 
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF 
from torchvision import io
from pathlib import Path
from typing import Tuple
import glob
import einops
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler, RandomSampler
from semseg.augmentations_mm import get_train_augmentation


class MF(Dataset):
    CLASSES = [ "unlabeled","car","person","bike", "curve", "car_stop", "guardrail", "color_cone", "bump"]
      
    def __init__(self, root: str="/data/tangchen/MFNet", split: str="train" , transform=None, modals = ['img', 'thermal'], case=None)->None:
        super(MF, self).__init__()

        assert split in ['train', 'val', 'test', 'test_day', 'test_night', 'val_test', 'most_wanted'], \
            'split must be "train"|"val"|"test"|"test_day"|"test_night"|"val_test"|"most_wanted"'  # test_day, test_night

        with open(os.path.join(root, split+'.txt'), 'r') as f:
            self.files = [os.path.join(root,"images","%s.png" % (name.strip())) for name in f.readlines()]
        
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        self.modals = modals
        self.split = split
        # get the scores for train data
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index):
        name  = self.files[index]
        item_name = self.files[index].split("/")[-1].split(".")[0]
        image = io.read_image(name)
        lbl_path = name.replace('/images', '/labels')
        label = io.read_image(lbl_path)[0,...].unsqueeze(0)
        # raw_sample = {}
        # raw_sample['img'] = image[:3, ...] 
        # raw_sample['thermal'] = image[3, ...].repeat(3, 1, 1)
        # raw_sample['mask'] = label
        sample = {}
        sample['img'] = image[:3, ...] 
        sample['thermal'] = image[3, ...].repeat(3, 1, 1)
        sample['mask'] = label
        if self.transform:
            sample = self.transform(sample)
        label = sample['mask']
        del sample['mask']
        label = self.encode(label.squeeze().numpy()).long()
        sample = [sample[k] for k in self.modals]
        # raw_sample = [raw_sample[k] for k in self.modals]
        # return raw_sample, sample, label, item_name
        return sample, label
              
    
    def _open_img(self, file):
        img = io.read_image(file)
        C, H, W = img.shape
        if C == 4:
            img = img[:3, ...]
        if C == 1:
            img = img.repeat(3, 1, 1)
        return img
    
    def encode(self, label: Tensor) -> Tensor:
        return torch.from_numpy(label)
    
if __name__ == '__main__':
    cases = [ "unlabeled","car","person","bike", "curve", "car_stop", "guardrail", "color_cone", "bump"]
    traintransform = get_train_augmentation((1024, 1024), seg_fill=255)
    trainset = MF(transform=traintransform, split='train')
    trainloader = DataLoader(trainset, batch_size=2, num_workers=2, drop_last=False, pin_memory=False)

    for i, (sample, lbl) in enumerate(trainloader):
            print(torch.unique(lbl))
    