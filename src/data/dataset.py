import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import autoaugment, InterpolationMode
from PIL import Image
import numpy as np


class ImageNetDataset:
    def __init__(self, config):
        self.config = config
        self.data_path = config['data']['data_path']
        self.dataset_type = config['data']['dataset']
        
    def get_transforms(self, is_training=True):
        if self.dataset_type == "tinyimagenet":
            # Tiny ImageNet uses 64x64 images
            if is_training:
                return transforms.Compose([
                    transforms.RandomCrop(64, padding=8),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225]),
                ])
            else:
                return transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225]),
                ])
        else:
            # Standard ImageNet transforms
            if is_training:
                return transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    # autoaugment.AutoAugment(autoaugment.AutoAugmentPolicy.IMAGENET),  # Commented out for now
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225]),
                ])
            else:
                return transforms.Compose([
                    transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225]),
                ])
    
    def get_datasets(self):
        train_transform = self.get_transforms(is_training=True)
        val_transform = self.get_transforms(is_training=False)
        
        if self.dataset_type == "imagenet1k":
            train_dataset = datasets.ImageFolder(
                root=os.path.join(self.data_path, 'train'),
                transform=train_transform
            )
            val_dataset = datasets.ImageFolder(
                root=os.path.join(self.data_path, 'val'),
                transform=val_transform
            )
        elif self.dataset_type == "imagenet100k":
            train_dataset = ImageNet100K(
                root=os.path.join(self.data_path, 'train'),
                transform=train_transform
            )
            val_dataset = ImageNet100K(
                root=os.path.join(self.data_path, 'val'),
                transform=val_transform
            )
        elif self.dataset_type == "tinyimagenet":
            train_dataset = datasets.ImageFolder(
                root=os.path.join(self.data_path, 'train'),
                transform=train_transform
            )
            val_dataset = TinyImageNetVal(
                root=os.path.join(self.data_path, 'val'),
                transform=val_transform
            )
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
            
        return train_dataset, val_dataset
    
    def get_dataloaders(self):
        train_dataset, val_dataset = self.get_datasets()
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=self.config['data']['pin_memory'],
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=self.config['data']['pin_memory']
        )
        
        return train_loader, val_loader


class ImageNet100K(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        
        self._make_dataset()
    
    def _make_dataset(self):
        classes = sorted([d for d in os.listdir(self.root) 
                         if os.path.isdir(os.path.join(self.root, d))])
        
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.class_to_idx = class_to_idx
        
        samples = []
        for class_name in classes:
            class_path = os.path.join(self.root, class_name)
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            images = images[:1000] if len(images) > 1000 else images
            
            for img_name in images:
                img_path = os.path.join(class_path, img_name)
                samples.append((img_path, class_to_idx[class_name]))
        
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        
        with Image.open(path) as img:
            img = img.convert('RGB')
            if self.transform:
                img = self.transform(img)
        
        return img, target


def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    
    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class TinyImageNetVal(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        
        self._load_annotations()
        
    def _load_annotations(self):
        # Load validation annotations
        annotations_file = os.path.join(self.root, 'val_annotations.txt')
        
        # First, create class mapping from training data
        train_root = os.path.join(os.path.dirname(self.root), 'train')
        classes = sorted([d for d in os.listdir(train_root) 
                         if os.path.isdir(os.path.join(train_root, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        # Load validation image mappings
        with open(annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img_name = parts[0]
                    class_name = parts[1]
                    img_path = os.path.join(self.root, 'images', img_name)
                    if class_name in self.class_to_idx:
                        self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        
        with Image.open(path) as img:
            img = img.convert('RGB')
            if self.transform:
                img = self.transform(img)
        
        return img, target