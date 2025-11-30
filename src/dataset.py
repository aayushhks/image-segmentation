import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class OxfordPetDataset(Dataset):
    def __init__(self, root_dir, list_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Load file IDs
        with open(list_file, 'r') as f:
            self.ids = [line.strip().split(' ')[0] for line in f]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_name = self.ids[idx]
        img_path = os.path.join(self.root_dir, "images", f"{img_name}.jpg")
        mask_path = os.path.join(self.root_dir, "annotations", "trimaps", f"{img_name}.png")

        # Load Image (RGB)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load Mask (Grayscale)
        mask = cv2.imread(mask_path, 0)

        # Preprocessing Logic:
        # The Oxford Pet dataset uses trimaps: 1=Pet, 2=BG, 3=Border.
        # We convert this to Binary Segmentation: 1=Pet, 0=Background/Border.
        binary_mask = np.where(mask == 1, 1.0, 0.0)

        if self.transform:
            augmented = self.transform(image=image, mask=binary_mask)
            image = augmented['image']
            mask = augmented['mask']

        # PyTorch expects channel-first format for masks (1, H, W)
        if mask.ndim == 2:
            mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask.float()

def get_transforms(phase, input_size):
    """Returns Albumentations transform pipeline."""
    # ImageNet normalization statistics
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if phase == 'train':
        return A.Compose([
            A.Resize(input_size, input_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(input_size, input_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

def get_loaders(config):
    train_ds = OxfordPetDataset(
        config.DATA_DIR, 
        config.TRAIN_LIST, 
        transform=get_transforms('train', config.INPUT_SIZE)
    )
    
    val_ds = OxfordPetDataset(
        config.DATA_DIR, 
        config.VAL_LIST, 
        transform=get_transforms('valid', config.INPUT_SIZE)
    )

    train_loader = DataLoader(
        train_ds, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS
    )
    
    return train_loader, val_loader
