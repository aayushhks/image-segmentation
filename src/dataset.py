import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


class OxfordPetDataset(Dataset):
    def __init__(self, root_dir, list_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        with open(list_file, 'r') as f:
            self.ids = [line.strip().split(' ')[0] for line in f]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_name = self.ids[idx]
        img_path = os.path.join(self.root_dir, "images", f"{img_name}.jpg")
        mask_path = os.path.join(self.root_dir, "annotations", "trimaps", f"{img_name}.png")

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)

        # Binary: 1=Pet, 0=Background
        binary_mask = np.where(mask == 1, 1.0, 0.0)

        if self.transform:
            augmented = self.transform(image=image, mask=binary_mask)
            image = augmented['image']
            mask = augmented['mask']

        # --- FIX START ---
        # Check if mask is already a tensor (from Albumentations ToTensorV2)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        # Ensure mask has channel dimension (1, H, W)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        # --- FIX END ---

        return image, mask.float()


class PascalVOCDataset(Dataset):
    """
    Handler for Pascal VOC 2012 Semantic Segmentation.
    Classes: 21 (0=Background, 1-20=Objects)
    Ignore Index: 255 (Object boundaries)
    """

    def __init__(self, root_dir, list_file, transform=None):
        self.root_dir = os.path.join(root_dir, "VOCdevkit", "VOC2012")
        self.transform = transform
        with open(list_file, 'r') as f:
            self.ids = [line.strip() for line in f]

        # VOC Class Colors (for visualization if needed)
        self.classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person',
                        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_name = self.ids[idx]
        img_path = os.path.join(self.root_dir, "JPEGImages", f"{img_name}.jpg")
        mask_path = os.path.join(self.root_dir, "SegmentationClass", f"{img_name}.png")

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask as PIL to preserve index values (0-21, 255)
        mask = np.array(Image.open(mask_path))

        # Handle 255 (Border/Ignore) -> Convert to 0 (Background) for simplicity
        # In strict research, we would mask these out in the loss function
        mask[mask == 255] = 0

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # For Multi-class, mask should be LongTensor (H, W) with values 0-N
        # No channel dimension needed for CrossEntropy
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        mask = mask.long()

        return image, mask


def get_transforms(phase, input_size):
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
    if config.DATASET == "PascalVOC":
        train_ds = PascalVOCDataset(config.DATA_DIR, "voc_train_list.txt",
                                    transform=get_transforms('train', config.INPUT_SIZE))
        val_ds = PascalVOCDataset(config.DATA_DIR, "voc_val_list.txt",
                                  transform=get_transforms('valid', config.INPUT_SIZE))
    else:
        train_ds = OxfordPetDataset(config.DATA_DIR, config.TRAIN_LIST,
                                    transform=get_transforms('train', config.INPUT_SIZE))
        val_ds = OxfordPetDataset(config.DATA_DIR, config.VAL_LIST,
                                  transform=get_transforms('valid', config.INPUT_SIZE))

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader