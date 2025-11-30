import os
import torch


class Config:
    # Project Settings
    PROJECT_NAME = "SegFormer_Research_Study"

    # Dataset Settings
    DATASET = "OxfordPets"  # Options: "OxfordPets", "PascalVOC"
    INPUT_SIZE = 256

    # Compute Settings
    BATCH_SIZE = 16
    EPOCHS = 15
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 2

    # Model Settings
    ENCODER = "mit_b0"
    ENCODER_WEIGHTS = "imagenet"
    ARCHITECTURE = "MAnet"

    # Hardware Detection
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"

    SEED = 42

    # Paths
    DATA_DIR = "."
    TRAIN_LIST = "train_list.txt"
    VAL_LIST = "val_list.txt"
    CHECKPOINT_DIR = "checkpoints"

    # Ensure checkpoint directory exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    # Dynamic Properties (Needed for Multi-class expansion later)
    @property
    def NUM_CLASSES(self):
        return 21 if self.DATASET == "PascalVOC" else 1

    @property
    def ACTIVATION(self):
        return "softmax2d" if self.DATASET == "PascalVOC" else "sigmoid"