import os
import torch


class Config:
    # Project Settings
    PROJECT_NAME = "SegFormer_OxfordPets_Study"
    INPUT_SIZE = 256

    # Mac M4 Pro has plenty of RAM, so we can use a decent batch size
    BATCH_SIZE = 16
    EPOCHS = 15
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 2

    # Model Settings
    ENCODER = "mit_b0"  # Mix Transformer (SegFormer)
    ENCODER_WEIGHTS = "imagenet"
    ARCHITECTURE = "MAnet"

    # Compute Settings - Optimized for Mac (MPS)
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"  # <--- Metal Performance Shaders (Apple Silicon GPU)
    else:
        DEVICE = "cpu"

    SEED = 42

    # Paths
    DATA_DIR = "."
    TRAIN_LIST = "train_list.txt"
    VAL_LIST = "val_list.txt"
    CHECKPOINT_DIR = "checkpoints"

    # Create checkpoint dir if it doesn't exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")