import torch

class Config:
    # Project Settings
    PROJECT_NAME = "SegFormer_OxfordPets_Study"
    INPUT_SIZE = 256
    BATCH_SIZE = 4        # Low batch size for Transformers
    EPOCHS = 15
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 2
    
    # Model Settings
    ENCODER = "mit_b0"    # Mix Transformer (SegFormer)
    ENCODER_WEIGHTS = "imagenet"
    ARCHITECTURE = "MAnet"
    
    # Compute Settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    
    # Paths
    DATA_DIR = "."
    TRAIN_LIST = "train_list.txt"
    VAL_LIST = "val_list.txt"
    CHECKPOINT_DIR = "checkpoints"
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
