import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler  # or torch.amp.GradScaler if PyTorch 2+
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from src.dataset import get_loaders
from src.model import get_model
from src.utils import SegmentationLoss, calculate_iou
from train import train_epoch, validate_epoch  # Reuse functions from your train.py


def run_specific_experiment(name, arch, encoder, save_name, epochs=5, batch_size=None):
    print(f"\n" + "=" * 40)
    print(f"🧪 STARTING EXPERIMENT: {name}")
    print(f"   Arch: {arch} | Encoder: {encoder}")
    print("=" * 40)

    # 1. Configure
    Config.ARCHITECTURE = arch
    Config.ENCODER = encoder
    if batch_size:
        Config.BATCH_SIZE = batch_size

    device = Config.DEVICE

    # 2. Data
    print("   Loading data...")
    train_loader, val_loader = get_loaders(Config)

    # 3. Model
    print("   Building model...")
    model = get_model(Config)
    model.to(device)

    # 4. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    criterion = SegmentationLoss()

    # Handle Scaler for Mac (MPS) vs CUDA
    if device == 'cuda':
        try:
            scaler = torch.amp.GradScaler('cuda')
        except:
            scaler = GradScaler()
    else:
        # MPS (Mac) doesn't strictly require scaler for float32,
        # but if using autocast it's good practice.
        # For simplicity on Mac CPU/MPS we can pass None if not using mixed precision strict
        scaler = None

        # 5. Training Loop
    best_iou = 0.0
    save_path = os.path.join(Config.CHECKPOINT_DIR, save_name)

    for epoch in range(epochs):
        # Train
        t_loss, t_iou = train_epoch(model, train_loader, optimizer, criterion, scaler, device)

        # Validate
        v_loss, v_iou = validate_epoch(model, val_loader, criterion, device)

        print(f"   Epoch {epoch + 1}/{epochs} | Val IoU: {v_iou:.4f}")

        if v_iou > best_iou:
            best_iou = v_iou
            torch.save(model.state_dict(), save_path)
            print(f"   Saved new best model: {save_name}")

    print(f"Experiment '{name}' Finished. Best IoU: {best_iou:.4f}")

    # Cleanup memory
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Experiment 1: Baseline U-Net (ResNet34)
    # Mac M4 Pro is powerful, so we can use standard batch size
    run_specific_experiment(
        name="Baseline CNN",
        arch="Unet",
        encoder="resnet34",
        save_name="best_unet.pth",
        epochs=10,  # Enough to converge
        batch_size=16
    )

    # Experiment 2: SOTA Transformer (SegFormer)
    # Transformers are memory hungry, we lower batch size slightly to be safe
    run_specific_experiment(
        name="SOTA Transformer",
        arch="MAnet",
        encoder="mit_b0",
        save_name="best_transformer.pth",
        epochs=10,
        batch_size=8  # Lower batch size for stability
    )