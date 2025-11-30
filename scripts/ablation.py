import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
import sys
import os
import pandas as pd

# Add project root to path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from src.dataset import get_loaders
from src.model import get_model
from src.utils import SegmentationLoss, calculate_iou

# Define experiments to run
experiments = [
    {"name": "Baseline (ResNet34)", "architecture": "Unet", "encoder": "resnet34", "size": 256},
    {"name": "Transformer (MiT-B0)", "architecture": "MAnet", "encoder": "mit_b0", "size": 256},
    {"name": "Transformer High-Res", "architecture": "MAnet", "encoder": "mit_b0", "size": 320},
]


def run_ablation():
    results = []
    device = Config.DEVICE
    print(f"Starting Ablation Study on {device.upper()}...")

    # Shared Criterion
    criterion = SegmentationLoss()

    for exp in experiments:
        print(f"\nRunning Experiment: {exp['name']}")

        # 1. Override Config for this run
        current_config = Config()
        current_config.ARCHITECTURE = exp["architecture"]
        current_config.ENCODER = exp["encoder"]
        current_config.INPUT_SIZE = exp["size"]
        # Reduce epochs for ablation speed (use full epochs for final paper)
        current_config.EPOCHS = 5

        # 2. Get Data and Model
        train_loader, val_loader = get_loaders(current_config)
        model = get_model(current_config)
        model.to(device)

        optimizer = optim.AdamW(model.parameters(), lr=current_config.LEARNING_RATE)
        scaler = GradScaler()

        best_iou = 0.0

        # 3. Training Loop
        for epoch in range(current_config.EPOCHS):
            model.train()
            # Train step
            for images, masks in train_loader:
                images = images.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()
                # Standard precision for simplicity in ablation, or use scaler if CUDA
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

            # Validation step
            model.eval()
            total_iou = 0
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    outputs = model(images)
                    total_iou += calculate_iou(outputs, masks).item()

            val_iou = total_iou / len(val_loader)
            if val_iou > best_iou:
                best_iou = val_iou

        print(f"Result: {best_iou:.4f} IoU")

        results.append({
            "Experiment": exp["name"],
            "Encoder": exp["encoder"],
            "Resolution": exp["size"],
            "Best Val IoU": round(best_iou, 4)
        })

        # Cleanup memory
        del model, optimizer, train_loader, val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 4. Display Results Table
    print("\n=== Ablation Study Results ===")
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))

    # Save to CSV for README
    df.to_csv("results/ablation_results.csv", index=False)


if __name__ == "__main__":
    run_ablation()