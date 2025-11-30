%%writefile
scripts / evaluate.py
import os
import sys
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add the project root to system path so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from src.model import get_model
from src.dataset import get_loaders


def visualize_results(model, loader, device, num_samples=5, save_dir="results"):
    """
    Runs inference on random samples and saves visualization grids.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # Get random samples from the validation loader
    # Note: This is a bit inefficient for large datasets but fine for demos
    all_samples = []
    for img, mask in loader.dataset:
        all_samples.append((img, mask))

    # Pick random indices
    indices = random.sample(range(len(all_samples)), num_samples)

    print(f"Generating {num_samples} visualizations in '{save_dir}'...")

    # De-normalization parameters (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, mask = all_samples[idx]
            input_tensor = image.unsqueeze(0).to(device)

            # Predict
            output = model(input_tensor)
            # Apply Sigmoid and Binarize
            output = torch.sigmoid(output)
            pred_mask = (output > 0.5).float().cpu().numpy()[0, 0]

            # Process Image for Display
            # 1. Convert Tensor (C, H, W) -> Numpy (H, W, C)
            img_display = image.permute(1, 2, 0).cpu().numpy()
            # 2. De-normalize: x * std + mean
            img_display = (img_display * std) + mean
            # 3. Clip to [0, 1] range to avoid Matplotlib warnings
            img_display = np.clip(img_display, 0, 1)

            # Prepare Ground Truth Mask
            mask_display = mask[0].numpy()

            # Plotting
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))

            # Original Image
            ax[0].imshow(img_display)
            ax[0].set_title("Input Image")
            ax[0].axis("off")

            # Ground Truth
            ax[1].imshow(mask_display, cmap="gray")
            ax[1].set_title("Ground Truth")
            ax[1].axis("off")

            # Prediction
            ax[2].imshow(pred_mask, cmap="gray")
            ax[2].set_title("Model Prediction")
            ax[2].axis("off")

            # Save
            save_path = os.path.join(save_dir, f"result_{i}.png")
            plt.savefig(save_path)
            plt.close(fig)
            print(f"Saved: {save_path}")


if __name__ == "__main__":
    # 1. Load Configuration
    # Ensure device matches training (MPS for Mac, CUDA for NVIDIA)
    device = Config.DEVICE

    # 2. Initialize Model
    # We must build the exact same architecture to load weights
    model = get_model(Config)

    # 3. Load Weights
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Loading model from {Config.BEST_MODEL_PATH}...")
        state_dict = torch.load(Config.BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("Error: Best model checkpoint not found. Train first!")
        exit()

    model.to(device)

    # 4. Get Data
    _, val_loader = get_loaders(Config)

    # 5. Run Visualization
    visualize_results(model, val_loader, device)