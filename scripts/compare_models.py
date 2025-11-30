import os
import sys
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from src.model import get_model
from src.dataset import get_loaders


def visualize_comparison(unet_model, tf_model, dataset, device, num_samples=5, save_dir="results/comparison"):
    os.makedirs(save_dir, exist_ok=True)
    indices = random.sample(range(len(dataset)), num_samples)
    print(f"Generating comparisons in '{save_dir}'...")

    # Denormalization stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, mask = dataset[idx]
            input_tensor = image.unsqueeze(0).to(device)

            # Predictions
            if tf_model:
                tf_out = tf_model(input_tensor)
                tf_pred = (torch.sigmoid(tf_out) > 0.5).float().cpu().numpy()[0, 0]
            else:
                tf_pred = np.zeros((256, 256))

            if unet_model:
                u_out = unet_model(input_tensor)
                u_pred = (torch.sigmoid(u_out) > 0.5).float().cpu().numpy()[0, 0]
            else:
                u_pred = np.zeros((256, 256))

            # Prepare Image
            img_show = image.permute(1, 2, 0).cpu().numpy()
            img_show = (img_show * std) + mean
            img_show = np.clip(img_show, 0, 1)

            # Plot
            fig, ax = plt.subplots(1, 4, figsize=(20, 5))
            ax[0].imshow(img_show);
            ax[0].set_title("Input")
            ax[1].imshow(mask[0], cmap="gray");
            ax[1].set_title("Ground Truth")
            ax[2].imshow(u_pred, cmap="gray");
            ax[2].set_title("U-Net (Baseline)")
            ax[3].imshow(tf_pred, cmap="gray");
            ax[3].set_title("SegFormer (Ours)")

            for a in ax: a.axis("off")
            plt.savefig(os.path.join(save_dir, f"compare_{i}.png"), bbox_inches='tight')
            plt.close(fig)


def load_model(arch, encoder, filename):
    path = os.path.join(Config.CHECKPOINT_DIR, filename)
    if not os.path.exists(path):
        print(f"Warning: {filename} not found. Did you run 'scripts/run_benchmark.py'?")
        return None

    # Override config temporarily to build correct architecture
    Config.ARCHITECTURE = arch
    Config.ENCODER = encoder

    model = get_model(Config)
    model.load_state_dict(torch.load(path, map_location=Config.DEVICE))
    model.to(Config.DEVICE)
    model.eval()
    return model


if __name__ == "__main__":
    # 1. Load Data
    _, val_loader = get_loaders(Config)

    # 2. Load Models
    print("Loading Baseline U-Net...")
    unet = load_model("Unet", "resnet34", "best_unet.pth")

    print("Loading SOTA Transformer...")
    transformer = load_model("MAnet", "mit_b0", "best_transformer.pth")

    # 3. Compare
    if unet or transformer:
        visualize_comparison(unet, transformer, val_loader.dataset, Config.DEVICE)