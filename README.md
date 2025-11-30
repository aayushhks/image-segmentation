# Semantic Segmentation: Transformers vs. CNNs

## Project Overview

This project implements a comparative study of modern Deep Learning architectures for Semantic Segmentation, applied to the Oxford-IIIT Pet Dataset. The codebase is structured as a modular Python package designed for reproducibility and scalability.

The primary objective is to benchmark a Transformer-based approach (SegFormer style) against a standard CNN baseline (U-Net) to analyze trade-offs between parameter efficiency and boundary precision.

## Key Features

- **Modular Architecture:** Separated logic for data loading, modeling, and training loops.
- **State-of-the-Art Models:** Utilizes `segmentation-models-pytorch` to implement U-Net (ResNet34) and MA-Net (Mix Transformer / SegFormer).
- **Hardware Acceleration:** Supports CUDA (NVIDIA) and MPS (Apple Silicon M-Series) for local acceleration.
- **Mixed Precision Training:** Implements `torch.amp` for memory efficiency and speed.
- **Rigorous Data Augmentation:** Uses `albumentations` for geometric and photometric distortions.
- **Experiment Tracking:** Integrated with Weights & Biases (W&B) for logging metrics.

## Directory Structure
```
.
├── src/
│   ├── config.py        # Hyperparameters and system settings
│   ├── dataset.py       # Custom PyTorch Dataset and Augmentation pipelines
│   ├── model.py         # Model factory for U-Net and Transformers
│   ├── utils.py         # Metric calculations (IoU) and Loss functions
│   └── __init__.py
├── scripts/
│   ├── prepare_data.py  # Downloads and splits the Oxford-IIIT Pet dataset
│   ├── evaluate.py      # Inference script to generate visual overlays
│   └── setup_dummy.py   # Generates synthetic data for pipeline testing
├── checkpoints/         # Stores the best model weights (.pth)
├── results/             # Stores output visualizations
├── train.py             # Main training entry point
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```

## Installation

### Clone the repository
```bash
git clone <repository-url>
cd image-segmentation
```

### Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation

Download and extract the Oxford-IIIT Pet dataset. This script will also generate the training and validation split files.
```bash
python scripts/prepare_data.py
```

### 2. Training

To start the training pipeline, run the main script. This will:

- Load the configuration from `src/config.py`.
- Initialize the model (default: MA-Net with MiT-B0 encoder).
- Train using AdamW optimizer and Cosine Annealing.
- Save the best weights to `checkpoints/best_model.pth`.
```bash
python train.py
```

Note: To modify hyperparameters (epochs, batch size, architecture), edit `src/config.py`.

### 3. Evaluation & Inference

After training, run the evaluation script to generate qualitative results. This script runs the model on unseen validation data and saves side-by-side comparisons (Input, Ground Truth, Prediction) to the `results/` folder.
```bash
python scripts/evaluate.py
```

## Experiment Results & Analysis

### 1. Quantitative Performance

We evaluated the Transformer-based SegFormer architecture against a standard CNN baseline. The SegFormer (MA-Net + MiT-B0) demonstrated superior performance, achieving a high Intersection-over-Union (IoU) score on the validation set.

| Model Architecture | Backbone (Encoder) | Pretrained Weights | Batch Size | Best Val IoU |
|:-------------------|:-------------------|:-------------------|:-----------|:-------------|
| MA-Net (SegFormer) | MiT-B0 (Mix Transformer) | ImageNet | 4 | 0.8852 |
| U-Net (Baseline) | ResNet34 | ImageNet | 16 | 0.8521 |

Note: The Transformer model achieved an IoU of 0.8852, indicating excellent overlap between the predicted segmentation masks and the ground truth.

### 2. Visual Inference (Qualitative Analysis)

The model demonstrates strong capabilities in distinguishing foreground (pets) from complex backgrounds, even with the lightweight `mit_b0` encoder.

#### Sample Predictions

Left: Input Image | Center: Ground Truth Mask | Right: Model Prediction

![Sample 0](results/result_0.png)

Figure 1: The model successfully segments the cat, capturing fine details around the ears despite the textured background.

![Sample 1](results/result_1.png)

Figure 2: Robust segmentation of the dog, accurately handling the contrast between the fur and the floor.

![Sample 2](results/result_2.png)

Figure 3: Handling varied poses. The model correctly identifies the animal's shape even in a non-standard posture.

### 3. Discussion & Findings

- **Transformer Efficacy:** The Mix Transformer (MiT) encoder utilizes self-attention mechanisms, allowing it to capture global context more effectively than standard CNNs. This results in sharper boundary delineation.
- **Memory Constraints:** While highly accurate, the attention mechanism is memory-intensive. We found that a batch size of 4 was required to train stably on standard GPUs, compared to 16+ for CNNs.
- **Convergence:** The model converged rapidly, reaching >0.80 IoU within the first few epochs, largely due to the effectiveness of the ImageNet-pretrained weights in the encoder.

## Technical Details

### Loss Function

The project utilizes a compound loss function to address class imbalance and structural consistency:

- **Dice Loss:** Optimizes the intersection-over-union directly.
- **Binary Cross Entropy (BCE):** Optimizes pixel-level classification probability.
- **Total Loss:** L = L_Dice + L_BCE

### Augmentation Strategy

To prevent overfitting on the relatively small Oxford Pets dataset, the following augmentations are applied during training:

- Horizontal Flip
- Shift, Scale, Rotate
- Random Brightness and Contrast
- Normalization (ImageNet statistics)

## License

This project uses the Oxford-IIIT Pet Dataset (Creative Commons Attribution-ShareAlike 4.0). Code is provided under the MIT License.