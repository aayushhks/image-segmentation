import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import wandb

# Import our custom modules
from src.config import Config
from src.dataset import get_loaders
from src.model import get_model
from src.utils import set_seed, SegmentationLoss, calculate_iou

def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    epoch_loss = 0
    epoch_iou = 0
    
    # Progress bar for visual feedback
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass with Mixed Precision
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
            
        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Metrics
        loss_val = loss.item()
        iou_val = calculate_iou(outputs, masks).item()
        
        epoch_loss += loss_val
        epoch_iou += iou_val
        
        pbar.set_postfix({"Loss": f"{loss_val:.4f}", "IoU": f"{iou_val:.4f}"})
        
    return epoch_loss / len(loader), epoch_iou / len(loader)

def validate_epoch(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_iou = 0
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            epoch_loss += loss.item()
            epoch_iou += calculate_iou(outputs, masks).item()
            
    return epoch_loss / len(loader), epoch_iou / len(loader)

def main():
    # 1. Setup
    set_seed(Config.SEED)
    print(f"Starting Project: {Config.PROJECT_NAME}")
    
    # Initialize W&B (Optional)
    try:
        wandb.init(project=Config.PROJECT_NAME, config=Config.__dict__)
    except:
        print("W&B not initialized.")

    # 2. Data
    train_loader, val_loader = get_loaders(Config)
    
    # 3. Model
    model = get_model(Config)
    model.to(Config.DEVICE)
    
    # 4. Optimizer & Loss
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=Config.LEARNING_RATE, 
        weight_decay=Config.WEIGHT_DECAY
    )
    criterion = SegmentationLoss()
    
    # Use standard GradScaler (updates automatically handle deprecations)
    scaler = GradScaler()
    
    # 5. Training Loop
    best_iou = 0.0
    
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")
        
        train_loss, train_iou = train_epoch(model, train_loader, optimizer, criterion, scaler, Config.DEVICE)
        val_loss, val_iou = validate_epoch(model, val_loader, criterion, Config.DEVICE)
        
        print(f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val IoU:   {val_iou:.4f}")
        
        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"--> Best model saved at {Config.BEST_MODEL_PATH}")
            
        # Log to W&B
        try:
            wandb.log({
                "train_loss": train_loss, "train_iou": train_iou,
                "val_loss": val_loss, "val_iou": val_iou
            })
        except:
            pass

    print(f"\nTraining Complete. Best Validation IoU: {best_iou:.4f}")

if __name__ == "__main__":
    main()
