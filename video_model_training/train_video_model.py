import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from dataset import VideoDeepfakeDataset
from model import VideoDeepfakeModel

# ================= Configuration =================
PROCESSED_DATA_DIR = r"D:\Celeb-DF-v2-processed"
SEQUENCE_LENGTH = 15

# VRAM-friendly 6GB GPU Settings
BATCH_SIZE = 2             # Keep very low to prevent OOM
ACCUMULATION_STEPS = 8     # Effectively trains at a batch size of 16 (2 * 8)
NUM_WORKERS = 2

# Training Strategy
EPOCHS_STAGE_1 = 2         # Train LSTM and Classification Head initially
EPOCHS_STAGE_2 = 3         # Unfreeze top layer of Backbone and fine-tune everything
LEARNING_RATE_1 = 1e-3     # Higher learning rate since Backbone is frozen
LEARNING_RATE_2 = 1e-5     # Very low learning rate so we don't break Backbone weights
# =================================================

def run_training_loop(model, train_loader, val_loader, criterion, optimizer, scaler, device, epochs, best_val_loss, stage_name):
    # This prevents an issue where accumulation keeps hanging gradients
    optimizer.zero_grad() 
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        
        print(f"\n[{stage_name}] Epoch {epoch+1}/{epochs}")
        loop = tqdm(train_loader, desc="Training")
        
        for i, (videos, labels) in enumerate(loop):
            videos, labels = videos.to(device), labels.to(device)
            
            # AMP: Mixed Precision Forward pass to save VRAM and train faster
            with torch.amp.autocast('cuda'):
                outputs = model(videos)
                loss = criterion(outputs, labels)
                loss = loss / ACCUMULATION_STEPS # Normalize loss for accumulation
            
            # AMP: Backward pass
            scaler.scale(loss).backward()
            
            # Gradient Accumulation: wait before stepping the optimizer
            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Track metrics
            total_train_loss += loss.item() * ACCUMULATION_STEPS
            preds = (torch.sigmoid(outputs) >= 0.5).float() # Apply sigmoid before calculating accuracy
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            
            loop.set_postfix(loss=loss.item() * ACCUMULATION_STEPS, acc=correct_train/total_train)
            
        # ----------------- Validation Phase -----------------
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        
        print("Running Validation...")
        with torch.no_grad():
            for videos, labels in tqdm(val_loader, desc="Validating"):
                videos, labels = videos.to(device), labels.to(device)
                
                # We can also use AMP on validation
                with torch.amp.autocast('cuda'):
                    outputs = model(videos)
                    loss = criterion(outputs, labels)
                    
                val_loss += loss.item()
                preds = (torch.sigmoid(outputs) >= 0.5).float() # Apply sigmoid before calculating accuracy
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct_val / total_val
        print(f"Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {val_acc:.4f}")
        
        # Checkpoint if we get a better model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save our weights out to file
            torch.save(model.state_dict(), 'video_deepfake_model.pth')
            print(f"[*] New Best Model Saved! (Val Loss: {best_val_loss:.4f})")

    return best_val_loss

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"==================================================")
    print(f"Starting Training Pipeline on hardware: {device}")
    print(f"==================================================")

    # Transformations matching Xception's pre-trained requirements
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # Standard ImageNet normalization figures
            std= [0.229, 0.224, 0.225]
        )
    ])

    print("Loading data...")
    dataset = VideoDeepfakeDataset(PROCESSED_DATA_DIR, SEQUENCE_LENGTH, transform)
    
    # 80/20 train/validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Split: {train_size} Training Videos, {val_size} Validation Videos")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

    # Initialize the model and place on GPU
    model = VideoDeepfakeModel(sequence_length=SEQUENCE_LENGTH).to(device)
    
    # BCEWithLogitsLoss is required for Mixed Precision (AMP) stability!
    # It combines Sigmoid and BCELoss in a single numerically stable class
    criterion = nn.BCEWithLogitsLoss()
    
    # PyTorch AMP scaler for Mixed Precision Training
    scaler = torch.amp.GradScaler('cuda')
    best_val_loss = float('inf')

    # ========================== STAGE 1 ==========================
    print("\n--- STAGE 1: Training LSTM & Classification Head (Backbone Frozen) ---")
    # Tell optimizer strictly to only update un-frozen parameters (LSTM + Linear Layers)
    optimizer_stage1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE_1)
    
    best_val_loss = run_training_loop(model, train_loader, val_loader, criterion, optimizer_stage1, scaler, device, EPOCHS_STAGE_1, best_val_loss, "Stage 1")

    # ========================== STAGE 2 ==========================
    print("\n--- STAGE 2: Fine-Tuning (Unfreezing Top Backbone Layers) ---")
    # Unfreeze the top 10 layers of the backbone to adapt it directly to Deepfakes
    model.unfreeze_backbone(num_layers=10) 
    
    # Re-initialize optimizer because model gradients have updated status
    optimizer_stage2 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE_2)
    
    best_val_loss = run_training_loop(model, train_loader, val_loader, criterion, optimizer_stage2, scaler, device, EPOCHS_STAGE_2, best_val_loss, "Stage 2")
    
    print("\n>>> Training fully complete! Best model is saved as: video_deepfake_model.pth")

if __name__ == "__main__":
    main()