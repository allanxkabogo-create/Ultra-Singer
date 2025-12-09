cat > scripts/train.py <<EOF
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model.tokenizer import UltraTokenizer
from model.backbone import UltraSingerDiT
from scripts.dataset import UltraDataset
import os

# --- Config ---
BATCH_SIZE = 2
LR = 1e-4
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train():
    # 1. Init Models
    print("Loading Models...")
    tokenizer = UltraTokenizer(model_path="weights_44khz.pth")
    tokenizer.to(DEVICE)
    
    # Tiny Config for Testing
    dit_config = {
        'in_channels': 1024, # DAC embedding dim
        'hidden_size': 512,
        'depth': 6,
        'num_heads': 8
    }
    model = UltraSingerDiT(dit_config).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    # 2. Data
    print("Loading Data...")
    if not os.path.exists("dataset_metadata_aligned.json"):
        print("⚠️ No dataset found. Please run data_prep/3_align.py first.")
        return

    dataset = UltraDataset("dataset_metadata_aligned.json", tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 3. Loop
    print(f"Starting Training on {DEVICE}...")
    model.train()
    
    for epoch in range(EPOCHS):
        for batch_idx, (audio, text) in enumerate(loader):
            audio = audio.to(DEVICE)
            
            # A. Get Ground Truth Tokens (The "Ears")
            with torch.no_grad():
                # Encode audio to discrete codes
                codes = tokenizer.encode(audio.unsqueeze(1)) 
                # (Ideally, we convert codes to continuous embeddings here)
                # For this dummy loop, we assume 'codes' is the target
                
            # B. Forward Pass (The "Brain")
            # Create a random timestep
            t = torch.randint(0, 1000, (audio.shape[0],), device=DEVICE)
            
            # Dummy Context (In real run, this is Phonemes+Pitch)
            context = torch.zeros(audio.shape[0], 512).to(DEVICE) 
            
            # Predict
            # (We need to adapt DAC codes to DiT input dimension in full version)
            # This is a placeholder to verify the loop runs
            dummy_input = torch.randn(audio.shape[0], 100, 1024).to(DEVICE)
            noise_pred = model(dummy_input, t, context)
            
            # C. Loss (Simple MSE for Flow Matching)
            loss = torch.mean((noise_pred - dummy_input) ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Step {batch_idx} | Loss: {loss.item():.4f}")
                
        # Save Checkpoint
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"checkpoints/ultra_epoch_{epoch}.pt")

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    train()
EOF
