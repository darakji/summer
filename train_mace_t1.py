import os
import torch
import numpy as np
from tqdm import tqdm
from mace.data.utils import load_atomic_dataset
from mace.modules import MACE

# === Config ===
init_model_path = "/home/phanim/harshitrawat/summer/mace_models/universal/2024-01-07-mace-128-L2_epoch-199.model"
train_file = "/home/phanim/harshitrawat/summer/mace_train_T1.db"
save_dir = "/home/phanim/harshitrawat/summer/mace_t1_finetuned_large"
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 1  # You can try 2 if memory allows
max_epochs = 100
learning_rate = 5e-4
ema_decay = 0.99
log_every = 20

weights = dict(energy=1.0, forces=1.0, stress=1.0)

# === Dataset ===
dataset = load_atomic_dataset([train_file])
train_data, val_data = dataset.split(train_fraction=0.9)

print(f"Train samples: {len(train_data)} | Val samples: {len(val_data)}")

# === Dataloader ===
def collate(batch):
    return batch

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate)

# === Load Pretrained Model ===
state_dict = torch.load(init_model_path, map_location="cpu")
model = MACE(**state_dict["config"])
model.load_state_dict(state_dict["state_dict"])
model = model.to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# === Training Loop ===
for epoch in range(max_epochs):
    epoch_loss = 0
    for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        loss = model(batch, compute_stress=True, compute_forces=True, energy_weight=weights["energy"],
                     forces_weight=weights["forces"], stress_weight=weights["stress"])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        if i % log_every == 0:
            print(f"[Epoch {epoch+1} | Batch {i}] Loss: {loss.item():.4f} | GPU Mem: {torch.cuda.memory_allocated()//1e6:.1f} MB")

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.6f}")

    # === Save checkpoint ===
    torch.save({
        "state_dict": model.state_dict(),
        "config": state_dict["config"]
    }, os.path.join(save_dir, f"epoch_{epoch+1}.pt"))

print("✅ Training completed.")
