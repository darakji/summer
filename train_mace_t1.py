import torch
import functools

# Patch torch.load to disable "safe" mode for now
_original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = patched_load

from mace.tools import torch_geometric
from mace.train import train
from mace.calculators import MACECalculator
from mace.data.utils import load_atomic_dataset
from mace.modules import MACE

import os
print("Imports done\n")
# ====== Configs (equivalent to CLI args) ======
init_model_path = "/home/phanim/harshitrawat/summer/mace_models/universal/2024-01-07-mace-128-L2_epoch-199.model"
train_file = "/home/phanim/harshitrawat/summer/mace_train_T1.db"
save_dir = "/home/phanim/harshitrawat/summer/mace_t1_finetuned_large"
device = "cuda:0"

batch_size = 8
valid_fraction = 0.1
max_epochs = 100
learning_rate = 5e-4
ema_decay = 0.99
log_every = 20
default_dtype = torch.float32

weights = dict(energy=1.0, forces=1.0, stress=1.0)

# ====== CUDA Info ======
assert torch.cuda.is_available(), "CUDA is not available!"
torch.cuda.set_device(device)
print(f"✅ Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}\n")

# ====== Load dataset ======
print("📦 Loading dataset...")
dataset = load_atomic_dataset([train_file])
train_data, val_data = dataset.split(train_fraction=1 - valid_fraction)

print(f"✅ Dataset loaded.")
print(f"   - Training samples: {len(train_data)}")
print(f"   - Validation samples: {len(val_data)}\n")

# ====== Load pretrained model ======
print("📥 Loading pretrained model...")
state_dict = torch.load(init_model_path, map_location="cpu")
model_config = state_dict["config"]
model = MACE(**model_config)
model.load_state_dict(state_dict["state_dict"])
model = model.to(device=device, dtype=default_dtype)
print("✅ Pretrained model loaded.\n")

# ====== Training ======
print("🚀 Starting training...\n")

train(
    model=model,
    train_data=train_data,
    valid_data=val_data,
    save_dir=save_dir,
    batch_size=batch_size,
    max_num_epochs=max_epochs,
    learning_rate=learning_rate,
    ema_decay=ema_decay,
    energy_weight=weights["energy"],
    forces_weight=weights["forces"],
    stress_weight=weights["stress"],
    log_every_n_steps=log_every,
    default_dtype=default_dtype,
    device=device,
)

print("\n✅ Training completed.")

# ====== GPU Memory Report ======
allocated = torch.cuda.memory_allocated() / 1e6
reserved = torch.cuda.memory_reserved() / 1e6
print(f"📊 Final GPU Memory Usage: Allocated = {allocated:.1f} MB | Reserved = {reserved:.1f} MB")
