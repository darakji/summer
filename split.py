import pandas as pd
import numpy as np

# === Load both Excel files ===
print("📥 Loading Excel files...")
df1 = pd.read_excel("/home/phanim/harshitrawat/summer/md/mdinfo_chgnet_predictions_forces.xlsx")
df2 = pd.read_excel("/home/phanim/harshitrawat/summer/md/strain_perturb_chgnet_predictions_forces.xlsx")

# === Concatenate all data ===
df_all = pd.concat([df1, df2], ignore_index=True)
unique_files = df_all["file"].unique()
print(f"📊 Total unique structures found: {len(unique_files)}")

# === Shuffle and Split ===
np.random.seed(42)
np.random.shuffle(unique_files)

split_idx = int(0.55 * len(unique_files))
t1_files = set(unique_files[:split_idx])
t2_files = set(unique_files[split_idx:])

print(f"🔀 Splitting into:")
print(f"   • T1 (train): {len(t1_files)} structures")
print(f"   • T2 (test):  {len(t2_files)} structures")
print(f"📄 Sample T1 files: {list(t1_files)[:3]}")
print(f"📄 Sample T2 files: {list(t2_files)[:3]}")

# === Assign rows to T1 and T2 ===
df_t1 = df_all[df_all["file"].isin(t1_files)].reset_index(drop=True)
df_t2 = df_all[df_all["file"].isin(t2_files)].reset_index(drop=True)

# === Save splits ===
print("💾 Saving splits...")
df_t1.to_excel("T1_chgnet_labeled.xlsx", index=False)
df_t2.to_excel("T2_chgnet_labeled.xlsx", index=False)

print("✅ Done!")
print(f"   • T1 → {df_t1.shape[0]} rows → T1_chgnet_labeled.xlsx")
print(f"   • T2 → {df_t2.shape[0]} rows → T2_chgnet_labeled.xlsx")
