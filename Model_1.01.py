"""
Model 1: Custom Time-Series Transformer (PyTorch) for F-round MoCap → Action Classes
NOW WITH CONFUSION MATRIX
(Labels derived from motion names, not per-subject numeric IDs)
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "mps"  # comment this out if you don't want MPS

WINDOW_SEC = 1.5
STEP_SEC = 0.5
FS = 120.0
T = int(round(WINDOW_SEC * FS))

BATCH_SIZE = 64
NUM_EPOCHS = 200
LR = 5e-5
VAL_RATIO = 0.2

D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 4
DIM_FF = 256
DROPOUT = 0.1

# >>> EDIT THIS ONLY <<<
DATA_DIR_F = "/Users/edmundtsou/Projects/F_IMU/"


# ============================================================
# DATASET
# ============================================================

class FRoundMocapDataset(Dataset):
    def __init__(self, windows, labels):
        self.X = torch.tensor(windows, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return {"input_values": self.X[idx], "labels": self.y[idx]}


# ============================================================
# POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)
    def forward(self, x): return x + self.pe[:x.size(0)]


# ============================================================
# TRANSFORMER MODEL
# ============================================================

class MocapTransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_ff, num_classes):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=DROPOUT,
            batch_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, input_values, labels=None):
        B, T, C = input_values.shape
        x = self.input_proj(input_values)       # (B, T, D)
        x = x.transpose(0, 1)                   # (T, B, D)
        x = self.pos_encoder(x)                 # add sin/cos
        enc = self.encoder(x)                   # (T, B, D)

        pooled = enc.mean(dim=0)                # (B, D)
        logits = self.cls_head(pooled)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}


# ============================================================
# CONFUSION MATRIX HELPER
# ============================================================

def compute_confusion_matrix(model, dataloader, device=DEVICE):
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for batch in dataloader:
            X = batch["input_values"].to(device)
            y = batch["labels"].cpu().numpy()

            logits = model(X)["logits"].cpu().numpy()
            preds = np.argmax(logits, axis=1)

            all_preds.extend(preds)
            all_true.extend(y)

    cm = confusion_matrix(all_true, all_preds)
    print("\nClassification Report:")
    print(classification_report(all_true, all_preds, digits=3))

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.title("Confusion Matrix (Validation Set)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    return cm


# ============================================================
# TRAINING LOOP
# ============================================================

def train_model_1(windows, labels, num_classes, batch_size=BATCH_SIZE,
                  num_epochs=NUM_EPOCHS, lr=LR):
    dataset = FRoundMocapDataset(windows, labels)

    n_total = len(dataset)
    n_val = int(n_total * VAL_RATIO)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    num_channels = windows.shape[2]
    model = MocapTransformerClassifier(
        input_dim=num_channels,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_ff=DIM_FF,
        num_classes=num_classes
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # --- TRAIN ---
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            X = batch["input_values"].to(DEVICE)
            y = batch["labels"].to(DEVICE)

            out = model(X, y)
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.size(0)

        avg_loss = total_loss / n_train

        # --- VAL ACC ---
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                X = batch["input_values"].to(DEVICE)
                y = batch["labels"].to(DEVICE)
                preds = torch.argmax(model(X)["logits"], axis=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss={avg_loss:.4f} | Val Acc={val_acc:.4f}")

    print("\n=== Confusion Matrix on Validation Set ===")
    compute_confusion_matrix(model, val_loader)

    return model


# ============================================================
# DATA LOADING (F-round) — USE MOTION NAMES AS LABELS
# ============================================================

def build_fround_windows_and_labels():
    """
    Build windows and labels for all F-round subjects.

    For each subject:
      - Use move_struct.motions_list[local_act_id] to get the motion_name string.
      - Assign that motion_name frame-wise, then at window level.
    Then:
      - Collect all motion_name labels across all windows.
      - Build a global mapping: motion_name -> int ID.
      - Return windows and integer labels, plus the list of class names.
    """
    f_paths = sorted(glob.glob(os.path.join(DATA_DIR_F, "F_v3d_Subject_*.mat")))
    if not f_paths:
        raise FileNotFoundError("No F_v3d_Subject_*.mat found.")

    all_windows = []
    all_label_names = []   # motion_name for each window

    T_frames = int(round(WINDOW_SEC * FS))
    step = int(round(STEP_SEC * FS))

    for f_path in f_paths:
        print(f"\nLoading {f_path}")
        mat = loadmat(f_path, struct_as_record=False, squeeze_me=False)

        subj_key = [k for k in mat.keys() if k.startswith("Subject") and k.endswith("_F")][0]
        subj = mat[subj_key][0, 0]

        move_struct = subj.move[0, 0][0, 0]
        marker_loc = move_struct.markerLocation
        marker_loc = np.nan_to_num(marker_loc)
        n_frames = marker_loc.shape[0]

        flags120 = move_struct.flags120

        # motions_list gives the local mapping from act_id → motion_name
        raw_motion_list = move_struct.motions_list  # e.g. shape (21,1)
        motion_names_local = [str(m[0][0]) for m in raw_motion_list]

        # frame_labels_names will hold the motion_name string for each frame
        frame_labels_names = np.array([""] * n_frames, dtype=object)

        for local_act_id, (s, e) in enumerate(flags120):
            s = int(s) - 1
            e = int(e)
            motion_name = motion_names_local[local_act_id]
            frame_labels_names[s:e] = motion_name

        # Sliding windows
        for start in range(0, n_frames - T_frames, step):
            end = start + T_frames
            center = start + T_frames // 2

            motion_name = frame_labels_names[center]
            if motion_name == "":
                continue  # unlabeled region

            window = marker_loc[start:end].reshape(T_frames, -1)
            all_windows.append(window)
            all_label_names.append(motion_name)

    windows = np.stack(all_windows)
    all_label_names = np.array(all_label_names, dtype=object)

    # Build class list from motion names (no manual global list)
    class_names = sorted(np.unique(all_label_names))
    name_to_id = {name: i for i, name in enumerate(class_names)}
    labels = np.array([name_to_id[name] for name in all_label_names], dtype=np.int64)

    print("\nBuilt dataset (labels based on motion_name):")
    print("windows:", windows.shape)
    print("labels (int):", labels.shape)
    print("num_classes:", len(class_names))
    print("class_names:", class_names)

    return windows, labels, class_names


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    windows, labels, class_names = build_fround_windows_and_labels()
    num_classes = len(class_names)

    model = train_model_1(windows, labels, num_classes=num_classes)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": class_names,
        },
        "model1_mocap_transformer_motionnames.pt",
    )
    print("\nSaved model → model1_mocap_transformer_motionnames.pt")
