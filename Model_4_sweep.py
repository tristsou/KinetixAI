"""
Model 4 Extended: IMU Student Model with Joint-Level Ablations

- Trains a transformer on S1 IMU windows using pseudo labels from Model 3.
- Then runs:
  1) Single-joint sweep: 21 runs, each using only one joint's IMU features.
  2) Pair-joint sweep: all C(21, 2) joint pairs.

For every run we:
  - Train from scratch on the selected feature subset.
  - Log best / final validation accuracy to a CSV.
  - Save a detailed classification report to a text file.
  - Save a model checkpoint (state_dict + feature_indices).

Assumptions:
- Pseudo labels are stored in PSEUDO_LABEL_CSV with columns:
    subject_id, center_frame, pred_label_name
- S1 IMU files are imu_Subject_*.mat with IMU.S1_Synched.data
  and IMU.S1_Synched.jointNames.

Adjust paths and high-level hyperparameters as needed.
"""

import os
import glob
import csv
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.io import loadmat
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "mps"  # comment out on non-Apple machines

WINDOW_SEC = 1.5
STEP_SEC = 0.5
FS = 120.0

BATCH_SIZE = 64
NUM_EPOCHS = 50
LR = 1e-4
VAL_RATIO = 0.2

D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 4
DIM_FF = 256
DROPOUT = 0.1

# Paths
DATA_DIR_S_IMU = "/Users/edmundtsou/Projects/S_IMU/"
PSEUDO_LABEL_CSV = "pseudo_labels_FtoS_model3_motionnames.csv"

CKPT_DIR = "model4_checkpoints"
LOG_DIR_SINGLE = "logs_single_joint"
LOG_DIR_PAIR = "logs_pair_joint"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR_SINGLE, exist_ok=True)
os.makedirs(LOG_DIR_PAIR, exist_ok=True)

LOG_CSV_SINGLE = os.path.join(LOG_DIR_SINGLE, "single_joint_results.csv")
LOG_CSV_PAIR = os.path.join(LOG_DIR_PAIR, "pair_joint_results.csv")

DO_SINGLE_JOINT_SWEEP = True
DO_PAIR_JOINT_SWEEP = True


# ============================================================
# DATASET
# ============================================================

class IMURoundDataset(Dataset):
    def __init__(self, windows, labels):
        self.X = torch.tensor(windows, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {"input_values": self.X[idx], "labels": self.y[idx]}


# ============================================================
# MODEL
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (T, B, D)
        return x + self.pe[:x.size(0)]


class IMUTransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_ff, num_classes):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=DROPOUT,
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, input_values, labels=None):
        # input_values: (B, T, C)
        B, T, C = input_values.shape
        x = self.input_proj(input_values)   # (B, T, D)
        x = x.transpose(0, 1)              # (T, B, D)
        x = self.pos_encoder(x)
        enc = self.encoder(x)              # (T, B, D)
        pooled = enc.mean(dim=0)           # (B, D)
        logits = self.cls_head(pooled)     # (B, num_classes)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}


# ============================================================
# HELPERS: IMU LOADING & JOINT NAMES
# ============================================================

def load_imu_s1_from_mat(path):
    """
    Load S1_Synched.data from imu_Subject_*.mat.
    Returns: np.ndarray (n_frames, n_channels) or None if unavailable.
    """
    mat = loadmat(path, struct_as_record=False, squeeze_me=False)
    if "IMU" not in mat:
        print(f"[WARN] 'IMU' key not in {path}, skipping.")
        return None

    imu_root = mat["IMU"][0, 0]
    if not hasattr(imu_root, "S1_Synched"):
        print(f"[WARN] No S1_Synched in {os.path.basename(path)}, skipping.")
        return None

    seg = imu_root.S1_Synched[0, 0]
    data = np.asarray(seg.data)

    if data.ndim != 2:
        print(f"[WARN] Expected 2D IMU array in {path}, got {data.shape}, skipping.")
        return None

    return np.nan_to_num(data)


def get_joint_names_from_any_file(data_dir_imu):
    """
    Read jointNames from the first imu_Subject_*.mat with S1_Synched.
    Returns: list of joint name strings.
    """
    paths = sorted(glob.glob(os.path.join(data_dir_imu, "imu_Subject_*.mat")))
    if not paths:
        raise FileNotFoundError(f"No imu_Subject_*.mat found in {data_dir_imu}")

    for p in paths:
        mat = loadmat(p, struct_as_record=False, squeeze_me=False)
        if "IMU" not in mat:
            continue
        imu_root = mat["IMU"][0, 0]
        if not hasattr(imu_root, "S1_Synched"):
            continue
        seg = imu_root.S1_Synched[0, 0]
        if not hasattr(seg, "jointNames"):
            continue

        jn = seg.jointNames
        # typically shape (1, 21)
        names = []
        for i in range(jn.shape[1]):
            names.append(str(jn[0, i][0]))
        return names

    raise RuntimeError("Could not find jointNames in any imu_Subject_*.mat")


# ============================================================
# BUILD WINDOWS + LABELS
# ============================================================

def build_sround_imu_windows_and_labels_from_pseudo(
    data_dir_imu,
    pseudo_csv,
    window_sec=WINDOW_SEC,
    step_sec=STEP_SEC,
    fs=FS,
):
    """
    Build S1 IMU windows and labels using pseudo labels from Model 3.

    Returns:
        windows: (N, T_frames, C_full)  -- normalized per channel
        labels:  (N,)
        class_names: list of motion labels
    """
    df = pd.read_csv(pseudo_csv)
    required = {"subject_id", "center_frame", "pred_label_name"}
    if not required.issubset(df.columns):
        raise ValueError(
            "Pseudo label CSV must contain subject_id, center_frame, pred_label_name"
        )

    label_names = df["pred_label_name"].astype(str).values
    class_names = sorted(np.unique(label_names))
    name_to_id = {name: i for i, name in enumerate(class_names)}

    label_map = {
        (str(sid), int(cf)): lbl
        for sid, cf, lbl in zip(df["subject_id"], df["center_frame"], label_names)
    }

    print(f"Loaded pseudo labels from {pseudo_csv}")
    print(f"Num pseudo windows: {len(df)}")
    print(f"Num classes: {len(class_names)}")
    print("class_names:", class_names)

    paths = sorted(glob.glob(os.path.join(data_dir_imu, "imu_Subject_*.mat")))
    if not paths:
        raise FileNotFoundError(f"No imu_Subject_*.mat found in {data_dir_imu}")

    T_frames = int(round(window_sec * fs))
    step = int(round(step_sec * fs))

    X_all = []
    y_all = []
    total_windows = 0
    matched_windows = 0

    for p in paths:
        print(f"[S1 IMU] Loading {p}")
        imu_arr = load_imu_s1_from_mat(p)
        if imu_arr is None:
            continue

        n_frames, _ = imu_arr.shape
        sid = p.split("Subject_")[-1].split(".")[0]

        for s in range(0, n_frames - T_frames, step):
            e = s + T_frames
            c = s + T_frames // 2
            total_windows += 1

            key = (str(sid), int(c))
            if key not in label_map:
                continue

            lbl_name = label_map[key]
            lbl_id = name_to_id[lbl_name]

            win = imu_arr[s:e, :]
            X_all.append(win)
            y_all.append(lbl_id)
            matched_windows += 1

    if not X_all:
        raise RuntimeError("No S1 IMU windows matched pseudo labels.")

    windows = np.stack(X_all)
    labels = np.array(y_all, dtype=np.int64)

    print("=========================================")
    print("Built S1 IMU windows (pseudo-labeled):")
    print(" total_windows_considered:", total_windows)
    print(" matched_windows (used):", matched_windows)
    print(" windows shape:", windows.shape)
    print(" labels shape:", labels.shape)
    print("=========================================")

    # Per-channel normalization
    C = windows.shape[2]
    mean = windows.reshape(-1, C).mean(axis=0)
    std = windows.reshape(-1, C).std(axis=0) + 1e-8
    windows = (windows - mean) / std

    print("Applied per-channel normalization.")
    return windows, labels, class_names


# ============================================================
# TRAIN / EVAL FOR A FEATURE SUBSET
# ============================================================

def evaluate_model(model, loader, class_names):
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for batch in loader:
            X = batch["input_values"].to(DEVICE)
            y = batch["labels"].cpu().numpy()
            out = model(X)["logits"].cpu().numpy()
            preds.extend(np.argmax(out, axis=1))
            trues.extend(y)

    trues = np.array(trues)
    preds = np.array(preds)

    if len(trues) == 0:
        # Degenerate case: empty validation set
        return 0.0, "Empty validation set; no report.", np.zeros((0, 0), dtype=int)

    acc = (trues == preds).mean()

    # Only use the classes that actually appear in trues/preds
    unique_classes = np.unique(np.concatenate([trues, preds]))
    label_names = [class_names[i] for i in unique_classes]

    report = classification_report(
        trues,
        preds,
        labels=unique_classes,
        target_names=label_names,
        digits=3,
    )
    cm = confusion_matrix(trues, preds, labels=unique_classes)
    return acc, report, cm

def train_student_model_imu_subset(
    windows,
    labels,
    class_names,
    feature_indices,
    run_tag,
    log_dir,
    log_csv,
):
    """
    Train transformer using only selected feature_indices.
    Logs metrics, saves report + checkpoint, appends to CSV.
    """
    X_sub = windows[:, :, feature_indices]
    num_classes = len(class_names)
    input_dim = X_sub.shape[2]

    dataset = IMURoundDataset(X_sub, labels)
    n_total = len(dataset)
    n_val = int(n_total * VAL_RATIO)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = IMUTransformerClassifier(
        input_dim=input_dim,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_ff=DIM_FF,
        num_classes=num_classes,
    ).to(DEVICE)

    optim = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val_acc = 0.0
    best_epoch = -1

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            X = batch["input_values"].to(DEVICE)
            y = batch["labels"].to(DEVICE)

            out = model(X, y)
            loss = out["loss"]

            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item() * X.size(0)

        avg_loss = total_loss / n_train

        # quick val acc
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                X = batch["input_values"].to(DEVICE)
                y = batch["labels"].to(DEVICE)
                preds = torch.argmax(model(X)["logits"], dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_acc = correct / total if total > 0 else 0.0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1

        print(
            f"[{run_tag}] Epoch {epoch+1}/{NUM_EPOCHS} "
            f"Loss={avg_loss:.4f} ValAcc={val_acc:.4f} "
            f"(best={best_val_acc:.4f} @ {best_epoch})"
        )

    final_val_acc, report_str, cm = evaluate_model(model, val_loader, class_names)

    # Save report
    report_path = os.path.join(log_dir, f"{run_tag}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Run: {run_tag}\n")
        f.write(f"Best ValAcc (epoch {best_epoch}): {best_val_acc:.4f}\n")
        f.write(f"Final ValAcc: {final_val_acc:.4f}\n\n")
        f.write(report_str)
    print(f"[{run_tag}] Saved report → {report_path}")

    # Save checkpoint
    ckpt_path = os.path.join(CKPT_DIR, f"{run_tag}_student_epoch{NUM_EPOCHS}.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": class_names,
            "feature_indices": np.array(feature_indices, dtype=int),
        },
        ckpt_path,
    )
    print(f"[{run_tag}] Saved checkpoint → {ckpt_path}")

    # Append to CSV
    header = [
        "run_tag",
        "n_features",
        "feature_indices",
        "best_val_acc",
        "best_epoch",
        "final_val_acc",
        "n_train",
        "n_val",
    ]
    file_exists = os.path.exists(log_csv)
    with open(log_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(
            [
                run_tag,
                len(feature_indices),
                ";".join(map(str, feature_indices)),
                f"{best_val_acc:.6f}",
                best_epoch,
                f"{final_val_acc:.6f}",
                n_train,
                n_val,
            ]
        )

    return best_val_acc, final_val_acc


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print(">>> Model 4 Extended (Joint Ablations) starting")
    print(f">>> Using device: {DEVICE}")

    # Build data
    windows, labels, class_names = build_sround_imu_windows_and_labels_from_pseudo(
        data_dir_imu=DATA_DIR_S_IMU,
        pseudo_csv=PSEUDO_LABEL_CSV,
        window_sec=WINDOW_SEC,
        step_sec=STEP_SEC,
        fs=FS,
    )
    N, T_frames, C_full = windows.shape
    print(f"Data ready: N={N}, T={T_frames}, C_full={C_full}")

    # Get joint names and feature grouping
    joint_names = get_joint_names_from_any_file(DATA_DIR_S_IMU)
    N_JOINTS = len(joint_names)
    if C_full % N_JOINTS != 0:
        print(
            f"[WARN] C_full={C_full} not divisible by N_JOINTS={N_JOINTS}. "
            "Feature grouping may be off."
        )
    FEAT_PER_JOINT = C_full // N_JOINTS

    print("Joint groups:")
    for idx, name in enumerate(joint_names):
        start = idx * FEAT_PER_JOINT
        end = (idx + 1) * FEAT_PER_JOINT
        print(f"  {idx:2d}: {name:12s} → channels [{start}:{end})")

    # ---------------- Single-joint sweep ----------------
    if DO_SINGLE_JOINT_SWEEP:
        print("\n>>> Starting single-joint sweep...")
        for j_idx, j_name in enumerate(joint_names):
            start = j_idx * FEAT_PER_JOINT
            end = (j_idx + 1) * FEAT_PER_JOINT
            feature_indices = list(range(start, end))
            run_tag = f"single_{j_idx:02d}_{j_name}"

            print(f"\n=== {run_tag}: channels [{start}:{end}) ===")
            train_student_model_imu_subset(
                windows,
                labels,
                class_names,
                feature_indices,
                run_tag,
                log_dir=LOG_DIR_SINGLE,
                log_csv=LOG_CSV_SINGLE,
            )

    # ---------------- Pair-joint sweep ----------------
    if DO_PAIR_JOINT_SWEEP:
        print("\n>>> Starting pair-joint sweep...")
        for (i_idx, j_idx) in combinations(range(N_JOINTS), 2):
            name_i = joint_names[i_idx]
            name_j = joint_names[j_idx]

            start_i, end_i = i_idx * FEAT_PER_JOINT, (i_idx + 1) * FEAT_PER_JOINT
            start_j, end_j = j_idx * FEAT_PER_JOINT, (j_idx + 1) * FEAT_PER_JOINT
            feature_indices = list(range(start_i, end_i)) + list(range(start_j, end_j))

            run_tag = f"pair_{i_idx:02d}_{name_i}__{j_idx:02d}_{name_j}"
            print(
                f"\n=== {run_tag}: "
                f"{name_i} [{start_i}:{end_i}), "
                f"{name_j} [{start_j}:{end_j}) ==="
            )

            train_student_model_imu_subset(
                windows,
                labels,
                class_names,
                feature_indices,
                run_tag,
                log_dir=LOG_DIR_PAIR,
                log_csv=LOG_CSV_PAIR,
            )

    print("\n>>> All sweeps done.")
