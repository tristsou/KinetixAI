"""
Model 3: Flexible Label Transfer with Minimal Human Checking Images

Pipeline:
1. Train a Transformer "Teacher" on F-round using only the 10 true overlapping markers.
   - Labels are motion names (from motions_list), made consistent across subjects.
   - If a trained teacher checkpoint exists, load it instead of retraining.
2. Apply the Teacher to a chosen target round (F or S) to create pseudo-labels.
   - Allows F → S (original) and F → F (sanity check).
3. Optionally save a few video frames (one per predicted-action change, max 20 per subject)
   with the predicted label overlaid, allowing fast human verification.

Adjust these paths for your system:
- DATA_DIR_F
- DATA_DIR_S
- VIDEO_DIR
- VIDEO_FILENAME_PATTERN
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
import pandas as pd
import cv2
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "mps"  # for Apple Silicon (comment out if on Windows/Linux GPU)

WINDOW_SEC = 1.5
STEP_SEC = 0.5
FS = 120.0
T = int(round(WINDOW_SEC * FS))

BATCH_SIZE = 64
NUM_EPOCHS = 200
LR = 1e-4
VAL_RATIO = 0.2

D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 4
DIM_FF = 256
DROPOUT = 0.1

# ====== EDIT THESE BASED ON YOUR MACHINE ======
DATA_DIR_F = "/Users/edmundtsou/Projects/F_IMU/"
DATA_DIR_S = "/Users/edmundtsou/Projects/F_IMU/"

VIDEO_DIR = "/Users/edmundtsou/Projects/Video"
VIDEO_FILENAME_PATTERN = "S_CP1_Subject_{sid}.mp4"

FRAME_OUTPUT_DIR = "sround_labeled_frames"

# Teacher checkpoint (stores state_dict + class_names)
TEACHER_CKPT_PATH = f"model3_teacher_transformer_10markers_epoch{NUM_EPOCHS}.pt"

# Round selection for label transfer
SOURCE_ROUND = "F"   # teacher is trained on F
TARGET_ROUND = "S"   # choose "F" or "S"

# 10 true overlapping markers
OVERLAP_MARKERS = [
    "LFHD","LBHD","RFHD","RBHD",
    "LWRA","LWRB","RWRA","RWRB",
    "LANK","RANK"
]

# ============================================================
# DATASET CLASSES
# ============================================================

class FRoundMocapDataset(Dataset):
    def __init__(self, windows, labels):
        self.X = torch.tensor(windows, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx):
        return {"input_values": self.X[idx], "labels": self.y[idx]}


class WindowOnlyDataset(Dataset):
    def __init__(self, windows):
        self.X = torch.tensor(windows, dtype=torch.float32)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx):
        return {"input_values": self.X[idx]}


# ============================================================
# POSITIONAL ENCODING
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
        return x + self.pe[:x.size(0)]


# ============================================================
# TRANSFORMER CLASSIFIER
# ============================================================

class MocapTransformerClassifier(nn.Module):
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
            nn.Linear(d_model, num_classes)
        )

        self.num_classes = num_classes

    def forward(self, input_values, labels=None):
        B, T, C = input_values.shape
        x = self.input_proj(input_values)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        enc = self.encoder(x)
        pooled = enc.mean(dim=0)
        logits = self.cls_head(pooled)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}


# ============================================================
# CONFUSION MATRIX
# ============================================================

def compute_confusion_matrix(model, loader, class_names=None):
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

    print("\nClassification Report:")
    if class_names is not None:
        # Use only the classes that actually appear in trues/preds
        unique_classes = np.unique(np.concatenate([trues, preds]))
        label_names = [class_names[i] for i in unique_classes]
        print(
            classification_report(
                trues,
                preds,
                labels=unique_classes,
                target_names=label_names,
                digits=3,
            )
        )
        cm = confusion_matrix(trues, preds, labels=unique_classes)
    else:
        print(classification_report(trues, preds, digits=3))
        cm = confusion_matrix(trues, preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", annot=False)
    plt.title("Teacher Confusion Matrix (F-round)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()



# ============================================================
# TRAIN TEACHER ON F-ROUND
# ============================================================

def train_teacher_model(windows, labels, num_classes, class_names=None):
    dataset = FRoundMocapDataset(windows, labels)
    n_total = len(dataset)
    n_val = int(n_total * VAL_RATIO)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = MocapTransformerClassifier(
        input_dim=windows.shape[2],
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_ff=DIM_FF,
        num_classes=num_classes
    ).to(DEVICE)

    optim = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        model.train()

        for batch in train_loader:
            X = batch["input_values"].to(DEVICE)
            y = batch["labels"].to(DEVICE)
            out = model(X, y)
            loss = out["loss"]

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item() * X.size(0)

        avg = total_loss / n_train

        # validation accuracy
        correct, total = 0, 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                X = batch["input_values"].to(DEVICE)
                y = batch["labels"].to(DEVICE)
                preds = torch.argmax(model(X)["logits"], dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss={avg:.4f} | ValAcc={correct/total:.4f}")

    # confusion matrix
    compute_confusion_matrix(model, val_loader, class_names=class_names)
    return model


# ============================================================
# MARKER SUBSET EXTRACTION (10 OVERLAP MARKERS)
# ============================================================

def _subset_marker_location(move_struct, target_markers):
    """
    Extract only target_markers from move_struct.markerLocation.
    Handles markerLocation as (frames, markers, 3) or (frames, markers*3).
    Returns array of shape (n_frames, len(target_markers), 3).
    """
    raw = move_struct.markerName[0]
    marker_names = [str(m[0]) for m in raw]

    missing = [m for m in target_markers if m not in marker_names]
    if missing:
        raise ValueError(f"Missing markers: {missing}")

    idxs = [marker_names.index(m) for m in target_markers]

    marker_loc = np.nan_to_num(move_struct.markerLocation)

    if marker_loc.ndim == 3:
        # (frames, markers_total, 3)
        marker_loc = marker_loc[:, idxs, :]
    elif marker_loc.ndim == 2:
        # (frames, markers_total*3) -> reshape
        n_frames, feat_dim = marker_loc.shape
        n_markers_total = len(marker_names)
        assert feat_dim % n_markers_total == 0, "markerLocation shape mismatch"
        C = feat_dim // n_markers_total
        assert C == 3, f"Expected 3D coords, got C={C}"
        marker_loc = marker_loc.reshape(n_frames, n_markers_total, C)
        marker_loc = marker_loc[:, idxs, :]
    else:
        raise ValueError(f"Unexpected markerLocation ndim={marker_loc.ndim}")

    return marker_loc  # (n_frames, len(target_markers), 3)


# ============================================================
# BUILD F-ROUND WINDOWS (FOR TRAINING) WITH MOTION-NAME LABELS
# ============================================================

def build_fround_windows_and_labels_10markers():
    """
    Build windows + labels for F-round using 10 overlapping markers
    and motion names from motions_list as the semantic labels.
    Returns:
      windows: (N, T, 30)
      labels:  (N,)
      class_names: sorted list of unique motion names
    """
    paths = sorted(glob.glob(os.path.join(DATA_DIR_F, "F_v3d_Subject_*.mat")))
    if not paths:
        raise FileNotFoundError("No F_v3d_Subject_*.mat found")

    T_frames = int(WINDOW_SEC * FS)
    step = int(STEP_SEC * FS)

    X_all = []
    label_names_all = []

    for p in paths:
        print(f"[F-round] Loading {p}")
        mat = loadmat(p, struct_as_record=False, squeeze_me=False)
        key = [k for k in mat.keys() if k.endswith("_F")][0]
        subj = mat[key][0, 0]
        move = subj.move[0, 0][0, 0]

        # 1) subset markers
        loc = _subset_marker_location(move, OVERLAP_MARKERS)
        n_frames = loc.shape[0]

        # 2) get motion names for each local act_id
        raw_motion_list = move.motions_list  # e.g. shape (21,1)
        motion_names_local = [str(m[0][0]) for m in raw_motion_list]

        # 3) build per-frame motion-name labels via flags120
        frame_labels_names = np.array([""] * n_frames, dtype=object)
        for local_act_id, (s, e) in enumerate(move.flags120):
            s_idx = int(s) - 1
            e_idx = int(e)
            motion_name = motion_names_local[local_act_id]
            frame_labels_names[s_idx:e_idx] = motion_name

        # 4) sliding windows
        for s in range(0, n_frames - T_frames, step):
            e = s + T_frames
            center = s + T_frames // 2

            motion_name = frame_labels_names[center]
            if motion_name == "":
                continue

            win = loc[s:e].reshape(T_frames, -1)  # (T_frames, 30)
            X_all.append(win)
            label_names_all.append(motion_name)

    windows = np.stack(X_all)
    label_names_all = np.array(label_names_all, dtype=object)

    # 5) map motion names to global IDs
    class_names = sorted(np.unique(label_names_all))
    name_to_id = {name: i for i, name in enumerate(class_names)}
    labels = np.array([name_to_id[name] for name in label_names_all], dtype=np.int64)

    print("Built F-round (training, 10 markers, motion-name labels):")
    print("windows:", windows.shape)
    print("labels:", labels.shape)
    print("num_classes:", len(class_names))
    print("class_names:", class_names)

    return windows, labels, class_names


# ============================================================
# BUILD WINDOWS FOR TARGET ROUND (F OR S) FOR LABEL TRANSFER
# ============================================================

def build_round_windows(round_type):
    """
    Build windows + meta for target round (F or S) for label transfer.
    Does NOT return labels, just windows and meta info.
    """
    if round_type not in ["F", "S"]:
        raise ValueError("round_type must be 'F' or 'S'")

    if round_type == "F":
        data_dir = DATA_DIR_F
        pattern = "F_v3d_Subject_*.mat"
        key_suffix = "_F"
    else:
        data_dir = DATA_DIR_S
        pattern = "S_v3d_Subject_*.mat"
        key_suffix = "_S"

    paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No {pattern} found in {data_dir}")

    T_frames = int(WINDOW_SEC * FS)
    step = int(STEP_SEC * FS)

    X_all = []
    meta = {"subject_id": [], "center_frame": []}

    for p in paths:
        print(f"[{round_type}-round] Loading {p}")
        mat = loadmat(p, struct_as_record=False, squeeze_me=False)
        key = [k for k in mat.keys() if k.endswith(key_suffix)][0]
        subj = mat[key][0, 0]
        move = subj.move[0, 0][0, 0]

        loc = _subset_marker_location(move, OVERLAP_MARKERS)
        n_frames = loc.shape[0]

        sid = p.split("Subject_")[-1].split(".")[0]

        for s in range(0, n_frames - T_frames, step):
            e = s + T_frames
            c = s + T_frames // 2
            win = loc[s:e].reshape(T_frames, -1)

            X_all.append(win)
            meta["subject_id"].append(sid)
            meta["center_frame"].append(c)

    X_all = np.stack(X_all)
    meta = {k: np.array(v) for k, v in meta.items()}

    print(f"Built {round_type}-round (target) windows:", X_all.shape)
    return X_all, meta


# ============================================================
# LABEL TRANSFER + MINIMAL IMAGE SAVING (GENERAL)
# ============================================================

def generate_pseudo_labels(
    model,
    windows,
    meta,
    class_names=None,
    output_csv="pseudo_labels.csv",
    save_frames=True,
    video_dir=None,
    video_filename_pattern=None,
    frame_output_dir=FRAME_OUTPUT_DIR,
):
    """
    Generic label transfer function:
    - model: trained teacher
    - windows: [N, T, C]
    - meta: dict with 'subject_id' and 'center_frame'
    - class_names: list of motion-name strings (maps class_id -> name)
    - If save_frames is True and video_dir / video_filename_pattern are set,
      save a few frames with predicted labels.
    """
    model.eval()
    loader = DataLoader(WindowOnlyDataset(windows), batch_size=BATCH_SIZE)

    preds_all = []
    last_label = {}
    saved_per_subject = defaultdict(int)
    MAX_IMAGES = 20  # max images per subject

    video_caps = {}

    if save_frames:
        os.makedirs(frame_output_dir, exist_ok=True)

    idx_global = 0
    with torch.no_grad():
        for batch in loader:
            X = batch["input_values"].to(DEVICE)
            logits = model(X)["logits"]
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            for i in range(len(preds)):
                pred = int(preds[i])
                preds_all.append(pred)

                sid = meta["subject_id"][idx_global]
                center = int(meta["center_frame"][idx_global])

                # Only save image when label changes
                save = False
                if sid not in last_label:
                    save = True
                elif pred != last_label[sid]:
                    save = True

                if save and save_frames and saved_per_subject[sid] < MAX_IMAGES:
                    last_label[sid] = pred
                    saved_per_subject[sid] += 1

                    if video_dir is None or video_filename_pattern is None:
                        pass
                    else:
                        video_file = video_filename_pattern.format(sid=sid)
                        video_path = os.path.join(video_dir, video_file)

                        if not os.path.exists(video_path):
                            print(f"[WARN] Missing video: {video_path}")
                        else:
                            if sid not in video_caps:
                                cap = cv2.VideoCapture(video_path)
                                if not cap.isOpened():
                                    video_caps[sid] = None
                                    print(f"[WARN] Could not open {video_path}")
                                else:
                                    video_caps[sid] = cap
                            cap = video_caps.get(sid, None)

                            if cap is not None:
                                fps = cap.get(cv2.CAP_PROP_FPS)
                                if fps <= 0:
                                    fps = 30.0

                                t = center / FS
                                vid_frame = int(round(t * fps))
                                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                vid_frame = max(0, min(vid_frame, n_frames - 1))

                                cap.set(cv2.CAP_PROP_POS_FRAMES, vid_frame)
                                ret, frame = cap.read()
                                if ret:
                                    if class_names is not None and 0 <= pred < len(class_names):
                                        label_text = class_names[pred]
                                    else:
                                        label_text = f"class {pred}"

                                    cv2.putText(
                                        frame,
                                        f"Pred: {label_text}",
                                        (30, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1.0,
                                        (0, 255, 0),
                                        2
                                    )
                                    out_name = (
                                        f"subj_{sid}_img{saved_per_subject[sid]:02d}_"
                                        f"cf{center}_pred{pred}.png"
                                    )
                                    cv2.imwrite(os.path.join(frame_output_dir, out_name), frame)
                                else:
                                    print(f"[WARN] Can't read frame {vid_frame} from {video_path}")

                idx_global += 1

    # release all videos
    for sid, cap in video_caps.items():
        if cap is not None:
            cap.release()

    preds_all = np.array(preds_all, dtype=int)

    if class_names is not None:
        pred_names = np.array([class_names[p] for p in preds_all], dtype=object)
    else:
        pred_names = np.array([""] * len(preds_all), dtype=object)

    df = pd.DataFrame({
        "subject_id": meta["subject_id"],
        "center_frame": meta["center_frame"],
        "pred_label_id": preds_all,
        "pred_label_name": pred_names,
    })
    df.to_csv(output_csv, index=False)
    print(f"Saved pseudo-labels → {output_csv}")
    if save_frames and video_dir is not None and video_filename_pattern is not None:
        print(f"Saved sample frames → {frame_output_dir}/")
    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print(">>> Model 3 Starting")
    print(f">>> Using device: {DEVICE}")

    # --------------------------------------------------------
    # 1) Build F-round windows/labels for teacher (source round)
    # --------------------------------------------------------
    print(">>> Building F-round (teacher training/loading data)...")
    Xf, yf, class_names = build_fround_windows_and_labels_10markers()
    input_dim = Xf.shape[2]
    num_classes = len(class_names)

    # --------------------------------------------------------
    # 2) Load existing teacher checkpoint OR train new teacher
    # --------------------------------------------------------
    if os.path.exists(TEACHER_CKPT_PATH):
        print(f">>> Found existing teacher checkpoint at {TEACHER_CKPT_PATH}. Loading...")
        ckpt = torch.load(TEACHER_CKPT_PATH, map_location=DEVICE)

        # If checkpoint has its own class_names, use them; else use current
        if isinstance(ckpt, dict) and "state_dict" in ckpt and "class_names" in ckpt:
            class_names = list(ckpt["class_names"])
            num_classes = len(class_names)
            state_dict = ckpt["state_dict"]
        else:
            # old-style checkpoint (just state_dict)
            state_dict = ckpt

        teacher = MocapTransformerClassifier(
            input_dim=input_dim,
            d_model=D_MODEL,
            nhead=NHEAD,
            num_layers=NUM_LAYERS,
            dim_ff=DIM_FF,
            num_classes=num_classes
        ).to(DEVICE)
        teacher.load_state_dict(state_dict)
    else:
        print(">>> No teacher checkpoint found. Training new teacher...")
        teacher = train_teacher_model(Xf, yf, num_classes=num_classes, class_names=class_names)
        torch.save(
            {
                "state_dict": teacher.state_dict(),
                "class_names": class_names,
            },
            TEACHER_CKPT_PATH
        )
        print(f">>> Saved teacher model → {TEACHER_CKPT_PATH}")

    # --------------------------------------------------------
    # 3) Build target round windows (F or S) for label transfer
    # --------------------------------------------------------
    print(f">>> Building target round ({TARGET_ROUND}) windows for label transfer...")
    Xt, meta_t = build_round_windows(TARGET_ROUND)

    # --------------------------------------------------------
    # 4) Run label transfer + optional minimal image saving
    # --------------------------------------------------------
    print(f">>> Running Label Transfer {SOURCE_ROUND} → {TARGET_ROUND} ...")
    output_csv = f"pseudo_labels_{SOURCE_ROUND}to{TARGET_ROUND}_model3_motionnames.csv"

    if TARGET_ROUND == "S":
        video_dir = VIDEO_DIR
        video_pattern = VIDEO_FILENAME_PATTERN
        save_frames = True
        frame_dir = FRAME_OUTPUT_DIR
    else:
        # F→F sanity check
        video_dir = VIDEO_DIR
        video_pattern = VIDEO_FILENAME_PATTERN
        save_frames = True
        frame_dir = FRAME_OUTPUT_DIR

    generate_pseudo_labels(
        teacher,
        Xt,
        meta_t,
        class_names=class_names,
        output_csv=output_csv,
        save_frames=save_frames,
        video_dir=video_dir,
        video_filename_pattern=video_pattern,
        frame_output_dir=frame_dir,
    )

    print(">>> Done.")
