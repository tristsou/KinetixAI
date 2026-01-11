import os
import glob
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
LOG_DIR_SINGLE = "logs_single_joint"  # folder with single_XX_XXX_classification_report.txt
OUTPUT_DIR = "analysis_single_joint"
os.makedirs(OUTPUT_DIR, exist_ok=True)

REPORT_GLOB = os.path.join(LOG_DIR_SINGLE, "single_*_classification_report.txt")

# Regex for data rows in classification_report
row_pattern = re.compile(
    r"^\s*(\S.+?)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s*$"
)


# -------------------------------------------------------------------
# PARSER FOR A SINGLE REPORT FILE
# -------------------------------------------------------------------
def parse_single_report(path: str):
    """
    Parse one classification report file like:
      single_00_Hip_classification_report.txt

    Assumes file structure like:

      Run: single_00_Hip
      Best ValAcc (epoch 45): 0.5799
      Final ValAcc: 0.5571

                            precision    recall  f1-score   support
            checking_watch      ...
            ...

                    accuracy                          0.557       219
                   macro avg      ...
                weighted avg      ...

    Returns:
        run_info: dict with run_tag, joint, best_val_acc, final_val_acc, path
        rows: list of dicts with columns:
              run_tag, joint, label, precision, recall, f1, support
    """
    with open(path, "r") as f:
        lines = [ln.rstrip("\n") for ln in f]

    if not lines:
        raise ValueError(f"Empty file: {path}")

    # ----- Run tag / joint name -----
    # First line: "Run: single_00_Hip"
    run_line = lines[0]
    if "Run:" not in run_line:
        raise ValueError(f"Unexpected first line in {path}: {run_line}")
    run_tag = run_line.split("Run:", 1)[1].strip()

    # Expect run_tag like "single_00_Hip"
    parts = run_tag.split("_")
    joint_name = parts[2] if len(parts) >= 3 else "UNKNOWN"

    # ----- Best / final val accuracy -----
    best_val_acc = None
    final_val_acc = None
    for ln in lines:
        if "Best ValAcc" in ln:
            try:
                best_val_acc = float(ln.split(":")[-1].strip())
            except ValueError:
                pass
        if "Final ValAcc" in ln:
            try:
                final_val_acc = float(ln.split(":")[-1].strip())
            except ValueError:
                pass

    # ----- Find header row with "precision  recall" -----
    header_idx = None
    for i, ln in enumerate(lines):
        if "precision" in ln and "recall" in ln and "f1-score" in ln:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Could not find classification_report header in {path}")

    # ----- Parse data rows after header -----
    data_rows = []
    for ln in lines[header_idx + 1 :]:
        # skip blank lines
        if not ln.strip():
            continue

        m = row_pattern.match(ln)
        if not m:
            # will skip "accuracy" line (doesn't have 3 metrics),
            # but will parse "macro avg" and "weighted avg" rows.
            continue

        label = m.group(1).strip()
        precision = float(m.group(2))
        recall = float(m.group(3))
        f1 = float(m.group(4))
        support = int(m.group(5))

        data_rows.append(
            {
                "run_tag": run_tag,
                "joint": joint_name,
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    if not data_rows:
        print(f"[WARN] No data rows parsed from {path}")

    run_info = {
        "path": path,
        "run_tag": run_tag,
        "joint": joint_name,
        "best_val_acc": best_val_acc,
        "final_val_acc": final_val_acc,
    }
    return run_info, data_rows


# -------------------------------------------------------------------
# LOAD ALL REPORTS
# -------------------------------------------------------------------
def load_all_reports():
    report_files = sorted(glob.glob(REPORT_GLOB))
    if not report_files:
        raise FileNotFoundError(f"No report files found matching {REPORT_GLOB}")

    all_run_info = []
    all_rows = []

    for path in report_files:
        print(f"Parsing {path}...")
        run_info, rows = parse_single_report(path)
        all_run_info.append(run_info)
        all_rows.extend(rows)

    df_runs = pd.DataFrame(all_run_info)
    df_rows = pd.DataFrame(all_rows)

    print("df_runs columns:", df_runs.columns.tolist())
    print("df_rows columns:", df_rows.columns.tolist())
    print("Number of per-label rows parsed:", len(df_rows))

    return df_runs, df_rows


# -------------------------------------------------------------------
# SUMMARIES
# -------------------------------------------------------------------
def make_run_summary(df_runs: pd.DataFrame, df_rows: pd.DataFrame):
    """
    Per-joint summary:
      - joint
      - best_val_acc
      - final_val_acc
      - macro_f1 (from "macro avg" row), if available
    """
    if df_rows.empty or "label" not in df_rows.columns:
        print("[WARN] df_rows is empty or has no 'label'; macro_f1 will be NaN.")
        df = df_runs.copy()
        df["macro_f1"] = np.nan
    else:
        macro_rows = df_rows[df_rows["label"] == "macro avg"].copy()
        macro_rows = macro_rows[["run_tag", "joint", "f1"]].rename(
            columns={"f1": "macro_f1"}
        )
        df = df_runs.merge(macro_rows, on=["run_tag", "joint"], how="left")

    out_csv = os.path.join(OUTPUT_DIR, "single_joint_run_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved run summary → {out_csv}")
    return df


def make_per_class_best_joint_table(df_rows: pd.DataFrame):
    """
    For each MoVi class, find which joint gives highest F1.
    Ignore summary rows like accuracy, macro avg, weighted avg.
    """
    if df_rows.empty or "label" not in df_rows.columns:
        print("[WARN] df_rows empty or no 'label'; skipping per-class best joint.")
        return pd.DataFrame()

    ignore_labels = {"accuracy", "macro avg", "weighted avg"}
    df = df_rows[~df_rows["label"].isin(ignore_labels)].copy()

    if df.empty:
        print("[WARN] No per-class rows found; skipping per-class best joint.")
        return pd.DataFrame()

    # For each label, pick the row (joint) with max F1
    idx = df.groupby("label")["f1"].idxmax()
    best_by_class = df.loc[idx, ["label", "joint", "f1", "recall", "precision"]]
    best_by_class = best_by_class.sort_values("label").reset_index(drop=True)

    out_csv = os.path.join(OUTPUT_DIR, "per_class_best_joint.csv")
    best_by_class.to_csv(out_csv, index=False)
    print(f"Saved per-class best joint table → {out_csv}")
    return best_by_class


# -------------------------------------------------------------------
# PLOTS
# -------------------------------------------------------------------
def plot_joint_overall_performance(df_runs: pd.DataFrame):
    """
    Bar plot of joint vs final_val_acc (+ optional macro_f1)
    """
    if df_runs.empty:
        print("[WARN] df_runs empty; skipping joint performance plots.")
        return

    df = df_runs.sort_values("final_val_acc", ascending=False)

    # Accuracy bar plot
    plt.figure(figsize=(10, 5))
    xs = np.arange(len(df))
    plt.bar(xs, df["final_val_acc"])
    plt.xticks(xs, df["joint"], rotation=45, ha="right")
    plt.ylabel("Final Val Accuracy")
    plt.title("Single-Joint Overall Performance (Final Val Acc)")
    plt.tight_layout()

    out_png = os.path.join(OUTPUT_DIR, "single_joint_final_val_acc_bar.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved joint final-acc bar plot → {out_png}")

    # Macro F1 bar plot (if available)
    if "macro_f1" in df.columns and df["macro_f1"].notna().any():
        plt.figure(figsize=(10, 5))
        plt.bar(xs, df["macro_f1"])
        plt.xticks(xs, df["joint"], rotation=45, ha="right")
        plt.ylabel("Macro F1")
        plt.title("Single-Joint Overall Performance (Macro F1)")
        plt.tight_layout()

        out_png = os.path.join(OUTPUT_DIR, "single_joint_macro_f1_bar.png")
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"Saved joint macro-F1 bar plot → {out_png}")


def plot_class_joint_heatmap(df_rows: pd.DataFrame):
    """
    Heatmap: MoVi class (rows) × joint (columns) of F1 scores.
    """
    if df_rows.empty or "label" not in df_rows.columns:
        print("[WARN] df_rows empty or no 'label'; skipping heatmap.")
        return

    ignore_labels = {"accuracy", "macro avg", "weighted avg"}
    df = df_rows[~df_rows["label"].isin(ignore_labels)].copy()
    if df.empty:
        print("[WARN] No per-class rows; skipping heatmap.")
        return

    # Average F1 across runs if needed
    df_mean = (
        df.groupby(["label", "joint"])["f1"]
        .mean()
        .reset_index()
    )
    pivot = df_mean.pivot(index="label", columns="joint", values="f1")

    labels = list(pivot.index)
    joints = list(pivot.columns)
    data = pivot.fillna(0.0).values

    plt.figure(
        figsize=(max(8, len(joints) * 0.5), max(6, len(labels) * 0.3))
    )
    im = plt.imshow(data, aspect="auto")
    plt.colorbar(im, label="F1 Score")

    plt.xticks(np.arange(len(joints)), joints, rotation=45, ha="right")
    plt.yticks(np.arange(len(labels)), labels)
    plt.xlabel("Joint (IMU location)")
    plt.ylabel("MoVi Class")
    plt.title("Per-Class F1 by Joint (Single-Joint Models)")
    plt.tight_layout()

    out_png = os.path.join(OUTPUT_DIR, "class_by_joint_f1_heatmap.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved class×joint F1 heatmap → {out_png}")


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    df_runs, df_rows = load_all_reports()

    # 1) Per-joint summary
    df_summary = make_run_summary(df_runs, df_rows)

    # 2) Per-class best joint
    best_by_class = make_per_class_best_joint_table(df_rows)

    # 3) Plots
    plot_joint_overall_performance(df_summary)
    plot_class_joint_heatmap(df_rows)

    print("\nDone. Outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
