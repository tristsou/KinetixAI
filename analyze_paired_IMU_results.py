import os
import glob
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
LOG_DIR_PAIR = "logs_pair_joint"  # folder with pair_XX_JointA__YY_JointB_classification_report.txt
OUTPUT_DIR = "analysis_pair_joint"
os.makedirs(OUTPUT_DIR, exist_ok=True)

REPORT_GLOB = os.path.join(LOG_DIR_PAIR, "pair_*_classification_report.txt")

# Regex for data rows in classification_report
row_pattern = re.compile(
    r"^\s*(\S.+?)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s*$"
)


# -------------------------------------------------------------------
# PARSER FOR A SINGLE PAIR REPORT
# -------------------------------------------------------------------
def parse_pair_run_tag(run_tag: str):
    """
    Expected format from Model_4_sweep:
      pair_{i_idx:02d}_{name_i}__{j_idx:02d}_{name_j}

    Example:
      pair_00_Hip__01_RightUpLeg
    """
    if not run_tag.startswith("pair_"):
        return "UNKNOWN", "UNKNOWN", "UNKNOWN"

    body = run_tag[len("pair_") :]  # e.g. "00_Hip__01_RightUpLeg"
    parts = body.split("__")
    if len(parts) != 2:
        return "UNKNOWN", "UNKNOWN", "UNKNOWN"

    left, right = parts
    # left: "00_Hip", right: "01_RightUpLeg"
    def parse_side(s):
        p = s.split("_", 1)
        if len(p) == 2:
            idx_str, name = p
            return name
        else:
            return s

    joint_a = parse_side(left)
    joint_b = parse_side(right)
    pair_name = f"{joint_a} + {joint_b}"
    return joint_a, joint_b, pair_name


def parse_pair_report(path: str):
    """
    Parse one classification report file like:
      pair_00_Hip__01_RightUpLeg_classification_report.txt

    Returns:
        run_info: dict with run_tag, joint_a, joint_b, pair_name,
                  best_val_acc, final_val_acc, path
        rows: list of dicts with columns:
              run_tag, joint_a, joint_b, pair_name,
              label, precision, recall, f1, support
    """
    with open(path, "r") as f:
        lines = [ln.rstrip("\n") for ln in f]

    if not lines:
        raise ValueError(f"Empty file: {path}")

    # ----- Run tag / pair name -----
    # First line: "Run: pair_00_Hip__01_RightUpLeg"
    run_line = lines[0]
    if "Run:" not in run_line:
        raise ValueError(f"Unexpected first line in {path}: {run_line}")
    run_tag = run_line.split("Run:", 1)[1].strip()

    joint_a, joint_b, pair_name = parse_pair_run_tag(run_tag)

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
        if not ln.strip():
            continue

        m = row_pattern.match(ln)
        if not m:
            # skip lines like "accuracy" that don't fit the pattern
            continue

        label = m.group(1).strip()
        precision = float(m.group(2))
        recall = float(m.group(3))
        f1 = float(m.group(4))
        support = int(m.group(5))

        data_rows.append(
            {
                "run_tag": run_tag,
                "joint_a": joint_a,
                "joint_b": joint_b,
                "pair_name": pair_name,
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
        "joint_a": joint_a,
        "joint_b": joint_b,
        "pair_name": pair_name,
        "best_val_acc": best_val_acc,
        "final_val_acc": final_val_acc,
    }
    return run_info, data_rows


# -------------------------------------------------------------------
# LOAD ALL REPORTS
# -------------------------------------------------------------------
def load_all_pair_reports():
    report_files = sorted(glob.glob(REPORT_GLOB))
    if not report_files:
        raise FileNotFoundError(f"No pair report files found matching {REPORT_GLOB}")

    all_run_info = []
    all_rows = []

    for path in report_files:
        print(f"Parsing {path}...")
        run_info, rows = parse_pair_report(path)
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
def make_pair_run_summary(df_runs: pd.DataFrame, df_rows: pd.DataFrame):
    """
    Per-pair summary:
      - pair_name
      - joint_a
      - joint_b
      - best_val_acc
      - final_val_acc
      - macro_f1 (from "macro avg"), if available
    """
    if df_rows.empty or "label" not in df_rows.columns:
        print("[WARN] df_rows is empty or has no 'label'; macro_f1 will be NaN.")
        df = df_runs.copy()
        df["macro_f1"] = np.nan
    else:
        macro_rows = df_rows[df_rows["label"] == "macro avg"].copy()
        macro_rows = macro_rows[["run_tag", "pair_name", "f1"]].rename(
            columns={"f1": "macro_f1"}
        )
        df = df_runs.merge(macro_rows, on=["run_tag", "pair_name"], how="left")

    out_csv = os.path.join(OUTPUT_DIR, "pair_run_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved pair run summary → {out_csv}")
    return df


def make_per_class_best_pair_table(df_rows: pd.DataFrame):
    """
    For each MoVi class, find which pair of joints gives highest F1.
    Ignore summary rows like accuracy, macro avg, weighted avg.
    """
    if df_rows.empty or "label" not in df_rows.columns:
        print("[WARN] df_rows empty or no 'label'; skipping per-class best pair.")
        return pd.DataFrame()

    ignore_labels = {"accuracy", "macro avg", "weighted avg"}
    df = df_rows[~df_rows["label"].isin(ignore_labels)].copy()

    if df.empty:
        print("[WARN] No per-class rows found; skipping per-class best pair.")
        return pd.DataFrame()

    # For each label, pick the row (pair) with max F1
    idx = df.groupby("label")["f1"].idxmax()
    best_by_class = df.loc[
        idx, ["label", "pair_name", "joint_a", "joint_b", "f1", "recall", "precision"]
    ]
    best_by_class = best_by_class.sort_values("label").reset_index(drop=True)

    out_csv = os.path.join(OUTPUT_DIR, "per_class_best_pair.csv")
    best_by_class.to_csv(out_csv, index=False)
    print(f"Saved per-class best pair table → {out_csv}")
    return best_by_class


# -------------------------------------------------------------------
# PLOTS
# -------------------------------------------------------------------
def plot_pair_overall_performance(df_runs: pd.DataFrame):
    """
    Bar plot of pair_name vs final_val_acc (+ optional macro_f1).
    """
    if df_runs.empty:
        print("[WARN] df_runs empty; skipping pair performance plots.")
        return

    df = df_runs.sort_values("final_val_acc", ascending=False)

    # Accuracy bar plot (may be wide; pairs are many)
    plt.figure(figsize=(max(10, len(df) * 0.4), 6))
    xs = np.arange(len(df))
    plt.bar(xs, df["final_val_acc"])
    plt.xticks(xs, df["pair_name"], rotation=90)
    plt.ylabel("Final Val Accuracy")
    plt.title("Pair-Joint Overall Performance (Final Val Acc)")
    plt.tight_layout()

    out_png = os.path.join(OUTPUT_DIR, "pair_joint_final_val_acc_bar.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved pair final-acc bar plot → {out_png}")

    # Macro F1 bar plot (if available)
    if "macro_f1" in df.columns and df["macro_f1"].notna().any():
        plt.figure(figsize=(max(10, len(df) * 0.4), 6))
        plt.bar(xs, df["macro_f1"])
        plt.xticks(xs, df["pair_name"], rotation=90)
        plt.ylabel("Macro F1")
        plt.title("Pair-Joint Overall Performance (Macro F1)")
        plt.tight_layout()

        out_png = os.path.join(OUTPUT_DIR, "pair_joint_macro_f1_bar.png")
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"Saved pair macro-F1 bar plot → {out_png}")


def plot_class_pair_heatmap(df_rows: pd.DataFrame):
    """
    Heatmap: MoVi class (rows) × pair_name (columns) of F1 scores.
    This can get wide (66 pairs), but is very informative.
    """
    if df_rows.empty or "label" not in df_rows.columns:
        print("[WARN] df_rows empty or no 'label'; skipping pair heatmap.")
        return

    ignore_labels = {"accuracy", "macro avg", "weighted avg"}
    df = df_rows[~df_rows["label"].isin(ignore_labels)].copy()
    if df.empty:
        print("[WARN] No per-class rows; skipping pair heatmap.")
        return

    # Average F1 across runs if needed (should be 1 per run_tag, but safe)
    df_mean = (
        df.groupby(["label", "pair_name"])["f1"]
        .mean()
        .reset_index()
    )
    pivot = df_mean.pivot(index="label", columns="pair_name", values="f1")

    labels = list(pivot.index)
    pairs = list(pivot.columns)
    data = pivot.fillna(0.0).values

    plt.figure(
        figsize=(max(10, len(pairs) * 0.4), max(6, len(labels) * 0.3))
    )
    im = plt.imshow(data, aspect="auto")
    plt.colorbar(im, label="F1 Score")

    plt.xticks(np.arange(len(pairs)), pairs, rotation=90)
    plt.yticks(np.arange(len(labels)), labels)
    plt.xlabel("IMU Pair (Joint A + Joint B)")
    plt.ylabel("MoVi Class")
    plt.title("Per-Class F1 by IMU Pair (Pair-Joint Models)")
    plt.tight_layout()

    out_png = os.path.join(OUTPUT_DIR, "class_by_pair_f1_heatmap.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved class×pair F1 heatmap → {out_png}")


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    df_runs, df_rows = load_all_pair_reports()

    # 1) Per-pair summary
    df_summary = make_pair_run_summary(df_runs, df_rows)

    # 2) Per-class best pair
    best_by_class = make_per_class_best_pair_table(df_rows)

    # 3) Plots
    plot_pair_overall_performance(df_summary)
    plot_class_pair_heatmap(df_rows)

    print("\nDone. Outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
