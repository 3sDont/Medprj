# -*- coding: utf-8 -*-
"""
simulate_l2r.py — Visual simulation of Learning-to-Rank training on MedPRS data.

Shows step-by-step how LambdaMART trains and how it changes ranking results.

Run:
    python simulate_l2r.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — saves PNG instead of opening a window
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import GradientBoostingRegressor
import warnings, os

warnings.filterwarnings("ignore")
OUTPUT_DIR = "l2r_simulation_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load(path):
    df = pd.read_csv(path)
    # Normalize column names (inference.py format)
    rename = {"True_Label": "True_Journal_ID", "Rank": "Base_Rank", "Score": "Base_Score"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["Aims_Scope_Sim"] = df.get("Aims_Scope_Sim", pd.Series(0.0, index=df.index))
    return df.sort_values(["Paper_ID", "Base_Rank"]).reset_index(drop=True)

print("=" * 60)
print("  L2R SIMULATION — MedPRS Journal Recommendation")
print("=" * 60)

print("\n[1] Loading data ...")
df_train = load("data/pred_train.csv")
df_val   = load("data/pred_val.csv")
df_test  = load("data/pred_test.csv")

n_train = df_train["Paper_ID"].nunique()
n_val   = df_val["Paper_ID"].nunique()
n_test  = df_test["Paper_ID"].nunique()
n_cands = df_train.groupby("Paper_ID").size().iloc[0]

print(f"    Train: {n_train} papers × {n_cands} candidates = {len(df_train)} rows")
print(f"    Val  : {n_val} papers × {n_cands} candidates = {len(df_val)} rows")
print(f"    Test : {n_test} papers × {n_cands} candidates = {len(df_test)} rows")

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df):
    X = np.column_stack([
        df["Base_Score"].values.astype(np.float32),
        df["Base_Rank"].values.astype(np.float32),
        df["Aims_Scope_Sim"].values.astype(np.float32),
    ])
    y = df["Is_Correct"].values.astype(np.float32)
    return X, y

X_train, y_train = build_features(df_train)
X_val,   y_val   = build_features(df_val)
X_test,  y_test  = build_features(df_test)

print(f"\n[2] Feature matrix built")
print(f"    Features: [base_score, base_rank, aims_scope_sim]")
print(f"    X_train shape: {X_train.shape}  — {y_train.sum():.0f} positive labels out of {len(y_train)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. BASELINE METRICS (before L2R)
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(df, score_col, ks=(1, 3, 5, 10)):
    df = df.copy()
    df["_rank"] = df.groupby("Paper_ID")[score_col].rank(method="first", ascending=False).astype(int)
    mrr = 0.0
    ndcg = {k: 0.0 for k in ks}
    acc  = {k: 0   for k in ks}
    n = 0
    for _, grp in df.groupby("Paper_ID"):
        rel = grp.sort_values("_rank")["Is_Correct"].values
        pos = np.where(rel == 1)[0]
        if len(pos) > 0:
            mrr += 1.0 / (pos[0] + 1)
        for k in ks:
            if rel[:k].sum() > 0:
                acc[k] += 1
            # NDCG (ideal=1 since one correct)
            r = rel[:k]
            ndcg[k] += float((r / np.log2(np.arange(2, len(r) + 2))).sum())
        n += 1
    results = {"MRR": mrr / n}
    for k in ks:
        results[f"NDCG@{k}"] = ndcg[k] / n
        results[f"Acc@{k}"]  = acc[k]  / n
    return results

print("\n[3] Baseline metrics (using Base_Score, no L2R) ...")
baseline_val  = compute_metrics(df_val,  "Base_Score")
baseline_test = compute_metrics(df_test, "Base_Score")

print(f"\n    VAL  — MRR: {baseline_val['MRR']:.4f} | "
      f"NDCG@5: {baseline_val['NDCG@5']:.4f} | Acc@1: {baseline_val['Acc@1']:.4f} | Acc@5: {baseline_val['Acc@5']:.4f}")
print(f"    TEST — MRR: {baseline_test['MRR']:.4f} | "
      f"NDCG@5: {baseline_test['NDCG@5']:.4f} | Acc@1: {baseline_test['Acc@1']:.4f} | Acc@5: {baseline_test['Acc@5']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. SIMULATE TRAINING — step through n_estimators, record val metric each round
# ─────────────────────────────────────────────────────────────────────────────

print("\n[4] Simulating L2R training (GradientBoosting proxy for LambdaMART) ...")
print("    Round   Val_MRR   Val_Acc@1   Val_NDCG@5")
print("    " + "-" * 42)

rounds      = [1, 5, 10, 20, 30, 50, 75, 100, 150, 200]
val_mrr     = []
val_acc1    = []
val_ndcg5   = []
train_mrr   = []

for n_est in rounds:
    model = GradientBoostingRegressor(
        n_estimators=n_est,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Score candidates
    df_val_copy = df_val.copy()
    df_val_copy["L2R_Score"] = model.predict(X_val)
    df_train_copy = df_train.copy()
    df_train_copy["L2R_Score"] = model.predict(X_train)

    m_val   = compute_metrics(df_val_copy,   "L2R_Score")
    m_train = compute_metrics(df_train_copy, "L2R_Score")

    val_mrr.append(m_val["MRR"])
    val_acc1.append(m_val["Acc@1"])
    val_ndcg5.append(m_val["NDCG@5"])
    train_mrr.append(m_train["MRR"])

    print(f"    {n_est:>5}   {m_val['MRR']:.4f}    {m_val['Acc@1']:.4f}      {m_val['NDCG@5']:.4f}")

# Final model (200 rounds)
model_final = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
model_final.fit(X_train, y_train)

df_test_copy  = df_test.copy()
df_test_copy["L2R_Score"] = model_final.predict(X_test)
final_test = compute_metrics(df_test_copy, "L2R_Score")

print(f"\n    FINAL TEST — MRR: {final_test['MRR']:.4f} | "
      f"NDCG@5: {final_test['NDCG@5']:.4f} | Acc@1: {final_test['Acc@1']:.4f} | Acc@5: {final_test['Acc@5']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. SHOW CONCRETE RANKING CHANGES (before vs after) on 5 papers
# ─────────────────────────────────────────────────────────────────────────────

print("\n[5] Concrete ranking changes on 5 sample papers ...")
print()

sample_ids = df_test["Paper_ID"].unique()[:5]
df_test_copy["L2R_Rank"] = (
    df_test_copy.groupby("Paper_ID")["L2R_Score"]
    .rank(method="first", ascending=False).astype(int)
)

for pid in sample_ids:
    grp = df_test_copy[df_test_copy["Paper_ID"] == pid].sort_values("Base_Rank")
    true_label = grp["True_Journal_ID"].iloc[0]
    correct_row = grp[grp["Is_Correct"] == 1].iloc[0]
    base_rank = int(correct_row["Base_Rank"])
    l2r_rank  = int(correct_row["L2R_Rank"])
    arrow = "✓ same" if base_rank == l2r_rank else (f"↑ {base_rank}→{l2r_rank}" if l2r_rank < base_rank else f"↓ {base_rank}→{l2r_rank}")
    print(f"    Paper {pid:>4}  |  True journal: {true_label:<6}  |  "
          f"Base rank: {base_rank}  →  L2R rank: {l2r_rank}  |  {arrow}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. PLOT
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[6] Generating plots ...")

fig = plt.figure(figsize=(16, 10))
fig.suptitle("L2R Training Simulation — MedPRS Journal Recommendation", fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Plot 1: MRR vs training rounds ────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(rounds, train_mrr, "o--", color="steelblue", label="Train MRR", linewidth=1.5)
ax1.plot(rounds, val_mrr,   "s-",  color="darkorange", label="Val MRR",   linewidth=2)
ax1.axhline(baseline_val["MRR"], linestyle=":", color="gray", label="Baseline (no L2R)")
ax1.set_title("MRR over Training Rounds")
ax1.set_xlabel("Boosting Rounds")
ax1.set_ylabel("MRR")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# ── Plot 2: Acc@1 vs rounds ───────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(rounds, val_acc1, "s-", color="seagreen", linewidth=2, label="Val Acc@1")
ax2.axhline(baseline_val["Acc@1"], linestyle=":", color="gray", label="Baseline Acc@1")
ax2.set_title("Accuracy@1 over Training Rounds")
ax2.set_xlabel("Boosting Rounds")
ax2.set_ylabel("Acc@1")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# ── Plot 3: NDCG@5 vs rounds ──────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(rounds, val_ndcg5, "^-", color="mediumpurple", linewidth=2, label="Val NDCG@5")
ax3.axhline(baseline_val["NDCG@5"], linestyle=":", color="gray", label="Baseline NDCG@5")
ax3.set_title("NDCG@5 over Training Rounds")
ax3.set_xlabel("Boosting Rounds")
ax3.set_ylabel("NDCG@5")
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# ── Plot 4: Before vs After metric comparison (bar chart) ─────────────────
ax4 = fig.add_subplot(gs[1, 0:2])
metrics_keys = ["MRR", "Acc@1", "Acc@3", "Acc@5", "Acc@10", "NDCG@1", "NDCG@3", "NDCG@5", "NDCG@10"]
before = [baseline_test.get(k, 0) for k in metrics_keys]
after  = [final_test.get(k, 0)   for k in metrics_keys]
x = np.arange(len(metrics_keys))
w = 0.35
bars1 = ax4.bar(x - w/2, before, w, label="Baseline (Base Score)", color="steelblue", alpha=0.8)
bars2 = ax4.bar(x + w/2, after,  w, label="After L2R",             color="darkorange", alpha=0.8)
ax4.set_title("Baseline vs L2R — Test Set Metrics")
ax4.set_xticks(x)
ax4.set_xticklabels(metrics_keys, rotation=30, ha="right", fontsize=8)
ax4.set_ylabel("Score")
ax4.legend()
ax4.grid(axis="y", alpha=0.3)
for bar in bars1:
    h = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.3f}", ha="center", va="bottom", fontsize=6)
for bar in bars2:
    h = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.3f}", ha="center", va="bottom", fontsize=6)

# ── Plot 5: Score distribution for correct vs incorrect candidates ─────────
ax5 = fig.add_subplot(gs[1, 2])
scores_correct   = df_train[df_train["Is_Correct"] == 1]["Base_Score"].values
scores_incorrect = df_train[df_train["Is_Correct"] == 0]["Base_Score"].values
ax5.hist(np.log1p(scores_incorrect), bins=40, alpha=0.6, color="steelblue", label="Wrong candidate", density=True)
ax5.hist(np.log1p(scores_correct),   bins=40, alpha=0.8, color="darkorange", label="Correct journal",  density=True)
ax5.set_title("Score Distribution\n(log scale, train set)")
ax5.set_xlabel("log(1 + Base_Score)")
ax5.set_ylabel("Density")
ax5.legend(fontsize=8)
ax5.grid(alpha=0.3)

plt.savefig(os.path.join(OUTPUT_DIR, "l2r_simulation.png"), dpi=150, bbox_inches="tight")
print(f"    Saved → {OUTPUT_DIR}/l2r_simulation.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  SUMMARY — Baseline vs L2R (Test Set)")
print("=" * 60)
print(f"  {'Metric':<12}  {'Baseline':>10}  {'L2R':>10}  {'Delta':>10}")
print("  " + "-" * 46)
for k in ["MRR", "Acc@1", "Acc@3", "Acc@5", "NDCG@5", "NDCG@10"]:
    b = baseline_test.get(k, 0)
    a = final_test.get(k, 0)
    d = a - b
    sign = "+" if d >= 0 else ""
    print(f"  {k:<12}  {b:>10.4f}  {a:>10.4f}  {sign}{d:>9.4f}")

print("\nDone. Output in:", OUTPUT_DIR)
