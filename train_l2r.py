# -*- coding: utf-8 -*-
"""
train_l2r.py — Learning-to-Rank training for MedPRS journal recommendation.

Pipeline:
    1. Load base model predictions (inference.py / main_simcprs_v2.py output)
    2. Merge optional LLM similarity features (domain_sim, research_focus_sim)
    3. Build feature vectors X(P, Ji) per (paper, candidate-journal) pair
    4. Train a LambdaMART (LightGBM) or XGBoost ranker
    5. Evaluate with MRR and NDCG@k on val/test splits
    6. Save model + ranked predictions

Feature vector X(P, Ji):
    [base_score, base_rank, aims_scope_sim, domain_sim*, research_focus_sim*]
    (* optional — zero-filled until the LLM pipeline is ready)

Accepted input CSV formats
--------------------------
From main_simcprs_v2.py (test_detailed_predictions.csv):
    Paper_ID, True_Journal_ID, Predicted_Journal_ID,
    Base_Rank, Base_Score, Aims_Scope_Sim, Is_Correct

From inference.py (predictions.csv):
    Paper_ID, True_Label, Rank, Predicted_Journal_ID, Score, Is_Correct
    (Aims_Scope_Sim column optional — zero-filled if absent)

Optional supplementary CSVs (added later):
    domain_similarity CSV      ->Paper_ID, Predicted_Journal_ID, domain_similarity
    research_focus_sim CSV     ->Paper_ID, Predicted_Journal_ID, research_focus_similarity

Usage examples
--------------
# Basic (no LLM features yet):
python train_l2r.py \\
    --train_csv predictions_train.csv \\
    --val_csv   predictions_val.csv   \\
    --test_csv  predictions_test.csv  \\
    --output_dir l2r_output

# With LLM features:
python train_l2r.py \\
    --train_csv predictions_train.csv \\
    --val_csv   predictions_val.csv   \\
    --test_csv  predictions_test.csv  \\
    --domain_sim_csv      domain_sim.csv \\
    --research_focus_sim_csv research_focus_sim.csv \\
    --output_dir l2r_output

# Re-rank new predictions with a saved model:
python train_l2r.py \\
    --mode predict \\
    --test_csv  new_predictions.csv \\
    --model_path l2r_output/l2r_lightgbm_model.pkl \\
    --output_dir l2r_output
"""

import argparse
import os
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Feature column order must stay fixed (model is trained on this order)
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "base_score",
    "base_rank",
    "aims_scope_sim",
    "domain_sim",
    "research_focus_sim",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename inference.py column names to the canonical names used here."""
    rename = {
        "True_Label":    "True_Journal_ID",
        "Rank":          "Base_Rank",
        "Score":         "Base_Score",
    }
    df = df.rename(columns={k: v for k, v in rename.items()
                             if k in df.columns and v not in df.columns})
    required = ["Paper_ID", "True_Journal_ID", "Predicted_Journal_ID",
                "Base_Rank", "Base_Score", "Is_Correct"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
    if "Aims_Scope_Sim" not in df.columns:
        df["Aims_Scope_Sim"] = 0.0
    return df


def load_predictions(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = _normalize_columns(df)
    # Sort so rows are grouped by paper (required for LightGBM/XGBoost group param)
    df = df.sort_values(["Paper_ID", "Base_Rank"]).reset_index(drop=True)
    return df


def merge_optional_feature(df: pd.DataFrame,
                            feature_csv: str | None,
                            feature_col: str,
                            default: float = 0.0) -> pd.DataFrame:
    """Left-join an optional per-(paper, journal) similarity CSV into df."""
    if feature_col in df.columns:
        df[feature_col] = df[feature_col].fillna(default)
        print(f"  '{feature_col}' already present in CSV (kept as-is)")
        return df
    if feature_csv and os.path.exists(feature_csv):
        feat_df = pd.read_csv(feature_csv, usecols=["Paper_ID", "Predicted_Journal_ID", feature_col])
        df = df.merge(feat_df, on=["Paper_ID", "Predicted_Journal_ID"], how="left")
        df[feature_col] = df[feature_col].fillna(default)
        print(f"  Merged '{feature_col}' from {feature_csv}")
    else:
        df[feature_col] = default
        if feature_csv:
            print(f"  WARNING: '{feature_csv}' not found - '{feature_col}' set to {default}")
        else:
            print(f"  '{feature_col}' not provided - set to {default}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature matrix construction
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(df: pd.DataFrame):
    """
    Returns
    -------
    X      : np.ndarray  (N, 5)   feature matrix
    y      : np.ndarray  (N,)     binary relevance labels (1 = correct journal)
    groups : np.ndarray  (Q,)     number of candidates per paper (for LTR models)
    """
    feat = pd.DataFrame({
        "base_score":         df["Base_Score"].values,
        "base_rank":          df["Base_Rank"].values,
        "aims_scope_sim":     df["Aims_Scope_Sim"].values,
        "domain_sim":         df.get("domain_similarity", pd.Series(0.0, index=df.index)).values,
        "research_focus_sim": df.get("research_focus_similarity", pd.Series(0.0, index=df.index)).values,
    })
    X      = feat[FEATURE_COLS].values.astype(np.float32)
    y      = df["Is_Correct"].values.astype(np.int32)
    groups = df.groupby("Paper_ID", sort=False).size().values
    return X, y, groups


# ─────────────────────────────────────────────────────────────────────────────
# Model training
# ─────────────────────────────────────────────────────────────────────────────

def train_lightgbm(X_train, y_train, groups_train,
                   X_val,   y_val,   groups_val,
                   args):
    import lightgbm as lgb

    params = {
        "objective":       "lambdarank",
        "metric":          "ndcg",
        "ndcg_eval_at":    [1, 3, 5, 10],
        "learning_rate":   args.lr,
        "num_leaves":      args.num_leaves,
        "min_child_samples": args.min_child_samples,
        "n_estimators":    args.n_estimators,
        "random_state":    42,
        "verbose":         -1,
    }
    train_data = lgb.Dataset(X_train, label=y_train, group=groups_train,
                             feature_name=FEATURE_COLS)
    val_data   = lgb.Dataset(X_val,   label=y_val,   group=groups_val,
                             reference=train_data)
    callbacks = [
        lgb.early_stopping(stopping_rounds=args.early_stopping, verbose=True),
        lgb.log_evaluation(period=20),
    ]
    model = lgb.train(params, train_data,
                      num_boost_round=args.n_estimators,
                      valid_sets=[val_data],
                      callbacks=callbacks)
    return model


def train_xgboost(X_train, y_train, groups_train,
                  X_val,   y_val,   groups_val,
                  args):
    import xgboost as xgb

    model = xgb.XGBRanker(
        objective="rank:ndcg",
        learning_rate=args.lr,
        n_estimators=args.n_estimators,
        max_depth=6,
        random_state=42,
        eval_metric="ndcg@10",
        early_stopping_rounds=args.early_stopping,
        verbosity=1,
    )
    model.fit(
        X_train, y_train, qid=_groups_to_qid(groups_train),
        eval_set=[(X_val, y_val)],
        eval_qid=[_groups_to_qid(groups_val)],
        verbose=20,
    )
    return model


def _groups_to_qid(groups: np.ndarray) -> np.ndarray:
    """Convert group sizes array to per-sample query-id array (XGBoost format)."""
    qid = np.repeat(np.arange(len(groups)), groups)
    return qid


def predict_scores(model, X: np.ndarray, model_type: str) -> np.ndarray:
    return model.predict(X)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────

def _dcg_at_k(sorted_relevances: np.ndarray, k: int) -> float:
    rel = sorted_relevances[:k]
    gains = rel / np.log2(np.arange(2, len(rel) + 2))
    return float(gains.sum())


def evaluate(df: pd.DataFrame, score_col: str, ks=(1, 3, 5, 10)):
    """
    Rank candidates by score_col within each paper, then compute MRR and NDCG@k.

    Returns (metrics_dict, df_with_l2r_rank)
    """
    df = df.copy()
    df["L2R_Rank"] = (
        df.groupby("Paper_ID")[score_col]
          .rank(method="first", ascending=False)
          .astype(int)
    )

    mrr = 0.0
    ndcg = {k: 0.0 for k in ks}
    acc  = {k: 0   for k in ks}
    n_papers = 0

    for _, grp in df.groupby("Paper_ID"):
        grp_sorted = grp.sort_values("L2R_Rank")
        relevances = grp_sorted["Is_Correct"].values

        # MRR
        correct_positions = np.where(relevances == 1)[0]
        if len(correct_positions) > 0:
            mrr += 1.0 / (correct_positions[0] + 1)

        # NDCG@k (ideal DCG = 1.0 since there is exactly one correct journal)
        for k in ks:
            ndcg[k] += _dcg_at_k(relevances, k)          # ideal_dcg = 1.0

        # Accuracy@k
        for k in ks:
            if relevances[:k].sum() > 0:
                acc[k] += 1

        n_papers += 1

    if n_papers == 0:
        return {}, df

    results = {"MRR": mrr / n_papers}
    for k in ks:
        results[f"NDCG@{k}"] = ndcg[k] / n_papers
        results[f"Acc@{k}"]  = acc[k]  / n_papers

    return results, df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_argparser():
    p = argparse.ArgumentParser(
        description="Train a Learning-to-Rank model on top of MedPRS predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["train", "predict"], default="train",
                   help="'train': fit and evaluate; 'predict': re-rank with saved model")

    # ── input data ──────────────────────────────────────────────────────────
    p.add_argument("--train_csv", type=str, default=None,
                   help="Predictions CSV for the training split")
    p.add_argument("--val_csv", type=str, default=None,
                   help="Predictions CSV for the validation split")
    p.add_argument("--test_csv", type=str, required=True,
                   help="Predictions CSV for the test split (also used in predict mode)")

    # ── optional LLM-based similarity features ───────────────────────────────
    p.add_argument("--domain_sim_csv", type=str, default=None,
                   help="CSV with domain_similarity "
                        "(columns: Paper_ID, Predicted_Journal_ID, domain_similarity)")
    p.add_argument("--research_focus_sim_csv", type=str, default=None,
                   help="CSV with research_focus_similarity "
                        "(columns: Paper_ID, Predicted_Journal_ID, research_focus_similarity)")

    # ── model ────────────────────────────────────────────────────────────────
    p.add_argument("--model_type", choices=["lightgbm", "xgboost"], default="lightgbm",
                   help="L2R algorithm")
    p.add_argument("--lr", type=float, default=0.05,
                   help="Learning rate")
    p.add_argument("--n_estimators", type=int, default=500,
                   help="Maximum number of boosting rounds")
    p.add_argument("--num_leaves", type=int, default=31,
                   help="LightGBM: max number of leaves per tree")
    p.add_argument("--min_child_samples", type=int, default=5,
                   help="LightGBM: min samples per leaf")
    p.add_argument("--early_stopping", type=int, default=30,
                   help="Early-stopping patience (rounds without improvement)")

    # ── predict mode ─────────────────────────────────────────────────────────
    p.add_argument("--model_path", type=str, default=None,
                   help="Path to a saved .pkl model (required for --mode predict)")

    # ── output ───────────────────────────────────────────────────────────────
    p.add_argument("--output_dir", type=str, default="l2r_output",
                   help="Directory for saved model, predictions, and metrics")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10],
                   help="k values for Acc@k and NDCG@k")

    return p


def _load_and_enrich(csv_path: str, args) -> pd.DataFrame:
    """Load a predictions CSV and merge optional similarity features."""
    df = load_predictions(csv_path)
    df = merge_optional_feature(df, args.domain_sim_csv,
                                 "domain_similarity")
    df = merge_optional_feature(df, args.research_focus_sim_csv,
                                 "research_focus_similarity")
    return df


def _print_metrics(split_name: str, metrics: dict):
    print(f"\n{'-'*40}")
    print(f"  {split_name.upper()} METRICS")
    print(f"{'-'*40}")
    for name, val in metrics.items():
        print(f"  {name:<15} {val:.4f}")


def _save_metrics(metrics: dict, path: str, split_name: str):
    with open(path, "w") as f:
        f.write(f"=== {split_name.upper()} Metrics ===\n")
        for name, val in metrics.items():
            f.write(f"{name}: {val:.4f}\n")


def run_train(args):
    if not args.train_csv or not args.val_csv:
        raise ValueError("--train_csv and --val_csv are required in train mode.")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── load ──────────────────────────────────────────────────────────────────
    print("\n[1/4] Loading data ...")
    print(f"  train: {args.train_csv}")
    df_train = _load_and_enrich(args.train_csv, args)
    print(f"  val  : {args.val_csv}")
    df_val   = _load_and_enrich(args.val_csv,   args)
    print(f"  test : {args.test_csv}")
    df_test  = _load_and_enrich(args.test_csv,  args)

    X_train, y_train, groups_train = build_dataset(df_train)
    X_val,   y_val,   groups_val   = build_dataset(df_val)
    X_test,  y_test,  groups_test  = build_dataset(df_test)

    print(f"\n  Papers  - train: {len(groups_train)}, val: {len(groups_val)}, test: {len(groups_test)}")
    print(f"  Samples - train: {len(X_train)},  val: {len(X_val)},  test: {len(X_test)}")
    print(f"  Features: {FEATURE_COLS}")
    active = [c for c in FEATURE_COLS
              if not (c in ("domain_sim", "research_focus_sim")
                      and df_train[{"domain_sim": "domain_similarity",
                                    "research_focus_sim": "research_focus_similarity"}.get(c, c)].eq(0.0).all())]
    inactive = [c for c in FEATURE_COLS if c not in active]
    if inactive:
        print(f"  (zero-filled / inactive): {inactive}")

    # ── baseline (before L2R) ─────────────────────────────────────────────────
    print("\n[2/4] Baseline evaluation (base model ranking) ...")
    for split_name, df_split in [("val", df_val), ("test", df_test)]:
        metrics, _ = evaluate(df_split, score_col="Base_Score", ks=args.ks)
        _print_metrics(f"Baseline {split_name}", metrics)

    # ── train ─────────────────────────────────────────────────────────────────
    print(f"\n[3/4] Training {args.model_type} ranker ...")
    if args.model_type == "lightgbm":
        model = train_lightgbm(X_train, y_train, groups_train,
                               X_val,   y_val,   groups_val,   args)
    else:
        model = train_xgboost(X_train, y_train, groups_train,
                              X_val,   y_val,   groups_val,   args)

    model_path = os.path.join(args.output_dir, f"l2r_{args.model_type}_model.pkl")
    joblib.dump(model, model_path)
    print(f"\n  Model saved ->{model_path}")

    # feature importance (LightGBM only)
    if args.model_type == "lightgbm":
        imp = model.feature_importance(importance_type="gain")
        print("\n  Feature importance (gain):")
        for feat, score in sorted(zip(FEATURE_COLS, imp), key=lambda x: -x[1]):
            print(f"    {feat:<25} {score:.2f}")

    # ── evaluate ──────────────────────────────────────────────────────────────
    print("\n[4/4] Evaluating L2R model ...")
    for split_name, df_split, X_split in [
        ("val",  df_val,  X_val),
        ("test", df_test, X_test),
    ]:
        scores = predict_scores(model, X_split, args.model_type)
        df_split = df_split.copy()
        df_split["L2R_Score"] = scores

        metrics, df_ranked = evaluate(df_split, score_col="L2R_Score", ks=args.ks)
        _print_metrics(f"L2R {split_name}", metrics)

        out_csv = os.path.join(args.output_dir, f"{split_name}_l2r_predictions.csv")
        df_ranked.to_csv(out_csv, index=False)

        out_txt = os.path.join(args.output_dir, f"{split_name}_l2r_metrics.txt")
        _save_metrics(metrics, out_txt, f"L2R {split_name}")
        print(f"  Saved ->{out_csv}")
        print(f"  Saved ->{out_txt}")

    print("\nDone.")


def run_predict(args):
    if not args.model_path:
        raise ValueError("--model_path is required in predict mode.")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nLoading model from {args.model_path} ...")
    model = joblib.load(args.model_path)
    model_type = "lightgbm" if "lightgbm" in os.path.basename(args.model_path) else "xgboost"

    print(f"Loading predictions from {args.test_csv} ...")
    df = _load_and_enrich(args.test_csv, args)
    X, y, groups = build_dataset(df)

    scores = predict_scores(model, X, model_type)
    df["L2R_Score"] = scores

    metrics, df_ranked = evaluate(df, score_col="L2R_Score", ks=args.ks)
    _print_metrics("Predict", metrics)

    out_csv = os.path.join(args.output_dir, "predict_l2r_predictions.csv")
    df_ranked.to_csv(out_csv, index=False)
    print(f"\nRanked predictions saved ->{out_csv}")

    out_txt = os.path.join(args.output_dir, "predict_l2r_metrics.txt")
    _save_metrics(metrics, out_txt, "Predict")
    print(f"Metrics saved ->{out_txt}")


def main():
    parser = build_argparser()
    args = parser.parse_args()

    if args.mode == "train":
        run_train(args)
    else:
        run_predict(args)


if __name__ == "__main__":
    main()
