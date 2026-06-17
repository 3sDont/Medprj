# -*- coding: utf-8 -*-
"""
recranker/evaluate.py
Compare zero-shot LLM reranking against baseline and L2R on hard test cases.

Usage
-----
python -m recranker.evaluate \
    --results  data/recranker_data/test_results.csv \
    --hard_csv data/subset_hard/test.csv \
    --l2r_csv  l2r_output_hard/test_l2r_predictions.csv
"""

import argparse
import json
import math

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _ndcg_at_k(ranked_ids: list[int], true_id: int, k: int) -> float:
    for i, jid in enumerate(ranked_ids[:k]):
        if jid == true_id:
            return 1.0 / math.log2(i + 2)
    return 0.0


def evaluate_ranking(df: pd.DataFrame, ranked_col: str, ks=(1, 3, 5, 10)) -> dict:
    """
    df must have: true_journal_id, <ranked_col> (JSON list of journal IDs)
    """
    mrr  = 0.0
    ndcg = {k: 0.0 for k in ks}
    acc  = {k: 0   for k in ks}
    n    = 0

    for _, row in df.iterrows():
        true_id = int(row["true_journal_id"])
        ranked  = json.loads(row[ranked_col])
        n += 1
        pos = next((i for i, jid in enumerate(ranked) if jid == true_id), None)
        if pos is not None:
            mrr += 1.0 / (pos + 1)
        for k in ks:
            ndcg[k] += _ndcg_at_k(ranked, true_id, k)
            if pos is not None and pos < k:
                acc[k] += 1

    if n == 0:
        return {}
    out = {"n": n, "MRR": mrr / n}
    for k in ks:
        out[f"NDCG@{k}"] = ndcg[k] / n
        out[f"Acc@{k}"]  = acc[k]  / n
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Build baseline ranked lists from hard_csv
# ─────────────────────────────────────────────────────────────────────────────

def build_baseline_df(hard_csv: str) -> pd.DataFrame:
    """
    For each paper, the baseline ranked list = sort candidates by Base_Rank ascending.
    Returns df with [paper_id, true_journal_id, base_ranked_ids, l2r_ranked_ids]
    """
    hard = pd.read_csv(hard_csv)
    rows = []
    for pid, grp in hard.groupby("Paper_ID"):
        true_id = int(grp[grp["Is_Correct"] == 1]["Predicted_Journal_ID"].iloc[0])
        grp_sorted   = grp.sort_values("Base_Rank")
        base_ranked  = grp_sorted["Predicted_Journal_ID"].astype(int).tolist()
        rows.append({
            "paper_id"        : pid,
            "true_journal_id" : true_id,
            "base_ranked_ids" : json.dumps(base_ranked),
        })
    return pd.DataFrame(rows)


def add_l2r_ranking(base_df: pd.DataFrame, l2r_csv: str) -> pd.DataFrame:
    l2r = pd.read_csv(l2r_csv)
    rows = []
    for pid, grp in l2r.groupby("Paper_ID"):
        true_id = int(grp[grp["Is_Correct"] == 1]["Predicted_Journal_ID"].iloc[0])
        grp_sorted  = grp.sort_values("L2R_Rank")
        l2r_ranked  = grp_sorted["Predicted_Journal_ID"].astype(int).tolist()
        rows.append({
            "paper_id"        : pid,
            "true_journal_id" : true_id,
            "l2r_ranked_ids"  : json.dumps(l2r_ranked),
        })
    l2r_df = pd.DataFrame(rows)
    return base_df.merge(l2r_df[["paper_id", "l2r_ranked_ids"]], on="paper_id", how="left")


# ─────────────────────────────────────────────────────────────────────────────
# Print helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison(metrics_dict: dict[str, dict], ks=(1, 3, 5, 10)):
    metric_keys = ["MRR"] + [f"NDCG@{k}" for k in ks] + [f"Acc@{k}" for k in ks]
    names = list(metrics_dict.keys())

    col_w = 14
    header = f"{'Metric':<12}" + "".join(f"{n:>{col_w}}" for n in names)
    print(header)
    print("-" * len(header))
    for mk in metric_keys:
        row = f"{mk:<12}"
        for name in names:
            v = metrics_dict[name].get(mk, float("nan"))
            row += f"{v:>{col_w}.4f}"
        print(row)
    print()

    # Delta columns (LLM vs baseline, LLM vs L2R)
    if "Baseline" in metrics_dict and "RecRanker (zero-shot)" in metrics_dict:
        print("Delta (RecRanker vs Baseline):")
        for mk in metric_keys:
            b = metrics_dict["Baseline"].get(mk, 0)
            r = metrics_dict["RecRanker (zero-shot)"].get(mk, 0)
            sign = "+" if r >= b else ""
            print(f"  {mk:<12}: {sign}{r-b:.4f}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate RecRanker vs baselines")
    parser.add_argument("--results",  required=True,
                        help="LLM inference results CSV (from inference.py)")
    parser.add_argument("--hard_csv", required=True,
                        help="Hard test cases CSV (data/subset_hard/test.csv)")
    parser.add_argument("--l2r_csv",  default=None,
                        help="L2R predictions CSV (l2r_output_hard/test_l2r_predictions.csv)")
    parser.add_argument("--ks",       type=int, nargs="+", default=[1, 3, 5, 10])
    args = parser.parse_args()

    ks = tuple(args.ks)
    print("Building baseline rankings ...")
    base_df = build_baseline_df(args.hard_csv)

    print("Loading LLM results ...")
    llm_df = pd.read_csv(args.results)

    merged = base_df.merge(
        llm_df[["paper_id", "llm_ranked_ids", "parse_ok"]],
        on="paper_id", how="inner"
    )

    print(f"Papers evaluated: {len(merged):,}  "
          f"(parse_ok: {merged['parse_ok'].sum():,})")

    metrics = {}
    metrics["Baseline"] = evaluate_ranking(merged, "base_ranked_ids", ks)

    if args.l2r_csv:
        base_df = add_l2r_ranking(base_df, args.l2r_csv)
        merged2 = merged.merge(
            base_df[["paper_id", "l2r_ranked_ids"]], on="paper_id", how="left"
        )
        merged2["l2r_ranked_ids"] = merged2["l2r_ranked_ids"].fillna(
            merged2["base_ranked_ids"]
        )
        metrics["L2R (hard-trained)"] = evaluate_ranking(merged2, "l2r_ranked_ids", ks)
        metrics["RecRanker (zero-shot)"] = evaluate_ranking(merged, "llm_ranked_ids", ks)
    else:
        metrics["RecRanker (zero-shot)"] = evaluate_ranking(merged, "llm_ranked_ids", ks)

    print("\n=== Full Comparison on Hard Test Cases ===\n")
    print_comparison(metrics, ks)

    # Breakdown: parse failures
    failed = merged[~merged["parse_ok"]]
    if len(failed):
        print(f"NOTE: {len(failed)} papers had LLM parse failures (fell back to original order).")


if __name__ == "__main__":
    main()
