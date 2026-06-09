import argparse
import math
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "Paper_ID",
    "True_Journal_ID",
    "Predicted_Journal_ID",
    "Base_Rank",
    "Base_Score",
    "Aims_Scope_Sim",
    "Is_Correct",
]


def dcg_at_k(relevances, k):
    """
    relevances: list of 0/1 in rank order
    """
    score = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        if rel > 0:
            score += rel / math.log2(i + 1)
    return score


def ndcg_at_k(relevances, k):
    """
    Binary relevance NDCG@k.
    """
    actual = dcg_at_k(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    ideal = dcg_at_k(ideal_relevances, k)
    if ideal == 0:
        return 0.0
    return actual / ideal


def reciprocal_rank(relevances):
    """
    Return 1/rank of the first relevant item, or 0 if none.
    """
    for idx, rel in enumerate(relevances, start=1):
        if rel > 0:
            return 1.0 / idx
    return 0.0


def validate_input(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def compute_metrics(df, ks):
    # Ensure correct ranking order inside each paper
    df = df.sort_values(["Paper_ID", "Base_Rank"], ascending=[True, True]).copy()

    per_paper_rows = []
    ndcg_sums = {k: 0.0 for k in ks}
    rr_sum = 0.0
    num_papers = 0

    for paper_id, group in df.groupby("Paper_ID", sort=False):
        group = group.sort_values("Base_Rank")
        relevances = group["Is_Correct"].astype(int).tolist()

        rr = reciprocal_rank(relevances)
        rr_sum += rr

        paper_result = {
            "Paper_ID": paper_id,
            "RR": rr,
            "First_Correct_Rank": None,
        }

        first_correct = next((i for i, rel in enumerate(relevances, start=1) if rel > 0), None)
        paper_result["First_Correct_Rank"] = first_correct

        for k in ks:
            val = ndcg_at_k(relevances, k)
            ndcg_sums[k] += val
            paper_result[f"NDCG@{k}"] = val

        per_paper_rows.append(paper_result)
        num_papers += 1

    summary = {
        "num_papers": num_papers,
        "MRR": rr_sum / num_papers if num_papers > 0 else 0.0,
    }
    for k in ks:
        summary[f"NDCG@{k}"] = ndcg_sums[k] / num_papers if num_papers > 0 else 0.0

    per_paper_df = pd.DataFrame(per_paper_rows)
    summary_df = pd.DataFrame([summary])

    return summary_df, per_paper_df


def main():
    parser = argparse.ArgumentParser(description="Compute MRR and NDCG@k from ranked prediction CSV")
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to CSV file with columns: Paper_ID, True_Journal_ID, Predicted_Journal_ID, Base_Rank, Base_Score, Aims_Scope_Sim, Is_Correct",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="List of k values for NDCG@k, e.g. --ks 1 3 5 10",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default=None,
        help="Prefix to save outputs. If omitted, files are saved next to input_csv.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    validate_input(df)

    # Optional sanity check: warn if some papers have no correct label in the file
    paper_has_correct = df.groupby("Paper_ID")["Is_Correct"].max()
    missing_correct = (paper_has_correct == 0).sum()
    if missing_correct > 0:
        print(f"[Warning] {missing_correct} papers have no Is_Correct = 1 row in the file. Their RR/NDCG will be 0.")

    max_rank_in_file = int(df["Base_Rank"].max())
    for k in args.ks:
        if k > max_rank_in_file:
            print(f"[Warning] NDCG@{k} requested, but max Base_Rank in file is {max_rank_in_file}. Metric will be computed on available rows only.")

    summary_df, per_paper_df = compute_metrics(df, args.ks)

    if args.output_prefix:
        out_prefix = Path(args.output_prefix)
    else:
        out_prefix = input_path.with_suffix("")

    summary_path = f"{out_prefix}_summary_metrics.csv"
    per_paper_path = f"{out_prefix}_per_paper_metrics.csv"

    summary_df.to_csv(summary_path, index=False)
    per_paper_df.to_csv(per_paper_path, index=False)

    print("\n=== Overall Metrics ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved per-paper metrics to: {per_paper_path}")


if __name__ == "__main__":
    main()