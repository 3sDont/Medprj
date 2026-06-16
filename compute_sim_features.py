# -*- coding: utf-8 -*-
"""
compute_sim_features.py — Generate domain_sim and research_focus_sim feature CSVs
for the L2R pipeline, using LLM-extracted paper and journal metadata.

Two similarity modes:
  jaccard  : Jaccard overlap on extracted label strings (fast, no GPU needed)
  embed    : Embed evidence texts with a sentence-transformer, compute cosine sim
             (follows the diagram: embed scientific_domains_evidence / research_focuses_evidence
              for both papers and journals, then cosine similarity)

Steps:
  1. Index train_set.csv by title -> Paper_ID (row index)
  2. Load extracted journals -> Journal_ID: {evidence texts / label sets}
  3. Load extracted papers -> match title -> Paper_ID: {evidence texts / label sets}
  4. [embed only] Precompute and normalise all embeddings
  5. Load pred_train.csv, filter to extracted papers, compute similarities
  6. Save domain_sim_train.csv, research_focus_sim_train.csv
  7. Split 80/20 by paper and save pred_feat_train.csv, pred_feat_eval.csv

Usage:
    # Jaccard (fast baseline):
    python compute_sim_features.py --sim_method jaccard

    # Embedding-based cosine similarity (as in the architecture diagram):
    python compute_sim_features.py --sim_method embed
    python compute_sim_features.py --sim_method embed --embed_model pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb
"""

import argparse
import json
import os
import warnings
from glob import glob

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Default paths
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_SET_CSV        = "data/train_set.csv"
EXTRACTED_JOURNALS   = "data/extracted_journals/extracted.jsonl"
EXTRACTED_PAPERS_DIR = "data/extracted_papers/success"
PRED_TRAIN_CSV       = "data/pred_train.csv"

OUT_DOMAIN_SIM       = "data/domain_sim_train.csv"
OUT_FOCUS_SIM        = "data/research_focus_sim_train.csv"
OUT_PRED_TRAIN       = "data/pred_feat_train.csv"
OUT_PRED_EVAL        = "data/pred_feat_eval.csv"

DEFAULT_EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return s.lower().strip()


def _norm_set(strings) -> set:
    return {_norm(s) for s in strings if s}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def evidence_dict_to_text(ev_dict: dict) -> str:
    """
    Flatten {domain/focus: [evidence phrases]} into one text string.
    Example: {"Neurology": ["brain", "nerve"]} -> "Neurology: brain, nerve"
    """
    parts = []
    for key, values in ev_dict.items():
        if isinstance(values, list):
            phrases = ", ".join(str(v) for v in values if v)
            parts.append(f"{key}: {phrases}" if phrases else str(key))
        else:
            parts.append(str(key))
    return "; ".join(parts)


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return mat / norms


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Index train_set.csv title -> Paper_ID
# ─────────────────────────────────────────────────────────────────────────────

def build_title_index(train_csv: str) -> dict:
    print("[1] Indexing train_set.csv by title ...")
    df = pd.read_csv(train_csv, usecols=["Title"])
    idx = {_norm(row["Title"]): paper_id for paper_id, row in df.iterrows()}
    print(f"    {len(idx):,} papers indexed")
    return idx


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Load extracted journals
# ─────────────────────────────────────────────────────────────────────────────

def load_journal_features(jsonl_path: str, sim_method: str) -> dict:
    """
    jaccard mode -> {jid: {"domains": set, "focuses": set}}
    embed   mode -> {jid: {"domain_text": str, "focus_text": str}}
    """
    print(f"[2] Loading extracted journals ({sim_method} mode) ...")
    journal_map = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            jid = int(rec["label"])
            if sim_method == "jaccard":
                journal_map[jid] = {
                    "domains": _norm_set(rec.get("scientific_domains", [])),
                    "focuses": _norm_set(rec.get("research_focuses", [])),
                }
            else:  # embed
                journal_map[jid] = {
                    "domain_text": evidence_dict_to_text(
                        rec.get("scientific_domains_evidence", {})
                    ),
                    "focus_text": evidence_dict_to_text(
                        rec.get("research_focuses_evidence", {})
                    ),
                }
    print(f"    {len(journal_map):,} journals loaded")
    return journal_map


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Load extracted papers -> Paper_ID
# ─────────────────────────────────────────────────────────────────────────────

def load_paper_features(papers_dir: str, title_index: dict, sim_method: str) -> dict:
    """
    jaccard mode -> {paper_id: {"domains": set, "focuses": set}}
    embed   mode -> {paper_id: {"domain_text": str, "focus_text": str}}
    """
    print(f"[3] Loading extracted papers ({sim_method} mode) ...")
    jsonl_files = glob(os.path.join(papers_dir, "*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in {papers_dir}")

    paper_map = {}
    n_total = n_matched = n_missed = 0

    for path in jsonl_files:
        print(f"    Reading {os.path.basename(path)} ...")
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_total += 1
                rec = json.loads(line)
                title_key = _norm(rec.get("title", ""))
                paper_id = title_index.get(title_key)
                if paper_id is None:
                    n_missed += 1
                    if n_missed <= 5:
                        print(f"    WARNING: not found in train_set.csv: "
                              f"'{rec.get('title', '')[:80]}'")
                    continue
                n_matched += 1
                if sim_method == "jaccard":
                    paper_map[paper_id] = {
                        "domains": _norm_set(
                            rec.get("scientific_domains_evidence", {}).keys()
                        ),
                        "focuses": _norm_set(
                            rec.get("research_focuses_evidence", {}).keys()
                        ),
                    }
                else:  # embed
                    paper_map[paper_id] = {
                        "domain_text": evidence_dict_to_text(
                            rec.get("scientific_domains_evidence", {})
                        ),
                        "focus_text": evidence_dict_to_text(
                            rec.get("research_focuses_evidence", {})
                        ),
                    }

    print(f"    {n_total:,} records read | {n_matched:,} matched | {n_missed:,} unmatched")
    return paper_map


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 (embed only): Precompute normalised embeddings
# ─────────────────────────────────────────────────────────────────────────────

def precompute_embeddings(paper_map: dict, journal_map: dict, model_name: str):
    """
    Returns four normalised numpy arrays + index mappings.
    paper_domain_embs  : (N_papers,  D)
    paper_focus_embs   : (N_papers,  D)
    journal_domain_embs: (N_journals, D)
    journal_focus_embs : (N_journals, D)
    """
    from sentence_transformers import SentenceTransformer

    print(f"\n[4] Loading sentence-transformer: {model_name} ...")
    model = SentenceTransformer(model_name)

    paper_ids   = list(paper_map.keys())
    journal_ids = list(journal_map.keys())

    print(f"    Encoding {len(paper_ids):,} paper domain texts ...")
    paper_domain_embs = model.encode(
        [paper_map[pid]["domain_text"] for pid in paper_ids],
        batch_size=256, show_progress_bar=True, convert_to_numpy=True
    )
    print(f"    Encoding {len(paper_ids):,} paper focus texts ...")
    paper_focus_embs = model.encode(
        [paper_map[pid]["focus_text"] for pid in paper_ids],
        batch_size=256, show_progress_bar=True, convert_to_numpy=True
    )
    print(f"    Encoding {len(journal_ids):,} journal domain texts ...")
    journal_domain_embs = model.encode(
        [journal_map[jid]["domain_text"] for jid in journal_ids],
        batch_size=256, show_progress_bar=True, convert_to_numpy=True
    )
    print(f"    Encoding {len(journal_ids):,} journal focus texts ...")
    journal_focus_embs = model.encode(
        [journal_map[jid]["focus_text"] for jid in journal_ids],
        batch_size=256, show_progress_bar=True, convert_to_numpy=True
    )

    # L2-normalise so dot product = cosine similarity
    paper_domain_embs   = l2_normalize(paper_domain_embs)
    paper_focus_embs    = l2_normalize(paper_focus_embs)
    journal_domain_embs = l2_normalize(journal_domain_embs)
    journal_focus_embs  = l2_normalize(journal_focus_embs)

    pid_to_idx = {pid: i for i, pid in enumerate(paper_ids)}
    jid_to_idx = {jid: i for i, jid in enumerate(journal_ids)}

    print(f"    Embedding dim: {paper_domain_embs.shape[1]}")
    return (paper_domain_embs, paper_focus_embs,
            journal_domain_embs, journal_focus_embs,
            pid_to_idx, jid_to_idx)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Load pred_csv, filter, compute similarities
# ─────────────────────────────────────────────────────────────────────────────

def compute_jaccard_similarities(pred_csv, paper_map, journal_map, chunk_size):
    print(f"\n[5] Computing Jaccard similarities from {pred_csv} ...")
    extracted_ids = set(paper_map.keys())
    _empty = {"domains": set(), "focuses": set()}
    chunks_out = []
    n_read = n_kept = 0

    for chunk in pd.read_csv(pred_csv, chunksize=chunk_size):
        n_read += len(chunk)
        sub = chunk[chunk["Paper_ID"].isin(extracted_ids)].copy()
        if sub.empty:
            continue
        pids = sub["Paper_ID"].astype(int).values
        jids = sub["Predicted_Journal_ID"].astype(int).values
        sub["domain_similarity"] = [
            jaccard(paper_map[p]["domains"], journal_map.get(j, _empty)["domains"])
            for p, j in zip(pids, jids)
        ]
        sub["research_focus_similarity"] = [
            jaccard(paper_map[p]["focuses"], journal_map.get(j, _empty)["focuses"])
            for p, j in zip(pids, jids)
        ]
        chunks_out.append(sub)
        n_kept += len(sub)
        if n_read % 5_000_000 == 0:
            print(f"    ... read {n_read:,} rows, kept {n_kept:,}")

    print(f"    Done - read {n_read:,} total, kept {n_kept:,}")
    return pd.concat(chunks_out, ignore_index=True)


def compute_embed_similarities(pred_csv, paper_map, chunk_size,
                               paper_domain_embs, paper_focus_embs,
                               journal_domain_embs, journal_focus_embs,
                               pid_to_idx, jid_to_idx):
    """
    Efficient cosine similarity using pre-normalised embeddings.
    For each (paper, journal) pair: cosine_sim = dot(p_emb, j_emb)
    Uses numpy row indexing for speed (no Python loops over rows).
    """
    print(f"\n[5] Computing embedding cosine similarities from {pred_csv} ...")
    extracted_ids = set(paper_map.keys())
    chunks_out = []
    n_read = n_kept = 0

    for chunk in pd.read_csv(pred_csv, chunksize=chunk_size):
        n_read += len(chunk)
        sub = chunk[chunk["Paper_ID"].isin(extracted_ids)].copy()
        if sub.empty:
            continue

        pids = sub["Paper_ID"].astype(int).values
        jids = sub["Predicted_Journal_ID"].astype(int).values

        p_idx = np.array([pid_to_idx[p] for p in pids])
        # journals not in extracted set get similarity 0
        j_idx = np.array([jid_to_idx.get(j, -1) for j in jids])

        valid = j_idx >= 0

        # domain cosine sim
        domain_sims = np.zeros(len(pids), dtype=np.float32)
        if valid.any():
            domain_sims[valid] = (
                paper_domain_embs[p_idx[valid]] *
                journal_domain_embs[j_idx[valid]]
            ).sum(axis=1)

        # focus cosine sim
        focus_sims = np.zeros(len(pids), dtype=np.float32)
        if valid.any():
            focus_sims[valid] = (
                paper_focus_embs[p_idx[valid]] *
                journal_focus_embs[j_idx[valid]]
            ).sum(axis=1)

        sub["domain_similarity"]         = domain_sims
        sub["research_focus_similarity"] = focus_sims
        chunks_out.append(sub)
        n_kept += len(sub)
        if n_read % 5_000_000 == 0:
            print(f"    ... read {n_read:,} rows, kept {n_kept:,}")

    print(f"    Done - read {n_read:,} total, kept {n_kept:,}")
    return pd.concat(chunks_out, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 7: Split 80/20 by Paper_ID
# ─────────────────────────────────────────────────────────────────────────────

def split_by_paper(df: pd.DataFrame, eval_frac: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    paper_ids = df["Paper_ID"].unique()
    rng.shuffle(paper_ids)
    n_eval    = int(len(paper_ids) * eval_frac)
    eval_ids  = set(paper_ids[:n_eval])
    train_ids = set(paper_ids[n_eval:])
    return (df[df["Paper_ID"].isin(train_ids)],
            df[df["Paper_ID"].isin(eval_ids)])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--sim_method",     choices=["jaccard", "embed"], default="jaccard",
                        help="jaccard: label-string overlap  |  embed: evidence-text cosine similarity")
    parser.add_argument("--embed_model",    default=DEFAULT_EMBED_MODEL,
                        help="Sentence-transformer model name (used only with --sim_method embed)")
    parser.add_argument("--train_set_csv",  default=TRAIN_SET_CSV)
    parser.add_argument("--journals_jsonl", default=EXTRACTED_JOURNALS)
    parser.add_argument("--papers_dir",     default=EXTRACTED_PAPERS_DIR)
    parser.add_argument("--pred_csv",       default=PRED_TRAIN_CSV)
    parser.add_argument("--out_domain_sim", default=OUT_DOMAIN_SIM)
    parser.add_argument("--out_focus_sim",  default=OUT_FOCUS_SIM)
    parser.add_argument("--out_pred_train", default=OUT_PRED_TRAIN)
    parser.add_argument("--out_pred_eval",  default=OUT_PRED_EVAL)
    parser.add_argument("--eval_frac",      type=float, default=0.2)
    parser.add_argument("--split_seed",     type=int,   default=42)
    parser.add_argument("--chunk_size",     type=int,   default=500_000)
    args = parser.parse_args()

    print("=" * 60)
    print(f"  compute_sim_features.py  [mode: {args.sim_method}]")
    print("=" * 60)

    title_index = build_title_index(args.train_set_csv)
    journal_map = load_journal_features(args.journals_jsonl, args.sim_method)
    paper_map   = load_paper_features(args.papers_dir, title_index, args.sim_method)

    if args.sim_method == "embed":
        (paper_domain_embs, paper_focus_embs,
         journal_domain_embs, journal_focus_embs,
         pid_to_idx, jid_to_idx) = precompute_embeddings(
            paper_map, journal_map, args.embed_model
        )
        df = compute_embed_similarities(
            args.pred_csv, paper_map, args.chunk_size,
            paper_domain_embs, paper_focus_embs,
            journal_domain_embs, journal_focus_embs,
            pid_to_idx, jid_to_idx,
        )
    else:
        df = compute_jaccard_similarities(
            args.pred_csv, paper_map, journal_map, args.chunk_size
        )

    # ── Stats ──────────────────────────────────────────────────────────────────
    n_papers  = df["Paper_ID"].nunique()
    nz_domain = (df["domain_similarity"] > 0).sum()
    nz_focus  = (df["research_focus_similarity"] > 0).sum()
    print(f"\n[6] Stats over {n_papers:,} papers ({len(df):,} rows)")
    print(f"    domain_similarity      > 0 : {nz_domain:,} rows ({100*nz_domain/len(df):.1f}%)")
    print(f"    research_focus_sim     > 0 : {nz_focus:,}  rows ({100*nz_focus/len(df):.1f}%)")
    print(f"    domain_similarity mean     : {df['domain_similarity'].mean():.4f}")
    print(f"    research_focus_sim mean    : {df['research_focus_similarity'].mean():.4f}")

    # ── Save similarity CSVs ───────────────────────────────────────────────────
    print(f"\n[7] Saving CSVs ...")
    sim_cols = ["Paper_ID", "Predicted_Journal_ID"]
    df[sim_cols + ["domain_similarity"]].to_csv(args.out_domain_sim, index=False)
    print(f"    Saved -> {args.out_domain_sim}  ({len(df):,} rows)")
    df[sim_cols + ["research_focus_similarity"]].to_csv(args.out_focus_sim, index=False)
    print(f"    Saved -> {args.out_focus_sim}  ({len(df):,} rows)")

    # ── Split 80/20 and save pred subsets ──────────────────────────────────────
    print(f"\n[8] Splitting {n_papers:,} papers 80/20 ...")
    df_train, df_eval = split_by_paper(df, args.eval_frac, args.split_seed)

    base_cols  = ["Paper_ID", "True_Label", "Rank", "Predicted_Journal_ID", "Score", "Is_Correct"]
    keep_cols  = [c for c in base_cols if c in df.columns]
    df_train[keep_cols].to_csv(args.out_pred_train, index=False)
    df_eval[keep_cols].to_csv(args.out_pred_eval,   index=False)
    print(f"    pred_feat_train : {df_train['Paper_ID'].nunique():,} papers, "
          f"{len(df_train):,} rows -> {args.out_pred_train}")
    print(f"    pred_feat_eval  : {df_eval['Paper_ID'].nunique():,} papers, "
          f"{len(df_eval):,} rows  -> {args.out_pred_eval}")

    print("\nDone. Run experiments:")
    print(f"  Exp A (baseline):")
    print(f"    python train_l2r.py --train_csv {args.out_pred_train} "
          f"--val_csv {args.out_pred_eval} --test_csv {args.out_pred_eval} "
          f"--output_dir l2r_output/exp_A_baseline")
    print(f"  Exp B (LLM features, {args.sim_method}):")
    print(f"    python train_l2r.py --train_csv {args.out_pred_train} "
          f"--val_csv {args.out_pred_eval} --test_csv {args.out_pred_eval} "
          f"--domain_sim_csv {args.out_domain_sim} "
          f"--research_focus_sim_csv {args.out_focus_sim} "
          f"--output_dir l2r_output/exp_B_{args.sim_method}")


if __name__ == "__main__":
    main()
