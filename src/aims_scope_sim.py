# -*- coding: utf-8 -*-
"""
aims_scope_sim.py — Compute Aims_Scope_Sim for a single user-provided paper
                    using allenai/specter2 (independent from BioBERT classifier).

Why SPECTER2:
  - Trained specifically for academic paper similarity
  - Independent from the BioBERT classifier → cosine sim is a true complementary feature
  - normalize_embeddings=True → cosine sim = dot product → fast

Usage (CLI):
    python aims_scope_sim.py \\
        --inference_json inference_result.json \\
        --data_path /path/to/data \\
        --aims_csv journal_category.csv \\
        --output_json inference_with_sim.json
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List


def load_specter2(model_name: str = "allenai/specter2_base") -> SentenceTransformer:
    """Load SPECTER2 sentence encoder."""
    print(f"Loading encoder: {model_name} ...")
    return SentenceTransformer(model_name)


def encode_journal_aims(
    encoder: SentenceTransformer,
    journal_df: pd.DataFrame,
    batch_size: int = 64,
    use_category: bool = False,
) -> np.ndarray:
    """
    Encode all journal aims with SPECTER2.

    Returns:
        np.ndarray [n_journals, hidden_size], float32, L2-normalized.
    """
    if use_category and "Categories" in journal_df.columns:
        texts = (journal_df["Aims"] + " " + journal_df["Categories"]).tolist()
    else:
        texts = journal_df["Aims"].tolist()

    print(f"Encoding {len(texts)} journal aims...")
    return encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def compute_aims_sim_single(
    title: str,
    abstract: str,
    keywords: str,
    top_journals: List[dict],
    encoder: SentenceTransformer,
    aims_embs: np.ndarray,
    features: str = "TAK",
) -> List[dict]:
    """
    Compute Aims_Scope_Sim for each journal in top_journals for a single paper.

    Each entry in top_journals must have a 'journal_idx' key (row index into aims_embs).
    Mutates and returns the list with 'Aims_Scope_Sim' added to each dict.
    """
    parts = {"T": title, "A": abstract, "K": keywords}
    text = " ".join(parts[c] for c in features if c in parts).strip()

    paper_emb = encoder.encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )  # [1, hidden_size], L2-normalized

    for j in top_journals:
        jid = j["journal_idx"]
        sim = float((paper_emb[0] * aims_embs[jid]).sum())
        j["Aims_Scope_Sim"] = round(sim, 6)

    return top_journals


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute Aims_Scope_Sim for a single paper using SPECTER2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Paper input (optional override; falls back to values stored in inference_json)
    parser.add_argument("--title",    type=str, default="")
    parser.add_argument("--abstract", type=str, default="")
    parser.add_argument("--keywords", type=str, default="")
    # Inference result from inference.py
    parser.add_argument("--inference_json", type=str, required=True,
                        help="JSON output from inference.py (contains paper + top_journals)")
    # Encoder
    parser.add_argument("--encoder_model", type=str, default="allenai/specter2_base")
    parser.add_argument("--batch_size",    type=int, default=64)
    # Journal / aims data
    parser.add_argument("--data_path",    type=str, required=True,
                        help="Folder containing aims CSV")
    parser.add_argument("--aims_csv",     type=str, default="journal_category.csv")
    parser.add_argument("--use_category", action="store_true",
                        help="Append Categories column to Aims when encoding")
    # Features
    parser.add_argument("--features", type=str, default="TAK")
    # Output
    parser.add_argument("--output_json", type=str, default="inference_with_sim.json")
    args = parser.parse_args()

    # Load inference result
    with open(args.inference_json, "r", encoding="utf-8") as f:
        inference_result = json.load(f)

    # Paper info: CLI args override values in JSON
    paper    = inference_result.get("paper", {})
    title    = args.title    or paper.get("title", "")
    abstract = args.abstract or paper.get("abstract", "")
    keywords = args.keywords or paper.get("keywords", "")
    top_journals = inference_result["top_journals"]

    # Load encoder and journal data
    encoder = load_specter2(args.encoder_model)
    journal_df = pd.read_csv(
        os.path.join(args.data_path, args.aims_csv), encoding="ISO-8859-1"
    )
    journal_df.fillna("", inplace=True)

    aims_embs = encode_journal_aims(encoder, journal_df, args.batch_size, args.use_category)

    # Compute similarity for each top journal
    top_journals = compute_aims_sim_single(
        title=title,
        abstract=abstract,
        keywords=keywords,
        top_journals=top_journals,
        encoder=encoder,
        aims_embs=aims_embs,
        features=args.features,
    )

    # Save updated result
    inference_result["paper"] = {"title": title, "abstract": abstract, "keywords": keywords}
    inference_result["top_journals"] = top_journals
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(inference_result, f, ensure_ascii=False, indent=2)

    print(f"\nSaved result with Aims_Scope_Sim → {args.output_json}")
    for j in top_journals[:5]:
        print(f"  #{j['Rank']:2d}  {j['Name'][:55]:55s}  base={j['Base_Score']:.4f}  sim={j['Aims_Scope_Sim']:.4f}")
