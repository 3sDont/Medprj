# -*- coding: utf-8 -*-
"""
pipeline.py — Full end-to-end journal recommendation pipeline for a single paper.

Steps:
  1. inference.py      → top-20 journals (Base_Rank, Base_Score)
  2. aims_scope_sim.py → Aims_Scope_Sim per journal (SPECTER2)
  3. llm_extract.py    → paper features via Qwen + coverage metrics per journal
  4. reasoning.py      → rerank by final_fit_score + Vietnamese explanations (Claude)

Load models once with load_pipeline(), then call run_pipeline() per request.

CLI usage:
    python pipeline.py \\
        --checkpoint_path model.pth \\
        --model_name roberta-base \\
        --data_path /path/to/data \\
        --title "..." --abstract "..." --keywords "..." \\
        --output_json result.json
"""
import os
import sys
import hashlib
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Allow running from project root (python src/pipeline.py) or from src/
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import argparse
import json
import uuid
from dataclasses import dataclass
from typing import List, Optional

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer


# ── Embedding cache helpers ───────────────────────────────────────────────────

def _cache_tag(model_name: str, csv_path: str) -> str:
    """Short hash that changes when model name or CSV file changes."""
    mtime = str(os.path.getmtime(csv_path)) if os.path.exists(csv_path) else "0"
    return hashlib.md5(f"{model_name}|{mtime}".encode()).hexdigest()[:10]


def _load_biobert_cache(cache_dir: str, model_name: str, csv_path: str, device):
    tag  = _cache_tag(model_name, csv_path)
    path = os.path.join(cache_dir, f"biobert_aims_{tag}.pt")
    if os.path.exists(path):
        print(f"[cache] Loading BioBERT aims embeddings from {path}")
        return torch.load(path, map_location=device, weights_only=True)
    return None


def _save_biobert_cache(cache_dir: str, model_name: str, csv_path: str, tensor):
    os.makedirs(cache_dir, exist_ok=True)
    tag  = _cache_tag(model_name, csv_path)
    path = os.path.join(cache_dir, f"biobert_aims_{tag}.pt")
    torch.save(tensor.cpu(), path)
    print(f"[cache] BioBERT aims embeddings saved → {path}")


def _load_specter_cache(cache_dir: str, model_name: str, csv_path: str):
    tag  = _cache_tag(model_name, csv_path)
    path = os.path.join(cache_dir, f"specter_aims_{tag}.npy")
    if os.path.exists(path):
        print(f"[cache] Loading SPECTER2 aims embeddings from {path}")
        return np.load(path)
    return None


def _save_specter_cache(cache_dir: str, model_name: str, csv_path: str, arr: np.ndarray):
    os.makedirs(cache_dir, exist_ok=True)
    tag  = _cache_tag(model_name, csv_path)
    path = os.path.join(cache_dir, f"specter_aims_{tag}.npy")
    np.save(path, arr)
    print(f"[cache] SPECTER2 aims embeddings saved → {path}")

from inference import load_model, run_inference_single
from aims_scope_sim import load_specter2, encode_journal_aims, compute_aims_sim_single
from llm_extract import QwenExtractor, load_journal_extract, process_llm_extraction
from reasoning import rerank_journals, generate_all_explanations


# ── Helper ────────────────────────────────────────────────────────────────────

def _to_grouped(items: List[str]) -> dict:
    """
    Convert a flat list to a numbered-group dict for the output JSON.
      ≤ 3 items → {"1": all_items}
      > 3 items → {"1": first_half, "2": second_half}
    """
    if not items:
        return {}
    if len(items) <= 3:
        return {"1": items}
    mid = (len(items) + 1) // 2
    result: dict = {"1": items[:mid]}
    if items[mid:]:
        result["2"] = items[mid:]
    return result


# ── Model container ──────────────────────────────────────────────────────────

@dataclass
class PipelineModels:
    """All pre-loaded models and data. Instantiate once, reuse per request."""
    # Classifier (BioBERT-based)
    classifier:      object
    base_model:      object
    tokenizer:       object
    aims_embeddings: object           # torch.Tensor on device
    device:          torch.device
    # SPECTER2 encoder
    specter2:          SentenceTransformer
    specter_aims_embs: np.ndarray     # [n_journals, hidden_size], L2-normalised
    # Data
    journal_df:       pd.DataFrame
    journal_extracts: dict
    # Qwen extractor (paper feature extraction)
    qwen_extractor:   QwenExtractor
    # Config — fields with defaults must come after fields without defaults
    features:     str  = "TAK"
    max_len:      int  = 512
    use_aim:      bool = True
    use_category: bool = False


# ── Loader ───────────────────────────────────────────────────────────────────

def load_pipeline(
    checkpoint_path: str,
    model_name: str,
    data_path: str,
    aims_csv: str              = "journal_full_info.csv",
    journal_extract_jsonl: str = "journal_extract.json",
    encoder_model: str         = "allenai/specter2_base",
    qwen_model: str            = "Qwen/Qwen3.5-2B",
    pooler_type: str           = "cls",
    features: str              = "TAK",
    max_len: int               = 512,
    use_aim: bool              = True,
    use_category: bool         = False,
    cache_dir: Optional[str]   = None,
) -> PipelineModels:
    """
    Load all models and data needed by the pipeline.
    Call once at startup; pass the returned PipelineModels to run_pipeline().

    Args:
        cache_dir: directory for pre-computed embedding caches.
                   Pass None to disable caching.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Journal metadata ──────────────────────────────────────────────────────
    csv_path   = os.path.join(data_path, aims_csv)
    journal_df = pd.read_csv(csv_path, encoding="ISO-8859-1")
    journal_df.fillna("", inplace=True)
    n_classes = len(journal_df)
    print(f"Loaded {n_classes} journals from {aims_csv}")

    if use_category and "Categories" in journal_df.columns:
        X_aims = (journal_df["Aims"] + " " + journal_df["Categories"]).tolist()
    else:
        X_aims = journal_df["Aims"].tolist()

    # ── Classifier ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier, base_model = load_model(
        checkpoint_path, model_name, pooler_type, n_classes, use_aim, device
    )

    # Load BioBERT aims embeddings from cache or recompute
    aims_embeddings = (
        _load_biobert_cache(cache_dir, model_name, csv_path, device)
        if cache_dir else None
    )
    if aims_embeddings is None:
        print("Encoding classifier aims embeddings...")
        aims_embeddings = base_model.encode(
            X_aims, show_progress_bar=True, convert_to_tensor=True,
            device=device, tokenizer=tokenizer, max_len=max_len,
        )
        if cache_dir:
            _save_biobert_cache(cache_dir, model_name, csv_path, aims_embeddings)
    aims_embeddings = aims_embeddings.to(device)

    # ── SPECTER2 ──────────────────────────────────────────────────────────────
    specter2 = load_specter2(encoder_model)

    # Load SPECTER2 aims embeddings from cache or recompute
    specter_aims_embs = (
        _load_specter_cache(cache_dir, encoder_model, csv_path)
        if cache_dir else None
    )
    if specter_aims_embs is None:
        specter_aims_embs = encode_journal_aims(
            specter2, journal_df, use_category=use_category
        )
        if cache_dir:
            _save_specter_cache(cache_dir, encoder_model, csv_path, specter_aims_embs)

    # ── Journal extract JSONL ─────────────────────────────────────────────────
    extract_path = os.path.join(data_path, journal_extract_jsonl)
    journal_extracts = load_journal_extract(extract_path)
    print(f"Loaded {len(journal_extracts)} journal extract entries")

    # ── Qwen extractor ────────────────────────────────────────────────────────
    qwen_extractor = QwenExtractor(model_name=qwen_model)

    return PipelineModels(
        classifier=classifier,
        base_model=base_model,
        tokenizer=tokenizer,
        aims_embeddings=aims_embeddings,
        device=device,
        specter2=specter2,
        specter_aims_embs=specter_aims_embs,
        journal_df=journal_df,
        journal_extracts=journal_extracts,
        qwen_extractor=qwen_extractor,
        features=features,
        max_len=max_len,
        use_aim=use_aim,
        use_category=use_category,
    )


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_pipeline(
    title: str,
    abstract: str,
    keywords: str,
    models: PipelineModels,
    topk: int               = 10,
    top_n_explain: int      = 10,
    paper_id: Optional[str] = None,
) -> dict:
    """
    Run the full pipeline for a single user-provided paper.

    Args:
        title / abstract / keywords: paper text inputs (separate strings)
        models:        pre-loaded PipelineModels from load_pipeline()
        topk:          number of candidate journals from classifier
        top_n_explain: number of journals to generate Vietnamese explanations for

    Returns:
        Final structured JSON dict matching the target schema.
    """
    if paper_id is None:
        paper_id = f"req_{uuid.uuid4().hex[:8]}"

    # ── Step 1: Classifier inference ──────────────────────────────────────────
    print("[1/4] Classifier inference...")
    top_journals: List[dict] = run_inference_single(
        title=title, abstract=abstract, keywords=keywords,
        model=models.classifier,
        tokenizer=models.tokenizer,
        aims_embeddings=models.aims_embeddings,
        journal_df=models.journal_df,
        device=models.device,
        features=models.features,
        max_len=models.max_len,
        topk=topk,
        use_aim=models.use_aim,
    )

    # Attach Categories list (used in coverage + output JSON)
    for j in top_journals:
        raw_cats = str(models.journal_df.iloc[j["journal_idx"]].get("Categories", ""))
        j["Categories"] = [c.strip() for c in raw_cats.split(",") if c.strip()]

    # ── Step 2: Aims/Scope similarity ─────────────────────────────────────────
    print("[2/4] Computing Aims_Scope_Sim (SPECTER2)...")
    top_journals = compute_aims_sim_single(
        title=title, abstract=abstract, keywords=keywords,
        top_journals=top_journals,
        encoder=models.specter2,
        aims_embs=models.specter_aims_embs,
        features=models.features,
    )

    # ── Step 3: Qwen extraction + coverage metrics ─────────────────────────────
    print("[3/4] Qwen extraction + coverage metrics...")
    paper_features, top_journals = process_llm_extraction(
        title=title, abstract=abstract, keywords=keywords,
        top_journals=top_journals,
        journal_extracts=models.journal_extracts,
        extractor=models.qwen_extractor,
        encoder=models.specter2,
    )

    # ── Step 4: Rerank + Vietnamese explanations ───────────────────────────────
    print("[4/4] Reranking + generating explanations...")
    top_journals = rerank_journals(top_journals)

    paper_info = {
        "title":              title,
        "abstract":           abstract,
        "keywords":           keywords,
        "extracted_features": paper_features,
    }
    top_journals = generate_all_explanations(
        paper_info=paper_info,
        top_journals=top_journals,
        extractor=models.qwen_extractor,
        top_n=top_n_explain,
    )

    # ── Assemble final JSON ────────────────────────────────────────────────────
    # Strip internal-only fields that must not appear in output
    for j in top_journals:
        j.pop("journal_idx", None)
        j.get("Rerank", {}).pop("rank_change", None)

    return {
        "paper_id": paper_id,
        "paper_information": {
            "inputs": {
                "T": title,
                "A": abstract,
                "K": [k.strip() for k in keywords.split(",")] if keywords else [],
            },
            "extracted_paper_features": {
                # Convert flat lists from Qwen output → grouped dict for output schema
                "sci_evidence":      _to_grouped(paper_features.get("scientific_domains", [])),
                "research_evidence": _to_grouped(paper_features.get("research_focuses", [])),
            },
        },
        "Top10_journals": top_journals,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end journal recommendation pipeline (single paper)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model / data paths
    parser.add_argument("--checkpoint_path",       type=str, required=True)
    parser.add_argument("--model_name",            type=str, default="roberta-base")
    parser.add_argument("--pooler_type",           type=str, default="cls")
    parser.add_argument("--data_path",             type=str, required=True,
                        help="Folder containing aims CSV and journal_extract.jsonl")
    parser.add_argument("--aims_csv",              type=str, default="journal_full_info.csv")
    parser.add_argument("--journal_extract_jsonl", type=str, default="journal_extract.jsonl")
    parser.add_argument("--encoder_model",         type=str, default="allenai/specter2_base")
    parser.add_argument("--qwen_model",            type=str, default="Qwen/Qwen3.5-2B",
                        help="HuggingFace model ID for the Qwen extractor")
    # Paper input
    parser.add_argument("--title",    type=str, default="")
    parser.add_argument("--abstract", type=str, default="")
    parser.add_argument("--keywords", type=str, default="",
                        help="Comma-separated keyword string")
    parser.add_argument("--paper_id", type=str, default=None)
    # Classifier options
    parser.add_argument("--features",     type=str, default="TAK")
    parser.add_argument("--use_aim",      action="store_true")
    parser.add_argument("--use_category", action="store_true")
    parser.add_argument("--max_len",      type=int, default=512)
    parser.add_argument("--topk",         type=int, default=10)
    # Explanation options
    parser.add_argument("--top_n_explain", type=int, default=10,
                        help="Generate explanations for top N journals")
    # Output
    parser.add_argument("--output_json", type=str, default="result.json")
    args = parser.parse_args()

    # Load all models once
    models = load_pipeline(
        checkpoint_path=args.checkpoint_path,
        model_name=args.model_name,
        data_path=args.data_path,
        aims_csv=args.aims_csv,
        journal_extract_jsonl=args.journal_extract_jsonl,
        encoder_model=args.encoder_model,
        qwen_model=args.qwen_model,
        pooler_type=args.pooler_type,
        features=args.features,
        max_len=args.max_len,
        use_aim=args.use_aim,
        use_category=args.use_category,
    )

    result = run_pipeline(
        title=args.title,
        abstract=args.abstract,
        keywords=args.keywords,
        models=models,
        topk=args.topk,
        top_n_explain=args.top_n_explain,
        paper_id=args.paper_id,
    )

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nResult saved → {args.output_json}")
    print("\nTop-5 journals after reranking:")
    for j in result["Top10_journals"][:5]:
        r = j["Rerank"]
        print(f"  #{r['new_rank']:2d}  {j['Name'][:55]:55s}  score={r['final_fit_score']:.1f}")
