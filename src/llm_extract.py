# -*- coding: utf-8 -*-
"""
llm_extract.py — Extract scientific features from a paper using Qwen (local),
                  look up pre-extracted journal features from journal_extract.jsonl,
                  and compute coverage metrics for each candidate journal.

Public API:
    QwenExtractor                       — wrapper for local Qwen model inference
    extract_paper_features(...)         -> dict  (scientific_domains, research_focuses, evidences)
    load_journal_extract(jsonl_path)    -> dict[label -> entry]
    compute_coverage_metrics(...)       -> dict  (6 metrics)
    process_llm_extraction(...)         -> (paper_features, top_journals)
"""
import json
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from prompt_builder_paper import build_prompt_paper, build_prompt_paper_no_system


# ── Qwen Extractor ────────────────────────────────────────────────────────────

class QwenExtractor:
    """
    Local Qwen model wrapper for JSON extraction tasks.
    Load once, call generate() per request.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-2B",
        device: Optional[str] = None,
    ):
        print(f"Loading Qwen extractor: {model_name} ...")
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # bfloat16 on CUDA: numerically stable + supported by RTX 3060
        # float32 on CPU: accuracy without CUDA float16 underflow risk
        dtype       = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        device_map  = self.device  # explicit device; "auto" would split across CPU+GPU

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            attn_implementation="eager",   # suppress flash-linear-attention warning
        )
        self.model.eval()
        vram = ""
        if self.device.startswith("cuda"):
            used = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            vram = f"  VRAM: {used:.1f}/{total:.1f} GB"
        print(f"  Qwen loaded on {self.device} ({dtype}){vram}")

    def generate(
        self,
        messages: List[dict],
        max_new_tokens: int = 1024,
    ) -> str:
        """
        Run chat-style inference and return the raw text of the new tokens.
        Handles Qwen3 enable_thinking flag gracefully.
        """
        try:
            # Qwen3 supports enable_thinking — disable for pure JSON output
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs.input_ids.shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── LLM Extraction ───────────────────────────────────────────────────────────

def extract_paper_features(
    title: str,
    abstract: str,
    keywords: str,
    extractor: QwenExtractor,
    use_system_role: bool = True,
) -> dict:
    """
    Use Qwen to extract scientific domains and research focuses from a paper.

    Returns:
        {
          "scientific_domains": [...],
          "scientific_domains_evidence": {domain: [phrase, ...], ...},
          "research_focuses": [...],
          "research_focuses_evidence": {focus: [phrase, ...], ...}
        }
    """
    if use_system_role:
        messages = build_prompt_paper(title=title, keywords=keywords, abstract=abstract)
    else:
        messages = build_prompt_paper_no_system(title=title, keywords=keywords, abstract=abstract)

    raw = extractor.generate(messages)

    # Strip Qwen3 thinking block if present
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # Strip markdown code fences
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Last resort: extract the first {...} block
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


# ── Journal Feature Lookup ───────────────────────────────────────────────────

def load_journal_extract(jsonl_path: str) -> Dict[str, dict]:
    """
    Load journal_extract.jsonl into a dict keyed by label (str).

    Each entry has: journal, label, categories, scientific_domains,
                    scientific_domains_evidence, research_focuses, research_focuses_evidence
    """
    extracts: Dict[str, dict] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            extracts[str(entry["label"])] = entry
    return extracts


def _to_text(value) -> str:
    """Accept str or list[str], return a single joined string."""
    if isinstance(value, list):
        return " ".join(value)
    return str(value) if value else ""


# ── Embedding-based similarity helpers ───────────────────────────────────────

def _encode(texts: List[str], encoder) -> np.ndarray:
    """Return L2-normalised embeddings, shape (n, hidden_dim)."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    vecs = encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vecs / norms).astype(np.float32)


def _best_match_mean(a: np.ndarray, b: np.ndarray) -> float:
    """For each row in a, find max cosine sim to any row in b; return mean."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return 0.0
    return float((a @ b.T).max(axis=1).mean())


def _mean_sim_to_vec(a: np.ndarray, v: Optional[np.ndarray]) -> float:
    """Mean cosine similarity of each row in a to a single vector v."""
    if a.shape[0] == 0 or v is None:
        return 0.0
    return float((a @ v).mean())


# ── Coverage Metrics ─────────────────────────────────────────────────────────

def compute_coverage_metrics(
    paper_features: dict,
    journal_entry: Optional[dict],
    journal_aims: str,
    journal_categories,
    encoder=None,
) -> dict:
    """
    Compute coverage metrics for one (paper, journal) pair.

    When encoder is provided (SentenceTransformer), all numeric metrics use
    cosine similarity on dense embeddings.  Falls back to keyword matching
    when encoder is None.

    Returns dict with 6 float metrics + missing_coverage list.
    """
    p_domains = paper_features.get("scientific_domains", [])
    p_focuses = paper_features.get("research_focuses", [])
    p_dom_evi: dict = paper_features.get("scientific_domains_evidence", {})

    p_dom_evi_phrases: List[str] = [
        phrase
        for phrases in p_dom_evi.values()
        for phrase in (phrases if isinstance(phrases, list) else [phrases])
    ]

    aims_text = journal_aims or ""
    cat_text  = _to_text(journal_categories)

    j_domains: List[str] = []
    j_focuses: List[str] = []
    if journal_entry:
        j_domains = journal_entry.get("scientific_domains", [])
        j_focuses = journal_entry.get("research_focuses", [])

    if encoder is not None:
        cats = (
            journal_categories if isinstance(journal_categories, list)
            else [c.strip() for c in cat_text.split(",") if c.strip()]
        )
        domain_vecs   = _encode(p_domains, encoder)
        focus_vecs    = _encode(p_focuses, encoder)
        evi_vecs      = _encode(p_dom_evi_phrases, encoder)
        cat_vecs      = _encode(cats, encoder)
        j_domain_vecs = _encode(j_domains, encoder)
        aims_vec      = _encode([aims_text], encoder)[0] if aims_text else None

        sci_dom_cat_cov     = round(_best_match_mean(domain_vecs, cat_vecs), 4)
        sci_dom_evi_cat_cov = round(
            _best_match_mean(evi_vecs, cat_vecs) if evi_vecs.shape[0] else sci_dom_cat_cov, 4
        )
        sci_dom_cov         = round(
            _best_match_mean(domain_vecs, j_domain_vecs)
            if j_domain_vecs.shape[0] else _mean_sim_to_vec(domain_vecs, aims_vec),
            4,
        )
        sci_dom_aimscope    = round(_mean_sim_to_vec(domain_vecs, aims_vec), 4)
        res_foc_cat_cov     = round(_best_match_mean(focus_vecs, cat_vecs), 4)
        res_foc_aimscope    = round(_mean_sim_to_vec(focus_vecs, aims_vec), 4)
    else:
        sci_dom_cat_cov     = _list_coverage(p_domains, cat_text)
        sci_dom_evi_cat_cov = (
            _list_coverage(p_dom_evi_phrases, cat_text) if p_dom_evi_phrases else sci_dom_cat_cov
        )
        sci_dom_cov         = _list_overlap(p_domains, j_domains)
        sci_dom_aimscope    = _list_coverage(p_domains, aims_text)
        res_foc_cat_cov     = _list_coverage(p_focuses, cat_text)
        res_foc_aimscope    = _list_coverage(p_focuses, aims_text)

    return {
        "scientific_domains_category_coverage":          sci_dom_cat_cov,
        "scientific_domains_evidence_category_coverage": sci_dom_evi_cat_cov,
        "scientific_domains_coverage":                   sci_dom_cov,
        "scientific_domains_aimscope":                   sci_dom_aimscope,
        "research_focuses_category_coverage":            res_foc_cat_cov,
        "research_focuses_coverage_aimscope":            res_foc_aimscope,
    }


# ── Main Process Function ─────────────────────────────────────────────────────

def process_llm_extraction(
    title: str,
    abstract: str,
    keywords: str,
    top_journals: List[dict],
    journal_extracts: Dict[str, dict],
    extractor: QwenExtractor,
    use_system_role: bool = True,
    encoder=None,
) -> Tuple[dict, List[dict]]:
    """
    Extract paper features via Qwen and compute coverage metrics for all journals.

    Mutates each journal dict in top_journals by adding:
      - extracted_journal_features: {sci_evi: [...], research_evi: [...]}
      - coverage_metrics: {6 float metrics + missing_coverage list}

    Pass encoder (SentenceTransformer) to use embedding-based similarity;
    omit to fall back to keyword matching.

    Returns:
        (paper_features, updated_top_journals)
    """
    paper_features = extract_paper_features(
        title, abstract, keywords, extractor, use_system_role
    )

    for j in top_journals:
        label = str(j.get("Label", ""))
        journal_entry = journal_extracts.get(label)

        j_domains = journal_entry.get("scientific_domains", []) if journal_entry else []
        j_focuses = journal_entry.get("research_focuses", [])   if journal_entry else []

        j["extracted_journal_features"] = {
            "sci_evi":      j_domains,
            "research_evi": j_focuses,
        }
        j["coverage_metrics"] = compute_coverage_metrics(
            paper_features=paper_features,
            journal_entry=journal_entry,
            journal_aims=j.get("Aims", ""),
            journal_categories=j.get("Categories", ""),
            encoder=encoder,
        )

    return paper_features, top_journals


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Qwen-based feature extraction + coverage metrics for top journals",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--title",    type=str, default="")
    parser.add_argument("--abstract", type=str, default="")
    parser.add_argument("--keywords", type=str, default="")
    parser.add_argument("--inference_json", type=str, required=True,
                        help="JSON output from aims_scope_sim.py")
    parser.add_argument("--journal_extract_jsonl", type=str, required=True,
                        help="Path to journal_extract.jsonl")
    parser.add_argument("--qwen_model", type=str, default="Qwen/Qwen3.5-2B",
                        help="HuggingFace model ID for Qwen extractor")
    parser.add_argument("--no_system_role", action="store_true",
                        help="Merge system prompt into user message (for models without system role support)")
    parser.add_argument("--output_json", type=str, default="inference_with_coverage.json")
    args = parser.parse_args()

    with open(args.inference_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    paper    = data.get("paper", {})
    title    = args.title    or paper.get("title", "")
    abstract = args.abstract or paper.get("abstract", "")
    keywords = args.keywords or paper.get("keywords", "")
    top_journals = data["top_journals"]

    journal_extracts = load_journal_extract(args.journal_extract_jsonl)
    print(f"Loaded {len(journal_extracts)} journal extract entries")

    extractor = QwenExtractor(model_name=args.qwen_model)

    print("Extracting paper features via Qwen...")
    paper_features, top_journals = process_llm_extraction(
        title=title, abstract=abstract, keywords=keywords,
        top_journals=top_journals,
        journal_extracts=journal_extracts,
        extractor=extractor,
        use_system_role=not args.no_system_role,
    )

    data["paper"]["extracted_features"] = paper_features
    data["top_journals"] = top_journals

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved → {args.output_json}")
    print(f"scientific_domains: {paper_features.get('scientific_domains', [])}")
    print(f"research_focuses  : {paper_features.get('research_focuses', [])}")
