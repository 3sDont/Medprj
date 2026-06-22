# -*- coding: utf-8 -*-
"""
llm_extract.py — Extract scientific features from a paper using Qwen (local),
                  look up pre-extracted journal features from journal_extract.jsonl,
                  and compute coverage metrics for each candidate journal.

Public API:
    QwenExtractor                       — wrapper for local Qwen model inference
    extract_paper_features(...)         -> dict  (scientific_domains, research_focuses, evidences)
    load_journal_extract(jsonl_path)    -> dict[label -> entry]
    compute_coverage_metrics(...)       -> dict  (7 metrics + missing_coverage)
    process_llm_extraction(...)         -> (paper_features, top_journals)
"""
import json
import re
from typing import Dict, List, Optional, Tuple

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
        # Use float32 on CPU for stability; float16 on CUDA for speed
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"  Qwen loaded on {self.device} ({dtype})")

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


# ── Coverage Helpers ──────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower())


def _keyword_match(term: str, target_text: str, min_word_len: int = 3) -> bool:
    """True if any significant word from term appears in target_text."""
    words = [w for w in _normalise(term).split() if len(w) >= min_word_len]
    target = _normalise(target_text)
    return bool(words) and any(w in target for w in words)


def _list_coverage(source_terms: List[str], target_text: str) -> float:
    """Fraction of source_terms keyword-matched in target_text."""
    if not source_terms:
        return 0.0
    hits = sum(1 for t in source_terms if _keyword_match(t, target_text))
    return round(hits / len(source_terms), 4)


def _list_overlap(list_a: List[str], list_b: List[str]) -> float:
    """Overlap coefficient |A∩B| / min(|A|,|B|) via fuzzy keyword matching."""
    if not list_a or not list_b:
        return 0.0
    combined_b = " ".join(list_b)
    hits = sum(1 for a in list_a if _keyword_match(a, combined_b))
    return round(hits / min(len(list_a), len(list_b)), 4)


def _to_text(value) -> str:
    """Accept str or list[str], return a single joined string."""
    if isinstance(value, list):
        return " ".join(value)
    return str(value) if value else ""


# ── Coverage Metrics ─────────────────────────────────────────────────────────

def compute_coverage_metrics(
    paper_features: dict,
    journal_entry: Optional[dict],
    journal_aims: str,
    journal_categories,
) -> dict:
    """
    Compute all coverage metrics for one (paper, journal) pair.

    Args:
        paper_features:     output of extract_paper_features()
                            {scientific_domains, scientific_domains_evidence,
                             research_focuses, research_focuses_evidence}
        journal_entry:      entry from journal_extract.jsonl, or None
        journal_aims:       Aims text from journal_full_info.csv
        journal_categories: str (comma-separated) or list[str]

    Returns dict with 7 metrics + missing_coverage list.
    """
    p_domains = paper_features.get("scientific_domains", [])
    p_focuses = paper_features.get("research_focuses", [])
    p_dom_evi: dict = paper_features.get("scientific_domains_evidence", {})

    # Flatten evidence phrases for evidence-based category coverage metric
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

    # 1. Paper scientific domains vs journal categories
    sci_dom_cat_cov = _list_coverage(p_domains, cat_text)

    # 2. Evidence phrases (from abstract) vs journal categories
    sci_dom_evi_cat_cov = (
        _list_coverage(p_dom_evi_phrases, cat_text)
        if p_dom_evi_phrases else sci_dom_cat_cov
    )

    # 3. Paper domains vs journal's extracted domains (JSONL)
    sci_dom_cov = _list_overlap(p_domains, j_domains)

    # 4. Paper domains present in journal aims/scope text
    sci_dom_aimscope = _list_coverage(p_domains, aims_text)

    # 5. Research focuses vs journal categories
    res_foc_cat_cov = _list_coverage(p_focuses, cat_text)

    # 6. Research focuses present in journal aims/scope text
    res_foc_aimscope = _list_coverage(p_focuses, aims_text)

    # 7. Missing: paper terms not found anywhere in the journal profile
    combined_journal = " ".join([aims_text, cat_text] + j_domains + j_focuses)
    missing = [
        t for t in (p_domains + p_focuses)
        if not _keyword_match(t, combined_journal)
    ]

    return {
        "scientific_domains_category_coverage":          sci_dom_cat_cov,
        "scientific_domains_evidence_category_coverage": sci_dom_evi_cat_cov,
        "scientific_domains_coverage":                   sci_dom_cov,
        "scientific_domains_aimscope":                   sci_dom_aimscope,
        "research_focuses_category_coverage":            res_foc_cat_cov,
        "research_focuses_coverage_aimscope":            res_foc_aimscope,
        "missing_coverage":                              missing,
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
) -> Tuple[dict, List[dict]]:
    """
    Extract paper features via Qwen and compute coverage metrics for all journals.

    Mutates each journal dict in top_journals by adding:
      - extracted_journal_features: {sci_evi: [...], research_evi: [...]}
      - coverage_metrics: {7 metric keys + missing_coverage}

    Returns:
        (paper_features, updated_top_journals)
        where paper_features = {scientific_domains, scientific_domains_evidence,
                                research_focuses, research_focuses_evidence}
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
