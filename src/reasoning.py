# -*- coding: utf-8 -*-
"""
reasoning.py — Rerank candidate journals by a weighted fit score
               and generate Vietnamese explanations using Qwen (local).

Public API:
    compute_final_score(journal)                              -> float  (0-100)
    rerank_journals(top_journals)                             -> List[dict]
    generate_explanation(paper_info, journal, extractor)     -> dict
    generate_all_explanations(paper_info, top_journals,
                              extractor, top_n)               -> List[dict]
"""
import json
import re
from typing import List

from llm_extract import QwenExtractor


# ── Scoring weights ───────────────────────────────────────────────────────────
# Must sum to 1.0
_WEIGHTS = {
    "base_score":                            0.25,
    "aims_scope_sim":                        0.20,
    "scientific_domains_coverage":           0.20,
    "scientific_domains_aimscope":           0.15,
    "scientific_domains_category_coverage":  0.10,
    "research_focuses_coverage_aimscope":    0.10,
}


def compute_final_score(journal: dict) -> float:
    """
    Weighted combination of classifier score, aims similarity, and coverage metrics.
    Returns a score in [0, 100].
    """
    cov = journal.get("coverage_metrics", {})
    score = (
        _WEIGHTS["base_score"]                             * journal.get("Base_Score", 0)          * 100
        + _WEIGHTS["aims_scope_sim"]                       * journal.get("Aims_Scope_Sim", 0)       * 100
        + _WEIGHTS["scientific_domains_coverage"]          * cov.get("scientific_domains_coverage", 0)          * 100
        + _WEIGHTS["scientific_domains_aimscope"]          * cov.get("scientific_domains_aimscope", 0)          * 100
        + _WEIGHTS["scientific_domains_category_coverage"] * cov.get("scientific_domains_category_coverage", 0) * 100
        + _WEIGHTS["research_focuses_coverage_aimscope"]   * cov.get("research_focuses_coverage_aimscope", 0)   * 100
    )
    return round(score, 2)


def rerank_journals(top_journals: List[dict]) -> List[dict]:
    """
    Compute final_fit_score for every journal, sort descending, assign new_rank.
    Adds a "Rerank" key to each dict: { final_fit_score, new_rank, rank_change }
    rank_change > 0 means the journal moved up relative to the classifier rank.
    Returns the sorted list (mutates in-place).
    """
    for j in top_journals:
        j.setdefault("Rerank", {})["final_fit_score"] = compute_final_score(j)

    top_journals.sort(key=lambda x: x["Rerank"]["final_fit_score"], reverse=True)

    for new_rank, j in enumerate(top_journals, start=1):
        old_rank = j.get("Rank", new_rank)
        j["Rerank"]["new_rank"]    = new_rank
        j["Rerank"]["rank_change"] = old_rank - new_rank  # positive = improved rank

    return top_journals


# ── LLM Explanation ───────────────────────────────────────────────────────────

_EXPLAIN_PROMPT = """\
You are an expert scientific publication advisor. Given the paper and journal information \
below, write a concise explanation for why this journal is (or is not) a good fit.

=== Paper Information ===
Title              : {title}
Scientific Domains : {sci_domains}
Research Focuses   : {research_focuses}

=== Journal Information ===
Name               : {journal_name}
New Rank           : #{new_rank}  (original #{old_rank}, change {rank_change:+d})
Fit Score          : {final_score:.1f}/100
Aims/Scope Sim.    : {aims_sim:.3f}
Domain Overlap     : {domain_cov:.0%}
Aims Coverage      : {aims_cov:.0%}
Missing Topics     : {missing}

Return JSON with exactly 3 fields (1–2 sentences each, in English):
  "main_reasoning"    : primary reason this journal fits (or does not fit) the paper
  "reranking_reasons" : reason the rank changed (or stayed) compared to the original rank
  "weakness_warning"  : key weakness or caveat (empty string "" if none)

Return only valid JSON, no additional text.\
"""


def _flatten_grouped(grouped: dict) -> List[str]:
    """Flatten {"1": [...], "2": [...]} → flat list preserving group order."""
    result = []
    for key in sorted(grouped.keys(), key=lambda k: int(k) if k.isdigit() else 0):
        items = grouped[key]
        result.extend(items if isinstance(items, list) else [items])
    return result


def generate_explanation(
    paper_info: dict,
    journal: dict,
    extractor: QwenExtractor,
) -> dict:
    """
    Generate a Vietnamese explanation dict for one journal recommendation using Qwen.

    Returns dict with keys: main_reasoning, reranking_reasons, weakness_warning.
    """
    features    = paper_info.get("extracted_features", {})
    cov         = journal.get("coverage_metrics", {})
    rerank      = journal.get("Rerank", {})
    rank_change = int(rerank.get("rank_change", 0))

    # Support both flat and grouped feature formats
    sci_domains_list = (
        _flatten_grouped(features.get("sci_evidence", {}))
        or features.get("scientific_domains", [])
    )
    research_list = (
        _flatten_grouped(features.get("research_evidence", {}))
        or features.get("research_focuses", [])
    )

    prompt = _EXPLAIN_PROMPT.format(
        title=paper_info.get("title", ""),
        sci_domains=", ".join(sci_domains_list) or "N/A",
        research_focuses=", ".join(research_list) or "N/A",
        journal_name=journal.get("Name", ""),
        new_rank=rerank.get("new_rank", "?"),
        old_rank=journal.get("Rank", "?"),
        rank_change=rank_change,
        final_score=rerank.get("final_fit_score", 0.0),
        aims_sim=journal.get("Aims_Scope_Sim", 0.0),
        domain_cov=cov.get("scientific_domains_coverage", 0.0),
        aims_cov=cov.get("scientific_domains_aimscope", 0.0),
        missing=", ".join(cov.get("missing_coverage", [])) or "không có",
    )

    raw = extractor.generate(
        messages=[{"role": "user", "content": prompt}],
        max_new_tokens=512,
    )

    # Strip thinking block and code fences
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {
            "main_reasoning":    raw[:300],
            "reranking_reasons": "",
            "weakness_warning":  "",
        }


def generate_all_explanations(
    paper_info: dict,
    top_journals: List[dict],
    extractor: QwenExtractor,
    top_n: int = 20,
) -> List[dict]:
    """Generate Vietnamese explanations for top_n journals (mutates each dict in-place)."""
    n = min(top_n, len(top_journals))
    for i, j in enumerate(top_journals[:n]):
        print(f"  Explanation {i + 1}/{n}: {j.get('Name', '')[:55]}")
        j["Explanation"] = generate_explanation(paper_info, j, extractor)
    return top_journals


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(
        description="Rerank journals and generate Vietnamese explanations via Qwen",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--inference_json", type=str, required=True,
                        help="JSON output from llm_extract.py")
    parser.add_argument("--qwen_model", type=str, default="Qwen/Qwen3.5-2B",
                        help="HuggingFace model ID for Qwen")
    parser.add_argument("--top_n",       type=int, default=20,
                        help="Generate explanations for top N journals only")
    parser.add_argument("--output_json", type=str, default="final_result.json")
    args = parser.parse_args()

    with open(args.inference_json, "r", encoding="utf-8") as f:
        data = _json.load(f)

    paper        = data.get("paper", {})
    top_journals = data["top_journals"]

    # Rerank
    print("Reranking journals by final fit score...")
    top_journals = rerank_journals(top_journals)

    # Generate explanations with Qwen
    extractor = QwenExtractor(model_name=args.qwen_model)
    print(f"Generating explanations for top {args.top_n} journals...")
    top_journals = generate_all_explanations(
        paper_info=paper,
        top_journals=top_journals,
        extractor=extractor,
        top_n=args.top_n,
    )

    data["top_journals"] = top_journals
    with open(args.output_json, "w", encoding="utf-8") as f:
        _json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved → {args.output_json}")
    print("\nTop-5 after reranking:")
    for j in top_journals[:5]:
        r = j["Rerank"]
        change = r.get("rank_change", 0)
        arrow  = f"↑{change}" if change > 0 else (f"↓{abs(change)}" if change < 0 else "—")
        print(f"  #{r['new_rank']:2d} ({arrow:>3})  {j['Name'][:55]:55s}  score={r['final_fit_score']:.1f}")
