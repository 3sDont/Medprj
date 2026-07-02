# -*- coding: utf-8 -*-
"""
reasoning.py — Rerank candidate journals by a trained LTR fit score and
                generate explanations using Qwen (local).

final_fit_score comes from a trained logistic-regression model (see
load_ltr_model / models/student_model.json) applied to the 8 features
produced earlier in the pipeline: Base_Score, Inverse_Base_Rank,
Aims_Scope_Sim, and the 5 coverage_metrics alignment scores. rerank_journals
also stores each journal's per-feature contribution to that score (see
Rerank.feature_contributions), sorted by impact, for UI display. If no
model is supplied, compute_final_score falls back to raw Base_Score.

generate_explanation is a scores-only experiment: the LLM sees the journal
name plus the same numeric signals (with a legend describing what each one
measures) but no paper/journal text content — it must interpret the score
meanings itself rather than reason over actual topical content.

Public API:
    load_ltr_model(path)                                      -> dict
    compute_final_score(journal, ltr_model=None)               -> float  (0-100)
    rerank_journals(top_journals, ltr_model=None)               -> List[dict]
    generate_explanation(paper_info, journal, extractor)     -> dict
    generate_all_explanations(paper_info, top_journals,
                              extractor, top_n)               -> List[dict]
"""
import json
import math
import os
import re
from typing import List, Optional

from llm_extract import QwenExtractor


# ── Trained LTR model ─────────────────────────────────────────────────────────
# Maps each feature name in the model file to how it's read off a journal dict.
_FEATURE_EXTRACTORS = {
    "Based_score":  lambda j: j.get("Base_Score", 0.0),
    "inverse_base_rank": lambda j: j.get("Inverse_Base_Rank", 0.0),
    "Aims_Scope_Sim": lambda j: j.get("Aims_Scope_Sim", 0.0),
    "Scientific_domain_profile_category_alignment":
        lambda j: j.get("coverage_metrics", {}).get("Scientific_domain_profile_category_alignment", 0.0),
    "Scientific_domain_profile_AimScope_alignment":
        lambda j: j.get("coverage_metrics", {}).get("Scientific_domain_profile_AimScope_alignment", 0.0),
    "abstract_category_alignment":
        lambda j: j.get("coverage_metrics", {}).get("abstract_category_alignment", 0.0),
    "research_focuses_profile_aimscope_alignment":
        lambda j: j.get("coverage_metrics", {}).get("research_focuses_profile_aimscope_alignment", 0.0),
    "research_focuses_profile_category_alignment":
        lambda j: j.get("coverage_metrics", {}).get("research_focuses_profile_category_alignment", 0.0),
}


def load_ltr_model(path: str) -> dict:
    """
    Load a trained LTR model: standardized logistic-regression coefficients
    fit on submission_ltr_dataset features. Expects keys: feature_names,
    weights, bias, feature_mean, feature_std (all same length except bias).
    """
    with open(path, "r", encoding="utf-8") as f:
        model = json.load(f)

    required = {"feature_names", "weights", "bias", "feature_mean", "feature_std"}
    missing = required - model.keys()
    if missing:
        raise ValueError(f"LTR model file {path!r} missing keys: {sorted(missing)}")
    unknown = set(model["feature_names"]) - _FEATURE_EXTRACTORS.keys()
    if unknown:
        raise ValueError(f"LTR model file {path!r} has unrecognized features: {sorted(unknown)}")

    return model


# Temperature for the logit -> score squashing (sigmoid(z / T)). T=1 is the
# model's native calibration, which saturates fast: once z exceeds ~3, every
# journal lands in the 95-100 band regardless of how much better one logit is
# than another (e.g. z=2.95 -> 95.0 but z=7.0 -> 99.9, barely distinguishable
# on a 0-100 scale). T>1 stretches the curve so good-but-not-best candidates
# spread out more, at the cost of the score no longer being the model's
# literal trained probability. Tune this constant to taste; ranking order and
# each feature's contribution/share_pct are unaffected either way.
_LOGIT_TEMPERATURE = 3.0


def _sigmoid(z: float, temperature: float = _LOGIT_TEMPERATURE) -> float:
    return 1.0 / (1.0 + math.exp(-z / temperature))


def _logit_and_contributions(journal: dict, ltr_model: dict):
    """
    Standardize each of the model's features and multiply by its learned
    weight. Returns (logit, contributions) where contributions is a list of
    {feature, raw, contribution, share_pct} — one entry per feature, in the
    model's own order. share_pct is the feature's share of the *total
    explanatory magnitude* (sum of |contribution|), signed to show whether
    it pushed the score up or down; |share_pct| values sum to 100.
    """
    contributions = []
    z = ltr_model["bias"]
    for name, weight, mean, std in zip(
        ltr_model["feature_names"], ltr_model["weights"],
        ltr_model["feature_mean"], ltr_model["feature_std"],
    ):
        x = _FEATURE_EXTRACTORS[name](journal)
        standardized = (x - mean) / std if std else 0.0
        contribution = weight * standardized
        z += contribution
        contributions.append({"feature": name, "raw": x, "contribution": contribution})

    total_abs = sum(abs(c["contribution"]) for c in contributions) or 1.0
    for c in contributions:
        c["share_pct"] = round(c["contribution"] / total_abs * 100, 1)

    return z, contributions


def compute_final_score(journal: dict, ltr_model: Optional[dict] = None) -> float:
    """
    Fit score in [0, 100]. With a trained ltr_model: standardize the 8
    pipeline features with the model's mean/std, take the weighted sum +
    bias, and squash with sigmoid. Without one: falls back to raw
    classifier confidence (Base_Score).
    """
    if ltr_model is None:
        return round(journal.get("Base_Score", 0) * 100, 2)

    z, _ = _logit_and_contributions(journal, ltr_model)
    return round(_sigmoid(z) * 100, 2)


def rerank_journals(top_journals: List[dict], ltr_model: Optional[dict] = None) -> List[dict]:
    """
    Compute final_fit_score for every journal, sort descending, assign new_rank.
    Adds a "Rerank" key to each dict:
      { final_fit_score, new_rank, rank_change, feature_contributions }
    feature_contributions is sorted by |contribution| descending (biggest
    driver of that journal's score first). rank_change > 0 means the journal
    moved up relative to the classifier rank. Returns the sorted list
    (mutates in-place).
    """
    for j in top_journals:
        rerank = j.setdefault("Rerank", {})
        if ltr_model is not None:
            z, contributions = _logit_and_contributions(j, ltr_model)
            contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)
            rerank["final_fit_score"] = round(_sigmoid(z) * 100, 2)
            rerank["feature_contributions"] = contributions
        else:
            rerank["final_fit_score"] = compute_final_score(j, None)
            rerank["feature_contributions"] = [{
                "feature": "Based_score",
                "raw": j.get("Base_Score", 0.0),
                "contribution": j.get("Base_Score", 0.0),
                "share_pct": 100.0,
            }]

    top_journals.sort(key=lambda x: x["Rerank"]["final_fit_score"], reverse=True)

    for new_rank, j in enumerate(top_journals, start=1):
        old_rank = j.get("Rank", new_rank)
        j["Rerank"]["new_rank"]    = new_rank
        j["Rerank"]["rank_change"] = old_rank - new_rank  # positive = improved rank

    return top_journals


# ── LLM Explanation ───────────────────────────────────────────────────────────

_EXPLAIN_PROMPT = """\
You are a publication advisor. You are NOT given the paper's or journal's actual \
text content — only the journal's name and numeric match scores (0 to 1, higher \
= stronger match, except Fit Score which is 0-100). Explain the recommendation by \
interpreting what these scores imply, using the legend below to know what each one \
measures. Write in plain language for a researcher unfamiliar with ML scoring — do \
not print metric names or raw numbers in your output; instead describe what a score \
in that range means (e.g. a high "Aims/Scope Similarity" becomes "the journal's \
overall scope closely matches your paper's topic").

=== Score legend ===
Fit Score                         : overall predicted match, combining every signal below
Rank quality among candidates     : how highly the initial classifier ranked this journal
                                     versus the other candidates it considered (1.0 = its
                                     top pick, near 0 = the weakest candidate considered)
Aims/Scope Similarity             : how closely the journal's stated aims & scope match
                                     the paper overall
Sci. Domain Profile → Categories  : how well the paper's scientific domain(s) match this
                                     journal's subject categories
Sci. Domain Profile → Aims/Scope  : how well the paper's scientific domain(s) match the
                                     journal's aims & scope description
Abstract → Categories             : how well the paper's abstract matches this journal's
                                     subject categories
Research Focuses → Aims/Scope     : how well the paper's specific research focuses match
                                     the journal's aims & scope
Research Focuses → Categories     : how well the paper's specific research focuses match
                                     this journal's subject categories

=== Journal: {journal_name} (Rank #{new_rank} of the candidates found) ===
Fit Score                         : {final_score:.1f}/100
Rank quality among candidates     : {inv_rank:.4f}
Aims/Scope Similarity             : {aims_sim:.4f}
Sci. Domain Profile → Categories  : {sci_dom_cat:.4f}
Sci. Domain Profile → Aims/Scope  : {sci_dom_aims:.4f}
Abstract → Categories             : {abs_cat:.4f}
Research Focuses → Aims/Scope     : {res_foc_aims:.4f}
Research Focuses → Categories     : {res_foc_cat:.4f}

Return JSON with exactly 2 fields, in English:
  "main_reasoning"   : 2-3 sentences. Open with a one-line verdict ("This is a strong/
                       moderate/weak fit because...") then explain, in plain language,
                       what the 1-2 highest scores mean for this match — without naming
                       the metric or printing its number.
  "weakness_warning" : 1-2 plain-language sentences interpreting the single lowest score
                       as a concern, or "" if every score is reasonably strong.

Return only valid JSON, no additional text.\
"""


def generate_explanation(
    paper_info: dict,
    journal: dict,
    extractor: QwenExtractor,
) -> dict:
    """
    Generate an explanation dict for one journal recommendation using Qwen.

    Experiment: the LLM only sees the journal name + numeric scores (with a
    legend explaining what each score measures) — no paper/journal text
    content — so paper_info's extracted_features are intentionally unused
    here. Returns dict with keys: main_reasoning, weakness_warning.
    """
    cov    = journal.get("coverage_metrics", {})
    rerank = journal.get("Rerank", {})

    prompt = _EXPLAIN_PROMPT.format(
        journal_name=journal.get("Name", ""),
        new_rank=rerank.get("new_rank", "?"),
        final_score=rerank.get("final_fit_score", 0.0),
        inv_rank=journal.get("Inverse_Base_Rank", 0.0),
        aims_sim=journal.get("Aims_Scope_Sim", 0.0),
        sci_dom_cat=cov.get("Scientific_domain_profile_category_alignment", 0.0),
        sci_dom_aims=cov.get("Scientific_domain_profile_AimScope_alignment", 0.0),
        abs_cat=cov.get("abstract_category_alignment", 0.0),
        res_foc_aims=cov.get("research_focuses_profile_aimscope_alignment", 0.0),
        res_foc_cat=cov.get("research_focuses_profile_category_alignment", 0.0),
    )

    raw = extractor.generate(
        messages=[{"role": "user", "content": prompt}],
        max_new_tokens=512,
    )

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
            "main_reasoning":  raw[:300],
            "weakness_warning": "",
        }


def generate_all_explanations(
    paper_info: dict,
    top_journals: List[dict],
    extractor: QwenExtractor,
    top_n: int = 20,
) -> List[dict]:
    """Generate explanations for top_n journals (mutates each dict in-place)."""
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
        description="Rerank journals and generate explanations via Qwen",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--inference_json", type=str, required=True,
                        help="JSON output from llm_extract.py")
    parser.add_argument("--qwen_model", type=str, default="Qwen/Qwen3.5-2B",
                        help="HuggingFace model ID for Qwen")
    parser.add_argument("--ltr_model_path", type=str, default="models/student_model.json",
                        help="Trained LTR model JSON (falls back to raw Base_Score if not found)")
    parser.add_argument("--top_n",       type=int, default=20,
                        help="Generate explanations for top N journals only")
    parser.add_argument("--output_json", type=str, default="final_result.json")
    args = parser.parse_args()

    with open(args.inference_json, "r", encoding="utf-8") as f:
        data = _json.load(f)

    paper        = data.get("paper", {})
    top_journals = data["top_journals"]

    ltr_model = None
    if args.ltr_model_path and os.path.exists(args.ltr_model_path):
        print(f"Loading LTR model from {args.ltr_model_path} ...")
        ltr_model = load_ltr_model(args.ltr_model_path)
    else:
        print("No LTR model found — falling back to raw Base_Score as fit score.")

    # Rerank
    print("Reranking journals by final fit score...")
    top_journals = rerank_journals(top_journals, ltr_model)

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
