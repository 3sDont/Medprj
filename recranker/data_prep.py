# -*- coding: utf-8 -*-
"""
recranker/data_prep.py
Build the natural-language dataset for RecRanker journal recommendation.

Inputs
------
- data/train_set.csv            : Paper_ID (row index) -> Title, Abstract, Keywords
- data/journal_full_info.csv    : Journal info (Label, Journal name, Aims, Scope, Categories)
- data/extracted_journals/extracted.jsonl : LLM-extracted research_focuses per journal
- data/subset_hard/{train,val,test}.csv   : hard cases with candidates (Paper_ID, Predicted_Journal_ID, ...)

Outputs (data/recranker_data/)
------
- {train,val,test}_nl.csv       : NL dataset for zero-shot inference / SFT training
- sft_train.jsonl               : SFT training format  (prompt + chosen)
- dpo_train.jsonl               : DPO training format  (prompt + chosen + rejected)
"""

import argparse
import json
import os
import random
import re

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_SET_CSV       = "data/train_set.csv"
JOURNAL_INFO_CSV    = "data/journal_full_info.csv"
JOURNAL_JSONL       = "data/extracted_journals/extracted.jsonl"
HARD_TRAIN_CSV      = "data/subset_hard/train.csv"
HARD_VAL_CSV        = "data/subset_hard/val.csv"
HARD_TEST_CSV       = "data/subset_hard/test.csv"
OUT_DIR             = "data/recranker_data"

# ─────────────────────────────────────────────────────────────────────────────
# Text helpers
# ─────────────────────────────────────────────────────────────────────────────
MAX_ABSTRACT_CHARS  = 600
MAX_AIMS_CHARS      = 300
MAX_KEYWORDS_CHARS  = 200


def _truncate(text: str, max_chars: int) -> str:
    text = str(text).strip() if pd.notna(text) else ""
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


# ─────────────────────────────────────────────────────────────────────────────
# Load resources
# ─────────────────────────────────────────────────────────────────────────────

def load_papers() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_SET_CSV)
    df.index.name = "Paper_ID"
    df = df.reset_index()          # Paper_ID becomes a column
    return df


def load_journals() -> dict:
    """Return {label_int: {name, aims, categories, research_focuses}}."""
    info = pd.read_csv(JOURNAL_INFO_CSV)

    # LLM-extracted research focuses
    focuses = {}
    with open(JOURNAL_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                label = int(obj.get("label", -1))
                rf = obj.get("research_focuses", [])
                focuses[label] = rf[:4] if isinstance(rf, list) else []
            except Exception:
                pass

    journals = {}
    for _, row in info.iterrows():
        label = int(row["Label"])
        aims  = _truncate(row.get("Aims", ""), MAX_AIMS_CHARS)
        cats  = str(row.get("Categories", row.get("Best Categories", ""))).strip()
        # clean bracket noise from categories
        cats  = re.sub(r"[\[\]']", "", cats)[:200]
        rf    = focuses.get(label, [])
        journals[label] = {
            "name"             : str(row["Journal"]).strip(),
            "aims"             : aims,
            "categories"       : cats,
            "research_focuses" : rf,
        }
    return journals


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an academic journal recommendation system for biomedical research. "
    "Based on a paper's content, rank the candidate journals from most to least "
    "suitable for publication."
)


def build_paper_text(paper_row: pd.Series) -> str:
    title    = _truncate(paper_row.get("Title", ""), 200)
    abstract = _truncate(paper_row.get("Abstract", ""), MAX_ABSTRACT_CHARS)
    keywords = _truncate(paper_row.get("Keywords", ""), MAX_KEYWORDS_CHARS)
    parts = [f"Title: {title}"]
    if abstract:
        parts.append(f"Abstract: {abstract}")
    if keywords and keywords.lower() not in ("nan", "none", ""):
        parts.append(f"Keywords: {keywords}")
    return "\n".join(parts)


def build_journal_text(label: int, journals: dict) -> str:
    j  = journals.get(label)
    if j is None:
        return f"Journal {label}: (no information available)"
    rf = "; ".join(j["research_focuses"]) if j["research_focuses"] else ""
    parts = [f"Journal {label}: {j['name']}"]
    if j["aims"]:
        parts.append(f"  Aims: {j['aims']}")
    if j["categories"]:
        parts.append(f"  Categories: {j['categories']}")
    if rf:
        parts.append(f"  Research focus: {rf}")
    return "\n".join(parts)


def build_prompt(paper_text: str, candidates: list[dict], journals: dict) -> str:
    """
    candidates: list of {'journal_id': int, 'text': str}  (already shuffled)
    """
    n = len(candidates)
    cand_block = "\n\n".join(
        f"[{i+1}] {c['text']}" for i, c in enumerate(candidates)
    )
    return (
        f"Paper to submit:\n{paper_text}\n\n"
        f"There are {n} candidate journals:\n{cand_block}\n\n"
        f"Rank these journals from most to least suitable for publication of this paper.\n"
        f"Think step by step about the match between the paper's topic and each journal's scope.\n\n"
        f"Strictly follow this output format (one line per rank):\n"
        + "\n".join(
            f"Rank {i+1}: Journal {{id}} - Reason: {{brief reason}}"
            for i in range(n)
        )
        + "\n\nYou MUST rank ALL given journals and cannot add journals not in the list.\n"
        "Begin with 'Rank 1:'."
    )


def build_chosen_response(candidates: list[dict], true_journal_id: int, journals: dict) -> str:
    """Build a positive (SFT/DPO chosen) response: correct journal ranked first."""
    ordered = [c for c in candidates if c["journal_id"] == true_journal_id]
    others  = [c for c in candidates if c["journal_id"] != true_journal_id]
    random.shuffle(others)
    ranked = ordered + others

    j = journals.get(true_journal_id, {})
    name = j.get("name", f"Journal {true_journal_id}") if j else f"Journal {true_journal_id}"

    lines = []
    lines.append(
        f"Rank 1: Journal {true_journal_id} - Reason: This journal ({name}) best matches "
        "the paper's topic and scope based on its stated aims and research focus."
    )
    for i, c in enumerate(others, start=2):
        jj = journals.get(c["journal_id"], {})
        jname = jj.get("name", f"Journal {c['journal_id']}") if jj else f"Journal {c['journal_id']}"
        lines.append(
            f"Rank {i}: Journal {c['journal_id']} - Reason: {jname} is ranked lower "
            "as its scope partially matches but is less aligned with the paper's specific topic."
        )
    return "\n".join(lines)


def build_rejected_response(candidates: list[dict], true_journal_id: int, journals: dict) -> str:
    """Build a negative (DPO rejected) response: a random wrong journal ranked first."""
    wrong = [c for c in candidates if c["journal_id"] != true_journal_id]
    if not wrong:
        return ""
    wrong_first = random.choice(wrong)
    rest = [c for c in candidates if c["journal_id"] != wrong_first["journal_id"]]
    random.shuffle(rest)
    ranked = [wrong_first] + rest

    lines = []
    for i, c in enumerate(ranked, start=1):
        jj = journals.get(c["journal_id"], {})
        jname = jj.get("name", f"Journal {c['journal_id']}") if jj else f"Journal {c['journal_id']}"
        lines.append(
            f"Rank {i}: Journal {c['journal_id']} - Reason: {jname} appears relevant "
            "to this research area."
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Build per-split NL dataset
# ─────────────────────────────────────────────────────────────────────────────

def build_nl_dataset(
    hard_csv: str,
    papers: pd.DataFrame,
    journals: dict,
    shuffle_candidates: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """One row per paper (all 20 candidates grouped together)."""
    rng  = random.Random(seed)
    hard = pd.read_csv(hard_csv)

    rows = []
    for paper_id, grp in hard.groupby("Paper_ID"):
        paper_idx = int(paper_id)
        if paper_idx >= len(papers):
            continue
        paper_row      = papers.iloc[paper_idx]
        paper_text     = build_paper_text(paper_row)
        correct_rows   = grp[grp["Is_Correct"] == 1]
        if correct_rows.empty:
            continue   # correct journal not in top-20 candidates, skip
        true_jid       = int(correct_rows["Predicted_Journal_ID"].iloc[0])
        candidate_jids = grp["Predicted_Journal_ID"].astype(int).tolist()

        cand_objects = [
            {"journal_id": jid, "text": build_journal_text(jid, journals)}
            for jid in candidate_jids
        ]
        if shuffle_candidates:
            rng.shuffle(cand_objects)

        prompt   = build_prompt(paper_text, cand_objects, journals)
        chosen   = build_chosen_response(cand_objects, true_jid, journals)
        rejected = build_rejected_response(cand_objects, true_jid, journals)

        rows.append({
            "paper_id"         : paper_id,
            "true_journal_id"  : true_jid,
            "candidate_ids"    : json.dumps([c["journal_id"] for c in cand_objects]),
            "paper_text"       : paper_text,
            "prompt"           : prompt,
            "chosen"           : chosen,
            "rejected"         : rejected,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Export helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_sft_jsonl(df: pd.DataFrame, path: str, system_prompt: str = SYSTEM_PROMPT):
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            obj = {
                "messages": [
                    {"role": "system",    "content": system_prompt},
                    {"role": "user",      "content": row["prompt"]},
                    {"role": "assistant", "content": row["chosen"]},
                ]
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"  Saved SFT JSONL -> {path}  ({len(df)} examples)")


def save_dpo_jsonl(df: pd.DataFrame, path: str, system_prompt: str = SYSTEM_PROMPT):
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            if not row["rejected"]:
                continue
            obj = {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": row["prompt"]},
                ],
                "chosen"  : [{"role": "assistant", "content": row["chosen"]}],
                "rejected": [{"role": "assistant", "content": row["rejected"]}],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"  Saved DPO JSONL  -> {path}  ({len(df)} examples)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare RecRanker NL dataset")
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[1/3] Loading papers and journals ...")
    papers   = load_papers()
    journals = load_journals()
    print(f"  Papers  : {len(papers):,}")
    print(f"  Journals: {len(journals):,}")

    for split, csv_path in [
        ("train", HARD_TRAIN_CSV),
        ("val",   HARD_VAL_CSV),
        ("test",  HARD_TEST_CSV),
    ]:
        print(f"\n[{split}] Building NL dataset from {csv_path} ...")
        df = build_nl_dataset(csv_path, papers, journals, seed=args.seed)
        print(f"  Papers: {len(df):,}")

        nl_path = os.path.join(args.out_dir, f"{split}_nl.csv")
        df.to_csv(nl_path, index=False, encoding="utf-8")
        print(f"  Saved NL CSV -> {nl_path}")

        if split == "train":
            save_sft_jsonl(df, os.path.join(args.out_dir, "sft_train.jsonl"))
            save_dpo_jsonl(df, os.path.join(args.out_dir, "dpo_train.jsonl"))
        elif split == "val":
            save_sft_jsonl(df, os.path.join(args.out_dir, "sft_val.jsonl"))

    print("\nDone.")


if __name__ == "__main__":
    main()
