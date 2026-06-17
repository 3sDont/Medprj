# -*- coding: utf-8 -*-
"""
recranker/inference.py
Zero-shot LLM reranking for journal recommendation.

Free options (no Anthropic key needed):
  Ollama (local):
    python -m recranker.inference --provider ollama --model qwen2.5:7b --n_papers 50

  Groq (cloud free, get key at console.groq.com):
    python -m recranker.inference --provider groq --model llama-3.3-70b-versatile

  Gemini (cloud free, get key at aistudio.google.com):
    python -m recranker.inference --provider gemini --model gemini-1.5-flash

  Claude (paid):
    python -m recranker.inference --provider claude --model claude-haiku-4-5-20251001

All commands use:
    --input  data/recranker_data/test_nl.csv
    --output data/recranker_data/test_results.csv
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# LLM client wrappers
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an academic journal recommendation system for biomedical research. "
    "Based on a paper's content, rank the candidate journals from most to least "
    "suitable for publication."
)


def call_ollama(prompt: str, model: str, max_tokens: int = 1024) -> str:
    """Ollama local server — free, no API key needed."""
    from openai import OpenAI
    client = OpenAI(
        api_key="ollama",                          # any non-empty string works
        base_url="http://localhost:11434/v1",
    )
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )
    return resp.choices[0].message.content


def call_groq(prompt: str, model: str, max_tokens: int = 1024) -> str:
    """Groq cloud — free tier at console.groq.com. Set GROQ_API_KEY env var."""
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ.get("GROQ_API_KEY", ""),
        base_url="https://api.groq.com/openai/v1",
    )
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )
    return resp.choices[0].message.content


def call_gemini(prompt: str, model: str, max_tokens: int = 1024) -> str:
    """Google Gemini — free tier at aistudio.google.com. Set GEMINI_API_KEY env var."""
    import google.generativeai as genai
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
    m = genai.GenerativeModel(
        model_name=model,
        system_instruction=SYSTEM_PROMPT,
    )
    resp = m.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(max_output_tokens=max_tokens, temperature=0.0),
    )
    return resp.text


def call_claude(prompt: str, model: str, max_tokens: int = 1024) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def call_openai(prompt: str, model: str, base_url: str | None,
                max_tokens: int = 1024) -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "sk-xxx"),
        base_url=base_url,
    )
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )
    return resp.choices[0].message.content


def call_litellm(prompt: str, model: str, max_tokens: int = 1024) -> str:
    import litellm
    resp = litellm.completion(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )
    return resp.choices[0].message.content


def get_llm_fn(provider: str, model: str, base_url: str | None):
    if provider == "ollama":
        return lambda p: call_ollama(p, model)
    if provider == "groq":
        return lambda p: call_groq(p, model)
    if provider == "gemini":
        return lambda p: call_gemini(p, model)
    if provider == "claude":
        return lambda p: call_claude(p, model)
    if provider == "openai":
        return lambda p: call_openai(p, model, base_url)
    if provider == "litellm":
        return lambda p: call_litellm(p, model)
    raise ValueError(f"Unknown provider: {provider}")


# ─────────────────────────────────────────────────────────────────────────────
# Output parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_ranking(response: str, candidate_ids: list[int]) -> list[int]:
    """
    Extract journal IDs in ranked order from LLM response.
    Expected format:  Rank 1: Journal {id} - Reason: ...
    Falls back to scanning for any mention of candidate IDs if strict parse fails.
    """
    ranked = []
    seen   = set()
    cand_set = set(candidate_ids)

    # Strict: match 'Rank N: Journal <id>'
    for m in re.finditer(
        r"Rank\s*\d+\s*:\s*Journal\s+(\d+)", response, re.IGNORECASE
    ):
        jid = int(m.group(1))
        if jid in cand_set and jid not in seen:
            ranked.append(jid)
            seen.add(jid)

    # Fallback: scan for candidate IDs in order of appearance
    if not ranked:
        for m in re.finditer(r"\bJournal\s+(\d+)\b", response, re.IGNORECASE):
            jid = int(m.group(1))
            if jid in cand_set and jid not in seen:
                ranked.append(jid)
                seen.add(jid)

    # Append any candidates not mentioned (preserve original order)
    for jid in candidate_ids:
        if jid not in seen:
            ranked.append(jid)

    return ranked


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _ndcg_at_k(ranked: list[int], true_id: int, k: int) -> float:
    for i, jid in enumerate(ranked[:k]):
        if jid == true_id:
            import math
            return 1.0 / math.log2(i + 2)
    return 0.0


def compute_metrics(results_df: pd.DataFrame, ks=(1, 3, 5, 10)) -> dict:
    mrr  = 0.0
    ndcg = {k: 0.0 for k in ks}
    acc  = {k: 0   for k in ks}
    n    = 0

    for _, row in results_df.iterrows():
        true_id = int(row["true_journal_id"])
        ranked  = json.loads(row["llm_ranked_ids"])
        if not ranked:
            continue
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
    out = {"MRR": mrr / n, "n_papers": n}
    for k in ks:
        out[f"NDCG@{k}"] = ndcg[k] / n
        out[f"Acc@{k}"]  = acc[k]  / n
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main inference loop
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    input_csv   : str,
    output_csv  : str,
    llm_fn,
    n_papers    : int | None = None,
    delay_sec   : float      = 0.5,
    resume      : bool       = True,
):
    df = pd.read_csv(input_csv)
    if n_papers:
        df = df.head(n_papers)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint
    done = set()
    if resume and out_path.exists():
        prev = pd.read_csv(out_path)
        done = set(prev["paper_id"].tolist())
        print(f"  Resuming — {len(done)} papers already done.")

    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        pid = row["paper_id"]
        if pid in done:
            continue

        candidate_ids = json.loads(row["candidate_ids"])
        prompt        = row["prompt"]

        try:
            response = llm_fn(prompt)
            ranked   = parse_ranking(response, candidate_ids)
            parse_ok = True
        except Exception as e:
            print(f"  [paper {pid}] ERROR: {e}")
            response = ""
            ranked   = candidate_ids  # fall back to original order
            parse_ok = False

        results.append({
            "paper_id"        : pid,
            "true_journal_id" : row["true_journal_id"],
            "candidate_ids"   : row["candidate_ids"],
            "llm_ranked_ids"  : json.dumps(ranked),
            "llm_response"    : response,
            "parse_ok"        : parse_ok,
        })

        # Save checkpoint every 10 papers
        if len(results) % 10 == 0:
            _append_results(results, out_path, done)
            print(f"  [{i+1}/{len(df)}] Saved checkpoint. "
                  f"Top-1 so far: {_quick_acc1(results):.1%}")
            results = []

        if delay_sec > 0:
            time.sleep(delay_sec)

    if results:
        _append_results(results, out_path, done)

    print(f"\nInference complete. Results saved -> {output_csv}")
    return pd.read_csv(output_csv)


def _append_results(results: list, path: Path, done: set):
    new_df = pd.DataFrame(results)
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["paper_id"])
    else:
        combined = new_df
    combined.to_csv(path, index=False, encoding="utf-8")
    done.update(r["paper_id"] for r in results)


def _quick_acc1(results: list) -> float:
    correct = sum(
        1 for r in results
        if r["parse_ok"] and json.loads(r["llm_ranked_ids"])[0] == r["true_journal_id"]
    )
    return correct / len(results) if results else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RecRanker zero-shot inference")
    parser.add_argument("--input",    required=True,
                        help="NL dataset CSV (output of data_prep.py)")
    parser.add_argument("--output",   required=True,
                        help="Output CSV with LLM rankings")
    parser.add_argument("--provider", default="ollama",
                        choices=["ollama", "groq", "gemini", "claude", "openai", "litellm"],
                        help=(
                            "ollama=local free | groq=cloud free (GROQ_API_KEY) | "
                            "gemini=cloud free (GEMINI_API_KEY) | claude=paid (ANTHROPIC_API_KEY)"
                        ))
    parser.add_argument("--model",    default="qwen2.5:7b",
                        help=(
                            "Model ID. Defaults per provider: "
                            "ollama->qwen2.5:7b | groq->llama-3.3-70b-versatile | "
                            "gemini->gemini-1.5-flash | claude->claude-haiku-4-5-20251001"
                        ))
    parser.add_argument("--base_url", default=None,
                        help="Base URL for OpenAI-compatible APIs")
    parser.add_argument("--n_papers", type=int, default=None,
                        help="Limit number of papers (for testing/cost control)")
    parser.add_argument("--delay",    type=float, default=0.5,
                        help="Delay in seconds between API calls")
    parser.add_argument("--no_resume", action="store_true",
                        help="Start fresh (ignore existing output file)")
    parser.add_argument("--ks",       type=int, nargs="+", default=[1, 3, 5, 10])
    args = parser.parse_args()

    llm_fn = get_llm_fn(args.provider, args.model, args.base_url)

    print(f"Provider : {args.provider} / {args.model}")
    print(f"Input    : {args.input}")
    print(f"Output   : {args.output}")
    if args.n_papers:
        print(f"Papers   : {args.n_papers} (limited)")

    results_df = run_inference(
        input_csv  = args.input,
        output_csv = args.output,
        llm_fn     = llm_fn,
        n_papers   = args.n_papers,
        delay_sec  = args.delay,
        resume     = not args.no_resume,
    )

    metrics = compute_metrics(results_df, ks=tuple(args.ks))
    print("\n=== RecRanker Zero-shot Metrics (Hard Test) ===")
    for k, v in metrics.items():
        if k == "n_papers":
            print(f"  Papers evaluated : {v}")
        else:
            print(f"  {k:<12} : {v:.4f}")

    # Compare with L2R baseline
    print("\n=== Context: L2R hard-case baseline (from l2r_output_hard) ===")
    print("  MRR    : 0.7471  (L2R trained on hard cases)")
    print("  Acc@1  : 0.6162")
    print("  NDCG@3 : 0.7586")


if __name__ == "__main__":
    main()
