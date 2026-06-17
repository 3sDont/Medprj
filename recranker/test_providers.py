# -*- coding: utf-8 -*-
"""
Quick smoke-test: send 1 sample prompt to a provider and print the ranked output.
Usage:
    python -m recranker.test_providers --provider ollama --model qwen2.5:7b
    python -m recranker.test_providers --provider groq   --model llama-3.3-70b-versatile
    python -m recranker.test_providers --provider gemini --model gemini-1.5-flash
"""
import argparse, json, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
from recranker.inference import get_llm_fn, parse_ranking

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model",    default="qwen2.5:7b")
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--n",        type=int, default=3, help="Number of papers to test")
    args = parser.parse_args()

    df = pd.read_csv("data/recranker_data/test_nl.csv").head(args.n)

    print(f"Provider : {args.provider}")
    print(f"Model    : {args.model}")
    print(f"Papers   : {args.n}")
    print("-" * 50)

    llm_fn = get_llm_fn(args.provider, args.model, args.base_url)

    correct = 0
    for i, (_, row) in enumerate(df.iterrows()):
        t0 = time.time()
        try:
            response = llm_fn(row["prompt"])
            elapsed  = time.time() - t0
        except Exception as e:
            print(f"[{i+1}] ERROR: {e}")
            continue

        cands  = json.loads(row["candidate_ids"])
        ranked = parse_ranking(response, cands)
        true_j = int(row["true_journal_id"])
        rank1  = ranked[0] if ranked else None
        hit    = rank1 == true_j
        if hit:
            correct += 1

        print(f"[{i+1}] Paper {row['paper_id']}  true={true_j}  "
              f"pred={rank1}  {'HIT' if hit else 'MISS'}  ({elapsed:.1f}s)")
        print(f"      Response preview: {response[:120].replace(chr(10),' ')}")

    print(f"\nAcc@1 on {args.n} papers: {correct}/{args.n} = {100*correct/args.n:.0f}%")
    print(f"Estimated time for 1337 papers: "
          f"~{1337 * (time.time()-t0) / 60:.0f} min")

if __name__ == "__main__":
    main()
