# -*- coding: utf-8 -*-
"""
aims_Scope_Sim.py — Tính Aims_Scope_Sim cho predictions từ inference.py
                     dùng allenai/specter2 (độc lập hoàn toàn với BioBERT classifier).

Tại sao dùng SPECTER2 thay vì BioBERT base:
  - Train riêng cho academic paper similarity → embed đúng semantic domain khoa học
  - Độc lập với BioBERT classifier → cosine sim là feature BỔ SUNG thực sự cho L2R
  - normalize_embeddings=True → cosine sim = dot product → nhanh, không cần torch

Variant nên dùng:
  allenai/specter2_proximity  — tối ưu cho retrieval/similarity (khuyên dùng)
  allenai/specter2            — base model, cũng dùng được

Usage:
    python aims_Scope_Sim.py \\
        --predictions_csv predictions.csv \\
        --input_csv data/test_set.csv \\
        --data_path data/ \\
        --output_csv predictions_with_sim.csv
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from inference import build_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Tính Aims_Scope_Sim dùng SPECTER2 (độc lập với base model).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Input
    parser.add_argument("--predictions_csv", type=str, required=True,
                        help="File predictions CSV từ inference.py")
    parser.add_argument("--input_csv", type=str, required=True,
                        help="CSV gốc của bài báo (cột: Title, Abstract, Keywords)")
    # Encoder
    parser.add_argument("--encoder_model", type=str, default="allenai/specter2_base",
                        help="SentenceTransformer model để encode papers và aims")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size khi encode (SPECTER2 nhẹ hơn BioBERT nhiều)")
    # Aims
    parser.add_argument("--data_path", type=str, required=True,
                        help="Thư mục chứa aims CSV")
    parser.add_argument("--aims_csv", type=str, default="journal_category.csv")
    parser.add_argument("--use_category", action="store_true",
                        help="Nối thêm cột Categories vào Aims khi encode")
    # Features
    parser.add_argument("--features", type=str, default="TAK",
                        help="Feature combination: TAK | TA | TK | AK | T | A | K")
    # Output
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Đường dẫn output. Mặc định: ghi đè lên predictions_csv.")
    args = parser.parse_args()

    # ── Load encoder ───────────────────────────────────────────────────────────
    print(f"Loading encoder: {args.encoder_model} ...")
    encoder = SentenceTransformer(args.encoder_model)

    # ── Load aims ──────────────────────────────────────────────────────────────
    data_aims = pd.read_csv(os.path.join(args.data_path, args.aims_csv), encoding="ISO-8859-1")
    data_aims.fillna("", inplace=True)
    n_classes = len(data_aims)
    if args.use_category and "Categories" in data_aims.columns:
        X_aims = (data_aims["Aims"] + " " + data_aims["Categories"]).tolist()
    else:
        X_aims = data_aims["Aims"].tolist()
    print(f"Loaded {n_classes} journals from {args.aims_csv}")

    # ── Encode aims một lần ────────────────────────────────────────────────────
    print("Encoding aims embeddings...")
    aims_embs = encoder.encode(
        X_aims,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )  # [n_classes, hidden_size], float32, L2-normalized

    # ── Load predictions ───────────────────────────────────────────────────────
    pred_df = pd.read_csv(args.predictions_csv)
    print(f"Predictions: {len(pred_df)} rows, {pred_df['Paper_ID'].nunique()} papers")

    # ── Load và encode papers ──────────────────────────────────────────────────
    input_df = pd.read_csv(args.input_csv, encoding="ISO-8859-1")
    input_df.fillna("", inplace=True)

    feature_cols = list(args.features)
    unique_paper_ids = sorted(pred_df["Paper_ID"].unique())

    if max(unique_paper_ids) >= len(input_df):
        raise ValueError(
            f"Paper_ID tối đa là {max(unique_paper_ids)} "
            f"nhưng input_csv chỉ có {len(input_df)} dòng. "
            f"Hãy chắc chắn dùng đúng file input_csv gốc."
        )

    texts = [build_text(input_df.iloc[pid], feature_cols) for pid in unique_paper_ids]

    print("Encoding paper embeddings...")
    paper_embs = encoder.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )  # [n_unique_papers, hidden_size], float32, L2-normalized

    paper_id_to_idx = {pid: i for i, pid in enumerate(unique_paper_ids)}

    # ── Tính cosine sim theo chunk (tránh OOM với dataset lớn) ────────────────
    # Vì đã normalize L2 → cosine_sim(u, v) = dot(u, v)
    print("Computing Aims_Scope_Sim...")
    paper_indices_arr   = np.array([paper_id_to_idx[pid] for pid in pred_df["Paper_ID"]], dtype=np.int32)
    journal_indices_arr = pred_df["Predicted_Journal_ID"].values.astype(np.int32)

    chunk_size = 200_000  # ~0.6 GB/chunk với hidden_size=768
    n_rows = len(pred_df)
    sims = np.empty(n_rows, dtype=np.float32)

    for start in tqdm(range(0, n_rows, chunk_size), desc="Computing sims"):
        end = min(start + chunk_size, n_rows)
        p_chunk = paper_embs[paper_indices_arr[start:end]]    # [chunk, hidden]
        j_chunk = aims_embs[journal_indices_arr[start:end]]   # [chunk, hidden]
        sims[start:end] = (p_chunk * j_chunk).sum(axis=1)

    pred_df["Aims_Scope_Sim"] = sims.round(6)

    # ── Lưu output ────────────────────────────────────────────────────────────
    out_path = args.output_csv or args.predictions_csv
    pred_df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
    print(f"Aims_Scope_Sim stats:")
    print(f"  mean = {pred_df['Aims_Scope_Sim'].mean():.4f}")
    print(f"  min  = {pred_df['Aims_Scope_Sim'].min():.4f}")
    print(f"  max  = {pred_df['Aims_Scope_Sim'].max():.4f}")
