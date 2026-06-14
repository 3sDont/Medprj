# -*- coding: utf-8 -*-
"""
aims_Scope_Sim.py — Thêm cột Aims_Scope_Sim vào predictions CSV từ inference.py.

Với mỗi cặp (paper, candidate_journal), tính cosine similarity giữa
CLS embedding của bài báo và aims embedding của tạp chí đó.

Cách tính nhất quán với main_simcprs_v2.py (dòng 800-819):
  paper_emb   = base_model(**features).last_hidden_state[:, 0, :]   (CLS token)
  aims_emb    = base_model.encode(aims_texts, ...)                   (pooler output)
  sim         = cosine_similarity(paper_emb, aims_emb)

Usage:
    python aims_Scope_Sim.py \\
        --predictions_csv predictions.csv \\
        --input_csv data/test_set.csv \\
        --checkpoint_path checkpoints/best_model.pth \\
        --model_name roberta-base \\
        --data_path data/ \\
        --features TAK \\
        --output_csv predictions_with_sim.csv
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from inference import load_model, build_text


def encode_papers_cls(base_model, tokenizer, texts, device, batch_size, max_len):
    """Encode paper texts, trả về CLS token embeddings [n_papers, hidden_size]."""
    base_model.eval()
    all_embs = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding papers"):
            batch = texts[start: start + batch_size]
            enc = tokenizer(batch, padding='max_length', truncation=True,
                            max_length=max_len, return_tensors='pt').to(device)
            out = base_model(**enc)
            all_embs.append(out.last_hidden_state[:, 0, :].cpu())
    return torch.cat(all_embs, dim=0)  # [n_papers, 768]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Tính Aims_Scope_Sim cho predictions từ inference.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Input
    parser.add_argument("--predictions_csv", type=str, required=True,
                        help="File predictions CSV từ inference.py")
    parser.add_argument("--input_csv", type=str, required=True,
                        help="CSV gốc của bài báo (cột: Title, Abstract, Keywords)")
    # Model
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Đường dẫn checkpoint .pth đã train")
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--pooler_type", type=str, default="cls")
    parser.add_argument("--use_aim", action="store_true")
    # Aims
    parser.add_argument("--data_path", type=str, required=True,
                        help="Thư mục chứa aims CSV")
    parser.add_argument("--aims_csv", type=str, default="journal_category.csv")
    parser.add_argument("--use_category", action="store_true",
                        help="Nối thêm cột Categories vào Aims khi encode")
    # Encoding
    parser.add_argument("--features", type=str, default="TAK",
                        help="Feature combination: TAK | TA | TK | AK | T | A | K")
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    # Output
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Đường dẫn output. Mặc định: ghi đè lên predictions_csv.")
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device: {device}")

    # ── Load aims ──────────────────────────────────────────────────────────────
    data_aims = pd.read_csv(os.path.join(args.data_path, args.aims_csv), encoding="ISO-8859-1")
    data_aims.fillna("", inplace=True)
    n_classes = len(data_aims)
    if args.use_category and "Categories" in data_aims.columns:
        X_aims = (data_aims["Aims"] + " " + data_aims["Categories"]).tolist()
    else:
        X_aims = data_aims["Aims"].tolist()
    print(f"Loaded {n_classes} journals from {args.aims_csv}")

    # ── Load tokenizer + base model ────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    _, base_model = load_model(
        args.checkpoint_path, args.model_name, args.pooler_type,
        n_classes, args.use_aim, device
    )

    # ── Encode aims (pooler output, nhất quán với inference.py) ───────────────
    print("Encoding aims embeddings...")
    aims_embeddings = base_model.encode(
        X_aims, show_progress_bar=True, convert_to_tensor=True,
        device=device, tokenizer=tokenizer, max_len=args.max_len
    ).to(device)  # [n_classes, 768]

    # ── Load predictions CSV ───────────────────────────────────────────────────
    pred_df = pd.read_csv(args.predictions_csv)
    n_papers = pred_df["Paper_ID"].nunique()
    print(f"Predictions: {len(pred_df)} rows, {n_papers} papers")

    # ── Load input papers và encode (CLS token, nhất quán với main_simcprs_v2) ─
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
    paper_embeddings = encode_papers_cls(
        base_model, tokenizer, texts, device, args.batch_size, args.max_len
    ).to(device)  # [n_unique_papers, 768]

    paper_id_to_idx = {pid: i for i, pid in enumerate(unique_paper_ids)}

    # ── Tính cosine similarity theo batch (vectorized) ─────────────────────────
    print("Computing Aims_Scope_Sim...")
    paper_indices  = torch.tensor(
        [paper_id_to_idx[pid] for pid in pred_df["Paper_ID"]], dtype=torch.long
    )
    journal_indices = torch.tensor(
        pred_df["Predicted_Journal_ID"].values.astype(int), dtype=torch.long
    )

    p_embs = paper_embeddings[paper_indices]   # [n_rows, 768]
    j_embs = aims_embeddings[journal_indices]  # [n_rows, 768]
    sims   = F.cosine_similarity(p_embs, j_embs, dim=1)  # [n_rows]

    pred_df["Aims_Scope_Sim"] = sims.cpu().numpy().round(6)

    # ── Lưu output ────────────────────────────────────────────────────────────
    out_path = args.output_csv or args.predictions_csv
    pred_df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
    print(f"Aims_Scope_Sim stats:")
    print(f"  mean = {pred_df['Aims_Scope_Sim'].mean():.4f}")
    print(f"  min  = {pred_df['Aims_Scope_Sim'].min():.4f}")
    print(f"  max  = {pred_df['Aims_Scope_Sim'].max():.4f}")
