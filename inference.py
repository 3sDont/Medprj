# -*- coding: utf-8 -*-
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from tqdm import tqdm
from typing import List, Union

# ──────────────────────────────────────────────
# Re-use model classes (copied from main_simcprs_v2)
# ──────────────────────────────────────────────

def sim_matrix(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


class Pooler(nn.Module):
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"]

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        hidden_states = outputs.hidden_states
        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            return ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            return ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)


class ModelForSE(nn.Module):
    def __init__(self, model_name_or_path, pooler_type):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        self.pooler_type = pooler_type
        self.pooler = Pooler(pooler_type)

    def forward(self, input_ids=None, attention_mask=None, return_dict=None, **kwargs):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask,
            output_hidden_states=True if self.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=return_dict,
        )
        pooler_output = self.pooler(attention_mask, outputs)
        return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output=pooler_output,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
        )

    def encode(self, sentences: Union[str, List[str]], batch_size=8,
               show_progress_bar=None, convert_to_numpy=True,
               convert_to_tensor=False, device=None, tokenizer=None, max_len=512):
        self.eval()
        if convert_to_tensor:
            convert_to_numpy = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'):
            sentences = [sentences]
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.to(device)
        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Batches", disable=not show_progress_bar):
            batch = sentences[start_index: start_index + batch_size]
            features = tokenizer(batch, padding='max_length', truncation=True,
                                 max_length=max_len, return_tensors='pt').to(device)
            with torch.no_grad():
                out = self.forward(**features)
                all_embeddings.extend([row.cpu() for row in out.pooler_output])
        if convert_to_tensor:
            return torch.vstack(all_embeddings)
        elif convert_to_numpy:
            return np.asarray([e.numpy() for e in all_embeddings])
        return all_embeddings


class Model_Classifier_WithAim(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.linear1_1 = nn.Linear(768, 512)
        self.act1_1 = nn.ReLU()
        self.drop1_1 = nn.Dropout(0.1)
        self.linear2_1 = nn.Linear(768, 512)
        self.act2_1 = nn.ReLU()
        self.linear_main_1 = nn.Linear(512 + num_classes, num_classes)
        self.act_main_1 = nn.LogSoftmax(dim=1)

    def forward(self, inputs_tak, inputs_aims):
        output_tak = self.base_model(**inputs_tak)
        x = self.linear1_1(output_tak.last_hidden_state[:, 0, :])
        x = self.act1_1(x)
        x = self.drop1_1(x)
        y = self.act2_1(self.linear2_1(inputs_aims))
        cosine_feats = sim_matrix(x, y)
        out = self.linear_main_1(torch.cat((x, cosine_feats), dim=1))
        return self.act_main_1(out)


class Model_Classifier_NoAim(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.linear1_1 = nn.Linear(768, 512)
        self.act1_1 = nn.ReLU()
        self.drop1_1 = nn.Dropout(0.1)
        self.linear1_2 = nn.Linear(512, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs_tak):
        output_tak = self.base_model(**inputs_tak)
        x = self.linear1_1(output_tak.last_hidden_state[:, 0, :])
        x = self.act1_1(x)
        x = self.drop1_1(x)
        return self.logsoftmax(self.linear1_2(x))


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────

def load_model(checkpoint_path, model_name, pooler_type, n_classes, use_aim, device):
    base_model = ModelForSE(model_name, pooler_type)
    if use_aim:
        model = Model_Classifier_WithAim(base_model, n_classes)
    else:
        model = Model_Classifier_NoAim(base_model, n_classes)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: epoch {checkpoint.get('epoch', '?')}")
    return model, base_model


def predict(texts: List[str], model, tokenizer, aims_embeddings, device,
            max_len=512, batch_size=32, topk=10, use_aim=True):
    """
    Args:
        texts: list of input strings (e.g. title + abstract + keywords)
        model: trained classifier
        aims_embeddings: tensor [n_journals, 768] pre-encoded on device
        topk: number of top journal predictions to return

    Returns:
        List of dicts with keys: top_journal_ids, top_scores
    """
    all_results = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start: start + batch_size]
            enc = tokenizer(batch_texts, padding='max_length', truncation=True,
                            max_length=max_len, return_tensors='pt')
            enc = {k: v.to(device) for k, v in enc.items()}

            if use_aim:
                logits = model(enc, aims_embeddings)
            else:
                logits = model(enc)

            probs = torch.exp(logits)
            scores, indices = torch.topk(probs, k=min(topk, probs.size(1)), dim=1)

            for i in range(len(batch_texts)):
                all_results.append({
                    "top_journal_ids": indices[i].cpu().tolist(),
                    "top_scores": scores[i].cpu().tolist(),
                })
    return all_results


def build_text(row, features):
    parts = {
        'T': str(row.get('Title', '')),
        'A': str(row.get('Abstract', '')),
        'K': str(row.get('Keywords', '')),
    }
    return " ".join(parts[c] for c in features if c in parts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to trained .pth checkpoint")
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--pooler_type", type=str, default="cls")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Folder containing aims CSV and optionally test_set.csv")
    parser.add_argument("--aims_csv", type=str, default="journal_category.csv",
                        help="Filename of aims CSV inside data_path (default: journal_category.csv)")
    parser.add_argument("--input_csv", type=str, default=None,
                        help="CSV file with papers to predict (columns: Title, Abstract, Keywords). "
                             "If omitted, uses test_set.csv from data_path.")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start row index (inclusive, 0-based). Default: 0")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="End row index (exclusive). Default: all rows after start_idx")
    parser.add_argument("--features", type=str, default="TAK",
                        help="Feature combination: TAK | TA | TK | AK | T | A | K")
    parser.add_argument("--use_aim", action="store_true")
    parser.add_argument("--use_category", action="store_true")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--output_csv", type=str, default="predictions.csv")
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device: {device}")

    # Load aims data
    data_aims = pd.read_csv(os.path.join(args.data_path, args.aims_csv), encoding="ISO-8859-1")
    data_aims.fillna("", inplace=True)
    n_classes = len(data_aims)

    if args.use_category and "Categories" in data_aims.columns:
        X_aims = (data_aims["Aims"] + " " + data_aims["Categories"]).tolist()
    else:
        X_aims = data_aims["Aims"].tolist()

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model, base_model = load_model(
        args.checkpoint_path, args.model_name, args.pooler_type,
        n_classes, args.use_aim, device
    )

    # Encode aims embeddings once
    print("Encoding aims embeddings...")
    aims_embeddings = base_model.encode(
        X_aims, show_progress_bar=True, convert_to_tensor=True,
        device=device, tokenizer=tokenizer, max_len=args.max_len
    )
    aims_embeddings = aims_embeddings.to(device)

    # Load input papers
    input_path = args.input_csv or os.path.join(args.data_path, "test_set.csv")
    data_input = pd.read_csv(input_path, encoding="ISO-8859-1")
    data_input.fillna("", inplace=True)

    # Slice the requested range
    total_rows = len(data_input)
    start = args.start_idx
    end = args.end_idx if args.end_idx is not None else total_rows
    end = min(end, total_rows)
    if start >= total_rows:
        raise ValueError(f"--start_idx {start} >= total rows {total_rows}")
    data_input = data_input.iloc[start:end].reset_index(drop=True)
    print(f"Processing rows [{start}, {end}) — {len(data_input)} samples (total file: {total_rows})")

    # Build texts from selected features
    feature_cols = list(args.features)  # e.g. 'TAK' → ['T','A','K']
    texts = [build_text(row, feature_cols) for _, row in data_input.iterrows()]

    # Run inference
    print(f"Running inference on {len(texts)} papers (top-{args.topk})...")
    results = predict(texts, model, tokenizer, aims_embeddings, device,
                      max_len=args.max_len, batch_size=args.batch_size,
                      topk=args.topk, use_aim=args.use_aim)

    # Build output dataframe
    rows = []
    for idx, (res, (_, paper)) in enumerate(zip(results, data_input.iterrows())):
        true_label = paper.get('Label', None)
        global_idx = start + idx  # preserve original row index across chunks
        for rank, (jid, score) in enumerate(zip(res['top_journal_ids'], res['top_scores']), start=1):
            rows.append({
                "Paper_ID": global_idx,
                "True_Label": true_label,
                "Rank": rank,
                "Predicted_Journal_ID": jid,
                "Score": round(score, 6),
                "Is_Correct": int(jid == true_label) if true_label is not None else None,
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to: {args.output_csv}")

    # Print summary accuracy if ground truth available
    if "Label" in data_input.columns:
        for k in [1, 3, 5, 10]:
            if k > args.topk:
                continue
            correct = df_out[df_out["Rank"] <= k].groupby("Paper_ID")["Is_Correct"].max().sum()
            acc = correct / len(texts)
            print(f"  Acc@{k:2d}: {acc:.4f}")

    print(f"\nDone. Chunk [{start}, {end}) written to: {args.output_csv}")
