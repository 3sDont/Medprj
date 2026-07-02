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
    def __init__(self, model_name_or_path, pooler_type, config=None):
        super().__init__()
        if config is not None:
            self.bert = AutoModel.from_config(config)
        else:
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

def _patch_config_from_ckpt(config, state_dict: dict):
    """
    Overwrite config embedding sizes with the shapes stored in the checkpoint.
    This prevents size-mismatch errors when the HuggingFace model revision differs
    from the version used during training.
    """
    _map = {
        'base_model.bert.embeddings.word_embeddings.weight':      'vocab_size',
        'base_model.bert.embeddings.position_embeddings.weight':  'max_position_embeddings',
        'base_model.bert.embeddings.token_type_embeddings.weight':'type_vocab_size',
    }
    for key, attr in _map.items():
        if key in state_dict:
            setattr(config, attr, state_dict[key].shape[0])
    return config


def load_model(checkpoint_path, model_name, pooler_type, n_classes, use_aim, device):
    from transformers import AutoConfig

    # Load checkpoint first so we can detect the exact architecture it was trained with
    checkpoint  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict  = checkpoint['model_state_dict']

    # Patch config to match checkpoint embedding dimensions (avoids size-mismatch errors)
    config = _patch_config_from_ckpt(AutoConfig.from_pretrained(model_name), state_dict)

    # Build model from patched config — weights come entirely from the checkpoint
    base_model = ModelForSE(model_name, pooler_type, config=config)
    if use_aim:
        model = Model_Classifier_WithAim(base_model, n_classes)
    else:
        model = Model_Classifier_NoAim(base_model, n_classes)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load_model] {len(missing)} missing keys  (e.g. {missing[:2]})")
    if unexpected:
        print(f"[load_model] {len(unexpected)} unexpected keys (e.g. {unexpected[:2]})")

    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: epoch={checkpoint.get('epoch','?')}  device={device}")
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


def run_inference_single(
    title: str,
    abstract: str,
    keywords: str,
    model: nn.Module,
    tokenizer,
    aims_embeddings,
    journal_df: pd.DataFrame,
    device,
    features: str = "TAK",
    max_len: int = 512,
    topk: int = 10,
    use_aim: bool = True,
) -> List[dict]:
    """
    Process a single user-provided paper and return top-k journal predictions.

    Returns:
        List of dicts with keys: journal_idx, Label, Name, Aims, Rank, Base_Score.
    """
    parts = {"T": title, "A": abstract, "K": keywords}
    text = " ".join(parts[c] for c in features if c in parts).strip()

    results = predict(
        [text], model, tokenizer, aims_embeddings, device,
        max_len=max_len, batch_size=1, topk=topk, use_aim=use_aim,
    )
    top_ids    = results[0]["top_journal_ids"]
    top_scores = results[0]["top_scores"]

    output = []
    for rank, (jid, score) in enumerate(zip(top_ids, top_scores), start=1):
        row = journal_df.iloc[jid]
        output.append({
            "journal_idx":       int(jid),
            "Label":             str(row.get("Label", str(jid))),
            "Name":              str(row.get("Journal", "")),
            "Aims":              str(row.get("Aims", "")),
            "Rank":              int(rank),
            "Base_Score":        round(float(score), 6),
            "Inverse_Base_Rank": round(_inverse_base_rank(rank, len(top_ids)), 6),
        })
    return output


def _inverse_base_rank(rank: int, top_k: int) -> float:
    """
    Linear-decay score in [0, 1] from a candidate's rank within the retrieved
    top-k: rank 1 -> 1.0, rank top_k -> 1/top_k. Mirrors the LTR feature of the
    same name in submission_ltr_dataset/features.py.
    """
    if rank is None or rank <= 0 or top_k <= 0:
        return 0.0
    return max(0.0, (float(top_k) - float(rank) + 1.0) / float(top_k))


if __name__ == '__main__':
    import json

    parser = argparse.ArgumentParser(description="Single-paper journal inference")
    # Model
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to trained .pth checkpoint")
    parser.add_argument("--model_name",  type=str, default="roberta-base")
    parser.add_argument("--pooler_type", type=str, default="cls")
    # Journal / aims data
    parser.add_argument("--data_path", type=str, required=True,
                        help="Folder containing aims CSV (default: journal_category.csv)")
    parser.add_argument("--aims_csv",    type=str, default="journal_category.csv")
    parser.add_argument("--use_category", action="store_true",
                        help="Append Categories column to Aims when encoding")
    # Single paper input
    parser.add_argument("--title",    type=str, default="", help="Paper title")
    parser.add_argument("--abstract", type=str, default="", help="Paper abstract")
    parser.add_argument("--keywords", type=str, default="", help="Paper keywords (comma-separated)")
    # Options
    parser.add_argument("--features", type=str, default="TAK",
                        help="Feature combination: TAK | TA | TK | AK | T | A | K")
    parser.add_argument("--use_aim", action="store_true")
    parser.add_argument("--max_len",  type=int, default=512)
    parser.add_argument("--topk",     type=int, default=10)
    parser.add_argument("--output_json", type=str, default="inference_result.json")
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device: {device}")

    # Load journal data
    journal_df = pd.read_csv(os.path.join(args.data_path, args.aims_csv), encoding="ISO-8859-1")
    journal_df.fillna("", inplace=True)
    n_classes = len(journal_df)
    print(f"Loaded {n_classes} journals from {args.aims_csv}")

    if args.use_category and "Categories" in journal_df.columns:
        X_aims = (journal_df["Aims"] + " " + journal_df["Categories"]).tolist()
    else:
        X_aims = journal_df["Aims"].tolist()

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model, base_model = load_model(
        args.checkpoint_path, args.model_name, args.pooler_type,
        n_classes, args.use_aim, device
    )

    # Encode all journal aims once
    print("Encoding aims embeddings...")
    aims_embeddings = base_model.encode(
        X_aims, show_progress_bar=True, convert_to_tensor=True,
        device=device, tokenizer=tokenizer, max_len=args.max_len
    )
    aims_embeddings = aims_embeddings.to(device)

    # Run inference on single paper
    print("Running inference...")
    top_journals = run_inference_single(
        title=args.title,
        abstract=args.abstract,
        keywords=args.keywords,
        model=model,
        tokenizer=tokenizer,
        aims_embeddings=aims_embeddings,
        journal_df=journal_df,
        device=device,
        features=args.features,
        max_len=args.max_len,
        topk=args.topk,
        use_aim=args.use_aim,
    )

    result = {
        "paper": {
            "title":    args.title,
            "abstract": args.abstract,
            "keywords": args.keywords,
        },
        "top_journals": top_journals,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved top-{args.topk} predictions → {args.output_json}")
    for j in top_journals[:5]:
        print(f"  #{j['Rank']:2d}  {j['Name'][:60]:60s}  score={j['Base_Score']:.4f}")
