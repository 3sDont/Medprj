import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import pickle
import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

def save_parameter(save_object, save_file):
    with open(save_file, 'wb') as f:
        pickle.dump(save_object, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_parameter(load_file):
    with open(load_file, 'rb') as f:
        output = pickle.load(f)
    return output

def sim_matrix(a, b, eps=1e-8):
    """
    Calculate cosine similarity between two matrices.
    Note: added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def batch2device(batch, device):
    for key, value in batch.items():
        batch[key] = batch[key].to(device)
    return batch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        x = {
            key: torch.tensor(val) for key, val in self.dataset[idx].items()
        }
        return x
    def __len__(self):
        return len(self.dataset)
class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

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
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError

class ModelForCL(nn.Module):
    def __init__(self, model_name_or_path, pooler_type):
        super(ModelForCL, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        self.pooler_type = pooler_type
        self.pooler = Pooler(self.pooler_type)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        batch_size = input_ids.size(0)
        # Number of sentences in one instance
        # 2: pair instance; 3: pair instance with a hard negative
        num_sent = input_ids.size(1)

        # Flatten input for encoding
        input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
        attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent, len)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

        # Get raw embeddings
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if self.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=return_dict,
        )

        # Pooling
        if self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"]:
            pooler_output = self.pooler(attention_mask, outputs)
        pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)

        return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output=pooler_output,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
        )

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.sim = nn.CosineSimilarity(dim=-1)

    def _eval_denom(self, z1, z2):
        cosine_vals = self.sim(z1.unsqueeze(1), z2.unsqueeze(0)) / self.temperature
        denom = torch.sum(torch.exp(cosine_vals), dim=1)
        return denom

    def _contrastive_loss(self, z1, z2):
        num = torch.exp(self.sim(z1, z2) / self.temperature)
        denom = self._eval_denom(z1, z2)

        # Kiểm tra giá trị NaN hoặc Inf
        if torch.isnan(num).any() or torch.isnan(denom).any():
            print('NaN values detected in num or denom')
        if torch.isinf(num).any() or torch.isinf(denom).any():
            print('Inf values detected in num or denom')

        loss = -torch.mean(torch.log(num / denom))
        return loss

    def forward(self, z1, z2):
        return self._contrastive_loss(z1, z2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tuning script with contrastive learning")
    parser.add_argument("--working_path", type=str, default="./QLoRAPSR/", help="working path")
    parser.add_argument("--data_path", type=str, default="./QLoRAPSR/", help="working path")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Pretrained model name")
    parser.add_argument("--checkpoint_path", type=str, help="Path to the contrastive learning checkpoint")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--max_len", type=int, default=300, help="max_len")
    parser.add_argument("--pooler_type", type=str, default='cls_before_pooler', help="Pooler type")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--num_epoch", type=int, default=10, help="number of epochs")
    parser.add_argument("--device", type=str, default="mps", choices=["cpu", "cuda", "mps"], help="Device to use (cpu, cuda, mps)")
    args = parser.parse_args()


        # GPU accelerator
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = args.device
    
    # Xác định thiết bị từ args
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")  # Mặc định là CPU nếu không khả dụng
    
    print(f"Using device: {device}")
    
    data_args = {
        "train_file": args.data_path,
        "preprocessing_num_workers": None
    }
    data_files = {
        "train": data_args["train_file"]
    }
    tokenizer_kwargs = {
        "pretrained_path": args.model_name,
        "use_fast": True,
        "max_seq_length": args.max_len,
        "pad_to_max_length": True,
        "truncation": True,
        "return_tensors": None
    }

    datasets = load_dataset("csv", data_files=data_files)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_kwargs["pretrained_path"],
        use_fast=tokenizer_kwargs["use_fast"]
    )
    column_names = datasets["train"].column_names
    def prepare_features(examples):
        total = len(examples[column_names[0]])
        for idx in range(total):
            if examples[column_names[0]][idx] is None:
                examples[column_names[0]][idx] = " "
            if examples[column_names[1]][idx] is None:
                examples[column_names[1]][idx] = " "
        sentences = examples[column_names[0]] + examples[column_names[1]]
        sent_features = tokenizer(
            sentences,
            max_length=tokenizer_kwargs["max_seq_length"],
            truncation=True,
            padding="max_length" if tokenizer_kwargs["pad_to_max_length"] else False,
            return_tensors=tokenizer_kwargs["return_tensors"]
        )
        features = {}
        for key in sent_features:
            features[key] = [[sent_features[key][i], sent_features[key][i + total]] for i in range(total)]
        return features


    train_dataset = datasets["train"].map(
        prepare_features,
        batched=True,
        num_proc=data_args["preprocessing_num_workers"],
        remove_columns=column_names
    )

    dataset = Dataset(train_dataset)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)

    model_args = {
        'model_name_or_path': args.model_name,
        'pooler_type': args.pooler_type
    }
    model = ModelForCL(**model_args)
    model.to(device)

    # Định nghĩa các tham số
    decayRate = 0.86
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=2, gamma=decayRate)
    loss_fn = SupervisedContrastiveLoss(0.1)

    epoch = 0
    save_path =  args.working_path + "checkpoint/saved_model/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Kiểm tra nếu checkpoint tồn tại thì load mô hình
    if args.checkpoint_path is not None and os.path.exists(args.checkpoint_path):
        print(f"Checkpoint found at {args.checkpoint_path}")
        checkpoint_cl = torch.load(args.checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint_cl["model_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint_cl["optimizer_state_dict"])
        min_loss = checkpoint_cl["min_loss"]
        epoch = checkpoint_cl["epoch"] + 1  # Tiếp tục từ epoch tiếp theo

        print(f"Model loaded successfully from {args.checkpoint_path}")
        print(f"Min loss at epoch {checkpoint_cl['epoch']}: {min_loss}")
    else:
        print(f"Checkpoint not found. Starting training from epoch {epoch}")
        min_loss = np.inf

    # Training Loop
    for epoch in range(epoch, args.num_epoch):
        loop = tqdm(data_loader, leave=True)
        train_loss = 0.0

        for batch in loop:
            optimizer.zero_grad()

            if device.type in ["cuda", "mps"]:
                inputs = batch2device(batch, device)
            else:
                inputs = batch  # if using CPU

            # Forward pass
            outputs = model(**inputs)
            z1, z2 = outputs.pooler_output[:, 0], outputs.pooler_output[:, 1]

            # Compute loss and backpropagate
            loss = loss_fn(z1, z2)
            loss.backward()
            train_loss += loss.item()

            # Update weights
            optimizer.step()

            loop.set_description(f'Epoch: {epoch} - lr: {optimizer.param_groups[0]["lr"]}')
            loop.set_postfix(loss=loss.item())

        train_loss = train_loss / len(data_loader)
        lr_scheduler.step()

        # Save best checkpoint if loss decreases
        if train_loss < min_loss:
            print(f">> Loss Decreased ({min_loss:.6f} ---> {train_loss:.6f})")
            min_loss = train_loss
            checkpoint_save_path = os.path.join(save_path, f"Epoch_{epoch:02d}_SupCL_{args.model_name}.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "min_loss": min_loss,
                "epoch": epoch
        }, checkpoint_save_path)
            print(f"Saving checkpoint to: {checkpoint_save_path}")
        
            # Kiểm tra xem file có tồn tại không
            import os
            if os.path.exists(checkpoint_save_path):
                print(f"✅ Checkpoint saved successfully at {checkpoint_save_path}")
            else:
                print("❌ Checkpoint NOT saved!")
            # Ghi kết quả huấn luyện vào file result.txt
            result_path = os.path.join(save_path, "result.txt")
            with open(result_path, "a") as f:
                f.write(f"Epoch: {epoch}, Train Loss: {train_loss:.6f}\n")