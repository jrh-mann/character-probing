#!/usr/bin/env python3
"""Train a dedicated text classifier for age/gender prediction.

Compares:
1. Fine-tuned DistilBERT (66M params) — pretrained + classification head
2. Fine-tuned TinyBERT (14M) — smaller pretrained
3. Logistic regression on frozen embeddings — no fine-tuning

This shows how a dedicated classifier compares to probing a frozen LLM.
"""
import os, sys, time, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import balanced_accuracy_score
from transformers import AutoTokenizer, AutoModel

BASE_DIR = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", "/root/hf_cache")

AGE_BIN_MAP = {1: 0, 2: 1, 3: 2}
GENDER_MAP = {"female": 0, "male": 1}


class TextDS(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    def __len__(self): return len(self.texts)
    def __getitem__(self, i): return self.texts[i], self.labels[i]


class ClassifierHead(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        return self.classifier(x)


def train_finetuned(model_name, train_texts, train_labels, val_texts, val_labels,
                     test_texts, test_labels, num_classes, epochs=3, lr=2e-5, bs=32, max_len=256):
    """Fine-tune a pretrained model with a classification head."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    head = ClassifierHead(model.config.hidden_size, num_classes).to(device)

    n_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head.parameters())
    print(f"  {model_name}: {n_params/1e6:.1f}M params")

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=lr)

    def encode_batch(texts, labels):
        enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        return enc.input_ids.to(device), enc.attention_mask.to(device), torch.tensor(labels, device=device)

    # Training
    model.train()
    for epoch in range(epochs):
        perm = np.random.permutation(len(train_texts))
        total_loss = 0
        for i in range(0, len(train_texts), bs):
            idx = perm[i:i+bs]
            batch_texts = [train_texts[j] for j in idx]
            batch_labels = [train_labels[j] for j in idx]
            ids, mask, labels = encode_batch(batch_texts, batch_labels)

            optimizer.zero_grad()
            outputs = model(input_ids=ids, attention_mask=mask)
            # Mean pool
            hidden = outputs.last_hidden_state
            pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            logits = head(pooled)
            loss = nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Val accuracy
        model.eval()
        val_preds = []
        with torch.no_grad():
            for i in range(0, len(val_texts), bs):
                batch = val_texts[i:i+bs]
                ids, mask, _ = encode_batch(batch, [0]*len(batch))
                outputs = model(input_ids=ids, attention_mask=mask)
                pooled = (outputs.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
                preds = head(pooled).argmax(-1)
                val_preds.extend(preds.cpu().tolist())
        val_acc = balanced_accuracy_score(val_labels, val_preds)
        print(f"    Epoch {epoch+1}: loss={total_loss/(len(train_texts)/bs):.4f}, val_acc={val_acc:.4f}")
        model.train()

    # Test accuracy
    model.eval()
    test_preds = []
    with torch.no_grad():
        for i in range(0, len(test_texts), bs):
            batch = test_texts[i:i+bs]
            ids, mask, _ = encode_batch(batch, [0]*len(batch))
            outputs = model(input_ids=ids, attention_mask=mask)
            pooled = (outputs.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            preds = head(pooled).argmax(-1)
            test_preds.extend(preds.cpu().tolist())
    test_acc = balanced_accuracy_score(test_labels, test_preds)

    del model, head
    torch.cuda.empty_cache()
    return test_acc, n_params


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--data_path", default=str(BASE_DIR / "data" / "processed" / "blog_corpus.parquet"))
    pa.add_argument("--max_train", type=int, default=20000)
    pa.add_argument("--max_test", type=int, default=5000)
    pa.add_argument("--epochs", type=int, default=3)
    args = pa.parse_args()

    df = pd.read_parquet(args.data_path)
    train = df[df["split"]=="train"].sample(min(args.max_train, len(df[df["split"]=="train"])), random_state=42)
    val = df[df["split"]=="val"].sample(min(args.max_test, len(df[df["split"]=="val"])), random_state=42)
    test = df[df["split"]=="test"].sample(min(args.max_test, len(df[df["split"]=="test"])), random_state=42)

    results = []

    for task, label_map, n_classes in [("gender", GENDER_MAP, 2), ("age_bin", AGE_BIN_MAP, 3)]:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}")

        train_labels = train[task].map(label_map).fillna(0).astype(int).tolist()
        val_labels = val[task].map(label_map).fillna(0).astype(int).tolist()
        test_labels = test[task].map(label_map).fillna(0).astype(int).tolist()
        train_texts = train["text"].tolist()
        val_texts = val["text"].tolist()
        test_texts = test["text"].tolist()

        models = [
            ("google/bert_uncased_L-2_H-128_A-2", 2e-5),   # ~4.4M - TinyBERT
            ("google/bert_uncased_L-4_H-256_A-4", 2e-5),   # ~11M
            ("google/bert_uncased_L-6_H-512_A-8", 2e-5),   # ~29M
            ("distilbert-base-uncased", 2e-5),               # ~66M
        ]

        for model_name, lr in models:
            short = model_name.split("/")[-1]
            print(f"\n  {short}:")
            t0 = time.time()
            try:
                acc, n_params = train_finetuned(
                    model_name, train_texts, train_labels,
                    val_texts, val_labels, test_texts, test_labels,
                    n_classes, epochs=args.epochs, lr=lr)
                elapsed = time.time() - t0
                print(f"    Test accuracy: {acc:.4f} ({elapsed:.0f}s)")
                results.append({
                    "model": short, "task": task, "n_params": n_params,
                    "test_balanced_acc": round(acc, 5), "time_s": round(elapsed, 1)
                })
            except Exception as e:
                print(f"    FAILED: {e}")

    rdf = pd.DataFrame(results)
    out_path = BASE_DIR / "results" / "classifier_comparison.csv"
    rdf.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")
    print(rdf.to_string(index=False))


if __name__ == "__main__":
    main()
