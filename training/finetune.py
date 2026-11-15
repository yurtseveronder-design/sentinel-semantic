"""Fine-tune a small prompt-injection classifier and score it on a file it never trained on.

Training data: the public Hugging Face dataset `deepset/prompt-injections` (text, label 0/1).
Evaluation: sentinel-eval's blind set (data/blind.jsonl), which is NEVER used for training or tuning.

    pip install "transformers>=4.46" torch datasets accelerate
    python training/finetune.py --blind ../sentinel-eval/data/blind.jsonl --out model_out

Prints catch rate (attacks flagged) and false-block rate (benign flagged) at several thresholds, so the
result is comparable with the numbers in sentinel-eval's RESULTS.md. Use the model with
SEMANTIC_CLASSIFIER_MODEL=<model_out dir or hub id>.
"""
import argparse
import json


def load_blind(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    return [(r["text"], 1 if r["label"] == "attack" else 0) for r in rows]


def rates(probs, labels, threshold):
    atk = [p for p, y in zip(probs, labels) if y == 1]
    ben = [p for p, y in zip(probs, labels) if y == 0]
    catch = sum(p >= threshold for p in atk) / max(len(atk), 1)
    fp = sum(p >= threshold for p in ben) / max(len(ben), 1)
    return catch, fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="distilbert-base-multilingual-cased")
    ap.add_argument("--dataset", default="deepset/prompt-injections")
    ap.add_argument("--blind", required=True)
    ap.add_argument("--out", default="model_out")
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer,
                              TrainingArguments)

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=2)
    ds = load_dataset(args.dataset)
    train = ds["train"].map(lambda b: tok(b["text"], truncation=True, max_length=256), batched=True)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
                               per_device_train_batch_size=16, save_strategy="no", report_to=[],
                               logging_steps=20),
        train_dataset=train, processing_class=tok)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)

    texts, labels = zip(*load_blind(args.blind))
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            enc = tok(list(texts[i:i + 16]), truncation=True, max_length=256, padding=True, return_tensors="pt")
            probs += torch.softmax(model(**enc).logits, dim=-1)[:, 1].tolist()
    print(f"\nBlind set ({sum(labels)} attacks, {len(labels) - sum(labels)} benign), model {args.base}")
    for t in (0.5, 0.7, 0.9):
        c, f = rates(probs, labels, t)
        print(f"threshold {t:.2f}: catch {c * 100:.1f}%  false-block {f * 100:.1f}%")


if __name__ == "__main__":
    main()
