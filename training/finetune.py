"""Fine-tune a prompt-injection classifier and score it honestly.

    python training/build_dataset.py --eval-dir ../sentinel-eval/data --out data/train.jsonl
    python training/finetune.py --train data/train.jsonl --eval-dir ../sentinel-eval/data --out model_out

Honesty rules:
  * 10% of the training file is held back as a validation split; the decision threshold is chosen THERE
    (highest catch rate with validation false-block rate <= --max-fp), never on an evaluation set;
  * sentinel-eval's dev, held-out and blind sets are only used for the final report;
  * the report also shows fixed thresholds so the choice can be checked.
"""
import argparse
import json
import random
from pathlib import Path


def load_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def eval_set(path):
    rows = load_jsonl(path)
    return [r["text"] for r in rows], [1 if r["label"] == "attack" else 0 for r in rows]


def rates(probs, labels, threshold):
    atk = [p for p, y in zip(probs, labels) if y == 1]
    ben = [p for p, y in zip(probs, labels) if y == 0]
    catch = sum(p >= threshold for p in atk) / max(len(atk), 1)
    fp = sum(p >= threshold for p in ben) / max(len(ben), 1)
    return catch, fp


def pick_threshold(probs, labels, max_fp):
    best = (None, -1.0)
    for t in [i / 100 for i in range(5, 100)]:
        catch, fp = rates(probs, labels, t)
        if fp <= max_fp and catch > best[1]:
            best = (t, catch)
    return best[0] if best[0] is not None else 0.9


def predict(model, tok, texts, max_length, torch):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tok(texts[i:i + 32], truncation=True, max_length=max_length, padding=True, return_tensors="pt")
            out += torch.softmax(model(**enc).logits, dim=-1)[:, 1].tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="jsonl from build_dataset.py")
    ap.add_argument("--eval-dir", required=True, help="sentinel-eval/data (dev, held-out, blind)")
    ap.add_argument("--base", default="distilbert-base-multilingual-cased")
    ap.add_argument("--out", default="model_out")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--max-train", type=int, default=6000, help="cap on training rows (CPU time)")
    ap.add_argument("--max-fp", type=float, default=0.02, help="validation false-block budget for the threshold")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer,
                              TrainingArguments)

    rows = load_jsonl(args.train)
    random.Random(args.seed).shuffle(rows)
    n_val = max(200, len(rows) // 10)
    val, train = rows[:n_val], rows[n_val:n_val + args.max_train]
    print(f"train {len(train)} rows ({sum(r['label'] for r in train)} attacks), validation {len(val)}")

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=2)
    ds = Dataset.from_list([{"text": r["text"], "label": r["label"]} for r in train])
    ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_length), batched=True)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
                               per_device_train_batch_size=16, save_strategy="no", report_to=[],
                               logging_steps=50, seed=args.seed),
        train_dataset=ds, processing_class=tok)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)

    vp = predict(model, tok, [r["text"] for r in val], args.max_length, torch)
    vl = [r["label"] for r in val]
    threshold = pick_threshold(vp, vl, args.max_fp)
    vc, vf = rates(vp, vl, threshold)
    Path(args.out, "threshold.json").write_text(json.dumps({"threshold": threshold}))
    print(f"\nChosen threshold {threshold:.2f} (validation: catch {vc * 100:.1f}%, false-block {vf * 100:.1f}%)")

    summary = {"threshold": threshold, "validation": {"catch": vc, "false_block": vf}, "sets": {}}
    for name in ("prompts", "heldout", "blind"):
        path = Path(args.eval_dir) / f"{name}.jsonl"
        if not path.exists():
            continue
        texts, labels = eval_set(path)
        probs = predict(model, tok, texts, args.max_length, torch)
        print(f"\n{name}: {sum(labels)} attacks, {len(labels) - sum(labels)} benign")
        summary["sets"][name] = {}
        for t in sorted({0.5, 0.9, round(threshold, 2)}):
            c, f = rates(probs, labels, t)
            tag = "  <- chosen" if abs(t - threshold) < 0.005 else ""
            print(f"  threshold {t:.2f}: catch {c * 100:.1f}%  false-block {f * 100:.1f}%{tag}")
            summary["sets"][name][f"{t:.2f}"] = {"catch": c, "false_block": f}
    Path(args.out, "report.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
