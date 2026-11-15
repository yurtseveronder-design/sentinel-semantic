"""Build one training file from several public Hugging Face datasets, with no leakage from sentinel-eval.

    python training/build_dataset.py --eval-dir ../sentinel-eval/data --out data/train.jsonl

Rules that keep the later benchmark honest:
  * exact duplicates (after lower-casing and whitespace folding) are removed;
  * any text that also appears in sentinel-eval's dev / held-out / blind files is dropped;
  * a dataset that fails to load is skipped with a warning, and the datacard says which ones were used.
Check each source's licence before redistributing the data or a model trained on it.
"""
import argparse
import json
import random
import re
from pathlib import Path

# name, split, text column, label column, label mapping (value -> 1 attack / 0 benign), optional cap
SOURCES = [
    ("deepset/prompt-injections", "train", "text", "label", {0: 0, 1: 1}, None),
    ("xTRam1/safe-guard-prompt-injection", "train", "text", "label", {0: 0, 1: 1}, None),
    ("jackhhao/jailbreak-classification", "train", "prompt", "type", {"benign": 0, "jailbreak": 1}, None),
    # Everyday requests: only there to teach the model what harmless prompts look like.
    ("databricks/databricks-dolly-15k", "train", "instruction", None, 0, 2500),
]


def norm(t):
    return re.sub(r"\s+", " ", t.strip().lower())


def eval_texts(eval_dir):
    seen = set()
    for f in Path(eval_dir).glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                seen.add(norm(json.loads(line)["text"]))
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out", default="data/train.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from datasets import load_dataset

    banned = eval_texts(args.eval_dir)
    rng = random.Random(args.seed)
    rows, seen, card = [], set(), []
    for name, split, tcol, lcol, mapping, cap in SOURCES:
        try:
            ds = load_dataset(name, split=split)
        except Exception as exc:  # network, renamed dataset, gated access...
            print(f"WARNING: skipped {name}: {exc}")
            card.append({"source": name, "status": f"skipped: {type(exc).__name__}"})
            continue
        got = dup = leaked = 0
        items = list(ds)
        rng.shuffle(items)
        for r in items:
            text = r.get(tcol)
            if not isinstance(text, str) or len(text.strip()) < 3:
                continue
            label = mapping if lcol is None else mapping.get(r.get(lcol))
            if label is None:
                continue
            key = norm(text)
            if key in banned:
                leaked += 1
                continue
            if key in seen:
                dup += 1
                continue
            seen.add(key)
            rows.append({"text": text.strip(), "label": label, "source": name})
            got += 1
            if cap and got >= cap:
                break
        card.append({"source": name, "status": "ok", "kept": got, "duplicates": dup, "removed_for_eval_overlap": leaked})
        print(card[-1])

    rng.shuffle(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    attacks = sum(r["label"] for r in rows)
    summary = {"total": len(rows), "attacks": attacks, "benign": len(rows) - attacks, "sources": card}
    (out.parent / "datacard.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "sources"}))


if __name__ == "__main__":
    main()
