#!/usr/bin/env python
"""
SPLADE sparse lexical/expansion smoke for Eta-group real-data analysis.

The output coordinates are vocabulary-level learned lexical/expansion tokens.
They should not be described as raw observed words. Dense embeddings are not
created by this script because coordinate-level interpretation is the target.

Default behavior is offline-safe. Pass --allow-download only after explicitly
approving a Hugging Face model download.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


SPECIAL_TOKEN_RE = re.compile(r"^\[.*\]$|^<.*>$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare SPLADE sparse BBC smoke features."
    )
    parser.add_argument("--bbc-root", default="data/bbc/raw/bbc")
    parser.add_argument("--classes", default="sport,entertainment,tech")
    parser.add_argument("--max-per-class", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--model-name", default="naver/splade-cocondenser-ensembledistil")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--top-features", type=int, default=500)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--min-weight", type=float, default=1e-8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default="results/splade_sparse_bbc3_smoke_260624")
    parser.add_argument("--label", default="splade_sparse_bbc3_smoke_260624")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow model download. Default is offline/local-cache only.",
    )
    return parser.parse_args()


def read_bbc_docs(root: Path, classes: Iterable[str], max_per_class: int, seed: int):
    rng = random.Random(seed)
    docs = []
    for cls in classes:
        cls_dir = root / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Missing BBC class directory: {cls_dir}")
        files = sorted(cls_dir.glob("*.txt"))
        if max_per_class > 0 and len(files) > max_per_class:
            files = sorted(rng.sample(files, max_per_class))
        for path in files:
            text = path.read_text(encoding="utf-8", errors="replace")
            text = " ".join(text.split())
            docs.append({"text": text, "label": cls, "source_path": str(path)})
    return docs


def import_transformers_and_torch():
    try:
        import torch  # type: ignore
        from transformers import AutoModelForMaskedLM, AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "torch and transformers are required. Install a Python environment "
            "with torch and transformers before running this smoke script."
        ) from exc
    return torch, AutoModelForMaskedLM, AutoTokenizer


def load_splade_model(args: argparse.Namespace):
    if not args.allow_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    torch, AutoModelForMaskedLM, AutoTokenizer = import_transformers_and_torch()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    model.eval()
    model.to(args.device)
    return torch, tokenizer, model


def clean_token(token: str) -> str:
    token = token.replace("##", "")
    token = token.replace("\u0120", "")
    return token.strip()


def keep_token(token: str) -> bool:
    if not token:
        return False
    if SPECIAL_TOKEN_RE.match(token):
        return False
    return any(ch.isalnum() for ch in token)


def splade_encode(args: argparse.Namespace, texts: List[str]):
    torch, tokenizer, model = load_splade_model(args)
    vocab = tokenizer.get_vocab()
    id_to_token = {idx: tok for tok, idx in vocab.items()}

    doc_weights: List[Dict[str, float]] = []
    with torch.no_grad():
        for start in range(0, len(texts), args.batch_size):
            batch = texts[start : start + args.batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(args.device) for k, v in encoded.items()}
            outputs = model(**encoded)

            # SPLADE-style sparse vector:
            # max over token positions of log(1 + ReLU(masked-LM logits)).
            sparse_values = torch.log1p(torch.relu(outputs.logits))
            mask = encoded["attention_mask"].unsqueeze(-1)
            sparse_values = sparse_values * mask
            pooled = torch.max(sparse_values, dim=1).values

            for row in pooled.cpu():
                nz = torch.nonzero(row > args.min_weight, as_tuple=False).view(-1).tolist()
                doc = {}
                for idx in nz:
                    token = clean_token(id_to_token.get(int(idx), str(idx)))
                    if keep_token(token):
                        doc[token] = float(row[idx].item())
                doc_weights.append(doc)
    return doc_weights


def select_features(
    token_docs: List[Dict[str, float]], top_features: int, min_df: int
) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    df = Counter()
    total_weight = defaultdict(float)
    max_weight = defaultdict(float)

    for doc in token_docs:
        for token, value in doc.items():
            df[token] += 1
            total_weight[token] += value
            max_weight[token] = max(max_weight[token], value)

    n_docs = len(token_docs)
    stats = {}
    for token, freq in df.items():
        if freq < min_df:
            continue
        stats[token] = {
            "df": float(freq),
            "df_prop": float(freq) / max(n_docs, 1),
            "mean_weight": total_weight[token] / max(n_docs, 1),
            "mean_nonzero_weight": total_weight[token] / max(freq, 1),
            "max_weight": max_weight[token],
        }

    ordered = sorted(
        stats,
        key=lambda t: (
            stats[t]["df"],
            stats[t]["mean_weight"],
            stats[t]["mean_nonzero_weight"],
            stats[t]["max_weight"],
            t,
        ),
        reverse=True,
    )
    return ordered[:top_features], stats


def l2_normalize_rows(matrix: List[List[float]]) -> List[List[float]]:
    out = []
    for row in matrix:
        norm = math.sqrt(sum(v * v for v in row))
        if norm > 0:
            out.append([v / norm for v in row])
        else:
            out.append(row)
    return out


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_matrix_csv_gz(path: Path, matrix: List[List[float]], header: List[str]):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id"] + header)
        for i, row in enumerate(matrix, start=1):
            writer.writerow([i] + [f"{v:.10g}" for v in row])


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = tuple(c.strip() for c in args.classes.split(",") if c.strip())
    docs = read_bbc_docs(Path(args.bbc_root), classes, args.max_per_class, args.seed)
    texts = [d["text"] for d in docs]
    labels = [d["label"] for d in docs]

    token_docs = splade_encode(args, texts)
    features, stats = select_features(token_docs, args.top_features, args.min_df)
    feature_index = {token: j for j, token in enumerate(features)}

    matrix = []
    nnz = 0
    for doc in token_docs:
        row = [0.0] * len(features)
        for token, value in doc.items():
            j = feature_index.get(token)
            if j is not None:
                row[j] += value
        nnz += sum(1 for value in row if value != 0.0)
        matrix.append(row)
    matrix = l2_normalize_rows(matrix)

    n = len(docs)
    d = len(features)
    density = nnz / max(n * d, 1)
    class_counts = Counter(labels)

    matrix_path = out_dir / f"{args.label}_matrix_top{d}.csv.gz"
    vocab_path = out_dir / f"{args.label}_vocabulary_top{d}.csv"
    meta_path = out_dir / f"{args.label}_metadata.csv"
    summary_path = out_dir / f"{args.label}_feature_summary.csv"
    notes_path = out_dir / f"{args.label}_embedding_notes.md"

    write_matrix_csv_gz(matrix_path, matrix, features)
    write_csv(
        vocab_path,
        [
            {
                "feature_id": i + 1,
                "token": token,
                "df": int(stats[token]["df"]),
                "df_prop": f"{stats[token]['df_prop']:.6f}",
                "mean_weight": f"{stats[token]['mean_weight']:.10g}",
                "mean_nonzero_weight": f"{stats[token]['mean_nonzero_weight']:.10g}",
                "max_weight": f"{stats[token]['max_weight']:.10g}",
            }
            for i, token in enumerate(features)
        ],
        [
            "feature_id",
            "token",
            "df",
            "df_prop",
            "mean_weight",
            "mean_nonzero_weight",
            "max_weight",
        ],
    )
    write_csv(
        meta_path,
        [
            {
                "doc_id": i + 1,
                "label": labels[i],
                "source_path": docs[i]["source_path"],
                "n_chars": len(texts[i]),
            }
            for i in range(n)
        ],
        ["doc_id", "label", "source_path", "n_chars"],
    )

    summary = [
        {
            "label": args.label,
            "model": args.model_name,
            "classes": ";".join(classes),
            "n": n,
            "d": d,
            "max_per_class": args.max_per_class,
            "min_df": args.min_df,
            "top_features": args.top_features,
            "nonzero_entries": nnz,
            "density": f"{density:.8f}",
            "class_counts": ";".join(f"{k}:{class_counts[k]}" for k in classes),
            "matrix_path": str(matrix_path),
            "vocabulary_path": str(vocab_path),
            "metadata_path": str(meta_path),
        }
    ]
    write_csv(summary_path, summary, list(summary[0].keys()))

    notes_path.write_text(
        "\n".join(
            [
                "# SPLADE Sparse BBC3 Smoke Notes",
                "",
                "This smoke uses SPLADE sparse lexical/expansion weights as token-level features.",
                "Coordinates are learned lexical/expansion tokens, not necessarily raw observed words.",
                "Dense embeddings are not produced by this script.",
                "",
                f"- Label: `{args.label}`",
                f"- Model: `{args.model_name}`",
                f"- Classes: `{';'.join(classes)}`",
                f"- n: {n}",
                f"- d: {d}",
                f"- Matrix density before row normalization: {density:.8f}",
                f"- Matrix: `{matrix_path}`",
                f"- Vocabulary: `{vocab_path}`",
                "",
                "Coordinate-level interpretation should be based on the selected SPLADE "
                "vocabulary/expansion token list.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary[0], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
