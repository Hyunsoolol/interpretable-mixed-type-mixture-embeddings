#!/usr/bin/env python
"""
BGE-M3 lexical/sparse embedding smoke for Eta-group real-data analysis.

This script prepares token-level sparse lexical features from local BBC text
documents. It intentionally treats dense BGE-M3 embeddings as optional
robustness output only; coordinate-level interpretation should use the lexical
weights.

Default behavior is offline-safe. Pass --allow-download only when the user has
explicitly approved downloading BAAI/bge-m3 from Hugging Face.
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
        description="Prepare BGE-M3 lexical/sparse BBC smoke features."
    )
    parser.add_argument("--bbc-root", default="data/bbc/raw/bbc")
    parser.add_argument("--classes", default="sport,entertainment,tech")
    parser.add_argument("--max-per-class", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--model-name", default="BAAI/bge-m3")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--top-features", type=int, default=500)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--out-dir", default="results/bge_m3_sparse_bbc3_smoke_260624")
    parser.add_argument("--label", default="bge_m3_sparse_bbc3_smoke_260624")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow model download. Default is offline/local-cache only.",
    )
    parser.add_argument(
        "--return-dense",
        action="store_true",
        help="Also save dense BGE-M3 vectors for robustness checks only.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-fp16", action="store_true")
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


def import_bge_model():
    try:
        from FlagEmbedding import BGEM3FlagModel  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "FlagEmbedding is not installed. Install it in a Python environment "
            "before running this smoke script."
        ) from exc
    return BGEM3FlagModel


def encode_bge_m3(args: argparse.Namespace, texts: List[str]):
    if not args.allow_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    BGEM3FlagModel = import_bge_model()
    model_kwargs = {"use_fp16": bool(args.use_fp16)}
    if args.device:
        model_kwargs["device"] = args.device
    model = BGEM3FlagModel(args.model_name, **model_kwargs)
    output = model.encode(
        texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        return_dense=args.return_dense,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    return model, output


def id_to_token(model, token_id) -> str:
    tok = str(token_id)
    try:
        tid = int(token_id)
        if hasattr(model, "tokenizer"):
            return model.tokenizer.convert_ids_to_tokens(tid)
    except Exception:
        pass
    return tok


def lexical_weights_to_token_dicts(model, lexical_weights) -> List[Dict[str, float]]:
    """
    Convert BGE-M3 lexical weights into token-keyed dictionaries.

    FlagEmbedding versions differ slightly: some expose token-id dictionaries,
    while some provide a convert_id_to_token helper. This function supports both
    without assuming one exact package version.
    """
    if hasattr(model, "convert_id_to_token"):
        try:
            converted = model.convert_id_to_token(lexical_weights)
            if isinstance(converted, list):
                return [
                    {str(k): float(v) for k, v in doc.items()}
                    for doc in converted
                ]
        except Exception:
            pass

    token_docs = []
    for doc in lexical_weights:
        out = {}
        for token_id, value in doc.items():
            out[id_to_token(model, token_id)] = float(value)
        token_docs.append(out)
    return token_docs


def clean_token(token: str) -> str:
    return token.replace("##", "").strip()


def keep_token(token: str) -> bool:
    if not token:
        return False
    if SPECIAL_TOKEN_RE.match(token):
        return False
    return any(ch.isalnum() for ch in token)


def select_features(
    token_docs: List[Dict[str, float]], top_features: int, min_df: int
) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    df = Counter()
    total_weight = defaultdict(float)
    max_weight = defaultdict(float)

    cleaned_docs = []
    for doc in token_docs:
        cleaned = defaultdict(float)
        for token, value in doc.items():
            tok = clean_token(str(token))
            if keep_token(tok):
                cleaned[tok] += float(value)
        cleaned_docs.append(dict(cleaned))
        for tok, value in cleaned.items():
            df[tok] += 1
            total_weight[tok] += value
            max_weight[tok] = max(max_weight[tok], value)

    stats = {}
    n_docs = len(cleaned_docs)
    for tok, freq in df.items():
        if freq < min_df:
            continue
        stats[tok] = {
            "df": float(freq),
            "df_prop": float(freq) / max(n_docs, 1),
            "mean_weight": total_weight[tok] / max(n_docs, 1),
            "mean_nonzero_weight": total_weight[tok] / max(freq, 1),
            "max_weight": max_weight[tok],
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


def write_matrix_csv_gz(path: Path, matrix: List[List[float]], header: List[str]):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id"] + header)
        for i, row in enumerate(matrix, start=1):
            writer.writerow([i] + [f"{v:.10g}" for v in row])


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = tuple(c.strip() for c in args.classes.split(",") if c.strip())
    docs = read_bbc_docs(Path(args.bbc_root), classes, args.max_per_class, args.seed)
    texts = [d["text"] for d in docs]
    labels = [d["label"] for d in docs]

    model, output = encode_bge_m3(args, texts)
    lexical_weights = output.get("lexical_weights")
    if lexical_weights is None:
        raise RuntimeError("BGE-M3 output did not contain lexical_weights.")

    token_docs = lexical_weights_to_token_dicts(model, lexical_weights)
    features, stats = select_features(token_docs, args.top_features, args.min_df)
    feature_index = {tok: j for j, tok in enumerate(features)}

    matrix = []
    nnz = 0
    for doc in token_docs:
        row = [0.0] * len(features)
        for token, value in doc.items():
            tok = clean_token(str(token))
            j = feature_index.get(tok)
            if j is not None:
                row[j] += float(value)
        nnz += sum(1 for v in row if v != 0.0)
        matrix.append(row)
    matrix = l2_normalize_rows(matrix)

    matrix_path = out_dir / f"{args.label}_matrix_top{len(features)}.csv.gz"
    vocab_path = out_dir / f"{args.label}_vocabulary_top{len(features)}.csv"
    meta_path = out_dir / f"{args.label}_metadata.csv"
    summary_path = out_dir / f"{args.label}_feature_summary.csv"
    notes_path = out_dir / f"{args.label}_embedding_notes.md"

    write_matrix_csv_gz(matrix_path, matrix, features)
    write_csv(
        vocab_path,
        [
            {
                "feature_id": i + 1,
                "token": tok,
                "df": int(stats[tok]["df"]),
                "df_prop": f"{stats[tok]['df_prop']:.6f}",
                "mean_weight": f"{stats[tok]['mean_weight']:.10g}",
                "mean_nonzero_weight": f"{stats[tok]['mean_nonzero_weight']:.10g}",
                "max_weight": f"{stats[tok]['max_weight']:.10g}",
            }
            for i, tok in enumerate(features)
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
            for i in range(len(docs))
        ],
        ["doc_id", "label", "source_path", "n_chars"],
    )

    n = len(docs)
    d = len(features)
    density = nnz / max(n * d, 1)
    class_counts = Counter(labels)
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
                "# BGE-M3 Lexical/Sparse Smoke Notes",
                "",
                "This smoke uses BGE-M3 lexical weights as token-level sparse features.",
                "Dense BGE-M3 vectors, if requested, are for semantic-geometry robustness only.",
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
                "Do not interpret dense embedding coordinates as selected variables. "
                "Coordinate-level interpretation should use the lexical token features.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    if args.return_dense and "dense_vecs" in output:
        dense_path = out_dir / f"{args.label}_dense_vectors_robustness_only.json"
        with dense_path.open("w", encoding="utf-8") as f:
            json.dump(output["dense_vecs"], f)

    print(json.dumps(summary[0], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
