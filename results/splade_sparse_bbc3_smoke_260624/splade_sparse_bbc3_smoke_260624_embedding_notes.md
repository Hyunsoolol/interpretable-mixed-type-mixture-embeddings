# SPLADE Sparse BBC3 Smoke Notes

Status: not executed in the current environment.

## Local Data Check

- Candidate dataset: BBC raw full text
- Preferred smoke subset: `sport`, `entertainment`, `tech`
- Available class counts: `sport:511; entertainment:386; tech:401`
- Existing TF-IDF comparison payload: `data/bbc/processed/bbc_tri_sport_entertainment_tech_d500_n1287.rds`
- Existing TF-IDF dimensions: `n = 1287`, `d = 500`

## Environment Check

- Python is not currently available.
  - `py -0p`: `No Installed Pythons Found`
  - `python --version`: WindowsApps Python launcher failed
- Hugging Face cache was not found under the usual local cache paths.
- SPLADE packages/models are therefore not available locally.
- SPLADE sparse embedding was not executed.

## Candidate Model

Default model for the smoke script:

- `naver/splade-cocondenser-ensembledistil`

This is an English SPLADE-style sparse lexical/expansion candidate. Before publication, the exact model license and citation should be confirmed.

## Reproducible Smoke Command

After a Python environment is available and `torch`/`transformers` are installed, run:

```powershell
py scripts\realdata\splade_sparse_smoke_260624.py `
  --bbc-root data\bbc\raw\bbc `
  --classes sport,entertainment,tech `
  --max-per-class 120 `
  --top-features 500 `
  --min-df 2 `
  --out-dir results\splade_sparse_bbc3_smoke_260624 `
  --label splade_sparse_bbc3_smoke_260624 `
  --allow-download
```

Without `--allow-download`, the script runs in local-cache/offline mode and will fail if the SPLADE model is not already cached.

## Interpretation Rule

- SPLADE coordinates are vocabulary-level learned lexical/expansion tokens.
- They are interpretable as model-expanded lexical features, not necessarily raw observed words.
- Dense LLM embeddings should remain semantic-geometry robustness checks only.
- For English text, SPLADE is the primary neural sparse candidate; BGE-M3 lexical/sparse remains a multilingual/long-document comparator.

