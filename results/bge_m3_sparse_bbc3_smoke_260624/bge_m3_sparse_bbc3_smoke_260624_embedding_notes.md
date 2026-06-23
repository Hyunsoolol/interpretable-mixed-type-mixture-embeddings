# BGE-M3 Lexical/Sparse Smoke Notes

Status: not executed in the current environment.

## Local Data Check

- Candidate dataset: BBC raw full text
- Preferred smoke subset: `sport`, `entertainment`, `tech`
- Existing processed TF-IDF payload: `data/bbc/processed/bbc_tri_sport_entertainment_tech_d500_n1287.rds`
- Existing processed dimensions: `n = 1287`, `d = 500`
- Existing class counts: `entertainment:386; sport:500; tech:401`

## Environment Check

- Rscript is available at `C:\Program Files\R\R-4.2.1\bin\x64\Rscript.exe`.
- Default Python is not available: `py -V` returned `Can't find a default Python`.
- Hugging Face cache for `BAAI/bge-m3` was not found under the usual local cache paths.
- Therefore BGE-M3 lexical/sparse embedding was not executed.

## Reproducible Smoke Command

After a Python environment is available and `FlagEmbedding`/`torch` are installed, run:

```powershell
py scripts\realdata\bge_m3_sparse_smoke_260624.py `
  --bbc-root data\bbc\raw\bbc `
  --classes sport,entertainment,tech `
  --max-per-class 120 `
  --top-features 500 `
  --min-df 2 `
  --out-dir results\bge_m3_sparse_bbc3_smoke_260624 `
  --label bge_m3_sparse_bbc3_smoke_260624 `
  --allow-download
```

Without `--allow-download`, the script runs in local-cache/offline mode and will fail if `BAAI/bge-m3` is not already cached.

## Interpretation Rule

- Use BGE-M3 lexical/sparse token weights for coordinate-level interpretation.
- Treat BGE-M3 dense embeddings only as semantic-geometry robustness checks.
- Do not interpret dense embedding coordinates as selected variables.

