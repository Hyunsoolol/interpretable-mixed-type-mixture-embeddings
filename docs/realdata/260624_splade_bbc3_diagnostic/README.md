# SPLADE BBC3 Real-data Diagnostic

This folder contains a compact, git-trackable summary of the SPLADE BBC3 d=500 Eta-group EBIC diagnostic.

## Purpose

The goal is to summarize whether SPLADE sparse lexical expansion is a plausible real-data representation for Eta-group coordinate support interpretation. This is diagnostic only and should not be treated as an official real-data result.

## Files

- `splade_bbc3_realdata_conclusion_260624.md`: main real-data diagnostic conclusion.
- `splade_d500_ebic_selected_tokens_overall.csv`: selected tokens ranked by centered eta norm.
- `splade_d500_ebic_selected_tokens_by_class.csv`: signed token scores by majority-mapped cluster.
- `splade_d500_ebic_cluster_class_table.csv`: cluster/class contingency table.

## Main Conclusion

SPLADE d=500 with Eta-group EBIC/RICc reduces support relative to Rossi while preserving clustering in the BBC3 smoke. BIC remains too weak and selects dense support. SPLADE tokens are learned lexical expansion tokens, not necessarily raw observed words.

## Large Files Not Tracked Here

- Payload RDS: `data/bbc/processed/splade_sparse_bbc3_alpha_min3_variance_top500_payload.rds`
- Fit object: `results/splade_bbc3_eta_smoke_top500_260624/cstr_eta_compare_fit.rds`
- SPLADE sparse matrices: `results/splade_sparse_bbc3_smoke_260624/*matrix_top*.csv.gz`
