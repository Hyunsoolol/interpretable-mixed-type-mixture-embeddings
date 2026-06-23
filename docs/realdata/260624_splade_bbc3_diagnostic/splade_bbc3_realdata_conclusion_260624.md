# SPLADE BBC3 Real-data Diagnostic 260624

## Status

- Diagnostic only; not an official real-data result.
- No new embedding or fit was run for this summary.
- Token tables are computed from the stored SPLADE d=500 EBIC Eta path fit because the refit fit object is not persisted in `cstr_eta_compare_fit.rds`.
- SPLADE coordinates are learned lexical expansion tokens, not necessarily raw observed words.
- Dense embedding coordinates are not used for interpretation.

## Input and Protocol

| Item | Value |
|:---|:---|
| Dataset | BBC 3-class subset |
| Classes | `sport`, `entertainment`, `tech` |
| n | 360 |
| d | 500 |
| Representation | SPLADE sparse lexical expansion |
| Token filter | `alpha_min3` |
| Feature ranking | variance |
| Main diagnostic criterion | EBIC |
| Secondary diagnostic | RICc |
| Baseline | Rossi shared-kappa path and matched TF-IDF diagnostic |

## Rossi vs Eta-group Summary

| Method | ARI | Selected q | Kappa ratio | Cluster size |
|:---|---:|---:|---:|:---|
| Rossi EBIC | 0.903 | 489 | 1.000 | 121;121;118 |
| Eta-group EBIC | 0.903 | 206 | 1.034 | 121;121;118 |
| Eta-group EBIC + refit | 0.911 | 206 | 1.092 | 121;120;119 |
| Eta-group BIC + refit | 0.919 | 500 | 1.112 | 121;119;120 |
| Eta-group RICc + refit | 0.903 | 181 | 1.076 | 121;121;118 |

BIC keeps full support for Eta-group (`selected q = 500`), so BIC is too weak for sparse real-data support in this smoke. EBIC and RICc are useful diagnostic criteria, not official tuning replacements.

## Cluster-majority Mapping

| Cluster | Majority class | Sport | Entertainment | Tech | Cluster size | Purity |
|---:|:---|---:|---:|---:|---:|---:|
| 1 | sport | 119 | 0 | 2 | 121 | 0.983 |
| 2 | tech | 0 | 6 | 115 | 121 | 0.950 |
| 3 | entertainment | 1 | 114 | 3 | 118 | 0.966 |

## Class-mapped Selected Tokens

### sport cluster

| Rank | Signed token | Centered eta score | Abs score |
|---:|:---|---:|---:|
| 1 | `+champion` | 29.318 | 29.318 |
| 2 | `+match` | 24.747 | 24.747 |
| 3 | `+team` | 23.730 | 23.730 |
| 4 | `+football` | 23.696 | 23.696 |
| 5 | `-film` | -17.344 | 17.344 |
| 6 | `+win` | 17.277 | 17.277 |
| 7 | `+club` | 17.252 | 17.252 |
| 8 | `+player` | 17.141 | 17.141 |
| 9 | `+coach` | 13.937 | 13.937 |
| 10 | `+won` | 13.266 | 13.266 |
| 11 | `-tech` | -12.733 | 12.733 |
| 12 | `+england` | 11.413 | 11.413 |

### tech cluster

| Rank | Signed token | Centered eta score | Abs score |
|---:|:---|---:|---:|
| 1 | `+tech` | 24.684 | 24.684 |
| 2 | `+software` | 21.813 | 21.813 |
| 3 | `+computer` | 20.778 | 20.778 |
| 4 | `-actor` | -19.928 | 19.928 |
| 5 | `+internet` | 18.979 | 18.979 |
| 6 | `-won` | -16.649 | 16.649 |
| 7 | `-champion` | -16.094 | 16.094 |
| 8 | `+computers` | 15.651 | 15.651 |
| 9 | `+people` | 14.587 | 14.587 |
| 10 | `+mobile` | 14.280 | 14.280 |
| 11 | `-win` | -14.207 | 14.207 |
| 12 | `+users` | 13.796 | 13.796 |

### entertainment cluster

| Rank | Signed token | Centered eta score | Abs score |
|---:|:---|---:|---:|
| 1 | `+film` | 30.143 | 30.143 |
| 2 | `+award` | 23.095 | 23.095 |
| 3 | `+actor` | 21.928 | 21.928 |
| 4 | `+awards` | 19.639 | 19.639 |
| 5 | `+oscar` | 17.457 | 17.457 |
| 6 | `+movie` | 16.632 | 16.632 |
| 7 | `+singer` | 15.276 | 15.276 |
| 8 | `+actors` | 15.080 | 15.080 |
| 9 | `-champion` | -13.224 | 13.224 |
| 10 | `-player` | -12.888 | 12.888 |
| 11 | `-match` | -12.512 | 12.512 |
| 12 | `+movies` | 12.396 | 12.396 |

## Overall Top Selected Tokens

| Rank | Token | Centered eta norm | Document frequency |
|---:|:---|---:|---:|
| 1 | `film` | 37.057 | 118 |
| 2 | `champion` | 35.965 | 145 |
| 3 | `match` | 30.310 | 104 |
| 4 | `tech` | 30.237 | 82 |
| 5 | `actor` | 29.697 | 212 |
| 6 | `team` | 29.065 | 131 |
| 7 | `football` | 29.022 | 106 |
| 8 | `award` | 28.317 | 99 |
| 9 | `software` | 26.715 | 69 |
| 10 | `computer` | 25.448 | 80 |
| 11 | `awards` | 24.054 | 64 |
| 12 | `internet` | 23.255 | 72 |
| 13 | `win` | 22.578 | 146 |
| 14 | `player` | 21.864 | 148 |
| 15 | `won` | 21.555 | 177 |
| 16 | `oscar` | 21.381 | 56 |
| 17 | `club` | 21.132 | 135 |
| 18 | `movie` | 20.471 | 68 |
| 19 | `computers` | 19.168 | 77 |
| 20 | `singer` | 18.722 | 76 |

## Interpretation

- The sport-majority cluster is driven by tokens such as `champion`, `match`, `team`, `football`, `club`, and `player`.
- The tech-majority cluster is driven by tokens such as `tech`, `software`, `computer`, `internet`, `computers`, `mobile`, and `users`.
- The entertainment-majority cluster is driven by tokens such as `film`, `award`, `actor`, `awards`, `oscar`, `movie`, and `singer`.
- Compared with Rossi EBIC, Eta-group EBIC + refit reduces selected support from 489 to 206 while keeping ARI around 0.911.
- Compared with matched TF-IDF, SPLADE d=500 is more compatible with the current Eta-group objective in this smoke; TF-IDF Eta-group became sparse but lost clustering quality.

## Conclusion

SPLADE sparse lexical representation is a plausible real-data representation candidate for Eta-group because it preserves token-level coordinate interpretability and maintains clustering under EBIC/RICc diagnostics. The current result is suitable for a meeting or appendix-level diagnostic table, but it should not be presented as final real-data validation. A fixed real-data protocol and stability checks are still needed before using it as a main-text empirical result.

## Files in This Folder

- `splade_d500_ebic_selected_tokens_overall.csv`: EBIC selected token ranking by centered eta norm.
- `splade_d500_ebic_selected_tokens_by_class.csv`: class-mapped signed centered eta scores.
- `splade_d500_ebic_cluster_class_table.csv`: cluster-majority class table.
- `splade_bbc3_realdata_conclusion_260624.md`: this diagnostic conclusion note.
