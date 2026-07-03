# Paper simulation S5_small_angle_heterogeneous_kappa_mild eta-first run

- Date: 2026-07-03
- Scenario: S5 small mean-direction difference with mild heterogeneous concentration.
- Setting: K=4, n=1000, d=200, common q=4, decision q=16 (4 per component), noise q=180.
- Target pairwise direction angle: 30.0 degrees.
- Kappa: (43, 44, 46, 47).
- Repetitions: 50.
- Tuning: BIC, all rows are support-refit results.
- Rcpp helpers: ON.

## Summary

| method | reps | valid | ARI | true q | selected q | TPR | FPR | Precision | F1 | MSE_mu | MSE_kappa | MSE_centered_eta | common false | noise FPR |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D-L | 50 | 50 | 0.031 | 16 | 198.56 | 1.000 | 0.992 | 0.081 | 0.149 | 0.002 | 112.405 | 4.395 | 1.000 | 0.992 |
| D-GL | 50 | 50 | 0.027 | 16 | 18.18 | 0.263 | 0.076 | 0.200 | 0.284 | 0.001 | 749.811 | 5.370 | 1.000 | 0.055 |
| D-AGL | 50 | 50 | 0.039 | 16 | 20.64 | 0.305 | 0.086 | 0.184 | 0.342 | 0.001 | 345.094 | 3.318 | 1.000 | 0.065 |
| E-L | 50 | 50 | 0.029 | 16 | 193.48 | 0.989 | 0.966 | 0.082 | 0.151 | 0.002 | 121.131 | 4.691 | 0.945 | 0.966 |
| E-GL | 50 | 0 | NA | 16 | 0.00 | 0.000 | 0.000 | NA | NA | NA | NA | NA | 0.000 | 0.000 |
| E-AGL | 50 | 1 | 0.015 | 16 | 0.02 | 0.001 | 0.000 | 1.000 | 0.118 | 0.009 | 1539.065 | 1.040 | 0.000 | 0.000 |

## Notes

- True q is the posterior decision support size, not the number of common signal coordinates.
- Common coordinates have equal eta values across components and should not be selected as decision coordinates.
- This run is a paper-simulation candidate; check smoke output before treating larger repetitions as final.
