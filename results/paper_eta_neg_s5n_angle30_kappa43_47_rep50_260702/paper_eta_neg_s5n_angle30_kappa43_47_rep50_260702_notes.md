# Paper simulation S5N_dense_decision_small_angle_heterogeneous_kappa_mild eta-first run

- Date: 2026-07-03
- Scenario: S5-N dense decision support, small mean-direction target, mild heterogeneous concentration.
- Setting: K=4, n=1000, d=200, common q=4, decision q=80 (20 per component), noise q=116.
- Target pairwise direction angle: 30.0 degrees.
- Kappa: (43, 44, 46, 47).
- Repetitions: 50.
- Tuning: BIC, all rows are support-refit results.
- Rcpp helpers: ON.

## Summary

| method | reps | valid | ARI | true q | selected q | TPR | FPR | Precision | F1 | MSE_mu | MSE_kappa | MSE_centered_eta | common false | noise FPR |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D-L | 50 | 50 | 0.023 | 80 | 198.52 | 0.994 | 0.992 | 0.400 | 0.571 | 0.002 | 113.941 | 4.499 | 1.000 | 0.992 |
| D-GL | 50 | 50 | 0.001 | 80 | 16.56 | 0.088 | 0.080 | 0.331 | 0.170 | 0.001 | 197.466 | 3.096 | 1.000 | 0.048 |
| D-AGL | 50 | 50 | 0.005 | 80 | 16.80 | 0.093 | 0.078 | 0.293 | 0.201 | 0.001 | 483.095 | 4.417 | 1.000 | 0.046 |
| E-L | 50 | 50 | 0.025 | 80 | 188.22 | 0.949 | 0.936 | 0.403 | 0.565 | 0.002 | 114.674 | 4.693 | 0.925 | 0.937 |
| E-GL | 50 | 0 | NA | 80 | 0.00 | 0.000 | 0.000 | NA | NA | NA | NA | NA | 0.000 | 0.000 |
| E-AGL | 50 | 3 | 0.001 | 80 | 0.06 | 0.000 | 0.000 | 0.333 | 0.025 | 0.010 | 1642.263 | 1.127 | 0.000 | 0.000 |

## Notes

- True q is the posterior decision support size, not the number of common signal coordinates.
- Common coordinates have equal eta values across components and should not be selected as decision coordinates.
- This run is a paper-simulation candidate; check smoke output before treating larger repetitions as final.
