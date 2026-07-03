# Paper simulation S5_small_angle_heterogeneous_kappa eta-first run

- Date: 2026-07-03
- Scenario: S5 small mean-direction difference with heterogeneous concentration.
- Setting: K=4, n=1000, d=200, common q=4, decision q=16 (4 per component), noise q=180.
- Target pairwise direction angle: 30.0 degrees.
- Kappa: (30, 40, 50, 60).
- Repetitions: 50.
- Tuning: BIC, all rows are support-refit results.
- Rcpp helpers: ON.

## Summary

| method | reps | valid | ARI | true q | selected q | TPR | FPR | Precision | F1 | MSE_mu | MSE_kappa | MSE_centered_eta | common false | noise FPR |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D-L | 50 | 50 | 0.546 | 16 | 199.94 | 1.000 | 1.000 | 0.080 | 0.148 | 0.001 | 119.199 | 2.365 | 1.000 | 1.000 |
| D-GL | 50 | 50 | 0.609 | 16 | 39.48 | 1.000 | 0.128 | 0.541 | 0.668 | 0.001 | 199.777 | 1.688 | 1.000 | 0.108 |
| D-AGL | 50 | 50 | 0.589 | 16 | 57.86 | 1.000 | 0.228 | 0.406 | 0.538 | 0.001 | 153.602 | 1.504 | 1.000 | 0.210 |
| E-L | 50 | 50 | 0.543 | 16 | 199.08 | 1.000 | 0.995 | 0.080 | 0.149 | 0.001 | 88.567 | 2.175 | 0.985 | 0.995 |
| E-GL | 50 | 50 | 0.606 | 16 | 45.44 | 0.998 | 0.160 | 0.478 | 0.609 | 0.004 | 153.502 | 0.764 | 0.170 | 0.160 |
| E-AGL | 50 | 50 | 0.630 | 16 | 20.78 | 0.938 | 0.031 | 0.876 | 0.881 | 0.004 | 235.356 | 0.270 | 0.025 | 0.032 |

## Notes

- True q is the posterior decision support size, not the number of common signal coordinates.
- Common coordinates have equal eta values across components and should not be selected as decision coordinates.
- This run is a paper-simulation candidate; check smoke output before treating larger repetitions as final.
