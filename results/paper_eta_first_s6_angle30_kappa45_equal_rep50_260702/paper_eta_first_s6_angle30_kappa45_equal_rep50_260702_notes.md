# Paper simulation S6_small_angle_equal_kappa eta-first run

- Date: 2026-07-03
- Scenario: S6 small mean-direction difference with equal concentration.
- Setting: K=4, n=1000, d=200, common q=4, decision q=16 (4 per component), noise q=180.
- Target pairwise direction angle: 30.0 degrees.
- Kappa: (45, 45, 45, 45).
- Repetitions: 50.
- Tuning: BIC, all rows are support-refit results.
- Rcpp helpers: ON.

## Summary

| method | reps | valid | ARI | true q | selected q | TPR | FPR | Precision | F1 | MSE_mu | MSE_kappa | MSE_centered_eta | common false | noise FPR |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D-L | 50 | 50 | 0.010 | 16 | 199.02 | 0.994 | 0.995 | 0.080 | 0.148 | 0.002 | 118.684 | 4.900 | 1.000 | 0.995 |
| D-GL | 50 | 50 | 0.004 | 16 | 15.24 | 0.096 | 0.074 | 0.077 | 0.152 | 0.001 | 619.062 | 4.879 | 1.000 | 0.054 |
| D-AGL | 50 | 50 | 0.005 | 16 | 10.28 | 0.081 | 0.049 | 0.057 | 0.192 | 0.001 | 527.888 | 3.965 | 1.000 | 0.028 |
| E-L | 50 | 50 | 0.012 | 16 | 191.94 | 0.975 | 0.958 | 0.081 | 0.150 | 0.002 | 101.513 | 4.729 | 0.955 | 0.958 |
| E-GL | 50 | 1 | 0.017 | 16 | 0.02 | 0.001 | 0.000 | 1.000 | 0.118 | 0.010 | 1662.998 | 1.020 | 0.000 | 0.000 |
| E-AGL | 50 | 2 | 0.012 | 16 | 0.56 | 0.004 | 0.003 | 0.537 | 0.105 | 0.008 | 946.052 | 2.354 | 0.005 | 0.003 |

## Notes

- True q is the posterior decision support size, not the number of common signal coordinates.
- Common coordinates have equal eta values across components and should not be selected as decision coordinates.
- This run is a paper-simulation candidate; check smoke output before treating larger repetitions as final.
