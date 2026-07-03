# Paper simulation S6N_dense_decision_small_angle_equal_kappa eta-first run

- Date: 2026-07-03
- Scenario: S6-N dense decision support, small mean-direction target, equal concentration.
- Setting: K=4, n=1000, d=200, common q=4, decision q=80 (20 per component), noise q=116.
- Target pairwise direction angle: 30.0 degrees.
- Kappa: (45, 45, 45, 45).
- Repetitions: 50.
- Tuning: BIC, all rows are support-refit results.
- Rcpp helpers: ON.

## Summary

| method | reps | valid | ARI | true q | selected q | TPR | FPR | Precision | F1 | MSE_mu | MSE_kappa | MSE_centered_eta | common false | noise FPR |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D-L | 50 | 50 | 0.012 | 80 | 197.56 | 0.989 | 0.987 | 0.401 | 0.570 | 0.002 | 165.369 | 5.141 | 1.000 | 0.987 |
| D-GL | 50 | 50 | 0.001 | 80 | 15.82 | 0.062 | 0.090 | 0.208 | 0.135 | 0.001 | 97.965 | 2.418 | 1.000 | 0.059 |
| D-AGL | 50 | 50 | 0.002 | 80 | 10.96 | 0.038 | 0.066 | 0.163 | 0.108 | 0.001 | 119.271 | 2.260 | 1.000 | 0.034 |
| E-L | 50 | 50 | 0.011 | 80 | 187.38 | 0.943 | 0.933 | 0.403 | 0.564 | 0.002 | 125.599 | 4.964 | 0.945 | 0.932 |
| E-GL | 50 | 0 | NA | 80 | 0.00 | 0.000 | 0.000 | NA | NA | NA | NA | NA | 0.000 | 0.000 |
| E-AGL | 50 | 3 | 0.005 | 80 | 2.04 | 0.011 | 0.010 | 0.807 | 0.172 | 0.009 | 1155.244 | 1.986 | 0.005 | 0.010 |

## Notes

- True q is the posterior decision support size, not the number of common signal coordinates.
- Common coordinates have equal eta values across components and should not be selected as decision coordinates.
- This run is a paper-simulation candidate; check smoke output before treating larger repetitions as final.
