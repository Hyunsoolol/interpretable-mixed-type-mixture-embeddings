# Rossi mu-group diagnostic

- Date: 2026-07-01
- Status: ablation diagnostic only; not Rossi 2022 official baseline.
- Setting: K=4, n=1000, d=100, common q=6, component-specific q=4 each, true union q=22.
- Kappa: (30, 45, 65, 90), specific weight: 0.500.
- Repetitions: 20.
- Tuning: mu-space group path + BIC.

## Summary

| method | reps | valid_reps | zero_refit | ARI | selected_q | TPR | FPR | Precision | F1 | MSE_mu | MSE_kappa | MSE_centered_eta | lambda_mu |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Rossi mu-group diagnostic BIC | 20 | 20 | 0 | 0.675 | 29.10 | 1.000 | 0.091 | 0.813 | 0.883 | 0.000 | 3.881 | 0.228 | 0.041 |
| Rossi mu-group diagnostic BIC + refit | 20 | 20 | 0 | 0.685 | 29.10 | 1.000 | 0.091 | 0.813 | 0.883 | 0.000 | 2.171 | 0.192 | 0.041 |

## Notes

- This variant applies coordinate-wise group shrinkage to the Rossi direction matrix mu.
- It is a diagnostic approximation for separating mu-space group-penalty effects from eta-centered contrast effects.
- Unit-norm rows are restored after shrinkage while preserving the selected coordinate mask.
- It should not be described as Rossi and Barbaro (2022) official baseline.
