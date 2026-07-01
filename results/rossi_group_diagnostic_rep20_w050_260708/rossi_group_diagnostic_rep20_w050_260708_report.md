# Rossi natural-group diagnostic smoke

- Date: 2026-07-01
- Status: ablation diagnostic only; not Rossi 2022 official baseline.
- Setting: K=4, n=1000, d=100, common q=6, component-specific q=4 each, true union q=22.
- Kappa: (30, 45, 65, 90), specific weight: 0.500.
- Repetitions: 20.
- Tuning: natural-scale group path + BIC.

## Summary

| method | reps | valid_reps | zero_refit | ARI | selected_q | TPR | FPR | Precision | F1 | MSE_mu | MSE_kappa | MSE_centered_eta | lambda |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Rossi natural-group diagnostic BIC | 20 | 20 | 0 | 0.634 | 34.30 | 1.000 | 0.158 | 0.704 | 0.809 | 0.000 | 28.217 | 0.294 | 1.772 |
| Rossi natural-group diagnostic BIC + refit | 20 | 20 | 0 | 0.678 | 34.30 | 1.000 | 0.158 | 0.704 | 0.809 | 0.000 | 2.044 | 0.240 | 1.772 |

## Notes

- This variant applies coordinate-wise group shrinkage to the raw natural-scale matrix g = kappa * mu.
- It is a diagnostic approximation for separating group-penalty effects from eta parameterization effects.
- It should not be described as Rossi and Barbaro (2022) official baseline.
