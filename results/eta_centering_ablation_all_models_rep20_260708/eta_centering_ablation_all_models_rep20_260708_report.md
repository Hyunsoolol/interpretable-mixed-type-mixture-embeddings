# Eta centering ablation all-model diagnostic

Diagnostic-only run. These rows separate mu/eta parameterization, raw/centered contrast, and entry-wise/group penalties.

- reps: 20
- K=4, n=1000, d=100
- common q=6, specific q/component=4, true union q=22
- kappa=(30,45,65,90)

| method | reps | valid | selected q | ARI | TPR | FPR | Precision | F1 | MSE_mu | MSE_kappa | MSE_eta | common q rate | specific q rate | noise q rate |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M-L BIC | 20 | 20 | 25.90 | 0.693 | 1.000 | 0.050 | 0.855 | 0.920 | 0.000 | 2.636 | 0.110 | 1.000 | 1.000 | 0.050 |
| M-L BIC + refit | 20 | 20 | 25.90 | 0.687 | 1.000 | 0.050 | 0.855 | 0.920 | 0.000 | 1.826 | 0.162 | 1.000 | 1.000 | 0.050 |
| M-GL BIC | 20 | 20 | 23.95 | 0.675 | 1.000 | 0.025 | 0.921 | 0.958 | 0.000 | 3.622 | 0.264 | 1.000 | 1.000 | 0.025 |
| M-GL BIC + refit | 20 | 20 | 23.95 | 0.687 | 1.000 | 0.025 | 0.921 | 0.958 | 0.000 | 1.855 | 0.146 | 1.000 | 1.000 | 0.025 |
| M-AGL BIC | 20 | 20 | 22.55 | 0.683 | 1.000 | 0.007 | 0.977 | 0.988 | 0.000 | 3.552 | 0.153 | 1.000 | 1.000 | 0.007 |
| M-AGL BIC + refit | 20 | 20 | 22.55 | 0.689 | 1.000 | 0.007 | 0.977 | 0.988 | 0.000 | 1.770 | 0.131 | 1.000 | 1.000 | 0.007 |
| E-L BIC | 20 | 20 | 30.85 | 0.651 | 1.000 | 0.113 | 0.722 | 0.836 | 0.000 | 36.547 | 0.172 | 1.000 | 1.000 | 0.113 |
| E-L BIC + refit | 20 | 20 | 30.85 | 0.680 | 1.000 | 0.113 | 0.722 | 0.836 | 0.000 | 1.923 | 0.226 | 1.000 | 1.000 | 0.113 |
| E-GL BIC | 20 | 20 | 23.20 | 0.650 | 1.000 | 0.015 | 0.950 | 0.974 | 0.000 | 30.810 | 0.332 | 1.000 | 1.000 | 0.015 |
| E-GL BIC + refit | 20 | 20 | 23.20 | 0.687 | 1.000 | 0.015 | 0.950 | 0.974 | 0.000 | 1.715 | 0.141 | 1.000 | 1.000 | 0.015 |
| E-CL BIC | 20 | 20 | 24.40 | 0.619 | 0.991 | 0.033 | 0.898 | 0.941 | 0.000 | 23.280 | 0.635 | 1.000 | 0.988 | 0.033 |
| E-CL BIC + refit | 20 | 20 | 24.40 | 0.688 | 0.991 | 0.033 | 0.898 | 0.941 | 0.000 | 1.915 | 0.177 | 1.000 | 0.988 | 0.033 |
| E-CGL BIC | 20 | 20 | 24.00 | 0.664 | 0.995 | 0.027 | 0.918 | 0.954 | 0.000 | 8.891 | 0.381 | 1.000 | 0.994 | 0.027 |
| E-CGL BIC + refit | 20 | 20 | 24.00 | 0.689 | 0.995 | 0.027 | 0.918 | 0.954 | 0.000 | 1.825 | 0.166 | 1.000 | 0.994 | 0.027 |
| E-CAGL BIC | 20 | 20 | 22.05 | 0.680 | 0.991 | 0.003 | 0.989 | 0.990 | 0.000 | 1.945 | 0.170 | 1.000 | 0.988 | 0.003 |
| E-CAGL BIC + refit | 20 | 20 | 22.05 | 0.687 | 0.991 | 0.003 | 0.989 | 0.990 | 0.000 | 1.792 | 0.139 | 1.000 | 0.988 | 0.003 |

Excluded from the main comparison table:

| excluded candidate | form | reason |
|:---|:---|:---|
| M-CGL | $\lambda_\mu\sum_j\lVert\mu_{\cdot j}-\bar\mu_j\mathbf{1}\rVert_2$ | The direction parameter $\mu$ is constrained to the unit sphere, while the posterior score uses $\eta_k=\kappa_k\mu_k$. Centering $\mu$ alone does not reflect concentration heterogeneity and is not aligned with posterior decision support. In this diagnostic, M-CGL + refit selected q=38.75 with FPR=0.215 and MSE_eta=1.359. |
