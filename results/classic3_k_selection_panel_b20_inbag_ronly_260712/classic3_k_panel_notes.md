# Classic3 extended K-selection panel diagnostic

- Candidate K={2,3,4,5,6,7,8,9,10}; bootstrap reps=20; bootstrap nstart=5.
- AIC/BIC/RIC/RICc/EBIC/ICL use train fits only.
- ICL is BIC + 2 times posterior classification entropy (lower is better).
- Held-out density uses bootstrap out-of-bag NLL; the external test split is not used for K selection.
- Every bootstrap initialization is estimated from the in-bag sample; full-train fits are excluded from initialization.
- Stability is mean pairwise ARI between full-train predictions from bootstrap fits.
- Token coherence is mean NPMI among the top 10 positive centered-Eta tokens per component.
- Supplied labels are used only after the diagnostic for ARI/NMI reporting.
- Bootstrap elapsed seconds: 604.9.

| kappa model | criterion | selected K | value |
|---|---|---:|---:|
| shared | AIC | 10 | -30606843.871 |
| shared | BIC | 10 | -30485989.881 |
| shared | RIC | 9 | -30346576.860 |
| shared | RICc | 8 | -30280954.547 |
| shared | EBIC_g0.5 | 8 | -30338790.338 |
| shared | EBIC_g1 | 8 | -30217175.899 |
| shared | ICL_BIC | 10 | -30485986.781 |
| shared | bootstrap_OOB_NLL_min | 10 | -4913.590 |
| shared | bootstrap_OOB_NLL_1SE | 10 | -4912.949 |
| shared | bootstrap_pairwise_stability | 3 | 0.996 |
| free | AIC | 10 | -30640102.567 |
| free | BIC | 10 | -30519194.193 |
| free | RIC | 9 | -30379292.891 |
| free | RICc | 7 | -30310931.982 |
| free | EBIC_g0.5 | 9 | -30371337.010 |
| free | EBIC_g1 | 7 | -30255101.747 |
| free | ICL_BIC | 10 | -30519187.728 |
| free | bootstrap_OOB_NLL_min | 10 | -4918.802 |
| free | bootstrap_OOB_NLL_1SE | 10 | -4918.182 |
| free | bootstrap_pairwise_stability | 3 | 0.992 |

- Audit checks passed: 11/11.
