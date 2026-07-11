# Real-data final validation freeze (260711)

## Scope

- Classic3: primary leakage-controlled held-out analysis with full exact BIC refit and B=20 support reselection stability.
- BBC5: all five official classes, normalized-text duplicates removed before splitting, used as an independent held-out robustness/limitation analysis.
- CSTR: transductive Rossi reproduction bridge; not a held-out predictive analysis.
- E-CGL is the main proposed method; E-ACGL is an adaptive extension.

| dataset | method | stage | selected q | held-out NLL/doc | ARI | NMI |
|---|---|---|---:|---:|---:|---:|
| Classic3 | Spherical k-means | train-only refit | 2000.00 | NA | 0.9856 | 0.9710 |
| Classic3 | Dense vMF shared-kappa | train-only refit | 2000.00 | -4871.6918 | 0.9856 | 0.9710 |
| Classic3 | Dense vMF free-kappa | train-only refit | 2000.00 | -4872.9015 | 0.9927 | 0.9863 |
| Classic3 | M-L | saved train path | 2000.00 | -4871.0937 | 0.9892 | 0.9787 |
| Classic3 | E-CGL | saved train exact path | 1347.00 | -4872.2942 | 0.9927 | 0.9863 |
| Classic3 | E-ACGL | saved train exact path | 1348.00 | -4872.2981 | 0.9927 | 0.9863 |
| BBC5 deduplicated | Spherical k-means | train-only refit | 1000.00 | NA | 0.8942 | 0.8719 |
| BBC5 deduplicated | Dense vMF shared-kappa | train-only refit | 1000.00 | -2125.6229 | 0.8942 | 0.8719 |
| BBC5 deduplicated | Dense vMF free-kappa | train-only refit | 1000.00 | -2126.2275 | 0.8959 | 0.8736 |
| BBC5 deduplicated | M-L | saved train path | 1000.00 | -2124.8031 | 0.8889 | 0.8667 |
| BBC5 deduplicated | E-CGL | saved train exact path | 679.00 | -2124.7525 | 0.8849 | 0.8615 |
| BBC5 deduplicated | E-ACGL | saved train exact path | 691.00 | -2124.8671 | 0.8849 | 0.8615 |
| CSTR | Dense shared-kappa vMF | dense | 1000.00 | NA | 0.8023 | 0.7650 |
| CSTR | Rossi M-L | penalized | 888.72 | NA | 0.8083 | 0.7703 |
| CSTR | E-CGL | penalized | 311.14 | NA | 0.6344 | 0.6496 |
| CSTR | E-ACGL | penalized | 313.28 | NA | 0.6095 | 0.6381 |
| CSTR | E-CGL | refit | 311.14 | NA | 0.6153 | 0.6449 |
| CSTR | E-ACGL | refit | 313.28 | NA | 0.6066 | 0.6401 |

## Frozen findings

- Classic3: E-CGL selected 1347/2000 coordinates and matched dense free-kappa test ARI (0.9927); its NLL was 0.6073 per document higher.
- Classic3 reselection stability: E-CGL mean q=1343.8 and Nogueira stability=0.884; E-ACGL mean q=1345.3 and stability=0.887.
- BBC5: E-CGL selected 679/1000 coordinates; versus dense free-kappa, test ARI changed by -0.0109 and NLL by 1.4750 per document.
- BBC5 adaptive extension: E-ACGL selected 691 coordinates and did not improve test ARI over E-CGL (difference 0.0000).
- BBC5 exact-BIC margins were 16.689 for E-CGL and 1.623 for E-ACGL; the adaptive support choice is therefore less decisive.
- CSTR reproduces the Rossi reference but favors the prototype-sparse M-L target over centered-Eta support.
- Classic3 K diagnostic (candidate K=2,...,10): AIC/BIC selected the upper boundary K=10; RICc and EBIC-gamma=1 selected K=7 or 8, while external-label ARI was highest at K=3. K selection remains separate from support selection and is not presented as solved.

## Audit status

- Protocol checks passed: 30/30.
- Result-consistency checks passed: 7/7.
- Train/test feature ranking is train-only; test labels are used only for ARI/NMI.
- All exact centered-refit candidates used in the held-out analyses converged without recorded Q or log-likelihood decreases.
- BBC5 duplicate removal occurs before splitting; no normalized duplicate group spans classes.

## Reporting boundary

- Real data do not provide true feature support, so support TPR/FPR/F1 are not reported.
- Selected-q values for M-L and E-CGL/E-ACGL refer to different estimands (prototype union versus posterior decision support).
- Negative continuous-density NLL values are valid; lower is better.
- The BBC5 and CSTR results preclude a claim of universal clustering or density superiority.
