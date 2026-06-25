# Negative-control Simulation Summary 260708

## 1. Purpose

This note summarizes diagnostic simulations designed to identify settings where the Eta-group penalty can be worse than Rossi or Separate penalties. These runs are negative-control diagnostics, not a change to the official algorithm.

Official method remains Eta-group path+BIC + selected support fixed unpenalized refit.

## 2. Existing Smoke Results

| Setting | Main design | Main result | Interpretation |
|:---|:---|:---|:---|
| A: direction-sparse smoke | $K=4$, $d=100$, true union $q=20$, $\kappa=(60,60,60,60)$ | Eta + refit ARI=0.976, selected q=28.8, F1=0.676; Rossi/Separate ARI=0.975, selected q=100, F1=0.333 | Not a strong Rossi/Separate-favorable setting. Eta still gives better support, but Eta refit has large MSE_kappa. |
| B: dense eta smoke | $K=4$, $d=100$, true union $q=80$, $\kappa=(30,45,65,90)$ | Eta + refit ARI=0.377, selected q=47.4, F1=0.702; Separate + refit ARI=0.396, selected q=100, F1=0.889 | Good negative-control candidate. Dense truth makes Eta-group over-shrink support and lose ARI/F1. |
| C: weak signal smoke | $K=4$, $d=100$, $w=0.20$, $\kappa=(25,30,35,40)$ | Eta BIC selected q=0 and refit invalid; Rossi/Separate ARI near 0 | Too difficult. This is a low-signal stress where almost all methods fail. |

## 3. Setting B Rep50: Dense Eta / Weak Sparsity Truth

Design:

| Item | Value |
|:---|:---|
| K | 4 |
| n | 1000 |
| d | 100 |
| rep | 50 |
| common q | 20 |
| specific q per component | 15 |
| true union q | 80 |
| specific weight | 0.25 |
| kappa | $(30,45,65,90)$ |
| selection | BIC |

Result:

| Method | ARI | Selected q | TPR | FPR | Precision | F1 | MSE_mu | MSE_kappa | MSE_centered_eta |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Rossi BIC | 0.384 | 99.90 | 1.000 | 0.996 | 0.801 | 0.889 | 0.000886 | 70.526 | 2.236 |
| Rossi BIC + refit | 0.380 | 99.90 | 1.000 | 0.996 | 0.801 | 0.889 | 0.000993 | 78.229 | 2.544 |
| Separate BIC | 0.381 | 99.74 | 1.000 | 0.988 | 0.802 | 0.890 | 0.000956 | 52.943 | 1.753 |
| Separate BIC + refit | 0.378 | 99.74 | 1.000 | 0.988 | 0.802 | 0.890 | 0.001029 | 50.549 | 2.150 |
| Eta-group BIC | 0.324 | 52.82 | 0.615 | 0.180 | 0.944 | 0.726 | 0.000557 | 186.886 | 2.895 |
| Eta-group BIC + refit | 0.368 | 52.82 | 0.615 | 0.180 | 0.944 | 0.726 | 0.000897 | 93.913 | 2.721 |

Conclusion:

- Setting B confirms a clear Eta-group failure mode.
- When the true decision support is dense, Eta-group selects a much smaller support than the true union q=80.
- This improves FPR and Precision but loses TPR, F1, ARI, and MSE_centered_eta relative to Separate.
- This is a useful negative-control result for the paper because it clarifies that Eta-group is not designed for dense weak-sparsity truth.

## 4. Setting C2 Smoke: Moderated Weak Signal

Design:

| Item | Value |
|:---|:---|
| K | 4 |
| n | 1000 |
| d | 100 |
| rep | 5 |
| common q | 6 |
| specific q per component | 4 |
| true union q | 22 |
| specific weight | 0.25 |
| kappa | $(35,45,55,65)$ |
| selection | BIC |

Result:

| Method | ARI | Selected q | TPR | FPR | Precision | F1 | MSE_mu | MSE_kappa | MSE_centered_eta |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Rossi BIC | 0.148 | 100.00 | 1.000 | 1.000 | 0.220 | 0.361 | 0.001067 | 47.891 | 2.885 |
| Rossi BIC + refit | 0.132 | 100.00 | 1.000 | 1.000 | 0.220 | 0.361 | 0.001449 | 51.147 | 3.859 |
| Separate BIC | 0.138 | 99.40 | 1.000 | 0.992 | 0.221 | 0.362 | 0.001010 | 69.650 | 2.383 |
| Separate BIC + refit | 0.138 | 99.40 | 1.000 | 0.992 | 0.221 | 0.362 | 0.001383 | 68.391 | 3.510 |
| Eta-group BIC | 0.000 | 0.00 | 0.000 | 0.000 | NA | NA | 0.000356 | 128.990 | 1.988 |
| Eta-group BIC + refit | NA | 0.00 | 0.000 | 0.000 | NA | NA | NA | NA | NA |
| Eta-group positive-support + refit | 0.136 | 15.80 | 0.491 | 0.064 | 0.770 | 0.577 | 0.000952 | 69.789 | 2.428 |

Conclusion:

- C2 is less degenerate than the first weak-signal smoke but still difficult.
- Standard Eta BIC selects zero support, so the official BIC-selected refit is invalid.
- Positive-support selection avoids zero support and gives sparse support with higher F1 than Rossi/Separate, but ARI remains low.
- C2 is useful as a tuning/zero-support failure diagnostic, not as a clean Rossi/Separate-favorable simulation.

## 5. Setting A Redesign

The first Setting A is not a strong Rossi/Separate-favorable setting because the current support metric is coordinate union support, which favors Eta-group when it removes common non-discriminating coordinates. Rossi and Separate select dense support, while Eta-group keeps a smaller decision-contrast support.

To make a fair Rossi-favorable setting, the target should be clarified:

- If the target is posterior decision support, Eta-group is naturally favored.
- If the target is prototype or direction support, Rossi may be more appropriate.

Suggested A2 design:

1. Use equal or near-equal concentration so $\eta$ contrast is close to $\mu$ contrast.
2. Construct component-specific direction patterns with minimal common support.
3. Evaluate both prototype support recovery and decision support recovery separately.
4. Report entry-level $\mu$ support metrics in addition to coordinate union support.

## 6. Setting A2 Smoke: Direction-Sparse / Equal Concentration

Design:

| Item | Value |
|:---|:---|
| K | 4 |
| n | 1000 |
| d | 100 |
| rep | 5 |
| common q | 1 |
| specific q per component | 5 |
| true union q | 21 |
| specific weight | 1.0 |
| kappa | $(60,60,60,60)$ |
| selection | BIC |

Result:

| Method | ARI | Selected q | TPR | FPR | Precision | F1 | entry_TPR | entry_FPR | MSE_mu | MSE_kappa | MSE_centered_eta |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Rossi BIC | 0.999 | 100.00 | 1.000 | 1.000 | 0.210 | 0.347 | 1.000 | 0.813 | 0.000091 | 1.344 | 0.262 |
| Rossi BIC + refit | 0.999 | 100.00 | 1.000 | 1.000 | 0.210 | 0.347 | NA | NA | 0.000130 | 1.383 | 0.373 |
| Separate BIC | 0.999 | 100.00 | 1.000 | 1.000 | 0.210 | 0.347 | 1.000 | 0.725 | 0.000077 | 1.306 | 0.222 |
| Separate BIC + refit | 0.999 | 100.00 | 1.000 | 1.000 | 0.210 | 0.347 | NA | NA | 0.000130 | 1.383 | 0.373 |
| Eta-group BIC | 0.998 | 40.60 | 0.962 | 0.258 | 0.502 | 0.658 | NA | NA | 0.000067 | 4.256 | 0.160 |
| Eta-group BIC + refit | 0.998 | 40.60 | 0.962 | 0.258 | 0.502 | 0.658 | NA | NA | 0.001473 | 43.610 | 0.250 |

Conclusion:

- A2 does not produce a clean Rossi/Separate-favorable result under the current coordinate union support metric.
- Rossi and Separate recover all true coordinates but also select almost every noise coordinate, so selected q=100 and FPR=1.000.
- Eta-group keeps ARI essentially unchanged while reducing selected q to 40.60 and improving union-support F1.
- The only metric where Rossi/Separate look structurally relevant is entry-level prototype support: entry_TPR=1.000 with entry_FPR=0.813 for Rossi and 0.725 for Separate.
- Therefore A2 shows a metric-definition issue rather than a clear method failure: to evaluate Rossi-style direction sparsity fairly, we need prototype/entry-level support recovery in addition to posterior decision support.

## 7. Fragmented Block-like Smoke: No Shared Coordinates

The proposed Rossi/Separate-favorable setting was tested in the current generator by setting common support to zero and using component-specific supports only. This is not a perfect binary block-diagonal generator, but it is the closest diagnostic available without modifying the core simulation code.

### 7.1 Low-dimensional fragmented smoke

Design: $K=4$, $n=1000$, $d=60$, rep=5, common q=0, specific q=10 per component, true union q=40, $\kappa=(60,60,60,60)$.

| Method | ARI | Selected q | FPR | Precision | F1 | MSE_mu | MSE_kappa | MSE_centered_eta |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| Rossi BIC | 1.000 | 59.00 | 0.950 | 0.678 | 0.808 | 0.000049 | 1.481 | 0.156 |
| Separate BIC | 1.000 | 56.80 | 0.840 | 0.705 | 0.827 | 0.000034 | 1.394 | 0.112 |
| Eta-group BIC | 0.999 | 42.80 | 0.140 | 0.936 | 0.967 | 0.000109 | 11.101 | 0.458 |
| Eta-group BIC + refit | 0.999 | 42.80 | 0.140 | 0.936 | 0.967 | 0.000079 | 1.490 | 0.244 |

Interpretation:

- This low-dimensional fragmented setting does not make Rossi/Separate clearly superior overall.
- Rossi/Separate have perfect ARI and better raw prototype-parameter MSE for Separate BIC, but they still select dense supports.
- Eta-group keeps ARI essentially unchanged and gives much sparser decision support.
- This again suggests that prototype-parameter accuracy and decision-support sparsity should be reported separately.

### 7.2 High-dimensional fragmented smoke

Design: $K=4$, $n=1000$, $d=400$, rep=3, common q=0, specific q=20 per component, true union q=80, $\kappa=(60,60,60,60)$.

| Method | ARI | Selected q | FPR | Precision | F1 | MSE_mu | MSE_kappa | MSE_centered_eta |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| Rossi BIC | 0.826 | 400.00 | 1.000 | 0.200 | 0.333 | 0.000425 | 48.931 | 1.417 |
| Separate BIC | 0.827 | 400.00 | 1.000 | 0.200 | 0.333 | 0.000425 | 25.462 | 1.329 |
| Eta-group BIC | 0.851 | 368.33 | 0.901 | 0.217 | 0.357 | 0.000310 | 4.283 | 0.736 |
| Eta-group BIC + refit | 0.832 | 368.33 | 0.901 | 0.217 | 0.357 | 0.000419 | 46.142 | 1.408 |

Interpretation:

- In this high-dimensional version, all methods become dense.
- Rossi/Separate do not gain the expected clear advantage; Eta-group is also dense but slightly less dense.
- Therefore the current generator does not reproduce the proposed "Rossi dominates under block-diagonal fragmented data" scenario.
- A truly Rossi-favorable test likely needs a dedicated block-diagonal or binary-style generator and prototype-support metrics.

### 7.3 d=800 attempt

A d=800 fragmented smoke was attempted with common q=0 and specific q=20 per component. The runs completed the repetitions but failed during summary binding with a column mismatch error. Because the R core algorithm should not be modified in this diagnostic step, this result is not used. The failure is recorded as a script robustness issue for very high-dimensional diagnostic settings.

## 8. Current Conclusion

The most useful negative-control result is Setting B rep50.

It shows that Eta-group is not universally better. When the true separation is dense and many weak coordinates matter, Eta-group can over-shrink the support and lose ARI/F1 relative to Rossi or Separate. This should be included as a limitation or negative-control diagnostic.

Setting C2 shows another limitation: BIC can select zero Eta support under weak signal, so positive-support or alternative tuning may be needed as a diagnostic. This should not be presented as official tuning.

Setting A2 shows that even an equal-concentration direction-sparse design still favors Eta-group under coordinate union support. This does not prove that Rossi/Separate cannot be better; it shows that prototype-level and decision-level support targets must be separated before making that comparison.

The fragmented low-dimensional and high-dimensional smokes do not yet create a clean Rossi/Separate-dominant result under the current generator. They show that all methods can cluster well or become dense, but they do not isolate the prototype-sparsity advantage. A dedicated block-diagonal generator is needed if the goal is to demonstrate a setting where Rossi/Separate are structurally favored.

## 9. Full Simulation Recommendation

| Candidate | Recommendation | Reason |
|:---|:---|:---|
| Setting B rep100 | Optional, not urgent | rep50 already gives stable negative-control evidence. |
| Setting C2 rep50 | Not yet | Need a cleaner weak-signal setting where Rossi/Separate do not collapse or go fully dense. |
| Setting A2 rep50 | Not recommended yet | Smoke result is not Rossi/Separate-favorable under union support. Need a prototype-support metric first. |
| Prototype-support metric | Recommended next | Needed to evaluate Rossi/Separate on their natural target. |
| Dedicated block-diagonal generator | Recommended after metric definition | Needed to test the proposed Rossi-favorable fragmented-data scenario cleanly. |
