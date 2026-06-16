# High-dimensional tuning sensitivity 260624

Diagnostic-only recalculation from stored Eta path candidates. No new simulation was run.

## d=200
- BIC selected_q_mean=120.06, FPR=0.552, F1=0.331.
- Best F1 criterion: BIC_current, selected_q_mean=120.06, FPR=0.552, F1=0.331.
- Most sparse/dense-controlled criterion: RIC_like, zero_rate=0.00, dense_rate=0.20, ARI=0.459, F1=0.331.
- Path structure: only 2% of replications contain a q=17-27 candidate, and the mean minimum path support is 119.92.

## d=400
- BIC selected_q_mean=262.95, FPR=0.642, F1=0.146.
- Best F1 criterion: RIC_like, selected_q_mean=262.90, FPR=0.642, F1=0.146.
- Most sparse/dense-controlled criterion: RIC_like, zero_rate=0.00, dense_rate=0.30, ARI=0.206, F1=0.146.
- Path structure: no replication contains a q=17-27 candidate, and the mean minimum path support is 262.90.

Interpretation: EBIC/RIC-like/log(d)-slope recalculation does not materially change the selected path candidates. The high-dimensional failure is therefore not just a BIC penalty-strength problem; the stored Eta path rarely contains a true-support-size candidate to select.
