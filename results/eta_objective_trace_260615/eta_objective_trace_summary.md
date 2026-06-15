# Eta Objective Trace Smoke Test 260615

## Setting

- K = 4
- n = 300
- d = 100
- common variables = 6
- component-specific variables = 4 per component
- w = 0.50
- kappa = (40, 50, 60, 70)
- random start = 5
- update = unpenalized eta M-step followed by centered eta proximal shrinkage

## Summary

| lambda_eta | iter | converged | active q | decreases | min objective diff | first objective | last objective |
|---:|---:|:---:|---:|---:|---:|---:|---:|
| 0 | 2 | yes | 100 | 0 | 0.00132605 | 29307.430149 | 29307.431475 |
| 0.41969 | 13 | yes | 97 | 0 | 0.00202857 | 29137.791064 | 29139.310185 |
| 2.54571 | 33 | yes | 27 | 22 | -0.048025 | 28666.857738 | 28758.424348 |
| 3.9433 | 16 | yes | 0 | 0 | 0 | 28644.728193 | 28909.366621 |
| 14.8299 | 2 | yes | 0 | 0 | 0 | 28909.366621 | 28909.366621 |

## Note

At least one lambda path candidate had a decreasing penalized objective step. The current update should be described as a proximal EM-type heuristic, not a guaranteed monotone EM algorithm. A line-search or MM safeguard is recommended for a paper-grade algorithm.
