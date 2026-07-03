# External baseline simulation S1-S6

Diagnostic external baselines for the paper eta-first S1-S6 simulation scenarios.

- Repetitions: 50
- n=1000, d=200, K=4
- Spherical k-means nstart=10
- Dense vMF free kappa nstart=10
- Sparse k-means package available: TRUE
- Sparse k-means tuning: KMeansSparseCluster.permute nperms=5, nvals=8
- dbmovMFs package available: FALSE

Support metrics for Sparse k-means are feature-support diagnostics, not posterior decision support.
Spherical k-means and Dense vMF do not estimate sparse support; their support columns are NA.
dbmovMFs was not run when the package was unavailable.

## Summary

| scenario | method | valid | ARI | NMI | purity | selected q | TPR | FPR | Precision | F1 | MSE_eta | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| S1 | Spherical k-means | 50 | 0.768 | 0.740 | 0.903 | NA | NA | NA | NA | NA | NA | ok |
| S1 | Dense vMF free kappa | 50 | 0.836 | 0.801 | 0.934 | NA | NA | NA | NA | NA | 0.741 | ok |
| S1 | Sparse k-means | 50 | 0.669 | 0.669 | 0.826 | 52.36 | 0.950 | 0.202 | 0.645 | 0.674 | NA | ok |
| S1 | dbmovMFs | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | not_available: dbmovMFs package not installed |
| S2 | Spherical k-means | 50 | 0.877 | 0.829 | 0.952 | NA | NA | NA | NA | NA | NA | ok |
| S2 | Dense vMF free kappa | 50 | 0.880 | 0.833 | 0.954 | NA | NA | NA | NA | NA | 0.709 | ok |
| S2 | Sparse k-means | 50 | 0.815 | 0.772 | 0.915 | 132.78 | 0.939 | 0.640 | 0.411 | 0.413 | NA | ok |
| S2 | dbmovMFs | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | not_available: dbmovMFs package not installed |
| S3 | Spherical k-means | 50 | 0.492 | 0.498 | 0.713 | NA | NA | NA | NA | NA | NA | ok |
| S3 | Dense vMF free kappa | 50 | 0.539 | 0.552 | 0.732 | NA | NA | NA | NA | NA | 2.291 | ok |
| S3 | Sparse k-means | 50 | 0.488 | 0.505 | 0.702 | 162.48 | 0.995 | 0.797 | 0.129 | 0.205 | NA | ok |
| S3 | dbmovMFs | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | not_available: dbmovMFs package not installed |
| S4 | Spherical k-means | 50 | 0.508 | 0.461 | 0.783 | NA | NA | NA | NA | NA | NA | ok |
| S4 | Dense vMF free kappa | 50 | 0.561 | 0.508 | 0.812 | NA | NA | NA | NA | NA | 1.021 | ok |
| S4 | Sparse k-means | 50 | 0.129 | 0.139 | 0.470 | 73.10 | 0.757 | 0.331 | 0.437 | 0.367 | NA | ok |
| S4 | dbmovMFs | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | not_available: dbmovMFs package not installed |
| S5 | Spherical k-means | 50 | 0.015 | 0.019 | 0.324 | NA | NA | NA | NA | NA | NA | ok |
| S5 | Dense vMF free kappa | 50 | 0.029 | 0.036 | 0.348 | NA | NA | NA | NA | NA | 4.215 | ok |
| S5 | Sparse k-means | 50 | 0.023 | 0.031 | 0.337 | 99.40 | 0.619 | 0.486 | 0.256 | 0.173 | NA | ok |
| S5 | dbmovMFs | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | not_available: dbmovMFs package not installed |
| S6 | Spherical k-means | 50 | 0.009 | 0.013 | 0.311 | NA | NA | NA | NA | NA | NA | ok |
| S6 | Dense vMF free kappa | 50 | 0.011 | 0.018 | 0.317 | NA | NA | NA | NA | NA | 4.679 | ok |
| S6 | Sparse k-means | 50 | 0.010 | 0.016 | 0.312 | 105.30 | 0.603 | 0.520 | 0.168 | 0.142 | NA | ok |
| S6 | dbmovMFs | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | not_available: dbmovMFs package not installed |
