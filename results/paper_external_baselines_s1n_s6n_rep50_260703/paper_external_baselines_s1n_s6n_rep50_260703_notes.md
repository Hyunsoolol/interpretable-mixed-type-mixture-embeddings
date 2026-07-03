# External baseline simulation S1-N-S6-N

Dense decision-support negative-control S1-N~S6-N에 대한 외부 clustering baseline rep=50 결과이다.

- Repetitions: 50
- n=1000, d=200, K=4
- common q=4, decision q=80, noise q=116
- Spherical k-means nstart=10
- Dense vMF free kappa nstart=10
- Sparse k-means tuning: KMeansSparseCluster.permute nperms=5, nvals=8
- dbmovMFs package available: FALSE

Sparse k-means의 support는 feature support이며 posterior decision support가 아니다. Spherical k-means와 Dense vMF free kappa는 sparse support를 추정하지 않는다.

## Summary

| scenario | method | valid | ARI | NMI | purity | selected q | TPR | FPR | Precision | F1 | MSE_eta | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| S1-N | Spherical k-means | 50 | 0.770 | 0.740 | 0.904 | NA | NA | NA | NA | NA | NA | ok |
| S1-N | Dense vMF free kappa | 50 | 0.835 | 0.802 | 0.934 | NA | NA | NA | NA | NA | 0.746 | ok |
| S1-N | Sparse k-means | 50 | 0.133 | 0.155 | 0.437 | 82.16 | 0.618 | 0.273 | 0.811 | 0.574 | NA | ok |
| S1-N | dbmovMFs | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | not_available: dbmovMFs package not installed |
| S2-N | Spherical k-means | 50 | 0.880 | 0.833 | 0.954 | NA | NA | NA | NA | NA | NA | ok |
| S2-N | Dense vMF free kappa | 50 | 0.886 | 0.839 | 0.956 | NA | NA | NA | NA | NA | 0.703 | ok |
| S2-N | Sparse k-means | 50 | 0.058 | 0.068 | 0.397 | 128.32 | 0.719 | 0.590 | 0.591 | 0.520 | NA | ok |
| S2-N | dbmovMFs | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | not_available: dbmovMFs package not installed |
| S3-N | Spherical k-means | 50 | 0.488 | 0.492 | 0.711 | NA | NA | NA | NA | NA | NA | ok |
| S3-N | Dense vMF free kappa | 50 | 0.545 | 0.559 | 0.741 | NA | NA | NA | NA | NA | 2.145 | ok |
| S3-N | Sparse k-means | 50 | 0.063 | 0.075 | 0.388 | 91.56 | 0.514 | 0.421 | 0.720 | 0.380 | NA | ok |
| S3-N | dbmovMFs | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | not_available: dbmovMFs package not installed |
| S4-N | Spherical k-means | 50 | 0.507 | 0.459 | 0.783 | NA | NA | NA | NA | NA | NA | ok |
| S4-N | Dense vMF free kappa | 50 | 0.562 | 0.510 | 0.812 | NA | NA | NA | NA | NA | 1.008 | ok |
| S4-N | Sparse k-means | 50 | 0.026 | 0.033 | 0.352 | 114.00 | 0.615 | 0.540 | 0.540 | 0.445 | NA | ok |
| S4-N | dbmovMFs | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | not_available: dbmovMFs package not installed |
| S5-N | Spherical k-means | 50 | 0.013 | 0.017 | 0.322 | NA | NA | NA | NA | NA | NA | ok |
| S5-N | Dense vMF free kappa | 50 | 0.024 | 0.032 | 0.343 | NA | NA | NA | NA | NA | 4.331 | ok |
| S5-N | Sparse k-means | 50 | 0.003 | 0.007 | 0.295 | 95.44 | 0.480 | 0.476 | 0.453 | 0.342 | NA | ok |
| S5-N | dbmovMFs | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | not_available: dbmovMFs package not installed |
| S6-N | Spherical k-means | 50 | 0.008 | 0.011 | 0.306 | NA | NA | NA | NA | NA | NA | ok |
| S6-N | Dense vMF free kappa | 50 | 0.012 | 0.018 | 0.318 | NA | NA | NA | NA | NA | 4.477 | ok |
| S6-N | Sparse k-means | 50 | 0.003 | 0.006 | 0.293 | 88.20 | 0.445 | 0.439 | 0.427 | 0.335 | NA | ok |
| S6-N | dbmovMFs | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | not_available: dbmovMFs package not installed |
