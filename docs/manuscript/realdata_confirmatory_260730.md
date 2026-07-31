# Locked repeated-holdout real-data analysis

## 1. Scope

The real-data analysis evaluates whether centered coordinate regularization
can reduce a vocabulary-aligned directional representation while retaining
held-out clustering structure. It does not treat selected coordinates as
ground-truth relevant variables.

The datasets, representation, and top-$d$ choices were fixed after exploratory
screening. The five locked splits therefore provide conditional repeated-
holdout validation, not an independently preregistered confirmatory study.

- **Classic3:** main illustration of sparse, interpretable posterior-score
  contrasts.
- **BBCSport:** contrast case in which useful information is more diffuse.
- **CSTR:** literature bridge used to validate the Rossi-style implementation;
  its earlier E-series diagnostic is not pooled with the locked analysis.

## 2. Locked design

| Dataset | $K$ | Raw $n$ | Exact duplicates removed | Near duplicates removed | Final $n$ | Train/test per split | $d$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classic3 | 3 | 3,890 | 2 | 5 | 3,883 | 3,105 / 778 | 2,000 |
| BBCSport | 5 | 737 | 13 | 20 | 704 | 561 / 143 | 500 |

Classic3 comprises the CISI, CRAN, and MED document collections used in the
directional-clustering study of
[Banerjee et al. (2005)](https://jmlr.org/papers/v6/banerjee05a.html).
BBCSport comprises athletics, cricket, football, rugby, and tennis articles
associated with the document-clustering benchmarks of
[Greene and Cunningham (2006)](https://doi.org/10.1145/1143844.1143892).

Both datasets used five locked stratified 80/20 splits. Near-duplicate removal
preceded splitting. Vocabulary ranking was performed on the training portion
of each split only. No test document had maximum cosine similarity of at least
0.98 with a training document.

Documents were represented by nonnegative SPLADE coordinates using
`naver/splade-cocondenser-ensembledistil` at revision
`49cf4c7b0db5b870a401ddf5e2669993ef3699c7`, followed by row normalization to
the unit sphere. The vocabulary consisted of the training-set variance-ranked
top 2,000 coordinates for Classic3 and top 500 coordinates for BBCSport.

The benchmark class count was fixed during fitting. Supplied labels were used
only for stratification, verification of $K$, post-fit ARI/NMI evaluation, and
post-fit component naming and visualization.

## 3. Methods and targets

| Method | Concentration | Coordinate target |
|---|---|---|
| Spherical $k$-means | not estimated | dense clustering baseline |
| Dense vMF | shared $\kappa$ | dense likelihood baseline |
| Dense vMF | free $\kappa_k$ | dense likelihood baseline |
| M-L | shared $\kappa$ | Rossi-style prototype support |
| M-CGL | free $\kappa_k$ | centered directional support |
| M-ACGL | free $\kappa_k$ | adaptive centered directional support |
| E-CGL | free $\kappa_k$ | centered posterior-score support |
| E-ACGL | free $\kappa_k$ | adaptive centered posterior-score support |

The three sparse targets differ:

$$
S_P
=
\left\{j:\|\mu_{\cdot j}\|_2>0\right\},
$$

$$
S_\mu
=
\left\{j:\|\mu_{\cdot j}-\bar\mu_j\mathbf 1_K\|_2>0\right\},
$$

$$
S_\eta
=
\left\{j:\|\eta_{\cdot j}-\bar\eta_j\mathbf 1_K\|_2>0\right\}.
$$

Consequently, selected $q$ is interpreted within each target rather than as a
common estimand across M-L, M-CGL, and E-CGL.

## 4. Estimation and evaluation

- Dense and spherical initialization budget: 30 random starts. Penalized paths
  were warm-started from the selected matched dense fit.
- Centered M/E paths: 240 candidates.
- M-L path: 600 candidates.
- Sparse-model selection: method-specific BIC computed from the training
  observed log-likelihood after a target-preserving support-constrained refit.
- Dense match for paired comparisons: shared-$\kappa$ for M-L and
  free-$\kappa_k$ for M-CGL, M-ACGL, E-CGL, and E-ACGL.
- Metrics: held-out ARI, held-out NMI, held-out negative log-likelihood per
  document, selected $q$, conditional support Jaccard, runtime, warnings,
  convergence, and path-boundary status.

For an active set of size $m$, the nominal centered-$\eta$ dimension was

$$
\operatorname{df}_\eta
=
d+(K-1)m+(K-1)\mathbf 1(m>0).
$$

The centered-$\mu$ nominal dimension was

$$
\operatorname{df}_\mu
=
\begin{cases}
d+2K-2, & m=0,\\
d+(K-1)m+K-1, & m>0.
\end{cases}
$$

M-L used the number of active prototype entries, consistent with the
Rossi-style implementation. An inactive centered-$\eta$ coordinate retained a
common natural-parameter baseline; an inactive centered-$\mu$ coordinate
retained a common directional coordinate under the row-unit-norm constraints.
These are practical nominal dimensions for support selection in a nonregular
finite mixture, not exact marginal-likelihood degrees of freedom.

The five holdouts overlap. Means, standard deviations, and paired
win/tie/loss counts are descriptive; no independent-replicate $t$-test is
reported.

Reported time is the mean recorded pipeline time across splits. It includes
dense initialization, path construction, and refitting. The jobs ran with up
to six splits in parallel on an Intel i7-11700K workstation with 32 GB memory
under R 4.2.1. Because M-L and centered paths used different path lengths and
different optimization routines, these values are workload records rather
than an algorithmic complexity comparison.

## 5. Classic3

| Method | Held-out ARI | Held-out NMI | Selected $q$ | $q/d$ | NLL/doc | Pipeline (s) |
|---|---:|---:|---:|---:|---:|---:|
| Spherical $k$-means | 0.970 (0.007) | 0.946 (0.010) | 2,000.0 | 1.000 | NA | 18.7 |
| Dense shared-$\kappa$ | 0.970 (0.007) | 0.946 (0.010) | 2,000.0 | 1.000 | -4,872.890 | 47.6 |
| Dense free-$\kappa_k$ | 0.973 (0.006) | 0.953 (0.011) | 2,000.0 | 1.000 | -4,873.968 | 60.6 |
| M-L | 0.970 (0.009) | 0.947 (0.013) | 1,924.6 | 0.962 | -4,872.239 | 2,320.7 |
| M-CGL | 0.970 (0.007) | 0.949 (0.011) | 1,376.8 | 0.688 | -4,873.337 | 6,063.2 |
| M-ACGL | 0.971 (0.007) | 0.950 (0.011) | 1,345.4 | 0.673 | -4,873.298 | 6,669.8 |
| **E-CGL** | **0.970 (0.007)** | **0.947 (0.012)** | **1,343.0** | **0.671** | **-4,873.297** | **2,429.6** |
| E-ACGL | 0.973 (0.006) | 0.953 (0.011) | 2,000.0 | 1.000 | -4,873.967 | 2,325.6 |

Relative to dense free-$\kappa_k$, E-CGL excluded 32.9% of the coordinates
from the estimated posterior-score contrast support while retaining their
common baselines. Its mean paired differences were

$$
\Delta\mathrm{ARI}
=
-0.0036,
\qquad
\Delta\mathrm{NLL/doc}
=
0.6705.
$$

ARI was not higher in one of five splits and lower in four; NLL was higher in
all five splits. The mean ARI decrease was 0.0036, and the mean held-out NLL
increase was 0.6705 per document. Conditional on coordinates available in both
training vocabularies, the mean support Jaccard was 0.933. E-ACGL selected the
dense support in every split and is therefore retained as an adaptive
sensitivity result rather than the primary fit.

The E-CGL centered-$\eta$ contrasts were stable across the locked splits.
After fitting, components were aligned to the benchmark classes for naming and
visualization. Tokens available and selected in all five splits included:

| Class | Leading positive contrast tokens |
|---|---|
| CISI | library, information, librarian, libraries, retrieval |
| CRAN | flow, mach, pressure, heat, theory |
| MED | tumor, inhibitor, rat, dose, cancer |

![Classic3 centered-Eta contrasts](figures/classic3_locked_ecgl_centered_eta_heatmap_260730.png)

## 6. BBCSport contrast case

| Method | Held-out ARI | Held-out NMI | Selected $q$ | $q/d$ | NLL/doc | Pipeline (s) |
|---|---:|---:|---:|---:|---:|---:|
| Spherical $k$-means | 0.894 (0.022) | 0.897 (0.021) | 500.0 | 1.000 | NA | 1.5 |
| Dense shared-$\kappa$ | 0.898 (0.024) | 0.901 (0.020) | 500.0 | 1.000 | -922.870 | 15.2 |
| Dense free-$\kappa_k$ | 0.877 (0.048) | 0.878 (0.033) | 500.0 | 1.000 | -922.622 | 15.8 |
| M-L | 0.907 (0.020) | 0.907 (0.019) | 498.2 | 0.996 | -921.605 | 116.7 |
| M-CGL | 0.880 (0.050) | 0.882 (0.035) | 303.2 | 0.606 | -921.086 | 1,516.6 |
| M-ACGL | 0.880 (0.050) | 0.882 (0.035) | 302.4 | 0.605 | -920.979 | 1,533.0 |
| **E-CGL** | **0.875 (0.053)** | **0.874 (0.042)** | **308.6** | **0.617** | **-921.211** | **491.6** |
| E-ACGL | 0.877 (0.048) | 0.878 (0.033) | 285.2 | 0.570 | -920.796 | 530.4 |

E-CGL excluded 38.3% of the coordinates from the estimated posterior-score
contrast support, but its mean paired differences from dense
free-$\kappa_k$ were

$$
\Delta\mathrm{ARI}
=
-0.0026,
\qquad
\Delta\mathrm{NLL/doc}
=
1.4105.
$$

Every sparse method had higher held-out NLL than its matched dense model in all
five splits. M-L had mean ARI 0.907 and retained 99.6% of the coordinates among
the evaluated path candidates. BBCSport therefore provides a contrast case:
BIC-selected sparse fits incurred held-out density loss in every split.

Raw cross-family ranks do not isolate a penalty effect because M-L used shared
$\kappa$, whereas centered M/E methods used free $\kappa_k$. The paired
sparse-minus-matched-dense differences are the primary comparison.

## 7. Numerical validation

| Dataset | Completed rows | Errors | Nonconverged | Terminal selections | Path fit stops | Known Bessel warning rows | Other warning rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classic3 | 40/40 | 0 | 0 | 0 | 0 | 10 | 0 |
| BBCSport | 40/40 | 0 | 0 | 0 | 5 | 0 | 0 |

Classic3 M-CGL/M-ACGL produced high-order Bessel precision warnings. A separate
audit over the selected-fit range, $d=2{,}000$ and
$\kappa\in[607,818]$, compared the production direct/fallback calculation with
the high-dimensional reference calculation:

$$
\max |\Delta A_d(\kappa)|
=
8.983\times10^{-11},
$$

$$
\max \operatorname{relative\ error}
=
2.850\times10^{-10}.
$$

The observed range passed the prespecified relative tolerance of $10^{-6}$.
This is a range-specific numerical audit, not a general error bound.

BBCSport M-L stopped when the next stronger penalty caused component-wise
prototype-support collapse. In every case, the BIC-selected support was an
interior path candidate rather than the failed endpoint. A sensitivity run
increased the maximum path updates from 600 to 1,200 and reduced the minimum
relative penalty increment from 0.02 to 0.005. Mean path rows increased from
331.8 to 771.2, while selected $q$ differed by at most one coordinate and
held-out ARI/NMI were unchanged in every split. The terminal evaluated
support had BIC at least 19,497.94 larger than the selected support. The denser
path required 1.76 times the M-L runtime on average and still ended in a later
fit failure. Thus the near-dense M-L selection is stable to the examined path
resolution, while inference beyond the evaluated path remains unsupported.

![Held-out ARI and coordinate retention](figures/realdata_locked_ari_retention_boxplots_260730.png)

## 8. CSTR literature bridge

Under the Rossi and Barbaro CSTR setting, the implementation reproduced the
reported ARI values:

| Method | Published ARI | Implementation ARI |
|---|---:|---:|
| Dense shared-$\kappa$ vMF | 0.804 | 0.8023 (0.0087) |
| Rossi M-L, BIC | 0.808 | 0.8083 (0.0079) |

The earlier centered-$\eta$ CSTR diagnostic used BIC before refit and is not
combined numerically with the present locked BIC-after analysis. CSTR is used
only as a reproduction check for the Rossi-style implementation.

## 9. Empirical conclusion

Classic3 is the main interpretive illustration: E-CGL retained 67.1% of the
coordinates in its estimated posterior-score contrast support, had
conditional support Jaccard 0.933, and produced coherent component contrasts
after post-fit class alignment. Its held-out ARI was lower and its NLL was
higher than the matched dense model on average. BBCSport showed held-out NLL
loss for every sparse method in every split. The real-data evidence concerns
training-selected coordinate compression, split-conditional stability, and
post-hoc interpretation, not support recovery or universal predictive and
clustering superiority.
