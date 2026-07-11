# Classic3 K-selection panel

- Selection diagnostics are label-free; labels are used only for post-fit ARI/NMI.
- Bootstrap results use B=10 and are an exploratory stability diagnostic.
- Bootstrap fits use in-bag-only initializations. The SPLADE top-2000 representation remains fixed from the full training split.
- An earlier full-train-initialized bootstrap diagnostic is superseded and is not used here.
- Dense-vMF likelihood/IC and out-of-bag density criteria favor finer partitions near the upper K boundary.
- Bootstrap partition stability peaks at K=3 for both shared- and free-kappa dense vMF fits.
- E-CGL is fitted separately at K=3, 7, 8, and 10; support is selected by exact BIC after fixed-support refit.

## Dense K choices

| kappa model | criterion | selected K | labels used |
|---|---|---:|---|
| shared | BIC | 10 | FALSE |
| shared | RICc |  8 | FALSE |
| shared | EBIC_g0.5 |  8 | FALSE |
| shared | EBIC_g1 |  8 | FALSE |
| shared | bootstrap_OOB_NLL_1SE | 10 | FALSE |
| shared | bootstrap_pairwise_stability |  3 | FALSE |
| free | BIC | 10 | FALSE |
| free | RICc |  7 | FALSE |
| free | EBIC_g0.5 |  9 | FALSE |
| free | EBIC_g1 |  7 | FALSE |
| free | bootstrap_OOB_NLL_1SE | 10 | FALSE |
| free | bootstrap_pairwise_stability |  3 | FALSE |

## E-CGL exact-BIC comparison

| K | selected q | test NLL/doc | test ARI | purity | homogeneity | completeness |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 1347 | -4872.2942 | 0.9927 | 0.9974 | 0.9867 | 0.9859 |
| 7 | 1105 | -4905.8263 | 0.5852 | 0.9961 | 0.9805 | 0.5861 |
| 8 | 1063 | -4910.6019 | 0.4927 | 0.9961 | 0.9804 | 0.5362 |
| 10 | 980 | -4917.5464 | 0.3982 | 0.9923 | 0.9718 | 0.4752 |

## Interpretation

The primary Classic3 benchmark is fixed at the three externally supplied topic categories. Exploratory in-bag bootstrap stability is highest at K=3, whereas likelihood-based criteria continue to reward finer partitions near the upper K boundary. E-CGL is therefore interpreted as conditional support selection at a fixed K rather than a method for selecting K itself.

Selected q is not interpreted as a cross-K sparsity ranking because the active-coordinate degrees-of-freedom cost changes with K.
