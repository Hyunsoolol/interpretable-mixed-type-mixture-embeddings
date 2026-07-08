# Study B difficulty sensitivity rep=50 diagnostic

Intermediate diagnostic before final rep=100/200 simulation. No document update or git staging was performed.

## Setting

- K=4, d=200, n in {300, 1000}
- target oracle Bayes error e_B in {0.025, 0.05, 0.10}
- common q=4, decision q=16, noise q=180
- scenarios: equal_kappa and heterogeneous_kappa
- methods: D-L, D-GL, D-AGL, E-L, E-GL, E-AGL
- rep=50, nstart=10, max_iter=100, path length=240, BIC-before-refit plus support refit, USE_RCPP_HELPERS=1

## Runtime

- e_B=0.025, n=300: 37.8 min
- e_B=0.025, n=1000: 67.4 min
- e_B=0.050, n=300: 42.1 min
- e_B=0.050, n=1000: 66.0 min
- e_B=0.100, n=300: 50 min
- e_B=0.100, n=1000: 70.6 min

## Calibration check

 target_eB    n            scenario achieved_oracle_error
     0.025  300         equal_kappa                0.0242
     0.025  300 heterogeneous_kappa                0.0254
     0.025 1000         equal_kappa                0.0242
     0.025 1000 heterogeneous_kappa                0.0254
     0.050  300         equal_kappa                0.0458
     0.050  300 heterogeneous_kappa                0.0454
     0.050 1000         equal_kappa                0.0458
     0.050 1000 heterogeneous_kappa                0.0454
     0.100  300         equal_kappa                0.1124
     0.100  300 heterogeneous_kappa                0.1052
     0.100 1000         equal_kappa                0.1124
     0.100 1000 heterogeneous_kappa                0.1052

The target e_B=0.10 equal-kappa calibration reached 0.1124, so that cell is slightly harder than the nominal 10% target. The heterogeneous e_B=0.10 cell reached 0.1052.

## Validation check

- valid_reps=50 for all rows: FALSE
- rows with zero_support_refit_reps > 0: 1
 target_eB   n    scenario method zero_support_refit_reps selected_q    F1 ARI
       0.1 300 equal_kappa  E-AGL                       1      16.98 0.942 0.7

## E-AGL rows

 target_eB achieved_eB    n            scenario selected_q common_q decision_q
     0.025      0.0242  300         equal_kappa      17.92     0.00      16.00
     0.025      0.0254  300 heterogeneous_kappa      18.00     0.02      16.00
     0.025      0.0242 1000         equal_kappa      16.04     0.00      16.00
     0.025      0.0254 1000 heterogeneous_kappa      16.04     0.00      16.00
     0.050      0.0458  300         equal_kappa      16.76     0.00      16.00
     0.050      0.0454  300 heterogeneous_kappa      18.24     0.04      16.00
     0.050      0.0458 1000         equal_kappa      16.12     0.00      16.00
     0.050      0.0454 1000 heterogeneous_kappa      16.06     0.00      16.00
     0.100      0.1124  300         equal_kappa      16.98     0.00      15.54
     0.100      0.1052  300 heterogeneous_kappa      21.48     0.12      14.88
     0.100      0.1124 1000         equal_kappa      16.24     0.00      16.00
     0.100      0.1052 1000 heterogeneous_kappa      21.66     0.18      16.00
 noise_q    F1   ARI MSE_eta
    1.92 0.943 0.931   0.257
    1.98 0.941 0.933   0.264
    0.04 0.999 0.935   0.053
    0.04 0.999 0.935   0.051
    0.76 0.977 0.875   0.231
    2.20 0.935 0.868   0.288
    0.12 0.996 0.879   0.058
    0.06 0.998 0.876   0.056
    1.44 0.942 0.700   0.348
    6.48 0.794 0.681   0.814
    0.24 0.993 0.720   0.069
    5.48 0.850 0.735   0.114

## Interpretation

- The easier e_B=0.025 setting increases ARI for all methods, as expected.
- E-AGL keeps most or all decision coordinates in the summary rows. At n=1000 it removes common coordinates and nearly all noise in easy/moderate settings, but the hard heterogeneous setting keeps additional noise coordinates.
- At n=300, E-AGL shows modest over-selection, especially in heterogeneous or harder settings. In the hard equal-kappa setting, E-AGL has one zero-support refit replicate and a small decision-q loss.
- D-GL/D-AGL tend to select the common q=4 coordinates because their target is prototype/direction support rather than posterior decision support.
- E-AGL remains the strongest candidate for the main proposed method; E-GL should remain the non-adaptive eta-group baseline.

## Recommendation

Before broad rep=100 final, prioritize checking the hard e_B=0.10 cells and keep achieved oracle Bayes error in all tables. If final runtime is constrained, run rep=100 first for e_B=0.10 and e_B=0.05, then decide whether e_B=0.025 needs final replication.
