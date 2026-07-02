# R-only vs Rcpp-helper rep=50 runtime diagnostic

- Scope: diagnostic-only runtime variability check.
- Comparison: same algorithm, low-level helper replacement only.
- Official source files modified: NO.
- Temporary runner copies: `results/rcpp_vs_r_runtime_benchmark_rep50_260708/temp_methods`.
- Loading route: `Rcpp::sourceCpp(cacheDir=..., rebuild=FALSE)` in the temporary copies.
- Full rep=100 simulation: NO.
- Setting: K=4, n=300, d=60, rep=50, nstart=3, max_iter=50, max_path_steps=40.
- Cache warm-up elapsed: 6.310 sec.
- Equality result: PASS at tolerance `1e-8`.
- Median OFF elapsed: 59.860 sec.
- Median ON elapsed: 25.380 sec.
- Median OFF/ON ratio: 2.359.

## Timing summary

 mode repeats mean_elapsed_sec sd_elapsed_sec median_elapsed_sec
  OFF       3         59.73667      0.3421013              59.86
   ON       3         25.38667      0.0305505              25.38
 min_elapsed_sec max_elapsed_sec
           59.35           60.00
           25.36           25.42

## Interpretation

- The repeated benchmark separates sourceCpp cache warm-up from the timed rep=50 runs.
- Results are diagnostic and should not be used as a publication speed claim.
- The R-only path remains the reference implementation and can be restored with `USE_RCPP_HELPERS=0`.
