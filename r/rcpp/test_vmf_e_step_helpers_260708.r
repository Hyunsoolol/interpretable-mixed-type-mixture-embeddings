#!/usr/bin/env Rscript

# Diagnostic-only equality and micro-benchmark tests for vMF E-step Rcpp helpers.
# These helpers are not wired into the official fitting pipeline.

`%||%` <- function(x, y) if (is.null(x) || length(x) == 0 || is.na(x)) y else x

cmd_file <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_file <- sub("^--file=", "", cmd_file[1] %||% "r/rcpp/test_vmf_e_step_helpers_260708.r")
repo_root <- normalizePath(file.path(dirname(script_file), "..", ".."), winslash = "/", mustWork = FALSE)
if (!file.exists(file.path(repo_root, "r", "rcpp", "vmf_e_step_helpers.cpp"))) {
  repo_root <- normalizePath(getwd(), winslash = "/", mustWork = FALSE)
}
setwd(repo_root)

if (!requireNamespace("Rcpp", quietly = TRUE)) {
  stop("Rcpp is not installed; cannot run vMF E-step helper diagnostic.")
}

Rcpp::sourceCpp(file.path("r", "rcpp", "vmf_e_step_helpers.cpp"))
source(file.path("r", "methods", "rossi_barbaro_2022_reproduction.r"), encoding = "UTF-8")

out_dir <- file.path("results", "rcpp_vmf_e_step_smoke_260708")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

max_abs_diff <- function(a, b) {
  if (!all(dim(a) == dim(b))) return(Inf)
  max(abs(as.numeric(a) - as.numeric(b)))
}

make_theta <- function(K, d, kappa_scale = 1) {
  alpha <- runif(K)
  alpha <- alpha / sum(alpha)
  mu <- normalize_rows(matrix(rnorm(K * d), nrow = K, ncol = d))
  kappa <- sort(runif(K, min = 0.1, max = 30 * kappa_scale))
  list(alpha = alpha, mu = mu, kappa = kappa)
}

make_X <- function(n, d) {
  normalize_rows(matrix(rnorm(n * d), nrow = n, ncol = d))
}

compare_row_logsumexp <- function() {
  cases <- list(
    random = matrix(rnorm(40), nrow = 10),
    very_negative = matrix(rnorm(40, mean = -1000, sd = 10), nrow = 10),
    very_positive = matrix(rnorm(40, mean = 1000, sd = 10), nrow = 10),
    mixed_scale = cbind(rnorm(10, -500, 2), rnorm(10), rnorm(10, 500, 2), rnorm(10, 100, 1))
  )
  rows <- lapply(names(cases), function(nm) {
    M <- cases[[nm]]
    r_val <- row_logsumexp(M)
    cpp_val <- row_logsumexp_cpp(M)
    data.frame(
      test = paste0("row_logsumexp_", nm),
      max_abs_diff = max(abs(r_val - cpp_val)),
      tolerance = 1e-10,
      pass = max(abs(r_val - cpp_val)) < 1e-10
    )
  })
  do.call(rbind, rows)
}

compare_e_step_case <- function(seed, n, d, K, kappa_scale = 1) {
  set.seed(seed)
  X <- make_X(n, d)
  theta <- make_theta(K, d, kappa_scale = kappa_scale)
  kappa <- theta$kappa
  if (length(kappa) == 1) kappa <- rep(kappa, K)
  log_const <- log_vmf_const(kappa, d)

  r_val <- e_step_vmf(X, theta)
  cpp_val <- e_step_vmf_cpp(X, theta$alpha, theta$mu, kappa, log_const)
  tau_diff <- max_abs_diff(r_val$tau, cpp_val$tau)
  loglik_diff <- abs(r_val$loglik - cpp_val$loglik)
  tau_row_sum_diff <- max(abs(rowSums(cpp_val$tau) - 1))

  data.frame(
    test = sprintf("e_step_seed%d_n%d_d%d_K%d", seed, n, d, K),
    seed = seed,
    n = n,
    d = d,
    K = K,
    tau_max_abs_diff = tau_diff,
    loglik_abs_diff = loglik_diff,
    tau_row_sum_max_abs_diff = tau_row_sum_diff,
    tolerance = 1e-10,
    pass = tau_diff < 1e-10 && loglik_diff < 1e-10 && tau_row_sum_diff < 1e-12
  )
}

benchmark_e_step <- function(seed = 20260708, n = 300, d = 80, K = 4, repeats = 250) {
  set.seed(seed)
  X <- make_X(n, d)
  theta <- make_theta(K, d, kappa_scale = 1.5)
  kappa <- theta$kappa
  log_const <- log_vmf_const(kappa, d)

  r_time <- system.time({
    for (i in seq_len(repeats)) {
      invisible(e_step_vmf(X, theta))
    }
  })
  cpp_time <- system.time({
    for (i in seq_len(repeats)) {
      invisible(e_step_vmf_cpp(X, theta$alpha, theta$mu, kappa, log_const))
    }
  })

  data.frame(
    seed = seed,
    n = n,
    d = d,
    K = K,
    repeats = repeats,
    r_elapsed_sec = unname(r_time[["elapsed"]]),
    cpp_elapsed_sec = unname(cpp_time[["elapsed"]]),
    speed_ratio_r_over_cpp = unname(r_time[["elapsed"]]) / unname(cpp_time[["elapsed"]])
  )
}

bind_rows_fill <- function(dfs) {
  all_names <- unique(unlist(lapply(dfs, names)))
  dfs <- lapply(dfs, function(x) {
    missing <- setdiff(all_names, names(x))
    for (nm in missing) x[[nm]] <- NA
    x[, all_names, drop = FALSE]
  })
  do.call(rbind, dfs)
}

set.seed(20260708)
row_tests <- compare_row_logsumexp()
e_step_tests <- do.call(rbind, list(
  compare_e_step_case(20260708, n = 30, d = 8, K = 3, kappa_scale = 0.5),
  compare_e_step_case(20260709, n = 120, d = 36, K = 4, kappa_scale = 1),
  compare_e_step_case(20260710, n = 180, d = 80, K = 4, kappa_scale = 1.5),
  compare_e_step_case(20260711, n = 80, d = 120, K = 5, kappa_scale = 2)
))
comparison <- bind_rows_fill(list(
  cbind(type = "row_logsumexp", row_tests),
  cbind(type = "e_step_vmf", e_step_tests)
))

benchmark <- benchmark_e_step()

comparison_path <- file.path(out_dir, "vmf_e_step_helper_comparison.csv")
benchmark_path <- file.path(out_dir, "vmf_e_step_runtime_benchmark.csv")
write.csv(comparison, comparison_path, row.names = FALSE)
write.csv(benchmark, benchmark_path, row.names = FALSE)

pass_all <- all(comparison$pass)
notes_path <- file.path(out_dir, "vmf_e_step_helper_notes.md")
notes <- c(
  "# vMF E-step Rcpp helper diagnostic",
  "",
  sprintf("- Date: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
  "- Scope: diagnostic-only Rcpp prototype for `row_logsumexp` and E-step tau/loglik core.",
  "- Official algorithm connection: NO.",
  "- Official method/source files modified: NO.",
  "- vMF normalizing constants are still computed by the existing R `log_vmf_const` function and passed into C++.",
  sprintf("- Equality result: %s.", if (pass_all) "PASS" else "FAIL"),
  sprintf("- Runtime benchmark repeats: %d.", benchmark$repeats[[1]]),
  sprintf("- Runtime R-only elapsed: %.6f sec.", benchmark$r_elapsed_sec[[1]]),
  sprintf("- Runtime Rcpp elapsed: %.6f sec.", benchmark$cpp_elapsed_sec[[1]]),
  sprintf("- Runtime speed ratio R/Rcpp: %.3f.", benchmark$speed_ratio_r_over_cpp[[1]]),
  "",
  "## Max equality differences",
  "",
  paste(capture.output(print(comparison)), collapse = "\n"),
  "",
  "## Interpretation",
  "",
  "- Equality is the primary criterion; runtime is a preliminary micro-benchmark.",
  "- This prototype does not yet replace the official `e_step_vmf` function.",
  "- If adopted later, official wiring should be guarded by a fallback switch."
)
writeLines(notes, notes_path, useBytes = TRUE)

message("[vmf-e-step-rcpp] Wrote: ", comparison_path)
message("[vmf-e-step-rcpp] Wrote: ", benchmark_path)
message("[vmf-e-step-rcpp] Wrote: ", notes_path)
message("[vmf-e-step-rcpp] Equality: ", if (pass_all) "PASS" else "FAIL")
message(sprintf("[vmf-e-step-rcpp] Runtime R=%.6f, Rcpp=%.6f, ratio=%.3f",
                benchmark$r_elapsed_sec[[1]],
                benchmark$cpp_elapsed_sec[[1]],
                benchmark$speed_ratio_r_over_cpp[[1]]))

if (!pass_all) {
  stop("vMF E-step Rcpp helper diagnostic failed. Do not connect to official code.")
}
