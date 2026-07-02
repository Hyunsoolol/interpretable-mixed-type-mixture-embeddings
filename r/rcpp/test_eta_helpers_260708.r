#!/usr/bin/env Rscript

# Standalone equality tests for Eta-group Rcpp helper prototypes.
# This script does not source or modify the official fitting pipeline.

if (!requireNamespace("Rcpp", quietly = TRUE)) {
  stop("Rcpp is required for this prototype test.")
}

cpp_path <- file.path("r", "rcpp", "eta_helpers.cpp")
if (!file.exists(cpp_path)) {
  stop("Missing Rcpp source: ", cpp_path)
}
Rcpp::sourceCpp(cpp_path, rebuild = TRUE)

eta_centered_penalty_value_r <- function(eta, adaptive_weights = NULL) {
  mean_eta <- colMeans(eta)
  centered <- sweep(eta, 2, mean_eta, "-")
  norms <- sqrt(colSums(centered * centered))
  if (is.null(adaptive_weights)) {
    adaptive_weights <- rep(1, length(norms))
  }
  sum(adaptive_weights * norms)
}

prox_eta_centered_r <- function(eta, lambda_eta, adaptive_weights = NULL) {
  mean_eta <- colMeans(eta)
  centered <- sweep(eta, 2, mean_eta, "-")
  norms <- sqrt(colSums(centered * centered))
  if (is.null(adaptive_weights)) {
    adaptive_weights <- rep(1, length(norms))
  }
  threshold <- lambda_eta * adaptive_weights
  scale <- ifelse(norms > 0, pmax(1 - threshold / norms, 0), 0)
  sweep(sweep(centered, 2, scale, "*"), 2, mean_eta, "+")
}

normalize_rows_r <- function(X, eps = 1e-12) {
  nr <- sqrt(rowSums(X * X))
  nr[nr < eps] <- 1
  sweep(X, 1, nr, "/")
}

max_abs_diff <- function(a, b) {
  max(abs(as.numeric(a) - as.numeric(b)))
}

check_close <- function(label, r_value, cpp_value, tol = 1e-10) {
  diff <- max_abs_diff(r_value, cpp_value)
  pass <- is.finite(diff) && diff <= tol
  data.frame(
    test = label,
    max_abs_diff = diff,
    tolerance = tol,
    pass = pass,
    stringsAsFactors = FALSE
  )
}

set.seed(20260708)
tol <- 1e-10
results <- list()

for (case_id in seq_len(10)) {
  K <- sample(2:6, 1)
  d <- sample(5:25, 1)
  eta <- matrix(rnorm(K * d), nrow = K)
  colnames(eta) <- paste0("v", seq_len(d))
  rownames(eta) <- paste0("k", seq_len(K))

  centered <- sweep(eta, 2, colMeans(eta), "-")
  norms <- sqrt(colSums(centered * centered))
  lambda_grid <- unique(c(
    0,
    max(norms) * 0.01,
    stats::median(norms),
    max(norms) * 1.5
  ))

  weights <- runif(d, min = 0.25, max = 2.0)

  results[[length(results) + 1]] <- check_close(
    sprintf("penalty_unweighted_case_%02d", case_id),
    eta_centered_penalty_value_r(eta),
    eta_centered_penalty_value_cpp(eta),
    tol
  )
  results[[length(results) + 1]] <- check_close(
    sprintf("penalty_weighted_case_%02d", case_id),
    eta_centered_penalty_value_r(eta, weights),
    eta_centered_penalty_value_cpp(eta, weights),
    tol
  )

  for (lambda_eta in lambda_grid) {
    lambda_label <- format(lambda_eta, digits = 4, scientific = TRUE)
    results[[length(results) + 1]] <- check_close(
      sprintf("prox_unweighted_case_%02d_lambda_%s", case_id, lambda_label),
      prox_eta_centered_r(eta, lambda_eta),
      prox_eta_centered_cpp(eta, lambda_eta),
      tol
    )
    results[[length(results) + 1]] <- check_close(
      sprintf("prox_weighted_case_%02d_lambda_%s", case_id, lambda_label),
      prox_eta_centered_r(eta, lambda_eta, weights),
      prox_eta_centered_cpp(eta, lambda_eta, weights),
      tol
    )
  }
}

norm_cases <- list(
  basic = matrix(c(3, 4, 0, 0, 1, -1), nrow = 3, byrow = TRUE),
  zero_and_near_zero = matrix(c(
    0, 0, 0,
    1e-14, -1e-14, 0,
    1, 2, 2,
    -3, 4, 0
  ), nrow = 4, byrow = TRUE),
  mixed_scale = matrix(rnorm(20), nrow = 5)
)

for (nm in names(norm_cases)) {
  X <- norm_cases[[nm]]
  rownames(X) <- paste0("r", seq_len(nrow(X)))
  colnames(X) <- paste0("x", seq_len(ncol(X)))
  results[[length(results) + 1]] <- check_close(
    paste0("normalize_rows_", nm),
    normalize_rows_r(X),
    normalize_rows_cpp(X),
    tol
  )
}

summary <- do.call(rbind, results)
print(summary, row.names = FALSE)

if (!all(summary$pass)) {
  failed <- summary[!summary$pass, , drop = FALSE]
  print(failed, row.names = FALSE)
  stop("Rcpp helper equality tests failed.")
}

cat("\nAll Rcpp helper equality tests passed.\n")
cat("Number of checks:", nrow(summary), "\n")
