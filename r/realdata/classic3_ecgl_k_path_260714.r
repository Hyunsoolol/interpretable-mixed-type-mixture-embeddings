#!/usr/bin/env Rscript

# E-CGL path at a fixed candidate K for the Classic3 K diagnostic.
# The dense free-kappa initialization comes from the existing train-only
# K=2,...,10 multistart fit. Supplied labels are used only for ARI/NMI.

options(stringsAsFactors = FALSE)

getenv_chr <- function(name, default) {
  value <- Sys.getenv(name, unset = "")
  if (nzchar(value)) value else default
}
getenv_int <- function(name, default) {
  value <- Sys.getenv(name, unset = "")
  if (nzchar(value)) as.integer(value) else default
}

K_target <- getenv_int("CLASSIC3_ECGL_K", 3L)
data_path <- getenv_chr(
  "CLASSIC3_ECGL_DATA",
  file.path(
    "data", "classic3", "processed",
    "classic3_splade_holdout_train_top2000_260711.rds"
  )
)
dense_fits_path <- getenv_chr(
  "CLASSIC3_ECGL_DENSE_FITS",
  file.path(
    "results", "classic3_splade_holdout_k_selection_k2_10_260711",
    "classic3_dense_k_fits.rds"
  )
)
out_dir <- getenv_chr(
  "CLASSIC3_ECGL_OUT_DIR",
  file.path(
    "results", paste0("classic3_ecgl_k", K_target, "_path300_260714")
  )
)
output_prefix <- getenv_chr(
  "CLASSIC3_ECGL_OUTPUT_PREFIX", paste0("classic3_ecgl_k", K_target)
)
max_path <- getenv_int("CLASSIC3_ECGL_MAX_PATH", 300L)
max_iter <- getenv_int("CLASSIC3_ECGL_MAX_ITER", 100L)

for (path in c(data_path, dense_fits_path)) {
  if (!file.exists(path)) stop("Missing input: ", path)
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

Sys.setenv(
  CLASSIC3_COMPARE_DATASET_LABEL = paste0("Classic3 K=", K_target),
  CLASSIC3_COMPARE_DATA = data_path,
  CLASSIC3_COMPARE_OUT_DIR = out_dir,
  CLASSIC3_COMPARE_STRICT_DIMENSIONS = "0",
  CLASSIC3_COMPARE_ALLOW_K_MISMATCH = "1",
  CLASSIC3_COMPARE_K = as.character(K_target),
  CLASSIC3_COMPARE_NSTART = "1",
  CLASSIC3_COMPARE_MAX_ITER = as.character(max_iter),
  CLASSIC3_COMPARE_MAX_PATH = as.character(max_path),
  CLASSIC3_COMPARE_VERBOSE = "0",
  CLASSIC3_COMPARE_SAVE_PATHS = "0",
  CLASSIC3_COMPARE_USE_RCPP = "0",
  USE_RCPP_HELPERS = "0"
)

runner_path <- file.path("r", "realdata", "classic3_vmf_compare_pilot_260711.r")
runner_lines <- readLines(runner_path, warn = FALSE, encoding = "UTF-8")
runner_lines[1L] <- sub("^\ufeff", "", runner_lines[1L])
runner_boundary <- grep("^timings <- list\\(\\)", runner_lines)[1L]
if (is.na(runner_boundary)) stop("Could not locate runner boundary.")
eval(parse(text = runner_lines[seq_len(runner_boundary - 1L)]), envir = .GlobalEnv)

dense_fits <- readRDS(dense_fits_path)
dense_key <- paste0("K", K_target, "__free")
dense_free <- dense_fits[[dense_key]]
if (is.null(dense_free)) stop("Missing dense free-kappa fit: ", dense_key)
if (nrow(dense_free$mu) != K_target || ncol(dense_free$mu) != d) {
  stop("Dense fit dimension mismatch.")
}

eta_dense <- fit_eta_centered_em(
  X, K_target, lambda_eta = 0, init = dense_free,
  max_iter = max_iter, tol = cfg$tol,
  adaptive_weights = rep(1, d)
)

started <- proc.time()[["elapsed"]]
ecgl <- fit_eta_path(eta_dense, rep(1, d), "E-CGL")
elapsed <- proc.time()[["elapsed"]] - started

candidate_rows <- lapply(seq_along(ecgl$fits), function(index) {
  fit <- ecgl$fits[[index]]
  active <- active_eta_centered(fit, cfg$zero_eps)
  m <- sum(active)
  cluster <- max.col(fit$tau, ties.method = "first")
  df <- d + (K_target - 1L) * m +
    (K_target - 1L) * as.integer(m > 0L)
  minus2 <- -2 * fit$loglik
  data.frame(
    method = "E-CGL",
    path_index = index,
    refit_mode = "centered_projection",
    selected_q = m,
    loglik = fit$loglik,
    df = df,
    ARI = adjusted_rand_index(y, cluster),
    NMI = normalized_mutual_information(y, cluster),
    converged = isTRUE(fit$converged),
    iter = fit$iter,
    n_halving = NA_integer_,
    min_loglik_diff = NA_real_,
    elapsed_sec = NA_real_,
    BIC = minus2 + log(n) * df,
    RICc = minus2 + 2 * (log(d) + log(log(d))) * df,
    Rossi_EBIC_g0.5 = minus2 + (log(n) + log(d)) * df,
    Rossi_EBIC_g1 = minus2 + (log(n) + 2 * log(d)) * df,
    Support_EBIC_g1 = minus2 + log(n) * df + 2 * lchoose(d, m),
    stringsAsFactors = FALSE
  )
})
candidates <- do.call(rbind, candidate_rows)

criteria <- c(
  "BIC", "RICc", "Rossi_EBIC_g0.5", "Rossi_EBIC_g1",
  "Support_EBIC_g1"
)
selection <- do.call(rbind, lapply(criteria, function(criterion) {
  index <- which.min(candidates[[criterion]])
  out <- candidates[index, , drop = FALSE]
  out$criterion <- criterion
  out$criterion_value <- out[[criterion]]
  out
}))
row.names(selection) <- NULL

paths_path <- file.path(out_dir, paste0(output_prefix, "_fitted_paths.rds"))
candidate_path <- file.path(
  out_dir, paste0(output_prefix, "_projected_candidates.csv")
)
selection_path <- file.path(
  out_dir, paste0(output_prefix, "_projected_selection.csv")
)
notes_path <- file.path(out_dir, paste0(output_prefix, "_path_notes.md"))

saveRDS(list(ecgl = ecgl), paths_path)
utils::write.csv(candidates, candidate_path, row.names = FALSE)
utils::write.csv(selection, selection_path, row.names = FALSE)

fmt <- function(x, digits = 3L) formatC(x, digits = digits, format = "f")
bic <- selection[selection$criterion == "BIC", , drop = FALSE]
notes <- c(
  paste0("# Classic3 E-CGL path at K=", K_target),
  "",
  sprintf("- Data: n=%d, d=%d; fitted K=%d.", n, d, K_target),
  "- Initialization: existing train-only dense free-kappa multistart fit.",
  sprintf("- Path candidates: %d; max path=%d.", nrow(candidates), max_path),
  sprintf("- Penalized-path BIC q=%d; ARI=%s.", bic$selected_q, fmt(bic$ARI)),
  "- Supplied labels are excluded from fitting and path selection.",
  sprintf("- E-CGL path elapsed seconds: %.3f.", elapsed),
  ""
)
writeLines(notes, notes_path, useBytes = TRUE)

print(selection, row.names = FALSE)
cat("Saved:", normalizePath(out_dir, winslash = "/"), "\n")
