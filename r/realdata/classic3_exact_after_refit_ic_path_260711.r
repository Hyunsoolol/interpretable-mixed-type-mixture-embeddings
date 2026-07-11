#!/usr/bin/env Rscript

# Refit each unique centered-Eta support exactly and rerank the path by IC.
# Diagnostic-only: official fitting and selection runners remain unchanged.

options(stringsAsFactors = FALSE)
Sys.setenv(USE_RCPP_HELPERS = "0")

if (!requireNamespace("Matrix", quietly = TRUE)) stop("Matrix is required.")
suppressPackageStartupMessages(library(Matrix))

getenv_chr <- function(name, default) {
  value <- Sys.getenv(name, unset = "")
  if (nzchar(value)) value else default
}
getenv_int <- function(name, default) {
  value <- Sys.getenv(name, unset = "")
  if (nzchar(value)) as.integer(value) else default
}
getenv_num <- function(name, default) {
  value <- Sys.getenv(name, unset = "")
  if (nzchar(value)) as.numeric(value) else default
}

cfg <- list(
  dataset_label = getenv_chr("CLASSIC3_EXACT_DATASET_LABEL", "Classic3"),
  output_prefix = getenv_chr(
    "CLASSIC3_EXACT_OUTPUT_PREFIX", "classic3_exact_after_refit_ic"
  ),
  data_path = getenv_chr(
    "CLASSIC3_EXACT_DATA",
    file.path("data", "classic3", "processed", "splade_classic3_n3890_top2000_260711.rds")
  ),
  fitted_paths = getenv_chr(
    "CLASSIC3_EXACT_PATHS",
    file.path(
      "results", "classic3_splade_top2000_nstart30_path300_260711",
      "classic3_splade_top2000_fitted_paths.rds"
    )
  ),
  projected_candidates = getenv_chr(
    "CLASSIC3_PROJECTED_CANDIDATES",
    file.path(
      "results", "classic3_splade_top2000_after_refit_ic_260711",
      "classic3_splade_top2000_after_refit_ic_candidate_refits.csv"
    )
  ),
  projected_selection = getenv_chr(
    "CLASSIC3_PROJECTED_SELECTION",
    file.path(
      "results", "classic3_splade_top2000_after_refit_ic_260711",
      "classic3_splade_top2000_after_refit_ic_selection_summary.csv"
    )
  ),
  out_dir = getenv_chr(
    "CLASSIC3_EXACT_PATH_OUT_DIR",
    file.path("results", "classic3_exact_after_refit_ic_path_260711")
  ),
  max_iter = getenv_int("CLASSIC3_EXACT_PATH_MAX_ITER", 500L),
  optim_maxit = getenv_int("CLASSIC3_EXACT_PATH_OPTIM_MAXIT", 500L),
  optim_factr = getenv_num("CLASSIC3_EXACT_PATH_OPTIM_FACTR", 1e5),
  optim_pgtol = getenv_num("CLASSIC3_EXACT_PATH_OPTIM_PGTOL", 1e-8),
  rel_tol = getenv_num("CLASSIC3_EXACT_PATH_REL_TOL", 1e-10),
  abs_tol = getenv_num("CLASSIC3_EXACT_PATH_ABS_TOL", 1e-6),
  zero_eps = getenv_num("CLASSIC3_EXACT_PATH_ZERO_EPS", 1e-8),
  min_step = getenv_num("CLASSIC3_EXACT_PATH_MIN_STEP", 2^-14),
  verbose = getenv_int("CLASSIC3_EXACT_PATH_VERBOSE", 1L) != 0L
)

for (path in c(
  cfg$data_path, cfg$fitted_paths,
  cfg$projected_candidates, cfg$projected_selection
)) {
  if (!file.exists(path)) stop("Missing input: ", path)
}
dir.create(cfg$out_dir, recursive = TRUE, showWarnings = FALSE)

source_no_bom <- function(path, envir = .GlobalEnv) {
  txt <- readLines(path, warn = FALSE, encoding = "UTF-8")
  if (length(txt) > 0L) txt[1L] <- sub("^\ufeff", "", txt[1L])
  eval(parse(text = txt), envir = envir)
}
source_no_bom(file.path("r", "methods", "rossi_barbaro_2022_reproduction.r"))
source_no_bom(file.path("r", "realdata", "exact_centered_refit_helpers_260711.r"))

log_vmf_const_one <- exact_refit_log_vmf_const_one
log_vmf_const <- exact_refit_log_vmf_const

payload <- readRDS(cfg$data_path)
X <- payload$X
if (!inherits(X, "sparseMatrix")) X <- Matrix::Matrix(X, sparse = TRUE)
y <- payload$y
paths <- readRDS(cfg$fitted_paths)
projected_candidates <- utils::read.csv(cfg$projected_candidates, check.names = FALSE)
projected_selection <- utils::read.csv(cfg$projected_selection, check.names = FALSE)
n <- nrow(X)
d <- ncol(X)
first_eta_path <- if (!is.null(paths$ecgl)) {
  paths$ecgl
} else if (!is.null(paths$ecagl)) {
  paths$ecagl
} else {
  stop("No centered-Eta path found in fitted_paths.")
}
K <- nrow(first_eta_path$fits[[1L]]$mu)

active_eta_centered <- function(theta, zero_eps = 1e-8) {
  eta <- sweep(theta$mu, 1L, theta$kappa, "*")
  centered <- sweep(eta, 2L, colMeans(eta), "-")
  sqrt(colSums(centered * centered)) > zero_eps
}

normalized_mutual_information <- function(truth, cluster) {
  tab <- table(truth, cluster)
  p <- tab / sum(tab)
  pi <- rowSums(p)
  pj <- colSums(p)
  nz <- which(p > 0, arr.ind = TRUE)
  mi <- sum(p[nz] * log(p[nz] / (pi[nz[, 1L]] * pj[nz[, 2L]])))
  h_truth <- -sum(pi[pi > 0] * log(pi[pi > 0]))
  h_cluster <- -sum(pj[pj > 0] * log(pj[pj > 0]))
  if (h_truth + h_cluster <= 0) return(1)
  2 * mi / (h_truth + h_cluster)
}

support_key <- function(active) paste(which(active), collapse = ",")

unique_support_specs <- function(method, object) {
  fits <- object$fits
  active <- lapply(fits, active_eta_centered, zero_eps = cfg$zero_eps)
  keys <- vapply(active, support_key, character(1))
  split_indices <- split(seq_along(fits), keys)
  projected <- projected_candidates[
    projected_candidates$method == method &
      projected_candidates$refit_mode == "centered_projection",
    , drop = FALSE
  ]
  specs <- lapply(split_indices, function(indices) {
    available <- projected[projected$path_index %in% indices, , drop = FALSE]
    if (nrow(available) > 0L && any(is.finite(available$loglik))) {
      index <- as.integer(available$path_index[which.max(available$loglik)])
    } else {
      index <- indices[1L]
    }
    list(
      path_index = index,
      duplicate_count = length(indices),
      duplicate_path_indices = paste(indices, collapse = ";"),
      active = active[[index]],
      init = fits[[index]]
    )
  })
  specs[order(vapply(specs, function(x) x$path_index, integer(1)))]
}

fit_exact_support <- function(init, active) {
  eta0 <- exact_refit_project(sweep(init$mu, 1L, init$kappa, "*"), active)
  theta <- exact_refit_eta_to_theta(init$alpha, eta0, init$mu)
  current <- e_step_vmf(X, theta)
  initial_loglik <- current$loglik
  converged <- FALSE
  failed <- FALSE
  stop_reason <- "max_iter"
  n_halving <- 0L
  min_loglik_diff <- Inf
  min_Q_diff <- Inf
  max_inner_convergence <- 0L
  max_inner_gradient_abs <- 0
  total_inner_function_evaluations <- 0L
  total_start <- proc.time()[["elapsed"]]
  completed_iter <- 0L

  for (iter in seq_len(cfg$max_iter)) {
    Nk <- colSums(current$tau)
    if (any(!is.finite(Nk)) || any(Nk < 1e-8)) {
      failed <- TRUE
      stop_reason <- "empty_component"
      break
    }
    r <- as.matrix(t(current$tau) %*% X)
    eta_old <- sweep(theta$mu, 1L, theta$kappa, "*")
    Q_old <- exact_refit_q(eta_old, r, Nk)
    inner <- exact_refit_mstep(
      r, Nk, active, eta_old,
      maxit = cfg$optim_maxit,
      factr = cfg$optim_factr,
      pgtol = cfg$optim_pgtol
    )
    max_inner_convergence <- max(max_inner_convergence, inner$convergence)
    max_inner_gradient_abs <- max(max_inner_gradient_abs, inner$gradient_max_abs)
    total_inner_function_evaluations <- total_inner_function_evaluations +
      unname(inner$counts["function"])

    step <- 1
    accepted <- FALSE
    repeat {
      eta_try <- eta_old + step * (inner$eta - eta_old)
      alpha_try <- theta$alpha + step * (Nk / n - theta$alpha)
      candidate <- exact_refit_eta_to_theta(alpha_try, eta_try, theta$mu)
      next_e <- e_step_vmf(X, candidate)
      if (is.finite(next_e$loglik) && next_e$loglik >= current$loglik - 1e-8) {
        accepted <- TRUE
        break
      }
      if (step <= cfg$min_step) break
      step <- step / 2
      n_halving <- n_halving + 1L
    }
    if (!accepted) {
      failed <- TRUE
      stop_reason <- "line_search_failed"
      break
    }

    Q_new <- exact_refit_q(eta_try, r, Nk)
    loglik_diff <- next_e$loglik - current$loglik
    Q_diff <- Q_new - Q_old
    min_loglik_diff <- min(min_loglik_diff, loglik_diff)
    min_Q_diff <- min(min_Q_diff, Q_diff)
    theta <- candidate
    current <- next_e
    completed_iter <- iter

    if (abs(loglik_diff) <= cfg$abs_tol + cfg$rel_tol * abs(current$loglik - loglik_diff)) {
      converged <- TRUE
      stop_reason <- "loglik_tolerance"
      break
    }
  }

  elapsed_sec <- proc.time()[["elapsed"]] - total_start
  cluster <- max.col(current$tau, ties.method = "first")
  eta_final <- sweep(theta$mu, 1L, theta$kappa, "*")
  c(theta, list(
    tau = current$tau,
    loglik = current$loglik,
    initial_loglik = initial_loglik,
    loglik_gain = current$loglik - initial_loglik,
    cluster = cluster,
    ARI = adjusted_rand_index(y, cluster),
    NMI = normalized_mutual_information(y, cluster),
    active = active,
    selected_q = sum(active),
    converged = converged,
    failed = failed,
    stop_reason = stop_reason,
    iter = completed_iter,
    elapsed_sec = elapsed_sec,
    n_halving = n_halving,
    min_loglik_diff = if (is.finite(min_loglik_diff)) min_loglik_diff else NA_real_,
    min_Q_diff = if (is.finite(min_Q_diff)) min_Q_diff else NA_real_,
    constraint_error = exact_refit_constraint_error(eta_final, active),
    max_inner_convergence = max_inner_convergence,
    max_inner_gradient_abs = max_inner_gradient_abs,
    total_inner_function_evaluations = total_inner_function_evaluations
  ))
}

criteria <- c(
  "BIC", "RICc", "Rossi_EBIC_g0.5", "Rossi_EBIC_g1", "Support_EBIC_g1"
)
specs <- list()
if (!is.null(paths$ecgl)) {
  specs[["E-CGL"]] <- unique_support_specs("E-CGL", paths$ecgl)
}
if (!is.null(paths$ecagl)) {
  specs[["E-CAGL"]] <- unique_support_specs("E-CAGL", paths$ecagl)
}

rows <- list()
best_fits <- list()
row_id <- 0L
total_candidates <- sum(vapply(specs, length, integer(1)))
completed_candidates <- 0L

for (method in names(specs)) {
  method_specs <- specs[[method]]
  for (ii in seq_along(method_specs)) {
    spec <- method_specs[[ii]]
    completed_candidates <- completed_candidates + 1L
    if (cfg$verbose) {
      cat(sprintf(
        "%s unique support %d/%d; overall %d/%d; path=%d q=%d duplicates=%d\n",
        method, ii, length(method_specs), completed_candidates, total_candidates,
        spec$path_index, sum(spec$active), spec$duplicate_count
      ))
      flush.console()
    }
    fit <- fit_exact_support(spec$init, spec$active)
    m <- fit$selected_q
    df <- d + (K - 1L) * m + (K - 1L) * as.integer(m > 0L)
    minus2loglik <- -2 * fit$loglik
    row <- data.frame(
      method = method,
      path_index = spec$path_index,
      duplicate_count = spec$duplicate_count,
      duplicate_path_indices = spec$duplicate_path_indices,
      selected_q = m,
      loglik = fit$loglik,
      initial_loglik = fit$initial_loglik,
      loglik_gain = fit$loglik_gain,
      df = df,
      BIC = minus2loglik + log(n) * df,
      RICc = minus2loglik + 2 * (log(d) + log(log(d))) * df,
      Rossi_EBIC_g0.5 = minus2loglik + (log(n) + log(d)) * df,
      Rossi_EBIC_g1 = minus2loglik + (log(n) + 2 * log(d)) * df,
      Support_EBIC_g1 = minus2loglik + log(n) * df + 2 * lchoose(d, m),
      ARI = fit$ARI,
      NMI = fit$NMI,
      converged = fit$converged,
      failed = fit$failed,
      stop_reason = fit$stop_reason,
      iter = fit$iter,
      elapsed_sec = fit$elapsed_sec,
      n_halving = fit$n_halving,
      min_loglik_diff = fit$min_loglik_diff,
      min_Q_diff = fit$min_Q_diff,
      constraint_error = fit$constraint_error,
      max_inner_convergence = fit$max_inner_convergence,
      max_inner_gradient_abs = fit$max_inner_gradient_abs,
      total_inner_function_evaluations = fit$total_inner_function_evaluations,
      cluster_size = paste(
        as.integer(table(factor(fit$cluster, levels = seq_len(K)))), collapse = "/"
      ),
      stringsAsFactors = FALSE
    )
    projected_row <- projected_candidates[
      projected_candidates$method == method &
        projected_candidates$refit_mode == "centered_projection" &
        projected_candidates$path_index == spec$path_index,
      , drop = FALSE
    ]
    row$projected_loglik <- if (nrow(projected_row) == 1L) projected_row$loglik else NA_real_
    row$exact_minus_projected_loglik <- row$loglik - row$projected_loglik
    row_id <- row_id + 1L
    rows[[row_id]] <- row

    if (!fit$failed && is.finite(fit$loglik)) {
      for (criterion in criteria) {
        key <- paste(method, criterion, sep = "__")
        if (is.null(best_fits[[key]]) || row[[criterion]] < best_fits[[key]]$value) {
          best_fits[[key]] <- list(value = row[[criterion]], row = row, fit = fit)
        }
      }
    }
  }
}

candidates <- do.call(rbind, rows)
# Build selection explicitly to retain the criterion name.
selection_rows <- list()
selection_id <- 0L
for (method in names(specs)) {
  for (criterion in criteria) {
    key <- paste(method, criterion, sep = "__")
    if (is.null(best_fits[[key]])) next
    selection_id <- selection_id + 1L
    out <- best_fits[[key]]$row
    out$criterion <- criterion
    out$criterion_value <- out[[criterion]]
    selection_rows[[selection_id]] <- out
  }
}
selection_exact <- do.call(rbind, selection_rows)
row.names(selection_exact) <- NULL
selection_exact$criterion_margin <- vapply(seq_len(nrow(selection_exact)), function(i) {
  tab <- candidates[candidates$method == selection_exact$method[i], , drop = FALSE]
  values <- sort(tab[[selection_exact$criterion[i]]])
  if (length(values) >= 2L) values[2L] - values[1L] else NA_real_
}, numeric(1))

projected_main <- projected_selection[
  projected_selection$method %in% names(specs) &
    projected_selection$refit_mode == "centered_projection" &
    projected_selection$criterion %in% criteria,
  , drop = FALSE
]
comparison <- merge(
  projected_main[, c(
    "method", "criterion", "path_index", "selected_q", "loglik", "ARI", "NMI",
    "converged", "elapsed_sec", "criterion_value"
  )],
  selection_exact[, c(
    "method", "criterion", "path_index", "selected_q", "loglik", "ARI", "NMI",
    "converged", "elapsed_sec", "criterion_value", "criterion_margin"
  )],
  by = c("method", "criterion"), suffixes = c("_projected", "_exact"), all = TRUE
)
comparison$selected_q_change <- comparison$selected_q_exact - comparison$selected_q_projected
comparison$loglik_change <- comparison$loglik_exact - comparison$loglik_projected
comparison$ARI_change <- comparison$ARI_exact - comparison$ARI_projected
comparison$exact_same_support_loglik_gain <- mapply(function(method, path_index) {
  row <- candidates[
    candidates$method == method & candidates$path_index == path_index,
    , drop = FALSE
  ]
  if (nrow(row) == 1L) row$exact_minus_projected_loglik else NA_real_
}, comparison$method, comparison$path_index_exact)

candidate_path <- file.path(cfg$out_dir, paste0(cfg$output_prefix, "_candidates.csv"))
selection_path <- file.path(cfg$out_dir, paste0(cfg$output_prefix, "_selection.csv"))
comparison_path <- file.path(cfg$out_dir, paste0(cfg$output_prefix, "_vs_projected_selection.csv"))
fits_path <- file.path(cfg$out_dir, paste0(cfg$output_prefix, "_selected_fits.rds"))
notes_path <- file.path(cfg$out_dir, paste0(cfg$output_prefix, "_notes.md"))
utils::write.csv(candidates, candidate_path, row.names = FALSE)
utils::write.csv(selection_exact, selection_path, row.names = FALSE)
utils::write.csv(comparison, comparison_path, row.names = FALSE)
saveRDS(lapply(best_fits, function(x) x$fit), fits_path)

bic_comparison <- comparison[comparison$criterion == "BIC", , drop = FALSE]
fmt <- function(x, digits = 3L) formatC(x, digits = digits, format = "f")
table_lines <- paste0(
  "| ", bic_comparison$method, " | ", bic_comparison$path_index_projected,
  " / ", bic_comparison$selected_q_projected, " | ", bic_comparison$path_index_exact,
  " / ", bic_comparison$selected_q_exact, " | ", fmt(bic_comparison$criterion_margin),
  " | ", fmt(bic_comparison$ARI_exact), " | ",
  " ", fmt(bic_comparison$exact_same_support_loglik_gain), " |"
)
notes <- c(
  paste0("# ", cfg$dataset_label, " exact after-refit IC path diagnostic"),
  "",
  sprintf("- Data: n=%d, d=%d, K=%d.", n, d, K),
  paste0(
    "- Unique supports: ",
    paste(
      paste0(names(specs), "=", vapply(specs, length, integer(1))),
      collapse = ", "
    ),
    "."
  ),
  "- Every unique support was refitted by the exact fixed-support centered-Eta M-step.",
  "- The main df approximation is d + (K-1)m + (K-1)1(m>0).",
  "- Official fitting and selection code is unchanged.",
  "",
  "| method | projected path / q | exact path / q | exact BIC margin | exact ARI | same-support exact-projected loglik |",
  "|---|---:|---:|---:|---:|---:|",
  table_lines,
  "",
  sprintf("- Failed candidates: %d.", sum(candidates$failed)),
  sprintf("- Nonconverged candidates: %d.", sum(!candidates$converged)),
  sprintf("- Exact Q-decrease candidates: %d.", sum(candidates$min_Q_diff < -1e-6, na.rm = TRUE)),
  sprintf("- Exact likelihood-decrease candidates: %d.", sum(candidates$min_loglik_diff < -1e-8, na.rm = TRUE)),
  sprintf("- Total exact-refit elapsed time: %.3f sec.", sum(candidates$elapsed_sec)),
  ""
)
writeLines(notes, notes_path, useBytes = TRUE)

print(comparison, row.names = FALSE)
cat("Saved:", normalizePath(cfg$out_dir, winslash = "/"), "\n")
