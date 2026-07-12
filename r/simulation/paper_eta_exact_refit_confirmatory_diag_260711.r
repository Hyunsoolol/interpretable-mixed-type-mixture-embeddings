# ==============================================================================
# Study B exact centered-Eta refit confirmatory diagnostic
# ------------------------------------------------------------------------------
# Diagnostic only. The official runner and existing results are not modified.
# The script compares the current projected refit with an exact fixed-support
# centered-Eta refit, and contrasts BIC-before-refit with BIC-after-exact-refit.
# ==============================================================================

options(stringsAsFactors = FALSE)

source_no_bom <- function(path, envir = .GlobalEnv) {
  txt <- readLines(path, warn = FALSE, encoding = "UTF-8")
  if (length(txt) > 0L) txt[1L] <- sub("^\ufeff", "", txt[1L])
  eval(parse(text = txt), envir = envir)
}

source_oracle_helpers_without_running <- function() {
  helper_file <- file.path("r", "simulation", "paper_eta_oracle_bayes_pilot_260714.r")
  lines <- readLines(helper_file, encoding = "UTF-8", warn = FALSE)
  if (length(lines) > 0L) lines[1L] <- sub("^\ufeff", "", lines[1L])
  stop_idx <- grep("^all_raw <- list\\(\\)", lines)[1L]
  if (is.na(stop_idx) || stop_idx <= 1L) {
    stop("Could not find execution boundary in oracle pilot runner.")
  }
  eval(parse(text = lines[seq_len(stop_idx - 1L)]), envir = .GlobalEnv)
}

Sys.setenv(USE_RCPP_HELPERS = Sys.getenv("USE_RCPP_HELPERS", "1"))
source_oracle_helpers_without_running()
source_no_bom(file.path("r", "realdata", "exact_centered_refit_helpers_260711.r"))

cfg$run_label <- Sys.getenv(
  "EXACT_DIAG_RUN_LABEL",
  "paper_eta_exact_refit_confirmatory_diag_260711"
)
cfg$out_dir <- Sys.getenv(
  "EXACT_DIAG_OUT_DIR",
  file.path("results", "paper_eta_exact_refit_confirmatory_diag_260711")
)
cfg$n <- as.integer(Sys.getenv("EXACT_DIAG_N", "300"))
cfg$d <- as.integer(Sys.getenv("EXACT_DIAG_D", "200"))
cfg$K <- 4L
cfg$n_rep <- as.integer(Sys.getenv("EXACT_DIAG_N_REP", "3"))
cfg$common_q <- 4L
cfg$decision_per_component <- 4L
cfg$nstart <- as.integer(Sys.getenv("EXACT_DIAG_NSTART", "10"))
cfg$max_iter <- as.integer(Sys.getenv("EXACT_DIAG_MAX_ITER", "100"))
cfg$eta_steps <- as.integer(Sys.getenv("EXACT_DIAG_ETA_STEPS", "40"))
cfg$target_oracle_error <- as.numeric(Sys.getenv("EXACT_DIAG_TARGET_EB", "0.10"))
cfg$base_seed <- as.integer(Sys.getenv("EXACT_DIAG_BASE_SEED", "20260711"))
cfg$calibration_iter <- as.integer(Sys.getenv("EXACT_DIAG_CALIBRATION_ITER", "18"))
cfg$calibration_mc_n <- as.integer(Sys.getenv("EXACT_DIAG_CALIBRATION_MC_N", "10000"))
cfg$validation_mc_n <- as.integer(Sys.getenv("EXACT_DIAG_VALIDATION_MC_N", "50000"))
cfg$exact_max_iter <- as.integer(Sys.getenv("EXACT_DIAG_REFIT_MAX_ITER", "80"))
cfg$optim_maxit <- as.integer(Sys.getenv("EXACT_DIAG_OPTIM_MAXIT", "80"))
cfg$optim_factr <- as.numeric(Sys.getenv("EXACT_DIAG_OPTIM_FACTR", "1e7"))
cfg$optim_pgtol <- as.numeric(Sys.getenv("EXACT_DIAG_OPTIM_PGTOL", "1e-6"))
cfg$abs_tol <- as.numeric(Sys.getenv("EXACT_DIAG_ABS_TOL", "1e-7"))
cfg$rel_tol <- as.numeric(Sys.getenv("EXACT_DIAG_REL_TOL", "1e-8"))
cfg$min_step <- as.numeric(Sys.getenv("EXACT_DIAG_MIN_STEP", "0.0009765625"))
cfg$kappa <- c(30, 40, 50, 60)
target_tag <- sprintf("%03d", round(1000 * cfg$target_oracle_error))
cfg$scenario_id <- paste0(
  "OBE", target_tag, "_heterogeneous_kappa_exact_refit_diag"
)
dir.create(cfg$out_dir, recursive = TRUE, showWarnings = FALSE)

fmt <- function(x, digits = 4L) {
  ifelse(is.na(x), "NA", formatC(as.numeric(x), format = "f", digits = digits))
}

oracle_error_crn <- function(A, kappa, n_mc, seed) {
  params <- make_oracle_eta_params_for_A(cfg, kappa, A)
  oracle_error_estimate(params, n_mc, seed)
}

calibrate_oracle_A_crn <- function(kappa, seed) {
  upper <- min(kappa) * 0.995
  target <- cfg$target_oracle_error
  err_at <- function(A) oracle_error_crn(A, kappa, cfg$calibration_mc_n, seed)
  err_low <- err_at(0)
  err_high <- err_at(upper)
  lo <- 0
  hi <- upper
  if (target <= err_low) {
    best_A <- lo
  } else if (target >= err_high) {
    best_A <- hi
  } else {
    for (iter in seq_len(cfg$calibration_iter)) {
      mid <- (lo + hi) / 2
      if (err_at(mid) < target) lo <- mid else hi <- mid
    }
    best_A <- (lo + hi) / 2
  }
  validation_seed <- seed + 100000L
  achieved <- oracle_error_crn(best_A, kappa, cfg$validation_mc_n, validation_seed)
  se <- sqrt(achieved * (1 - achieved) / cfg$validation_mc_n)
  data.frame(
    scenario = cfg$scenario_id,
    target_oracle_error = target,
    common_norm = best_A,
    endpoint_error_low = err_low,
    endpoint_error_high = err_high,
    achieved_oracle_error = achieved,
    achieved_mcse = se,
    achieved_ci_low = pmax(0, achieved - 1.96 * se),
    achieved_ci_high = pmin(1, achieved + 1.96 * se),
    calibration_mc_n = cfg$calibration_mc_n,
    validation_mc_n = cfg$validation_mc_n,
    common_random_numbers = TRUE
  )
}

main_df <- function(active, K = cfg$K, d = cfg$d) {
  m <- sum(active)
  d + (K - 1L) * m + (K - 1L) * as.integer(m > 0L)
}

main_ic <- function(loglik, active, n = cfg$n) {
  df <- main_df(active)
  data.frame(df = df, BIC = -2 * loglik + log(n) * df)
}

support_key <- function(active) {
  if (!any(active)) return("<empty>")
  paste(which(active), collapse = ";")
}

fit_eta_candidate_path <- function(X, dense, method) {
  adaptive <- identical(method, "E-ACGL")
  weights <- rep(1, ncol(X))
  if (adaptive) {
    eta0 <- eta_matrix(dense)
    weights <- group_weights_from_norm(
      sqrt(colSums(center_eta(eta0)^2)), cfg$adaptive_gamma, cfg$adaptive_eps
    )
  }

  fit <- fit_eta_centered_em(
    X, cfg$K, 0, init = dense, max_iter = cfg$max_iter,
    adaptive_weights = weights
  )
  fits <- list(fit)
  actives <- list(active_eta_centered(fit))
  lambdas <- 0

  if (cfg$eta_steps > 1L) {
    for (step in 2:cfg$eta_steps) {
      e <- e_step_vmf(X, fit)
      mstep <- unpenalized_eta_mstep(X, e$tau)
      centered <- center_eta(mstep$eta)
      lambda <- tail(lambdas, 1L)
      thresholds <- sqrt(colSums(centered^2)) / pmax(weights, 1e-12)
      next_values <- thresholds[thresholds > lambda + 1e-10]
      if (!length(next_values)) break
      lambda_next <- min(next_values)
      if (lambda > 0) lambda_next <- max(lambda_next, lambda * (1 + cfg$min_rel_lambda))
      if (!is.finite(lambda_next) || lambda_next <= lambda) break
      fit_next <- tryCatch(
        fit_eta_centered_em(
          X, cfg$K, lambda_next, init = fit, max_iter = cfg$max_iter,
          adaptive_weights = weights
        ),
        error = function(e) NULL
      )
      if (is.null(fit_next) || isTRUE(fit_next$failed)) break
      fit <- fit_next
      fits[[length(fits) + 1L]] <- fit
      actives[[length(actives) + 1L]] <- active_eta_centered(fit)
      lambdas <- c(lambdas, lambda_next)
      if (sum(tail(actives, 1L)[[1L]]) <= 1L) break
    }
  }

  path <- do.call(rbind, lapply(seq_along(fits), function(i) {
    current_ic <- eta_centered_ic(fits[[i]], nrow(X), ncol(X), fits[[i]]$loglik)
    revised_ic <- main_ic(fits[[i]]$loglik, actives[[i]], nrow(X))
    data.frame(
      path_index = i,
      lambda_eta = lambdas[i],
      support_key = support_key(actives[[i]]),
      selected_q = sum(actives[[i]]),
      loglik_before = fits[[i]]$loglik,
      df_current = current_ic$df,
      BIC_before_current = current_ic$BIC,
      df_main = revised_ic$df,
      BIC_before_main = revised_ic$BIC
    )
  }))
  list(path = path, fits = fits, actives = actives, weights = weights)
}

fit_exact_centered_support <- function(X, active, init) {
  n <- nrow(X)
  eta0 <- exact_refit_project(eta_matrix(init), active)
  theta <- exact_refit_eta_to_theta(init$alpha, eta0, init$mu)
  current <- e_step_vmf(X, theta)
  initial_loglik <- current$loglik
  converged <- FALSE
  failed <- FALSE
  stop_reason <- "max_iter"
  n_halving <- 0L
  min_loglik_diff <- Inf
  max_inner_convergence <- 0L
  max_inner_gradient_abs <- 0
  completed_iter <- 0L
  start <- proc.time()[["elapsed"]]

  for (iter in seq_len(cfg$exact_max_iter)) {
    Nk <- colSums(current$tau)
    if (any(!is.finite(Nk)) || any(Nk < 1e-8)) {
      failed <- TRUE
      stop_reason <- "empty_component"
      break
    }
    r <- as.matrix(t(current$tau) %*% X)
    eta_old <- eta_matrix(theta)
    old_loglik <- current$loglik
    inner <- exact_refit_mstep(
      r, Nk, active, eta_old, maxit = cfg$optim_maxit,
      factr = cfg$optim_factr, pgtol = cfg$optim_pgtol
    )
    max_inner_convergence <- max(max_inner_convergence, inner$convergence)
    max_inner_gradient_abs <- max(max_inner_gradient_abs, inner$gradient_max_abs)

    step <- 1
    accepted <- FALSE
    repeat {
      eta_try <- eta_old + step * (inner$eta - eta_old)
      alpha_try <- theta$alpha + step * (Nk / n - theta$alpha)
      candidate <- exact_refit_eta_to_theta(alpha_try, eta_try, theta$mu)
      next_e <- e_step_vmf(X, candidate)
      if (is.finite(next_e$loglik) && next_e$loglik >= old_loglik - 1e-8) {
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

    diff <- next_e$loglik - old_loglik
    min_loglik_diff <- min(min_loglik_diff, diff)
    theta <- candidate
    current <- next_e
    completed_iter <- iter
    if (abs(diff) <= cfg$abs_tol + cfg$rel_tol * abs(old_loglik)) {
      converged <- TRUE
      stop_reason <- "loglik_tolerance"
      break
    }
  }

  c(theta, list(
    failed = failed,
    failed_reason = stop_reason,
    converged = converged,
    iter = completed_iter,
    loglik = current$loglik,
    pen_loglik = NA_real_,
    tau = current$tau,
    initial_loglik = initial_loglik,
    loglik_gain = current$loglik - initial_loglik,
    elapsed_sec = proc.time()[["elapsed"]] - start,
    n_halving = n_halving,
    min_loglik_diff = if (is.finite(min_loglik_diff)) min_loglik_diff else NA_real_,
    max_inner_convergence = max_inner_convergence,
    max_inner_gradient_abs = max_inner_gradient_abs,
    constraint_error = exact_refit_constraint_error(eta_matrix(theta), active)
  ))
}

evaluate_refit <- function(fit, X, active, z, params, method, rule, lambda_eta) {
  ic <- if (is.finite(fit$loglik)) main_ic(fit$loglik, active, length(z)) else {
    data.frame(df = main_df(active), BIC = NA_real_)
  }
  if (is.finite(fit$loglik)) {
    row <- method_row(method, fit, X, z, params, active, lambda_eta = lambda_eta, ic = ic)
    row <- append_type_metrics(row, active, params)
  } else {
    sm <- support_metrics(active, colSums(params$support) > 0)
    row <- cbind(data.frame(
      method = method, ARI = NA_real_, loglik = NA_real_, converged = FALSE,
      iter = 0L, MSE_mu = NA_real_, MSE_kappa = NA_real_,
      MSE_centered_eta = NA_real_, kappa_hat_mean = NA_real_
    ), sm, ic, selection_type_metrics(active, params))
  }
  if (sum(active) == 0L) row$F1 <- 0
  eta <- if (is.null(fit$mu) || is.null(fit$kappa)) {
    matrix(NA_real_, cfg$K, cfg$d)
  } else {
    eta_matrix(fit)
  }
  baseline <- if (all(is.finite(eta))) colMeans(eta) else rep(NA_real_, cfg$d)
  row$rule <- rule
  row$common_q_selected <- sum(active[params$common_idx])
  row$decision_q_selected <- sum(active[params$decision_idx])
  row$noise_q_selected <- sum(active[params$noise_idx])
  row$common_baseline_l2 <- sqrt(sum(baseline[params$common_idx]^2))
  row$inactive_baseline_l2 <- if (any(!active)) sqrt(sum(baseline[!active]^2)) else 0
  row$constraint_error <- if (all(is.finite(eta))) {
    exact_refit_constraint_error(eta, active)
  } else {
    NA_real_
  }
  row$refit_elapsed_sec <- ifelse(is.null(fit$elapsed_sec), NA_real_, fit$elapsed_sec)
  row$stop_reason <- ifelse(is.null(fit$failed_reason), NA_character_, fit$failed_reason)
  row
}

calibration <- calibrate_oracle_A_crn(cfg$kappa, cfg$base_seed + 5000L)
params <- make_oracle_eta_params_for_A(cfg, cfg$kappa, calibration$common_norm[1L])

candidate_rows <- list()
best_rows <- list()
run_start <- proc.time()[["elapsed"]]

for (rep_id in seq_len(cfg$n_rep)) {
  set.seed(cfg$base_seed + 10000L + rep_id)
  dat <- simulate_from_params(cfg$n, params)
  set.seed(cfg$base_seed + 20000L + rep_id)
  dense <- fit_svMF_multistart(
    dat$X, cfg$K, beta = 0, nstart = cfg$nstart, max_iter = cfg$max_iter
  )
  cat(sprintf("[%s] rep %d/%d, shared dense initialization complete\n",
              cfg$scenario_id, rep_id, cfg$n_rep))

  for (method in c("E-CGL", "E-ACGL")) {
    path <- fit_eta_candidate_path(dat$X, dense, method)
    path_tab <- path$path
    current_idx <- which.min(path_tab$BIC_before_current)

    unique_keys <- unique(path_tab$support_key)
    exact_by_key <- list()
    projected_by_key <- list()
    source_index <- integer(length(unique_keys))

    for (u in seq_along(unique_keys)) {
      indices <- which(path_tab$support_key == unique_keys[u])
      index <- indices[which.max(path_tab$loglik_before[indices])]
      source_index[u] <- index
      active <- path$actives[[index]]
      init <- path$fits[[index]]
      exact <- fit_exact_centered_support(dat$X, active, init)
      exact_by_key[[unique_keys[u]]] <- exact
      projected_by_key[[unique_keys[u]]] <- if (any(active)) {
        fit_support_refit(dat$X, cfg$K, active, init, max_iter = cfg$max_iter)
      } else {
        zero <- init
        zero$failed <- TRUE
        zero$failed_reason <- "zero_active_support"
        zero$converged <- FALSE
        zero$iter <- 0L
        zero$loglik <- NA_real_
        zero$tau <- init$tau
        zero
      }

      exact_ic <- main_ic(exact$loglik, active, nrow(dat$X))
      candidate_rows[[length(candidate_rows) + 1L]] <- data.frame(
        scenario = cfg$scenario_id,
        rep = rep_id,
        method = method,
        support_key = unique_keys[u],
        source_path_index = index,
        lambda_eta = path_tab$lambda_eta[index],
        selected_q = sum(active),
        common_q = sum(active[params$common_idx]),
        decision_q = sum(active[params$decision_idx]),
        noise_q = sum(active[params$noise_idx]),
        loglik_before = path_tab$loglik_before[index],
        BIC_before_current = path_tab$BIC_before_current[index],
        BIC_before_main = path_tab$BIC_before_main[index],
        loglik_after_exact = exact$loglik,
        BIC_after_exact = exact_ic$BIC,
        exact_converged = exact$converged,
        exact_failed = exact$failed,
        exact_iter = exact$iter,
        exact_elapsed_sec = exact$elapsed_sec,
        exact_constraint_error = exact$constraint_error
      )
    }

    cand <- do.call(rbind, candidate_rows)
    cand <- cand[cand$rep == rep_id & cand$method == method, , drop = FALSE]
    exact_best_key <- cand$support_key[which.min(cand$BIC_after_exact)]
    current_key <- path_tab$support_key[current_idx]
    current_active <- path$actives[[current_idx]]
    exact_best_source <- source_index[match(exact_best_key, unique_keys)]
    exact_best_active <- path$actives[[exact_best_source]]

    projected_row <- evaluate_refit(
      projected_by_key[[current_key]], dat$X, current_active, dat$z, params, method,
      "current_BIC_before_projected_refit", path_tab$lambda_eta[current_idx]
    )
    exact_current_row <- evaluate_refit(
      exact_by_key[[current_key]], dat$X, current_active, dat$z, params, method,
      "current_BIC_before_exact_refit", path_tab$lambda_eta[current_idx]
    )
    exact_after_row <- evaluate_refit(
      exact_by_key[[exact_best_key]], dat$X, exact_best_active, dat$z, params, method,
      "BIC_after_exact_refit", path_tab$lambda_eta[exact_best_source]
    )
    rows <- rbind(projected_row, exact_current_row, exact_after_row)
    rows$scenario <- cfg$scenario_id
    rows$rep <- rep_id
    rows$n <- cfg$n
    rows$d <- cfg$d
    rows$target_oracle_error <- cfg$target_oracle_error
    rows$achieved_oracle_error <- calibration$achieved_oracle_error[1L]
    rows$current_path_index <- current_idx
    rows$exact_after_path_index <- exact_best_source
    rows$support_changed <- current_key != exact_best_key
    best_rows[[length(best_rows) + 1L]] <- rows
  }
}

candidates <- do.call(rbind, candidate_rows)
best <- do.call(rbind, best_rows)

metrics <- c(
  "selected_q", "common_q_selected", "decision_q_selected", "noise_q_selected",
  "TPR", "FPR", "Precision", "F1", "ARI", "loglik", "MSE_mu",
  "MSE_kappa", "MSE_centered_eta", "common_baseline_l2",
  "inactive_baseline_l2", "constraint_error", "refit_elapsed_sec"
)
groups <- unique(best[, c("method", "rule")])
summary_rows <- lapply(seq_len(nrow(groups)), function(i) {
  sub <- best[best$method == groups$method[i] & best$rule == groups$rule[i], , drop = FALSE]
  out <- data.frame(
    scenario = cfg$scenario_id,
    method = groups$method[i],
    rule = groups$rule[i],
    reps = length(unique(sub$rep)),
    valid_reps = sum(is.finite(sub$ARI)),
    zero_support_reps = sum(sub$selected_q == 0, na.rm = TRUE),
    support_changed_reps = sum(sub$support_changed, na.rm = TRUE)
  )
  for (metric in metrics) {
    value <- sub[[metric]]
    n_valid <- sum(is.finite(value))
    out[[paste0(metric, "_mean")]] <- if (n_valid) mean(value[is.finite(value)]) else NA_real_
    out[[paste0(metric, "_sd")]] <- if (n_valid > 1L) stats::sd(value[is.finite(value)]) else NA_real_
    out[[paste0(metric, "_mcse")]] <- if (n_valid > 1L) {
      stats::sd(value[is.finite(value)]) / sqrt(n_valid)
    } else {
      NA_real_
    }
  }
  out
})
summary <- do.call(rbind, summary_rows)

candidate_path <- file.path(cfg$out_dir, paste0(cfg$run_label, "_candidate_supports.csv"))
best_path <- file.path(cfg$out_dir, paste0(cfg$run_label, "_best_by_rep.csv"))
summary_path <- file.path(cfg$out_dir, paste0(cfg$run_label, "_summary.csv"))
calibration_path <- file.path(cfg$out_dir, paste0(cfg$run_label, "_calibration.csv"))
notes_path <- file.path(cfg$out_dir, paste0(cfg$run_label, "_notes.md"))
write.csv(candidates, candidate_path, row.names = FALSE)
write.csv(best, best_path, row.names = FALSE)
write.csv(summary, summary_path, row.names = FALSE)
write.csv(calibration, calibration_path, row.names = FALSE)

notes <- c(
  "# Study B exact centered-Eta refit confirmatory diagnostic",
  "",
  "This is a separate diagnostic result. It does not replace the existing Study B tables.",
  "",
  "## Setting",
  "",
  sprintf("- K=%d, n=%d, d=%d, reps=%d.", cfg$K, cfg$n, cfg$d, cfg$n_rep),
  sprintf("- Target oracle Bayes error=%.3f; achieved=%.4f (95%% MC interval %.4f to %.4f).",
          cfg$target_oracle_error, calibration$achieved_oracle_error,
          calibration$achieved_ci_low, calibration$achieved_ci_high),
  sprintf("- kappa=(%s), path steps=%d, nstart=%d.",
          paste(cfg$kappa, collapse = ","), cfg$eta_steps, cfg$nstart),
  "- E-CGL and E-ACGL use the same simulated data and the same dense initialization within each replicate.",
  "- The exact refit preserves inactive common natural-parameter baselines.",
  "- Main diagnostic df: d + (K-1)m + (K-1) I(m>0).",
  "",
  "## Summary",
  "",
  "| method | selection/refit rule | selected q | common q | decision q | noise q | F1 | ARI | MSE eta | MSE kappa | loglik | common baseline norm |",
  "|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
)
for (i in seq_len(nrow(summary))) {
  notes <- c(notes, sprintf(
    "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |",
    summary$method[i], summary$rule[i],
    fmt(summary$selected_q_mean[i], 2), fmt(summary$common_q_selected_mean[i], 2),
    fmt(summary$decision_q_selected_mean[i], 2), fmt(summary$noise_q_selected_mean[i], 2),
    fmt(summary$F1_mean[i]), fmt(summary$ARI_mean[i]),
    fmt(summary$MSE_centered_eta_mean[i]), fmt(summary$MSE_kappa_mean[i]),
    fmt(summary$loglik_mean[i], 2), fmt(summary$common_baseline_l2_mean[i])
  ))
}
notes <- c(
  notes,
  "",
  "## Interpretation boundary",
  "",
  "- Means are replicate-level means; F1 is set to zero for a zero-support replicate.",
  "- The result is a focused implementation diagnostic, not a final publication simulation.",
  "- Existing Study B result directories were not modified.",
  sprintf("- Total elapsed time: %.1f seconds.", proc.time()[["elapsed"]] - run_start)
)
writeLines(notes, notes_path)

cat("Wrote separate diagnostic results:\n")
cat("  ", normalizePath(summary_path, winslash = "/"), "\n", sep = "")
cat("  ", normalizePath(best_path, winslash = "/"), "\n", sep = "")
cat("  ", normalizePath(calibration_path, winslash = "/"), "\n", sep = "")
cat("  ", normalizePath(notes_path, winslash = "/"), "\n", sep = "")
print(summary[, c(
  "method", "rule", "reps", "valid_reps", "selected_q_mean",
  "common_q_selected_mean", "decision_q_selected_mean", "noise_q_selected_mean",
  "F1_mean", "ARI_mean", "MSE_centered_eta_mean", "MSE_kappa_mean",
  "loglik_mean", "common_baseline_l2_mean"
)])
