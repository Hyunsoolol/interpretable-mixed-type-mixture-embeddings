# ==============================================================================
# Study B confirmatory v2 pilot
# ------------------------------------------------------------------------------
# Separate diagnostic runner. Existing paper runners and results are unchanged.
# E-CGL/E-ACGL use exact fixed-support centered-Eta refits. Both current
# BIC-before-refit and BIC-after-exact-refit selectors are retained.
# ==============================================================================

options(stringsAsFactors = FALSE)

parse_num_grid <- function(x) as.numeric(strsplit(x, ",", fixed = TRUE)[[1L]])

source_exact_helpers_without_running <- function() {
  path <- file.path("r", "simulation", "paper_eta_exact_refit_confirmatory_diag_260711.r")
  lines <- readLines(path, warn = FALSE, encoding = "UTF-8")
  if (length(lines) > 0L) lines[1L] <- sub("^\ufeff", "", lines[1L])
  stop_idx <- grep("^calibration <- calibrate_oracle_A_crn", lines)[1L]
  if (is.na(stop_idx) || stop_idx <= 1L) stop("Cannot find exact diagnostic execution boundary.")
  eval(parse(text = lines[seq_len(stop_idx - 1L)]), envir = .GlobalEnv)
}

Sys.setenv(USE_RCPP_HELPERS = Sys.getenv("USE_RCPP_HELPERS", "1"))
source_exact_helpers_without_running()
source(file.path("r", "methods", "eta_centered_exact_refit_260711.r"))

cfg$run_label <- Sys.getenv("V2_RUN_LABEL", "paper_eta_studyb_confirmatory_v2_rep5_260711")
cfg$out_dir <- Sys.getenv(
  "V2_OUT_DIR", file.path("results", "paper_eta_studyb_confirmatory_v2_rep5_260711")
)
cfg$n_values <- as.integer(parse_num_grid(Sys.getenv("V2_N_VALUES", "300,1000")))
cfg$target_eb_values <- parse_num_grid(Sys.getenv("V2_EB_VALUES", "0.05,0.10"))
cfg$kappa <- parse_num_grid(Sys.getenv("V2_KAPPA", "30,40,50,60"))
cfg$kappa_label <- Sys.getenv("V2_KAPPA_LABEL", "heterogeneous")
cfg$methods <- strsplit(
  Sys.getenv("V2_METHODS", "D-L,D-GL,D-AGL,E-L,E-CGL,E-ACGL"),
  ",", fixed = TRUE
)[[1L]]
cfg$n_rep <- as.integer(Sys.getenv("V2_N_REP", "5"))
cfg$d <- as.integer(Sys.getenv("V2_D", "200"))
cfg$K <- 4L
cfg$common_q <- 4L
cfg$decision_per_component <- 4L
cfg$nstart <- as.integer(Sys.getenv("V2_NSTART", "10"))
cfg$max_iter <- as.integer(Sys.getenv("V2_MAX_ITER", "100"))
cfg$d_l_steps <- as.integer(Sys.getenv("V2_D_L_STEPS", "240"))
cfg$group_steps <- as.integer(Sys.getenv("V2_GROUP_STEPS", "240"))
cfg$eta_steps <- as.integer(Sys.getenv("V2_ETA_STEPS", "240"))
cfg$select_ic <- "BIC"
cfg$base_seed <- as.integer(Sys.getenv("V2_BASE_SEED", "20260711"))
cfg$calibration_iter <- as.integer(Sys.getenv("V2_CALIBRATION_ITER", "18"))
cfg$calibration_mc_n <- as.integer(Sys.getenv("V2_CALIBRATION_MC_N", "10000"))
cfg$validation_mc_n <- as.integer(Sys.getenv("V2_VALIDATION_MC_N", "50000"))
cfg$test_n <- as.integer(Sys.getenv("V2_TEST_N", "2000"))
cfg$exact_max_iter <- as.integer(Sys.getenv("V2_REFIT_MAX_ITER", "160"))
cfg$exact_retry_max_iter <- as.integer(Sys.getenv("V2_REFIT_RETRY_MAX_ITER", "840"))
cfg$optim_maxit <- as.integer(Sys.getenv("V2_OPTIM_MAXIT", "80"))
cfg$refit_shortlist <- as.integer(Sys.getenv("V2_REFIT_SHORTLIST", "0"))
guard_default <- if (cfg$refit_shortlist > 0L) {
  max(1L, cfg$refit_shortlist - 2L)
} else {
  0L
}
cfg$refit_guard_rank <- as.integer(Sys.getenv(
  "V2_REFIT_GUARD_RANK", as.character(guard_default)
))
if (length(cfg$kappa) != cfg$K) stop("V2_KAPPA length must match K.")
valid_methods <- c("D-L", "D-GL", "D-AGL", "E-L", "E-CGL", "E-ACGL")
if (!length(cfg$methods) || any(!cfg$methods %in% valid_methods)) {
  stop("V2_METHODS contains an unsupported method.")
}
if (!is.finite(cfg$refit_shortlist) || cfg$refit_shortlist < 0L) {
  stop("V2_REFIT_SHORTLIST must be a nonnegative integer.")
}
if (!is.finite(cfg$refit_guard_rank) || cfg$refit_guard_rank < 0L) {
  stop("V2_REFIT_GUARD_RANK must be a nonnegative integer.")
}
if (!is.finite(cfg$exact_retry_max_iter) || cfg$exact_retry_max_iter < 0L) {
  stop("V2_REFIT_RETRY_MAX_ITER must be a nonnegative integer.")
}
dir.create(cfg$out_dir, recursive = TRUE, showWarnings = FALSE)

rbind_fill <- function(xs) {
  xs <- Filter(Negate(is.null), xs)
  if (!length(xs)) return(data.frame())
  all_names <- unique(unlist(lapply(xs, names), use.names = FALSE))
  xs <- lapply(xs, function(x) {
    missing <- setdiff(all_names, names(x))
    for (nm in missing) x[[nm]] <- NA
    x[, all_names, drop = FALSE]
  })
  do.call(rbind, xs)
}

heldout_nll <- function(fit, X_test) {
  if (is.null(fit) || !is.finite(fit$loglik)) return(NA_real_)
  e <- e_step_vmf(X_test, fit)
  -e$loglik / nrow(X_test)
}

fit_exact_with_retry <- function(X, active, init) {
  first <- fit_eta_centered_support_refit_exact(
    X, active, init, max_iter = cfg$exact_max_iter,
    optim_maxit = cfg$optim_maxit
  )
  first$retry_used <- FALSE
  first$initial_iter <- first$iter
  first$retry_iter <- 0L
  if (isTRUE(first$failed) || isTRUE(first$converged) ||
      cfg$exact_retry_max_iter <= 0L || !is.finite(first$loglik)) {
    return(first)
  }

  retry <- fit_eta_centered_support_refit_exact(
    X, active, first, max_iter = cfg$exact_retry_max_iter,
    optim_maxit = cfg$optim_maxit
  )
  retry$retry_used <- TRUE
  retry$initial_iter <- first$iter
  retry$retry_iter <- retry$iter
  retry$iter <- first$iter + retry$iter
  retry$elapsed_sec <- first$elapsed_sec + retry$elapsed_sec
  retry$initial_loglik <- first$initial_loglik
  retry$loglik_gain <- retry$loglik - first$initial_loglik
  retry$n_halving <- first$n_halving + retry$n_halving
  retry$min_loglik_diff <- suppressWarnings(min(
    c(first$min_loglik_diff, retry$min_loglik_diff), na.rm = TRUE
  ))
  retry$min_q_diff <- suppressWarnings(min(
    c(first$min_q_diff, retry$min_q_diff), na.rm = TRUE
  ))
  retry$max_inner_convergence <- max(
    first$max_inner_convergence, retry$max_inner_convergence, na.rm = TRUE
  )
  retry$max_inner_gradient_abs <- max(
    first$max_inner_gradient_abs, retry$max_inner_gradient_abs, na.rm = TRUE
  )
  retry
}

add_counts_and_meta <- function(row, params, cell, rep_id, achieved, elapsed, rule) {
  if (!nrow(row)) return(row)
  row$method <- sub("^E-GL$", "E-CGL", row$method)
  row$method <- sub("^E-AGL$", "E-ACGL", row$method)
  if (is.na(row$F1) && is.finite(row$selected_q) && row$selected_q == 0) row$F1 <- 0
  row$common_q_selected <- round(row$common_false_selection_rate * length(params$common_idx))
  row$decision_q_selected <- round(row$decision_selection_rate * length(params$decision_idx))
  row$noise_q_selected <- round(row$noise_false_selection_rate * length(params$noise_idx))
  row$cell <- cell
  row$scenario <- cell
  row$rep <- rep_id
  row$n <- cfg$n
  row$d <- cfg$d
  row$K_true <- cfg$K
  row$target_oracle_error <- cfg$target_oracle_error
  row$achieved_oracle_error <- achieved
  row$rule <- rule
  row$method_elapsed_sec <- elapsed
  row$test_NLL <- NA_real_
  row$shared_initialization_seed <- cfg$current_init_seed
  row
}

run_standard_method <- function(fun, method, X, z, params, cell, rep_id, achieved) {
  set.seed(cfg$current_init_seed)
  start <- proc.time()[["elapsed"]]
  row <- tryCatch(
    fun(),
    error = function(e) data.frame(
      method = method, ARI = NA_real_, selected_q = NA_real_, TPR = NA_real_,
      FPR = NA_real_, Precision = NA_real_, F1 = NA_real_, loglik = NA_real_,
      MSE_mu = NA_real_, MSE_kappa = NA_real_, MSE_centered_eta = NA_real_,
      common_false_selection_rate = NA_real_, decision_selection_rate = NA_real_,
      noise_false_selection_rate = NA_real_, converged = FALSE, iter = NA_real_,
      refit_status = paste0("ERROR: ", conditionMessage(e))
    )
  )
  add_counts_and_meta(
    row, params, cell, rep_id, achieved,
    proc.time()[["elapsed"]] - start, "current_BIC_before_support_refit"
  )
}

fit_e_group_rules <- function(X, X_test, z, params, dense, method, cell, rep_id, achieved) {
  start <- proc.time()[["elapsed"]]
  path <- fit_eta_candidate_path(X, dense, method)
  path_tab <- path$path
  current_idx <- which.min(path_tab$BIC_before_main)
  unique_keys <- unique(path_tab$support_key)
  source_index <- integer(length(unique_keys))

  for (u in seq_along(unique_keys)) {
    indices <- which(path_tab$support_key == unique_keys[u])
    source_index[u] <- indices[which.max(path_tab$loglik_before[indices])]
  }

  support_order <- order(
    path_tab$BIC_before_main[source_index],
    path_tab$selected_q[source_index],
    source_index,
    na.last = TRUE
  )
  support_rank <- integer(length(unique_keys))
  support_rank[support_order] <- seq_along(support_order)
  shortlist_applied <-
    cfg$refit_shortlist > 0L && cfg$refit_shortlist < length(unique_keys)
  candidate_u <- if (shortlist_applied) {
    support_order[seq_len(cfg$refit_shortlist)]
  } else {
    seq_along(unique_keys)
  }
  current_key <- path_tab$support_key[current_idx]
  candidate_u <- unique(c(match(current_key, unique_keys), candidate_u))

  exact_by_key <- list()
  refit_candidates <- function(candidate_indices) {
    rows <- vector("list", length(candidate_indices))
    for (r in seq_along(candidate_indices)) {
      u <- candidate_indices[r]
      index <- source_index[u]
      active <- path$actives[[index]]
      exact <- fit_exact_with_retry(X, active, path$fits[[index]])
      exact_by_key[[unique_keys[u]]] <<- exact
      ic_after <- main_ic(exact$loglik, active, nrow(X))
      eligible <- isTRUE(exact$converged) && !isTRUE(exact$failed) &&
        is.finite(exact$loglik) && is.finite(ic_after$BIC) &&
        is.finite(exact$constraint_error) && exact$constraint_error <= 1e-8
      rows[[r]] <- data.frame(
        cell = cell, rep = rep_id, method = method,
        support_key = unique_keys[u], source_path_index = index,
        rank_before_main = support_rank[u],
        lambda_eta = path_tab$lambda_eta[index], selected_q = sum(active),
        common_q = sum(active[params$common_idx]),
        decision_q = sum(active[params$decision_idx]),
        noise_q = sum(active[params$noise_idx]),
        loglik_before = path_tab$loglik_before[index],
        BIC_before_current = path_tab$BIC_before_current[index],
        BIC_before_main = path_tab$BIC_before_main[index],
        loglik_after_exact = exact$loglik, BIC_after_exact = ic_after$BIC,
        BIC_after_selection = if (eligible) ic_after$BIC else Inf,
        test_NLL_after_exact = heldout_nll(exact, X_test),
        exact_eligible = eligible,
        exact_converged = exact$converged, exact_failed = exact$failed,
        exact_iter = exact$iter, exact_constraint_error = exact$constraint_error,
        exact_retry_used = exact$retry_used,
        exact_initial_iter = exact$initial_iter,
        exact_retry_iter = exact$retry_iter,
        exact_stop_reason = exact$failed_reason,
        exact_min_loglik_diff = exact$min_loglik_diff,
        exact_min_q_diff = exact$min_q_diff,
        exact_max_inner_convergence = exact$max_inner_convergence,
        exact_max_inner_gradient_abs = exact$max_inner_gradient_abs,
        exact_elapsed_sec = exact$elapsed_sec,
        unique_support_count = length(unique_keys),
        shortlist_requested = cfg$refit_shortlist,
        shortlist_applied = shortlist_applied
      )
    }
    do.call(rbind, rows)
  }

  candidates <- refit_candidates(candidate_u)
  eligible <- candidates$exact_eligible %in% TRUE &
    is.finite(candidates$BIC_after_selection)
  fallback_reason <- "none"
  if (shortlist_applied) {
    if (!any(eligible)) {
      fallback_reason <- "no_eligible_shortlist_candidate"
    } else {
      initial_best <- which.min(candidates$BIC_after_selection)
      if (cfg$refit_guard_rank > 0L &&
          candidates$rank_before_main[initial_best] >= cfg$refit_guard_rank) {
        fallback_reason <- "winner_near_shortlist_boundary"
      }
    }
  }
  shortlist_fallback_full <- fallback_reason != "none"
  if (shortlist_fallback_full) {
    remaining_u <- setdiff(seq_along(unique_keys), candidate_u)
    if (length(remaining_u)) {
      candidates <- rbind(candidates, refit_candidates(remaining_u))
    }
  }
  candidates$shortlist_fallback_full <- shortlist_fallback_full
  candidates$shortlist_fallback_reason <- fallback_reason
  eligible <- candidates$exact_eligible %in% TRUE &
    is.finite(candidates$BIC_after_selection)
  if (!any(eligible)) {
    stop(sprintf("No eligible exact centered-Eta refit for %s rep %d.", method, rep_id))
  }
  after_row <- which.min(candidates$BIC_after_selection)
  after_key <- candidates$support_key[after_row]
  after_source <- candidates$source_path_index[after_row]

  make_row <- function(key, source, rule) {
    active <- path$actives[[source]]
    fit <- exact_by_key[[key]]
    row <- evaluate_refit(
      fit, X, active, z, params, method, rule, path_tab$lambda_eta[source]
    )
    row$test_NLL <- heldout_nll(fit, X_test)
    row$cell <- cell
    row$scenario <- cell
    row$rep <- rep_id
    row$n <- cfg$n
    row$d <- cfg$d
    row$K_true <- cfg$K
    row$target_oracle_error <- cfg$target_oracle_error
    row$achieved_oracle_error <- achieved
    row$method_elapsed_sec <- proc.time()[["elapsed"]] - start
    row$shared_initialization_seed <- cfg$current_init_seed
    row$current_path_index <- current_idx
    row$selected_path_index <- source
    row$support_changed <- key != current_key
    row$exact_candidate_count <- nrow(candidates)
    row$exact_unique_support_count <- length(unique_keys)
    row$exact_shortlist_requested <- cfg$refit_shortlist
    row$exact_shortlist_applied <- shortlist_applied
    row$exact_shortlist_fallback_full <- shortlist_fallback_full
    row$exact_shortlist_fallback_reason <- fallback_reason
    row$exact_invalid_candidate_count <- sum(!eligible)
    row
  }

  rows <- rbind(
    make_row(current_key, current_idx, "current_BIC_before_exact_refit"),
    make_row(after_key, after_source, "BIC_after_exact_refit")
  )
  list(rows = rows, candidates = candidates)
}

summarize_results <- function(raw) {
  if (!nrow(raw)) return(data.frame())
  metrics <- c(
    "selected_q", "common_q_selected", "decision_q_selected", "noise_q_selected",
    "TPR", "FPR", "Precision", "F1", "ARI", "loglik", "test_NLL",
    "MSE_mu", "MSE_kappa", "MSE_centered_eta", "method_elapsed_sec"
  )
  groups <- unique(raw[, c("cell", "target_oracle_error", "n", "method", "rule")])
  out <- lapply(seq_len(nrow(groups)), function(i) {
    g <- groups[i, ]
    sub <- raw[
      raw$cell == g$cell & raw$method == g$method & raw$rule == g$rule,
      , drop = FALSE
    ]
    one <- data.frame(
      cell = g$cell, target_oracle_error = g$target_oracle_error,
      n = g$n, method = g$method, rule = g$rule,
      reps = length(unique(sub$rep)), valid_reps = sum(is.finite(sub$ARI)),
      zero_support_reps = sum(sub$selected_q == 0, na.rm = TRUE),
      convergence_rate = mean(sub$converged %in% TRUE, na.rm = TRUE)
    )
    for (metric in metrics) {
      value <- suppressWarnings(as.numeric(sub[[metric]]))
      valid <- is.finite(value)
      one[[paste0(metric, "_mean")]] <- if (any(valid)) mean(value[valid]) else NA_real_
      one[[paste0(metric, "_sd")]] <- if (sum(valid) > 1L) sd(value[valid]) else NA_real_
      one[[paste0(metric, "_mcse")]] <- if (sum(valid) > 1L) sd(value[valid]) / sqrt(sum(valid)) else NA_real_
    }
    one
  })
  do.call(rbind, out)
}

selector_deltas <- function(raw) {
  out <- list()
  groups <- unique(raw[raw$method %in% c("E-CGL", "E-ACGL"), c("cell", "rep", "method")])
  metrics <- c("selected_q", "F1", "ARI", "MSE_centered_eta", "MSE_kappa", "test_NLL")
  for (i in seq_len(nrow(groups))) {
    g <- groups[i, ]
    sub <- raw[raw$cell == g$cell & raw$rep == g$rep & raw$method == g$method, ]
    before <- sub[sub$rule == "current_BIC_before_exact_refit", ]
    after <- sub[sub$rule == "BIC_after_exact_refit", ]
    if (!nrow(before) || !nrow(after)) next
    row <- data.frame(cell = g$cell, rep = g$rep, method = g$method)
    for (metric in metrics) row[[paste0("delta_", metric)]] <- after[[metric]] - before[[metric]]
    out[[length(out) + 1L]] <- row
  }
  rbind_fill(out)
}

raw_rows <- list()
candidate_rows <- list()
calibration_rows <- list()
status_rows <- list()
run_start <- proc.time()[["elapsed"]]

raw_path <- file.path(cfg$out_dir, paste0(cfg$run_label, "_raw.csv"))
candidate_path <- file.path(cfg$out_dir, paste0(cfg$run_label, "_candidate_supports.csv"))
summary_path <- file.path(cfg$out_dir, paste0(cfg$run_label, "_summary.csv"))
calibration_path <- file.path(cfg$out_dir, paste0(cfg$run_label, "_calibration.csv"))
delta_path <- file.path(cfg$out_dir, paste0(cfg$run_label, "_selector_deltas.csv"))
status_path <- file.path(cfg$out_dir, paste0(cfg$run_label, "_status.csv"))
notes_path <- file.path(cfg$out_dir, paste0(cfg$run_label, "_notes.md"))
completion_path <- file.path(cfg$out_dir, paste0(cfg$run_label, "_complete.ok"))
if (file.exists(completion_path)) unlink(completion_path)

checkpoint <- function() {
  raw <- rbind_fill(raw_rows)
  candidates <- rbind_fill(candidate_rows)
  if (nrow(raw)) {
    write.csv(raw, raw_path, row.names = FALSE)
    write.csv(summarize_results(raw), summary_path, row.names = FALSE)
    write.csv(selector_deltas(raw), delta_path, row.names = FALSE)
  }
  if (nrow(candidates)) write.csv(candidates, candidate_path, row.names = FALSE)
  if (length(calibration_rows)) write.csv(do.call(rbind, calibration_rows), calibration_path, row.names = FALSE)
  if (length(status_rows)) write.csv(do.call(rbind, status_rows), status_path, row.names = FALSE)
}

for (eb_index in seq_along(cfg$target_eb_values)) {
  cfg$target_oracle_error <- cfg$target_eb_values[eb_index]
  cfg$scenario_id <- sprintf(
    "OBE%03d_%s_calibration",
    round(1000 * cfg$target_oracle_error), cfg$kappa_label
  )
  calibration <- calibrate_oracle_A_crn(
    cfg$kappa, cfg$base_seed + 5000L + 100L * eb_index
  )
  calibration$calibration_id <- cfg$scenario_id
  calibration_rows[[length(calibration_rows) + 1L]] <- calibration
  params <- make_oracle_eta_params_for_A(cfg, cfg$kappa, calibration$common_norm[1L])

  for (n_value in cfg$n_values) {
    cfg$n <- n_value
    cell <- sprintf(
      "eB%03d_n%d_%s",
      round(1000 * cfg$target_oracle_error), cfg$n, cfg$kappa_label
    )
    cat(sprintf("[%s] start: reps=%d, path=%d\n", cell, cfg$n_rep, cfg$eta_steps))
    cell_start <- proc.time()[["elapsed"]]

    for (rep_id in seq_len(cfg$n_rep)) {
      seed_offset <- 100000L * eb_index + 1000L * match(n_value, cfg$n_values) + rep_id
      set.seed(cfg$base_seed + seed_offset)
      dat <- simulate_from_params(cfg$n, params)
      set.seed(cfg$base_seed + 500000L + seed_offset)
      test_dat <- simulate_from_params(cfg$test_n, params)
      cfg$current_init_seed <- cfg$base_seed + 900000L + seed_offset

      standard_methods <- list(
        "D-L" = function() fit_d_l(dat$X, dat$z, params, cfg),
        "D-GL" = function() fit_d_group_path(dat$X, dat$z, params, cfg, adaptive = FALSE),
        "D-AGL" = function() fit_d_group_path(dat$X, dat$z, params, cfg, adaptive = TRUE),
        "E-L" = function() fit_eta_path(dat$X, dat$z, params, cfg, "E-L")
      )
      standard_methods <- standard_methods[
        intersect(names(standard_methods), cfg$methods)
      ]
      for (method in names(standard_methods)) {
        raw_rows[[length(raw_rows) + 1L]] <- run_standard_method(
          standard_methods[[method]], method, dat$X, dat$z, params, cell, rep_id,
          calibration$achieved_oracle_error[1L]
        )
      }

      e_group_methods <- intersect(c("E-CGL", "E-ACGL"), cfg$methods)
      if (length(e_group_methods)) {
        set.seed(cfg$current_init_seed)
        dense <- fit_svMF_multistart(
          dat$X, cfg$K, beta = 0, nstart = cfg$nstart, max_iter = cfg$max_iter
        )
        for (method in e_group_methods) {
          result <- fit_e_group_rules(
            dat$X, test_dat$X, dat$z, params, dense, method, cell, rep_id,
            calibration$achieved_oracle_error[1L]
          )
          raw_rows[[length(raw_rows) + 1L]] <- result$rows
          candidate_rows[[length(candidate_rows) + 1L]] <- result$candidates
        }
      }
      cat(sprintf("[%s] rep %d/%d complete\n", cell, rep_id, cfg$n_rep))
      checkpoint()
    }

    cell_raw <- rbind_fill(raw_rows)
    cell_raw <- cell_raw[cell_raw$cell == cell, , drop = FALSE]
    standard_expected <- intersect(
      c("D-L", "D-GL", "D-AGL", "E-L"), cfg$methods
    )
    group_expected <- intersect(c("E-CGL", "E-ACGL"), cfg$methods)
    expected <- rbind_fill(list(
      if (length(standard_expected)) expand.grid(
        rep = seq_len(cfg$n_rep), method = standard_expected,
        rule = "current_BIC_before_support_refit", stringsAsFactors = FALSE
      ),
      if (length(group_expected)) expand.grid(
        rep = seq_len(cfg$n_rep), method = group_expected,
        rule = c("current_BIC_before_exact_refit", "BIC_after_exact_refit"),
        stringsAsFactors = FALSE
      )
    ))
    expected_key <- paste(expected$rep, expected$method, expected$rule, sep = "|")
    actual_key <- paste(cell_raw$rep, cell_raw$method, cell_raw$rule, sep = "|")
    key_counts <- table(factor(actual_key, levels = expected_key))
    row_structure_ok <- nrow(cell_raw) == nrow(expected) && all(key_counts == 1L)
    finite_columns <- intersect(
      c("ARI", "selected_q", "loglik", "MSE_centered_eta"), names(cell_raw)
    )
    finite_rows <- if (length(finite_columns)) {
      apply(cell_raw[, finite_columns, drop = FALSE], 1L, function(x) {
        all(is.finite(as.numeric(x)))
      })
    } else {
      rep(FALSE, nrow(cell_raw))
    }
    zero_support_rows <- is.finite(cell_raw$selected_q) & cell_raw$selected_q == 0
    numeric_rows_ok <- finite_rows | zero_support_rows
    error_rows <- if ("refit_status" %in% names(cell_raw)) {
      grepl("^ERROR:", cell_raw$refit_status)
    } else {
      rep(FALSE, nrow(cell_raw))
    }
    group_rows <- cell_raw[
      cell_raw$method %in% group_expected &
        cell_raw$rule %in% c(
          "current_BIC_before_exact_refit", "BIC_after_exact_refit"
        ), , drop = FALSE
    ]
    group_rows_ok <- !length(group_expected) || (
      nrow(group_rows) == 2L * cfg$n_rep * length(group_expected) &&
        all(group_rows$converged %in% TRUE) &&
        all(is.finite(group_rows$constraint_error)) &&
        all(group_rows$constraint_error <= 1e-8)
    )
    candidate_all <- rbind_fill(candidate_rows)
    candidate_cell <- candidate_all[candidate_all$cell == cell, , drop = FALSE]
    invalid_candidates <- if (nrow(candidate_cell)) {
      sum(!(candidate_cell$exact_eligible %in% TRUE))
    } else {
      0L
    }
    cell_ok <- row_structure_ok && all(numeric_rows_ok) && !any(error_rows) &&
      group_rows_ok && invalid_candidates == 0L
    status_rows[[length(status_rows) + 1L]] <- data.frame(
      cell = cell,
      completed_reps = length(unique(cell_raw$rep)),
      expected_rows = nrow(expected), actual_rows = nrow(cell_raw),
      error_rows = sum(error_rows), zero_support_rows = sum(zero_support_rows),
      invalid_candidate_refits = invalid_candidates,
      group_selector_refits_converged = if (nrow(group_rows)) {
        sum(group_rows$converged %in% TRUE)
      } else {
        0L
      },
      status = if (cell_ok) "complete" else "failed_validation",
      elapsed_sec = proc.time()[["elapsed"]] - cell_start
    )
    checkpoint()
    if (!cell_ok) {
      stop(sprintf("Cell-level validation failed for %s; see status and raw CSV.", cell))
    }
  }
}

raw <- rbind_fill(raw_rows)
summary <- summarize_results(raw)
deltas <- selector_deltas(raw)
delta_summary <- if (nrow(deltas)) {
  do.call(rbind, lapply(split(deltas, interaction(deltas$cell, deltas$method, drop = TRUE)), function(x) {
    data.frame(
      cell = x$cell[1L], method = x$method[1L], reps = nrow(x),
      delta_selected_q = mean(x$delta_selected_q),
      delta_F1 = mean(x$delta_F1), delta_ARI = mean(x$delta_ARI),
      delta_MSE_eta = mean(x$delta_MSE_centered_eta),
      delta_MSE_kappa = mean(x$delta_MSE_kappa),
      delta_test_NLL = mean(x$delta_test_NLL)
    )
  }))
} else data.frame()

checkpoint()
delta_summary_path <- file.path(cfg$out_dir, paste0(cfg$run_label, "_selector_delta_summary.csv"))
write.csv(delta_summary, delta_summary_path, row.names = FALSE)

fmt <- function(x, digits = 3L) ifelse(is.na(x), "NA", formatC(x, digits = digits, format = "f"))
notes <- c(
  "# Study B confirmatory v2 pilot",
  "",
  "Separate diagnostic result; existing Study B outputs are unchanged.",
  "",
  "## Design",
  "",
  sprintf("- K=4, d=%d, n=(%s), reps=%d.", cfg$d, paste(cfg$n_values, collapse = ","), cfg$n_rep),
  sprintf("- Target oracle errors=(%s); %s kappa=(%s).",
          paste(cfg$target_eb_values, collapse = ","), cfg$kappa_label,
          paste(cfg$kappa, collapse = ",")),
  sprintf("- Methods=(%s).", paste(cfg$methods, collapse = ",")),
  sprintf("- Paths: D-L=%d, D-group=%d, Eta=%d; nstart=%d.",
          cfg$d_l_steps, cfg$group_steps, cfg$eta_steps, cfg$nstart),
  "- All methods use the same initialization seed within each replicate.",
  "- E-CGL/E-ACGL use exact centered-Eta support refits.",
  sprintf(
    "- Exact refit maximum iterations=%d; BIC-before shortlist=%s (0 means all unique supports).",
    cfg$exact_max_iter, cfg$refit_shortlist
  ),
  sprintf(
    "- Shortlist guard rank=%d; a winner at or beyond this rank triggers full-support refitting.",
    cfg$refit_guard_rank
  ),
  "- Failed, non-converged, non-finite, or constraint-violating exact refits are ineligible for selection.",
  "- E-CGL/E-ACGL retain both BIC-before-refit and BIC-after-exact-refit rules.",
  "- Summary metrics are replicate means with SD and Monte Carlo SE in the CSV.",
  "",
  "## Calibration",
  "",
  "| target eB | achieved eB | 95% MC interval | common norm |",
  "|---:|---:|:---|---:|"
)
calibration_all <- do.call(rbind, calibration_rows)
for (i in seq_len(nrow(calibration_all))) {
  notes <- c(notes, sprintf(
    "| %s | %s | %s to %s | %s |",
    fmt(calibration_all$target_oracle_error[i]), fmt(calibration_all$achieved_oracle_error[i]),
    fmt(calibration_all$achieved_ci_low[i]), fmt(calibration_all$achieved_ci_high[i]),
    fmt(calibration_all$common_norm[i])
  ))
}
notes <- c(
  notes, "", "## Selector deltas: BIC-after-exact minus current BIC-before", "",
  "| cell | method | delta q | delta F1 | delta ARI | delta MSE eta | delta MSE kappa | delta test NLL |",
  "|:---|:---|---:|---:|---:|---:|---:|---:|"
)
if (nrow(delta_summary)) {
  for (i in seq_len(nrow(delta_summary))) {
    notes <- c(notes, sprintf(
      "| %s | %s | %s | %s | %s | %s | %s | %s |",
      delta_summary$cell[i], delta_summary$method[i], fmt(delta_summary$delta_selected_q[i]),
      fmt(delta_summary$delta_F1[i]), fmt(delta_summary$delta_ARI[i]),
      fmt(delta_summary$delta_MSE_eta[i]), fmt(delta_summary$delta_MSE_kappa[i]),
      fmt(delta_summary$delta_test_NLL[i])
    ))
  }
}
notes <- c(
  notes, "", "## Interpretation boundary", "",
  "- This pilot checks implementation and selector behavior; it is not a final method ranking.",
  "- E-L uses an entry-wise penalty; the coordinate union selected by that path receives the same centered-support B refit for post-selection comparison.",
  "- Candidate-support output is an audit artifact and should not be committed.",
  sprintf("- Total elapsed time: %.1f seconds.", proc.time()[["elapsed"]] - run_start)
)
writeLines(notes, notes_path)
writeLines(
  c(cfg$run_label, paste0("completed_at=", format(Sys.time(), tz = "UTC", usetz = TRUE))),
  completion_path
)

cat("Wrote Study B confirmatory v2 outputs:\n")
for (path in c(
  summary_path, delta_summary_path, calibration_path, notes_path, status_path,
  completion_path
)) {
  cat("  ", normalizePath(path, winslash = "/"), "\n", sep = "")
}
print(summary[, c(
  "cell", "method", "rule", "reps", "valid_reps", "selected_q_mean",
  "decision_q_selected_mean", "noise_q_selected_mean", "F1_mean", "ARI_mean",
  "MSE_centered_eta_mean", "MSE_kappa_mean", "test_NLL_mean"
)])
