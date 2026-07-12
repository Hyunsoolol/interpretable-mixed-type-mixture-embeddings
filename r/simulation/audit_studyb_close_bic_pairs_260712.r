options(stringsAsFactors = FALSE)

source_v2_helpers_without_running <- function() {
  path <- file.path("r", "simulation", "paper_eta_studyb_confirmatory_v2_260711.r")
  lines <- readLines(path, warn = FALSE, encoding = "UTF-8")
  if (length(lines) > 0L) lines[1L] <- sub("^\ufeff", "", lines[1L])
  stop_idx <- grep("^raw_rows <- list\\(\\)", lines)[1L]
  if (is.na(stop_idx) || stop_idx <= 1L) stop("Cannot find v2 execution boundary.")
  eval(parse(text = lines[seq_len(stop_idx - 1L)]), envir = .GlobalEnv)
}

Sys.setenv(USE_RCPP_HELPERS = "1")
source_v2_helpers_without_running()

input_path <- file.path(
  "results", "paper_eta_studyb_v2_refitB_guard40_all6_rep100_260712",
  "studyb_close_bic_pairs_for_tight_audit.csv"
)
output_dir <- dirname(input_path)
pairs <- read.csv(input_path, check.names = FALSE)
if (nrow(pairs) != 9L) stop("Expected nine close-BIC method-replicates.")

cfg$nstart <- 10L
cfg$max_iter <- 100L
cfg$eta_steps <- 240L
cfg$base_seed <- 20260711L
cfg$calibration_iter <- 18L
cfg$calibration_mc_n <- 10000L
cfg$validation_mc_n <- 50000L
cfg$optim_maxit <- 80L
cfg$adaptive_gamma <- 1
cfg$adaptive_eps <- 1e-6
cfg$min_rel_lambda <- 1e-3

parse_design <- function(row) {
  kappa_pattern <- if (grepl("_equal$", row$cell)) "equal" else "heterogeneous"
  kappa <- if (kappa_pattern == "equal") rep(45, 4L) else c(30, 40, 50, 60)
  target <- if (grepl("^eB025_", row$cell)) {
    0.025
  } else if (grepl("^eB050_", row$cell)) {
    0.05
  } else {
    0.10
  }
  n <- as.integer(sub("^.*_n([0-9]+)_.*$", "\\1", row$cell))
  list(kappa_pattern = kappa_pattern, kappa = kappa, target = target, n = n)
}

parameter_cache <- new.env(parent = emptyenv())
get_parameters <- function(design) {
  key <- paste(design$kappa_pattern, design$target, sep = "|")
  if (!exists(key, parameter_cache, inherits = FALSE)) {
    cfg$kappa <- design$kappa
    cfg$target_oracle_error <- design$target
    cfg$scenario_id <- sprintf(
      "OBE%03d_%s_calibration", round(1000 * design$target), design$kappa_pattern
    )
    calibration <- calibrate_oracle_A_crn(cfg$kappa, cfg$base_seed + 5100L)
    params <- make_oracle_eta_params_for_A(
      cfg, cfg$kappa, calibration$common_norm[1L]
    )
    assign(key, list(params = params, calibration = calibration), parameter_cache)
  }
  get(key, parameter_cache, inherits = FALSE)
}

fit_rows <- list()
pair_rows <- list()
for (i in seq_len(nrow(pairs))) {
  pair <- pairs[i, , drop = FALSE]
  design <- parse_design(pair)
  cfg$n <- design$n
  cfg$n_values <- design$n
  cfg$kappa <- design$kappa
  cfg$kappa_label <- design$kappa_pattern
  cfg$target_oracle_error <- design$target
  cached <- get_parameters(design)
  params <- cached$params

  seed_offset <- 101000L + as.integer(pair$rep)
  set.seed(cfg$base_seed + seed_offset)
  dat <- simulate_from_params(cfg$n, params)
  cfg$current_init_seed <- cfg$base_seed + 900000L + seed_offset
  set.seed(cfg$current_init_seed)
  dense <- fit_svMF_multistart(
    dat$X, cfg$K, beta = 0, nstart = cfg$nstart, max_iter = cfg$max_iter
  )
  path <- fit_eta_candidate_path(dat$X, dense, pair$method)

  roles <- c("winner", "runnerup")
  sources <- as.integer(c(pair$winner_source, pair$runnerup_source))
  expected_keys <- c(pair$winner_key, pair$runnerup_key)
  old_bic <- as.numeric(c(pair$winner_bic, pair$runnerup_bic))
  one_pair <- vector("list", 2L)
  for (j in seq_along(roles)) {
    source <- sources[j]
    if (source < 1L || source > nrow(path$path)) stop("Source index outside path.")
    active <- path$actives[[source]]
    observed_key <- support_key(active)
    if (!identical(observed_key, expected_keys[j])) {
      stop(sprintf("Support mismatch for %s rep %d %s.", pair$cell, pair$rep, roles[j]))
    }
    exact <- fit_eta_centered_support_refit_exact(
      dat$X, active, path$fits[[source]], max_iter = 1000L,
      optim_maxit = cfg$optim_maxit, abs_tol = 1e-9, rel_tol = 1e-10
    )
    ic <- main_ic(exact$loglik, active, cfg$n)
    one_pair[[j]] <- data.frame(
      cell = pair$cell, rep = as.integer(pair$rep), method = pair$method,
      role = roles[j], source_path_index = source, support_key = observed_key,
      selected_q = sum(active), old_BIC = old_bic[j], tight_BIC = ic$BIC,
      BIC_change = ic$BIC - old_bic[j], loglik = exact$loglik,
      df = ic$df, converged = exact$converged, failed = exact$failed,
      stop_reason = exact$failed_reason, iter = exact$iter,
      constraint_error = exact$constraint_error,
      min_loglik_diff = exact$min_loglik_diff,
      min_q_diff = exact$min_q_diff,
      max_inner_convergence = exact$max_inner_convergence,
      max_inner_gradient_abs = exact$max_inner_gradient_abs,
      stringsAsFactors = FALSE
    )
  }
  one_pair <- do.call(rbind, one_pair)
  fit_rows[[i]] <- one_pair
  tight_order <- order(one_pair$tight_BIC, one_pair$selected_q, one_pair$source_path_index)
  pair_rows[[i]] <- data.frame(
    cell = pair$cell, rep = as.integer(pair$rep), method = pair$method,
    original_margin = as.numeric(pair$margin),
    tight_margin_original_order = one_pair$tight_BIC[2L] - one_pair$tight_BIC[1L],
    tight_selected_role = one_pair$role[tight_order[1L]],
    tight_selected_support = one_pair$support_key[tight_order[1L]],
    original_winner_retained = identical(one_pair$role[tight_order[1L]], "winner"),
    both_converged = all(one_pair$converged %in% TRUE) && !any(one_pair$failed %in% TRUE),
    stringsAsFactors = FALSE
  )
  cat(sprintf("[%d/%d] %s rep %d %s complete\n", i, nrow(pairs), pair$cell, pair$rep, pair$method))
}

fits <- do.call(rbind, fit_rows)
comparison <- do.call(rbind, pair_rows)
fit_path <- file.path(output_dir, "studyb_close_bic_tight_refits.csv")
comparison_path <- file.path(output_dir, "studyb_close_bic_tight_comparison.csv")
notes_path <- file.path(output_dir, "studyb_close_bic_tight_audit.md")
write.csv(fits, fit_path, row.names = FALSE)
write.csv(comparison, comparison_path, row.names = FALSE)

pass <- all(comparison$both_converged) && all(comparison$original_winner_retained) &&
  all(is.finite(fits$constraint_error)) && max(fits$constraint_error) <= 1e-8 &&
  all(fits$max_inner_convergence == 0L) &&
  all(fits$min_loglik_diff >= -1e-8) && all(fits$min_q_diff >= -1e-6)
notes <- c(
  "# Study B close-BIC tight-tolerance audit",
  "",
  "- Diagnostic only; the main path and candidate set are unchanged.",
  "- The top two BIC-after supports were refit from their original path states.",
  "- Tight outer tolerances: abs_tol=1e-9, rel_tol=1e-10; maximum iterations=1000.",
  sprintf("- Audited method-replicates: %d.", nrow(comparison)),
  sprintf("- Both top-two fits converged: %d/%d.", sum(comparison$both_converged), nrow(comparison)),
  sprintf("- Original winner retained: %d/%d.", sum(comparison$original_winner_retained), nrow(comparison)),
  sprintf("- Minimum tight BIC margin in original order: %.6f.", min(comparison$tight_margin_original_order)),
  sprintf("- Maximum constraint error: %.3e.", max(fits$constraint_error)),
  sprintf("- Overall audit: %s.", if (pass) "PASS" else "FAIL")
)
writeLines(notes, notes_path)
cat(sprintf("Tight close-BIC audit: %s\n", if (pass) "PASS" else "FAIL"))
if (!pass) quit(status = 1L)
