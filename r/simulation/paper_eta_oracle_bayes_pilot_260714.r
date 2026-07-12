# ==============================================================================
# Oracle Bayes error calibrated Eta-contrast pilot for Study B
# ------------------------------------------------------------------------------
# Diagnostic pilot only. This script reuses the current paper simulation fitting
# helpers without modifying the official runner.
# ==============================================================================

options(stringsAsFactors = FALSE)

parse_num_grid <- function(x) as.numeric(strsplit(x, ",", fixed = TRUE)[[1]])

source_paper_helpers_without_running <- function() {
  helper_file <- file.path("r", "simulation", "paper_eta_first_s1_run_260702.r")
  lines <- readLines(helper_file, encoding = "UTF-8", warn = FALSE)
  if (length(lines) > 0) lines[1] <- sub("^\ufeff", "", lines[1])
  run_idx <- grep("Running paper %s eta-first simulation", lines, fixed = TRUE)[1]
  if (is.na(run_idx)) stop("Could not find execution boundary in paper S1 runner.")
  cat_idx <- max(grep("^cat\\(sprintf\\(", lines[seq_len(run_idx)], perl = TRUE))
  if (!is.finite(cat_idx) || is.na(cat_idx) || cat_idx <= 1L) {
    stop("Could not find cat(sprintf execution boundary in paper S1 runner.")
  }
  eval(parse(text = lines[seq_len(cat_idx - 1L)]), envir = .GlobalEnv)
}

Sys.setenv(
  PAPER_S1_LABEL = "paper_eta_oracle_bayes_pilot_260714",
  PAPER_S1_SCENARIO_ID = "oracle_bayes_placeholder",
  PAPER_S1_SCENARIO_DESC = "Oracle Bayes error calibrated eta-contrast pilot.",
  PAPER_S1_N_REP = "5",
  PAPER_S1_N = "300",
  PAPER_S1_D = "200",
  PAPER_S1_K = "4",
  PAPER_S1_COMMON_Q = "4",
  PAPER_S1_DECISION_PER_COMPONENT = "4",
  PAPER_S1_NSTART = "10",
  PAPER_S1_MAX_ITER = "100",
  PAPER_S1_D_L_STEPS = "120",
  PAPER_S1_GROUP_STEPS = "120",
  PAPER_S1_ETA_STEPS = "120",
  PAPER_S1_SELECT_IC = Sys.getenv("ORACLE_PILOT_SELECT_IC", "BIC"),
  PAPER_S1_ETA_REFIT_MODE = Sys.getenv(
    "ORACLE_PILOT_ETA_REFIT_MODE", "BIC_AFTER_EXACT"
  ),
  PAPER_S1_ETA_REFIT_SHORTLIST = Sys.getenv(
    "ORACLE_PILOT_ETA_REFIT_SHORTLIST", "0"
  ),
  PAPER_S1_EXACT_REFIT_MAX_ITER = Sys.getenv(
    "ORACLE_PILOT_EXACT_REFIT_MAX_ITER", "160"
  ),
  PAPER_S1_BASE_SEED = "20260714",
  PAPER_S1_USE_RCPP = "1",
  PAPER_S1_OUT_DIR = "results/paper_eta_oracle_bayes_pilot_260714"
)

source_paper_helpers_without_running()

cfg$n_rep <- as.integer(Sys.getenv("ORACLE_PILOT_N_REP", "5"))
cfg$n <- as.integer(Sys.getenv("ORACLE_PILOT_N", "300"))
cfg$d <- as.integer(Sys.getenv("ORACLE_PILOT_D", "200"))
cfg$K <- as.integer(Sys.getenv("ORACLE_PILOT_K", "4"))
cfg$common_q <- as.integer(Sys.getenv("ORACLE_PILOT_COMMON_Q", as.character(round(0.02 * cfg$d))))
cfg$decision_per_component <- as.integer(Sys.getenv(
  "ORACLE_PILOT_DECISION_PER_COMPONENT",
  as.character(round(0.08 * cfg$d / cfg$K))
))
cfg$nstart <- as.integer(Sys.getenv("ORACLE_PILOT_NSTART", "10"))
cfg$max_iter <- as.integer(Sys.getenv("ORACLE_PILOT_MAX_ITER", "100"))
cfg$d_l_steps <- as.integer(Sys.getenv("ORACLE_PILOT_D_L_STEPS", "120"))
cfg$group_steps <- as.integer(Sys.getenv("ORACLE_PILOT_GROUP_STEPS", "120"))
cfg$eta_steps <- as.integer(Sys.getenv("ORACLE_PILOT_ETA_STEPS", "120"))
cfg$target_angle_deg <- NA_real_
cfg$target_oracle_error <- as.numeric(Sys.getenv("ORACLE_PILOT_TARGET_EB", "0.05"))
cfg$oracle_mc_n <- as.integer(Sys.getenv("ORACLE_PILOT_ORACLE_MC_N", "2500"))
cfg$calibration_iter <- as.integer(Sys.getenv("ORACLE_PILOT_CALIBRATION_ITER", "16"))
cfg$select_ic <- toupper(Sys.getenv("ORACLE_PILOT_SELECT_IC", cfg$select_ic))
cfg$run_label <- Sys.getenv("ORACLE_PILOT_RUN_LABEL", "paper_eta_oracle_bayes_pilot_260714")
cfg$out_dir <- Sys.getenv("ORACLE_PILOT_OUT_DIR", "results/paper_eta_oracle_bayes_pilot_260714")
dir.create(cfg$out_dir, recursive = TRUE, showWarnings = FALSE)

make_oracle_eta_params_for_A <- function(cfg, kappa, A) {
  K <- cfg$K
  d <- cfg$d
  common_q <- cfg$common_q
  per_k <- cfg$decision_per_component
  decision_q <- K * per_k

  common_idx <- seq_len(common_q)
  decision_idx <- common_q + seq_len(decision_q)
  noise_idx <- if (common_q + decision_q < d) seq.int(common_q + decision_q + 1L, d) else integer(0)

  contrast <- matrix(0, nrow = K, ncol = decision_q)
  for (g in seq_len(K)) {
    idx <- ((g - 1L) * per_k) + seq_len(per_k)
    contrast[, idx] <- -1 / (K - 1)
    contrast[g, idx] <- 1
  }

  v_norm <- sqrt(sum(contrast[1, ]^2))
  common_value <- if (common_q > 0) A / sqrt(common_q) else 0
  scale_by_k <- sqrt(pmax(kappa^2 - A^2, 0)) / v_norm

  eta <- matrix(0, nrow = K, ncol = d)
  if (common_q > 0) eta[, common_idx] <- common_value
  for (k in seq_len(K)) eta[k, decision_idx] <- scale_by_k[k] * contrast[k, ]

  kappa_actual <- sqrt(rowSums(eta^2))
  mu <- sweep(eta, 1, pmax(kappa_actual, 1e-12), "/")
  support <- matrix(FALSE, nrow = K, ncol = d)
  support[, decision_idx] <- TRUE

  pair_cos <- tcrossprod(mu)
  pair_angle <- acos(pmin(pmax(pair_cos[upper.tri(pair_cos)], -1), 1)) * 180 / pi

  list(
    alpha = rep(1 / K, K),
    mu = mu,
    kappa = kappa_actual,
    eta = eta,
    support = support,
    common_idx = common_idx,
    decision_idx = decision_idx,
    noise_idx = noise_idx,
    decision_blocks = split(decision_idx, rep(seq_len(K), each = per_k)),
    common_value = common_value,
    contrast_scale = scale_by_k,
    common_norm = A,
    decision_norm = sqrt(pmax(kappa_actual^2 - A^2, 0)),
    mu_pairwise_angle_mean = mean(pair_angle),
    mu_pairwise_angle_min = min(pair_angle),
    mu_pairwise_angle_max = max(pair_angle),
    mu_pairwise_cos_mean = mean(pair_cos[upper.tri(pair_cos)]),
    target_angle_deg = NA_real_
  )
}

oracle_error_estimate <- function(params, n_mc, seed) {
  set.seed(seed)
  dat <- simulate_from_params(n_mc, params)
  X <- dat$X
  z <- dat$z
  eta <- sweep(params$mu, 1, params$kappa, "*")
  logdens <- X %*% t(eta)
  logdens <- sweep(logdens, 2, log_vmf_const(params$kappa, ncol(X)), "+")
  logdens <- sweep(logdens, 2, log(params$alpha), "+")
  pred <- max.col(logdens, ties.method = "first")
  mean(pred != z)
}

calibrate_oracle_A <- function(cfg, kappa, scenario_seed) {
  upper <- min(kappa) * 0.995
  target <- cfg$target_oracle_error
  err_at <- function(A, offset) {
    params <- make_oracle_eta_params_for_A(cfg, kappa, A)
    oracle_error_estimate(params, cfg$oracle_mc_n, scenario_seed + offset)
  }

  err_low <- err_at(0, 11)
  err_high <- err_at(upper, 13)

  if (!is.finite(err_low) || !is.finite(err_high)) {
    stop("Oracle error calibration produced non-finite endpoints.")
  }

  lo <- 0
  hi <- upper
  if (target <= err_low) {
    best_A <- lo
  } else if (target >= err_high) {
    best_A <- hi
  } else {
    for (iter in seq_len(cfg$calibration_iter)) {
      mid <- (lo + hi) / 2
      err_mid <- err_at(mid, 100 + iter)
      if (err_mid < target) lo <- mid else hi <- mid
    }
    best_A <- (lo + hi) / 2
  }

  params <- make_oracle_eta_params_for_A(cfg, kappa, best_A)
  achieved <- oracle_error_estimate(params, cfg$oracle_mc_n * 2L, scenario_seed + 999)
  data.frame(
    common_norm = best_A,
    endpoint_error_low = err_low,
    endpoint_error_high = err_high,
    achieved_oracle_error = achieved,
    target_oracle_error = target
  )
}

target_tag <- sprintf("%03d", round(1000 * cfg$target_oracle_error))
scenario_grid <- data.frame(
  scenario = paste0(
    "OBE", target_tag, c("_equal_kappa", "_heterogeneous_kappa")
  ),
  scenario_desc = c(
    "Study B pilot: target oracle Bayes error 5%, equal kappa.",
    "Study B pilot: target oracle Bayes error 5%, heterogeneous kappa."
  ),
  kappa = c("45,45,45,45", "30,40,50,60"),
  stringsAsFactors = FALSE
)

safe_mean <- function(x) if (sum(!is.na(x)) == 0) NA_real_ else mean(x, na.rm = TRUE)

aggregate_support_all_reps <- function(sub) {
  reps <- length(unique(sub$rep))
  true_q <- safe_mean(sub$true_q)
  d_val <- safe_mean(sub$d)
  selected_q <- ifelse(is.na(sub$selected_q), 0, sub$selected_q)
  tpr <- ifelse(is.na(sub$TPR), 0, sub$TPR)
  if (!is.finite(reps) || reps == 0 || !is.finite(true_q) || true_q <= 0 ||
      !is.finite(d_val) || d_val <= true_q) {
    return(data.frame(
      valid_support_reps = sum(selected_q >= 1, na.rm = TRUE),
      valid_support_rate = NA_real_,
      zero_selected_reps = sum(selected_q < 1, na.rm = TRUE),
      Precision_all_reps = NA_real_,
      F1_all_reps = NA_real_
    ))
  }
  tp <- sum(tpr * true_q)
  selected <- sum(selected_q)
  fp <- selected - tp
  fn <- reps * true_q - tp
  precision <- if ((tp + fp) > 0) tp / (tp + fp) else NA_real_
  f1 <- if ((2 * tp + fp + fn) > 0) 2 * tp / (2 * tp + fp + fn) else NA_real_
  data.frame(
    valid_support_reps = sum(selected_q >= 1, na.rm = TRUE),
    valid_support_rate = sum(selected_q >= 1, na.rm = TRUE) / reps,
    zero_selected_reps = sum(selected_q < 1, na.rm = TRUE),
    Precision_all_reps = precision,
    F1_all_reps = f1
  )
}

summarize_raw <- function(raw) {
  num_cols <- names(raw)[vapply(raw, is.numeric, logical(1))]
  groups <- unique(raw[, c("scenario", "method")])
  summary <- do.call(rbind, lapply(seq_len(nrow(groups)), function(i) {
    sub <- raw[raw$scenario == groups$scenario[i] & raw$method == groups$method[i], ]
    means <- as.data.frame(as.list(vapply(sub[, num_cols, drop = FALSE], safe_mean, numeric(1))))
    support_all <- aggregate_support_all_reps(sub)
    data.frame(
      scenario = groups$scenario[i],
      method = groups$method[i],
      reps = length(unique(sub$rep)),
      valid_reps = sum(!is.na(sub$ARI)),
      error_reps = sum(sub$method == "ERROR"),
      zero_support_refit_reps = sum(sub$selected_q == 0, na.rm = TRUE),
      support_all,
      means,
      row.names = NULL
    )
  }))
  method_order <- c("D-L", "D-GL", "D-AGL", "E-L", "E-GL", "E-AGL", "ERROR")
  summary$method <- factor(summary$method, levels = method_order)
  summary <- summary[order(summary$scenario, summary$method), ]
  summary$method <- as.character(summary$method)
  summary
}

all_raw <- list()
calibration_rows <- list()

for (s in seq_len(nrow(scenario_grid))) {
  cfg$scenario_id <- scenario_grid$scenario[s]
  cfg$scenario_desc <- scenario_grid$scenario_desc[s]
  cfg$kappa <- parse_num_grid(scenario_grid$kappa[s])
  scenario_seed <- cfg$base_seed + 10000L * s

  calibration <- calibrate_oracle_A(cfg, cfg$kappa, scenario_seed)
  calibration$scenario <- cfg$scenario_id
  calibration$kappa <- paste(cfg$kappa, collapse = ",")
  calibration_rows[[s]] <- calibration

  make_eta_first_s1_params <- function(cfg_local) {
    make_oracle_eta_params_for_A(
      cfg_local,
      kappa = cfg$kappa,
      A = calibration$common_norm[1]
    )
  }

  cat(sprintf(
    "[%s] calibrated common norm=%.4f, achieved oracle error=%.4f, kappa=(%s)\n",
    cfg$scenario_id, calibration$common_norm[1], calibration$achieved_oracle_error[1],
    paste(cfg$kappa, collapse = ",")
  ))

  rows <- vector("list", cfg$n_rep)
  for (rep_id in seq_len(cfg$n_rep)) {
    rows[[rep_id]] <- tryCatch(
      run_one(rep_id, cfg),
      error = function(e) {
        message(sprintf("[ERROR] %s rep %d: %s", cfg$scenario_id, rep_id, conditionMessage(e)))
        data.frame(
          method = "ERROR", K_fit = NA_real_, beta = NA_real_,
          lambda_mu = NA_real_, lambda_kappa = NA_real_, lambda_eta = NA_real_,
          ARI = NA_real_, loglik = NA_real_, pen_loglik = NA_real_,
          converged = NA, iter = NA_real_, true_union_q = NA_real_,
          selected_q = NA_real_, TPR = NA_real_, FPR = NA_real_,
          Precision = NA_real_, F1 = NA_real_, entry_TPR = NA_real_,
          entry_FPR = NA_real_, entry_Precision = NA_real_, entry_F1 = NA_real_,
          MSE_mu = NA_real_, MSE_kappa = NA_real_, MSE_centered_eta = NA_real_,
          kappa_hat_mean = NA_real_, df = NA_real_, BIC = NA_real_,
          EBIC = NA_real_, common_false_selection_rate = NA_real_,
          decision_selection_rate = NA_real_, noise_false_selection_rate = NA_real_,
          objective = NA_real_, n_decrease = NA_real_, min_objective_diff = NA_real_,
          line_search_halving = NA_real_, line_search_accepted = NA,
          adaptive_penalty = NA_integer_, adaptive_gamma = NA_real_,
          adaptive_eps = NA_real_, adaptive_weight_min = NA_real_,
          adaptive_weight_median = NA_real_, adaptive_weight_max = NA_real_,
          refit_status = conditionMessage(e), penalty_target = NA_character_,
          penalty_group = NA_integer_, penalty_adaptive = NA_integer_,
          selector_rule = NA_character_, refit_type = NA_character_,
          current_bic_before_selected_q = NA_real_, support_changed_after_refit = NA,
          exact_converged = NA, exact_failed = NA, exact_iter = NA_integer_,
          exact_constraint_error = NA_real_, exact_min_loglik_diff = NA_real_,
          exact_min_q_diff = NA_real_, exact_refit_elapsed_sec = NA_real_,
          exact_candidate_count = NA_integer_, exact_unique_support_count = NA_integer_,
          exact_shortlist_requested = NA_integer_, exact_shortlist_applied = NA,
          scenario = cfg$scenario_id, rep = rep_id,
          n = cfg$n, d = cfg$d, K_true = cfg$K, common_q = cfg$common_q,
          decision_q = cfg$K * cfg$decision_per_component,
          decision_q_per_component = cfg$decision_per_component,
          noise_q = cfg$d - cfg$common_q - cfg$K * cfg$decision_per_component,
          true_q = cfg$K * cfg$decision_per_component,
          target_angle_deg = NA_real_,
          mu_pairwise_angle_mean = NA_real_, mu_pairwise_angle_min = NA_real_,
          mu_pairwise_angle_max = NA_real_, kappa_true_min = min(cfg$kappa),
          kappa_true_max = max(cfg$kappa), kappa_true_mean = mean(cfg$kappa),
          kappa_true_ratio = max(cfg$kappa) / min(cfg$kappa),
          common_eta_value = NA_real_, common_norm = calibration$common_norm[1],
          target_oracle_error = cfg$target_oracle_error,
          achieved_oracle_error = calibration$achieved_oracle_error[1]
        )
      }
    )
    rows[[rep_id]]$common_norm <- calibration$common_norm[1]
    rows[[rep_id]]$target_oracle_error <- cfg$target_oracle_error
    rows[[rep_id]]$achieved_oracle_error <- calibration$achieved_oracle_error[1]
  }
  all_raw[[s]] <- do.call(rbind, rows)
}

raw <- do.call(rbind, all_raw)
summary <- summarize_raw(raw)
calibration <- do.call(rbind, calibration_rows)

raw_path <- file.path(cfg$out_dir, sprintf("%s_raw.csv", cfg$run_label))
summary_path <- file.path(cfg$out_dir, sprintf("%s_summary.csv", cfg$run_label))
calibration_path <- file.path(cfg$out_dir, sprintf("%s_calibration.csv", cfg$run_label))
notes_path <- file.path(cfg$out_dir, sprintf("%s_notes.md", cfg$run_label))

write.csv(raw, raw_path, row.names = FALSE)
write.csv(summary, summary_path, row.names = FALSE)
write.csv(calibration, calibration_path, row.names = FALSE)

fmt <- function(x, digits = 3) ifelse(is.na(x), "NA", formatC(as.numeric(x), digits = digits, format = "f"))

notes <- c(
  "# Oracle Bayes Error Calibrated Eta-Contrast Pilot 260714",
  "",
  "## Purpose",
  "",
  "This pilot tests whether the Study B DGP can be calibrated by oracle Bayes error rather than by a fixed angle. It is a pilot run only, not a full simulation.",
  "",
  "## Setting",
  "",
  sprintf("- K=%d, n=%d, d=%d.", cfg$K, cfg$n, cfg$d),
  sprintf("- common q=%d, decision q=%d, noise q=%d.", cfg$common_q, cfg$K * cfg$decision_per_component, cfg$d - cfg$common_q - cfg$K * cfg$decision_per_component),
  sprintf("- target oracle Bayes error: %.3f.", cfg$target_oracle_error),
  sprintf("- oracle Monte Carlo n per calibration check: %d.", cfg$oracle_mc_n),
  sprintf("- reps per scenario: %d.", cfg$n_rep),
  sprintf("- nstart=%d, max_iter=%d, path steps=%d.", cfg$nstart, cfg$max_iter, cfg$eta_steps),
  sprintf("- selector: %s.", cfg$select_ic),
  sprintf("- exact Eta refit: max_iter=%d; shortlist=%d (0 means B-full).",
          cfg$exact_refit_max_iter, cfg$eta_refit_shortlist),
  "",
  "## Calibration",
  "",
  "| scenario | kappa | common norm | endpoint low | endpoint high | achieved oracle error |",
  "|:---|:---|---:|---:|---:|---:|"
)
for (i in seq_len(nrow(calibration))) {
  notes <- c(notes, sprintf(
    "| %s | %s | %s | %s | %s | %s |",
    calibration$scenario[i], calibration$kappa[i], fmt(calibration$common_norm[i]),
    fmt(calibration$endpoint_error_low[i]), fmt(calibration$endpoint_error_high[i]),
    fmt(calibration$achieved_oracle_error[i])
  ))
}

notes <- c(
  notes,
  "",
  "## Pilot Summary",
  "",
  "| scenario | method | valid reps | ARI | selected q | TPR | FPR | Precision all | F1 all | F1 valid | MSE_eta | common false | noise FPR | zero support |",
  "|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
)
for (i in seq_len(nrow(summary))) {
  notes <- c(notes, sprintf(
    "| %s | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %d |",
    summary$scenario[i], summary$method[i], summary$valid_reps[i],
    fmt(summary$ARI[i]), fmt(summary$selected_q[i], 2), fmt(summary$TPR[i]),
    fmt(summary$FPR[i]), fmt(summary$Precision_all_reps[i]),
    fmt(summary$F1_all_reps[i]), fmt(summary$F1[i]),
    fmt(summary$MSE_centered_eta[i]), fmt(summary$common_false_selection_rate[i]),
    fmt(summary$noise_false_selection_rate[i]), summary$zero_support_refit_reps[i]
  ))
}

notes <- c(
  notes,
  "",
  "## Notes",
  "",
  "- The generator preserves the specified kappa pattern and tunes the common-background norm.",
  paste0("- Eta refit rule: ", cfg$eta_refit_mode, "."),
  "- E-series main rows use information-criterion selection after exact centered-Eta support refitting.",
  "- Raw output is produced for audit but should not be committed."
)
writeLines(notes, notes_path)

cat("Wrote:\n")
cat("  ", normalizePath(raw_path, winslash = "/"), "\n", sep = "")
cat("  ", normalizePath(summary_path, winslash = "/"), "\n", sep = "")
cat("  ", normalizePath(calibration_path, winslash = "/"), "\n", sep = "")
cat("  ", normalizePath(notes_path, winslash = "/"), "\n", sep = "")

print(calibration)
print(summary[, c(
  "scenario", "method", "valid_reps", "ARI", "selected_q", "TPR", "FPR",
  "Precision_all_reps", "F1_all_reps", "F1", "MSE_centered_eta",
  "common_false_selection_rate", "noise_false_selection_rate",
  "zero_support_refit_reps"
)])
