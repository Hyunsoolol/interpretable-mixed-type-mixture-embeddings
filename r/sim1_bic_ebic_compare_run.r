# ==============================================================================
# Simulation 1 tuning criterion comparison: BIC vs EBIC
# ------------------------------------------------------------------------------
# Reuses the K=2 fair path-tuning implementation and compares BIC/EBIC choices
# for Rossi, separate mu/kappa penalty, and eta-contrast penalty.
# ==============================================================================

source_until_before <- function(file, marker, back = 1L) {
  lines <- readLines(file, encoding = "UTF-8", warn = FALSE)
  idx <- tail(grep(marker, lines, fixed = TRUE), 1L)
  if (is.na(idx)) stop(sprintf("Marker not found in %s: %s", file, marker))
  keep <- seq_len(max(1L, idx - back))
  eval(parse(text = lines[keep]), envir = .GlobalEnv)
}

source_until_before(file.path("r", "eta_path_tuning_compare_run.r"), "raw_rows <- list()", back = 1L)

parse_chr_grid <- function(x) {
  trimws(strsplit(x, ",", fixed = TRUE)[[1]])
}

cfg$run_label <- Sys.getenv("SIM1_CRIT_LABEL", "sim1_bic_ebic_compare_260622")
cfg$n_rep <- as.integer(Sys.getenv("SIM1_CRIT_N_REP", "20"))
cfg$n <- as.integer(Sys.getenv("SIM1_CRIT_N", "1000"))
cfg$d <- as.integer(Sys.getenv("SIM1_CRIT_D", "100"))
cfg$q <- as.integer(Sys.getenv("SIM1_CRIT_Q", "10"))
cfg$nstart <- as.integer(Sys.getenv("SIM1_CRIT_NSTART", "5"))
cfg$max_iter <- as.integer(Sys.getenv("SIM1_CRIT_MAX_ITER", "120"))
cfg$rossi_max_path_steps <- as.integer(Sys.getenv("SIM1_CRIT_ROSSI_STEPS", "220"))
cfg$eta_max_path_steps <- as.integer(Sys.getenv("SIM1_CRIT_ETA_STEPS", "120"))
cfg$sep_mu_path_steps <- as.integer(Sys.getenv("SIM1_CRIT_SEP_MU_STEPS", "220"))
cfg$base_seed <- as.integer(Sys.getenv("SIM1_CRIT_BASE_SEED", "20260622"))
cfg$out_dir <- Sys.getenv("SIM1_CRIT_OUT_DIR", "results/sim1_bic_ebic_compare_260622")

scenario_filter <- parse_chr_grid(Sys.getenv("SIM1_CRIT_SCENARIOS", "all"))
if (!(length(scenario_filter) == 1L && identical(scenario_filter, "all"))) {
  scenarios <- scenarios[scenarios$scenario %in% scenario_filter, , drop = FALSE]
}

dir.create(cfg$out_dir, recursive = TRUE, showWarnings = FALSE)

rename_selected <- function(out, method_prefix, criterion) {
  out$method <- c(
    sprintf("%s %s", method_prefix, criterion),
    sprintf("%s %s + refit", method_prefix, criterion)
  )
  out
}

fit_rossi_criteria_pair <- function(X, z, params, cfg) {
  path <- fit_svMF_path(
    X = X,
    K = cfg$K,
    labels_true = z,
    mu_true = params$mu,
    support_true = params$support,
    nstart = cfg$nstart,
    max_path_steps = cfg$rossi_max_path_steps,
    max_iter = cfg$max_iter,
    gamma = 0.5,
    verbose = FALSE
  )

  select_one <- function(criterion) {
    idx <- which.min(path$path[[criterion]])
    fit <- path$fits[[idx]]
    beta <- path$path$beta[idx]
    active <- colSums(abs(fit$mu) > 1e-8) > 0
    out <- evaluate_row(
      "tmp", fit, X, z, params$support[1, ],
      params, cfg$q, beta = beta, active = active
    )
    refit <- fit_support_constrained_vmf(
      X, cfg$K, active, fit, max_iter = cfg$max_iter
    )
    out_refit <- evaluate_row(
      "tmp refit", refit, X, z, params$support[1, ],
      params, cfg$q, beta = beta, active = active,
      use_support_ic = TRUE
    )
    rename_selected(rbind(out, out_refit), "Rossi path", criterion)
  }

  rbind(select_one("BIC"), select_one("EBIC"))
}

fit_separate_criteria_pair <- function(X, z, params, cfg) {
  dense <- fit_svMF_multistart(
    X, cfg$K, beta = 0, nstart = cfg$nstart, max_iter = cfg$max_iter
  )
  lambda_kappa_grid <- separate_lambda_kappa_grid(X, dense, cfg$sep_kappa_fracs)
  all_rows <- list()
  all_fits <- list()

  for (lambda_kappa in lambda_kappa_grid) {
    path <- fit_separate_path_for_kappa(
      X, z, params, cfg, lambda_kappa = lambda_kappa, dense_fit = dense
    )
    if (is.null(path$rows)) next
    for (i in seq_along(path$fits)) {
      all_fits[[length(all_fits) + 1L]] <- path$fits[[i]]
    }
    all_rows[[length(all_rows) + 1L]] <- path$rows
  }
  if (length(all_rows) == 0) stop("Separate path produced no valid fit.")
  grid_path <- do.call(rbind, all_rows)

  select_one <- function(criterion) {
    idx <- which.min(grid_path[[criterion]])
    fit <- all_fits[[idx]]
    active <- separate_penalty_active(fit)
    out <- grid_path[idx, , drop = FALSE]
    out$method <- sprintf("Separate path/grid %s", criterion)
    refit <- fit_support_constrained_vmf(
      X, cfg$K, active, fit, max_iter = cfg$max_iter
    )
    out_refit <- evaluate_row(
      sprintf("Separate path/grid %s + refit", criterion),
      refit, X, z, params$support[1, ], params, cfg$q,
      lambda_mu = out$lambda_mu,
      lambda_kappa = out$lambda_kappa,
      active = active,
      use_support_ic = TRUE
    )
    rbind(out, out_refit)
  }

  rbind(select_one("BIC"), select_one("EBIC"))
}

fit_eta_criteria_pair <- function(X, z, params, cfg) {
  dense <- fit_svMF_multistart(
    X, cfg$K, beta = 0, nstart = cfg$nstart, max_iter = cfg$max_iter
  )
  lambda <- 0
  fit <- fit_eta_penalty_em(
    X, lambda_eta = lambda, init = dense, max_iter = cfg$max_iter
  )
  fits <- list(fit)
  rows <- list(evaluate_row(
    "Eta path candidate", fit, X, z, params$support[1, ],
    params, cfg$q, lambda_eta = lambda
  ))

  for (step in 2:cfg$eta_max_path_steps) {
    e <- e_step_vmf(X, fit)
    mstep <- unpenalized_eta_mstep(X, e$tau)
    delta_abs <- abs(mstep$eta[2, ] - mstep$eta[1, ])
    candidates <- delta_abs[delta_abs > lambda + 1e-10]
    if (length(candidates) == 0) break
    lambda_next <- min(candidates)
    if (lambda > 0) {
      lambda_next <- max(lambda_next, lambda * (1 + cfg$min_rel_lambda))
    }
    if (!is.finite(lambda_next) || lambda_next <= lambda) break
    fit_next <- tryCatch(
      fit_eta_penalty_em(
        X, lambda_eta = lambda_next, init = fit, max_iter = cfg$max_iter
      ),
      error = function(e) NULL
    )
    if (is.null(fit_next) || isTRUE(fit_next$failed)) break
    fit <- fit_next
    lambda <- lambda_next
    fits[[length(fits) + 1L]] <- fit
    rows[[length(rows) + 1L]] <- evaluate_row(
      "Eta path candidate", fit, X, z, params$support[1, ],
      params, cfg$q, lambda_eta = lambda
    )
    if (sum(eta_contrast_active(fit)) <= 1) break
  }

  path <- do.call(rbind, rows)

  select_one <- function(criterion) {
    idx <- which.min(path[[criterion]])
    fit_best <- fits[[idx]]
    lambda_best <- path$lambda_eta[idx]
    active <- eta_contrast_active(fit_best)
    out <- path[idx, , drop = FALSE]
    out$method <- sprintf("Eta path %s", criterion)
    refit <- fit_support_constrained_vmf(
      X, cfg$K, active, fit_best, max_iter = cfg$max_iter
    )
    out_refit <- evaluate_row(
      sprintf("Eta path %s + refit", criterion),
      refit, X, z, params$support[1, ], params, cfg$q,
      lambda_eta = lambda_best,
      active = active,
      use_support_ic = TRUE
    )
    rbind(out, out_refit)
  }

  rbind(select_one("BIC"), select_one("EBIC"))
}

run_one_criteria <- function(scenario_row, rep_id, cfg) {
  set.seed(cfg$base_seed + rep_id + 1000L * match(scenario_row$scenario, scenarios$scenario))
  dat <- simulate_kappa_contrast_data(
    n = cfg$n, d = cfg$d, q = cfg$q,
    kappa_low = scenario_row$kappa_low,
    kappa_high = scenario_row$kappa_high,
    mu_cos = scenario_row$mu_cos
  )
  cat(sprintf("[%s] rep %d/%d\n", scenario_row$scenario, rep_id, cfg$n_rep))
  rows <- rbind(
    fit_rossi_criteria_pair(dat$X, dat$z, dat$params, cfg),
    fit_separate_criteria_pair(dat$X, dat$z, dat$params, cfg),
    fit_eta_criteria_pair(dat$X, dat$z, dat$params, cfg)
  )
  rows$scenario <- scenario_row$scenario
  rows$rep <- rep_id
  rows$n <- cfg$n
  rows$d <- cfg$d
  rows$q_true <- cfg$q
  rows$mu_cos_true <- scenario_row$mu_cos
  rows$kappa_low_true <- scenario_row$kappa_low
  rows$kappa_high_true <- scenario_row$kappa_high
  rows$true_eta_contrast_norm <- l2_norm(
    dat$params$kappa[2] * dat$params$mu[2, ] -
      dat$params$kappa[1] * dat$params$mu[1, ]
  )
  rows$error <- NA_character_
  rows
}

cat(sprintf(
  "Running simulation 1 BIC/EBIC comparison: scenarios=%s, reps=%d, n=%d, d=%d, q=%d, nstart=%d\n",
  paste(scenarios$scenario, collapse = ","),
  cfg$n_rep, cfg$n, cfg$d, cfg$q, cfg$nstart
))

raw_path <- file.path(cfg$out_dir, sprintf("%s_raw.csv", cfg$run_label))
summary_path <- file.path(cfg$out_dir, sprintf("%s_summary.csv", cfg$run_label))
if (file.exists(raw_path)) file.remove(raw_path)
if (file.exists(summary_path)) file.remove(summary_path)

append_raw <- function(chunk, path) {
  write.table(
    chunk,
    file = path,
    sep = ",",
    row.names = FALSE,
    col.names = !file.exists(path),
    append = file.exists(path)
  )
}

raw_rows <- list()
idx <- 1L
for (s in seq_len(nrow(scenarios))) {
  scenario_row <- scenarios[s, ]
  for (rep_id in seq_len(cfg$n_rep)) {
    raw_rows[[idx]] <- tryCatch(
      run_one_criteria(scenario_row, rep_id, cfg),
      error = function(e) {
        data.frame(
          method = "ERROR", beta = NA_real_, lambda_eta = NA_real_,
          lambda_mu = NA_real_, lambda_kappa = NA_real_,
          ARI = NA_real_, loglik = NA_real_,
          converged = NA, iter = NA_real_, nnz_fraction = NA_real_,
          selected_q = NA_real_, TPR = NA_real_, FPR = NA_real_,
          Precision = NA_real_, F1 = NA_real_,
          mu_contrast_norm = NA_real_, eta_contrast_norm = NA_real_,
          kappa_ratio_hat = NA_real_, mu_topq_recall = NA_real_,
          eta_topq_recall = NA_real_, MSE_mu = NA_real_,
          MSE_kappa = NA_real_, MSE_eta_contrast = NA_real_,
          df = NA_real_, BIC = NA_real_, EBIC = NA_real_,
          scenario = scenario_row$scenario, rep = rep_id,
          n = cfg$n, d = cfg$d, q_true = cfg$q,
          mu_cos_true = scenario_row$mu_cos,
          kappa_low_true = scenario_row$kappa_low,
          kappa_high_true = scenario_row$kappa_high,
          true_eta_contrast_norm = NA_real_,
          error = conditionMessage(e)
        )
      }
    )
    append_raw(raw_rows[[idx]], raw_path)
    idx <- idx + 1L
  }
}

raw <- read.csv(raw_path, stringsAsFactors = FALSE)

metric_cols <- c(
  "ARI", "selected_q", "TPR", "FPR", "Precision", "F1",
  "eta_contrast_norm", "kappa_ratio_hat",
  "MSE_mu", "MSE_kappa", "MSE_eta_contrast",
  "BIC", "EBIC", "beta", "lambda_eta", "lambda_mu", "lambda_kappa"
)
safe_mean <- function(x) if (all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE)
safe_se <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) <= 1) return(NA_real_)
  stats::sd(x) / sqrt(length(x))
}

groups <- unique(raw[, c("scenario", "method")])
summary <- do.call(rbind, lapply(seq_len(nrow(groups)), function(i) {
  sub <- raw[raw$scenario == groups$scenario[i] & raw$method == groups$method[i], ]
  out <- data.frame(
    scenario = groups$scenario[i],
    method = groups$method[i],
    reps = length(unique(sub$rep))
  )
  for (m in metric_cols) {
    out[[paste0(m, "_mean")]] <- safe_mean(sub[[m]])
    out[[paste0(m, "_se")]] <- safe_se(sub[[m]])
  }
  out
}))

method_order <- c(
  "Rossi path BIC", "Rossi path BIC + refit",
  "Rossi path EBIC", "Rossi path EBIC + refit",
  "Separate path/grid BIC", "Separate path/grid BIC + refit",
  "Separate path/grid EBIC", "Separate path/grid EBIC + refit",
  "Eta path BIC", "Eta path BIC + refit",
  "Eta path EBIC", "Eta path EBIC + refit",
  "ERROR"
)
summary$method <- factor(summary$method, levels = method_order)
summary <- summary[order(summary$scenario, summary$method), ]
summary$method <- as.character(summary$method)

write.csv(summary, summary_path, row.names = FALSE)

cat("\nWrote:\n")
cat("  ", normalizePath(raw_path, winslash = "/"), "\n", sep = "")
cat("  ", normalizePath(summary_path, winslash = "/"), "\n", sep = "")
print(summary[, c(
  "scenario", "method", "reps", "ARI_mean", "selected_q_mean",
  "TPR_mean", "FPR_mean", "Precision_mean", "F1_mean",
  "MSE_mu_mean", "MSE_kappa_mean", "MSE_eta_contrast_mean",
  "eta_contrast_norm_mean", "kappa_ratio_hat_mean"
)])
