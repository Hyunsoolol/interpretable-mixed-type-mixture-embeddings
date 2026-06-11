# ==============================================================================
# Eta path criterion comparison: BIC vs EBIC
# ------------------------------------------------------------------------------
# This script reuses the K=4 fair path-tuning implementation and evaluates the
# same eta lambda path under two selection criteria:
#   1. minimum BIC
#   2. minimum EBIC
#
# The path is generated once per replicate. BIC and EBIC then choose among the
# same candidate fits, so any difference is due to the criterion, not a different
# optimizer path.
# ==============================================================================

source_until_before <- function(file, marker, back = 1L) {
  lines <- readLines(file, encoding = "UTF-8", warn = FALSE)
  idx <- tail(grep(marker, lines, fixed = TRUE), 1L)
  if (is.na(idx)) stop(sprintf("Marker not found in %s: %s", file, marker))
  keep <- seq_len(max(1L, idx - back))
  eval(parse(text = lines[keep]), envir = .GlobalEnv)
}

source_until_before(file.path("r", "k4_path_tuning_compare_run.r"), "rows <- list()", back = 1L)

parse_chr_grid <- function(x) {
  trimws(strsplit(x, ",", fixed = TRUE)[[1]])
}

cfg$run_label <- Sys.getenv("ETA_EBIC_LABEL", "eta_ebic_compare_260622")
cfg$n_rep <- as.integer(Sys.getenv("ETA_EBIC_N_REP", "20"))
cfg$n <- as.integer(Sys.getenv("ETA_EBIC_N", "1000"))
cfg$d <- as.integer(Sys.getenv("ETA_EBIC_D", "100"))
cfg$K <- as.integer(Sys.getenv("ETA_EBIC_K", "4"))
cfg$nstart <- as.integer(Sys.getenv("ETA_EBIC_NSTART", "5"))
cfg$max_iter <- as.integer(Sys.getenv("ETA_EBIC_MAX_ITER", "100"))
cfg$eta_path_steps <- as.integer(Sys.getenv("ETA_EBIC_ETA_STEPS", "120"))
cfg$min_rel_lambda <- as.numeric(Sys.getenv("ETA_EBIC_MIN_REL_LAMBDA", "1e-3"))
cfg$base_seed <- as.integer(Sys.getenv("ETA_EBIC_BASE_SEED", "20260622"))
cfg$out_dir <- Sys.getenv("ETA_EBIC_OUT_DIR", "results/eta_ebic_compare_260622")

scenario_filter <- parse_chr_grid(Sys.getenv(
  "ETA_EBIC_SCENARIOS",
  "realistic_concdom"
))
if (length(scenario_filter) == 1L && identical(scenario_filter, "all")) {
  scenario_filter <- scenario_table$scenario
}

dir.create(cfg$out_dir, recursive = TRUE, showWarnings = FALSE)

fit_eta_centered_path_bic_ebic_pair <- function(X, z, params, cfg) {
  dense <- fit_svMF_multistart(
    X, cfg$K, beta = 0, nstart = cfg$nstart, max_iter = cfg$max_iter
  )
  lambda_eta <- 0
  fit <- fit_eta_centered_em(
    X, cfg$K, lambda_eta = lambda_eta, init = dense, max_iter = cfg$max_iter
  )
  rows <- list()
  fits <- list(fit)

  add_row <- function(fit, lambda_eta) {
    active <- active_eta_centered(fit)
    ic <- eta_centered_ic(fit, nrow(X), ncol(X), fit$loglik)
    eval_method(
      "Eta path candidate", fit, X, z, params, active, NULL,
      lambda_eta = lambda_eta,
      ic = ic
    )
  }

  rows[[1L]] <- add_row(fit, lambda_eta)

  for (step in 2:cfg$eta_path_steps) {
    e <- e_step_vmf(X, fit)
    mstep <- unpenalized_eta_mstep(X, e$tau)
    thresholds <- sqrt(colSums(center_eta(mstep$eta)^2))
    candidates <- thresholds[thresholds > lambda_eta + 1e-10]
    if (length(candidates) == 0) break
    lambda_next <- min(candidates)
    if (lambda_eta > 0) {
      lambda_next <- max(lambda_next, lambda_eta * (1 + cfg$min_rel_lambda))
    }
    if (!is.finite(lambda_next) || lambda_next <= lambda_eta) break

    fit_next <- tryCatch(
      fit_eta_centered_em(
        X, cfg$K, lambda_eta = lambda_next, init = fit,
        max_iter = cfg$max_iter
      ),
      error = function(e) NULL
    )
    if (is.null(fit_next) || isTRUE(fit_next$failed)) break

    fit <- fit_next
    lambda_eta <- lambda_next
    fits[[length(fits) + 1L]] <- fit
    rows[[length(rows) + 1L]] <- add_row(fit, lambda_eta)
    if (sum(active_eta_centered(fit)) <= 1) break
  }

  path <- do.call(rbind, rows)

  select_one <- function(criterion) {
    best <- which.min(path[[criterion]])
    fit_best <- fits[[best]]
    active <- active_eta_centered(fit_best)
    out <- path[best, , drop = FALSE]
    out$method <- sprintf("Eta path %s", criterion)

    refit <- fit_support_refit(X, cfg$K, active, fit_best, max_iter = cfg$max_iter)
    out_refit <- eval_method(
      sprintf("Eta path %s + refit", criterion),
      refit, X, z, params, active, NULL,
      lambda_eta = out$lambda_eta
    )
    rbind(out, out_refit)
  }

  rbind(select_one("BIC"), select_one("EBIC"))
}

run_one_eta_criteria <- function(scenario, rep_id, cfg) {
  set.seed(cfg$base_seed + rep_id + 1000L * match(scenario, scenario_table$scenario))
  dat <- simulate_scenario(scenario, cfg)
  cat(sprintf("[%s] rep %d/%d: true union q=%d\n",
              scenario, rep_id, cfg$n_rep,
              sum(colSums(dat$params$support) > 0)))
  out <- fit_eta_centered_path_bic_ebic_pair(dat$X, dat$z, dat$params, cfg)
  out$scenario <- scenario
  out$rep <- rep_id
  out$n <- cfg$n
  out$d <- cfg$d
  out$K_true <- cfg$K
  out
}

cat(sprintf(
  "Running eta BIC/EBIC comparison: scenarios=%s, reps=%d, n=%d, d=%d, K=%d, nstart=%d\n",
  paste(scenario_filter, collapse = ","),
  cfg$n_rep, cfg$n, cfg$d, cfg$K, cfg$nstart
))

rows <- list()
idx <- 1L
for (s in scenario_filter) {
  for (rep_id in seq_len(cfg$n_rep)) {
    rows[[idx]] <- tryCatch(
      run_one_eta_criteria(s, rep_id, cfg),
      error = function(e) {
        data.frame(
          method = "ERROR", K_fit = NA_real_, beta = NA_real_,
          lambda_mu = NA_real_, lambda_kappa = NA_real_,
          lambda_eta = NA_real_, ARI = NA_real_, loglik = NA_real_,
          pen_loglik = NA_real_, converged = NA, iter = NA_real_,
          true_union_q = NA_real_, selected_q = NA_real_,
          TPR = NA_real_, FPR = NA_real_, Precision = NA_real_,
          F1 = NA_real_, entry_TPR = NA_real_, entry_FPR = NA_real_,
          entry_Precision = NA_real_, entry_F1 = NA_real_,
          MSE_mu = NA_real_, MSE_kappa = NA_real_,
          MSE_centered_eta = NA_real_, kappa_hat_mean = NA_real_,
          df = NA_real_, BIC = NA_real_, EBIC = NA_real_,
          scenario = s, rep = rep_id, n = cfg$n, d = cfg$d,
          K_true = cfg$K, error = conditionMessage(e)
        )
      }
    )
    idx <- idx + 1L
  }
}

raw <- do.call(rbind, rows)
raw_path <- file.path(cfg$out_dir, sprintf("%s_raw.csv", cfg$run_label))
write.csv(raw, raw_path, row.names = FALSE)

num_cols <- names(raw)[vapply(raw, is.numeric, logical(1))]
groups <- unique(raw[, c("scenario", "method")])
summary <- do.call(rbind, lapply(seq_len(nrow(groups)), function(i) {
  sub <- raw[raw$scenario == groups$scenario[i] & raw$method == groups$method[i], ]
  means <- aggregate(
    sub[, num_cols, drop = FALSE],
    list(dummy = rep(1, nrow(sub))),
    mean,
    na.rm = TRUE
  )
  means$dummy <- NULL
  data.frame(
    scenario = groups$scenario[i],
    method = groups$method[i],
    reps = length(unique(sub$rep)),
    means,
    row.names = NULL
  )
}))

method_order <- c(
  "Eta path BIC", "Eta path BIC + refit",
  "Eta path EBIC", "Eta path EBIC + refit",
  "ERROR"
)
summary$method <- factor(summary$method, levels = method_order)
summary <- summary[order(summary$scenario, summary$method), ]
summary$method <- as.character(summary$method)

summary_path <- file.path(cfg$out_dir, sprintf("%s_summary.csv", cfg$run_label))
write.csv(summary, summary_path, row.names = FALSE)

cat("\nWrote:\n")
cat("  ", normalizePath(raw_path, winslash = "/"), "\n", sep = "")
cat("  ", normalizePath(summary_path, winslash = "/"), "\n", sep = "")
print(summary[, c(
  "scenario", "method", "reps", "ARI", "true_union_q", "selected_q",
  "TPR", "FPR", "Precision", "F1", "MSE_mu", "MSE_kappa",
  "MSE_centered_eta", "kappa_hat_mean", "BIC", "EBIC", "lambda_eta"
)])
