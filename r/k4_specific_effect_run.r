# ==============================================================================
# K=4 specific-effect variable simulation
# ------------------------------------------------------------------------------
# Variable construction:
#   common variables: v_kj = 1.0 for all components
#   component-specific variables: v_kj = 0.5 only for component k
#   noise variables: v_kj = 0
#   mu_k = v_k / ||v_k||
#
# This pilot keeps the official tuning rule:
#   path-based candidates + BIC selection + optional support refit.
# ==============================================================================

source_until_before <- function(file, marker, back = 1L) {
  lines <- readLines(file, encoding = "UTF-8", warn = FALSE)
  idx <- tail(grep(marker, lines, fixed = TRUE), 1L)
  if (is.na(idx)) stop(sprintf("Marker not found in %s: %s", file, marker))
  keep <- seq_len(max(1L, idx - back))
  eval(parse(text = lines[keep]), envir = .GlobalEnv)
}

source_until_before(file.path("r", "k4_path_tuning_compare_run.r"), "rows <- list()", back = 1L)

cfg$run_label <- Sys.getenv("K4_SPECIFIC_LABEL", "k4_specific_effect")
cfg$n_rep <- as.integer(Sys.getenv("K4_SPECIFIC_N_REP", "3"))
cfg$n <- as.integer(Sys.getenv("K4_SPECIFIC_N", "600"))
cfg$d <- as.integer(Sys.getenv("K4_SPECIFIC_D", "100"))
cfg$K <- as.integer(Sys.getenv("K4_SPECIFIC_K", "4"))
cfg$nstart <- as.integer(Sys.getenv("K4_SPECIFIC_NSTART", "2"))
cfg$max_iter <- as.integer(Sys.getenv("K4_SPECIFIC_MAX_ITER", "80"))
cfg$max_path_steps <- as.integer(Sys.getenv("K4_SPECIFIC_ROSSI_STEPS", "100"))
cfg$sep_mu_path_steps <- as.integer(Sys.getenv("K4_SPECIFIC_SEP_MU_STEPS", "140"))
cfg$eta_path_steps <- as.integer(Sys.getenv("K4_SPECIFIC_ETA_STEPS", "80"))
cfg$base_seed <- as.integer(Sys.getenv("K4_SPECIFIC_BASE_SEED", "20260623"))
cfg$select_ic <- toupper(Sys.getenv("K4_SPECIFIC_SELECT_IC", "BIC"))
if (!cfg$select_ic %in% c("BIC", "EBIC")) {
  stop("K4_SPECIFIC_SELECT_IC must be either BIC or EBIC.")
}
cfg$out_dir <- Sys.getenv(
  "K4_SPECIFIC_OUT_DIR",
  "results/k4_specific_effect"
)

common_q <- as.integer(Sys.getenv("K4_SPECIFIC_COMMON_Q", "6"))
specific_q <- as.integer(Sys.getenv("K4_SPECIFIC_SPECIFIC_Q", "4"))
specific_weight <- as.numeric(Sys.getenv("K4_SPECIFIC_WEIGHT", "0.5"))
kappa_vec <- as.numeric(strsplit(
  Sys.getenv("K4_SPECIFIC_KAPPA", "30,45,65,90"),
  ",",
  fixed = TRUE
)[[1]])

dir.create(cfg$out_dir, recursive = TRUE, showWarnings = FALSE)

make_specific_effect_params <- function(d, K, common_q, specific_q,
                                        specific_weight, kappa) {
  if (length(kappa) != K) stop("length(kappa) must match K.")
  if (common_q + K * specific_q > d) {
    stop("common_q + K * specific_q must be <= d.")
  }

  mu_raw <- matrix(0, nrow = K, ncol = d)
  support <- matrix(FALSE, nrow = K, ncol = d)

  common_idx <- seq_len(common_q)
  mu_raw[, common_idx] <- 1.0
  support[, common_idx] <- TRUE

  start <- common_q + 1L
  specific_index <- vector("list", K)
  for (k in seq_len(K)) {
    idx <- start + ((k - 1L) * specific_q) + seq_len(specific_q) - 1L
    specific_index[[k]] <- idx
    mu_raw[k, idx] <- specific_weight
    support[k, idx] <- TRUE
  }

  list(
    alpha = rep(1 / K, K),
    mu = normalize_rows(mu_raw),
    kappa = kappa,
    support = support,
    common_idx = common_idx,
    specific_idx = unlist(specific_index, use.names = FALSE),
    noise_idx = seq.int(common_q + K * specific_q + 1L, d)
  )
}

selection_type_metrics <- function(active, params) {
  common_rate <- mean(active[params$common_idx])
  specific_rate <- mean(active[params$specific_idx])
  noise_rate <- mean(active[params$noise_idx])
  data.frame(
    common_selection_rate = common_rate,
    specific_selection_rate = specific_rate,
    noise_selection_rate = noise_rate
  )
}

append_type_metrics <- function(row, active, params) {
  cbind(row, selection_type_metrics(active, params))
}

method_name <- function(base, cfg, refit = FALSE) {
  paste0(base, " ", cfg$select_ic, if (refit) " + refit" else "")
}

best_ic_index <- function(tab, cfg) {
  which.min(tab[[cfg$select_ic]])
}

fit_rossi_specific_pair <- function(X, z, params, cfg) {
  path <- fit_svMF_path(
    X = X,
    K = cfg$K,
    labels_true = z,
    mu_true = params$mu,
    support_true = params$support,
    nstart = cfg$nstart,
    max_path_steps = cfg$max_path_steps,
    max_iter = cfg$max_iter,
    gamma = 0.5,
    verbose = FALSE
  )
  idx <- best_ic_index(path$path, cfg)
  fit <- path$fits[[idx]]
  prow <- path$path[idx, , drop = FALSE]
  active <- active_mu_coord(fit)
  support_entry <- abs(fit$mu) > 1e-8
  out <- eval_method(
    method_name("Rossi path", cfg), fit, X, z, params, active, support_entry,
    beta = prow$beta,
    ic = prow[, c("df", "BIC", "EBIC"), drop = FALSE]
  )
  refit <- fit_support_refit(X, cfg$K, active, fit, max_iter = cfg$max_iter)
  out_refit <- eval_method(
    method_name("Rossi path", cfg, refit = TRUE), refit, X, z, params, active, NULL,
    beta = prow$beta
  )
  rbind(
    append_type_metrics(out, active, params),
    append_type_metrics(out_refit, active, params)
  )
}

fit_separate_specific_pair <- function(X, z, params, cfg) {
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
    all_rows[[length(all_rows) + 1L]] <- path$rows
    for (i in seq_along(path$fits)) {
      all_fits[[length(all_fits) + 1L]] <- path$fits[[i]]
    }
  }
  if (length(all_rows) == 0) stop("Separate path produced no valid fit.")
  tab <- do.call(rbind, all_rows)
  best <- best_ic_index(tab, cfg)
  fit <- all_fits[[best]]
  active <- active_mu_coord(fit)
  out <- tab[best, , drop = FALSE]
  out$method <- method_name("Separate 2D path/grid", cfg)
  refit <- fit_support_refit(X, cfg$K, active, fit, max_iter = cfg$max_iter)
  out_refit <- eval_method(
    method_name("Separate 2D path/grid", cfg, refit = TRUE), refit, X, z, params, active, NULL,
    lambda_mu = out$lambda_mu,
    lambda_kappa = out$lambda_kappa
  )
  rbind(
    append_type_metrics(out, active, params),
    append_type_metrics(out_refit, active, params)
  )
}

fit_eta_specific_pair <- function(X, z, params, cfg) {
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
      method_name("Eta centered path", cfg), fit, X, z, params, active, NULL,
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

  tab <- do.call(rbind, rows)
  best <- best_ic_index(tab, cfg)
  fit <- fits[[best]]
  active <- active_eta_centered(fit)
  out <- tab[best, , drop = FALSE]
  refit <- fit_support_refit(X, cfg$K, active, fit, max_iter = cfg$max_iter)
  out_refit <- eval_method(
    method_name("Eta centered path", cfg, refit = TRUE), refit, X, z, params, active, NULL,
    lambda_eta = out$lambda_eta
  )
  rbind(
    append_type_metrics(out, active, params),
    append_type_metrics(out_refit, active, params)
  )
}

run_one_specific <- function(rep_id, cfg) {
  set.seed(cfg$base_seed + rep_id)
  params <- make_specific_effect_params(
    d = cfg$d, K = cfg$K, common_q = common_q,
    specific_q = specific_q, specific_weight = specific_weight,
    kappa = kappa_vec
  )
  dat <- simulate_from_params(cfg$n, params)
  pairwise_cos <- as.numeric(crossprod(t(params$mu))[upper.tri(diag(cfg$K))])
  cat(sprintf(
    "[specific_effect] rep %d/%d: union_q=%d, common=%d, specific_union=%d, mean cos=%.3f, kappa=(%s)\n",
    rep_id, cfg$n_rep, sum(colSums(params$support) > 0),
    common_q, cfg$K * specific_q, mean(pairwise_cos),
    paste(kappa_vec, collapse = ",")
  ))

  out <- rbind(
    fit_rossi_specific_pair(dat$X, dat$z, dat$params, cfg),
    fit_separate_specific_pair(dat$X, dat$z, dat$params, cfg),
    fit_eta_specific_pair(dat$X, dat$z, dat$params, cfg)
  )

  out$scenario <- "specific_effect_common6_specific4"
  out$rep <- rep_id
  out$n <- cfg$n
  out$d <- cfg$d
  out$K_true <- cfg$K
  out$common_q <- common_q
  out$specific_q_per_component <- specific_q
  out$specific_weight <- specific_weight
  out$true_union_q <- sum(colSums(params$support) > 0)
  out$true_entry_q <- sum(params$support)
  out$mu_pairwise_cos_mean <- mean(pairwise_cos)
  out$kappa_true_min <- min(kappa_vec)
  out$kappa_true_max <- max(kappa_vec)
  out$kappa_true_ratio <- max(kappa_vec) / min(kappa_vec)
  out$error <- NA_character_
  out
}

cat(sprintf(
  "Running K=4 specific-effect simulation: reps=%d, n=%d, d=%d, common=%d, specific/component=%d, weight=%.2f, nstart=%d, select=%s\n",
  cfg$n_rep, cfg$n, cfg$d, common_q, specific_q, specific_weight, cfg$nstart,
  cfg$select_ic
))

rows <- list()
for (rep_id in seq_len(cfg$n_rep)) {
  rows[[rep_id]] <- tryCatch(
    run_one_specific(rep_id, cfg),
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
        common_selection_rate = NA_real_,
        specific_selection_rate = NA_real_,
        noise_selection_rate = NA_real_,
        scenario = "specific_effect_common6_specific4",
        rep = rep_id, n = cfg$n, d = cfg$d, K_true = cfg$K,
        common_q = common_q,
        specific_q_per_component = specific_q,
        specific_weight = specific_weight,
        true_entry_q = NA_real_,
        mu_pairwise_cos_mean = NA_real_,
        kappa_true_min = min(kappa_vec),
        kappa_true_max = max(kappa_vec),
        kappa_true_ratio = max(kappa_vec) / min(kappa_vec),
        error = conditionMessage(e)
      )
    }
  )
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
  method_name("Rossi path", cfg), method_name("Rossi path", cfg, refit = TRUE),
  method_name("Separate 2D path/grid", cfg),
  method_name("Separate 2D path/grid", cfg, refit = TRUE),
  method_name("Eta centered path", cfg), method_name("Eta centered path", cfg, refit = TRUE),
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
  "TPR", "FPR", "Precision", "F1", "entry_TPR", "entry_FPR",
  "MSE_mu", "MSE_kappa", "MSE_centered_eta", "kappa_hat_mean",
  "BIC", "lambda_mu", "lambda_kappa", "lambda_eta",
  "common_q", "specific_q_per_component", "specific_weight",
  "true_entry_q", "mu_pairwise_cos_mean", "kappa_true_ratio"
)])
