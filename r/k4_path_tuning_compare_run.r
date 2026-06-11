# ==============================================================================
# K=4 fair path-tuning comparison
# ------------------------------------------------------------------------------
# Scenarios:
#   1. Rossi 2022-like sparse active-variable setting
#   2. K=4 stress setting: identical mean direction, concentration contrast only
#   3. K=4 realistic concentration-dominant setting
#
# Methods:
#   1. Rossi sparse vMF, beta path + BIC
#   2. Rossi sparse vMF + support refit
#   3. Separate mu/kappa penalty EM, lambda_kappa grid and lambda_mu path + BIC
#   4. Separate mu/kappa penalty EM + support refit
#   5. Centered eta penalty EM, lambda_eta path + BIC
#   6. Centered eta penalty EM + support refit
# ==============================================================================

source_until_before <- function(file, marker, back = 2L) {
  lines <- readLines(file, encoding = "UTF-8", warn = FALSE)
  idx <- grep(marker, lines, fixed = TRUE)[1]
  if (is.na(idx)) stop(sprintf("Marker not found in %s: %s", file, marker))
  keep <- seq_len(max(1L, idx - back))
  eval(parse(text = lines[keep]), envir = .GlobalEnv)
}

source_until_before(file.path("r", "rb2022_k4_pilot_compare_run.r"), "Running K=4 pilot", back = 2L)

parse_num_grid <- function(x) as.numeric(strsplit(x, ",", fixed = TRUE)[[1]])

cfg <- list(
  run_label = Sys.getenv("K4_PATH_LABEL", "k4_path_tuning_compare_260622"),
  n_rep = as.integer(Sys.getenv("K4_PATH_N_REP", "20")),
  n = as.integer(Sys.getenv("K4_PATH_N", "1000")),
  d = as.integer(Sys.getenv("K4_PATH_D", "100")),
  K = as.integer(Sys.getenv("K4_PATH_K", "4")),
  nstart = as.integer(Sys.getenv("K4_PATH_NSTART", "5")),
  max_path_steps = as.integer(Sys.getenv("K4_PATH_ROSSI_STEPS", "220")),
  max_iter = as.integer(Sys.getenv("K4_PATH_MAX_ITER", "100")),
  sep_mu_path_steps = as.integer(Sys.getenv("K4_PATH_SEP_MU_STEPS", "300")),
  sep_kappa_fracs = parse_num_grid(Sys.getenv(
    "K4_PATH_SEP_KAPPA_FRACS",
    "0,0.05,0.1,0.2"
  )),
  eta_path_steps = as.integer(Sys.getenv("K4_PATH_ETA_STEPS", "120")),
  min_rel_lambda = as.numeric(Sys.getenv("K4_PATH_MIN_REL_LAMBDA", "1e-3")),
  path_patience = as.integer(Sys.getenv("K4_PATH_PATIENCE", "60")),
  base_seed = as.integer(Sys.getenv("K4_PATH_BASE_SEED", "20260622")),
  out_dir = Sys.getenv("K4_PATH_OUT_DIR", "results/k4_path_tuning_compare_260622")
)

dir.create(cfg$out_dir, recursive = TRUE, showWarnings = FALSE)

scenario_table <- data.frame(
  scenario = c("paperlike_sparse_active", "same_mu_kappa_stress", "realistic_concdom"),
  stringsAsFactors = FALSE
)

simulate_from_params <- function(n, params) {
  K <- nrow(params$mu)
  z <- sample.int(K, size = n, replace = TRUE, prob = params$alpha)
  X <- matrix(0, nrow = n, ncol = ncol(params$mu))
  for (k in seq_len(K)) {
    idx <- which(z == k)
    if (length(idx) > 0) {
      X[idx, ] <- rvMF(length(idx), params$mu[k, ], params$kappa[k])
    }
  }
  list(X = X, z = z, params = params)
}

make_same_mu_kappa_params <- function(d, q, kappa) {
  K <- length(kappa)
  support <- matrix(FALSE, nrow = K, ncol = d)
  support[, seq_len(q)] <- TRUE
  mu <- matrix(0, nrow = K, ncol = d)
  mu[, seq_len(q)] <- 1 / sqrt(q)
  list(
    alpha = rep(1 / K, K),
    mu = mu,
    kappa = kappa,
    support = support
  )
}

make_realistic_concdom_params <- function(d, K, shared_q = 7,
                                          specific_q = 3,
                                          shared_energy = 0.90,
                                          kappa = c(25, 40, 65, 100)) {
  support <- matrix(FALSE, nrow = K, ncol = d)
  mu <- matrix(0, nrow = K, ncol = d)
  shared <- seq_len(shared_q)
  mu[, shared] <- sqrt(shared_energy / shared_q)
  support[, shared] <- TRUE

  start <- shared_q + 1L
  for (k in seq_len(K)) {
    spec <- start + ((k - 1L) * specific_q) + seq_len(specific_q) - 1L
    mu[k, spec] <- sqrt((1 - shared_energy) / specific_q)
    support[k, spec] <- TRUE
  }

  list(
    alpha = rep(1 / K, K),
    mu = normalize_rows(mu),
    kappa = kappa,
    support = support
  )
}

simulate_scenario <- function(scenario, cfg) {
  if (identical(scenario, "paperlike_sparse_active")) {
    return(simulate_rb2022_data(
      n = cfg$n, K = cfg$K, d = cfg$d,
      overlap = 0.05, nonzero_fraction = 0.10
    ))
  }
  if (identical(scenario, "same_mu_kappa_stress")) {
    params <- make_same_mu_kappa_params(
      d = cfg$d, q = 10, kappa = c(20, 35, 60, 100)
    )
    return(simulate_from_params(cfg$n, params))
  }
  if (identical(scenario, "realistic_concdom")) {
    params <- make_realistic_concdom_params(
      d = cfg$d, K = cfg$K,
      shared_q = 7, specific_q = 3, shared_energy = 0.90,
      kappa = c(25, 40, 65, 100)
    )
    return(simulate_from_params(cfg$n, params))
  }
  stop(sprintf("Unknown scenario: %s", scenario))
}

method_name_map <- c(
  "Rossi" = "Rossi path BIC",
  "Rossi + refit" = "Rossi path BIC + refit"
)

fit_rossi_path_pair <- function(X, z, params, cfg) {
  out <- fit_rossi_pair(X, z, params, cfg)
  out$method <- ifelse(
    out$method %in% names(method_name_map),
    unname(method_name_map[out$method]),
    out$method
  )
  out
}

separate_lambda_kappa_grid <- function(X, dense_fit, fracs) {
  e <- e_step_vmf(X, dense_fit)
  r <- t(e$tau) %*% X
  s <- numeric(nrow(dense_fit$mu))
  for (k in seq_len(nrow(dense_fit$mu))) {
    s[k] <- as.numeric(crossprod(dense_fit$mu[k, ], r[k, ]))
  }
  max_feasible <- max(0, min(s, na.rm = TRUE))
  sort(unique(pmax(0, fracs * max_feasible)))
}

fit_separate_path_for_kappa <- function(X, z, params, cfg, lambda_kappa,
                                        dense_fit) {
  lambda_mu <- 0
  fit <- fit_separate_penalty_em(
    X, cfg$K, lambda_mu = lambda_mu, lambda_kappa = lambda_kappa,
    init = dense_fit, max_iter = cfg$max_iter
  )
  if (isTRUE(fit$failed)) return(list(rows = NULL, fits = list()))

  fits <- list(fit)
  rows <- list()
  best_bic <- Inf
  no_improve <- 0L

  add_row <- function(fit, lambda_mu, lambda_kappa) {
    active <- active_mu_coord(fit)
    ic <- separate_model_ic(fit, nrow(X), ncol(X), fit$loglik)
    eval_method(
      "Separate 2D path/grid BIC", fit, X, z, params, active,
      abs(fit$mu) > 1e-8,
      lambda_mu = lambda_mu,
      lambda_kappa = lambda_kappa,
      ic = ic
    )
  }

  rows[[1L]] <- add_row(fit, lambda_mu, lambda_kappa)
  best_bic <- rows[[1L]]$BIC

  for (step in 2:cfg$sep_mu_path_steps) {
    e <- e_step_vmf(X, fit)
    r <- t(e$tau) %*% X
    thresholds <- as.vector(abs(sweep(r, 1, fit$kappa, "*")))
    candidates <- thresholds[thresholds > lambda_mu + 1e-10]
    if (length(candidates) == 0) break
    lambda_next <- min(candidates)
    if (lambda_mu > 0) {
      lambda_next <- max(lambda_next, lambda_mu * (1 + cfg$min_rel_lambda))
    }
    if (!is.finite(lambda_next) || lambda_next <= lambda_mu) break

    fit_next <- tryCatch(
      fit_separate_penalty_em(
        X, cfg$K, lambda_mu = lambda_next, lambda_kappa = lambda_kappa,
        init = fit, max_iter = cfg$max_iter
      ),
      error = function(e) NULL
    )
    if (is.null(fit_next) || isTRUE(fit_next$failed)) break

    fit <- fit_next
    lambda_mu <- lambda_next
    fits[[length(fits) + 1L]] <- fit
    row <- add_row(fit, lambda_mu, lambda_kappa)
    rows[[length(rows) + 1L]] <- row

    if (is.finite(row$BIC) && row$BIC < best_bic - 1e-7) {
      best_bic <- row$BIC
      no_improve <- 0L
    } else {
      no_improve <- no_improve + 1L
    }

    if (sum(active_mu_coord(fit)) <= 1) break
    if (no_improve >= cfg$path_patience) break
  }

  list(rows = do.call(rbind, rows), fits = fits)
}

fit_separate_path_grid_pair <- function(X, z, params, cfg) {
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
  best <- which.min(tab$BIC)
  fit <- all_fits[[best]]
  active <- active_mu_coord(fit)
  out <- tab[best, , drop = FALSE]
  refit <- fit_support_refit(X, cfg$K, active, fit, max_iter = cfg$max_iter)
  out_refit <- eval_method(
    "Separate 2D path/grid BIC + refit", refit, X, z, params, active, NULL,
    lambda_mu = out$lambda_mu,
    lambda_kappa = out$lambda_kappa
  )
  rbind(out, out_refit)
}

fit_eta_centered_path_pair <- function(X, z, params, cfg) {
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
      "Eta centered path BIC", fit, X, z, params, active, NULL,
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
  best <- which.min(tab$BIC)
  fit <- fits[[best]]
  active <- active_eta_centered(fit)
  out <- tab[best, , drop = FALSE]
  refit <- fit_support_refit(X, cfg$K, active, fit, max_iter = cfg$max_iter)
  out_refit <- eval_method(
    "Eta centered path BIC + refit", refit, X, z, params, active, NULL,
    lambda_eta = out$lambda_eta
  )
  rbind(out, out_refit)
}

run_one <- function(scenario, rep_id, cfg) {
  set.seed(cfg$base_seed + rep_id + 1000L * match(scenario, scenario_table$scenario))
  dat <- simulate_scenario(scenario, cfg)
  cat(sprintf("[%s] rep %d/%d: true union q=%d\n",
              scenario, rep_id, cfg$n_rep,
              sum(colSums(dat$params$support) > 0)))
  out <- rbind(
    fit_rossi_path_pair(dat$X, dat$z, dat$params, cfg),
    fit_separate_path_grid_pair(dat$X, dat$z, dat$params, cfg),
    fit_eta_centered_path_pair(dat$X, dat$z, dat$params, cfg)
  )
  out$scenario <- scenario
  out$rep <- rep_id
  out$n <- cfg$n
  out$d <- cfg$d
  out$K_true <- cfg$K
  out
}

rows <- list()
idx <- 1L
for (s in scenario_table$scenario) {
  for (rep_id in seq_len(cfg$n_rep)) {
    rows[[idx]] <- tryCatch(
      run_one(s, rep_id, cfg),
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
  "Rossi path BIC", "Rossi path BIC + refit",
  "Separate 2D path/grid BIC", "Separate 2D path/grid BIC + refit",
  "Eta centered path BIC", "Eta centered path BIC + refit",
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
  "MSE_centered_eta", "kappa_hat_mean", "BIC", "lambda_mu",
  "lambda_kappa", "lambda_eta"
)])
