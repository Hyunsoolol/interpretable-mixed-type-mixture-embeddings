# ==============================================================================
# K=4 paper-like pilot comparison
# ------------------------------------------------------------------------------
# Data: Rossi & Barbaro (2022) artificial simulation style.
# Methods:
#   1. Rossi sparse vMF
#   2. Rossi sparse vMF + coordinate-support refit
#   3. Separate mu/kappa penalty EM
#   4. Separate mu/kappa penalty EM + coordinate-support refit
#   5. Centered eta penalty EM
#   6. Centered eta penalty EM + coordinate-support refit
# ==============================================================================

source("rossi_barbaro_2022_reproduction.r")

parse_num_grid <- function(x) {
  as.numeric(strsplit(x, ",", fixed = TRUE)[[1]])
}

cfg <- list(
  run_label = Sys.getenv("RB2022_K4_PILOT_LABEL", "rb2022_k4_pilot_compare_260608"),
  n_rep = as.integer(Sys.getenv("RB2022_K4_PILOT_N_REP", "5")),
  n = as.integer(Sys.getenv("RB2022_K4_PILOT_N", "1000")),
  d = as.integer(Sys.getenv("RB2022_K4_PILOT_D", "100")),
  K = as.integer(Sys.getenv("RB2022_K4_PILOT_K", "4")),
  overlap = as.numeric(Sys.getenv("RB2022_K4_PILOT_OVERLAP", "0.05")),
  nonzero_fraction = as.numeric(Sys.getenv("RB2022_K4_PILOT_NONZERO", "0.10")),
  nstart = as.integer(Sys.getenv("RB2022_K4_PILOT_NSTART", "3")),
  max_path_steps = as.integer(Sys.getenv("RB2022_K4_PILOT_MAX_PATH_STEPS", "120")),
  max_iter = as.integer(Sys.getenv("RB2022_K4_PILOT_MAX_ITER", "100")),
  lambda_mu_grid = parse_num_grid(Sys.getenv(
    "RB2022_K4_PILOT_LAMBDA_MU", "0,100,200,300,400,600"
  )),
  lambda_kappa_grid = parse_num_grid(Sys.getenv(
    "RB2022_K4_PILOT_LAMBDA_KAPPA", "0,5,10,25"
  )),
  lambda_eta_grid = parse_num_grid(Sys.getenv(
    "RB2022_K4_PILOT_LAMBDA_ETA", "0,1,2,5,10,20,30"
  )),
  base_seed = as.integer(Sys.getenv("RB2022_K4_PILOT_BASE_SEED", "20260608")),
  out_dir = Sys.getenv(
    "RB2022_K4_PILOT_OUT_DIR",
    "results/rb2022_k4_pilot_compare_260608"
  )
)

dir.create(cfg$out_dir, recursive = TRUE, showWarnings = FALSE)

eta_matrix <- function(theta) {
  sweep(theta$mu, 1, theta$kappa, "*")
}

center_eta <- function(eta) {
  sweep(eta, 2, colMeans(eta), "-")
}

active_eta_centered <- function(theta, zero_eps = 1e-8) {
  sqrt(colSums(center_eta(eta_matrix(theta))^2)) > zero_eps
}

active_mu_coord <- function(theta, zero_eps = 1e-8) {
  colSums(abs(theta$mu) > zero_eps) > 0
}

support_metrics <- function(active, support_true) {
  tp <- sum(active & support_true)
  fp <- sum(active & !support_true)
  fn <- sum(!active & support_true)
  tn <- sum(!active & !support_true)
  precision <- ifelse(tp + fp > 0, tp / (tp + fp), NA_real_)
  tpr <- ifelse(tp + fn > 0, tp / (tp + fn), NA_real_)
  fpr <- ifelse(fp + tn > 0, fp / (fp + tn), NA_real_)
  f1 <- ifelse(is.na(precision + tpr) || precision + tpr == 0,
               NA_real_, 2 * precision * tpr / (precision + tpr))
  data.frame(
    selected_q = sum(active),
    TPR = tpr,
    FPR = fpr,
    Precision = precision,
    F1 = f1
  )
}

entry_metrics <- function(support_est, support_true, mu_est, mu_true) {
  if (is.null(support_est)) {
    return(data.frame(
      entry_TPR = NA_real_,
      entry_FPR = NA_real_,
      entry_Precision = NA_real_,
      entry_F1 = NA_real_
    ))
  }
  perm <- best_perm_by_cosine(mu_est, mu_true)
  est <- support_est[perm, , drop = FALSE]
  tp <- sum(est & support_true)
  fp <- sum(est & !support_true)
  fn <- sum(!est & support_true)
  tn <- sum(!est & !support_true)
  precision <- ifelse(tp + fp > 0, tp / (tp + fp), NA_real_)
  tpr <- ifelse(tp + fn > 0, tp / (tp + fn), NA_real_)
  fpr <- ifelse(fp + tn > 0, fp / (fp + tn), NA_real_)
  f1 <- ifelse(is.na(precision + tpr) || precision + tpr == 0,
               NA_real_, 2 * precision * tpr / (precision + tpr))
  data.frame(
    entry_TPR = tpr,
    entry_FPR = fpr,
    entry_Precision = precision,
    entry_F1 = f1
  )
}

parameter_metrics <- function(theta, true_params) {
  K <- nrow(true_params$mu)
  if (nrow(theta$mu) != K) {
    return(data.frame(
      MSE_mu = NA_real_,
      MSE_kappa = NA_real_,
      MSE_centered_eta = NA_real_,
      kappa_hat_mean = NA_real_
    ))
  }
  perm <- best_perm_by_cosine(theta$mu, true_params$mu)
  mu_hat <- theta$mu[perm, , drop = FALSE]
  kappa_hat <- theta$kappa[perm]
  eta_hat <- sweep(mu_hat, 1, kappa_hat, "*")
  eta_true <- sweep(true_params$mu, 1, true_params$kappa, "*")
  data.frame(
    MSE_mu = mean((mu_hat - true_params$mu)^2),
    MSE_kappa = mean((kappa_hat - true_params$kappa)^2),
    MSE_centered_eta = mean((center_eta(eta_hat) - center_eta(eta_true))^2),
    kappa_hat_mean = mean(kappa_hat)
  )
}

support_df <- function(K, active_count) {
  (K - 1) + K + K * max(active_count - 1, 1)
}

support_ic <- function(loglik, n, d, K, active_count, gamma = 0.5) {
  df <- support_df(K, active_count)
  data.frame(
    df = df,
    BIC = log(n) * df - 2 * loglik,
    EBIC = (log(n) + 2 * gamma * log(d)) * df - 2 * loglik
  )
}

separate_model_df <- function(theta, zero_eps = 1e-8) {
  K <- nrow(theta$mu)
  active_kappa <- theta$kappa > zero_eps
  nnz <- rowSums(abs(theta$mu) > zero_eps)
  (K - 1) + sum(active_kappa) + sum(ifelse(active_kappa, pmax(1, nnz - 1), 0))
}

separate_model_ic <- function(theta, n, d, loglik, gamma = 0.5) {
  df <- separate_model_df(theta)
  data.frame(
    df = df,
    BIC = log(n) * df - 2 * loglik,
    EBIC = (log(n) + 2 * gamma * log(d)) * df - 2 * loglik
  )
}

eta_centered_df <- function(theta, zero_eps = 1e-8) {
  K <- nrow(theta$mu)
  d <- ncol(theta$mu)
  m <- sum(active_eta_centered(theta, zero_eps))
  (K - 1) + d + (K - 1) * m
}

eta_centered_ic <- function(theta, n, d, loglik, gamma = 0.5) {
  df <- eta_centered_df(theta)
  data.frame(
    df = df,
    BIC = log(n) * df - 2 * loglik,
    EBIC = (log(n) + 2 * gamma * log(d)) * df - 2 * loglik
  )
}

mask_and_normalize_mu <- function(mu, active, fallback = NULL) {
  K <- nrow(mu)
  d <- ncol(mu)
  out <- matrix(0, nrow = K, ncol = d)
  out[, active] <- mu[, active, drop = FALSE]
  for (k in seq_len(K)) {
    norm_k <- l2_norm(out[k, ])
    if (norm_k > 1e-10) {
      out[k, ] <- out[k, ] / norm_k
    } else if (!is.null(fallback) && l2_norm(fallback[k, active]) > 1e-10) {
      out[k, active] <- fallback[k, active] / l2_norm(fallback[k, active])
    } else {
      out[k, which(active)[1]] <- 1
    }
  }
  out
}

fit_support_refit <- function(X, K, active, init, max_iter = 100,
                              tol = 1e-7, kappa_cap = 1e6) {
  if (!any(active)) stop("No active coordinate selected.")
  n <- nrow(X)
  d <- ncol(X)
  theta <- init
  theta$mu <- mask_and_normalize_mu(theta$mu, active)
  theta$alpha <- pmax(theta$alpha, 1e-12)
  theta$alpha <- theta$alpha / sum(theta$alpha)
  if (length(theta$kappa) == 1) theta$kappa <- rep(theta$kappa, K)
  prev <- -Inf
  last_e <- NULL
  for (iter in seq_len(max_iter)) {
    e <- e_step_vmf(X, theta)
    tau <- e$tau
    Nk <- colSums(tau)
    if (any(Nk < 1e-8)) {
      return(c(theta, list(failed = TRUE, converged = FALSE, iter = iter,
                           loglik = e$loglik, tau = tau)))
    }
    r <- t(tau) %*% X
    alpha_new <- pmax(Nk / n, 1e-12)
    alpha_new <- alpha_new / sum(alpha_new)
    mu_new <- matrix(0, nrow = K, ncol = d)
    kappa_new <- numeric(K)
    for (k in seq_len(K)) {
      rk <- r[k, ]
      rk[!active] <- 0
      mu_new[k, ] <- rk / max(l2_norm(rk), 1e-12)
      rho <- as.numeric(crossprod(mu_new[k, ], r[k, ])) / max(Nk[k], 1e-12)
      kappa_new[k] <- estimate_kappa(rho, d, kappa_cap)
    }
    theta <- list(alpha = alpha_new, mu = mu_new, kappa = kappa_new)
    e_new <- e_step_vmf(X, theta)
    last_e <- e_new
    if (is.finite(prev) &&
        abs(e_new$loglik - prev) / max(1, abs(prev)) < tol) {
      return(c(theta, list(failed = FALSE, converged = TRUE, iter = iter,
                           loglik = e_new$loglik, tau = e_new$tau)))
    }
    prev <- e_new$loglik
  }
  if (is.null(last_e)) last_e <- e_step_vmf(X, theta)
  c(theta, list(failed = FALSE, converged = FALSE, iter = max_iter,
                loglik = last_e$loglik, tau = last_e$tau))
}

update_mu_kappa_separate_one <- function(r_k, Nk, kappa_start, lambda_mu,
                                         lambda_kappa, d, kappa_cap,
                                         inner_max_iter = 80,
                                         inner_tol = 1e-8,
                                         zero_eps = 1e-8) {
  kappa <- kappa_start
  mu <- r_k / max(l2_norm(r_k), 1e-12)
  for (inner in seq_len(inner_max_iter)) {
    shrink <- pmax(kappa * abs(r_k) - lambda_mu, 0)
    shrink_norm <- l2_norm(shrink)
    if (shrink_norm <= zero_eps) {
      return(list(failed = TRUE, mu = mu, kappa = kappa))
    }
    mu_new <- sign(r_k) * shrink / shrink_norm
    s_k <- as.numeric(crossprod(mu_new, r_k))
    rho <- (s_k - lambda_kappa) / max(Nk, 1e-12)
    if (!is.finite(rho) || rho <= 1e-10) {
      return(list(failed = TRUE, mu = mu, kappa = kappa))
    }
    kappa_new <- estimate_kappa(rho, d, kappa_cap)
    diff <- max(max(abs(mu_new - mu)),
                abs(kappa_new - kappa) / max(1, abs(kappa)))
    mu <- mu_new
    kappa <- kappa_new
    if (diff < inner_tol) break
  }
  list(failed = FALSE, mu = mu, kappa = kappa)
}

fit_separate_penalty_em <- function(X, K, lambda_mu, lambda_kappa,
                                    init = NULL, max_iter = 100,
                                    tol = 1e-7, kappa_cap = 1e6) {
  n <- nrow(X)
  d <- ncol(X)
  theta <- if (is.null(init)) init_vmf_mixture(X, K, kappa_cap) else init
  theta$mu <- normalize_rows(theta$mu)
  theta$alpha <- pmax(theta$alpha, 1e-12)
  theta$alpha <- theta$alpha / sum(theta$alpha)
  if (length(theta$kappa) == 1) theta$kappa <- rep(theta$kappa, K)
  prev_obj <- -Inf
  last_e <- NULL
  for (iter in seq_len(max_iter)) {
    e <- e_step_vmf(X, theta)
    tau <- e$tau
    Nk <- colSums(tau)
    if (any(Nk < 1e-8)) {
      return(c(theta, list(failed = TRUE, converged = FALSE, iter = iter,
                           loglik = e$loglik, pen_loglik = NA_real_,
                           tau = tau)))
    }
    r <- t(tau) %*% X
    alpha_new <- pmax(Nk / n, 1e-12)
    alpha_new <- alpha_new / sum(alpha_new)
    mu_new <- matrix(0, nrow = K, ncol = d)
    kappa_new <- theta$kappa
    for (k in seq_len(K)) {
      upd <- update_mu_kappa_separate_one(
        r[k, ], Nk[k], theta$kappa[k], lambda_mu, lambda_kappa,
        d, kappa_cap
      )
      if (isTRUE(upd$failed)) {
        return(c(theta, list(failed = TRUE, converged = FALSE, iter = iter,
                             loglik = e$loglik, pen_loglik = NA_real_,
                             tau = tau)))
      }
      mu_new[k, ] <- upd$mu
      kappa_new[k] <- upd$kappa
    }
    theta_new <- list(alpha = alpha_new, mu = mu_new, kappa = kappa_new)
    e_new <- e_step_vmf(X, theta_new)
    obj <- e_new$loglik -
      lambda_mu * sum(abs(theta_new$mu)) -
      lambda_kappa * sum(theta_new$kappa)
    theta <- theta_new
    last_e <- e_new
    if (is.finite(prev_obj) &&
        abs(obj - prev_obj) / max(1, abs(prev_obj)) < tol) {
      return(c(theta, list(failed = FALSE, converged = TRUE, iter = iter,
                           loglik = e_new$loglik, pen_loglik = obj,
                           tau = e_new$tau)))
    }
    prev_obj <- obj
  }
  if (is.null(last_e)) last_e <- e_step_vmf(X, theta)
  c(theta, list(failed = FALSE, converged = FALSE, iter = max_iter,
                loglik = last_e$loglik,
                pen_loglik = last_e$loglik -
                  lambda_mu * sum(abs(theta$mu)) -
                  lambda_kappa * sum(theta$kappa),
                tau = last_e$tau))
}

eta_to_theta <- function(alpha, eta, fallback_mu = NULL, zero_eps = 1e-10) {
  K <- nrow(eta)
  d <- ncol(eta)
  kappa <- sqrt(rowSums(eta * eta))
  mu <- matrix(0, nrow = K, ncol = d)
  for (k in seq_len(K)) {
    if (kappa[k] > zero_eps) {
      mu[k, ] <- eta[k, ] / kappa[k]
    } else if (!is.null(fallback_mu)) {
      mu[k, ] <- fallback_mu[k, ]
      kappa[k] <- zero_eps
    } else {
      mu[k, 1] <- 1
      kappa[k] <- zero_eps
    }
  }
  list(alpha = alpha / sum(alpha), mu = normalize_rows(mu), kappa = kappa)
}

unpenalized_eta_mstep <- function(X, tau, kappa_cap = 1e6) {
  n <- nrow(X)
  K <- ncol(tau)
  Nk <- colSums(tau)
  r <- t(tau) %*% X
  if (any(Nk < 1e-8)) stop("Empty component in eta M-step.")
  mu <- normalize_rows(r)
  kappa <- numeric(K)
  for (k in seq_len(K)) {
    rho <- l2_norm(r[k, ]) / Nk[k]
    kappa[k] <- estimate_kappa(rho, ncol(X), kappa_cap)
  }
  list(alpha = pmax(Nk / n, 1e-12), eta = sweep(mu, 1, kappa, "*"))
}

prox_eta_centered <- function(eta, lambda_eta) {
  mean_eta <- colMeans(eta)
  centered <- sweep(eta, 2, mean_eta, "-")
  norms <- sqrt(colSums(centered * centered))
  scale <- ifelse(norms > 0, pmax(1 - lambda_eta / norms, 0), 0)
  sweep(sweep(centered, 2, scale, "*"), 2, mean_eta, "+")
}

fit_eta_centered_em <- function(X, K, lambda_eta, init = NULL,
                                max_iter = 100, tol = 1e-7) {
  theta <- if (is.null(init)) {
    fit_svMF_multistart(X, K, beta = 0, nstart = 1, max_iter = max_iter)
  } else {
    init
  }
  theta$mu <- normalize_rows(theta$mu)
  theta$alpha <- pmax(theta$alpha, 1e-12)
  theta$alpha <- theta$alpha / sum(theta$alpha)
  if (length(theta$kappa) == 1) theta$kappa <- rep(theta$kappa, K)
  prev_obj <- -Inf
  last_e <- NULL
  for (iter in seq_len(max_iter)) {
    e <- e_step_vmf(X, theta)
    tau <- e$tau
    mstep <- unpenalized_eta_mstep(X, tau)
    eta_new <- prox_eta_centered(mstep$eta, lambda_eta)
    theta_new <- eta_to_theta(mstep$alpha, eta_new, fallback_mu = theta$mu)
    e_new <- e_step_vmf(X, theta_new)
    penalty <- sum(sqrt(colSums(center_eta(eta_matrix(theta_new))^2)))
    obj <- e_new$loglik - lambda_eta * penalty
    theta <- theta_new
    last_e <- e_new
    if (is.finite(prev_obj) &&
        abs(obj - prev_obj) / max(1, abs(prev_obj)) < tol) {
      return(c(theta, list(failed = FALSE, converged = TRUE, iter = iter,
                           loglik = e_new$loglik, pen_loglik = obj,
                           tau = e_new$tau)))
    }
    prev_obj <- obj
  }
  if (is.null(last_e)) last_e <- e_step_vmf(X, theta)
  penalty <- sum(sqrt(colSums(center_eta(eta_matrix(theta))^2)))
  c(theta, list(failed = FALSE, converged = FALSE, iter = max_iter,
                loglik = last_e$loglik,
                pen_loglik = last_e$loglik - lambda_eta * penalty,
                tau = last_e$tau))
}

eval_method <- function(method, fit, X, z, params, active_coord,
                        support_entry = NULL, lambda_mu = NA_real_,
                        lambda_kappa = NA_real_, lambda_eta = NA_real_,
                        beta = NA_real_, ic = NULL) {
  n <- nrow(X)
  d <- ncol(X)
  K <- nrow(params$mu)
  if (is.null(ic)) {
    ic <- support_ic(fit$loglik, n, d, K, sum(active_coord))
  }
  cluster <- max.col(fit$tau, ties.method = "first")
  cbind(
    data.frame(
      method = method,
      K_fit = nrow(fit$mu),
      beta = beta,
      lambda_mu = lambda_mu,
      lambda_kappa = lambda_kappa,
      lambda_eta = lambda_eta,
      ARI = adjusted_rand_index(z, cluster),
      loglik = fit$loglik,
      pen_loglik = ifelse(is.null(fit$pen_loglik), NA_real_, fit$pen_loglik),
      converged = fit$converged,
      iter = fit$iter,
      true_union_q = sum(colSums(params$support) > 0)
    ),
    support_metrics(active_coord, colSums(params$support) > 0),
    entry_metrics(support_entry, params$support, fit$mu, params$mu),
    parameter_metrics(fit, params),
    ic
  )
}

fit_rossi_pair <- function(X, z, params, cfg) {
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
  idx <- which.min(path$path$BIC)
  fit <- path$fits[[idx]]
  prow <- path$path[idx, , drop = FALSE]
  active <- active_mu_coord(fit)
  support_entry <- abs(fit$mu) > 1e-8
  out <- eval_method(
    "Rossi", fit, X, z, params, active, support_entry,
    beta = prow$beta,
    ic = prow[, c("df", "BIC", "EBIC"), drop = FALSE]
  )
  refit <- fit_support_refit(X, cfg$K, active, fit, max_iter = cfg$max_iter)
  out_refit <- eval_method(
    "Rossi + refit", refit, X, z, params, active, NULL,
    beta = prow$beta
  )
  rbind(out, out_refit)
}

fit_separate_pair <- function(X, z, params, cfg) {
  dense <- fit_svMF_multistart(
    X, cfg$K, beta = 0, nstart = cfg$nstart, max_iter = cfg$max_iter
  )
  rows <- list()
  fits <- list()
  idx <- 1L
  for (lambda_kappa in cfg$lambda_kappa_grid) {
    warm <- dense
    for (lambda_mu in cfg$lambda_mu_grid) {
      fit <- tryCatch(
        fit_separate_penalty_em(
          X, cfg$K, lambda_mu, lambda_kappa, init = warm,
          max_iter = cfg$max_iter
        ),
        error = function(e) NULL
      )
      if (is.null(fit) || isTRUE(fit$failed)) next
      warm <- fit
      active <- active_mu_coord(fit)
      ic <- separate_model_ic(fit, nrow(X), ncol(X), fit$loglik)
      rows[[idx]] <- eval_method(
        "분리 패널티", fit, X, z, params, active,
        abs(fit$mu) > 1e-8,
        lambda_mu = lambda_mu,
        lambda_kappa = lambda_kappa,
        ic = ic
      )
      fits[[idx]] <- fit
      idx <- idx + 1L
    }
  }
  tab <- do.call(rbind, rows)
  best <- which.min(tab$BIC)
  fit <- fits[[best]]
  active <- active_mu_coord(fit)
  out <- tab[best, , drop = FALSE]
  refit <- fit_support_refit(X, cfg$K, active, fit, max_iter = cfg$max_iter)
  out_refit <- eval_method(
    "분리 패널티 + refit", refit, X, z, params, active, NULL,
    lambda_mu = out$lambda_mu,
    lambda_kappa = out$lambda_kappa
  )
  rbind(out, out_refit)
}

fit_eta_pair <- function(X, z, params, cfg) {
  dense <- fit_svMF_multistart(
    X, cfg$K, beta = 0, nstart = cfg$nstart, max_iter = cfg$max_iter
  )
  rows <- list()
  fits <- list()
  warm <- dense
  for (i in seq_along(cfg$lambda_eta_grid)) {
    lambda_eta <- cfg$lambda_eta_grid[i]
    fit <- tryCatch(
      fit_eta_centered_em(
        X, cfg$K, lambda_eta, init = warm, max_iter = cfg$max_iter
      ),
      error = function(e) NULL
    )
    if (is.null(fit) || isTRUE(fit$failed)) next
    warm <- fit
    active <- active_eta_centered(fit)
    ic <- eta_centered_ic(fit, nrow(X), ncol(X), fit$loglik)
    rows[[length(rows) + 1L]] <- eval_method(
      "에타 패널티", fit, X, z, params, active, NULL,
      lambda_eta = lambda_eta,
      ic = ic
    )
    fits[[length(fits) + 1L]] <- fit
  }
  tab <- do.call(rbind, rows)
  best <- which.min(tab$BIC)
  fit <- fits[[best]]
  active <- active_eta_centered(fit)
  out <- tab[best, , drop = FALSE]
  refit <- fit_support_refit(X, cfg$K, active, fit, max_iter = cfg$max_iter)
  out_refit <- eval_method(
    "에타 패널티 + refit", refit, X, z, params, active, NULL,
    lambda_eta = out$lambda_eta
  )
  rbind(out, out_refit)
}

run_one <- function(rep_id, cfg) {
  set.seed(cfg$base_seed + rep_id)
  dat <- simulate_rb2022_data(
    n = cfg$n,
    K = cfg$K,
    d = cfg$d,
    overlap = cfg$overlap,
    nonzero_fraction = cfg$nonzero_fraction
  )
  cat(sprintf("rep %d/%d: true union q=%d\n",
              rep_id, cfg$n_rep, sum(colSums(dat$params$support) > 0)))
  out <- rbind(
    fit_rossi_pair(dat$X, dat$z, dat$params, cfg),
    fit_separate_pair(dat$X, dat$z, dat$params, cfg),
    fit_eta_pair(dat$X, dat$z, dat$params, cfg)
  )
  out$rep <- rep_id
  out$n <- cfg$n
  out$d <- cfg$d
  out$K_true <- cfg$K
  out$overlap <- cfg$overlap
  out$nonzero_fraction <- cfg$nonzero_fraction
  out
}

cat(sprintf(
  "Running K=4 pilot: reps=%d, n=%d, d=%d, overlap=%.3f, nonzero=%.2f\n",
  cfg$n_rep, cfg$n, cfg$d, cfg$overlap, cfg$nonzero_fraction
))

all <- vector("list", cfg$n_rep)
for (rep_id in seq_len(cfg$n_rep)) {
  all[[rep_id]] <- run_one(rep_id, cfg)
}
raw <- do.call(rbind, all)

num_cols <- names(raw)[vapply(raw, is.numeric, logical(1))]
mean_by_method <- aggregate(raw[, num_cols, drop = FALSE],
                            raw[, "method", drop = FALSE],
                            mean, na.rm = TRUE)

method_order <- c(
  "Rossi", "Rossi + refit",
  "분리 패널티", "분리 패널티 + refit",
  "에타 패널티", "에타 패널티 + refit"
)
mean_by_method$method <- factor(mean_by_method$method, levels = method_order)
mean_by_method <- mean_by_method[order(mean_by_method$method), ]
mean_by_method$method <- as.character(mean_by_method$method)

raw_path <- file.path(cfg$out_dir, sprintf("%s_raw.csv", cfg$run_label))
summary_path <- file.path(cfg$out_dir, sprintf("%s_summary.csv", cfg$run_label))
write.csv(raw, raw_path, row.names = FALSE)
write.csv(mean_by_method, summary_path, row.names = FALSE)

print(mean_by_method[, c(
  "method", "ARI", "true_union_q", "selected_q", "TPR", "FPR",
  "Precision", "F1", "MSE_mu", "MSE_kappa", "MSE_centered_eta",
  "kappa_hat_mean"
)])

cat("\nWrote:\n")
cat("  ", normalizePath(raw_path, winslash = "/"), "\n", sep = "")
cat("  ", normalizePath(summary_path, winslash = "/"), "\n", sep = "")
