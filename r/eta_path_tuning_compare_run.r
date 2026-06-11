# ==============================================================================
# Fair tuning comparison: Rossi path, separate 2D path/grid, eta path
# ------------------------------------------------------------------------------
# Goal:
#   Compare Rossi & Barbaro's path-following beta selection with analogous
#   data-adaptive tuning paths for the proposed alternatives.
#
# Methods:
#   1. Rossi sparse vMF, beta path + BIC
#   2. Rossi sparse vMF + support refit
#   3. Separate mu/kappa penalty EM, lambda_kappa grid and lambda_mu path + BIC
#   4. Separate mu/kappa penalty EM + support refit
#   5. Eta-contrast proximal EM, lambda path + BIC
#   6. Eta-contrast proximal EM + support refit
#
# This script focuses on K=2 scenarios, where the eta contrast is
# delta = eta_2 - eta_1.  The eta lambda path and separate lambda_mu path are
# generated from current M-step thresholds, analogous to the Rossi threshold path.
# ==============================================================================

source(file.path("r", "rossi_barbaro_2022_reproduction.r"))

parse_num_grid <- function(x) as.numeric(strsplit(x, ",", fixed = TRUE)[[1]])

cfg <- list(
  run_label = Sys.getenv("ETA_PATH_LABEL", "eta_path_tuning_compare_260622"),
  n_rep = as.integer(Sys.getenv("ETA_PATH_N_REP", "20")),
  n = as.integer(Sys.getenv("ETA_PATH_N", "1000")),
  d = as.integer(Sys.getenv("ETA_PATH_D", "100")),
  q = as.integer(Sys.getenv("ETA_PATH_Q", "10")),
  K = 2L,
  nstart = as.integer(Sys.getenv("ETA_PATH_NSTART", "5")),
  max_iter = as.integer(Sys.getenv("ETA_PATH_MAX_ITER", "120")),
  rossi_max_path_steps = as.integer(Sys.getenv("ETA_PATH_ROSSI_STEPS", "220")),
  eta_max_path_steps = as.integer(Sys.getenv("ETA_PATH_ETA_STEPS", "120")),
  sep_mu_path_steps = as.integer(Sys.getenv("ETA_PATH_SEP_MU_STEPS", "220")),
  sep_kappa_fracs = parse_num_grid(Sys.getenv(
    "ETA_PATH_SEP_KAPPA_FRACS",
    "0,0.05,0.1,0.2,0.35,0.5"
  )),
  sep_inner_max_iter = as.integer(Sys.getenv("ETA_PATH_SEP_INNER_ITER", "80")),
  min_rel_lambda = as.numeric(Sys.getenv("ETA_PATH_MIN_REL_LAMBDA", "1e-3")),
  base_seed = as.integer(Sys.getenv("ETA_PATH_BASE_SEED", "20260622")),
  out_dir = Sys.getenv("ETA_PATH_OUT_DIR", "results/eta_path_tuning_compare_260622")
)

if (!dir.exists(cfg$out_dir)) dir.create(cfg$out_dir, recursive = TRUE)

scenarios <- data.frame(
  scenario = c(
    "concentration_dominant",
    "weak_concentration",
    "mean_and_concentration"
  ),
  mu_cos = c(1, 1, 0.95),
  kappa_low = c(20, 20, 20),
  kappa_high = c(200, 40, 100),
  stringsAsFactors = FALSE
)

make_kappa_contrast_params <- function(d, q, kappa_low, kappa_high, mu_cos = 1) {
  support <- rep(FALSE, d)
  support[seq_len(q)] <- TRUE

  mu1 <- rep(0, d)
  mu1[support] <- 1 / sqrt(q)

  if (mu_cos >= 1 - 1e-12) {
    mu2 <- mu1
  } else {
    v <- rep(0, d)
    v[support] <- seq(-1, 1, length.out = q)
    v <- v - as.numeric(crossprod(v, mu1)) * mu1
    v <- v / l2_norm(v)
    mu2 <- mu_cos * mu1 + sqrt(1 - mu_cos^2) * v
    mu2 <- mu2 / l2_norm(mu2)
  }

  list(
    alpha = c(0.5, 0.5),
    mu = rbind(mu1, mu2),
    kappa = c(kappa_low, kappa_high),
    support = rbind(support, support)
  )
}

simulate_kappa_contrast_data <- function(n, d, q, kappa_low, kappa_high,
                                         mu_cos = 1) {
  params <- make_kappa_contrast_params(d, q, kappa_low, kappa_high, mu_cos)
  z <- sample.int(2, size = n, replace = TRUE, prob = params$alpha)
  X <- matrix(0, nrow = n, ncol = d)
  for (k in 1:2) {
    idx <- which(z == k)
    if (length(idx) > 0) {
      X[idx, ] <- rvMF(length(idx), params$mu[k, ], params$kappa[k])
    }
  }
  list(X = X, z = z, params = params)
}

eta_matrix <- function(theta) sweep(theta$mu, 1, theta$kappa, "*")

support_metrics <- function(active, support_true) {
  tp <- sum(active & support_true)
  fp <- sum(active & !support_true)
  fn <- sum(!active & support_true)
  precision <- ifelse(tp + fp > 0, tp / (tp + fp), NA_real_)
  recall <- ifelse(tp + fn > 0, tp / (tp + fn), NA_real_)
  f1 <- ifelse(is.na(precision + recall) || precision + recall == 0,
               NA_real_, 2 * precision * recall / (precision + recall))
  data.frame(
    selected_q = sum(active),
    TPR = recall,
    FPR = fp / sum(!support_true),
    Precision = precision,
    F1 = f1
  )
}

safe_topq_recall <- function(score, support_true, q) {
  if (all(!is.finite(score)) || max(abs(score), na.rm = TRUE) < 1e-12) {
    return(NA_real_)
  }
  top <- order(abs(score), decreasing = TRUE)[seq_len(min(q, length(score)))]
  mean(support_true[top])
}

contrast_metrics <- function(theta, support_true, q) {
  ord <- order(theta$kappa)
  mu <- theta$mu[ord, , drop = FALSE]
  kappa <- theta$kappa[ord]
  eta <- sweep(mu, 1, kappa, "*")
  mu_delta <- mu[2, ] - mu[1, ]
  eta_delta <- eta[2, ] - eta[1, ]
  data.frame(
    mu_contrast_norm = l2_norm(mu_delta),
    eta_contrast_norm = l2_norm(eta_delta),
    mu_topq_recall = safe_topq_recall(mu_delta, support_true, q),
    eta_topq_recall = safe_topq_recall(eta_delta, support_true, q),
    kappa_low_hat = kappa[1],
    kappa_high_hat = kappa[2],
    kappa_ratio_hat = kappa[2] / max(kappa[1], 1e-12)
  )
}

parameter_mse_metrics <- function(theta, true_params) {
  ord_hat <- order(theta$kappa)
  ord_true <- order(true_params$kappa)
  mu_hat <- theta$mu[ord_hat, , drop = FALSE]
  kappa_hat <- theta$kappa[ord_hat]
  mu_true <- true_params$mu[ord_true, , drop = FALSE]
  kappa_true <- true_params$kappa[ord_true]
  eta_hat <- sweep(mu_hat, 1, kappa_hat, "*")
  eta_true <- sweep(mu_true, 1, kappa_true, "*")
  eta_delta_hat <- eta_hat[2, ] - eta_hat[1, ]
  eta_delta_true <- eta_true[2, ] - eta_true[1, ]
  data.frame(
    MSE_mu = mean((mu_hat - mu_true)^2),
    MSE_kappa = mean((kappa_hat - kappa_true)^2),
    MSE_eta_contrast = mean((eta_delta_hat - eta_delta_true)^2)
  )
}

eta_contrast_active <- function(theta, zero_eps = 1e-8) {
  eta <- eta_matrix(theta)
  abs(eta[2, ] - eta[1, ]) > zero_eps
}

eta_penalty_df <- function(theta, zero_eps = 1e-8) {
  active <- eta_contrast_active(theta, zero_eps)
  K <- nrow(theta$mu)
  d <- ncol(theta$mu)
  (K - 1) + d + sum(active)
}

eta_penalty_ic <- function(theta, n, d, loglik, gamma = 0.5) {
  df <- eta_penalty_df(theta)
  data.frame(
    df = df,
    BIC = log(n) * df - 2 * loglik,
    EBIC = (log(n) + 2 * gamma * log(d)) * df - 2 * loglik
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
  d <- ncol(X)
  K <- ncol(tau)
  Nk <- colSums(tau)
  r <- t(tau) %*% X
  if (any(Nk < 1e-8)) stop("Empty component in eta M-step.")
  mu <- normalize_rows(r)
  kappa <- numeric(K)
  for (k in seq_len(K)) {
    rho <- l2_norm(r[k, ]) / Nk[k]
    kappa[k] <- estimate_kappa(rho, d)
  }
  list(alpha = pmax(Nk / n, 1e-12), eta = sweep(mu, 1, kappa, "*"))
}

soft_threshold <- function(x, lambda) {
  sign(x) * pmax(abs(x) - lambda, 0)
}

prox_eta_contrast_k2 <- function(eta, lambda_eta) {
  eta_bar <- 0.5 * (eta[1, ] + eta[2, ])
  delta <- eta[2, ] - eta[1, ]
  delta_shrunk <- soft_threshold(delta, lambda_eta)
  rbind(
    eta_bar - 0.5 * delta_shrunk,
    eta_bar + 0.5 * delta_shrunk
  )
}

fit_eta_penalty_em <- function(X, lambda_eta, init = NULL, max_iter = 120,
                               tol = 1e-7) {
  K <- 2L
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
    mstep <- unpenalized_eta_mstep(X, e$tau)
    eta_shrunk <- prox_eta_contrast_k2(mstep$eta, lambda_eta)
    theta_new <- eta_to_theta(mstep$alpha, eta_shrunk, fallback_mu = theta$mu)
    e_new <- e_step_vmf(X, theta_new)
    delta <- eta_matrix(theta_new)[2, ] - eta_matrix(theta_new)[1, ]
    obj <- e_new$loglik - lambda_eta * sum(abs(delta))
    theta <- theta_new
    last_e <- e_new
    if (is.finite(prev_obj) &&
        abs(obj - prev_obj) / max(1, abs(prev_obj)) < tol) {
      return(c(theta, list(
        failed = FALSE, converged = TRUE, iter = iter,
        loglik = e_new$loglik, pen_loglik = obj, tau = e_new$tau
      )))
    }
    prev_obj <- obj
  }
  if (is.null(last_e)) last_e <- e_step_vmf(X, theta)
  delta <- eta_matrix(theta)[2, ] - eta_matrix(theta)[1, ]
  c(theta, list(
    failed = FALSE, converged = FALSE, iter = max_iter,
    loglik = last_e$loglik,
    pen_loglik = last_e$loglik - lambda_eta * sum(abs(delta)),
    tau = last_e$tau
  ))
}

separate_penalty_active <- function(theta, zero_eps = 1e-8) {
  colSums(abs(theta$mu) > zero_eps) > 0
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

update_mu_kappa_separate_one <- function(r_k, Nk, kappa_start, lambda_mu,
                                         lambda_kappa, d, kappa_cap = 1e6,
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
    diff <- max(
      max(abs(mu_new - mu)),
      abs(kappa_new - kappa) / max(1, abs(kappa))
    )
    mu <- mu_new
    kappa <- kappa_new
    if (diff < inner_tol) break
  }

  list(failed = FALSE, mu = mu, kappa = kappa)
}

fit_separate_penalty_em <- function(X, K, lambda_mu, lambda_kappa, init = NULL,
                                    max_iter = 120, inner_max_iter = 80,
                                    tol = 1e-7, inner_tol = 1e-8,
                                    zero_eps = 1e-8, kappa_cap = 1e6) {
  n <- nrow(X)
  d <- ncol(X)
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
    Nk <- colSums(tau)
    if (any(Nk < 1e-8)) {
      return(c(theta, list(
        failed = TRUE, converged = FALSE, iter = iter,
        loglik = e$loglik, pen_loglik = NA_real_, tau = tau
      )))
    }

    r <- t(tau) %*% X
    alpha_new <- pmax(Nk / n, 1e-12)
    alpha_new <- alpha_new / sum(alpha_new)
    mu_new <- matrix(0, nrow = K, ncol = d)
    kappa_new <- theta$kappa

    for (k in seq_len(K)) {
      upd <- update_mu_kappa_separate_one(
        r_k = r[k, ], Nk = Nk[k], kappa_start = theta$kappa[k],
        lambda_mu = lambda_mu, lambda_kappa = lambda_kappa, d = d,
        kappa_cap = kappa_cap, inner_max_iter = inner_max_iter,
        inner_tol = inner_tol, zero_eps = zero_eps
      )
      if (isTRUE(upd$failed)) {
        return(c(theta, list(
          failed = TRUE, converged = FALSE, iter = iter,
          loglik = e$loglik, pen_loglik = NA_real_, tau = tau
        )))
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
      return(c(theta, list(
        failed = FALSE, converged = TRUE, iter = iter,
        loglik = e_new$loglik, pen_loglik = obj, tau = e_new$tau
      )))
    }
    prev_obj <- obj
  }

  if (is.null(last_e)) last_e <- e_step_vmf(X, theta)
  c(theta, list(
    failed = FALSE, converged = FALSE, iter = max_iter,
    loglik = last_e$loglik,
    pen_loglik = last_e$loglik -
      lambda_mu * sum(abs(theta$mu)) -
      lambda_kappa * sum(theta$kappa),
    tau = last_e$tau
  ))
}

mask_and_normalize_mu <- function(mu, active, fallback = NULL) {
  K <- nrow(mu)
  d <- ncol(mu)
  out <- matrix(0, nrow = K, ncol = d)
  out[, active] <- mu[, active, drop = FALSE]
  for (k in seq_len(K)) {
    if (l2_norm(out[k, ]) < 1e-10) {
      if (!is.null(fallback) && l2_norm(fallback[k, active]) > 1e-10) {
        out[k, active] <- fallback[k, active]
      } else {
        out[k, which(active)[1]] <- 1
      }
    }
    out[k, ] <- out[k, ] / l2_norm(out[k, ])
  }
  out
}

fit_support_constrained_vmf <- function(X, K, active, init, max_iter = 120,
                                        tol = 1e-7, kappa_cap = 1e6) {
  n <- nrow(X)
  d <- ncol(X)
  theta <- list(
    alpha = init$alpha / sum(init$alpha),
    mu = mask_and_normalize_mu(init$mu, active, fallback = init$mu),
    kappa = init$kappa
  )
  if (length(theta$kappa) == 1) theta$kappa <- rep(theta$kappa, K)
  prev <- -Inf
  last_e <- NULL
  for (iter in seq_len(max_iter)) {
    e <- e_step_vmf(X, theta)
    tau <- e$tau
    Nk <- colSums(tau)
    if (any(Nk < 1e-8)) {
      return(c(theta, list(
        failed = TRUE, converged = FALSE, iter = iter,
        loglik = e$loglik, tau = tau
      )))
    }
    r <- t(tau) %*% X
    r[, !active] <- 0
    mu_new <- normalize_rows(r)
    kappa_new <- numeric(K)
    for (k in seq_len(K)) {
      rho <- as.numeric(crossprod(mu_new[k, ], r[k, ])) / Nk[k]
      kappa_new[k] <- estimate_kappa(rho, d, kappa_cap)
    }
    theta <- list(
      alpha = pmax(Nk / n, 1e-12),
      mu = mu_new,
      kappa = kappa_new
    )
    theta$alpha <- theta$alpha / sum(theta$alpha)
    e_new <- e_step_vmf(X, theta)
    last_e <- e_new
    if (is.finite(prev) &&
        abs(e_new$loglik - prev) / max(1, abs(prev)) < tol) {
      return(c(theta, list(
        failed = FALSE, converged = TRUE, iter = iter,
        loglik = e_new$loglik, tau = e_new$tau
      )))
    }
    prev <- e_new$loglik
  }
  if (is.null(last_e)) last_e <- e_step_vmf(X, theta)
  c(theta, list(
    failed = FALSE, converged = FALSE, iter = max_iter,
    loglik = last_e$loglik, tau = last_e$tau
  ))
}

evaluate_row <- function(method, fit, X, z, support_true, true_params, q,
                         lambda_eta = NA_real_, beta = NA_real_,
                         lambda_mu = NA_real_, lambda_kappa = NA_real_,
                         active = NULL, use_support_ic = FALSE) {
  if (is.null(active)) active <- eta_contrast_active(fit)
  n <- nrow(X)
  d <- ncol(X)
  ic <- if (use_support_ic) {
    support_ic(fit$loglik, n, d, nrow(fit$mu), sum(active))
  } else if (!is.na(lambda_mu) || !is.na(lambda_kappa)) {
    separate_model_ic(fit, n, d, fit$loglik)
  } else if (is.na(lambda_eta)) {
    model_ic(fit, n, d, fit$loglik)
  } else {
    eta_penalty_ic(fit, n, d, fit$loglik)
  }
  ic <- ic[, c("df", "BIC", "EBIC"), drop = FALSE]
  cluster <- max.col(fit$tau, ties.method = "first")
  data.frame(
    method = method,
    beta = beta,
    lambda_eta = lambda_eta,
    lambda_mu = lambda_mu,
    lambda_kappa = lambda_kappa,
    ARI = adjusted_rand_index(z, cluster),
    loglik = fit$loglik,
    converged = fit$converged,
    iter = fit$iter,
    nnz_fraction = mean(active),
    support_metrics(active, support_true),
    contrast_metrics(fit, support_true, q),
    parameter_mse_metrics(fit, true_params),
    ic
  )
}

fit_rossi_bic_pair <- function(X, z, params, cfg) {
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
  idx <- which.min(path$path$BIC)
  fit <- path$fits[[idx]]
  beta <- path$path$beta[idx]
  active <- colSums(abs(fit$mu) > 1e-8) > 0
  out <- evaluate_row("Rossi path BIC", fit, X, z, params$support[1, ],
                      params, cfg$q, beta = beta, active = active)
  refit <- fit_support_constrained_vmf(X, cfg$K, active, fit,
                                       max_iter = cfg$max_iter)
  out_refit <- evaluate_row("Rossi path BIC + refit", refit, X, z,
                            params$support[1, ], params, cfg$q,
                            beta = beta, active = active,
                            use_support_ic = TRUE)
  rbind(out, out_refit)
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
    init = dense_fit, max_iter = cfg$max_iter,
    inner_max_iter = cfg$sep_inner_max_iter
  )
  if (isTRUE(fit$failed)) return(list(rows = NULL, fits = list()))

  fits <- list(fit)
  rows <- list(evaluate_row(
    "Separate 2D path/grid BIC", fit, X, z, params$support[1, ],
    params, cfg$q, lambda_mu = lambda_mu, lambda_kappa = lambda_kappa,
    active = separate_penalty_active(fit)
  ))

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
        init = fit, max_iter = cfg$max_iter,
        inner_max_iter = cfg$sep_inner_max_iter
      ),
      error = function(e) NULL
    )
    if (is.null(fit_next) || isTRUE(fit_next$failed)) break

    fit <- fit_next
    lambda_mu <- lambda_next
    fits[[length(fits) + 1L]] <- fit
    rows[[length(rows) + 1L]] <- evaluate_row(
      "Separate 2D path/grid BIC", fit, X, z, params$support[1, ],
      params, cfg$q, lambda_mu = lambda_mu, lambda_kappa = lambda_kappa,
      active = separate_penalty_active(fit)
    )
    if (sum(separate_penalty_active(fit)) <= 1) break
  }

  list(rows = do.call(rbind, rows), fits = fits)
}

fit_separate_path_grid <- function(X, z, params, cfg) {
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

  if (length(all_rows) == 0) {
    stop("Separate 2D path/grid produced no valid fit.")
  }

  grid_path <- do.call(rbind, all_rows)
  idx <- which.min(grid_path$BIC)
  fit_best <- all_fits[[idx]]
  lambda_mu_best <- grid_path$lambda_mu[idx]
  lambda_kappa_best <- grid_path$lambda_kappa[idx]
  active <- separate_penalty_active(fit_best)
  out <- grid_path[idx, , drop = FALSE]

  refit <- fit_support_constrained_vmf(
    X, cfg$K, active, fit_best, max_iter = cfg$max_iter
  )
  out_refit <- evaluate_row(
    "Separate 2D path/grid BIC + refit", refit, X, z,
    params$support[1, ], params, cfg$q,
    lambda_mu = lambda_mu_best, lambda_kappa = lambda_kappa_best,
    active = active, use_support_ic = TRUE
  )

  list(rows = rbind(out, out_refit), path = grid_path)
}

fit_eta_lambda_path <- function(X, z, params, cfg) {
  dense <- fit_svMF_multistart(
    X, cfg$K, beta = 0, nstart = cfg$nstart, max_iter = cfg$max_iter
  )
  lambda <- 0
  fit <- fit_eta_penalty_em(X, lambda_eta = lambda, init = dense,
                            max_iter = cfg$max_iter)
  fits <- list(fit)
  rows <- list(evaluate_row("Eta path BIC", fit, X, z, params$support[1, ],
                            params, cfg$q, lambda_eta = lambda))

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
      fit_eta_penalty_em(X, lambda_eta = lambda_next, init = fit,
                         max_iter = cfg$max_iter),
      error = function(e) NULL
    )
    if (is.null(fit_next) || isTRUE(fit_next$failed)) break
    fit <- fit_next
    lambda <- lambda_next
    fits[[length(fits) + 1L]] <- fit
    rows[[length(rows) + 1L]] <- evaluate_row(
      "Eta path BIC", fit, X, z, params$support[1, ], params, cfg$q,
      lambda_eta = lambda
    )
    if (sum(eta_contrast_active(fit)) <= 1) break
  }

  path <- do.call(rbind, rows)
  idx <- which.min(path$BIC)
  fit_best <- fits[[idx]]
  lambda_best <- path$lambda_eta[idx]
  active <- eta_contrast_active(fit_best)
  out <- path[idx, , drop = FALSE]
  refit <- fit_support_constrained_vmf(X, cfg$K, active, fit_best,
                                       max_iter = cfg$max_iter)
  out_refit <- evaluate_row("Eta path BIC + refit", refit, X, z,
                            params$support[1, ], params, cfg$q,
                            lambda_eta = lambda_best,
                            active = active,
                            use_support_ic = TRUE)
  list(rows = rbind(out, out_refit), path = path)
}

run_one <- function(scenario_row, rep_id, cfg) {
  set.seed(cfg$base_seed + rep_id + 1000L * match(scenario_row$scenario, scenarios$scenario))
  dat <- simulate_kappa_contrast_data(
    n = cfg$n, d = cfg$d, q = cfg$q,
    kappa_low = scenario_row$kappa_low,
    kappa_high = scenario_row$kappa_high,
    mu_cos = scenario_row$mu_cos
  )
  rossi <- fit_rossi_bic_pair(dat$X, dat$z, dat$params, cfg)
  separate <- fit_separate_path_grid(dat$X, dat$z, dat$params, cfg)
  eta <- fit_eta_lambda_path(dat$X, dat$z, dat$params, cfg)
  rows <- rbind(rossi, separate$rows, eta$rows)
  rows$scenario <- scenario_row$scenario
  rows$rep <- rep_id
  rows$n <- cfg$n
  rows$d <- cfg$d
  rows$q <- cfg$q
  rows$mu_cos <- scenario_row$mu_cos
  rows$kappa_low <- scenario_row$kappa_low
  rows$kappa_high <- scenario_row$kappa_high
  rows$true_eta_contrast_norm <- l2_norm(
    dat$params$kappa[2] * dat$params$mu[2, ] -
      dat$params$kappa[1] * dat$params$mu[1, ]
  )
  rows
}

raw_rows <- list()
idx <- 1L
for (s in seq_len(nrow(scenarios))) {
  for (rep_id in seq_len(cfg$n_rep)) {
    cat(sprintf("[%s] rep %d/%d\n", scenarios$scenario[s], rep_id, cfg$n_rep))
    raw_rows[[idx]] <- tryCatch(
      run_one(scenarios[s, ], rep_id, cfg),
      error = function(e) {
        data.frame(
          method = "ERROR", beta = NA_real_, lambda_eta = NA_real_,
          lambda_mu = NA_real_, lambda_kappa = NA_real_,
          ARI = NA_real_, loglik = NA_real_, converged = NA, iter = NA_real_,
          nnz_fraction = NA_real_, selected_q = NA_real_, TPR = NA_real_,
          FPR = NA_real_, Precision = NA_real_, F1 = NA_real_,
          mu_contrast_norm = NA_real_, eta_contrast_norm = NA_real_,
          mu_topq_recall = NA_real_, eta_topq_recall = NA_real_,
          kappa_low_hat = NA_real_, kappa_high_hat = NA_real_,
          kappa_ratio_hat = NA_real_, MSE_mu = NA_real_,
          MSE_kappa = NA_real_, MSE_eta_contrast = NA_real_,
          df = NA_real_, BIC = NA_real_, EBIC = NA_real_,
          scenario = scenarios$scenario[s], rep = rep_id,
          n = cfg$n, d = cfg$d, q = cfg$q, mu_cos = scenarios$mu_cos[s],
          kappa_low = scenarios$kappa_low[s],
          kappa_high = scenarios$kappa_high[s],
          true_eta_contrast_norm = NA_real_,
          error = conditionMessage(e)
        )
      }
    )
    idx <- idx + 1L
  }
}

raw <- do.call(rbind, raw_rows)
raw_path <- file.path(cfg$out_dir, sprintf("%s_raw.csv", cfg$run_label))
write.csv(raw, raw_path, row.names = FALSE)

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
summary_path <- file.path(cfg$out_dir, sprintf("%s_summary.csv", cfg$run_label))
write.csv(summary, summary_path, row.names = FALSE)

cat("\nWrote:\n")
cat("  ", normalizePath(raw_path, winslash = "/"), "\n", sep = "")
cat("  ", normalizePath(summary_path, winslash = "/"), "\n", sep = "")
print(summary[, c(
  "scenario", "method", "reps", "ARI_mean", "selected_q_mean",
  "TPR_mean", "FPR_mean", "Precision_mean", "F1_mean",
  "eta_contrast_norm_mean", "kappa_ratio_hat_mean",
  "MSE_eta_contrast_mean", "BIC_mean", "lambda_eta_mean",
  "lambda_mu_mean", "lambda_kappa_mean"
)])
