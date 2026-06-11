# ==============================================================================
# Eta-contrast proximal EM under the kappa-contrast limitation scenario
# ------------------------------------------------------------------------------
# Prototype objective for K = 2:
#
#   log L - lambda_eta * sum_j |eta_2j - eta_1j|,
#   eta_k = kappa_k * mu_k.
#
# The exact penalized M-step is not closed form.  This script uses a practical
# proximal EM prototype:
#   1. Do the usual unpenalized vMF M-step in eta-space.
#   2. Soft-threshold delta = eta_2 - eta_1 coordinate-wise.
#   3. Convert eta_k back to (mu_k, kappa_k).
#
# This is meant to test whether a direct eta-contrast penalty improves
# coordinate-level recovery beyond mu-only or separate mu/kappa penalties.
# ==============================================================================

source(file.path("r", "rossi_barbaro_2022_reproduction.r"))

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

parse_num_grid <- function(x) as.numeric(strsplit(x, ",")[[1]])

cfg <- list(
  run_label = Sys.getenv("ETA_PEN_RUN_LABEL", "eta_penalty_vmf"),
  n_rep = as.integer(Sys.getenv("ETA_PEN_N_REP", "30")),
  n = as.integer(Sys.getenv("ETA_PEN_N", "1000")),
  d = as.integer(Sys.getenv("ETA_PEN_D", "100")),
  q = as.integer(Sys.getenv("ETA_PEN_Q", "10")),
  K_true = 2,
  K_fit = 2,
  kappa_low = as.numeric(Sys.getenv("ETA_PEN_KAPPA_LOW", "20")),
  kappa_high = as.numeric(Sys.getenv("ETA_PEN_KAPPA_HIGH", "200")),
  mu_cos = as.numeric(Sys.getenv("ETA_PEN_MU_COS", "1")),
  lambda_eta_grid = parse_num_grid(Sys.getenv(
    "ETA_PEN_LAMBDA_GRID",
    "0,1,2,5,10,15,20,30,40,50"
  )),
  nstart = as.integer(Sys.getenv("ETA_PEN_NSTART", "5")),
  max_iter = as.integer(Sys.getenv("ETA_PEN_MAX_ITER", "120")),
  workers = as.integer(Sys.getenv("ETA_PEN_WORKERS", "1")),
  base_seed = as.integer(Sys.getenv("ETA_PEN_BASE_SEED", "20260602")),
  out_dir = Sys.getenv("ETA_PEN_OUT_DIR", "results/eta_penalty_vmf_260604")
)

if (!dir.exists(cfg$out_dir)) dir.create(cfg$out_dir, recursive = TRUE)
cell_dir <- file.path(cfg$out_dir, "cells")
if (!dir.exists(cell_dir)) dir.create(cell_dir, recursive = TRUE)

# ------------------------------------------------------------------------------
# Data generation
# ------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------
# Evaluation helpers
# ------------------------------------------------------------------------------

support_metrics <- function(active, support_true) {
  tp <- sum(active & support_true)
  fp <- sum(active & !support_true)
  fn <- sum(!active & support_true)

  precision <- ifelse(tp + fp > 0, tp / (tp + fp), NA_real_)
  recall <- ifelse(tp + fn > 0, tp / (tp + fn), NA_real_)
  f1 <- ifelse(is.na(precision + recall) || precision + recall == 0,
               NA_real_, 2 * precision * recall / (precision + recall))

  data.frame(
    shat = sum(active),
    TPR = recall,
    FPR = fp / sum(!support_true),
    precision = precision,
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

eta_matrix <- function(theta) {
  sweep(theta$mu, 1, theta$kappa, "*")
}

contrast_metrics <- function(theta, support_true, q) {
  if (nrow(theta$mu) != 2) {
    return(data.frame(
      mu_contrast_norm = NA_real_,
      eta_contrast_norm = NA_real_,
      mu_topq_recall = NA_real_,
      eta_topq_recall = NA_real_,
      kappa_low_hat = NA_real_,
      kappa_high_hat = NA_real_,
      kappa_ratio_hat = NA_real_
    ))
  }

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
  if (nrow(theta$mu) != nrow(true_params$mu)) {
    return(data.frame(
      MSE_mu = NA_real_,
      MSE_kappa = NA_real_,
      MSE_eta_contrast = NA_real_
    ))
  }

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

# ------------------------------------------------------------------------------
# Eta-contrast proximal update
# ------------------------------------------------------------------------------

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
    kappa[k] <- estimate_kappa(rho, d, kappa_cap)
  }

  list(
    alpha = pmax(Nk / n, 1e-12),
    eta = sweep(mu, 1, kappa, "*"),
    mu = mu,
    kappa = kappa
  )
}

soft_threshold <- function(x, lambda) {
  sign(x) * pmax(abs(x) - lambda, 0)
}

prox_eta_contrast_k2 <- function(eta, lambda_eta) {
  if (nrow(eta) != 2) stop("Eta contrast proximal update is implemented for K=2.")

  eta_bar <- 0.5 * (eta[1, ] + eta[2, ])
  delta <- eta[2, ] - eta[1, ]
  delta_shrunk <- soft_threshold(delta, lambda_eta)

  rbind(
    eta_bar - 0.5 * delta_shrunk,
    eta_bar + 0.5 * delta_shrunk
  )
}

fit_eta_penalty_em <- function(X,
                               lambda_eta,
                               init = NULL,
                               max_iter = 120,
                               tol = 1e-7,
                               zero_eps = 1e-8) {
  K <- 2
  theta <- if (is.null(init)) {
    fit_svMF_multistart(X, K, beta = 0, nstart = 1)
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
        failed = FALSE,
        converged = TRUE,
        iter = iter,
        loglik = e_new$loglik,
        pen_loglik = obj,
        tau = e_new$tau
      )))
    }

    prev_obj <- obj
  }

  if (is.null(last_e)) last_e <- e_step_vmf(X, theta)
  delta <- eta_matrix(theta)[2, ] - eta_matrix(theta)[1, ]
  c(theta, list(
    failed = FALSE,
    converged = FALSE,
    iter = max_iter,
    loglik = last_e$loglik,
    pen_loglik = last_e$loglik - lambda_eta * sum(abs(delta)),
    tau = last_e$tau
  ))
}

eta_contrast_active <- function(theta, zero_eps = 1e-8) {
  eta <- eta_matrix(theta)
  abs(eta[2, ] - eta[1, ]) > zero_eps
}

eta_penalty_df <- function(theta, zero_eps = 1e-8) {
  d <- ncol(theta$mu)
  m <- sum(eta_contrast_active(theta, zero_eps))

  # Reparameterization for K=2:
  #   eta_bar has d common natural-parameter coordinates,
  #   delta has m active contrast coordinates,
  #   alpha contributes one mixing parameter.
  1 + d + m
}

eta_penalty_ic <- function(theta, n, d, loglik, gamma = 0.5) {
  df <- eta_penalty_df(theta)
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
    if (l2_norm(out[k, ]) < 1e-10) {
      if (!is.null(fallback) && l2_norm(fallback[k, active]) > 1e-10) {
        out[k, active] <- fallback[k, active]
      } else {
        out[k, active] <- rnorm(sum(active))
      }
    }
    out[k, ] <- out[k, ] / l2_norm(out[k, ])
  }

  out
}

fit_support_constrained_vmf <- function(X,
                                        K,
                                        active,
                                        init,
                                        max_iter = 200,
                                        tol = 1e-7,
                                        kappa_cap = 1e6) {
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
        failed = TRUE,
        converged = FALSE,
        iter = iter,
        loglik = e$loglik,
        tau = tau
      )))
    }

    # Refit step: the eta-penalty support is fixed, and inactive coordinates
    # are held at zero as in the eta screening + refit comparison.
    r <- t(tau) %*% X
    r[, !active] <- 0
    mu_new <- normalize_rows(r)

    kappa_new <- numeric(K)
    for (k in seq_len(K)) {
      rho <- as.numeric(crossprod(mu_new[k, ], r[k, ])) / Nk[k]
      kappa_new[k] <- estimate_kappa(rho, d, kappa_cap)
    }

    theta_new <- list(
      alpha = pmax(Nk / n, 1e-12),
      mu = mu_new,
      kappa = kappa_new
    )
    theta_new$alpha <- theta_new$alpha / sum(theta_new$alpha)
    e_new <- e_step_vmf(X, theta_new)

    theta <- theta_new
    last_e <- e_new

    if (is.finite(prev) &&
        abs(e_new$loglik - prev) / max(1, abs(prev)) < tol) {
      return(c(theta, list(
        failed = FALSE,
        converged = TRUE,
        iter = iter,
        loglik = e_new$loglik,
        tau = e_new$tau
      )))
    }
    prev <- e_new$loglik
  }

  if (is.null(last_e)) last_e <- e_step_vmf(X, theta)
  c(theta, list(
    failed = FALSE,
    converged = FALSE,
    iter = max_iter,
    loglik = last_e$loglik,
    tau = last_e$tau
  ))
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

evaluate_eta_penalty_fit <- function(fit,
                                     X,
                                     z,
                                     support_true,
                                     true_params,
                                     q,
                                     lambda_eta) {
  n <- nrow(X)
  d <- ncol(X)
  active <- eta_contrast_active(fit)
  ic <- eta_penalty_ic(fit, n, d, fit$loglik)
  cluster <- max.col(fit$tau, ties.method = "first")
  sm <- support_metrics(active, support_true)
  cm <- contrast_metrics(fit, support_true, q)
  pm <- parameter_mse_metrics(fit, true_params)

  data.frame(
    method = "eta_penalty_EM_BIC",
    lambda_eta = lambda_eta,
    ARI = adjusted_rand_index(z, cluster),
    loglik = fit$loglik,
    pen_loglik = fit$pen_loglik,
    converged = fit$converged,
    iter = fit$iter,
    nnz_fraction = mean(active),
    sm,
    cm,
    pm,
    ic
  )
}

evaluate_eta_penalty_refit <- function(fit,
                                       X,
                                       z,
                                       support_true,
                                       true_params,
                                       q,
                                       lambda_eta,
                                       active_selected) {
  n <- nrow(X)
  d <- ncol(X)
  K <- nrow(fit$mu)
  ic <- support_ic(fit$loglik, n, d, K, sum(active_selected))
  cluster <- max.col(fit$tau, ties.method = "first")
  sm <- support_metrics(active_selected, support_true)
  cm <- contrast_metrics(fit, support_true, q)
  pm <- parameter_mse_metrics(fit, true_params)

  data.frame(
    method = "eta_penalty_EM_BIC_refit",
    lambda_eta = lambda_eta,
    ARI = adjusted_rand_index(z, cluster),
    loglik = fit$loglik,
    pen_loglik = NA_real_,
    converged = fit$converged,
    iter = fit$iter,
    nnz_fraction = sum(active_selected) / d,
    sm,
    cm,
    pm,
    ic
  )
}

fit_eta_penalty_grid <- function(X, z, support_true, true_params, cfg) {
  rows <- list()
  fits <- list()

  dense <- fit_svMF_multistart(X, cfg$K_fit, beta = 0, nstart = cfg$nstart)
  init <- dense

  for (lambda_eta in cfg$lambda_eta_grid) {
    fit <- tryCatch(
      fit_eta_penalty_em(
        X = X,
        lambda_eta = lambda_eta,
        init = init,
        max_iter = cfg$max_iter
      ),
      error = function(e) NULL
    )

    if (is.null(fit) || isTRUE(fit$failed)) next

    row <- evaluate_eta_penalty_fit(
      fit,
      X,
      z,
      support_true,
      true_params,
      cfg$q,
      lambda_eta = lambda_eta
    )

    rows[[length(rows) + 1L]] <- row
    fits[[length(fits) + 1L]] <- fit
    init <- fit
  }

  if (length(rows) == 0L) {
    stop("All eta-penalty grid fits failed.")
  }

  path <- do.call(rbind, rows)
  idx <- which.min(path$BIC)
  list(path = path, row = path[idx, , drop = FALSE], fit = fits[[idx]])
}

# ------------------------------------------------------------------------------
# Simulation driver
# ------------------------------------------------------------------------------

run_one <- function(rep_id, cfg, cell_dir) {
  out_file <- file.path(cell_dir, sprintf("cell_%03d.csv", rep_id))
  path_file <- file.path(cell_dir, sprintf("path_%03d.csv", rep_id))
  if (file.exists(out_file)) return(out_file)

  set.seed(cfg$base_seed + rep_id)
  dat <- simulate_kappa_contrast_data(
    n = cfg$n,
    d = cfg$d,
    q = cfg$q,
    kappa_low = cfg$kappa_low,
    kappa_high = cfg$kappa_high,
    mu_cos = cfg$mu_cos
  )

  fit <- tryCatch(
    fit_eta_penalty_grid(dat$X, dat$z, dat$params$support[1, ], dat$params, cfg),
    error = function(e) e
  )

  if (inherits(fit, "error")) {
    out <- data.frame(
      method = c("eta_penalty_EM_BIC", "eta_penalty_EM_BIC_refit"),
      lambda_eta = NA_real_,
      ARI = NA_real_,
      loglik = NA_real_,
      pen_loglik = NA_real_,
      converged = NA,
      iter = NA_real_,
      nnz_fraction = NA_real_,
      shat = NA_real_,
      TPR = NA_real_,
      FPR = NA_real_,
      precision = NA_real_,
      F1 = NA_real_,
      mu_contrast_norm = NA_real_,
      eta_contrast_norm = NA_real_,
      MSE_mu = NA_real_,
      MSE_kappa = NA_real_,
      MSE_eta_contrast = NA_real_,
      mu_topq_recall = NA_real_,
      eta_topq_recall = NA_real_,
      kappa_low_hat = NA_real_,
      kappa_high_hat = NA_real_,
      kappa_ratio_hat = NA_real_,
      df = NA_real_,
      BIC = NA_real_,
      EBIC = NA_real_,
      error = conditionMessage(fit)
    )
  } else {
    out_pen <- fit$row
    out_pen$error <- NA_character_

    active <- eta_contrast_active(fit$fit)
    refit <- if (any(active)) {
      tryCatch(
        fit_support_constrained_vmf(
          dat$X,
          K = cfg$K_fit,
          active = active,
          init = fit$fit,
          max_iter = cfg$max_iter
        ),
        error = function(e) e
      )
    } else {
      structure(
        list(message = "Eta penalty selected no active coordinates."),
        class = c("simpleError", "error", "condition")
      )
    }

    if (inherits(refit, "error") || isTRUE(refit$failed)) {
      out_refit <- data.frame(
        method = "eta_penalty_EM_BIC_refit",
        lambda_eta = fit$row$lambda_eta,
        ARI = NA_real_,
        loglik = NA_real_,
        pen_loglik = NA_real_,
        converged = NA,
        iter = NA_real_,
        nnz_fraction = sum(active) / cfg$d,
        shat = sum(active),
        TPR = NA_real_,
        FPR = NA_real_,
        precision = NA_real_,
        F1 = NA_real_,
        mu_contrast_norm = NA_real_,
        eta_contrast_norm = NA_real_,
        MSE_mu = NA_real_,
        MSE_kappa = NA_real_,
        MSE_eta_contrast = NA_real_,
        mu_topq_recall = NA_real_,
        eta_topq_recall = NA_real_,
        kappa_low_hat = NA_real_,
        kappa_high_hat = NA_real_,
        kappa_ratio_hat = NA_real_,
        df = NA_real_,
        BIC = NA_real_,
        EBIC = NA_real_,
        error = if (inherits(refit, "error")) conditionMessage(refit) else "Support refit failed"
      )
    } else {
      out_refit <- evaluate_eta_penalty_refit(
        refit,
        dat$X,
        dat$z,
        dat$params$support[1, ],
        dat$params,
        cfg$q,
        lambda_eta = fit$row$lambda_eta,
        active_selected = active
      )
      out_refit$error <- NA_character_
    }

    out <- rbind(out_pen, out_refit)
    path <- fit$path
    path$rep <- rep_id
    write.csv(path, path_file, row.names = FALSE)
  }

  out$rep <- rep_id
  out$n <- cfg$n
  out$d <- cfg$d
  out$q <- cfg$q
  out$mu_cos <- cfg$mu_cos
  out$kappa_low <- cfg$kappa_low
  out$kappa_high <- cfg$kappa_high
  out$true_mu_contrast_norm <- l2_norm(dat$params$mu[2, ] - dat$params$mu[1, ])
  out$true_eta_contrast_norm <- l2_norm(
    dat$params$kappa[2] * dat$params$mu[2, ] -
      dat$params$kappa[1] * dat$params$mu[1, ]
  )

  write.csv(out, out_file, row.names = FALSE)
  out_file
}

cat(sprintf(
  "Running eta-penalty proximal EM: reps=%d, n=%d, d=%d, q=%d, lambda_eta={%s}\n",
  cfg$n_rep,
  cfg$n,
  cfg$d,
  cfg$q,
  paste(cfg$lambda_eta_grid, collapse = ",")
))

tasks <- seq_len(cfg$n_rep)
workers <- max(1L, min(cfg$workers, cfg$n_rep))

if (workers == 1L) {
  files <- character(length(tasks))
  for (i in tasks) {
    cat(sprintf("[%03d/%03d]\n", i, cfg$n_rep))
    files[i] <- run_one(i, cfg, cell_dir)
  }
} else {
  cl <- parallel::makeCluster(workers)
  on.exit(parallel::stopCluster(cl), add = TRUE)
  parallel::clusterEvalQ(cl, source(file.path("r", "rossi_barbaro_2022_reproduction.r")))
  parallel::clusterExport(cl, setdiff(ls(), "cl"), envir = environment())
  files <- parallel::parLapplyLB(cl, tasks, function(i) run_one(i, cfg, cell_dir))
}

cell_files <- list.files(cell_dir, pattern = "^cell_[0-9]+\\.csv$", full.names = TRUE)
cell_tables <- lapply(cell_files, read.csv)
all_cols <- Reduce(union, lapply(cell_tables, names))
cell_tables <- lapply(cell_tables, function(tab) {
  missing <- setdiff(all_cols, names(tab))
  for (col in missing) tab[[col]] <- NA
  tab[, all_cols]
})
raw <- do.call(rbind, cell_tables)

raw_path <- file.path(cfg$out_dir, sprintf("%s_raw.csv", cfg$run_label))
summary_path <- file.path(cfg$out_dir, sprintf("%s_summary.csv", cfg$run_label))
write.csv(raw, raw_path, row.names = FALSE)

safe_mean <- function(x) if (all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE)
safe_se <- function(x) {
  nn <- sum(!is.na(x))
  if (nn > 1) sd(x, na.rm = TRUE) / sqrt(nn) else NA_real_
}

raw$failed <- !is.na(raw$error) & nzchar(raw$error)
raw$valid <- !raw$failed

keys <- c("method", "n", "d", "q", "mu_cos", "kappa_low", "kappa_high")
summary <- unique(raw[, keys])

fail_rate <- aggregate(as.numeric(raw$failed), raw[, keys], safe_mean)
names(fail_rate)[ncol(fail_rate)] <- "fail_rate"
valid_reps <- aggregate(as.numeric(raw$valid), raw[, keys], sum)
names(valid_reps)[ncol(valid_reps)] <- "valid_reps"
total_reps <- aggregate(raw[, "rep", drop = FALSE], raw[, keys],
                        function(x) length(unique(x)))
names(total_reps)[ncol(total_reps)] <- "total_reps"

summary <- merge(summary, fail_rate, by = keys, all.x = TRUE)
summary <- merge(summary, valid_reps, by = keys, all.x = TRUE)
summary <- merge(summary, total_reps, by = keys, all.x = TRUE)

metrics <- c(
  "ARI", "lambda_eta", "df", "BIC", "EBIC", "nnz_fraction",
  "shat", "TPR", "FPR", "precision", "F1",
  "mu_contrast_norm", "eta_contrast_norm",
  "MSE_mu", "MSE_kappa", "MSE_eta_contrast",
  "mu_topq_recall", "eta_topq_recall",
  "kappa_low_hat", "kappa_high_hat", "kappa_ratio_hat",
  "true_mu_contrast_norm", "true_eta_contrast_norm", "iter"
)

for (m in metrics) {
  agg_mean <- aggregate(raw[, m, drop = FALSE], raw[, keys], safe_mean)
  agg_se <- aggregate(raw[, m, drop = FALSE], raw[, keys], safe_se)
  names(agg_mean)[ncol(agg_mean)] <- paste0(m, "_mean")
  names(agg_se)[ncol(agg_se)] <- paste0(m, "_se")
  summary <- merge(summary, agg_mean, by = keys, all.x = TRUE)
  summary <- merge(summary, agg_se, by = keys, all.x = TRUE)
}

write.csv(summary, summary_path, row.names = FALSE)

shown <- summary[, c(
  "method", "fail_rate", "valid_reps", "total_reps",
  "ARI_mean", "lambda_eta_mean", "shat_mean", "TPR_mean",
  "FPR_mean", "precision_mean", "F1_mean",
  "mu_contrast_norm_mean", "eta_contrast_norm_mean",
  "MSE_mu_mean", "MSE_kappa_mean", "MSE_eta_contrast_mean",
  "kappa_low_hat_mean", "kappa_high_hat_mean", "kappa_ratio_hat_mean"
)]
num <- vapply(shown, is.numeric, logical(1))
shown[num] <- lapply(shown[num], round, 3)

cat("\nDone.\n")
cat("Raw: ", normalizePath(raw_path, winslash = "/"), "\n", sep = "")
cat("Summary: ", normalizePath(summary_path, winslash = "/"), "\n", sep = "")
print(shown, row.names = FALSE)
