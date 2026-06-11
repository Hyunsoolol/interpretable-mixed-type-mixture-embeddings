# ==============================================================================
# Three-method comparison under the kappa-contrast limitation scenario
# ------------------------------------------------------------------------------
# Methods compared in this file:
#   1. Rossi & Barbaro (2022) sparse vMF mixture
#   2. Proposed eta-contrast support screening + refit
#   3. Mu/kappa separate screening proxy + refit
#
# Scenario:
#   The two components have the same directional mean, but very different
#   concentration.  The discriminating signal is therefore in
#       eta_k = kappa_k * mu_k
#   rather than in mu_k alone.
#
# This file is intentionally written as a clean, readable comparison script.
# The lower-level vMF sampler and Rossi sparse-vMF EM are sourced from
# rossi_barbaro_2022_reproduction.r.
# ==============================================================================

source(file.path("r", "rossi_barbaro_2022_reproduction.r"))

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

parse_int_grid <- function(x) as.integer(strsplit(x, ",")[[1]])

cfg <- list(
  run_label = Sys.getenv("CMP_RUN_LABEL", "compare_three_methods_kappa_limit"),
  n_rep = as.integer(Sys.getenv("CMP_N_REP", "30")),
  n = as.integer(Sys.getenv("CMP_N", "1000")),
  d = as.integer(Sys.getenv("CMP_D", "100")),
  q = as.integer(Sys.getenv("CMP_Q", "10")),
  K_true = 2,
  K_fit_grid = parse_int_grid(Sys.getenv("CMP_K_GRID", "1,2,3")),
  kappa_low = as.numeric(Sys.getenv("CMP_KAPPA_LOW", "20")),
  kappa_high = as.numeric(Sys.getenv("CMP_KAPPA_HIGH", "200")),
  mu_cos = as.numeric(Sys.getenv("CMP_MU_COS", "1")),
  nstart = as.integer(Sys.getenv("CMP_NSTART", "5")),
  max_path_steps = as.integer(Sys.getenv("CMP_MAX_PATH_STEPS", "250")),
  max_active = as.integer(Sys.getenv("CMP_MAX_ACTIVE", "35")),
  workers = as.integer(Sys.getenv("CMP_WORKERS", "1")),
  base_seed = as.integer(Sys.getenv("CMP_BASE_SEED", "20260602")),
  out_dir = Sys.getenv("CMP_OUT_DIR", "results/compare_three_methods_kappa_limit_260603")
)

if (!dir.exists(cfg$out_dir)) dir.create(cfg$out_dir, recursive = TRUE)
cell_dir <- file.path(cfg$out_dir, "cells")
if (!dir.exists(cell_dir)) dir.create(cell_dir, recursive = TRUE)

# ------------------------------------------------------------------------------
# Data-generating mechanism
# ------------------------------------------------------------------------------

make_kappa_contrast_params <- function(d, q, kappa_low, kappa_high, mu_cos = 1) {
  support <- rep(FALSE, d)
  support[seq_len(q)] <- TRUE

  mu1 <- rep(0, d)
  mu1[support] <- 1 / sqrt(q)

  # mu_cos = 1 gives the clean limitation case: mu_1 and mu_2 are identical.
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
# Shared evaluation helpers
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

  # Ordering by kappa makes component labels comparable across random starts.
  ord <- order(theta$kappa)
  mu <- theta$mu[ord, , drop = FALSE]
  kappa <- theta$kappa[ord]
  mu_delta <- mu[2, ] - mu[1, ]
  eta_delta <- kappa[2] * mu[2, ] - kappa[1] * mu[1, ]

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

empty_support_metrics <- function() {
  data.frame(
    shat = NA_real_,
    TPR = NA_real_,
    FPR = NA_real_,
    precision = NA_real_,
    F1 = NA_real_
  )
}

empty_contrast_metrics <- function() {
  data.frame(
    mu_contrast_norm = NA_real_,
    eta_contrast_norm = NA_real_,
    mu_topq_recall = NA_real_,
    eta_topq_recall = NA_real_,
    kappa_low_hat = NA_real_,
    kappa_high_hat = NA_real_,
    kappa_ratio_hat = NA_real_
  )
}

make_method_row <- function(method,
                            stage = NA_character_,
                            failed = FALSE,
                            error = NA_character_,
                            ARI = NA_real_,
                            K_fit = NA_real_,
                            beta = NA_real_,
                            BIC = NA_real_,
                            EBIC = NA_real_,
                            df = NA_real_,
                            m = NA_real_,
                            nnz_fraction = NA_real_,
                            sm = empty_support_metrics(),
                            cm = empty_contrast_metrics()) {
  data.frame(
    method = method,
    stage = stage,
    failed = failed,
    error = error,
    ARI = ARI,
    K_fit = K_fit,
    beta = beta,
    BIC = BIC,
    EBIC = EBIC,
    df = df,
    m = m,
    nnz_fraction = nnz_fraction,
    sm,
    cm
  )
}

failure_stage <- function(method) {
  if (grepl("pre_refit", method)) return("pre_refit_topq")
  if (grepl("BIC_refit", method)) return("BIC_refit")
  if (identical(method, "rossi_sparse_vmf_BIC")) return("path_BIC")
  "failed"
}

failure_row <- function(method, err) {
  make_method_row(
    method = method,
    stage = failure_stage(method),
    failed = TRUE,
    error = conditionMessage(err)
  )
}

# ------------------------------------------------------------------------------
# Method 1: Rossi & Barbaro (2022) sparse vMF mixture
# ------------------------------------------------------------------------------

fit_rossi_sparse_bic <- function(X, z, cfg) {
  rows <- list()
  fits <- list()

  for (K in cfg$K_fit_grid) {
    # Rossi path: beta = 0 is dense vMF; increasing beta gives sparse prototypes.
    path <- fit_svMF_path(
      X,
      K,
      labels_true = z,
      nstart = cfg$nstart,
      max_path_steps = cfg$max_path_steps
    )

    ptab <- path$path
    idx <- which.min(ptab$BIC)
    best <- ptab[idx, , drop = FALSE]
    best$K_fit <- K

    rows[[length(rows) + 1L]] <- best
    fits[[as.character(K)]] <- path$fits[[idx]]
  }

  tab <- do.call(rbind, rows)
  best_idx <- which.min(tab$BIC)
  best <- tab[best_idx, , drop = FALSE]

  list(
    row = best,
    fit = fits[[as.character(best$K_fit)]]
  )
}

rossi_method_row <- function(X, z, support_true, cfg) {
  fit <- fit_rossi_sparse_bic(X, z, cfg)
  theta <- fit$fit

  # Rossi sparsity is on entries of mu_k.  For coordinate recovery, a coordinate
  # is active if any fitted component uses it.
  active <- colSums(abs(theta$mu) > 1e-8) > 0
  sm <- support_metrics(active, support_true)
  cm <- contrast_metrics(theta, support_true, cfg$q)

  make_method_row(
    method = "rossi_sparse_vmf_BIC",
    stage = "path_BIC",
    ARI = fit$row$ARI,
    K_fit = fit$row$K_fit,
    beta = fit$row$beta,
    BIC = fit$row$BIC,
    EBIC = fit$row$EBIC,
    df = fit$row$df,
    nnz_fraction = fit$row$nnz_fraction,
    sm = sm,
    cm = cm
  )
}

# ------------------------------------------------------------------------------
# Methods 2 and 3: score a support path, then refit vMF on selected coordinates
# ------------------------------------------------------------------------------

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
    mu = mask_and_normalize_mu(init$mu, active),
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

    # M-step under the selected support: inactive coordinates are fixed at zero.
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
  # Same effective-dimension convention as the Rossi reproduction code:
  # alpha: K - 1, kappa: K, each active mu_k: max(m - 1, 1).
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

eta_contrast_score <- function(theta) {
  ord <- order(theta$kappa)
  eta <- sweep(theta$mu[ord, , drop = FALSE], 1, theta$kappa[ord], "*")
  abs(eta[2, ] - eta[1, ])
}

mu_kappa_separate_score <- function(theta) {
  # kappa is component-level, not coordinate-level.  Therefore this separate
  # proxy can rank coordinates only through the mu contrast; kappa is still
  # estimated in the dense initialization and constrained refit.
  ord <- order(theta$kappa)
  mu <- theta$mu[ord, , drop = FALSE]
  abs(mu[2, ] - mu[1, ])
}

score_coordinates <- function(theta, score_type) {
  if (score_type == "eta") return(eta_contrast_score(theta))
  if (score_type == "mu_kappa_separate") return(mu_kappa_separate_score(theta))
  stop("Unknown score_type: ", score_type)
}

score_refit_method_name <- function(score_type) {
  if (score_type == "eta") return("eta_contrast_BIC_refit")
  if (score_type == "mu_kappa_separate") return("mu_kappa_separate_BIC_refit")
  stop("Unknown score_type: ", score_type)
}

score_prefit_method_name <- function(score_type) {
  if (score_type == "eta") return("eta_contrast_pre_refit_topq")
  if (score_type == "mu_kappa_separate") return("mu_kappa_separate_pre_refit_topq")
  stop("Unknown score_type: ", score_type)
}

score_pre_refit_row <- function(X, z, support_true, cfg, score_type, dense) {
  d <- ncol(X)
  score <- score_coordinates(dense, score_type)
  active <- rep(FALSE, d)
  active[order(score, decreasing = TRUE)[seq_len(cfg$q)]] <- TRUE

  cluster <- max.col(dense$tau, ties.method = "first")
  sm <- support_metrics(active, support_true)
  cm <- contrast_metrics(dense, support_true, cfg$q)

  # This is a screening diagnostic, not a selected sparse model.  Therefore
  # BIC/EBIC/df are left empty; the selected-model IC is reported after refit.
  make_method_row(
    method = score_prefit_method_name(score_type),
    stage = "pre_refit_topq",
    ARI = adjusted_rand_index(z, cluster),
    K_fit = 2,
    m = cfg$q,
    nnz_fraction = cfg$q / d,
    sm = sm,
    cm = cm
  )
}

fit_score_refit_bic <- function(X, z, support_true, cfg, score_type, dense = NULL) {
  n <- nrow(X)
  d <- ncol(X)
  K <- 2

  # Dense vMF estimates kappa first; the score then decides which coordinates
  # are allowed into the constrained refit.
  if (is.null(dense)) {
    dense <- fit_svMF_multistart(X, K, beta = 0, nstart = cfg$nstart)
  }
  score <- score_coordinates(dense, score_type)
  ord <- order(score, decreasing = TRUE)
  m_grid <- seq_len(min(cfg$max_active, d))

  rows <- list()
  fits <- list()

  for (m in m_grid) {
    active <- rep(FALSE, d)
    active[ord[seq_len(m)]] <- TRUE

    fit <- fit_support_constrained_vmf(X, K, active, dense)
    if (isTRUE(fit$failed)) next

    ic <- support_ic(fit$loglik, n, d, K, m)
    cluster <- max.col(fit$tau, ties.method = "first")
    sm <- support_metrics(active, support_true)

    row <- data.frame(
      m = m,
      ARI = adjusted_rand_index(z, cluster),
      loglik = fit$loglik,
      ic,
      sm
    )

    rows[[length(rows) + 1L]] <- row
    fits[[length(fits) + 1L]] <- list(fit = fit, active = active, row = row)
  }

  if (length(rows) == 0L) {
    stop("All support-constrained refits failed for score_type=", score_type)
  }

  path <- do.call(rbind, rows)
  idx <- which.min(path$BIC)
  selected <- fits[[idx]]
  cm <- contrast_metrics(selected$fit, support_true, cfg$q)

  make_method_row(
    method = score_refit_method_name(score_type),
    stage = "BIC_refit",
    ARI = selected$row$ARI,
    K_fit = K,
    BIC = selected$row$BIC,
    EBIC = selected$row$EBIC,
    df = selected$row$df,
    m = selected$row$m,
    nnz_fraction = selected$row$m / d,
    sm = selected$row[, c("shat", "TPR", "FPR", "precision", "F1")],
    cm = cm
  )
}

# ------------------------------------------------------------------------------
# One replication: same simulated data, three methods
# ------------------------------------------------------------------------------

run_one <- function(rep_id, cfg, cell_dir) {
  out_file <- file.path(cell_dir, sprintf("cell_%03d.csv", rep_id))
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

  X <- dat$X
  z <- dat$z
  support_true <- dat$params$support[1, ]

  rossi <- tryCatch(
    rossi_method_row(X, z, support_true, cfg),
    error = function(e) failure_row("rossi_sparse_vmf_BIC", e)
  )

  # Use one dense initialization for both score diagnostics so that eta and
  # mu/kappa-separate screening differ only by the coordinate score.
  dense_for_scores <- tryCatch(
    fit_svMF_multistart(X, 2, beta = 0, nstart = cfg$nstart),
    error = function(e) e
  )

  if (inherits(dense_for_scores, "error")) {
    eta_pre <- failure_row("eta_contrast_pre_refit_topq", dense_for_scores)
    eta_refit <- failure_row("eta_contrast_BIC_refit", dense_for_scores)
    mu_kappa_pre <- failure_row("mu_kappa_separate_pre_refit_topq", dense_for_scores)
    mu_kappa_refit <- failure_row("mu_kappa_separate_BIC_refit", dense_for_scores)
  } else {
    eta_pre <- tryCatch(
      score_pre_refit_row(X, z, support_true, cfg, score_type = "eta",
                          dense = dense_for_scores),
      error = function(e) failure_row("eta_contrast_pre_refit_topq", e)
    )

    eta_refit <- tryCatch(
      fit_score_refit_bic(X, z, support_true, cfg, score_type = "eta",
                          dense = dense_for_scores),
      error = function(e) failure_row("eta_contrast_BIC_refit", e)
    )

    mu_kappa_pre <- tryCatch(
      score_pre_refit_row(X, z, support_true, cfg, score_type = "mu_kappa_separate",
                          dense = dense_for_scores),
      error = function(e) failure_row("mu_kappa_separate_pre_refit_topq", e)
    )

    mu_kappa_refit <- tryCatch(
      fit_score_refit_bic(X, z, support_true, cfg,
                          score_type = "mu_kappa_separate",
                          dense = dense_for_scores),
      error = function(e) failure_row("mu_kappa_separate_BIC_refit", e)
    )
  }

  rows <- rbind(rossi, eta_pre, eta_refit, mu_kappa_pre, mu_kappa_refit)
  rows$rep <- rep_id
  rows$n <- cfg$n
  rows$d <- cfg$d
  rows$q <- cfg$q
  rows$mu_cos <- cfg$mu_cos
  rows$kappa_low <- cfg$kappa_low
  rows$kappa_high <- cfg$kappa_high
  rows$true_mu_contrast_norm <- l2_norm(dat$params$mu[2, ] - dat$params$mu[1, ])
  rows$true_eta_contrast_norm <- l2_norm(
    dat$params$kappa[2] * dat$params$mu[2, ] -
      dat$params$kappa[1] * dat$params$mu[1, ]
  )

  rows <- rows[, c(
    "rep", "method", "stage", "failed", "error",
    "n", "d", "q", "mu_cos", "kappa_low", "kappa_high",
    "ARI", "K_fit", "beta", "BIC", "EBIC", "df", "m",
    "nnz_fraction", "shat", "TPR", "FPR", "precision", "F1",
    "mu_contrast_norm", "eta_contrast_norm",
    "mu_topq_recall", "eta_topq_recall",
    "kappa_low_hat", "kappa_high_hat", "kappa_ratio_hat",
    "true_mu_contrast_norm", "true_eta_contrast_norm"
  )]

  write.csv(rows, out_file, row.names = FALSE)
  out_file
}

# ------------------------------------------------------------------------------
# Run all replications and summarize
# ------------------------------------------------------------------------------

cat(sprintf(
  "Running three-method comparison: reps=%d, n=%d, d=%d, q=%d, kappa=(%.1f, %.1f), mu_cos=%.3f\n",
  cfg$n_rep, cfg$n, cfg$d, cfg$q, cfg$kappa_low, cfg$kappa_high, cfg$mu_cos
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
raw <- do.call(rbind, lapply(cell_files, read.csv))

raw_path <- file.path(cfg$out_dir, sprintf("%s_raw.csv", cfg$run_label))
summary_path <- file.path(cfg$out_dir, sprintf("%s_summary.csv", cfg$run_label))
write.csv(raw, raw_path, row.names = FALSE)

safe_mean <- function(x) if (all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE)
safe_se <- function(x) {
  nn <- sum(!is.na(x))
  if (nn > 1) sd(x, na.rm = TRUE) / sqrt(nn) else NA_real_
}

keys <- c("method", "stage", "n", "d", "q", "mu_cos", "kappa_low", "kappa_high")
raw$failed_num <- as.numeric(raw$failed)
raw$valid_num <- as.numeric(!raw$failed)

summary <- unique(raw[, keys])

fail_rate <- aggregate(raw[, "failed_num", drop = FALSE], raw[, keys], safe_mean)
names(fail_rate)[ncol(fail_rate)] <- "fail_rate"
valid_reps <- aggregate(raw[, "valid_num", drop = FALSE], raw[, keys], sum)
names(valid_reps)[ncol(valid_reps)] <- "valid_reps"
total_reps <- aggregate(raw[, "rep", drop = FALSE], raw[, keys],
                        function(x) length(unique(x)))
names(total_reps)[ncol(total_reps)] <- "total_reps"

summary <- merge(summary, fail_rate, by = keys, all.x = TRUE)
summary <- merge(summary, valid_reps, by = keys, all.x = TRUE)
summary <- merge(summary, total_reps, by = keys, all.x = TRUE)

metrics <- c(
  "ARI", "K_fit", "beta", "BIC", "EBIC", "df", "m",
  "nnz_fraction", "shat", "TPR", "FPR", "precision", "F1",
  "mu_contrast_norm", "eta_contrast_norm",
  "mu_topq_recall", "eta_topq_recall",
  "kappa_low_hat", "kappa_high_hat", "kappa_ratio_hat",
  "true_mu_contrast_norm", "true_eta_contrast_norm"
)

for (m in metrics) {
  agg_mean <- aggregate(raw[, m, drop = FALSE], raw[, keys], safe_mean)
  agg_se <- aggregate(raw[, m, drop = FALSE], raw[, keys], safe_se)
  names(agg_mean)[ncol(agg_mean)] <- paste0(m, "_mean")
  names(agg_se)[ncol(agg_se)] <- paste0(m, "_se")
  summary <- merge(summary, agg_mean, by = keys, all.x = TRUE)
  summary <- merge(summary, agg_se, by = keys, all.x = TRUE)
}

summary <- summary[order(summary$method), ]
write.csv(summary, summary_path, row.names = FALSE)

shown <- summary[, c(
  "method", "stage", "fail_rate", "valid_reps", "total_reps",
  "ARI_mean", "K_fit_mean", "m_mean", "shat_mean",
  "TPR_mean", "FPR_mean", "precision_mean", "F1_mean",
  "eta_contrast_norm_mean", "kappa_ratio_hat_mean"
)]
num <- vapply(shown, is.numeric, logical(1))
shown[num] <- lapply(shown[num], round, 3)

cat("\nDone.\n")
cat("Raw: ", normalizePath(raw_path, winslash = "/"), "\n", sep = "")
cat("Summary: ", normalizePath(summary_path, winslash = "/"), "\n", sep = "")
print(shown, row.names = FALSE)
