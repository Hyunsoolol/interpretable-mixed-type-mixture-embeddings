#!/usr/bin/env Rscript

# Classic3 real-data pilot for sparse vMF and centered-Eta methods.
# Supplied class labels are used only for external ARI/NMI evaluation.

options(stringsAsFactors = FALSE)

if (!requireNamespace("Matrix", quietly = TRUE)) {
  stop("The Matrix package is required.")
}
suppressPackageStartupMessages(library(Matrix))

parse_bool_env <- function(name, default = "0") {
  tolower(Sys.getenv(name, default)) %in% c("1", "true", "t", "yes", "y", "on")
}

# The current C++ E-step accepts dense matrices. Classic3 is retained as a
# 98.9% sparse dgCMatrix, so this diagnostic intentionally uses Matrix algebra.
if (parse_bool_env("CLASSIC3_COMPARE_USE_RCPP", "0")) {
  warning("The current Rcpp E-step is dense-only; using R/Matrix for sparse text input.")
}
Sys.setenv(USE_RCPP_HELPERS = "0")

source_no_bom <- function(path, envir = .GlobalEnv) {
  txt <- readLines(path, warn = FALSE, encoding = "UTF-8")
  if (length(txt) > 0L) txt[1L] <- sub("^\ufeff", "", txt[1L])
  eval(parse(text = txt), envir = envir)
}

source_method_helpers <- function() {
  source_no_bom(file.path("r", "methods", "rossi_barbaro_2022_reproduction.r"))
  helper_path <- file.path("r", "methods", "rb2022_k4_pilot_compare_run.r")
  lines <- readLines(helper_path, warn = FALSE, encoding = "UTF-8")
  if (length(lines) > 0L) lines[1L] <- sub("^\ufeff", "", lines[1L])
  boundary <- grep("fit_rossi_pair <-", lines, fixed = TRUE)[1L]
  if (is.na(boundary)) stop("Could not locate helper boundary in ", helper_path)
  eval(parse(text = lines[seq_len(boundary - 1L)]), envir = .GlobalEnv)
}

source_method_helpers()

# R's besselI loses precision when both the dimension and Bessel order are
# large. Use the second-order uniform asymptotic expansion for I_nu(nu * z)
# in this Classic3 runner; the core implementation remains unchanged.
log_vmf_const_one <- function(kappa, d) {
  if (kappa < 1e-8) {
    return(lgamma(d / 2) - (d / 2) * log(2 * pi))
  }
  nu <- d / 2 - 1
  if (nu >= 50) {
    z <- kappa / nu
    root <- sqrt(1 + z * z)
    t <- 1 / root
    eta <- root + log(z / (1 + root))
    u1 <- (3 * t - 5 * t^3) / 24
    u2 <- (81 * t^2 - 462 * t^4 + 385 * t^6) / 1152
    correction <- 1 + u1 / nu + u2 / (nu^2)
    log_bessel <- -0.5 * log(2 * pi * nu) -
      0.25 * log1p(z * z) + nu * eta + log(correction)
    return(nu * log(kappa) - (d / 2) * log(2 * pi) - log_bessel)
  }
  scaled_bessel <- besselI(kappa, nu, expon.scaled = TRUE)
  if (!is.finite(scaled_bessel) || scaled_bessel <= 0) {
    return(((d - 1) / 2) * (log(kappa) - log(2 * pi)) - kappa)
  }
  nu * log(kappa) - (d / 2) * log(2 * pi) -
    (log(scaled_bessel) + kappa)
}

log_vmf_const <- function(kappa, d) {
  vapply(kappa, log_vmf_const_one, numeric(1), d = d)
}

getenv_num <- function(name, default) {
  value <- Sys.getenv(name, unset = NA_character_)
  if (is.na(value) || !nzchar(value)) return(default)
  as.numeric(value)
}

getenv_int <- function(name, default) as.integer(getenv_num(name, default))

cfg <- list(
  dataset_label = Sys.getenv("CLASSIC3_COMPARE_DATASET_LABEL", "Classic3"),
  output_prefix = Sys.getenv("CLASSIC3_COMPARE_OUTPUT_PREFIX", "classic3"),
  strict_classic3_dimensions = parse_bool_env(
    "CLASSIC3_COMPARE_STRICT_DIMENSIONS", "1"
  ),
  allow_K_mismatch = parse_bool_env(
    "CLASSIC3_COMPARE_ALLOW_K_MISMATCH", "0"
  ),
  run_label = Sys.getenv(
    "CLASSIC3_COMPARE_LABEL",
    "classic3_vmf_compare_pilot_nstart10_path300_260711"
  ),
  data_path = Sys.getenv(
    "CLASSIC3_COMPARE_DATA",
    file.path("data", "classic3", "processed", "classic3_rcoclust_tfidf_l2.rds")
  ),
  out_dir = Sys.getenv(
    "CLASSIC3_COMPARE_OUT_DIR",
    file.path("results", "classic3_vmf_compare_pilot_nstart10_path300_260711")
  ),
  K = getenv_int("CLASSIC3_COMPARE_K", 3L),
  nstart = getenv_int("CLASSIC3_COMPARE_NSTART", 10L),
  max_iter = getenv_int("CLASSIC3_COMPARE_MAX_ITER", 100L),
  max_path_steps = getenv_int("CLASSIC3_COMPARE_MAX_PATH", 300L),
  tol = getenv_num("CLASSIC3_COMPARE_TOL", 1e-7),
  min_rel_lambda = getenv_num("CLASSIC3_COMPARE_MIN_REL_LAMBDA", 0.05),
  lambda_eps = getenv_num("CLASSIC3_COMPARE_LAMBDA_EPS", 1e-8),
  zero_eps = getenv_num("CLASSIC3_COMPARE_ZERO_EPS", 1e-8),
  adaptive_gamma = getenv_num("CLASSIC3_COMPARE_ADAPTIVE_GAMMA", 1),
  adaptive_eps = getenv_num("CLASSIC3_COMPARE_ADAPTIVE_EPS", 1e-6),
  ebic_gamma = getenv_num("CLASSIC3_COMPARE_EBIC_GAMMA", 0.5),
  base_seed = getenv_int("CLASSIC3_COMPARE_BASE_SEED", 20260711L),
  save_paths = parse_bool_env("CLASSIC3_COMPARE_SAVE_PATHS", "0"),
  verbose = parse_bool_env("CLASSIC3_COMPARE_VERBOSE", "1")
)

if (!file.exists(cfg$data_path)) stop("Prepared text data not found: ", cfg$data_path)
dir.create(cfg$out_dir, recursive = TRUE, showWarnings = FALSE)

payload <- readRDS(cfg$data_path)
X <- payload$X
y <- as.integer(payload$y)
n <- nrow(X)
d <- ncol(X)

input_matrix_class <- paste(class(X), collapse = "/")
if (!inherits(X, "sparseMatrix")) {
  if (!is.matrix(X) || !is.numeric(X)) {
    stop("Comparison input must be a numeric matrix or Matrix sparse input.")
  }
  X <- Matrix::Matrix(X, sparse = TRUE)
}
if (!cfg$allow_K_mismatch && cfg$K != length(unique(y))) {
  stop("K does not match the number of supplied classes.")
}
if (cfg$strict_classic3_dimensions && (n != 3891L || d != 4303L || cfg$K != 3L)) {
  stop("Unexpected Classic3 dimensions: n=", n, ", d=", d)
}
row_norm_error <- max(abs(sqrt(Matrix::rowSums(X * X)) - 1))
if (!is.finite(row_norm_error) || row_norm_error > 1e-8) {
  stop("Input rows are not L2 normalized; max error=", row_norm_error)
}

# Sparse-compatible initialization. Only the K sampled prototype rows and the
# K-by-d sufficient statistic are converted to ordinary matrices.
init_vmf_mixture <- function(X, K, kappa_cap = 1e6) {
  n_local <- nrow(X)
  d_local <- ncol(X)
  idx <- sample.int(n_local, K, replace = FALSE)
  mu <- as.matrix(X[idx, , drop = FALSE])
  sim <- as.matrix(X %*% t(mu))
  cl <- max.col(sim, ties.method = "random")
  tau <- matrix(0, nrow = n_local, ncol = K)
  tau[cbind(seq_len(n_local), cl)] <- 1
  Nk <- colSums(tau)
  if (any(Nk == 0)) stop("Empty component in sparse initialization.")
  r <- as.matrix(t(tau) %*% X)
  kappa <- numeric(K)
  for (k in seq_len(K)) {
    rho <- as.numeric(crossprod(mu[k, ], r[k, ])) / Nk[k]
    kappa[k] <- estimate_kappa(rho, d_local, kappa_cap)
  }
  list(alpha = Nk / n_local, mu = mu, kappa = kappa)
}

unpenalized_eta_mstep <- function(X, tau, kappa_cap = 1e6) {
  n_local <- nrow(X)
  K_local <- ncol(tau)
  Nk <- colSums(tau)
  if (any(Nk < 1e-8)) stop("Empty component in eta M-step.")
  r <- as.matrix(t(tau) %*% X)
  mu <- normalize_rows(r)
  kappa <- numeric(K_local)
  for (k in seq_len(K_local)) {
    rho <- l2_norm(r[k, ]) / Nk[k]
    kappa[k] <- estimate_kappa(rho, ncol(X), kappa_cap)
  }
  list(alpha = pmax(Nk / n_local, 1e-12), eta = sweep(mu, 1, kappa, "*"))
}

fit_support_refit_sparse <- function(X, K, active, init, max_iter = 100,
                                     tol = 1e-7, kappa_cap = 1e6) {
  if (!any(active)) return(NULL)
  n_local <- nrow(X)
  d_local <- ncol(X)
  theta <- init
  theta$mu <- mask_and_normalize_mu(theta$mu, active)
  theta$alpha <- pmax(theta$alpha, 1e-12)
  theta$alpha <- theta$alpha / sum(theta$alpha)
  if (length(theta$kappa) == 1L) theta$kappa <- rep(theta$kappa, K)
  prev <- -Inf
  last_e <- NULL
  for (iter in seq_len(max_iter)) {
    e <- e_step_vmf(X, theta)
    tau <- e$tau
    Nk <- colSums(tau)
    if (any(Nk < 1e-8)) return(NULL)
    r <- as.matrix(t(tau) %*% X)
    alpha_new <- pmax(Nk / n_local, 1e-12)
    alpha_new <- alpha_new / sum(alpha_new)
    mu_new <- matrix(0, nrow = K, ncol = d_local)
    kappa_new <- numeric(K)
    for (k in seq_len(K)) {
      rk <- r[k, ]
      rk[!active] <- 0
      mu_new[k, ] <- rk / max(l2_norm(rk), 1e-12)
      rho <- as.numeric(crossprod(mu_new[k, ], r[k, ])) / max(Nk[k], 1e-12)
      kappa_new[k] <- estimate_kappa(rho, d_local, kappa_cap)
    }
    theta <- list(alpha = alpha_new, mu = mu_new, kappa = kappa_new)
    e_new <- e_step_vmf(X, theta)
    last_e <- e_new
    if (is.finite(prev) &&
        abs(e_new$loglik - prev) / max(1, abs(prev)) < tol) {
      return(c(theta, list(
        failed = FALSE, converged = TRUE, iter = iter,
        loglik = e_new$loglik, pen_loglik = NA_real_, tau = e_new$tau
      )))
    }
    prev <- e_new$loglik
  }
  if (is.null(last_e)) last_e <- e_step_vmf(X, theta)
  c(theta, list(
    failed = FALSE, converged = FALSE, iter = max_iter,
    loglik = last_e$loglik, pen_loglik = NA_real_, tau = last_e$tau
  ))
}

normalized_mutual_information <- function(truth, cluster) {
  tab <- table(truth, cluster)
  total <- sum(tab)
  pij <- tab / total
  pi <- rowSums(pij)
  pj <- colSums(pij)
  expected <- outer(pi, pj)
  keep <- pij > 0 & expected > 0
  mi <- sum(pij[keep] * log(pij[keep] / expected[keep]))
  h_truth <- -sum(pi[pi > 0] * log(pi[pi > 0]))
  h_cluster <- -sum(pj[pj > 0] * log(pj[pj > 0]))
  denom <- h_truth + h_cluster
  if (denom <= 0) return(0)
  2 * mi / denom
}

group_weights_from_norm <- function(norms, gamma, eps) {
  weights <- (norms + eps)^(-gamma)
  center <- median(weights[is.finite(weights) & weights > 0])
  if (is.finite(center) && center > 0) weights <- weights / center
  weights
}

start_seeds <- cfg$base_seed + seq_len(cfg$nstart) * 1009L

vmf_init_from_cluster <- function(cluster) {
  tau <- matrix(0, nrow = n, ncol = cfg$K)
  tau[cbind(seq_len(n), cluster)] <- 1
  Nk <- colSums(tau)
  if (any(Nk == 0)) stop("Empty spherical k-means cluster.")
  r <- as.matrix(t(tau) %*% X)
  mu <- normalize_rows(r)
  kappa <- vapply(
    seq_len(cfg$K),
    function(k) estimate_kappa(l2_norm(r[k, ]) / Nk[k], d),
    numeric(1)
  )
  list(alpha = Nk / n, mu = mu, kappa = kappa)
}

fit_vmf_common_starts <- function(shared_kappa, spherical_starts) {
  best <- NULL
  best_obj <- -Inf
  best_start <- NA_integer_
  for (s in seq_along(spherical_starts)) {
    fit <- tryCatch({
      init <- vmf_init_from_cluster(spherical_starts[[s]]$cluster)
      fit_svMF_em(
        X, cfg$K, beta = 0, init = init,
        shared_kappa = shared_kappa,
        max_iter = cfg$max_iter, tol = cfg$tol,
        zero_eps = cfg$zero_eps
      )
    }, error = function(e) NULL)
    if (is.null(fit) || isTRUE(fit$failed) || !is.finite(fit$loglik)) next
    if (fit$loglik > best_obj) {
      best <- fit
      best_obj <- fit$loglik
      best_start <- s
    }
  }
  if (is.null(best)) stop("All vMF initializations failed.")
  best$best_start <- best_start
  best
}

make_sparse_spherical_starts <- function() {
  best <- NULL
  best_obj <- -Inf
  starts <- vector("list", length(start_seeds))
  for (s in seq_along(start_seeds)) {
    set.seed(start_seeds[s])
    mu <- as.matrix(X[sample.int(n, cfg$K, replace = FALSE), , drop = FALSE])
    cl <- rep(NA_integer_, n)
    for (iter in seq_len(cfg$max_iter)) {
      sim <- as.matrix(X %*% t(mu))
      cl_new <- max.col(sim, ties.method = "random")
      mu_new <- matrix(0, nrow = cfg$K, ncol = d)
      for (k in seq_len(cfg$K)) {
        idx <- which(cl_new == k)
        if (length(idx) == 0L) {
          mu_new[k, ] <- as.numeric(X[sample.int(n, 1L), ])
        } else {
          mu_new[k, ] <- Matrix::colMeans(X[idx, , drop = FALSE])
        }
      }
      mu_new <- normalize_rows(mu_new)
      if ((!anyNA(cl) && all(cl_new == cl)) || max(abs(mu_new - mu)) < 1e-8) {
        mu <- mu_new
        cl <- cl_new
        break
      }
      mu <- mu_new
      cl <- cl_new
    }
    sim <- as.matrix(X %*% t(mu))
    cl <- max.col(sim, ties.method = "first")
    obj <- sum(sim[cbind(seq_len(n), cl)])
    starts[[s]] <- list(
      cluster = cl, mu = mu, objective = obj, iter = iter, start = s
    )
    if (obj > best_obj) {
      best <- starts[[s]]
      best$best_start <- s
      best_obj <- obj
    }
  }
  list(best = best, starts = starts)
}

fit_m_l_path <- function(dense_fit) {
  beta <- 0
  fit <- dense_fit
  fits <- list(fit)
  path <- evaluate_fit(
    fit, X, beta, labels_true = y, shared_kappa = TRUE,
    gamma = cfg$ebic_gamma, zero_eps = cfg$zero_eps
  )
  path$step <- 1L
  if (cfg$max_path_steps >= 2L) {
    for (step in 2:cfg$max_path_steps) {
      e <- e_step_vmf(X, fit)
      r <- as.matrix(t(e$tau) %*% X)
      kappa <- fit$kappa
      if (length(kappa) == 1L) kappa <- rep(kappa, cfg$K)
      margin <- matrix(0, nrow = cfg$K, ncol = d)
      for (k in seq_len(cfg$K)) margin[k, ] <- kappa[k] * abs(r[k, ]) - beta
      candidates <- margin[margin > cfg$lambda_eps]
      if (length(candidates) == 0L) break
      beta_next <- beta + min(candidates)
      if (beta > 0) beta_next <- max(beta_next, beta * (1 + cfg$min_rel_lambda))
      if (!is.finite(beta_next) || beta_next <= beta) break
      fit_next <- tryCatch(
        fit_svMF_em(
          X, cfg$K, beta = beta_next, init = fit,
          shared_kappa = TRUE, max_iter = cfg$max_iter,
          tol = cfg$tol, zero_eps = cfg$zero_eps
        ),
        error = function(e) NULL
      )
      if (is.null(fit_next) || isTRUE(fit_next$failed)) break
      fit_next$mu[abs(fit_next$mu) < cfg$zero_eps] <- 0
      fit_next$mu <- normalize_rows(fit_next$mu)
      e_next <- e_step_vmf(X, fit_next)
      fit_next$loglik <- e_next$loglik
      fit_next$pen_loglik <- fit_next$loglik - beta_next * sum(abs(fit_next$mu))
      fit_next$tau <- e_next$tau
      fit <- fit_next
      beta <- beta_next
      fits[[length(fits) + 1L]] <- fit
      one <- evaluate_fit(
        fit, X, beta, labels_true = y, shared_kappa = TRUE,
        gamma = cfg$ebic_gamma, zero_eps = cfg$zero_eps
      )
      one$step <- step
      path <- rbind(path, one)
      if (all(rowSums(abs(fit$mu) > cfg$zero_eps) <= 1L)) break
    }
  }
  best_idx <- which.min(path$BIC)
  list(path = path, fits = fits, best_idx = best_idx, best_fit = fits[[best_idx]])
}

fit_eta_path <- function(init, weights, method) {
  lambda <- 0
  fit <- fit_eta_centered_em(
    X, cfg$K, lambda_eta = lambda, init = init,
    max_iter = cfg$max_iter, tol = cfg$tol,
    adaptive_weights = weights
  )
  rows <- list()
  fits <- list()
  add_row <- function(fit, lambda, step) {
    active <- active_eta_centered(fit, cfg$zero_eps)
    cluster <- max.col(fit$tau, ties.method = "first")
    ic <- eta_centered_ic(fit, n, d, fit$loglik, gamma = cfg$ebic_gamma)
    data.frame(
      method = method,
      step = step,
      lambda_eta = lambda,
      selected_q = sum(active),
      ARI = adjusted_rand_index(y, cluster),
      NMI = normalized_mutual_information(y, cluster),
      loglik = fit$loglik,
      pen_loglik = fit$pen_loglik,
      converged = fit$converged,
      iter = fit$iter,
      kappa_ratio = max(fit$kappa) / max(min(fit$kappa), 1e-12),
      ic,
      row.names = NULL
    )
  }
  rows[[1L]] <- add_row(fit, lambda, 1L)
  fits[[1L]] <- fit
  if (cfg$max_path_steps >= 2L) {
    for (step in 2:cfg$max_path_steps) {
      e <- e_step_vmf(X, fit)
      mstep <- unpenalized_eta_mstep(X, e$tau)
      centered <- center_eta(mstep$eta)
      thresholds <- sqrt(colSums(centered * centered)) / pmax(weights, 1e-12)
      candidates <- thresholds[thresholds > lambda + cfg$lambda_eps]
      if (length(candidates) == 0L) break
      lambda_next <- min(candidates)
      if (lambda > 0) {
        lambda_next <- max(lambda_next, lambda * (1 + cfg$min_rel_lambda))
      }
      if (!is.finite(lambda_next) || lambda_next <= lambda) break
      fit_next <- tryCatch(
        fit_eta_centered_em(
          X, cfg$K, lambda_eta = lambda_next, init = fit,
          max_iter = cfg$max_iter, tol = cfg$tol,
          adaptive_weights = weights
        ),
        error = function(e) NULL
      )
      if (is.null(fit_next) || isTRUE(fit_next$failed)) break
      fit <- fit_next
      lambda <- lambda_next
      rows[[length(rows) + 1L]] <- add_row(fit, lambda, step)
      fits[[length(fits) + 1L]] <- fit
      if (sum(active_eta_centered(fit, cfg$zero_eps)) <= 1L) break
    }
  }
  path <- do.call(rbind, rows)
  best_idx <- which.min(path$BIC)
  list(
    path = path, fits = fits, best_idx = best_idx,
    best_fit = fits[[best_idx]],
    best_active = active_eta_centered(fits[[best_idx]], cfg$zero_eps),
    weights = weights
  )
}

cluster_metrics <- function(cluster) {
  c(
    ARI = adjusted_rand_index(y, cluster),
    NMI = normalized_mutual_information(y, cluster)
  )
}

equalize_kappa_for_scoring <- function(fit) {
  theta <- list(
    alpha = fit$alpha,
    mu = fit$mu,
    kappa = rep(mean(fit$kappa), length(fit$kappa))
  )
  e <- e_step_vmf(X, theta)
  c(theta, list(
    failed = FALSE,
    converged = fit$converged,
    iter = 0L,
    loglik = e$loglik,
    pen_loglik = NA_real_,
    tau = e$tau
  ))
}

fit_row <- function(method, support_target, stage, fit, active,
                    criterion = NA_character_, beta = NA_real_,
                    lambda_eta = NA_real_, selection_df = NA_real_,
                    selection_bic = NA_real_, path_candidates = NA_integer_,
                    selected_at_path_end = NA, elapsed_sec = NA_real_) {
  cluster <- max.col(fit$tau, ties.method = "first")
  metric <- cluster_metrics(cluster)
  prototype_entries_applicable <- support_target %in%
    c("prototype union", "none (all coordinates)")
  data.frame(
    method = method,
    support_target = support_target,
    stage = stage,
    criterion = criterion,
    ARI = unname(metric["ARI"]),
    NMI = unname(metric["NMI"]),
    selected_q = sum(active),
    coordinate_sparsity = 1 - sum(active) / d,
    prototype_selected_entries = if (prototype_entries_applicable) {
      sum(abs(fit$mu) > cfg$zero_eps)
    } else {
      NA_integer_
    },
    prototype_entry_sparsity = if (prototype_entries_applicable) {
      mean(abs(fit$mu) <= cfg$zero_eps)
    } else {
      NA_real_
    },
    loglik = fit$loglik,
    beta = beta,
    lambda_eta = lambda_eta,
    selection_df = selection_df,
    selection_BIC = selection_bic,
    converged = fit$converged,
    iter = fit$iter,
    kappa_min = min(fit$kappa),
    kappa_median = median(fit$kappa),
    kappa_max = max(fit$kappa),
    kappa_ratio = max(fit$kappa) / max(min(fit$kappa), 1e-12),
    cluster_size = paste(
      as.integer(table(factor(cluster, levels = seq_len(cfg$K)))),
      collapse = ";"
    ),
    path_candidates = path_candidates,
    selected_at_path_end = selected_at_path_end,
    elapsed_sec = elapsed_sec,
    stringsAsFactors = FALSE
  )
}

if (cfg$verbose) {
  cat(cfg$dataset_label, "vMF comparison pilot\n")
  cat("representation:", payload$representation, "\n")
  cat("n=", n, " d=", d, " K=", cfg$K, "\n", sep = "")
  cat("nstart=", cfg$nstart, " max_iter=", cfg$max_iter,
      " max_path=", cfg$max_path_steps, "\n", sep = "")
}

timings <- list()

tic <- proc.time()[["elapsed"]]
skm_collection <- make_sparse_spherical_starts()
skm <- skm_collection$best
timings$skm <- proc.time()[["elapsed"]] - tic

tic <- proc.time()[["elapsed"]]
dense_shared <- fit_vmf_common_starts(
  shared_kappa = TRUE,
  spherical_starts = skm_collection$starts
)
timings$dense_shared <- proc.time()[["elapsed"]] - tic

tic <- proc.time()[["elapsed"]]
m_l <- fit_m_l_path(dense_shared)
timings$m_l <- proc.time()[["elapsed"]] - tic

tic <- proc.time()[["elapsed"]]
dense_free <- fit_vmf_common_starts(
  shared_kappa = FALSE,
  spherical_starts = skm_collection$starts
)
timings$dense_free <- proc.time()[["elapsed"]] - tic

eta_dense <- fit_eta_centered_em(
  X, cfg$K, lambda_eta = 0, init = dense_free,
  max_iter = cfg$max_iter, tol = cfg$tol,
  adaptive_weights = rep(1, d)
)
eta_norm0 <- sqrt(colSums(center_eta(eta_matrix(eta_dense))^2))
adaptive_weights <- group_weights_from_norm(
  eta_norm0, cfg$adaptive_gamma, cfg$adaptive_eps
)

tic <- proc.time()[["elapsed"]]
ecgl <- fit_eta_path(eta_dense, rep(1, d), "E-CGL")
timings$ecgl <- proc.time()[["elapsed"]] - tic

tic <- proc.time()[["elapsed"]]
ecagl <- fit_eta_path(eta_dense, adaptive_weights, "E-CAGL")
timings$ecagl <- proc.time()[["elapsed"]] - tic

ecgl_refit <- fit_support_refit_sparse(
  X, cfg$K, ecgl$best_active, ecgl$best_fit,
  max_iter = cfg$max_iter, tol = cfg$tol
)
ecagl_refit <- fit_support_refit_sparse(
  X, cfg$K, ecagl$best_active, ecagl$best_fit,
  max_iter = cfg$max_iter, tol = cfg$tol
)
ecgl_equalized_score <- equalize_kappa_for_scoring(ecgl$best_fit)
ecagl_equalized_score <- equalize_kappa_for_scoring(ecagl$best_fit)

m_idx <- m_l$best_idx
m_fit <- m_l$best_fit
m_active <- active_mu_coord(m_fit, cfg$zero_eps)
m_ic <- m_l$path[m_idx, , drop = FALSE]
ecgl_ic <- ecgl$path[ecgl$best_idx, , drop = FALSE]
ecagl_ic <- ecagl$path[ecagl$best_idx, , drop = FALSE]

skm_metric <- cluster_metrics(skm$cluster)
rows <- list(
  data.frame(
    method = "Spherical k-means",
    support_target = "none (all coordinates)",
    stage = "clustering baseline",
    criterion = "cosine objective",
    ARI = unname(skm_metric["ARI"]),
    NMI = unname(skm_metric["NMI"]),
    selected_q = d,
    coordinate_sparsity = 0,
    prototype_selected_entries = NA_integer_,
    prototype_entry_sparsity = NA_real_,
    loglik = NA_real_, beta = NA_real_, lambda_eta = NA_real_,
    selection_df = NA_real_, selection_BIC = NA_real_,
    converged = NA, iter = skm$iter,
    kappa_min = NA_real_, kappa_median = NA_real_, kappa_max = NA_real_,
    kappa_ratio = NA_real_,
    cluster_size = paste(
      as.integer(table(factor(skm$cluster, levels = seq_len(cfg$K)))),
      collapse = ";"
    ),
    path_candidates = NA_integer_, selected_at_path_end = NA,
    elapsed_sec = timings$skm,
    stringsAsFactors = FALSE
  ),
  fit_row(
    "Dense vMF shared-kappa", "none (all coordinates)", "dense",
    dense_shared, rep(TRUE, d), criterion = "unpenalized likelihood",
    path_candidates = 1L, selected_at_path_end = FALSE,
    elapsed_sec = timings$dense_shared
  ),
  fit_row(
    "Dense vMF free-kappa", "none (all coordinates)", "dense",
    dense_free, rep(TRUE, d), criterion = "unpenalized likelihood",
    path_candidates = 1L, selected_at_path_end = FALSE,
    elapsed_sec = timings$dense_free
  ),
  fit_row(
    "M-L", "prototype union", "penalized",
    m_fit, m_active, criterion = "BIC",
    beta = m_ic$beta, selection_df = m_ic$df,
    selection_bic = m_ic$BIC, path_candidates = nrow(m_l$path),
    selected_at_path_end = m_idx == nrow(m_l$path),
    elapsed_sec = timings$m_l
  ),
  fit_row(
    "E-CGL", "posterior decision", "penalized",
    ecgl$best_fit, ecgl$best_active, criterion = "BIC before refit",
    lambda_eta = ecgl_ic$lambda_eta, selection_df = ecgl_ic$df,
    selection_bic = ecgl_ic$BIC, path_candidates = nrow(ecgl$path),
    selected_at_path_end = ecgl$best_idx == nrow(ecgl$path),
    elapsed_sec = timings$ecgl
  ),
  fit_row(
    "E-CAGL", "posterior decision", "penalized",
    ecagl$best_fit, ecagl$best_active, criterion = "BIC before refit",
    lambda_eta = ecagl_ic$lambda_eta, selection_df = ecagl_ic$df,
    selection_bic = ecagl_ic$BIC, path_candidates = nrow(ecagl$path),
    selected_at_path_end = ecagl$best_idx == nrow(ecagl$path),
    elapsed_sec = timings$ecagl
  ),
  fit_row(
    "E-CGL", "posterior decision", "posthoc shared-kappa score",
    ecgl_equalized_score, ecgl$best_active,
    criterion = "BIC fit; kappa equalized post hoc",
    lambda_eta = ecgl_ic$lambda_eta, selection_df = ecgl_ic$df,
    selection_bic = ecgl_ic$BIC, path_candidates = nrow(ecgl$path),
    selected_at_path_end = ecgl$best_idx == nrow(ecgl$path),
    elapsed_sec = timings$ecgl
  ),
  fit_row(
    "E-CAGL", "posterior decision", "posthoc shared-kappa score",
    ecagl_equalized_score, ecagl$best_active,
    criterion = "BIC fit; kappa equalized post hoc",
    lambda_eta = ecagl_ic$lambda_eta, selection_df = ecagl_ic$df,
    selection_bic = ecagl_ic$BIC, path_candidates = nrow(ecagl$path),
    selected_at_path_end = ecagl$best_idx == nrow(ecagl$path),
    elapsed_sec = timings$ecagl
  )
)

if (!is.null(ecgl_refit)) {
  rows[[length(rows) + 1L]] <- fit_row(
    "E-CGL", "posterior decision", "refit",
    ecgl_refit, ecgl$best_active, criterion = "BIC before refit",
    lambda_eta = ecgl_ic$lambda_eta, selection_df = ecgl_ic$df,
    selection_bic = ecgl_ic$BIC, path_candidates = nrow(ecgl$path),
    selected_at_path_end = ecgl$best_idx == nrow(ecgl$path),
    elapsed_sec = timings$ecgl
  )
}
if (!is.null(ecagl_refit)) {
  rows[[length(rows) + 1L]] <- fit_row(
    "E-CAGL", "posterior decision", "refit",
    ecagl_refit, ecagl$best_active, criterion = "BIC before refit",
    lambda_eta = ecagl_ic$lambda_eta, selection_df = ecagl_ic$df,
    selection_bic = ecagl_ic$BIC, path_candidates = nrow(ecagl$path),
    selected_at_path_end = ecagl$best_idx == nrow(ecagl$path),
    elapsed_sec = timings$ecagl
  )
}

comparison <- do.call(rbind, rows)
comparison$run_label <- cfg$run_label
comparison$representation <- payload$representation
comparison$n <- n
comparison$d <- d
comparison$K <- cfg$K
comparison$nstart <- cfg$nstart
comparison$max_iter <- cfg$max_iter
comparison$max_path_steps <- cfg$max_path_steps
comparison$adaptive_gamma <- ifelse(comparison$method == "E-CAGL", cfg$adaptive_gamma, NA_real_)
comparison$adaptive_eps <- ifelse(comparison$method == "E-CAGL", cfg$adaptive_eps, NA_real_)
comparison$uses_supplied_labels_for_selection <- FALSE

m_path <- m_l$path
m_path$method <- "M-L"
m_path$selected_q <- vapply(m_l$fits, function(f) sum(active_mu_coord(f, cfg$zero_eps)), numeric(1))
m_path$NMI <- vapply(
  m_l$fits,
  function(f) normalized_mutual_information(y, max.col(f$tau, ties.method = "first")),
  numeric(1)
)
m_path$lambda_eta <- NA_real_
m_path$path_parameter <- m_path$beta

e_path <- rbind(ecgl$path, ecagl$path)
e_path$beta <- NA_real_
e_path$path_parameter <- e_path$lambda_eta
path_keep <- c(
  "method", "step", "path_parameter", "beta", "lambda_eta", "selected_q",
  "ARI", "NMI", "loglik", "pen_loglik", "df", "BIC", "EBIC",
  "converged", "iter"
)
path_diagnostic <- rbind(
  m_path[, path_keep, drop = FALSE],
  e_path[, path_keep, drop = FALSE]
)

comparison_path <- file.path(
  cfg$out_dir, paste0(cfg$output_prefix, "_method_comparison.csv")
)
path_path <- file.path(
  cfg$out_dir, paste0(cfg$output_prefix, "_path_diagnostic.csv")
)
notes_path <- file.path(
  cfg$out_dir, paste0(cfg$output_prefix, "_method_comparison_notes.md")
)
utils::write.csv(comparison, comparison_path, row.names = FALSE)
utils::write.csv(path_diagnostic, path_path, row.names = FALSE)

if (cfg$save_paths) {
  saveRDS(
    list(m_l = m_l, ecgl = ecgl, ecagl = ecagl),
    file.path(cfg$out_dir, paste0(cfg$output_prefix, "_fitted_paths.rds"))
  )
}

fmt <- function(x, digits = 3L) {
  ifelse(is.na(x), "NA", formatC(x, digits = digits, format = "f"))
}
main_rows <- comparison[
  comparison$stage %in% c("clustering baseline", "dense", "penalized", "refit"),
  , drop = FALSE
]
note_lines <- c(
  paste0("# ", cfg$dataset_label, " vMF comparison pilot"),
  "",
  sprintf("- Data: N=%d, d=%d, supplied classes K=%d.", n, d, cfg$K),
  sprintf("- Input matrix class: %s; fitting uses Matrix algebra.", input_matrix_class),
  sprintf("- Representation: %s.", payload$representation),
  sprintf("- Fitting: %d common start seeds; at most %d path candidates.",
          cfg$nstart, cfg$max_path_steps),
  "- Each vMF fit is initialized from the corresponding converged spherical k-means partition.",
  "- Supplied labels are used only for ARI/NMI evaluation, not for initialization or tuning.",
  "- M-L is the shared-kappa, entry-wise L1 sparse-prototype path used for the Rossi reference.",
  "- E-CGL is the main centered-Eta coordinate-group method; E-CAGL is its adaptive extension (gamma=1).",
  "- E methods select support by BIC before fixed-support refitting.",
  "- Posthoc shared-kappa scoring is a diagnostic only: fitted directions and support are fixed while kappa values are replaced by their mean.",
  "- M-L selected_q is prototype-union support; E selected_q is centered-Eta posterior-decision support.",
  "- M-L prototype_entry_sparsity is also reported because entry-wise zeros can increase while the coordinate union remains dense.",
  "- The data have no ground-truth feature support, so support TPR, FPR, and F1 are not reported.",
  if (is.null(payload$vocab) && is.null(payload$vocabulary)) {
    "- Vocabulary terms are unavailable; selected coordinates cannot be named."
  } else {
    "- Vocabulary terms are available for post-selection interpretation."
  },
  "- Sparse Matrix algebra is used. The dense-only Rcpp E-step is not connected in this pilot.",
  "- For high dimension, the vMF log-normalizer uses a second-order uniform asymptotic Bessel expansion in this runner.",
  "",
  "## Main rows",
  "",
  paste0(
    "- ", main_rows$method, " [", main_rows$stage, "]: ARI=",
    fmt(main_rows$ARI), ", NMI=", fmt(main_rows$NMI),
    ", selected_q=", fmt(main_rows$selected_q, 0L), "."
  ),
  "",
  "## Interpretation boundary",
  "",
  "Clustering metrics are directly comparable across methods. Support counts are reported with their estimand because prototype-union support and posterior-decision support are different targets.",
  ""
)
writeLines(note_lines, notes_path, useBytes = TRUE)

cat("\n", cfg$dataset_label, " comparison\n", sep = "")
print(comparison[, c(
  "method", "stage", "support_target", "ARI", "NMI", "selected_q",
  "coordinate_sparsity", "prototype_entry_sparsity", "kappa_ratio",
  "cluster_size", "path_candidates",
  "selected_at_path_end", "elapsed_sec"
)], row.names = FALSE)
cat("\nSaved outputs to:", normalizePath(cfg$out_dir, winslash = "/"), "\n")
