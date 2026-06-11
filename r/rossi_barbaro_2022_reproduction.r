# ==============================================================================
# Rossi & Barbaro (2022) sparse vMF mixture simulation reproduction
# ------------------------------------------------------------------------------
# Paper:
#   Fabrice Rossi and Florian Barbaro (2022/2023 arXiv version),
#   "Mixture of von Mises-Fisher distribution with sparse prototypes"
#   Neurocomputing 501, 41-74.
#
# Goal:
#   Reproduce the paper's simulation logic before implementing the thesis model:
#   - data from vMF mixtures on the unit sphere
#   - L1 penalty on directional means mu_k
#   - EM algorithm with the fixed point M-step in Algorithm 2
#   - beta path following in Algorithm 3
#   - AIC/BIC/RIC/RICc/EBIC summaries
#
# The file is intentionally dependency-free. It uses only base R.
# ==============================================================================

options(stringsAsFactors = FALSE)

# ------------------------------------------------------------------------------
# Basic helpers
# ------------------------------------------------------------------------------

l2_norm <- function(x) sqrt(sum(x * x))

normalize_rows <- function(X, eps = 1e-12) {
  nr <- sqrt(rowSums(X * X))
  nr[nr < eps] <- 1
  sweep(X, 1, nr, "/")
}

logsumexp <- function(x) {
  m <- max(x)
  m + log(sum(exp(x - m)))
}

row_logsumexp <- function(M) {
  apply(M, 1, logsumexp)
}

comb2 <- function(x) x * (x - 1) / 2

adjusted_rand_index <- function(labels_true, labels_pred) {
  labels_true <- as.factor(labels_true)
  labels_pred <- as.factor(labels_pred)
  tab <- table(labels_true, labels_pred)
  n <- sum(tab)
  if (n <= 1) return(NA_real_)

  sum_nij <- sum(comb2(as.vector(tab)))
  sum_ai <- sum(comb2(rowSums(tab)))
  sum_bj <- sum(comb2(colSums(tab)))
  total <- comb2(n)

  expected <- sum_ai * sum_bj / total
  max_index <- 0.5 * (sum_ai + sum_bj)
  denom <- max_index - expected
  if (abs(denom) < 1e-12) return(0)
  (sum_nij - expected) / denom
}

all_permutations <- function(n) {
  if (n == 1) return(matrix(1L, nrow = 1))
  prev <- all_permutations(n - 1)
  out <- vector("list", n * nrow(prev))
  idx <- 1L
  for (i in seq_len(nrow(prev))) {
    for (pos in seq_len(n)) {
      out[[idx]] <- append(prev[i, ], n, after = pos - 1L)
      idx <- idx + 1L
    }
  }
  do.call(rbind, out)
}

best_perm_by_cosine <- function(mu_est, mu_true) {
  K <- nrow(mu_true)
  if (nrow(mu_est) != K || K > 8) return(seq_len(nrow(mu_est)))
  perms <- all_permutations(K)
  score <- rep(NA_real_, nrow(perms))
  for (i in seq_len(nrow(perms))) {
    score[i] <- sum(rowSums(mu_est[perms[i, ], , drop = FALSE] *
                              mu_true, na.rm = TRUE))
  }
  perms[which.max(score), ]
}

# ------------------------------------------------------------------------------
# vMF density and sampling
# ------------------------------------------------------------------------------

estimate_kappa <- function(rho, d, kappa_cap = 1e6) {
  rho <- min(max(rho, 1e-10), 1 - 1e-8)
  kappa <- (d * rho - rho^3) / (1 - rho^2)
  min(max(kappa, 1e-10), kappa_cap)
}

log_vmf_const_one <- function(kappa, d) {
  if (kappa < 1e-8) {
    return(lgamma(d / 2) - (d / 2) * log(2 * pi))
  }
  nu <- d / 2 - 1
  scaled_bessel <- besselI(kappa, nu, expon.scaled = TRUE)
  if (!is.finite(scaled_bessel) || scaled_bessel <= 0) {
    return(((d - 1) / 2) * (log(kappa) - log(2 * pi)) - kappa)
  }
  log_bessel <- log(scaled_bessel) + kappa
  if (!is.finite(log_bessel)) {
    return(((d - 1) / 2) * (log(kappa) - log(2 * pi)) - kappa)
  }
  nu * log(kappa) - (d / 2) * log(2 * pi) - log_bessel
}

log_vmf_const <- function(kappa, d) {
  vapply(kappa, log_vmf_const_one, numeric(1), d = d)
}

runif_sphere <- function(n, d) {
  X <- matrix(rnorm(n * d), nrow = n, ncol = d)
  normalize_rows(X)
}

householder_to_mu <- function(X, mu) {
  d <- length(mu)
  e <- rep(0, d)
  e[d] <- 1
  if (l2_norm(mu - e) < 1e-12) return(X)
  v <- e - mu
  v <- v / l2_norm(v)
  X - 2 * tcrossprod(X %*% v, v)
}

rvMF <- function(n, mu, kappa) {
  d <- length(mu)
  mu <- mu / l2_norm(mu)
  if (kappa < 1e-8) return(runif_sphere(n, d))

  b <- (-2 * kappa + sqrt(4 * kappa^2 + (d - 1)^2)) / (d - 1)
  x0 <- (1 - b) / (1 + b)
  c_const <- kappa * x0 + (d - 1) * log(1 - x0^2)

  X <- matrix(0, nrow = n, ncol = d)
  for (i in seq_len(n)) {
    repeat {
      z <- rbeta(1, (d - 1) / 2, (d - 1) / 2)
      w <- (1 - (1 + b) * z) / (1 - (1 - b) * z)
      u <- runif(1)
      accept_log <- kappa * w + (d - 1) * log(1 - x0 * w) - c_const
      if (accept_log >= log(u)) break
    }
    v <- runif_sphere(1, d - 1)
    X[i, ] <- c(sqrt(max(0, 1 - w^2)) * v, w)
  }
  householder_to_mu(X, mu)
}

# ------------------------------------------------------------------------------
# Rossi-Barbaro simulated data generation
# ------------------------------------------------------------------------------

base_kappa_lookup <- function(d, overlap) {
  if (d == 100 && abs(overlap - 0.025) < 1e-12) return(17.34)
  if (d == 100 && abs(overlap - 0.05) < 1e-12) return(15.09)
  if (d == 10 && abs(overlap - 0.05) < 1e-12) return(5.37)

  k100 <- if (overlap <= 0.025) 17.34 else 15.09
  k100 * d / 100
}

select_separated_directions <- function(K, d, candidate_factor = 20) {
  candidates <- runif_sphere(candidate_factor * K, d)
  if (K == 1) return(candidates[1, , drop = FALSE])

  S <- tcrossprod(candidates)
  S[lower.tri(S, diag = TRUE)] <- Inf
  first_pair <- arrayInd(which.min(S), dim(S))[1, ]
  selected <- as.integer(first_pair)

  while (length(selected) < K) {
    remaining <- setdiff(seq_len(nrow(candidates)), selected)
    max_ip <- apply(S[remaining, selected, drop = FALSE], 1, max)
    selected <- c(selected, remaining[which.min(max_ip)])
  }
  candidates[selected, , drop = FALSE]
}

sparsify_directions <- function(mu, nonzero_fraction, max_tries = 1000) {
  K <- nrow(mu)
  d <- ncol(mu)
  q <- max(1L, min(d, round(nonzero_fraction * d)))

  for (attempt in seq_len(max_tries)) {
    support <- matrix(FALSE, nrow = K, ncol = d)
    for (k in seq_len(K)) support[k, sample.int(d, q)] <- TRUE
    mu_sparse <- mu * support
    norms <- sqrt(rowSums(mu_sparse * mu_sparse))
    if (any(norms < 1e-10)) next
    mu_sparse <- sweep(mu_sparse, 1, norms, "/")
    distinct <- TRUE
    if (K > 1) {
      ip <- tcrossprod(mu_sparse)
      ip[lower.tri(ip, diag = TRUE)] <- NA
      distinct <- all(abs(ip[!is.na(ip)]) < 1 - 1e-8)
    }
    if (distinct) {
      return(list(mu = mu_sparse, support = support))
    }
  }
  stop("Could not generate distinct sparse directional means.")
}

make_rb2022_parameters <- function(K = 4,
                                   d = 100,
                                   overlap = 0.05,
                                   nonzero_fraction = 0.10,
                                   base_kappa = NULL) {
  mu_dense <- select_separated_directions(K, d)
  sparse <- sparsify_directions(mu_dense, nonzero_fraction)
  mu <- sparse$mu

  if (is.null(base_kappa)) base_kappa <- base_kappa_lookup(d, overlap)
  kappa_raw <- rnorm(K, mean = base_kappa, sd = 0.025 * base_kappa)
  kappa_raw <- pmax(kappa_raw, 1e-6)

  ip <- tcrossprod(mu)
  diag(ip) <- -Inf
  max_ip <- apply(ip, 1, max)
  kappa <- 2 * kappa_raw / pmax(1e-8, 1 - max_ip)

  list(
    alpha = rep(1 / K, K),
    mu = mu,
    kappa = kappa,
    support = sparse$support,
    base_kappa = base_kappa
  )
}

simulate_rb2022_data <- function(n = 200,
                                 K = 4,
                                 d = 100,
                                 overlap = 0.05,
                                 nonzero_fraction = 0.10,
                                 params = NULL) {
  if (is.null(params)) {
    params <- make_rb2022_parameters(K, d, overlap, nonzero_fraction)
  }

  z <- sample.int(K, size = n, replace = TRUE, prob = params$alpha)
  X <- matrix(0, nrow = n, ncol = d)
  for (k in seq_len(K)) {
    idx <- which(z == k)
    if (length(idx) > 0) {
      X[idx, ] <- rvMF(length(idx), params$mu[k, ], params$kappa[k])
    }
  }

  list(X = X, z = z, params = params)
}

# ------------------------------------------------------------------------------
# Sparse vMF EM
# ------------------------------------------------------------------------------

e_step_vmf <- function(X, theta) {
  n <- nrow(X)
  d <- ncol(X)
  K <- nrow(theta$mu)
  kappa <- theta$kappa
  if (length(kappa) == 1) kappa <- rep(kappa, K)

  logdens <- X %*% t(theta$mu)
  logdens <- sweep(logdens, 2, kappa, "*")
  logdens <- sweep(logdens, 2, log_vmf_const(kappa, d), "+")
  logdens <- sweep(logdens, 2, log(pmax(theta$alpha, 1e-300)), "+")

  lse <- row_logsumexp(logdens)
  tau <- exp(sweep(logdens, 1, lse, "-"))
  list(tau = tau, loglik = sum(lse))
}

init_vmf_mixture <- function(X, K, kappa_cap = 1e6) {
  n <- nrow(X)
  d <- ncol(X)
  idx <- sample.int(n, K, replace = FALSE)
  mu <- X[idx, , drop = FALSE]

  cl <- max.col(X %*% t(mu), ties.method = "random")
  tau <- matrix(0, nrow = n, ncol = K)
  tau[cbind(seq_len(n), cl)] <- 1

  Nk <- colSums(tau)
  if (any(Nk == 0)) stop("Empty component in initialization.")

  alpha <- Nk / n
  r <- t(tau) %*% X
  kappa <- numeric(K)
  for (k in seq_len(K)) {
    rho <- as.numeric(crossprod(mu[k, ], r[k, ])) / Nk[k]
    kappa[k] <- estimate_kappa(rho, d, kappa_cap)
  }

  list(alpha = alpha, mu = mu, kappa = kappa)
}

update_mu_kappa_one <- function(r_k,
                                Nk,
                                kappa_start,
                                beta,
                                d,
                                kappa_cap,
                                inner_max_iter,
                                inner_tol,
                                zero_eps) {
  kappa <- kappa_start
  mu <- r_k / max(l2_norm(r_k), 1e-12)

  for (inner in seq_len(inner_max_iter)) {
    shrink <- pmax(kappa * abs(r_k) - beta, 0)
    shrink_norm <- l2_norm(shrink)
    if (shrink_norm <= zero_eps) {
      return(list(failed = TRUE, mu = mu, kappa = kappa))
    }

    mu_new <- sign(r_k) * shrink / shrink_norm
    rho <- as.numeric(crossprod(mu_new, r_k)) / max(Nk, 1e-12)
    if (!is.finite(rho) || rho <= 0) {
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

fit_svMF_em <- function(X,
                        K,
                        beta = 0,
                        init = NULL,
                        shared_kappa = FALSE,
                        max_iter = 300,
                        inner_max_iter = 100,
                        tol = 1e-7,
                        inner_tol = 1e-8,
                        zero_eps = 1e-8,
                        kappa_cap = 1e6,
                        verbose = FALSE) {
  n <- nrow(X)
  d <- ncol(X)

  theta <- if (is.null(init)) init_vmf_mixture(X, K, kappa_cap) else init
  theta$mu <- normalize_rows(theta$mu)
  theta$alpha <- pmax(theta$alpha, 1e-12)
  theta$alpha <- theta$alpha / sum(theta$alpha)
  if (length(theta$kappa) == 1) theta$kappa <- rep(theta$kappa, K)
  if (shared_kappa) theta$kappa <- rep(mean(theta$kappa), K)

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

    if (!shared_kappa) {
      for (k in seq_len(K)) {
        upd <- update_mu_kappa_one(
          r_k = r[k, ],
          Nk = Nk[k],
          kappa_start = theta$kappa[k],
          beta = beta,
          d = d,
          kappa_cap = kappa_cap,
          inner_max_iter = inner_max_iter,
          inner_tol = inner_tol,
          zero_eps = zero_eps
        )
        if (upd$failed) {
          return(c(theta, list(
            failed = TRUE, converged = FALSE, iter = iter,
            loglik = e$loglik, pen_loglik = NA_real_, tau = tau
          )))
        }
        mu_new[k, ] <- upd$mu
        kappa_new[k] <- upd$kappa
      }
    } else {
      kappa_shared <- mean(theta$kappa)
      for (inner in seq_len(inner_max_iter)) {
        for (k in seq_len(K)) {
          shrink <- pmax(kappa_shared * abs(r[k, ]) - beta, 0)
          shrink_norm <- l2_norm(shrink)
          if (shrink_norm <= zero_eps) {
            return(c(theta, list(
              failed = TRUE, converged = FALSE, iter = iter,
              loglik = e$loglik, pen_loglik = NA_real_, tau = tau
            )))
          }
          mu_new[k, ] <- sign(r[k, ]) * shrink / shrink_norm
        }
        rho <- sum(rowSums(mu_new * r)) / n
        kappa_next <- estimate_kappa(rho, d, kappa_cap)
        if (abs(kappa_next - kappa_shared) / max(1, kappa_shared) < inner_tol) {
          kappa_shared <- kappa_next
          break
        }
        kappa_shared <- kappa_next
      }
      kappa_new <- rep(kappa_shared, K)
    }

    theta_new <- list(alpha = alpha_new, mu = mu_new, kappa = kappa_new)
    e_new <- e_step_vmf(X, theta_new)
    obj <- e_new$loglik - beta * sum(abs(theta_new$mu))

    if (verbose) {
      cat(sprintf("iter=%d beta=%.6g loglik=%.6f obj=%.6f\n",
                  iter, beta, e_new$loglik, obj))
    }

    theta <- theta_new
    last_e <- e_new

    if (is.finite(prev_obj) &&
        abs(obj - prev_obj) / max(1, abs(prev_obj)) < tol) {
      return(c(theta, list(
        failed = FALSE, converged = TRUE, iter = iter,
        loglik = e_new$loglik, pen_loglik = obj,
        tau = e_new$tau
      )))
    }
    prev_obj <- obj
  }

  if (is.null(last_e)) last_e <- e_step_vmf(X, theta)
  c(theta, list(
    failed = FALSE, converged = FALSE, iter = max_iter,
    loglik = last_e$loglik,
    pen_loglik = last_e$loglik - beta * sum(abs(theta$mu)),
    tau = last_e$tau
  ))
}

fit_svMF_multistart <- function(X,
                                K,
                                beta = 0,
                                nstart = 10,
                                shared_kappa = FALSE,
                                seed = NULL,
                                ...) {
  if (!is.null(seed)) set.seed(seed)
  best <- NULL
  best_obj <- -Inf
  for (s in seq_len(nstart)) {
    fit <- tryCatch(
      fit_svMF_em(X, K, beta = beta, shared_kappa = shared_kappa, ...),
      error = function(e) NULL
    )
    if (is.null(fit) || isTRUE(fit$failed)) next
    obj <- fit$pen_loglik
    if (is.finite(obj) && obj > best_obj) {
      best <- fit
      best_obj <- obj
    }
  }
  if (is.null(best)) stop("All EM initializations failed.")
  best
}

model_df <- function(theta, shared_kappa = FALSE, zero_eps = 1e-8) {
  K <- nrow(theta$mu)
  alpha_kappa_df <- if (shared_kappa) K else (2 * K - 1)
  nnz <- rowSums(abs(theta$mu) > zero_eps)
  alpha_kappa_df + sum(pmax(1, nnz - 1))
}

model_ic <- function(theta,
                     n,
                     d,
                     loglik,
                     shared_kappa = FALSE,
                     gamma = 0.5,
                     zero_eps = 1e-8) {
  df <- model_df(theta, shared_kappa, zero_eps)
  data.frame(
    df = df,
    AIC = 2 * df - 2 * loglik,
    BIC = log(n) * df - 2 * loglik,
    RIC = 2 * log(d) * df - 2 * loglik,
    RICc = 2 * (log(d) + log(log(d))) * df - 2 * loglik,
    EBIC = (log(n) + 2 * gamma * log(d)) * df - 2 * loglik
  )
}

evaluate_fit <- function(fit,
                         X,
                         beta,
                         labels_true = NULL,
                         mu_true = NULL,
                         support_true = NULL,
                         shared_kappa = FALSE,
                         gamma = 0.5,
                         zero_eps = 1e-8) {
  n <- nrow(X)
  d <- ncol(X)
  K <- nrow(fit$mu)
  ic <- model_ic(fit, n, d, fit$loglik, shared_kappa, gamma, zero_eps)
  cluster <- max.col(fit$tau, ties.method = "first")
  ari <- if (is.null(labels_true)) NA_real_ else adjusted_rand_index(labels_true, cluster)

  support_est <- abs(fit$mu) > zero_eps
  nnz_fraction <- mean(support_est)
  coord_support_est <- colSums(support_est) > 0

  coord_precision <- coord_recall <- coord_f1 <- NA_real_
  entry_precision <- entry_recall <- entry_f1 <- NA_real_

  if (!is.null(support_true)) {
    coord_support_true <- colSums(support_true) > 0
    tp <- sum(coord_support_est & coord_support_true)
    fp <- sum(coord_support_est & !coord_support_true)
    fn <- sum(!coord_support_est & coord_support_true)
    coord_precision <- ifelse(tp + fp > 0, tp / (tp + fp), NA_real_)
    coord_recall <- ifelse(tp + fn > 0, tp / (tp + fn), NA_real_)
    coord_f1 <- ifelse(is.na(coord_precision + coord_recall) ||
                         coord_precision + coord_recall == 0,
                       NA_real_,
                       2 * coord_precision * coord_recall /
                         (coord_precision + coord_recall))

    if (!is.null(mu_true) && nrow(support_true) == K) {
      perm <- best_perm_by_cosine(fit$mu, mu_true)
      support_est_perm <- support_est[perm, , drop = FALSE]
      tp_e <- sum(support_est_perm & support_true)
      fp_e <- sum(support_est_perm & !support_true)
      fn_e <- sum(!support_est_perm & support_true)
      entry_precision <- ifelse(tp_e + fp_e > 0, tp_e / (tp_e + fp_e), NA_real_)
      entry_recall <- ifelse(tp_e + fn_e > 0, tp_e / (tp_e + fn_e), NA_real_)
      entry_f1 <- ifelse(is.na(entry_precision + entry_recall) ||
                           entry_precision + entry_recall == 0,
                         NA_real_,
                         2 * entry_precision * entry_recall /
                           (entry_precision + entry_recall))
    }
  }

  data.frame(
    beta = beta,
    loglik = fit$loglik,
    pen_loglik = fit$pen_loglik,
    converged = fit$converged,
    failed = fit$failed,
    iter = fit$iter,
    ARI = ari,
    nnz_fraction = nnz_fraction,
    coord_precision = coord_precision,
    coord_recall = coord_recall,
    coord_f1 = coord_f1,
    entry_precision = entry_precision,
    entry_recall = entry_recall,
    entry_f1 = entry_f1,
    ic,
    row.names = NULL
  )
}

fit_svMF_path <- function(X,
                          K,
                          labels_true = NULL,
                          mu_true = NULL,
                          support_true = NULL,
                          nstart = 10,
                          max_path_steps = 200,
                          min_rel_beta = 1e-3,
                          shared_kappa = FALSE,
                          zero_eps = 1e-8,
                          beta_eps = 1e-10,
                          gamma = 0.5,
                          verbose = FALSE,
                          ...) {
  d <- ncol(X)
  beta <- 0
  fit <- fit_svMF_multistart(
    X, K, beta = beta, nstart = nstart,
    shared_kappa = shared_kappa, zero_eps = zero_eps, ...
  )

  fits <- list(fit)
  betas <- beta
  path_table <- evaluate_fit(
    fit, X, beta,
    labels_true = labels_true,
    mu_true = mu_true,
    support_true = support_true,
    shared_kappa = shared_kappa,
    gamma = gamma,
    zero_eps = zero_eps
  )
  path_table$step <- 1L

  for (step in 2:max_path_steps) {
    e <- e_step_vmf(X, fit)
    r <- t(e$tau) %*% X
    kappa <- fit$kappa
    if (length(kappa) == 1) kappa <- rep(kappa, K)

    margin <- matrix(0, nrow = K, ncol = d)
    for (k in seq_len(K)) margin[k, ] <- kappa[k] * abs(r[k, ]) - beta
    candidates <- margin[margin > beta_eps]
    if (length(candidates) == 0) break

    beta_next <- beta + min(candidates)
    if (beta > 0) beta_next <- max(beta_next, beta * (1 + min_rel_beta))
    if (!is.finite(beta_next) || beta_next <= beta) break

    fit_next <- tryCatch(
      fit_svMF_em(
        X, K, beta = beta_next, init = fit,
        shared_kappa = shared_kappa, zero_eps = zero_eps, ...
      ),
      error = function(e) NULL
    )
    if (is.null(fit_next) || isTRUE(fit_next$failed)) break

    fit_next$mu[abs(fit_next$mu) < zero_eps] <- 0
    fit_next$mu <- normalize_rows(fit_next$mu)
    fit_next_e <- e_step_vmf(X, fit_next)
    fit_next$loglik <- fit_next_e$loglik
    fit_next$pen_loglik <- fit_next$loglik - beta_next * sum(abs(fit_next$mu))
    fit_next$tau <- fit_next_e$tau

    fit <- fit_next
    beta <- beta_next
    fits[[length(fits) + 1L]] <- fit
    betas <- c(betas, beta)

    one_row <- evaluate_fit(
      fit, X, beta,
      labels_true = labels_true,
      mu_true = mu_true,
      support_true = support_true,
      shared_kappa = shared_kappa,
      gamma = gamma,
      zero_eps = zero_eps
    )
    one_row$step <- step
    path_table <- rbind(path_table, one_row)

    if (verbose) {
      cat(sprintf("K=%d step=%d beta=%.6g nnz=%.4f BIC=%.3f\n",
                  K, step, beta, one_row$nnz_fraction, one_row$BIC))
    }

    if (all(rowSums(abs(fit$mu) > zero_eps) <= 1)) break
  }

  list(fits = fits, betas = betas, path = path_table)
}

# ------------------------------------------------------------------------------
# Reference spherical k-means baseline
# ------------------------------------------------------------------------------

spherical_kmeans <- function(X, K, nstart = 10, max_iter = 100, tol = 1e-8) {
  n <- nrow(X)
  best <- NULL
  best_obj <- -Inf

  for (s in seq_len(nstart)) {
    mu <- X[sample.int(n, K, replace = FALSE), , drop = FALSE]
    cl <- rep(NA_integer_, n)
    for (iter in seq_len(max_iter)) {
      sim <- X %*% t(mu)
      cl_new <- max.col(sim, ties.method = "random")
      mu_new <- matrix(0, nrow = K, ncol = ncol(X))
      for (k in seq_len(K)) {
        idx <- which(cl_new == k)
        if (length(idx) == 0) {
          mu_new[k, ] <- X[sample.int(n, 1), ]
        } else {
          mu_new[k, ] <- colMeans(X[idx, , drop = FALSE])
        }
      }
      mu_new <- normalize_rows(mu_new)
      if (!any(is.na(cl)) && all(cl_new == cl)) break
      if (max(abs(mu_new - mu)) < tol) {
        mu <- mu_new
        cl <- cl_new
        break
      }
      mu <- mu_new
      cl <- cl_new
    }
    obj <- sum(apply(X %*% t(mu), 1, max))
    if (obj > best_obj) {
      best <- list(cluster = cl, mu = mu, objective = obj, iter = iter)
      best_obj <- obj
    }
  }
  best
}

# ------------------------------------------------------------------------------
# Simulation drivers
# ------------------------------------------------------------------------------

run_rb2022_one <- function(rep_id = 1,
                           n = 200,
                           d = 100,
                           K_true = 4,
                           K_fit_grid = 4,
                           overlap = 0.05,
                           nonzero_fraction = 0.10,
                           nstart = 10,
                           max_path_steps = 200,
                           shared_kappa = FALSE,
                           gamma = 0.5,
                           seed = NULL,
                           verbose = FALSE,
                           ...) {
  if (!is.null(seed)) set.seed(seed)

  dat <- simulate_rb2022_data(
    n = n,
    K = K_true,
    d = d,
    overlap = overlap,
    nonzero_fraction = nonzero_fraction
  )
  X <- dat$X
  z <- dat$z
  params <- dat$params

  baseline_rows <- list()
  if (K_true %in% K_fit_grid) {
    skm <- spherical_kmeans(X, K_true, nstart = max(1, min(nstart, 10)))
    baseline_rows[[1]] <- data.frame(
      rep = rep_id,
      n = n,
      d = d,
      K_true = K_true,
      K_fit = K_true,
      overlap = overlap,
      nonzero_fraction = nonzero_fraction,
      method = "spherical_kmeans",
      criterion = NA_character_,
      selected = TRUE,
      step = NA_integer_,
      beta = NA_real_,
      loglik = NA_real_,
      pen_loglik = NA_real_,
      converged = TRUE,
      failed = FALSE,
      iter = skm$iter,
      ARI = adjusted_rand_index(z, skm$cluster),
      nnz_fraction = NA_real_,
      coord_precision = NA_real_,
      coord_recall = NA_real_,
      coord_f1 = NA_real_,
      entry_precision = NA_real_,
      entry_recall = NA_real_,
      entry_f1 = NA_real_,
      df = NA_real_,
      AIC = NA_real_,
      BIC = NA_real_,
      RIC = NA_real_,
      RICc = NA_real_,
      EBIC = NA_real_
    )
  }

  path_rows <- list()
  selection_rows <- list()

  for (K_fit in K_fit_grid) {
    if (verbose) {
      cat(sprintf(
        "\n[rep=%d] n=%d overlap=%.3f nonzero=%.3f K_fit=%d\n",
        rep_id, n, overlap, nonzero_fraction, K_fit
      ))
    }

    path <- fit_svMF_path(
      X = X,
      K = K_fit,
      labels_true = z,
      mu_true = if (K_fit == K_true) params$mu else NULL,
      support_true = if (K_fit == K_true) params$support else NULL,
      nstart = nstart,
      max_path_steps = max_path_steps,
      shared_kappa = shared_kappa,
      gamma = gamma,
      verbose = verbose,
      ...
    )

    ptab <- path$path
    ptab$rep <- rep_id
    ptab$n <- n
    ptab$d <- d
    ptab$K_true <- K_true
    ptab$K_fit <- K_fit
    ptab$overlap <- overlap
    ptab$nonzero_fraction <- nonzero_fraction
    ptab$method <- "sparse_vmf_path"
    ptab$criterion <- NA_character_
    ptab$selected <- FALSE
    ptab <- ptab[, c(
      "rep", "n", "d", "K_true", "K_fit", "overlap", "nonzero_fraction",
      "method", "criterion", "selected", "step", "beta", "loglik",
      "pen_loglik", "converged", "failed", "iter", "ARI", "nnz_fraction",
      "coord_precision", "coord_recall", "coord_f1",
      "entry_precision", "entry_recall", "entry_f1",
      "df", "AIC", "BIC", "RIC", "RICc", "EBIC"
    )]
    path_rows[[length(path_rows) + 1L]] <- ptab
  }

  all_paths <- do.call(rbind, path_rows)
  criteria <- c("AIC", "BIC", "RIC", "RICc", "EBIC")

  for (crit in criteria) {
    dense <- subset(all_paths, beta == 0)
    dense_best <- dense[which.min(dense[[crit]]), , drop = FALSE]
    dense_best$method <- "dense_vmf_selected_K"
    dense_best$criterion <- crit
    dense_best$selected <- TRUE
    selection_rows[[length(selection_rows) + 1L]] <- dense_best

    sparse_best_per_K <- do.call(rbind, lapply(split(all_paths, all_paths$K_fit),
      function(tab) tab[which.min(tab[[crit]]), , drop = FALSE]))
    sparse_best <- sparse_best_per_K[which.min(sparse_best_per_K[[crit]]), , drop = FALSE]
    sparse_best$method <- "sparse_vmf_selected_K_beta"
    sparse_best$criterion <- crit
    sparse_best$selected <- TRUE
    selection_rows[[length(selection_rows) + 1L]] <- sparse_best
  }

  out <- rbind(
    if (length(baseline_rows) > 0) do.call(rbind, baseline_rows) else NULL,
    all_paths,
    do.call(rbind, selection_rows)
  )
  row.names(out) <- NULL
  out
}

summarize_rb2022 <- function(results) {
  keys <- unique(results[, c(
    "n", "d", "K_true", "K_fit", "overlap", "nonzero_fraction",
    "method", "criterion", "selected"
  )])

  safe_mean <- function(x) if (sum(!is.na(x)) > 0) mean(x, na.rm = TRUE) else NA_real_
  safe_se <- function(x) {
    nn <- sum(!is.na(x))
    if (nn > 1) sd(x, na.rm = TRUE) / sqrt(nn) else NA_real_
  }

  rows <- vector("list", nrow(keys))
  metrics <- c(
    "beta", "ARI", "nnz_fraction", "coord_precision", "coord_recall",
    "coord_f1", "entry_precision", "entry_recall", "entry_f1",
    "df", "AIC", "BIC", "RIC", "RICc", "EBIC", "iter"
  )

  for (i in seq_len(nrow(keys))) {
    idx <- rep(TRUE, nrow(results))
    for (nm in names(keys)) {
      val <- keys[i, nm]
      if (is.na(val)) {
        idx <- idx & is.na(results[[nm]])
      } else {
        idx <- idx & results[[nm]] == val
      }
    }
    tmp <- results[idx, , drop = FALSE]
    row <- keys[i, , drop = FALSE]
    row$reps <- length(unique(tmp$rep))
    for (m in metrics) {
      row[[paste0(m, "_mean")]] <- safe_mean(tmp[[m]])
      row[[paste0(m, "_se")]] <- safe_se(tmp[[m]])
    }
    rows[[i]] <- row
  }

  do.call(rbind, rows)
}

run_rb2022_reproduction <- function(n_rep = 3,
                                    n_grid = c(200),
                                    d = 100,
                                    K_true = 4,
                                    K_fit_grid = 4,
                                    overlap_grid = c(0.05),
                                    nonzero_fraction_grid = c(0.10),
                                    nstart = 3,
                                    max_path_steps = 40,
                                    shared_kappa = FALSE,
                                    base_seed = 20260601,
                                    out_dir = "results/rb2022",
                                    verbose = TRUE,
                                    ...) {
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

  all <- list()
  idx <- 1L
  for (rep_id in seq_len(n_rep)) {
    for (n in n_grid) {
      for (overlap in overlap_grid) {
        for (nonzero_fraction in nonzero_fraction_grid) {
          seed <- base_seed + 100000L * rep_id +
            1000L * as.integer(1000 * overlap) +
            as.integer(1000 * nonzero_fraction) + n
          if (verbose) {
            cat(sprintf(
              "\n=== rep %d/%d | n=%d | overlap=%.3f | nonzero=%.3f ===\n",
              rep_id, n_rep, n, overlap, nonzero_fraction
            ))
          }
          all[[idx]] <- run_rb2022_one(
            rep_id = rep_id,
            n = n,
            d = d,
            K_true = K_true,
            K_fit_grid = K_fit_grid,
            overlap = overlap,
            nonzero_fraction = nonzero_fraction,
            nstart = nstart,
            max_path_steps = max_path_steps,
            shared_kappa = shared_kappa,
            seed = seed,
            verbose = verbose,
            ...
          )
          idx <- idx + 1L
        }
      }
    }
  }

  results <- do.call(rbind, all)
  summary <- summarize_rb2022(results)

  raw_path <- file.path(out_dir, "rb2022_reproduction_raw.csv")
  summary_path <- file.path(out_dir, "rb2022_reproduction_summary.csv")
  write.csv(results, raw_path, row.names = FALSE)
  write.csv(summary, summary_path, row.names = FALSE)

  list(results = results, summary = summary,
       raw_path = raw_path, summary_path = summary_path)
}

# ------------------------------------------------------------------------------
# Script entry point
# ------------------------------------------------------------------------------

if (sys.nframe() == 0) {
  cat("Running a quick Rossi-Barbaro 2022 reproduction smoke test.\n")
  cat("For the full paper grid, source this file and call:\n")
  cat("run_rb2022_reproduction(n_rep = 100, n_grid = c(200, 1000),\n")
  cat("  K_fit_grid = 1:6, overlap_grid = c(0.025, 0.05),\n")
  cat("  nonzero_fraction_grid = c(0.05, 0.10, 0.15),\n")
  cat("  nstart = 10, max_path_steps = 200)\n\n")

  res <- run_rb2022_reproduction(
    n_rep = 1,
    n_grid = c(200),
    d = 100,
    K_true = 4,
    K_fit_grid = 4,
    overlap_grid = c(0.05),
    nonzero_fraction_grid = c(0.10),
    nstart = 2,
    max_path_steps = 12,
    verbose = TRUE
  )

  print(res$summary)
  cat("\nWrote:\n")
  cat("  ", normalizePath(res$raw_path, winslash = "/"), "\n", sep = "")
  cat("  ", normalizePath(res$summary_path, winslash = "/"), "\n", sep = "")
}
