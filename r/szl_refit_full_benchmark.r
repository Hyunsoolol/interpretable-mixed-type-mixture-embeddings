############################################################
# Debiased Sum-to-Zero Lasso Mixture Clustering
# Full benchmark version
#
# Added comparison models:
#   Traditional        : K-means, PCA + K-means
#   Model-based        : Unpenalized diagonal GMM (Sigma = I_p prototype)
#   Sparse clustering  : Sparse K-means via sparcl if installed; fallback proxy otherwise
#   Spectral/screening : SC-FS proxy implemented in base R
#   Model-based VS     : SelvarMix proxy + actual roles S/nonW/nonW-refit if installed
#   Penalized GMM      : Naive Lasso self-tuned
#   Ablation           : Naive Lasso at refit lambda
#   Proposed           : SZL-Refit
#   Proposed auxiliary : ASZL-Refit
#   Oracle             : Oracle-feature GMM, True-parameter oracle
#
# First prototype assumes common known diagonal covariance Sigma = I_p.
# This is intentional for the first shrinkage/refit sanity check.
############################################################

set.seed(20260511)

############################################################
# 0. Utility functions
############################################################

soft_threshold <- function(x, t) {
  sign(x) * pmax(abs(x) - t, 0)
}

logsumexp_vec <- function(v) {
  m <- max(v)
  m + log(sum(exp(v - m)))
}

row_logsumexp <- function(A) {
  apply(A, 1, logsumexp_vec)
}

safe_mean <- function(x) {
  if (all(is.na(x))) return(NA_real_)
  mean(x, na.rm = TRUE)
}

safe_se <- function(x) {
  if (all(is.na(x))) return(NA_real_)
  x <- x[!is.na(x)]
  if (length(x) <= 1) return(NA_real_)
  sd(x) / sqrt(length(x))
}

adjusted_rand_index <- function(z_true, z_hat) {
  tab <- table(z_true, z_hat)
  choose2 <- function(x) x * (x - 1) / 2
  a <- sum(choose2(tab))
  b <- sum(choose2(rowSums(tab)))
  c <- sum(choose2(colSums(tab)))
  d <- choose2(sum(tab))
  expected <- b * c / d
  max_index <- 0.5 * (b + c)
  if (abs(max_index - expected) < 1e-12) return(0)
  (a - expected) / (max_index - expected)
}

entropy_mean <- function(R) {
  R2 <- pmax(R, 1e-15)
  -mean(rowSums(R2 * log(R2)))
}

all_perms <- function(K) {
  permute_vec <- function(v) {
    if (length(v) == 1) return(matrix(v, 1, 1))
    out <- NULL
    for (i in seq_along(v)) {
      sub <- permute_vec(v[-i])
      out <- rbind(out, cbind(v[i], sub))
    }
    out
  }
  permute_vec(1:K)
}

align_mu_to_truth <- function(mu_hat, mu_true) {
  K <- nrow(mu_true)
  perms <- all_perms(K)
  best_loss <- Inf
  best_mu <- mu_hat
  for (r in 1:nrow(perms)) {
    mu_perm <- mu_hat[perms[r, ], , drop = FALSE]
    loss <- sum((mu_perm - mu_true)^2)
    if (loss < best_loss) {
      best_loss <- loss
      best_mu <- mu_perm
    }
  }
  best_mu
}

standardize_X <- function(X) {
  Xs <- scale(X)
  Xs[, is.na(colMeans(Xs)), drop = FALSE] <- 0
  Xs
}

############################################################
# 1. Data generation
############################################################

make_data <- function(n = 300, p = 100, q = 5, a = 1.2, K = 3, pi = NULL) {
  if (is.null(pi)) pi <- rep(1 / K, K)
  z <- sample(1:K, n, replace = TRUE, prob = pi)
  
  mu <- matrix(0, K, p)
  
  base <- seq(-(K - 1) / 2, (K - 1) / 2, length.out = K)
  base <- base / max(abs(base))
  
  active <- 1:q
  for (k in active) {
    mu[, k] <- a * base
  }
  
  X <- matrix(rnorm(n * p), n, p)
  for (i in 1:n) X[i, ] <- X[i, ] + mu[z[i], ]
  
  delta <- mu - matrix(colMeans(mu), K, p, byrow = TRUE)
  
  list(X = X, z = z, mu = mu, delta = delta, S0 = active, pi = pi)
}

############################################################
# 2. GMM likelihood and E-step (Sigma = I_p)
############################################################

loglik_gmm <- function(X, pi, mu) {
  n <- nrow(X)
  K <- nrow(mu)
  logdens <- matrix(0, n, K)
  const <- -0.5 * ncol(X) * log(2 * base::pi)
  for (j in 1:K) {
    diff <- X - matrix(mu[j, ], n, ncol(X), byrow = TRUE)
    logdens[, j] <- log(pi[j] + 1e-15) + const - 0.5 * rowSums(diff^2)
  }
  sum(row_logsumexp(logdens))
}

estep_gmm <- function(X, pi, mu) {
  n <- nrow(X)
  K <- nrow(mu)
  logdens <- matrix(0, n, K)
  for (j in 1:K) {
    diff <- X - matrix(mu[j, ], n, ncol(X), byrow = TRUE)
    logdens[, j] <- log(pi[j] + 1e-15) - 0.5 * rowSums(diff^2)
  }
  lse <- row_logsumexp(logdens)
  R <- exp(logdens - lse)
  R / rowSums(R)
}

init_from_kmeans <- function(X, K, nstart = 10) {
  km <- kmeans(X, centers = K, nstart = nstart, iter.max = 100)
  labels <- km$cluster
  R <- matrix(0, nrow(X), K)
  R[cbind(1:nrow(X), labels)] <- 1
  pi <- colMeans(R)
  mu <- matrix(0, K, ncol(X))
  for (j in 1:K) {
    if (sum(labels == j) == 0) {
      mu[j, ] <- colMeans(X)
    } else {
      mu[j, ] <- colMeans(X[labels == j, , drop = FALSE])
    }
  }
  list(pi = pi, mu = mu, R = R, labels = labels)
}

fit_from_labels <- function(X, labels, K) {
  n <- nrow(X)
  p <- ncol(X)
  labels <- as.integer(labels)
  R <- matrix(0, n, K)
  R[cbind(1:n, labels)] <- 1
  pi <- colMeans(R)
  mu <- matrix(0, K, p)
  for (j in 1:K) {
    idx <- which(labels == j)
    if (length(idx) == 0) {
      mu[j, ] <- colMeans(X)
    } else {
      mu[j, ] <- colMeans(X[idx, , drop = FALSE])
    }
  }
  pi_safe <- pmax(pi, 1e-15)
  pi_safe <- pi_safe / sum(pi_safe)
  R_soft <- estep_gmm(X, pi_safe, mu)
  list(pi = pi_safe, mu = mu, R = R_soft, labels = labels,
       delta = mu - matrix(colMeans(mu), K, p, byrow = TRUE),
       loglik = loglik_gmm(X, pi_safe, mu))
}

fit_from_labels_support <- function(X, labels, K, S) {
  n <- nrow(X)
  p <- ncol(X)
  labels <- as.integer(labels)
  S <- sort(unique(S))
  R <- matrix(0, n, K)
  R[cbind(1:n, labels)] <- 1
  pi <- colMeans(R)
  mu <- matrix(colMeans(X), K, p, byrow = TRUE)
  if (length(S) > 0) {
    for (j in 1:K) {
      idx <- which(labels == j)
      if (length(idx) > 0) {
        mu[j, S] <- colMeans(X[idx, S, drop = FALSE])
      }
    }
  }
  pi_safe <- pmax(pi, 1e-15)
  pi_safe <- pi_safe / sum(pi_safe)
  R_soft <- estep_gmm(X, pi_safe, mu)
  list(pi = pi_safe, mu = mu, R = R_soft, labels = labels,
       delta = mu - matrix(colMeans(mu), K, p, byrow = TRUE),
       S = S, loglik = loglik_gmm(X, pi_safe, mu))
}

bic_support_fit <- function(fit, n, p, K, alpha = 0.5) {
  S <- if (!is.null(fit$S)) fit$S else integer(0)
  df_eff <- (K - 1) * length(S)
  -2 * fit$loglik + log(n) * df_eff + 2 * alpha * length(S) * log(p)
}

############################################################
# 3. Sum-to-zero lasso coordinate update
############################################################

update_delta_sumzero_lasso <- function(xbar, N, lambda_vec,
                                       max_iter = 100, tol = 1e-8) {
  K <- length(xbar)
  N <- pmax(N, 1e-8)
  lambda_vec <- pmax(lambda_vec, 0)
  
  if (max(lambda_vec) < 1e-12) {
    mu0 <- mean(xbar)
    delta <- xbar - mu0
    return(list(mu0 = mu0, delta = delta))
  }
  
  mu0 <- mean(xbar)
  delta <- xbar - mu0
  
  for (it in 1:max_iter) {
    old <- c(mu0, delta)
    u <- xbar - mu0
    
    f_eta <- function(eta) {
      sum(soft_threshold(u - eta / N, lambda_vec / N))
    }
    
    if (abs(f_eta(0)) < 1e-10) {
      eta <- 0
    } else {
      lo <- -1
      hi <- 1
      cnt <- 0
      while (f_eta(lo) < 0 && cnt < 100) {
        lo <- lo * 2
        cnt <- cnt + 1
      }
      cnt <- 0
      while (f_eta(hi) > 0 && cnt < 100) {
        hi <- hi * 2
        cnt <- cnt + 1
      }
      eta <- uniroot(f_eta, lower = lo, upper = hi, tol = 1e-10)$root
    }
    
    delta <- soft_threshold(u - eta / N, lambda_vec / N)
    # Numerical projection. For exact KKT studies, this line can be removed.
    if (abs(sum(delta)) > 1e-8) delta <- delta - mean(delta)
    
    mu0 <- sum(N * (xbar - delta)) / sum(N)
    
    if (max(abs(c(mu0, delta) - old)) < tol) break
  }
  
  list(mu0 = mu0, delta = delta)
}

############################################################
# 4. EM for sum-to-zero lasso screening
############################################################

em_sz_lasso <- function(X, K, lambda, adaptive_w = NULL, init = NULL,
                        max_iter = 60, tol = 1e-5, nstart = 10) {
  n <- nrow(X)
  p <- ncol(X)
  
  if (is.null(adaptive_w)) adaptive_w <- matrix(1, K, p)
  if (is.null(init)) init <- init_from_kmeans(X, K, nstart = nstart)
  
  pi <- init$pi
  mu <- init$mu
  ll_old <- -Inf
  
  for (iter in 1:max_iter) {
    R <- estep_gmm(X, pi, mu)
    N <- colSums(R) + 1e-8
    pi <- N / n
    
    mu_new <- matrix(0, K, p)
    for (k in 1:p) {
      xbar <- numeric(K)
      for (j in 1:K) {
        xbar[j] <- sum(R[, j] * X[, k]) / N[j]
      }
      lambda_vec <- lambda * adaptive_w[, k]
      upd <- update_delta_sumzero_lasso(xbar, N, lambda_vec)
      mu_new[, k] <- upd$mu0 + upd$delta
    }
    mu <- mu_new
    
    ll <- loglik_gmm(X, pi, mu)
    if (abs(ll - ll_old) < tol * (1 + abs(ll_old))) break
    ll_old <- ll
  }
  
  R <- estep_gmm(X, pi, mu)
  labels <- max.col(R)
  delta <- mu - matrix(colMeans(mu), K, p, byrow = TRUE)
  
  list(pi = pi, mu = mu, delta = delta, R = R, labels = labels,
       loglik = loglik_gmm(X, pi, mu), lambda = lambda, iter = iter)
}

############################################################
# 5. Support selection and unpenalized refit
############################################################

select_support_maxcontrast <- function(mu, tau = 1e-4) {
  K <- nrow(mu)
  p <- ncol(mu)
  S <- integer(0)
  for (k in 1:p) {
    m <- 0
    for (j in 1:(K - 1)) {
      for (l in (j + 1):K) {
        m <- max(m, abs(mu[j, k] - mu[l, k]))
      }
    }
    if (m > tau) S <- c(S, k)
  }
  S
}

em_refit_support <- function(X, K, S, init = NULL,
                             max_iter = 80, tol = 1e-5, nstart = 10) {
  n <- nrow(X)
  p <- ncol(X)
  S <- sort(unique(S))
  
  if (is.null(init)) {
    if (length(S) > 0) {
      init_sub <- init_from_kmeans(X[, S, drop = FALSE], K, nstart = nstart)
      mu0 <- matrix(colMeans(X), K, p, byrow = TRUE)
      for (j in 1:K) mu0[j, S] <- init_sub$mu[j, ]
      init <- list(pi = init_sub$pi, mu = mu0)
      init$R <- estep_gmm(X, init$pi, init$mu)
    } else {
      init <- init_from_kmeans(X, K, nstart = nstart)
      init$mu <- matrix(colMeans(X), K, p, byrow = TRUE)
      init$R <- estep_gmm(X, init$pi, init$mu)
    }
  } else {
    # Sanitize init so that non-selected variables satisfy common mean constraint.
    notS <- setdiff(1:p, S)
    if (length(notS) > 0) {
      common <- colMeans(X)
      init$mu[, notS] <- matrix(common[notS], K, length(notS), byrow = TRUE)
    }
    init$R <- estep_gmm(X, init$pi, init$mu)
  }
  
  pi <- init$pi
  mu <- init$mu
  ll_old <- -Inf
  
  for (iter in 1:max_iter) {
    R <- estep_gmm(X, pi, mu)
    N <- colSums(R) + 1e-8
    pi <- N / n
    
    mu_new <- matrix(colMeans(X), K, p, byrow = TRUE)
    if (length(S) > 0) {
      for (k in S) {
        for (j in 1:K) {
          mu_new[j, k] <- sum(R[, j] * X[, k]) / N[j]
        }
      }
    }
    mu <- mu_new
    
    ll <- loglik_gmm(X, pi, mu)
    if (abs(ll - ll_old) < tol * (1 + abs(ll_old))) break
    ll_old <- ll
  }
  
  R <- estep_gmm(X, pi, mu)
  labels <- max.col(R)
  delta <- mu - matrix(colMeans(mu), K, p, byrow = TRUE)
  
  list(pi = pi, mu = mu, delta = delta, R = R, labels = labels,
       loglik = loglik_gmm(X, pi, mu), S = S, iter = iter)
}

############################################################
# 6. Lambda grid and path fitting (with dual EBIC)
############################################################

make_lambda_grid <- function(X, K, nlambda = 12, lambda_min_ratio = 0.03) {
  init <- init_from_kmeans(X, K, nstart = 10)
  R <- init$R
  N <- colSums(R) + 1e-8
  p <- ncol(X)
  K <- length(N)
  raw <- matrix(0, K, p)
  for (k in 1:p) {
    xbar <- numeric(K)
    for (j in 1:K) xbar[j] <- sum(R[, j] * X[, k]) / N[j]
    d <- xbar - mean(xbar)
    raw[, k] <- abs(d) * N
  }
  lam_max <- max(raw)
  lam_max <- max(lam_max, 1e-3)
  exp(seq(log(lam_max), log(lam_max * lambda_min_ratio), length.out = nlambda))
}

fit_szl_refit_path <- function(X, K, lambda_grid, alpha = 0.5,
                               adaptive_w = NULL, init = NULL,
                               tau = 1e-4, verbose = FALSE) {
  n <- nrow(X)
  p <- ncol(X)
  
  path        <- vector("list", length(lambda_grid))
  ebic_refit <- rep(NA, length(lambda_grid))
  ebic_lasso <- rep(NA, length(lambda_grid))
  
  for (m in seq_along(lambda_grid)) {
    lam <- lambda_grid[m]
    if (verbose) cat("  lambda", m, "/", length(lambda_grid),
                     "=", round(lam, 3), "\n")
    
    # Fresh kmeans init at every lambda. No warm-start.
    fit_lasso <- em_sz_lasso(X, K, lam, adaptive_w = adaptive_w, init = NULL)
    S <- select_support_maxcontrast(fit_lasso$mu, tau = tau)
    fit_refit <- em_refit_support(X, K, S, init = fit_lasso)
    
    df_eff <- (K - 1) * length(S)
    pen <- log(n) * df_eff + 2 * alpha * length(S) * log(p)
    
    ebic_refit[m] <- -2 * fit_refit$loglik + pen
    ebic_lasso[m] <- -2 * fit_lasso$loglik + pen
    
    path[[m]] <- list(lambda = lam, fit_lasso = fit_lasso,
                      S = S, fit_refit = fit_refit,
                      ebic_refit = ebic_refit[m],
                      ebic_lasso = ebic_lasso[m])
  }
  
  best_refit_idx <- which.min(ebic_refit)
  best_lasso_idx <- which.min(ebic_lasso)
  
  list(best_refit = path[[best_refit_idx]],
       best_lasso = path[[best_lasso_idx]],
       path = path, ebic_refit = ebic_refit, ebic_lasso = ebic_lasso,
       best_refit_idx = best_refit_idx,
       best_lasso_idx = best_lasso_idx)
}

fit_aszl_refit_path <- function(X, K, lambda_grid, pilot_delta,
                                gamma = 1, eps = 1e-3, weight_cap = 100,
                                alpha = 0.5, tau = 1e-4, verbose = FALSE) {
  adaptive_w <- (abs(pilot_delta) + eps)^(-gamma)
  adaptive_w <- pmin(adaptive_w, weight_cap)
  fit_szl_refit_path(X, K, lambda_grid, alpha = alpha,
                     adaptive_w = adaptive_w, tau = tau, verbose = verbose)
}

path_diagnostics <- function(fit_path) {
  data.frame(
    idx = seq_along(fit_path$path),
    lambda = sapply(fit_path$path, function(x) x$lambda),
    Shat = sapply(fit_path$path, function(x) length(x$S)),
    loglik_lasso = sapply(fit_path$path, function(x) x$fit_lasso$loglik),
    loglik_refit = sapply(fit_path$path, function(x) x$fit_refit$loglik),
    ebic_lasso = fit_path$ebic_lasso,
    ebic_refit = fit_path$ebic_refit
  )
}

############################################################
# 7. Additional benchmark methods
############################################################

fit_kmeans_baseline <- function(X, K, nstart = 20) {
  km <- kmeans(X, centers = K, nstart = nstart, iter.max = 100)
  fit_from_labels(X, km$cluster, K)
}

fit_pca_kmeans <- function(X, K, var_explained = 0.80, nstart = 20) {
  pc <- prcomp(X, center = TRUE, scale. = FALSE)
  eig <- pc$sdev^2
  cum <- cumsum(eig) / sum(eig)
  nPC <- max(K - 1, which(cum >= var_explained)[1])
  nPC <- min(nPC, ncol(pc$x))
  km <- kmeans(pc$x[, 1:nPC, drop = FALSE], centers = K,
               nstart = nstart, iter.max = 100)
  fit <- fit_from_labels(X, km$cluster, K)
  fit$p_fit <- nPC
  fit
}

fit_unpenalized_gmm <- function(X, K, nstart = 10) {
  em_refit_support(X, K, S = 1:ncol(X), init = NULL, nstart = nstart)
}

feature_r2 <- function(X, labels, K) {
  n <- nrow(X)
  p <- ncol(X)
  out <- rep(0, p)
  grand <- colMeans(X)
  for (k in 1:p) {
    tss <- sum((X[, k] - grand[k])^2)
    if (tss <= 1e-12) {
      out[k] <- 0
    } else {
      bss <- 0
      for (j in 1:K) {
        idx <- which(labels == j)
        if (length(idx) > 0) {
          bss <- bss + length(idx) * (mean(X[idx, k]) - grand[k])^2
        }
      }
      out[k] <- bss / tss
    }
  }
  out
}

spectral_cluster <- function(X, K, nstart = 10) {
  n <- nrow(X)
  if (n <= K) return(rep(1:K, length.out = n))
  D <- as.matrix(dist(X))
  dpos <- D[D > 0]
  sigma <- ifelse(length(dpos) > 0, median(dpos), 1)
  if (!is.finite(sigma) || sigma <= 1e-12) sigma <- 1
  A <- exp(-(D^2) / (2 * sigma^2))
  diag(A) <- 0
  deg <- rowSums(A)
  deg <- pmax(deg, 1e-12)
  L <- diag(1 / sqrt(deg)) %*% A %*% diag(1 / sqrt(deg))
  eg <- eigen(L, symmetric = TRUE)
  U <- eg$vectors[, 1:K, drop = FALSE]
  nr <- sqrt(rowSums(U^2))
  U <- U / pmax(nr, 1e-12)
  kmeans(U, centers = K, nstart = nstart, iter.max = 100)$cluster
}

choose_support_by_bic_from_rank <- function(X, K, labels_init, rank_scores,
                                            m_grid = NULL, alpha = 0.5,
                                            final_cluster = c("kmeans", "spectral", "gmm_refit"),
                                            nstart = 10) {
  final_cluster <- match.arg(final_cluster)
  n <- nrow(X)
  p <- ncol(X)
  if (is.null(m_grid)) {
    m_grid <- unique(round(c(K, 5, 8, 10, 15, 20, sqrt(p), p / 20, p / 10)))
    m_grid <- m_grid[m_grid >= 1 & m_grid <= p]
  }
  m_grid <- sort(unique(pmax(1, pmin(p, m_grid))))
  ord <- order(rank_scores, decreasing = TRUE)
  best_bic <- Inf
  best_fit <- NULL
  best_S <- integer(0)
  
  for (m in m_grid) {
    S <- sort(ord[1:m])
    if (final_cluster == "spectral") {
      labels <- spectral_cluster(X[, S, drop = FALSE], K, nstart = nstart)
      fit <- fit_from_labels_support(X, labels, K, S)
    } else if (final_cluster == "gmm_refit") {
      fit0 <- fit_from_labels_support(X, labels_init, K, S)
      fit <- em_refit_support(X, K, S, init = fit0, nstart = nstart)
    } else {
      km <- kmeans(X[, S, drop = FALSE], centers = K, nstart = nstart, iter.max = 100)
      fit <- fit_from_labels_support(X, km$cluster, K, S)
    }
    bic <- bic_support_fit(fit, n, p, K, alpha = alpha)
    if (bic < best_bic) {
      best_bic <- bic
      best_fit <- fit
      best_S <- S
    }
  }
  best_fit$S <- best_S
  best_fit$bic <- best_bic
  best_fit
}

fit_scfs_proxy <- function(X, K, alpha = 0.5, m_grid = NULL, nstart = 10) {
  labels0 <- spectral_cluster(X, K, nstart = nstart)
  r2 <- feature_r2(X, labels0, K)
  fit <- choose_support_by_bic_from_rank(X, K, labels0, r2,
                                         m_grid = m_grid, alpha = alpha,
                                         final_cluster = "spectral",
                                         nstart = nstart)
  fit$method_detail <- "SC-FS proxy: spectral init -> top R2 features -> spectral reclustering"
  fit
}

fit_sparse_kmeans_proxy <- function(X, K, alpha = 0.5, m_grid = NULL, nstart = 20) {
  km0 <- kmeans(X, centers = K, nstart = nstart, iter.max = 100)
  r2 <- feature_r2(X, km0$cluster, K)
  fit <- choose_support_by_bic_from_rank(X, K, km0$cluster, r2,
                                         m_grid = m_grid, alpha = alpha,
                                         final_cluster = "kmeans",
                                         nstart = nstart)
  fit$method_detail <- "Sparse K-means proxy: kmeans ranking by R2, BIC-selected top features"
  fit
}

fit_sparse_kmeans <- function(X, K, alpha = 0.5, wbounds_grid = NULL,
                              nstart = 20, weight_tol = 1e-8) {
  n <- nrow(X)
  p <- ncol(X)
  if (is.null(wbounds_grid)) {
    wbounds_grid <- unique(seq(1.2, sqrt(p), length.out = 8))
    wbounds_grid <- wbounds_grid[wbounds_grid > 1]
  }
  
  if (!requireNamespace("sparcl", quietly = TRUE)) {
    fit <- fit_sparse_kmeans_proxy(X, K, alpha = alpha, nstart = nstart)
    fit$method_name <- "Sparse_Kmeans_proxy"
    return(fit)
  }
  
  Xs <- scale(X)
  obj <- tryCatch(
    sparcl::KMeansSparseCluster(Xs, K = K, wbounds = wbounds_grid,
                                nstart = nstart, silent = TRUE),
    error = function(e) NULL
  )
  if (is.null(obj)) {
    fit <- fit_sparse_kmeans_proxy(X, K, alpha = alpha, nstart = nstart)
    fit$method_name <- "Sparse_Kmeans_proxy"
    return(fit)
  }
  
  fits <- obj
  if (!is.list(obj) || (!is.null(obj$Cs) && !is.null(obj$ws))) {
    fits <- list(obj)
  }
  
  best_bic <- Inf
  best_fit <- NULL
  for (f in fits) {
    if (is.null(f$Cs) || is.null(f$ws)) next
    labels <- as.integer(f$Cs)
    S <- which(abs(f$ws) > weight_tol)
    if (length(S) == 0) S <- which.max(abs(f$ws))
    fit <- fit_from_labels_support(X, labels, K, S)
    bic <- bic_support_fit(fit, n, p, K, alpha = alpha)
    if (bic < best_bic) {
      best_bic <- bic
      best_fit <- fit
      best_fit$weights <- f$ws
      best_fit$S <- S
      best_fit$bic <- bic
    }
  }
  if (is.null(best_fit)) {
    fit <- fit_sparse_kmeans_proxy(X, K, alpha = alpha, nstart = nstart)
    fit$method_name <- "Sparse_Kmeans_proxy"
    return(fit)
  }
  best_fit$method_name <- "Sparse_Kmeans"
  best_fit
}

find_vector_recursive <- function(obj, len, numeric_ok = TRUE, depth = 0, max_depth = 4) {
  if (depth > max_depth) return(NULL)
  if ((is.numeric(obj) || is.integer(obj) || is.factor(obj)) && length(obj) == len) {
    return(obj)
  }
  if (is.list(obj)) {
    for (nm in names(obj)) {
      val <- find_vector_recursive(obj[[nm]], len, numeric_ok, depth + 1, max_depth)
      if (!is.null(val)) return(val)
    }
  }
  NULL
}

fit_selvarmix_proxy <- function(X, K, alpha = 0.5, m_grid = NULL, nstart = 10) {
  # Model-based variable-selection proxy:
  # kmeans labels -> rank by R2 -> EBIC over top features -> GMM refit on selected support.
  km0 <- kmeans(X, centers = K, nstart = nstart, iter.max = 100)
  r2 <- feature_r2(X, km0$cluster, K)
  fit <- choose_support_by_bic_from_rank(X, K, km0$cluster, r2,
                                         m_grid = m_grid, alpha = alpha,
                                         final_cluster = "gmm_refit",
                                         nstart = nstart)
  fit$method_detail <- "SelvarMix proxy: ranking + model-based support refit"
  fit
}

fit_selvarmix_optional <- function(X, K, alpha = 0.5, nstart = 10) {
  if (!requireNamespace("SelvarMix", quietly = TRUE)) {
    fit <- fit_selvarmix_proxy(X, K, alpha = alpha, nstart = nstart)
    fit$method_name <- "SelvarMix_proxy"
    return(fit)
  }
  obj <- tryCatch(
    SelvarMix::SelvarClustLasso(x = X, nbcluster = K, nbcores = 1),
    error = function(e) NULL
  )
  if (is.null(obj)) {
    fit <- fit_selvarmix_proxy(X, K, alpha = alpha, nstart = nstart)
    fit$method_name <- "SelvarMix_proxy"
    return(fit)
  }
  labels <- find_vector_recursive(obj, nrow(X))
  S <- find_vector_recursive(obj, ncol(X))
  if (is.null(labels)) {
    fit <- fit_selvarmix_proxy(X, K, alpha = alpha, nstart = nstart)
    fit$method_name <- "SelvarMix_proxy"
    return(fit)
  }
  labels <- as.integer(as.factor(labels))
  if (length(unique(labels)) != K) {
    fit <- fit_selvarmix_proxy(X, K, alpha = alpha, nstart = nstart)
    fit$method_name <- "SelvarMix_proxy"
    return(fit)
  }
  if (is.null(S)) {
    # If selected variables cannot be extracted, infer support from full-space labels.
    r2 <- feature_r2(X, labels, K)
    S <- order(r2, decreasing = TRUE)[1:min(10, ncol(X))]
  } else {
    if (is.logical(S)) S <- which(S)
    S <- as.integer(S)
    S <- S[S >= 1 & S <= ncol(X)]
  }
  fit <- fit_from_labels_support(X, labels, K, S)
  fit$method_name <- "SelvarMix"
  fit$raw_object <- obj
  fit
}

############################################################
# 8. Metrics
############################################################

compute_metrics <- function(fit, data, S_hat = NULL, selection_applicable = TRUE) {
  X <- data$X
  z <- data$z
  mu_true <- data$mu
  delta_true <- data$delta
  S0 <- data$S0
  K <- nrow(mu_true)
  p <- ncol(mu_true)
  
  mu_hat <- align_mu_to_truth(fit$mu, mu_true)
  delta_hat <- mu_hat - matrix(colMeans(mu_hat), K, p, byrow = TRUE)
  
  if (selection_applicable) {
    if (is.null(S_hat)) {
      if (!is.null(fit$S)) S_hat <- fit$S else S_hat <- select_support_maxcontrast(fit$mu)
    }
    TP <- length(intersect(S_hat, S0))
    FP <- length(setdiff(S_hat, S0))
    FN <- length(setdiff(S0, S_hat))
    TPR <- TP / length(S0)
    FPR <- ifelse((p - length(S0)) > 0, FP / (p - length(S0)), 0)
    Shat <- length(S_hat)
  } else {
    TPR <- NA_real_
    FPR <- NA_real_
    Shat <- NA_real_
  }
  
  Rk <- rep(NA, length(S0))
  for (ii in seq_along(S0)) {
    k <- S0[ii]
    denom <- sqrt(sum(delta_true[, k]^2))
    Rk[ii] <- sqrt(sum(delta_hat[, k]^2)) / denom
  }
  
  list(
    ARI = adjusted_rand_index(z, fit$labels),
    TPR = TPR,
    FPR = FPR,
    Shat = Shat,
    MSE_mu = sum((mu_hat - mu_true)^2) / (K * p),
    MSE_delta_S = sum((delta_hat[, S0, drop = FALSE] - delta_true[, S0, drop = FALSE])^2) / (K * length(S0)),
    R_mean = mean(Rk),
    R_median = median(Rk),
    Entropy = entropy_mean(fit$R)
  )
}

metrics_to_row <- function(group, method, met, p, a, rep_id, p_fit = NA_real_) {
  data.frame(
    group = group,
    method = method,
    p = p,
    a = a,
    rep = rep_id,
    p_fit = p_fit,
    ARI = met$ARI,
    TPR = met$TPR,
    FPR = met$FPR,
    Shat = met$Shat,
    MSE_mu = met$MSE_mu,
    MSE_delta_S = met$MSE_delta_S,
    R_mean = met$R_mean,
    R_median = met$R_median,
    Entropy = met$Entropy
  )
}

summarize_results <- function(res) {
  num_cols <- c("p_fit", "ARI", "TPR", "FPR", "Shat", "MSE_mu", "MSE_delta_S",
                "R_mean", "R_median", "Entropy")
  split_res <- split(res, list(res$p, res$a, res$method), drop = TRUE)
  out <- lapply(split_res, function(df) {
    vals <- sapply(num_cols, function(v) safe_mean(df[[v]]))
    ses <- sapply(num_cols, function(v) safe_se(df[[v]]))
    data.frame(
      group = df$group[1],
      p = df$p[1],
      a = df$a[1],
      method = df$method[1],
      t(vals),
      t(setNames(ses, paste0(num_cols, "_SE")))
    )
  })
  do.call(rbind, out)
}


############################################################
# 8. Actual package benchmark wrappers
############################################################

# Optional helper. Run manually if the packages are not installed.
install_benchmark_packages <- function(install_scfs = TRUE) {
  if (!requireNamespace("sparcl", quietly = TRUE)) {
    install.packages("sparcl")
  }
  if (!requireNamespace("SelvarMix", quietly = TRUE)) {
    install.packages("SelvarMix", repos = "http://R-Forge.R-project.org")
  }
  if (install_scfs && !requireNamespace("SCFS", quietly = TRUE)) {
    if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
    remotes::install_github("TerenceLiu4444/SCFS")
  }
}

scale_no_na <- function(X) {
  Xs <- scale(X)
  Xs[is.na(Xs)] <- 0
  Xs
}

normalize_support_vector <- function(S, p, cn = NULL) {
  if (is.null(S)) return(integer(0))
  if (is.logical(S)) {
    if (length(S) == p) return(which(S))
    S <- which(S)
  }
  if (is.factor(S)) S <- as.character(S)
  if (is.character(S)) {
    if (!is.null(cn)) {
      m <- match(S, cn)
      S <- m[!is.na(m)]
    } else {
      S <- suppressWarnings(as.integer(gsub("[^0-9]", "", S)))
    }
  }
  S <- suppressWarnings(as.integer(S))
  S <- S[!is.na(S)]
  S <- S[S >= 1 & S <= p]
  sort(unique(S))
}

find_named_recursive <- function(obj, target_names, depth = 0, max_depth = 8) {
  if (depth > max_depth) return(NULL)
  if (!is.list(obj)) return(NULL)
  nms <- names(obj)
  if (!is.null(nms)) {
    hit <- which(tolower(nms) %in% tolower(target_names))
    if (length(hit) > 0) return(obj[[hit[1]]])
  }
  for (ii in seq_along(obj)) {
    val <- find_named_recursive(obj[[ii]], target_names, depth + 1, max_depth)
    if (!is.null(val)) return(val)
  }
  NULL
}

# --- Sparse K-means: official package, EBIC over wbounds grid ---
fit_sparse_kmeans_pkg_ebic <- function(X, K, alpha = 0.5, wbounds_grid = NULL,
                                       nstart = 20, weight_tol = 1e-8) {
  n <- nrow(X); p <- ncol(X)
  if (!requireNamespace("sparcl", quietly = TRUE)) return(NULL)
  if (is.null(wbounds_grid)) {
    wbounds_grid <- unique(seq(1.2, sqrt(p), length.out = 10))
    wbounds_grid <- wbounds_grid[wbounds_grid > 1]
  }
  Xs <- scale_no_na(X)
  obj <- tryCatch(
    sparcl::KMeansSparseCluster(Xs, K = K, wbounds = wbounds_grid,
                                nstart = nstart, silent = TRUE),
    error = function(e) NULL
  )
  if (is.null(obj)) return(NULL)
  fits <- obj
  if (!is.list(obj) || (!is.null(obj$Cs) && !is.null(obj$ws))) fits <- list(obj)
  best_bic <- Inf; best_fit <- NULL
  for (f in fits) {
    if (is.null(f$Cs) || is.null(f$ws)) next
    labels <- as.integer(f$Cs)
    S <- which(abs(f$ws) > weight_tol)
    if (length(S) == 0) S <- which.max(abs(f$ws))
    fit <- fit_from_labels_support(X, labels, K, S)
    bic <- bic_support_fit(fit, n, p, K, alpha = alpha)
    if (bic < best_bic) {
      best_bic <- bic
      best_fit <- fit
      best_fit$weights <- f$ws
      best_fit$S <- S
      best_fit$bic <- bic
    }
  }
  if (is.null(best_fit)) return(NULL)
  best_fit$method_name <- "Sparse_Kmeans_pkg_EBIC"
  best_fit
}

# --- Sparse K-means: official package, permutation/gap tuning ---
fit_sparse_kmeans_pkg_gap <- function(X, K, wbounds_grid = NULL, nperms = 5,
                                      nstart = 20, weight_tol = 1e-8) {
  p <- ncol(X)
  if (!requireNamespace("sparcl", quietly = TRUE)) return(NULL)
  if (is.null(wbounds_grid)) {
    wbounds_grid <- unique(seq(1.2, sqrt(p), length.out = 10))
    wbounds_grid <- wbounds_grid[wbounds_grid > 1]
  }
  Xs <- scale_no_na(X)
  perm <- tryCatch(
    sparcl::KMeansSparseCluster.permute(Xs, K = K, nperms = nperms,
                                        wbounds = wbounds_grid, silent = TRUE),
    error = function(e) NULL
  )
  if (is.null(perm) || is.null(perm$bestw)) return(NULL)
  obj <- tryCatch(
    sparcl::KMeansSparseCluster(Xs, K = K, wbounds = perm$bestw,
                                nstart = nstart, silent = TRUE),
    error = function(e) NULL
  )
  if (is.null(obj) || is.null(obj$Cs) || is.null(obj$ws)) return(NULL)
  labels <- as.integer(obj$Cs)
  S <- which(abs(obj$ws) > weight_tol)
  if (length(S) == 0) S <- which.max(abs(obj$ws))
  fit <- fit_from_labels_support(X, labels, K, S)
  fit$S <- S
  fit$weights <- obj$ws
  fit$bestw <- perm$bestw
  fit$perm_object <- perm
  fit$method_name <- "Sparse_Kmeans_pkg_Gap"
  fit
}

# --- SC-FS: actual GitHub package if installed ---
fit_scfs_pkg <- function(X, K, use_lloyd_iteration = TRUE, scor_thresh = 0.9,
                         scale_input = TRUE) {
  if (!requireNamespace("SCFS", quietly = TRUE)) return(NULL)
  Xs <- if (scale_input) scale_no_na(X) else X
  obj <- tryCatch(
    SCFS::SpectralClusterFeatureSelection(
      data = Xs,
      num_clusters = K,
      init_cluster_ids = NULL,
      use_lloyd_iteration = use_lloyd_iteration,
      scor_thresh = scor_thresh
    ),
    error = function(e) NULL
  )
  if (is.null(obj)) return(NULL)
  labels <- obj$cluster_ids
  S <- normalize_support_vector(obj$info_feat_ids, ncol(X), colnames(X))
  if (is.null(labels) || length(labels) != nrow(X) || length(S) == 0) return(NULL)
  labels <- as.integer(as.factor(labels))
  fit <- fit_from_labels_support(X, labels, K, S)
  fit$S <- S
  fit$raw_object <- obj
  fit$method_name <- if (use_lloyd_iteration) "SCFS_pkg_Lloyd" else "SCFS_pkg"
  fit
}

# --- SelvarMix: actual R-Forge package if installed ---
# SelvarMix has role-specific outputs:
#   S: clustering-relevant variables selected by the package
#   R: redundant variables
#   U: variables useful for explaining clustering variables
#   W: irrelevant/noise variables
# Our simulation target S0 = {k: delta_.k != 0} is closer to non-W = {1,...,p}\W
# than to strict S. Therefore, we report actual-package variants:
#   SelvarMix_pkg_S          : strict S support, package partition
#   SelvarMix_pkg_nonW       : non-W support, package partition
#   SelvarMix_nonW_Refit     : non-W support + our unpenalized GMM refit

extract_selvarmix_roles <- function(obj, p) {
  get_int <- function(x) {
    if (is.null(x)) return(integer(0))
    if (is.logical(x)) x <- which(x)
    if (is.factor(x)) x <- as.character(x)
    if (is.character(x)) x <- suppressWarnings(as.integer(gsub("[^0-9]", "", x)))
    x <- suppressWarnings(as.integer(x))
    x <- x[!is.na(x) & x >= 1 & x <= p]
    sort(unique(x))
  }
  
  S <- get_int(obj$S)
  R <- get_int(obj$R)
  U <- get_int(obj$U)
  W <- get_int(obj$W)
  nonW <- setdiff(seq_len(p), W)
  SRU <- sort(unique(c(S, R, U)))
  
  list(S = S, R = R, U = U, W = W, SRU = SRU, nonW = nonW)
}

fit_selvarmix_actual_roles <- function(X, K, criterion = "BIC", nbcores = 1) {
  if (!requireNamespace("SelvarMix", quietly = TRUE)) return(NULL)
  
  Xdf <- as.data.frame(X)
  colnames(Xdf) <- paste0("V", seq_len(ncol(Xdf)))
  
  obj <- tryCatch(
    SelvarMix::SelvarClustLasso(x = Xdf, nbcluster = K,
                                criterion = criterion, nbcores = nbcores),
    error = function(e) NULL
  )
  if (is.null(obj)) return(NULL)
  if (is.null(obj$partition) || length(obj$partition) != nrow(X)) return(NULL)
  
  labels <- as.integer(as.factor(obj$partition))
  if (length(unique(labels)) != K) return(NULL)
  
  proba <- obj$proba
  roles <- extract_selvarmix_roles(obj, p = ncol(X))
  out <- list()
  
  attach_common <- function(fit, method_name) {
    if (!is.null(proba) && is.matrix(proba) && nrow(proba) == nrow(X) && ncol(proba) == K) {
      fit$R <- proba / pmax(rowSums(proba), 1e-15)
      fit$labels <- max.col(fit$R)
    }
    fit$raw_object <- obj
    fit$roles <- roles
    fit$method_name <- method_name
    fit
  }
  
  # 1) Strict S: package's parsimonious clustering-relevant variables.
  S_strict <- roles$S
  fit_S <- fit_from_labels_support(X, labels, K, S_strict)
  fit_S$S <- S_strict
  out$S <- attach_common(fit_S, "SelvarMix_pkg_S")
  
  # 2) non-W: variables not classified as irrelevant/noise by SelvarMix.
  S_nonW <- roles$nonW
  fit_nonW <- fit_from_labels_support(X, labels, K, S_nonW)
  fit_nonW$S <- S_nonW
  out$nonW <- attach_common(fit_nonW, "SelvarMix_pkg_nonW")
  
  # 3) non-W + unpenalized refit: sensitivity aligned with our refit framework.
  init_nonW <- fit_from_labels_support(X, labels, K, S_nonW)
  fit_nonW_refit <- em_refit_support(X, K, S_nonW, init = init_nonW)
  fit_nonW_refit$S <- S_nonW
  out$nonW_refit <- attach_common(fit_nonW_refit, "SelvarMix_nonW_Refit")
  
  # 4) SRU union: explicit role union; usually identical to non-W if W is exhaustive.
  S_SRU <- roles$SRU
  if (!setequal(S_SRU, S_nonW)) {
    fit_SRU <- fit_from_labels_support(X, labels, K, S_SRU)
    fit_SRU$S <- S_SRU
    out$SRU <- attach_common(fit_SRU, "SelvarMix_pkg_SRU")
  }
  
  out
}

# Backward-compatible wrapper. It returns strict S by default.
fit_selvarmix_pkg <- function(X, K, criterion = "BIC", nbcores = 1) {
  obj_list <- fit_selvarmix_actual_roles(X, K, criterion = criterion, nbcores = nbcores)
  if (is.null(obj_list)) return(NULL)
  obj_list$S
}

############################################################
# 9. One simulation replicate: actual packages vs proxies
############################################################

run_one_rep <- function(n = 300, p = 100, q = 5, a = 1.2, K = 3,
                        nlambda = 10, alpha = 0.5, tau = 1e-4,
                        rep_id = 1, verbose = FALSE,
                        include_traditional = TRUE,
                        include_proxy_benchmarks = TRUE,
                        include_actual_packages = TRUE,
                        include_sparse_kmeans = TRUE,
                        include_scfs = TRUE,
                        include_selvarmix = TRUE,
                        sparse_gap_nperms = 5) {
  data <- make_data(n = n, p = p, q = q, a = a, K = K)
  X <- data$X
  rows <- list()
  row_counter <- 1

  add_row <- function(group, method, fit, S_hat = NULL,
                      selection_applicable = TRUE, p_fit = NA_real_) {
    met <- compute_metrics(fit, data, S_hat = S_hat,
                           selection_applicable = selection_applicable)
    rows[[row_counter]] <<- metrics_to_row(group, method, met, p, a, rep_id, p_fit = p_fit)
    row_counter <<- row_counter + 1
  }

  ##########################################################
  # Traditional benchmarks
  ##########################################################
  if (include_traditional) {
    km_fit <- fit_kmeans_baseline(X, K)
    add_row("Traditional", "Kmeans", km_fit,
            selection_applicable = FALSE, p_fit = p)

    pca_fit <- fit_pca_kmeans(X, K, var_explained = 0.80)
    add_row("Traditional", "PCA_Kmeans", pca_fit,
            selection_applicable = FALSE, p_fit = pca_fit$p_fit)

    gmm_fit <- fit_unpenalized_gmm(X, K)
    add_row("Model_based", "Unpenalized_GMM", gmm_fit,
            selection_applicable = FALSE, p_fit = p)
  }

  ##########################################################
  # Sparse K-means: proxy vs actual package
  ##########################################################
  if (include_sparse_kmeans) {
    if (include_proxy_benchmarks) {
      skm_proxy <- fit_sparse_kmeans_proxy(X, K, alpha = alpha)
      add_row("Sparse_clustering_proxy", "Sparse_Kmeans_proxy_EBIC",
              skm_proxy, S_hat = skm_proxy$S,
              selection_applicable = TRUE, p_fit = length(skm_proxy$S))
    }
    if (include_actual_packages) {
      skm_pkg_ebic <- fit_sparse_kmeans_pkg_ebic(X, K, alpha = alpha)
      if (!is.null(skm_pkg_ebic)) {
        add_row("Sparse_clustering_pkg", "Sparse_Kmeans_pkg_EBIC",
                skm_pkg_ebic, S_hat = skm_pkg_ebic$S,
                selection_applicable = TRUE, p_fit = length(skm_pkg_ebic$S))
      } else if (verbose) cat("  [skip] sparcl package EBIC result unavailable\n")

      skm_pkg_gap <- fit_sparse_kmeans_pkg_gap(X, K, nperms = sparse_gap_nperms)
      if (!is.null(skm_pkg_gap)) {
        add_row("Sparse_clustering_pkg", "Sparse_Kmeans_pkg_Gap",
                skm_pkg_gap, S_hat = skm_pkg_gap$S,
                selection_applicable = TRUE, p_fit = length(skm_pkg_gap$S))
      } else if (verbose) cat("  [skip] sparcl package Gap result unavailable\n")
    }
  }

  ##########################################################
  # SC-FS: proxy vs actual package
  ##########################################################
  if (include_scfs) {
    if (include_proxy_benchmarks) {
      scfs_proxy <- fit_scfs_proxy(X, K, alpha = alpha)
      add_row("Spectral_screening_proxy", "SCFS_proxy_EBIC",
              scfs_proxy, S_hat = scfs_proxy$S,
              selection_applicable = TRUE, p_fit = length(scfs_proxy$S))
    }
    if (include_actual_packages) {
      scfs_pkg <- fit_scfs_pkg(X, K, use_lloyd_iteration = TRUE, scor_thresh = 0.9)
      if (!is.null(scfs_pkg)) {
        add_row("Spectral_screening_pkg", scfs_pkg$method_name,
                scfs_pkg, S_hat = scfs_pkg$S,
                selection_applicable = TRUE, p_fit = length(scfs_pkg$S))
      } else if (verbose) cat("  [skip] SCFS package result unavailable\n")
    }
  }

  ##########################################################
  # SelvarMix: proxy vs actual package
  ##########################################################
  if (include_selvarmix) {
    if (include_proxy_benchmarks) {
      sel_proxy <- fit_selvarmix_proxy(X, K, alpha = alpha)
      add_row("Model_based_VS_proxy", "SelvarMix_proxy_EBIC",
              sel_proxy, S_hat = sel_proxy$S,
              selection_applicable = TRUE, p_fit = length(sel_proxy$S))
    }
    if (include_actual_packages) {
      sel_roles <- fit_selvarmix_actual_roles(X, K, criterion = "BIC", nbcores = 1)
      if (!is.null(sel_roles)) {
        # Strict S: SelvarMix's parsimonious clustering variables.
        add_row("Model_based_VS_pkg", sel_roles$S$method_name,
                sel_roles$S, S_hat = sel_roles$S$S,
                selection_applicable = TRUE,
                p_fit = length(sel_roles$S$S))
        
        # non-W: variables not classified as irrelevant/noise.
        add_row("Model_based_VS_pkg", sel_roles$nonW$method_name,
                sel_roles$nonW, S_hat = sel_roles$nonW$S,
                selection_applicable = TRUE,
                p_fit = length(sel_roles$nonW$S))
        
        # non-W + our unpenalized refit, useful as sensitivity.
        add_row("Model_based_VS_pkg", sel_roles$nonW_refit$method_name,
                sel_roles$nonW_refit, S_hat = sel_roles$nonW_refit$S,
                selection_applicable = TRUE,
                p_fit = length(sel_roles$nonW_refit$S))
        
        # If SRU differs from non-W, also report SRU.
        if (!is.null(sel_roles$SRU)) {
          add_row("Model_based_VS_pkg", sel_roles$SRU$method_name,
                  sel_roles$SRU, S_hat = sel_roles$SRU$S,
                  selection_applicable = TRUE,
                  p_fit = length(sel_roles$SRU$S))
        }
      } else if (verbose) cat("  [skip] SelvarMix package result unavailable\n")
    }
  }

  ##########################################################
  # Proposed SZL family
  ##########################################################
  lambda_grid <- make_lambda_grid(X, K, nlambda = nlambda)

  if (verbose) cat("Plain SZL path\n")
  szl <- fit_szl_refit_path(X, K, lambda_grid, alpha = alpha, tau = tau,
                            verbose = verbose)

  naive_atrlam_fit <- szl$best_refit$fit_lasso
  naive_atrlam_S   <- szl$best_refit$S
  naive_self_fit   <- szl$best_lasso$fit_lasso
  naive_self_S     <- szl$best_lasso$S
  szl_refit_fit    <- szl$best_refit$fit_refit
  szl_refit_S      <- szl$best_refit$S

  if (verbose) cat("Adaptive SZL path\n")
  aszl <- fit_aszl_refit_path(X, K, lambda_grid,
                              pilot_delta = naive_atrlam_fit$delta,
                              alpha = alpha, tau = tau, verbose = verbose)
  aszl_refit_fit <- aszl$best_refit$fit_refit
  aszl_refit_S   <- aszl$best_refit$S

  add_row("Ablation", "Naive_Lasso_at_refit_lambda",
          naive_atrlam_fit, S_hat = naive_atrlam_S,
          selection_applicable = TRUE, p_fit = p)

  add_row("Penalized_GMM", "Naive_Lasso_self_tuned",
          naive_self_fit, S_hat = naive_self_S,
          selection_applicable = TRUE, p_fit = p)

  add_row("Proposed", "SZL_Refit",
          szl_refit_fit, S_hat = szl_refit_S,
          selection_applicable = TRUE, p_fit = p)

  add_row("Proposed_auxiliary", "ASZL_Refit",
          aszl_refit_fit, S_hat = aszl_refit_S,
          selection_applicable = TRUE, p_fit = p)

  ##########################################################
  # Oracles
  ##########################################################
  oracle_fit <- em_refit_support(X, K, data$S0, init = NULL)
  add_row("Oracle", "Oracle_feature_GMM",
          oracle_fit, S_hat = data$S0,
          selection_applicable = TRUE, p_fit = length(data$S0))

  R_true <- estep_gmm(X, data$pi, data$mu)
  true_oracle_fit <- list(pi = data$pi,
                          mu = data$mu,
                          delta = data$delta,
                          R = R_true,
                          labels = max.col(R_true),
                          loglik = loglik_gmm(X, data$pi, data$mu))
  add_row("Oracle", "True_parameter_oracle",
          true_oracle_fit, S_hat = data$S0,
          selection_applicable = TRUE, p_fit = length(data$S0))

  list(rows = do.call(rbind, rows), data = data, szl = szl, aszl = aszl,
       oracle_fit = oracle_fit, true_oracle_fit = true_oracle_fit)
}

############################################################
# 10. Main simulation loop
############################################################

run_simulation <- function(R_rep = 10,
                           n = 300,
                           p_list = c(100, 300),
                           a_list = c(1.6, 1.4, 1.2),
                           K = 3,
                           nlambda = 10,
                           alpha = 0.5,
                           tau = 1e-4,
                           verbose = TRUE,
                           include_traditional = TRUE,
                           include_proxy_benchmarks = TRUE,
                           include_actual_packages = TRUE,
                           include_sparse_kmeans = TRUE,
                           include_scfs = TRUE,
                           include_selvarmix = TRUE,
                           sparse_gap_nperms = 5) {
  all_rows <- list()
  counter <- 1

  for (p in p_list) {
    q <- ifelse(p == 20, 3, 5)
    for (a in a_list) {
      for (r in 1:R_rep) {
        if (verbose) {
          cat("\n=== rep", r, "/", R_rep, "p=", p, "q=", q, "a=", a, "===\n")
        }
        one <- run_one_rep(n = n, p = p, q = q, a = a, K = K,
                           nlambda = nlambda, alpha = alpha, tau = tau,
                           rep_id = r, verbose = FALSE,
                           include_traditional = include_traditional,
                           include_proxy_benchmarks = include_proxy_benchmarks,
                           include_actual_packages = include_actual_packages,
                           include_sparse_kmeans = include_sparse_kmeans,
                           include_scfs = include_scfs,
                           include_selvarmix = include_selvarmix,
                           sparse_gap_nperms = sparse_gap_nperms)
        all_rows[[counter]] <- one$rows
        counter <- counter + 1
      }
    }
  }

  res <- do.call(rbind, all_rows)
  summ <- summarize_results(res)
  list(raw = res, summary = summ)
}

############################################################
# 11. Optional quick run
############################################################

# Before using actual package rows, install packages manually if needed:
# install_benchmark_packages(install_scfs = TRUE)
#
# For the first actual-vs-proxy check, keep R_rep small because
# Sparse_Kmeans_pkg_Gap and SelvarMix can be slow.

sim <- run_simulation(
  R_rep = 10,
  n = 300,
  p_list = c(100),
  a_list = c(1.4),
  K = 3,
  nlambda = 8,
  alpha = 0.5,
  tau = 1e-4,
  verbose = TRUE,
  include_traditional = TRUE,
  include_proxy_benchmarks = TRUE,
  include_actual_packages = TRUE,
  include_sparse_kmeans = TRUE,
  include_scfs = TRUE,
  include_selvarmix = TRUE,
  sparse_gap_nperms = 3
)

print(sim$raw)
print(sim$summary) %>% view()

write.csv(sim$raw, "szl_actual_vs_proxy_raw_results.csv", row.names = FALSE)
write.csv(sim$summary, "szl_actual_vs_proxy_summary_results.csv", row.names = FALSE)

############################################################
# 12. Optional plot
############################################################

boxplot(R_mean ~ method, data = sim$raw,
        las = 2,
        main = "Recovery ratio R_k across methods",
        ylab = "Mean recovery ratio over active variables")
abline(h = 1, lty = 2)
############################################################
# Visualization code for SZL-Refit simulation results
# Project: Debiased Sum-to-Zero Lasso Mixture Clustering
############################################################

# 1. 필요 패키지 로드 및 설치
packages <- c("ggplot2", "dplyr", "tidyr", "stringr", "forcats", "scales", "gridExtra")
for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
}

library(ggplot2)
library(dplyr)
library(tidyr)
library(stringr)
library(forcats)
library(scales)

# 2. 데이터 준비 (sim 객체 확인)
if (!exists("sim")) stop("객체 'sim'이 존재하지 않습니다. 시뮬레이션을 먼저 실행해주세요.")
raw <- sim$raw
summ <- sim$summary

# 출력 디렉토리 생성
plot_dir <- "figures_szl_refit"
if (!dir.exists(plot_dir)) dir.create(plot_dir)

############################################################
# 1. Method labels and ordering: corrected version
############################################################

method_labels <- c(
  "Kmeans" = "K-means",
  "PCA_Kmeans" = "PCA + K-means",
  "Unpenalized_GMM" = "Unpenalized GMM",
  
  "Sparse_Kmeans_proxy_EBIC" = "Sparse K-means\n(Proxy)",
  "Sparse_Kmeans_pkg_EBIC" = "Sparse K-means\n(Package, EBIC)",
  "Sparse_Kmeans_pkg_Gap" = "Sparse K-means\n(Package, Gap)",
  
  "SCFS_proxy_EBIC" = "SCFS\n(Proxy)",
  "SCFS_pkg_Lloyd" = "SCFS\n(Package)",
  
  "SelvarMix_proxy_EBIC" = "SelvarMix\n(Proxy)",
  "SelvarMix_pkg_S" = "SelvarMix\n(Strict S)",
  "SelvarMix_pkg_nonW" = "SelvarMix\n(non-W)",
  "SelvarMix_nonW_Refit" = "SelvarMix\n(non-W + Refit)",
  "SelvarMix_pkg_SRU" = "SelvarMix\n(SRU)",
  
  "Naive_Lasso_at_refit_lambda" = "Naive Lasso\n(Ablation)",
  "Naive_Lasso_self_tuned" = "Naive Lasso\n(Self-tuned)",
  
  "SZL_Refit" = "SZL-Refit\n(Proposed)",
  "ASZL_Refit" = "ASZL-Refit\n(Auxiliary)",
  
  "Oracle_feature_GMM" = "Oracle\n(Feature)",
  "True_parameter_oracle" = "Oracle\n(Parameter)"
)

method_order <- c(
  "Kmeans",
  "PCA_Kmeans",
  "Unpenalized_GMM",
  
  "Sparse_Kmeans_proxy_EBIC",
  "Sparse_Kmeans_pkg_EBIC",
  "Sparse_Kmeans_pkg_Gap",
  
  "SCFS_proxy_EBIC",
  "SCFS_pkg_Lloyd",
  
  "SelvarMix_proxy_EBIC",
  "SelvarMix_pkg_S",
  "SelvarMix_pkg_nonW",
  "SelvarMix_nonW_Refit",
  "SelvarMix_pkg_SRU",
  
  "Naive_Lasso_at_refit_lambda",
  "Naive_Lasso_self_tuned",
  
  "SZL_Refit",
  "ASZL_Refit",
  
  "Oracle_feature_GMM",
  "True_parameter_oracle"
)

prep_plot_data <- function(df) {
  df %>%
    mutate(
      method_label = ifelse(
        method %in% names(method_labels),
        unname(method_labels[method]),
        method
      ),
      method_label = factor(
        method_label,
        levels = unname(method_labels[method_order[method_order %in% names(method_labels)]])
      ),
      scenario = paste0("p=", p, ", a=", a)
    )
}

method_order <- names(method_labels)

# 데이터 가공 함수
prep_plot_data <- function(df) {
  df %>%
    mutate(
      method_label = ifelse(method %in% names(method_labels),
                            method_labels[method],
                            method),
      method_label = factor(method_label, levels = method_labels[method_order]),
      scenario = paste0("p=", p, ", a=", a)
    )
}

raw_p <- prep_plot_data(raw)
summ_p <- prep_plot_data(summ)

# 4. 핵심 분석용 데이터 필터링 (Proposed vs Baselines)
core_methods <- c("Naive_Lasso_at_refit_lambda", "Naive_Lasso_self_tuned", 
                  "SZL_Refit", "ASZL_Refit", "Oracle_feature_GMM", "True_parameter_oracle")

raw_core <- raw_p %>% filter(method %in% core_methods)
summ_core <- summ_p %>% filter(method %in% core_methods)

# ----------------------------------------------------------
# [그림 1] Hero Figure: Recovery Ratio (R_mean) 분포
# ----------------------------------------------------------
p1 <- ggplot(raw_core, aes(x = method_label, y = R_mean, fill = method_label)) +
  geom_boxplot(outlier.shape = 21, width = 0.6, alpha = 0.7) +
  geom_jitter(width = 0.1, alpha = 0.5) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
  facet_wrap(~ scenario) +
  labs(title = "Figure 1. Recovery of Mean-Heterogeneity Effect",
       subtitle = "Lasso shrinks effects (R < 1), SZL-Refit restores them (R ≈ 1)",
       x = NULL, y = expression(R[mean])) +
  theme_bw() + theme(axis.text.x = element_text(angle = 30, hjust = 1), legend.position = "none")

p1
ggsave(file.path(plot_dir, "01_Rmean_recovery.png"), p1, width = 10, height = 6)

# ----------------------------------------------------------
# [그림 2] ARI 비교 (핵심 모델)
# ----------------------------------------------------------
p2 <- ggplot(summ_core, aes(x = method_label, y = ARI, fill = method_label)) +
  geom_col(width = 0.6, alpha = 0.8) +
  geom_errorbar(aes(ymin = ARI - ARI_SE, ymax = ARI + ARI_SE), width = 0.2) +
  facet_wrap(~ scenario) +
  labs(title = "Figure 2. Clustering Performance (ARI)",
       subtitle = "Post-selection Refit closes the oracle gap",
       x = NULL, y = "Adjusted Rand Index") +
  theme_bw() + theme(axis.text.x = element_text(angle = 30, hjust = 1), legend.position = "none")
p2
ggsave(file.path(plot_dir, "02_ARI_comparison.png"), p2, width = 10, height = 6)

# ----------------------------------------------------------
# [그림 3] MSE_delta_S 비교 (효과 추정 정확도)
# ----------------------------------------------------------
p3 <- ggplot(summ_core, aes(x = method_label, y = MSE_delta_S, fill = method_label)) +
  geom_col(width = 0.6, alpha = 0.8) +
  geom_errorbar(aes(ymin = 0, ymax = MSE_delta_S + MSE_delta_S_SE), width = 0.2) +
  facet_wrap(~ scenario) +
  labs(title = "Figure 3. Estimation Error of Mean-Shift Effects",
       subtitle = "Refit significantly reduces estimation MSE",
       x = NULL, y = expression(MSE[Delta*S])) +
  theme_bw() + theme(axis.text.x = element_text(angle = 30, hjust = 1), legend.position = "none")
p3
ggsave(file.path(plot_dir, "03_MSE_delta_S.png"), p3, width = 10, height = 6)

# ----------------------------------------------------------
# [그림 4] 전체 Benchmark ARI 비교 (Ranking)
# ----------------------------------------------------------
summ_bench_clean <- summ_p %>%
  filter(!is.na(ARI)) %>%
  filter(!is.na(method_label)) %>%
  mutate(method_label = fct_drop(method_label))

p4 <- ggplot(summ_p, aes(x = fct_reorder(method_label, ARI), y = ARI)) +
  geom_col(fill = "steelblue", alpha = 0.8) +
  geom_errorbar(aes(ymin = ARI - ARI_SE, ymax = ARI + ARI_SE), width = 0.2) +
  coord_cartesian(ylim = c(0, 0.8))+
  labs(title = "Figure 4. ARI Ranking across All Benchmarks",
       x = "Methods", y = "Adjusted Rand Index") +
  theme_minimal()
p4
ggsave(file.path(plot_dir, "04_full_benchmark_ARI.png"), p4, width = 8, height = 8)

# ----------------------------------------------------------
# [그림 5] Support Recovery (TPR/FPR)
# ----------------------------------------------------------
support_long <- summ_p %>%
  filter(!is.na(TPR_SE)) %>% 
  select(method_label, scenario, TPR, FPR) %>%
  pivot_longer(cols = c(TPR, FPR), names_to = "metric", values_to = "value")

p5 <- ggplot(support_long, aes(x = method_label, y = value, fill = metric)) +
  geom_col(position = "dodge", alpha = 0.8) +
  facet_wrap(~ scenario) +
  labs(title = "Figure 5. Variable Selection Performance",
       subtitle = "TPR and FPR comparisons",
       x = NULL, y = "Rate") +
  theme_bw() + theme(axis.text.x = element_text(angle = 45, hjust = 1))

p5
ggsave(file.path(plot_dir, "05_support_recovery.png"), p5, width = 12, height = 6)

# ----------------------------------------------------------
# [표 1] 미팅용 요약 테이블 생성
# ----------------------------------------------------------
meeting_table <- summ_p %>%
  select(method, ARI, TPR, FPR, R_mean, MSE_delta_S) %>%
  mutate(across(where(is.numeric), ~ round(., 3)))

# gridExtra를 이용한 표 이미지 저장
if (requireNamespace("gridExtra", quietly = TRUE)) {
  png(file.path(plot_dir, "meeting_summary_table.png"), width = 800, height = 400)
  gridExtra::grid.table(meeting_table)
  dev.off()
}

print(meeting_table)
cat("\n시각화가 완료되었습니다. 결과물은 'figures_szl_refit' 폴더에서 확인하세요.\n")
