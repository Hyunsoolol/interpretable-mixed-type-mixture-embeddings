# ============================================================
# Li et al. (Biometrics) simulation reproduction (NO packages)
# - Methods: Mix-L, Mix-AL, Mix-HP-L, Mix-HP-AL
# - Selection: BIC over m in {2,3,4} and lambda grid
# - EM + coordinate descent (weighted lasso)
#
# REFLECTED FIXES (requested):
# ③ Pilot estimate design for Adaptive Lasso weights:
#    - Use PILOT ridge-EM (non-sparse, stable) to get beta^(0)
#    - Adaptive weights = 1/(|beta^(0)|^gamma + eps), CLAMP + NORMALIZE
# ⑤ Ensure CD sufficiently converges:
#    - Enforce minimum sweeps (min_cd)
#    - Stricter tol (tol_cd)
#    - For HP M-step, periodically REBUILD residuals to avoid drift
# ⑥ BIC df tol adjustment:
#    - df counts use tol_df = max(1e-6, 0.5*min(lambda_grid))
#
# Also keeps earlier important fixes:
# - EM ll_old update bug fixed
# - Mix-HP partial_m sign bug fixed
# - HP df uses Bfree (independent params)
# - FPR/TPR selection uses PRE-matching B_hat
# - simulate_dataset mu vectorized (dimension-safe)
# ============================================================


# -------------------------
# Basic helpers
# -------------------------
soft_thresh <- function(z, g) {
  if (z >  g) return(z - g)
  if (z < -g) return(z + g)
  0
}

logsumexp <- function(v) {
  m <- max(v)
  m + log(sum(exp(v - m)))
}

clamp <- function(x, lo, hi) pmax(lo, pmin(hi, x))


# -------------------------
# Covariance matrices
# -------------------------
cov_mat <- function(p, correlated = FALSE) {
  if (!correlated) return(diag(p))
  toeplitz(0.5^(0:(p - 1)))
}


# -------------------------
# SNR <-> delta mapping (paper)
# SNR = 25 / delta  => delta = 25 / SNR
# -------------------------
delta_from_snr <- function(snr) 25 / snr


# -------------------------
# True scaled betas (paper setting)
# p0 = 10 relevant; among them p00 = 3 hetero (positions 8,9,10)
# -------------------------
true_scaled_betas <- function(p, delta) {
  stopifnot(p >= 10)
  beta0  <- rep(0, p)
  beta1  <- rep(0, p)
  beta2  <- rep(0, p)
  beta3  <- rep(0, p)

  beta0[1:10] <- c(rep(1, 7), rep(0, 3)) / sqrt(delta)

  beta1[1:10] <- c(rep(0, 7),  0, -3,  3) / sqrt(delta)
  beta2[1:10] <- c(rep(0, 7), -3,  3,  0) / sqrt(delta)
  beta3[1:10] <- c(rep(0, 7),  3,  0, -3) / sqrt(delta)

  list(beta0 = beta0,
       beta_het = cbind(beta1, beta2, beta3)) # p x 3
}


# -------------------------
# Data generator
# Model: f(y|x)= sum_j pi_j * rho_j/sqrt(2pi) * exp(-1/2 (rho_j y - x^T beta_tilde_j)^2)
# beta_tilde_j = beta0 + beta_j (scaled)
# unscaled b_j = beta_tilde_j / rho_j
# sigma^2 = 1/rho^2, paper: sigma^2 = delta*(0.1,0.1,0.4)
# -------------------------
simulate_dataset <- function(n = 200, p = 60, snr = 50, correlated = FALSE, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)

  delta <- delta_from_snr(snr)
  Sigma <- cov_mat(p, correlated)

  X <- matrix(rnorm(n * p), n, p)
  L <- chol(Sigma)
  X <- X %*% L

  m_true <- 3
  pi_true <- rep(1 / m_true, m_true)

  sigma2_true <- delta * c(0.1, 0.1, 0.4)
  sigma_true  <- sqrt(sigma2_true)
  rho_true    <- 1 / sigma_true

  bet <- true_scaled_betas(p, delta)
  beta0 <- bet$beta0
  beta_het <- bet$beta_het

  beta_tilde_true <- sapply(1:m_true, function(j) beta0 + beta_het[, j])
  beta_tilde_true <- matrix(beta_tilde_true, nrow = p, ncol = m_true)

  b_true <- sweep(beta_tilde_true, 2, rho_true, "/")

  z <- sample.int(m_true, size = n, replace = TRUE, prob = pi_true)

  # SAFE + fast: b_true[, z] is p x n -> t() makes n x p
  mu <- rowSums(X * t(b_true[, z, drop = FALSE]))
  y <- rnorm(n, mean = mu, sd = sigma_true[z])

  list(
    y = y, X = X, z = z,
    delta = delta, snr = snr, correlated = correlated,
    b_true = b_true,
    beta_tilde_true = beta_tilde_true,
    pi_true = pi_true,
    sigma2_true = sigma2_true,
    rho_true = rho_true,
    S_relevant = 1:10,
    S_common = 1:7,
    S_hetero = 8:10
  )
}


# -------------------------
# Mixture log-likelihood (scaled parameterization)
# beta_tilde_hat: p x m, rho_hat: m, pi_hat: m
# -------------------------
loglik_mix_scaled <- function(y, X, beta_tilde_hat, rho_hat, pi_hat) {
  n <- length(y)
  m <- length(pi_hat)
  ll <- 0
  for (i in 1:n) {
    lvec <- rep(NA_real_, m)
    for (j in 1:m) {
      mu <- sum(X[i, ] * beta_tilde_hat[, j])
      r  <- rho_hat[j] * y[i] - mu
      lvec[j] <- log(pi_hat[j] + 1e-300) + log(rho_hat[j] + 1e-300) -
        0.5 * r * r - 0.5 * log(2*pi)
    }
    ll <- ll + logsumexp(lvec)
  }
  ll
}


# -------------------------
# E-step responsibilities (scaled)
# tau_ij proportional to pi_j * rho_j * exp(-1/2 (rho_j y - x beta)^2)
# -------------------------
estep_tau <- function(y, X, beta_tilde, rho, pi) {
  n <- length(y); m <- length(pi)
  logw <- matrix(0, n, m)
  for (j in 1:m) {
    mu <- as.vector(X %*% beta_tilde[, j])
    r  <- rho[j] * y - mu
    logw[, j] <- log(pi[j] + 1e-300) + log(rho[j] + 1e-300) - 0.5 * r * r
  }
  tau <- matrix(0, n, m)
  for (i in 1:n) {
    den <- logsumexp(logw[i, ])
    tau[i, ] <- exp(logw[i, ] - den)
  }
  tau
}


# -------------------------
# Weighted ridge WLS (pilot)
# beta = argmin 1/2 sum_i w_i (y_i - x_i^T beta)^2 + 1/2 * alpha * ||beta||^2
# -------------------------
ridge_wls <- function(X, y, w, alpha) {
  # alpha scaled later; should be >=0
  XtW <- t(X) * w
  A <- XtW %*% X
  diag(A) <- diag(A) + alpha
  b <- XtW %*% y
  out <- tryCatch(as.vector(solve(A, b)), error = function(e) NULL)
  if (is.null(out)) out <- as.vector(qr.solve(A, b))
  out
}


# -------------------------
# Weighted lasso by coordinate descent (with min sweeps)
# minimize 1/2 sum_i w_i (y_i - x_i^T b)^2 + lam * sum_k pen_w[k] * |b_k|
# -------------------------
lasso_cd_weighted <- function(X, y, w, lam, pen_w = NULL,
                              b_init = NULL,
                              maxit = 500, tol = 1e-8, minit = 10) {
  n <- nrow(X); p <- ncol(X)
  if (is.null(pen_w)) pen_w <- rep(1, p)
  if (is.null(b_init)) b <- rep(0, p) else b <- b_init

  r <- y - as.vector(X %*% b)
  a <- colSums((X^2) * w)

  for (it in 1:maxit) {
    b_old <- b
    for (k in 1:p) {
      if (a[k] < 1e-12) next
      xk <- X[, k]
      ck <- sum(w * xk * (r + xk * b[k]))
      bk_new <- soft_thresh(ck, lam * pen_w[k]) / a[k]
      d <- bk_new - b[k]
      if (abs(d) > 0) {
        b[k] <- bk_new
        r <- r - xk * d
      }
    }
    if (it >= minit && max(abs(b - b_old)) < tol) break
  }
  b
}


# -------------------------
# 1D update with TWO absolute terms (HP coordinate)
# minimize 0.5*a*b^2 - c*b + lam1*|b| + lam2*|b+s|
# -------------------------
update_two_abs_1d <- function(a, c, s, lam1, lam2) {
  if (a < 1e-12) return(0)

  cand <- c(0, -s)

  for (sgn1 in c(-1, 1)) {
    for (sgn2 in c(-1, 1)) {
      b <- (c - (lam1 * sgn1 + lam2 * sgn2)) / a
      ok1 <- (b > 0 && sgn1 == 1) || (b < 0 && sgn1 == -1) || abs(b) < 1e-12
      ok2 <- ((b + s) > 0 && sgn2 == 1) || ((b + s) < 0 && sgn2 == -1) || abs(b + s) < 1e-12
      if (ok1 && ok2) cand <- c(cand, b)
    }
  }

  f <- function(b) 0.5 * a * b * b - c * b + lam1 * abs(b) + lam2 * abs(b + s)
  vals <- sapply(cand, f)
  cand[which.min(vals)]
}


# -------------------------
# M-step for Mix-L / Mix-AL
# - update pi
# - update beta_tilde_j via weighted lasso on (rho_j*y) ~ X
# - update sigma^2 and rho via unscaled residuals (ECM-style)
# -------------------------
mstep_mix_lasso <- function(y, X, tau, beta_tilde, rho, pi,
                            lam, pen_w_list,
                            max_cd = 500, tol_cd = 1e-8, min_cd = 10,
                            min_pi = 1e-6,
                            sigma2_floor = 1e-10) {
  n <- length(y); m <- ncol(tau); p <- ncol(X)

  pi_new <- colMeans(tau)
  pi_new <- pmax(pi_new, min_pi)
  pi_new <- pi_new / sum(pi_new)

  beta_new <- beta_tilde
  rho_new <- rho

  for (j in 1:m) {
    w <- tau[, j]
    yj <- rho_new[j] * y
    pen_w <- pen_w_list[[j]]
    beta_new[, j] <- lasso_cd_weighted(
      X, yj, w,
      lam = n * lam,
      pen_w = pen_w,
      b_init = beta_new[, j],
      maxit = max_cd, tol = tol_cd, minit = min_cd
    )
  }

  sigma2_new <- rep(NA_real_, m)
  for (j in 1:m) {
    bj <- beta_new[, j] / rho_new[j]
    rj <- y - as.vector(X %*% bj)
    num <- sum(tau[, j] * rj^2)
    den <- sum(tau[, j]) + 1e-12
    s2 <- num / den
    s2 <- max(s2, sigma2_floor)
    sigma2_new[j] <- s2
    rho_new[j] <- 1 / sqrt(s2 + 1e-12)
  }

  list(beta_tilde = beta_new, rho = rho_new, pi = pi_new, sigma2 = sigma2_new)
}


# -------------------------
# PILOT ridge-EM for adaptive weights (Mix)
# returns beta_tilde_pilot (p x m), rho, pi
# -------------------------
pilot_ridge_em_mix <- function(y, X, m,
                               n_start = 1, seed_base = 1,
                               max_em = 30,
                               ridge_scale = 1e-3,
                               sigma2_floor = 1e-10,
                               min_pi = 1e-6) {
  n <- length(y); p <- ncol(X)

  best <- NULL
  best_ll <- -Inf

  for (s in 1:n_start) {
    set.seed(seed_base + 777*s + 31*m)
    z <- sample.int(m, n, replace = TRUE)
    tau <- matrix(0, n, m)
    tau[cbind(1:n, z)] <- 1

    pi <- colMeans(tau); pi <- pmax(pi, min_pi); pi <- pi / sum(pi)
    rho <- rep(1 / (sd(y) + 1e-12), m)
    beta_tilde <- matrix(0, p, m)

    for (iter in 1:max_em) {
      tau <- estep_tau(y, X, beta_tilde, rho, pi)

      # pi update
      pi <- colMeans(tau); pi <- pmax(pi, min_pi); pi <- pi / sum(pi)

      # ridge update per component
      for (j in 1:m) {
        w <- tau[, j]
        yj <- rho[j] * y
        # scale alpha to data curvature
        xtwx_diag_mean <- mean(colSums((X^2) * w)) + 1e-12
        alpha <- ridge_scale * xtwx_diag_mean
        beta_tilde[, j] <- ridge_wls(X, yj, w, alpha)
      }

      # sigma2/rho update (unscaled)
      for (j in 1:m) {
        bj <- beta_tilde[, j] / rho[j]
        rj <- y - as.vector(X %*% bj)
        s2 <- sum(tau[, j] * rj^2) / (sum(tau[, j]) + 1e-12)
        s2 <- max(s2, sigma2_floor)
        rho[j] <- 1 / sqrt(s2 + 1e-12)
      }
    }

    ll <- loglik_mix_scaled(y, X, beta_tilde, rho, pi)
    if (ll > best_ll) {
      best_ll <- ll
      best <- list(beta_tilde = beta_tilde, rho = rho, pi = pi, ll = ll)
    }
  }

  best
}


# -------------------------
# M-step for Mix-HP-L / Mix-HP-AL
# (a) min sweeps + strict tol
# (b) periodic residual rebuild to avoid numeric drift
# -------------------------
mstep_mix_hp <- function(y, X, tau, beta0, Bfree, rho, pi,
                         lam, pen_w0, pen_wB, # pen_wB: p x m weights
                         max_cd = 500, tol_cd = 1e-8, min_cd = 10,
                         rebuild_every = 5,
                         min_pi = 1e-6,
                         sigma2_floor = 1e-10) {
  n <- length(y); m <- ncol(tau); p <- ncol(X)
  stopifnot(ncol(Bfree) == m - 1)

  pi_new <- colMeans(tau)
  pi_new <- pmax(pi_new, min_pi)
  pi_new <- pi_new / sum(pi_new)

  rho_new <- rho
  beta0_new <- beta0
  Bfree_new <- Bfree

  make_Bfull <- function(Bf) {
    Bm <- -rowSums(Bf)
    cbind(Bf, Bm)
  }

  Xcols <- lapply(1:p, function(k) X[, k])

  build_residuals <- function(beta0v, Bfull) {
    r <- matrix(0, n, m)
    for (j in 1:m) {
      bt <- beta0v + Bfull[, j]
      r[, j] <- rho_new[j] * y - as.vector(X %*% bt)
    }
    r
  }

  Bfull <- make_Bfull(Bfree_new)
  rmat <- build_residuals(beta0_new, Bfull)

  for (it in 1:max_cd) {
    maxchg <- 0

    # ---- beta0 coordinates ----
    for (k in 1:p) {
      xk <- Xcols[[k]]
      a <- sum((xk^2) * rowSums(tau))
      if (a < 1e-12) next

      cval <- 0
      for (j in 1:m) {
        cval <- cval + sum(tau[, j] * xk * (rmat[, j] + xk * beta0_new[k]))
      }

      bnew <- soft_thresh(cval, n * lam * pen_w0[k]) / a
      d <- bnew - beta0_new[k]
      if (abs(d) > 0) {
        beta0_new[k] <- bnew
        maxchg <- max(maxchg, abs(d))
        for (j in 1:m) rmat[, j] <- rmat[, j] - xk * d
      }
    }

    # ---- Bfree coordinates ----
    for (l in 1:(m - 1)) {
      for (k in 1:p) {
        xk <- Xcols[[k]]

        s_minus <- if (m - 1 == 1) 0 else sum(Bfree_new[k, -l])
        bcur <- Bfree_new[k, l]

        a <- sum((tau[, l] + tau[, m]) * (xk^2))
        if (a < 1e-12) next

        partial_l <- rmat[, l] + xk * bcur

        # FIXED: component m design is (-xk)
        partial_m <- rmat[, m] - xk * bcur

        cval <- sum(tau[, l] * xk * partial_l + tau[, m] * (-xk) * partial_m)

        lam1 <- n * lam * pen_wB[k, l]
        lam2 <- n * lam * pen_wB[k, m]

        bnew <- update_two_abs_1d(a = a, c = cval, s = s_minus, lam1 = lam1, lam2 = lam2)
        d <- bnew - bcur
        if (abs(d) > 0) {
          Bfree_new[k, l] <- bnew
          maxchg <- max(maxchg, abs(d))
          rmat[, l] <- rmat[, l] - xk * d
          rmat[, m] <- rmat[, m] + xk * d
        }
      }
    }

    # periodic rebuild to prevent drift
    if (rebuild_every > 0 && (it %% rebuild_every) == 0) {
      Bfull <- make_Bfull(Bfree_new)
      rmat <- build_residuals(beta0_new, Bfull)
    }

    if (it >= min_cd && maxchg < tol_cd) break
  }

  Bfull <- make_Bfull(Bfree_new)

  sigma2_new <- rep(NA_real_, m)
  for (j in 1:m) {
    beta_tilde_j <- beta0_new + Bfull[, j]
    bj <- beta_tilde_j / rho_new[j]
    rj <- y - as.vector(X %*% bj)
    s2 <- sum(tau[, j] * rj^2) / (sum(tau[, j]) + 1e-12)
    s2 <- max(s2, sigma2_floor)
    sigma2_new[j] <- s2
    rho_new[j] <- 1 / sqrt(s2 + 1e-12)
  }

  list(beta0 = beta0_new, Bfull = Bfull, Bfree = Bfree_new,
       rho = rho_new, pi = pi_new, sigma2 = sigma2_new)
}


# -------------------------
# PILOT ridge-EM for adaptive weights (HP)
# Strategy:
# 1) Fit per-component ridge beta_tilde (unconstrained)
# 2) Project to constraint: beta0=rowMeans, B=beta_tilde-beta0 (sum-to-zero)
# -------------------------
pilot_ridge_em_hp <- function(y, X, m,
                              n_start = 1, seed_base = 1,
                              max_em = 30,
                              ridge_scale = 1e-3,
                              sigma2_floor = 1e-10,
                              min_pi = 1e-6) {
  fit <- pilot_ridge_em_mix(
    y, X, m,
    n_start = n_start, seed_base = seed_base,
    max_em = max_em,
    ridge_scale = ridge_scale,
    sigma2_floor = sigma2_floor,
    min_pi = min_pi
  )
  beta_tilde <- fit$beta_tilde
  beta0 <- rowMeans(beta_tilde)
  Bfull <- beta_tilde - beta0
  # enforce exact sum-to-zero (numerical)
  Bfull <- sweep(Bfull, 1, rowMeans(Bfull), "-")
  list(beta0 = beta0, Bfull = Bfull, rho = fit$rho, pi = fit$pi, ll = fit$ll)
}


# -------------------------
# Adaptive weights builder (clamp + normalize)
# normalize so mean weight ~ 1 (keeps lambda scale more comparable)
# -------------------------
make_adaptive_weights <- function(beta_mat, gamma = 1,
                                  eps = 1e-4,
                                  w_min = 1e-2,
                                  w_max = 1e2) {
  w <- 1 / (abs(beta_mat)^gamma + eps)
  w <- clamp(w, w_min, w_max)
  w <- w / mean(w)
  w
}


# -------------------------
# Fit mixture model for fixed (m, lam) by EM
# Mix-L / Mix-AL (adaptive uses PILOT ridge-EM)
# -------------------------
fit_mix_L_or_AL <- function(y, X, m, lam,
                            adaptive = FALSE, gamma = 1,
                            pilot_em = 30, pilot_ridge_scale = 1e-3,
                            w_eps = 1e-4, w_min = 1e-2, w_max = 1e2,
                            max_em = 200, tol_em = 1e-5,
                            max_cd = 500, tol_cd = 1e-8, min_cd = 10,
                            n_start = 5, seed_base = 1,
                            sigma2_floor = 1e-10) {
  n <- length(y); p <- ncol(X)

  best <- NULL
  best_obj <- -Inf

  for (s in 1:n_start) {
    set.seed(seed_base + 1000*s + 17*m)

    z <- sample.int(m, n, replace = TRUE)
    tau <- matrix(0, n, m)
    tau[cbind(1:n, z)] <- 1

    pi <- colMeans(tau); pi <- pmax(pi, 1e-6); pi <- pi / sum(pi)
    rho <- rep(1 / (sd(y) + 1e-12), m)
    beta_tilde <- matrix(0, p, m)

    # penalty weights (per component)
    pen_w_list <- vector("list", m)
    for (j in 1:m) pen_w_list[[j]] <- rep(1, p)

    if (adaptive) {
      # pilot ridge-EM -> stable nonzero beta^(0)
      pil <- pilot_ridge_em_mix(
        y, X, m,
        n_start = 1,
        seed_base = seed_base + 50000 + 1000*s,
        max_em = pilot_em,
        ridge_scale = pilot_ridge_scale,
        sigma2_floor = sigma2_floor
      )
      beta_init <- pil$beta_tilde
      rho <- pil$rho; pi <- pil$pi
      beta_tilde <- beta_init

      for (j in 1:m) {
        wj <- make_adaptive_weights(beta_init[, j], gamma = gamma,
                                   eps = w_eps, w_min = w_min, w_max = w_max)
        pen_w_list[[j]] <- as.vector(wj)
      }
    }

    ll_old <- -Inf
    for (iter in 1:max_em) {
      tau <- estep_tau(y, X, beta_tilde, rho, pi)

      ms <- mstep_mix_lasso(
        y, X, tau, beta_tilde, rho, pi,
        lam = lam,
        pen_w_list = pen_w_list,
        max_cd = max_cd, tol_cd = tol_cd, min_cd = min_cd,
        sigma2_floor = sigma2_floor
      )

      beta_tilde <- ms$beta_tilde
      rho <- ms$rho
      pi <- ms$pi

      ll <- loglik_mix_scaled(y, X, beta_tilde, rho, pi)

      if (abs(ll - ll_old) < tol_em) { ll_old <- ll; break }
      ll_old <- ll
    }

    pen <- 0
    for (j in 1:m) pen <- pen + sum(pen_w_list[[j]] * abs(beta_tilde[, j]))
    obj <- ll_old - n * lam * pen

    if (obj > best_obj) {
      best_obj <- obj
      best <- list(beta_tilde = beta_tilde, rho = rho, pi = pi, ll = ll_old,
                   pen_w_list = pen_w_list, m = m, lam = lam, adaptive = adaptive)
    }
  }

  best
}


# -------------------------
# Fit mixture model for fixed (m, lam) by EM
# Mix-HP-L / Mix-HP-AL (adaptive uses PILOT ridge-EM + projection)
# -------------------------
fit_mix_HP_L_or_AL <- function(y, X, m, lam,
                               adaptive = FALSE, gamma = 1,
                               pilot_em = 30, pilot_ridge_scale = 1e-3,
                               w_eps = 1e-4, w_min = 1e-2, w_max = 1e2,
                               max_em = 200, tol_em = 1e-5,
                               max_cd = 500, tol_cd = 1e-8, min_cd = 10,
                               rebuild_every = 5,
                               n_start = 5, seed_base = 1,
                               sigma2_floor = 1e-10) {
  n <- length(y); p <- ncol(X)

  best <- NULL
  best_obj <- -Inf

  for (s in 1:n_start) {
    set.seed(seed_base + 2000*s + 23*m)

    z <- sample.int(m, n, replace = TRUE)
    tau <- matrix(0, n, m)
    tau[cbind(1:n, z)] <- 1

    pi <- colMeans(tau); pi <- pmax(pi, 1e-6); pi <- pi / sum(pi)
    rho <- rep(1 / (sd(y) + 1e-12), m)

    beta0 <- rep(0, p)
    Bfree <- matrix(0, p, m - 1)

    pen_w0 <- rep(1, p)
    pen_wB <- matrix(1, p, m)

    if (adaptive) {
      pil <- pilot_ridge_em_hp(
        y, X, m,
        n_start = 1,
        seed_base = seed_base + 70000 + 2000*s,
        max_em = pilot_em,
        ridge_scale = pilot_ridge_scale,
        sigma2_floor = sigma2_floor
      )
      beta0 <- pil$beta0
      Bfull_init <- pil$Bfull
      # set free params
      Bfree <- Bfull_init[, 1:(m - 1), drop = FALSE]
      rho <- pil$rho; pi <- pil$pi

      pen_w0 <- as.vector(make_adaptive_weights(beta0, gamma = gamma,
                                               eps = w_eps, w_min = w_min, w_max = w_max))
      pen_wB <- make_adaptive_weights(Bfull_init, gamma = gamma,
                                      eps = w_eps, w_min = w_min, w_max = w_max)
    }

    ll_old <- -Inf
    for (iter in 1:max_em) {
      Bfull <- cbind(Bfree, -rowSums(Bfree))
      beta_tilde <- sapply(1:m, function(j) beta0 + Bfull[, j])
      beta_tilde <- matrix(beta_tilde, nrow = p, ncol = m)

      tau <- estep_tau(y, X, beta_tilde, rho, pi)

      ms <- mstep_mix_hp(
        y, X, tau, beta0, Bfree, rho, pi,
        lam = lam,
        pen_w0 = pen_w0,
        pen_wB = pen_wB,
        max_cd = max_cd, tol_cd = tol_cd, min_cd = min_cd,
        rebuild_every = rebuild_every,
        sigma2_floor = sigma2_floor
      )

      beta0 <- ms$beta0
      Bfree <- ms$Bfree
      rho <- ms$rho
      pi <- ms$pi

      Bfull <- ms$Bfull
      beta_tilde <- sapply(1:m, function(j) beta0 + Bfull[, j])
      beta_tilde <- matrix(beta_tilde, nrow = p, ncol = m)

      ll <- loglik_mix_scaled(y, X, beta_tilde, rho, pi)

      if (abs(ll - ll_old) < tol_em) { ll_old <- ll; break }
      ll_old <- ll
    }

    Bfull <- cbind(Bfree, -rowSums(Bfree))
    pen <- sum(pen_w0 * abs(beta0)) + sum(pen_wB * abs(Bfull))
    obj <- ll_old - n * lam * pen

    if (obj > best_obj) {
      best_obj <- obj
      best <- list(beta0 = beta0, Bfull = Bfull, Bfree = Bfree,
                   rho = rho, pi = pi, ll = ll_old,
                   pen_w0 = pen_w0, pen_wB = pen_wB,
                   m = m, lam = lam, adaptive = adaptive)
    }
  }

  best
}


# -------------------------
# BIC helpers
# -------------------------
bic_mix <- function(ll, df, n) -2 * ll + df * log(n)

count_nz <- function(v, tol = 1e-6) sum(abs(v) > tol)


# -------------------------
# Lambda grid
# -------------------------
lambda_grid <- function(y, X, nlambda = 25, ratio = 0.02) {
  n <- length(y)
  rho0 <- 1 / (sd(y) + 1e-12)
  yt <- rho0 * y
  lam_max <- max(abs(colSums(X * yt))) / (n + 1e-12)
  lam_min <- lam_max * ratio
  exp(seq(log(lam_max), log(lam_min), length.out = nlambda))
}


# -------------------------
# Fit one method over m in {2,3,4} and lambda grid; pick best by BIC
# (df tol uses tol_df = max(1e-6, 0.5*min(lams)))
# HP df uses Bfree (independent params)
# -------------------------
fit_by_bic <- function(y, X, method,
                       m_set = 2:4,
                       nlambda = 25, lam_ratio = 0.02,
                       n_start = 5,
                       seed_base = 1,
                       # adaptive pilot controls
                       pilot_em = 30, pilot_ridge_scale = 1e-3,
                       w_eps = 1e-4, w_min = 1e-2, w_max = 1e2,
                       # CD controls
                       max_cd = 500, tol_cd = 1e-8, min_cd = 10,
                       rebuild_every = 5,
                       sigma2_floor = 1e-10) {
  n <- length(y); p <- ncol(X)

  lams <- lambda_grid(y, X, nlambda = nlambda, ratio = lam_ratio)
  tol_df <- max(1e-6, 0.5 * min(lams))

  best_all <- NULL
  best_bic <- Inf

  for (m in m_set) {
    for (lam in lams) {

      fit <- tryCatch({
        if (method == "Mix-L") {
          fit_mix_L_or_AL(
            y, X, m, lam, adaptive = FALSE,
            n_start = n_start, seed_base = seed_base,
            max_cd = max_cd, tol_cd = tol_cd, min_cd = min_cd,
            sigma2_floor = sigma2_floor
          )
        } else if (method == "Mix-AL") {
          fit_mix_L_or_AL(
            y, X, m, lam, adaptive = TRUE,
            pilot_em = pilot_em, pilot_ridge_scale = pilot_ridge_scale,
            w_eps = w_eps, w_min = w_min, w_max = w_max,
            n_start = n_start, seed_base = seed_base,
            max_cd = max_cd, tol_cd = tol_cd, min_cd = min_cd,
            sigma2_floor = sigma2_floor
          )
        } else if (method == "Mix-HP-L") {
          fit_mix_HP_L_or_AL(
            y, X, m, lam, adaptive = FALSE,
            n_start = n_start, seed_base = seed_base,
            max_cd = max_cd, tol_cd = tol_cd, min_cd = min_cd,
            rebuild_every = rebuild_every,
            sigma2_floor = sigma2_floor
          )
        } else if (method == "Mix-HP-AL") {
          fit_mix_HP_L_or_AL(
            y, X, m, lam, adaptive = TRUE,
            pilot_em = pilot_em, pilot_ridge_scale = pilot_ridge_scale,
            w_eps = w_eps, w_min = w_min, w_max = w_max,
            n_start = n_start, seed_base = seed_base,
            max_cd = max_cd, tol_cd = tol_cd, min_cd = min_cd,
            rebuild_every = rebuild_every,
            sigma2_floor = sigma2_floor
          )
        } else {
          stop("Unknown method")
        }
      }, error = function(e) NULL)

      if (is.null(fit)) next

      if (method %in% c("Mix-L", "Mix-AL")) {
        beta_tilde <- fit$beta_tilde
        ll <- fit$ll
        df <- count_nz(beta_tilde, tol = tol_df) + (m - 1) + m
        bic <- bic_mix(ll, df, n)
        if (bic < best_bic) {
          best_bic <- bic
          best_all <- list(method = method, m = m, lam = lam,
                           beta_tilde = beta_tilde,
                           rho = fit$rho, pi = fit$pi, ll = ll, bic = bic)
        }
      } else {
        beta0 <- fit$beta0
        Bfree <- fit$Bfree
        Bfull <- fit$Bfull
        beta_tilde <- sapply(1:m, function(j) beta0 + Bfull[, j])
        beta_tilde <- matrix(beta_tilde, nrow = p, ncol = m)
        ll <- fit$ll

        df <- count_nz(beta0, tol = tol_df) + count_nz(Bfree, tol = tol_df) + (m - 1) + m
        bic <- bic_mix(ll, df, n)
        if (bic < best_bic) {
          best_bic <- bic
          best_all <- list(method = method, m = m, lam = lam,
                           beta0 = beta0, Bfull = Bfull, Bfree = Bfree,
                           beta_tilde = beta_tilde,
                           rho = fit$rho, pi = fit$pi, ll = ll, bic = bic)
        }
      }
    }
  }

  best_all
}


# -------------------------
# Component matching for MSE (m<=4)
# -------------------------
perms <- function(v) {
  if (length(v) == 1) return(list(v))
  out <- list()
  for (i in seq_along(v)) {
    rest <- v[-i]
    ps <- perms(rest)
    for (p in ps) out[[length(out) + 1]] <- c(v[i], p)
  }
  out
}

match_to_true <- function(B_hat, pi_hat, s2_hat, B_true, pi_true, s2_true) {
  p <- nrow(B_true)
  m_true <- ncol(B_true)
  m_hat <- ncol(B_hat)

  best_sse <- Inf
  best <- NULL

  if (m_hat >= m_true) {
    idx_list <- combn(m_hat, m_true, simplify = FALSE)
    pset <- perms(1:m_true)
    for (ss in idx_list) {
      for (pp in pset) {
        idx <- ss[pp]
        Bh <- B_hat[, idx, drop = FALSE]
        sse <- sum((Bh - B_true)^2)
        if (sse < best_sse) {
          best_sse <- sse
          best <- list(B = Bh, pi = pi_hat[idx], s2 = s2_hat[idx], m_hat = m_hat)
        }
      }
    }
    return(best)
  }

  true_subs <- combn(m_true, m_hat, simplify = FALSE)
  pset <- perms(1:m_hat)
  for (ts in true_subs) {
    Bt_sub <- B_true[, ts, drop = FALSE]
    for (pp in pset) {
      Bh <- B_hat[, pp, drop = FALSE]
      sse <- sum((Bh - Bt_sub)^2)
      miss <- setdiff(1:m_true, ts)
      if (length(miss) > 0) sse <- sse + sum((B_true[, miss, drop = FALSE])^2)
      if (sse < best_sse) {
        Bpad <- matrix(0, p, m_true)
        pipad <- rep(0, m_true)
        s2pad <- rep(0, m_true)
        Bpad[, ts] <- Bh
        pipad[ts] <- pi_hat[pp]
        s2pad[ts] <- s2_hat[pp]
        best_sse <- sse
        best <- list(B = Bpad, pi = pipad, s2 = s2pad, m_hat = m_hat)
      }
    }
  }
  best
}


# -------------------------
# Metrics
# - MSEs computed after matching (Bh)
# - FPR/TPR computed using pre-matching B_hat
# -------------------------
compute_metrics <- function(fit, truth, tol_sel = 1e-8) {
  B_true <- truth$b_true
  pi_true <- truth$pi_true
  s2_true <- truth$sigma2_true

  beta_tilde <- fit$beta_tilde
  rho <- fit$rho
  pi_hat <- fit$pi

  s2_hat <- (1 / (rho + 1e-12))^2

  # PRE-matching coefficients
  B_hat <- sweep(beta_tilde, 2, rho, "/")

  matched <- match_to_true(B_hat, pi_hat, s2_hat, B_true, pi_true, s2_true)
  Bh <- matched$B; pih <- matched$pi; s2h <- matched$s2

  mse_b  <- mean((Bh - B_true)^2)
  mse_pi <- mean((pih - pi_true)^2)
  mse_s2 <- mean((s2h - s2_true)^2)

  p <- nrow(B_true)
  S_R <- truth$S_relevant
  S_C <- truth$S_common

  sel_R <- which(apply(abs(B_hat), 1, function(v) any(v > tol_sel)))

  if (!is.null(fit$method) && fit$method %in% c("Mix-HP-L", "Mix-HP-AL")) {
    Bfull <- fit$Bfull
    sel_H <- which(apply(abs(Bfull), 1, function(v) any(v > tol_sel)))
  } else {
    sel_H <- which(apply(beta_tilde, 1, function(v) sd(v) > tol_sel))
  }

  FPR <- length(intersect(sel_R, setdiff(1:p, S_R))) / (p - length(S_R))
  TPR <- length(intersect(sel_R, S_R)) / length(S_R)
  FHR <- length(intersect(sel_H, S_C)) / length(S_C)

  list(
    mse_b = mse_b,
    mse_pi = mse_pi,
    mse_sigma2 = mse_s2,
    FPR = FPR, TPR = TPR, FHR = FHR,
    m_hat = matched$m_hat
  )
}


# -------------------------
# One replicate: fit 4 methods and evaluate
# -------------------------
one_rep <- function(rep_id, n, p, snr, correlated,
                    nlambda = 25, lam_ratio = 0.02,
                    n_start = 5, seed_base = 2026,
                    pilot_em = 30, pilot_ridge_scale = 1e-3,
                    w_eps = 1e-4, w_min = 1e-2, w_max = 1e2,
                    max_cd = 500, tol_cd = 1e-8, min_cd = 10,
                    rebuild_every = 5,
                    sigma2_floor = 1e-10) {

  dat <- simulate_dataset(n = n, p = p, snr = snr, correlated = correlated,
                          seed = seed_base + rep_id)
  y <- dat$y; X <- dat$X

  methods <- c("Mix-L", "Mix-AL", "Mix-HP-L", "Mix-HP-AL")
  out <- vector("list", length(methods))

  for (i in seq_along(methods)) {
    meth <- methods[i]

    fit <- fit_by_bic(
      y, X, meth,
      m_set = 2:4,
      nlambda = nlambda,
      lam_ratio = lam_ratio,
      n_start = n_start,
      seed_base = seed_base + 999*rep_id,
      pilot_em = pilot_em,
      pilot_ridge_scale = pilot_ridge_scale,
      w_eps = w_eps, w_min = w_min, w_max = w_max,
      max_cd = max_cd, tol_cd = tol_cd, min_cd = min_cd,
      rebuild_every = rebuild_every,
      sigma2_floor = sigma2_floor
    )

    fit$method <- meth
    fit$y <- y; fit$X <- X

    met <- compute_metrics(fit, dat)

    out[[i]] <- data.frame(
      rep = rep_id, method = meth,
      snr = snr, p = p, correlated = correlated,
      mse_b = met$mse_b,
      mse_sigma2 = met$mse_sigma2,
      mse_pi = met$mse_pi,
      FPR = met$FPR, TPR = met$TPR, FHR = met$FHR,
      m_hat = met$m_hat,
      stringsAsFactors = FALSE
    )
  }

  do.call(rbind, out)
}


# -------------------------
# Run simulation grid
# -------------------------
run_sim <- function(R = 20, n = 200, p = 60,
                    snr_grid = c(200, 100, 50, 25, 12.5),
                    correlated = FALSE,
                    nlambda = 25, lam_ratio = 0.02,
                    n_start = 5, seed_base = 2026,
                    pilot_em = 30, pilot_ridge_scale = 1e-3,
                    w_eps = 1e-4, w_min = 1e-2, w_max = 1e2,
                    max_cd = 500, tol_cd = 1e-8, min_cd = 10,
                    rebuild_every = 5,
                    sigma2_floor = 1e-10) {

  res_all <- list()
  idx <- 1

  for (snr in snr_grid) {
    for (r in 1:R) {
      cat("SNR=", snr, " rep=", r, " / ", R, "\n")
      res_all[[idx]] <- one_rep(
        rep_id = r + 10000*snr,
        n = n, p = p, snr = snr, correlated = correlated,
        nlambda = nlambda, lam_ratio = lam_ratio,
        n_start = n_start, seed_base = seed_base,
        pilot_em = pilot_em, pilot_ridge_scale = pilot_ridge_scale,
        w_eps = w_eps, w_min = w_min, w_max = w_max,
        max_cd = max_cd, tol_cd = tol_cd, min_cd = min_cd,
        rebuild_every = rebuild_every,
        sigma2_floor = sigma2_floor
      )
      idx <- idx + 1
    }
  }

  do.call(rbind, res_all)
}


# -------------------------
# Summary like Table (paper reports MSE*100 and rates in %)
# -------------------------
summarize_res <- function(res) {
  methods <- unique(res$method)
  out <- data.frame()

  for (snr in sort(unique(res$snr))) {
    for (meth in methods) {
      sub <- res[res$snr == snr & res$method == meth, ]
      if (nrow(sub) == 0) next
      out <- rbind(out, data.frame(
        snr = snr,
        method = meth,
        mse_b_100 = 100 * mean(sub$mse_b, na.rm = TRUE),
        mse_s2_100 = 100 * mean(sub$mse_sigma2, na.rm = TRUE),
        mse_pi_100 = 100 * mean(sub$mse_pi, na.rm = TRUE),
        FPR_pct = 100 * mean(sub$FPR, na.rm = TRUE),
        FHR_pct = 100 * mean(sub$FHR, na.rm = TRUE),
        TPR_pct = 100 * mean(sub$TPR, na.rm = TRUE),
        m_hat_mode = {
          mh <- sub$m_hat[is.finite(sub$m_hat)]
          if (length(mh) == 0) NA else as.integer(names(sort(table(mh), decreasing = TRUE)[1]))
        },
        stringsAsFactors = FALSE
      ))
    }
  }
  out
}


# -------------------------
# Figure-1 style: log(MSE) boxplot for coefficients (base R)
# -------------------------
plot_fig1_like <- function(res, snr_value) {
  sub <- res[res$snr == snr_value, ]
  sub <- sub[is.finite(sub$mse_b), ]
  if (nrow(sub) == 0) return(invisible(NULL))
  boxplot(log(sub$mse_b) ~ sub$method,
          main = paste0("log(MSE) of coefficients, SNR=", snr_value),
          xlab = "", ylab = "log(MSE)")
}


# -------------------------
# Paper Table-1 style summary (mean (SE), rates in %)
# -------------------------
summarize_paper_style <- function(res) {
  methods_order <- c("Mix-L", "Mix-AL", "Mix-HP-L", "Mix-HP-AL")
  out <- data.frame()

  for (snr in sort(unique(res$snr), decreasing = TRUE)) {
    for (meth in methods_order) {
      sub <- res[res$snr == snr & res$method == meth, ]
      if (nrow(sub) == 0) next

      n_rep <- nrow(sub)

      format_stat <- function(values, scale = 1) {
        val <- values * scale
        avg <- mean(val, na.rm = TRUE)
        se  <- sd(val, na.rm = TRUE) / sqrt(n_rep)
        sprintf("%.2f (%.2f)", avg, se)
      }

      format_rate <- function(values) {
        sprintf("%.1f", mean(values, na.rm = TRUE) * 100)
      }

      out <- rbind(out, data.frame(
        SNR = snr,
        Method = meth,
        b = format_stat(sub$mse_b, 100),
        sigma2 = format_stat(sub$mse_sigma2, 100),
        pi = format_stat(sub$mse_pi, 100),
        FPR = format_rate(sub$FPR),
        FHR = format_rate(sub$FHR),
        TPR = format_rate(sub$TPR),
        m_hat = {
          mh <- sub$m_hat[is.finite(sub$m_hat)]
          if (length(mh) == 0) NA else as.integer(names(sort(table(mh), decreasing = TRUE)[1]))
        },
        stringsAsFactors = FALSE
      ))
    }
  }

  colnames(out) <- c("SNR", "Method", "MSE_b(SE)", "MSE_sigma2(SE)", "MSE_pi(SE)",
                     "FPR(%)", "FHR(%)", "TPR(%)", "m_mode")
  out
}


# ============================================================
# Example run (sanity check)
# ============================================================
set.seed(1)
res <- run_sim(
  R = 5, n = 200, p = 60,
  snr_grid = c(50),
  correlated = FALSE,
  nlambda = 10, lam_ratio = 0.05,
  n_start = 2, seed_base = 2026,
  # requested knobs
  pilot_em = 30, pilot_ridge_scale = 1e-3,
  max_cd = 500, tol_cd = 1e-8, min_cd = 10,
  rebuild_every = 5
)

print(summarize_res(res))
plot_fig1_like(res, 50)

final_table <- summarize_paper_style(res)
print(final_table, row.names = FALSE)
