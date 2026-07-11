# Reusable diagnostic helpers for exact fixed-support centered-Eta refitting.
# These functions are not sourced by official fitting runners.

exact_refit_log_vmf_const_one <- function(kappa, d) {
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
  scaled <- besselI(kappa, nu, expon.scaled = TRUE)
  if (!is.finite(scaled) || scaled <= 0) {
    return(((d - 1) / 2) * (log(kappa) - log(2 * pi)) - kappa)
  }
  nu * log(kappa) - (d / 2) * log(2 * pi) - (log(scaled) + kappa)
}

exact_refit_log_vmf_const <- function(kappa, d) {
  vapply(kappa, exact_refit_log_vmf_const_one, numeric(1), d = d)
}

exact_refit_A <- function(kappa, d) {
  if (kappa < 1e-8) return(kappa / d)
  nu <- d / 2 - 1
  if (nu < 50) {
    den <- besselI(kappa, nu, expon.scaled = TRUE)
    num <- besselI(kappa, nu + 1, expon.scaled = TRUE)
    if (is.finite(den) && den > 0 && is.finite(num)) return(num / den)
  }
  # Numerical derivative keeps the gradient consistent with the high-d
  # normalizer approximation used in the objective.
  h <- 1e-5 * max(1, kappa)
  if (kappa > h) {
    return(-(
      exact_refit_log_vmf_const_one(kappa + h, d) -
        exact_refit_log_vmf_const_one(kappa - h, d)
    ) / (2 * h))
  }
  -(
    exact_refit_log_vmf_const_one(kappa + h, d) -
      exact_refit_log_vmf_const_one(kappa, d)
  ) / h
}

exact_refit_pack <- function(eta, active) {
  K <- nrow(eta)
  baseline <- colMeans(eta)
  centered <- sweep(eta, 2L, baseline, "-")
  c(baseline, as.vector(centered[seq_len(K - 1L), active, drop = FALSE]))
}

exact_refit_unpack <- function(par, K, d, active) {
  baseline <- par[seq_len(d)]
  eta <- matrix(baseline, nrow = K, ncol = d, byrow = TRUE)
  m <- sum(active)
  if (m > 0L) {
    free <- matrix(par[d + seq_len((K - 1L) * m)], nrow = K - 1L, ncol = m)
    contrast <- rbind(free, -colSums(free))
    eta[, active] <- sweep(contrast, 2L, baseline[active], "+")
  }
  eta
}

exact_refit_project <- function(eta, active) {
  baseline <- colMeans(eta)
  centered <- sweep(eta, 2L, baseline, "-")
  centered[, !active] <- 0
  sweep(centered, 2L, baseline, "+")
}

exact_refit_constraint_error <- function(eta, active) {
  if (all(active)) return(0)
  centered <- sweep(eta, 2L, colMeans(eta), "-")
  max(abs(centered[, !active, drop = FALSE]))
}

exact_refit_eta_to_theta <- function(alpha, eta, fallback_mu = NULL) {
  K <- nrow(eta)
  kappa <- sqrt(rowSums(eta * eta))
  mu <- matrix(0, K, ncol(eta))
  for (k in seq_len(K)) {
    if (kappa[k] > 1e-12) {
      mu[k, ] <- eta[k, ] / kappa[k]
    } else if (!is.null(fallback_mu)) {
      mu[k, ] <- fallback_mu[k, ]
    } else {
      mu[k, 1L] <- 1
    }
  }
  alpha <- pmax(alpha, 1e-12)
  list(alpha = alpha / sum(alpha), mu = mu, kappa = kappa)
}

exact_refit_q <- function(eta, r, Nk) {
  kappa <- sqrt(rowSums(eta * eta))
  sum(eta * r) + sum(Nk * exact_refit_log_vmf_const(kappa, ncol(eta)))
}

exact_refit_eta_gradient <- function(eta, r, Nk) {
  K <- nrow(eta)
  d <- ncol(eta)
  out <- r
  kappa <- sqrt(rowSums(eta * eta))
  for (k in seq_len(K)) {
    if (kappa[k] > 1e-12) {
      out[k, ] <- out[k, ] -
        Nk[k] * exact_refit_A(kappa[k], d) * eta[k, ] / kappa[k]
    }
  }
  out
}

exact_refit_value_gradient <- function(par, K, d, active, r, Nk) {
  eta <- exact_refit_unpack(par, K, d, active)
  grad_eta <- exact_refit_eta_gradient(eta, r, Nk)
  grad_baseline <- colSums(grad_eta)
  grad_free <- if (any(active)) {
    grad_eta[seq_len(K - 1L), active, drop = FALSE] -
      matrix(grad_eta[K, active], nrow = K - 1L, ncol = sum(active), byrow = TRUE)
  } else {
    matrix(numeric(), nrow = K - 1L, ncol = 0L)
  }
  list(
    value = exact_refit_q(eta, r, Nk),
    gradient = c(grad_baseline, as.vector(grad_free)),
    eta = eta
  )
}

exact_refit_mstep <- function(r, Nk, active, eta_start, maxit = 100L,
                              factr = 1e7, pgtol = 1e-6) {
  K <- nrow(r)
  d <- ncol(r)
  eta_start <- exact_refit_project(eta_start, active)
  par0 <- exact_refit_pack(eta_start, active)
  fn <- function(par) -exact_refit_value_gradient(par, K, d, active, r, Nk)$value
  gr <- function(par) -exact_refit_value_gradient(par, K, d, active, r, Nk)$gradient
  start <- proc.time()[["elapsed"]]
  opt <- stats::optim(
    par0, fn, gr, method = "L-BFGS-B",
    control = list(maxit = maxit, factr = factr, pgtol = pgtol)
  )
  elapsed <- proc.time()[["elapsed"]] - start
  eta <- exact_refit_unpack(opt$par, K, d, active)
  gradient <- exact_refit_value_gradient(opt$par, K, d, active, r, Nk)$gradient
  list(
    eta = eta,
    Q = -opt$value,
    convergence = opt$convergence,
    counts = opt$counts,
    message = opt$message,
    elapsed_sec = elapsed,
    gradient_max_abs = max(abs(gradient)),
    parameter_count = length(opt$par)
  )
}

exact_refit_projected_target <- function(r, Nk, active, estimate_kappa_fn) {
  K <- nrow(r)
  d <- ncol(r)
  mu <- r
  norms <- sqrt(rowSums(r * r))
  for (k in seq_len(K)) {
    if (norms[k] > 1e-12) mu[k, ] <- r[k, ] / norms[k]
  }
  kappa <- vapply(
    seq_len(K),
    function(k) estimate_kappa_fn(norms[k] / Nk[k], d, 1e6),
    numeric(1)
  )
  exact_refit_project(sweep(mu, 1L, kappa, "*"), active)
}
