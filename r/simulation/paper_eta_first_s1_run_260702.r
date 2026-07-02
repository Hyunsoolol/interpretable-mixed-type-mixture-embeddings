# ==============================================================================
# Paper simulation S1: eta-first decision-support design
# ------------------------------------------------------------------------------
# Scenario S1:
#   K = 4, n = 1000, d = 200
#   common q = 4, decision q = 16 (4 per component), noise q = 180
#   target mean pairwise direction angle ~= 80 degrees
#   heterogeneous kappa = (20, 30, 40, 50)
#
# Methods, all selected by BIC and then support-refit:
#   D-L    : direction entry-wise lasso
#   D-GL   : direction group lasso
#   D-AGL  : adaptive direction group lasso
#   E-L    : centered eta entry-wise lasso
#   E-GL   : centered eta group lasso
#   E-AGL  : adaptive centered eta group lasso
#
# This runner is separate from the existing diagnostic runners and does not modify
# the official method files.
# ==============================================================================

source_k4_helpers <- function() {
  source_no_bom <- function(file) {
    txt <- readLines(file, encoding = "UTF-8", warn = FALSE)
    if (length(txt) > 0) txt[1] <- sub("^\ufeff", "", txt[1])
    eval(parse(text = txt), envir = .GlobalEnv)
  }
  source_no_bom(file.path("r", "methods", "rossi_barbaro_2022_reproduction.r"))

  lines <- readLines(
    file.path("r", "methods", "rb2022_k4_pilot_compare_run.r"),
    encoding = "UTF-8",
    warn = FALSE
  )
  if (length(lines) > 0) lines[1] <- sub("^\ufeff", "", lines[1])
  idx <- grep("fit_rossi_pair <-", lines, fixed = TRUE)[1]
  if (is.na(idx)) stop("Could not find helper boundary in rb2022_k4_pilot_compare_run.r.")
  eval(parse(text = lines[seq_len(idx - 1L)]), envir = .GlobalEnv)
}

source_k4_helpers()

parse_num_grid <- function(x) as.numeric(strsplit(x, ",", fixed = TRUE)[[1]])

parse_bool_env <- function(name, default = "0") {
  tolower(Sys.getenv(name, default)) %in% c("1", "true", "t", "yes", "y", "on")
}

cfg <- list(
  run_label = Sys.getenv("PAPER_S1_LABEL", "paper_eta_first_s1_smoke_260702"),
  scenario_id = Sys.getenv("PAPER_S1_SCENARIO_ID", "S1_moderate_angle_heterogeneous_kappa"),
  scenario_desc = Sys.getenv("PAPER_S1_SCENARIO_DESC", "S1 moderate mean-direction difference with heterogeneous concentration."),
  n_rep = as.integer(Sys.getenv("PAPER_S1_N_REP", "1")),
  n = as.integer(Sys.getenv("PAPER_S1_N", "1000")),
  d = as.integer(Sys.getenv("PAPER_S1_D", "200")),
  K = as.integer(Sys.getenv("PAPER_S1_K", "4")),
  common_q = as.integer(Sys.getenv("PAPER_S1_COMMON_Q", "4")),
  decision_per_component = as.integer(Sys.getenv("PAPER_S1_DECISION_PER_COMPONENT", "4")),
  target_angle_deg = as.numeric(Sys.getenv("PAPER_S1_TARGET_ANGLE_DEG", "80")),
  kappa = parse_num_grid(Sys.getenv("PAPER_S1_KAPPA", "20,30,40,50")),
  nstart = as.integer(Sys.getenv("PAPER_S1_NSTART", "3")),
  max_iter = as.integer(Sys.getenv("PAPER_S1_MAX_ITER", "80")),
  d_l_steps = as.integer(Sys.getenv("PAPER_S1_D_L_STEPS", "60")),
  group_steps = as.integer(Sys.getenv("PAPER_S1_GROUP_STEPS", "60")),
  eta_steps = as.integer(Sys.getenv("PAPER_S1_ETA_STEPS", "60")),
  min_rel_lambda = as.numeric(Sys.getenv("PAPER_S1_MIN_REL_LAMBDA", "1e-3")),
  adaptive_gamma = as.numeric(Sys.getenv("PAPER_S1_ADAPTIVE_GAMMA", "1")),
  adaptive_eps = as.numeric(Sys.getenv("PAPER_S1_ADAPTIVE_EPS", "1e-6")),
  select_ic = toupper(Sys.getenv("PAPER_S1_SELECT_IC", "BIC")),
  base_seed = as.integer(Sys.getenv("PAPER_S1_BASE_SEED", "20260702")),
  use_rcpp = parse_bool_env("PAPER_S1_USE_RCPP", "1"),
  out_dir = Sys.getenv("PAPER_S1_OUT_DIR", "results/paper_eta_first_s1_smoke_260702")
)

if (!cfg$select_ic %in% c("BIC", "EBIC")) stop("PAPER_S1_SELECT_IC must be BIC or EBIC.")
if (length(cfg$kappa) != cfg$K) stop("PAPER_S1_KAPPA length must match K.")
if (cfg$common_q + cfg$K * cfg$decision_per_component > cfg$d) {
  stop("common_q + K * decision_per_component must be <= d.")
}
if (cfg$use_rcpp) Sys.setenv(USE_RCPP_HELPERS = "1")

dir.create(cfg$out_dir, recursive = TRUE, showWarnings = FALSE)

mean_pairwise_cos_for_H2 <- function(H2, kappa, v_norm2, v_dot) {
  t <- sqrt(pmax(kappa^2 - H2, 0) / v_norm2)
  K <- length(kappa)
  vals <- numeric(0)
  for (i in seq_len(K - 1L)) {
    for (j in (i + 1L):K) {
      vals <- c(vals, (H2 + t[i] * t[j] * v_dot) / (kappa[i] * kappa[j]))
    }
  }
  mean(vals)
}

calibrate_common_norm2 <- function(kappa, target_angle_deg, v_norm2, v_dot) {
  target <- cos(target_angle_deg * pi / 180)
  upper <- min(kappa)^2 * 0.999
  f <- function(H2) mean_pairwise_cos_for_H2(H2, kappa, v_norm2, v_dot) - target
  f0 <- f(0)
  f1 <- f(upper)
  if (is.finite(f0) && is.finite(f1) && f0 * f1 <= 0) {
    return(uniroot(f, lower = 0, upper = upper, tol = 1e-10)$root)
  }
  grid <- seq(0, upper, length.out = 2000)
  grid[which.min(abs(vapply(grid, f, numeric(1))))]
}

make_eta_first_s1_params <- function(cfg) {
  K <- cfg$K
  d <- cfg$d
  common_q <- cfg$common_q
  per_k <- cfg$decision_per_component
  decision_q <- K * per_k
  kappa <- cfg$kappa

  common_idx <- seq_len(common_q)
  decision_idx <- common_q + seq_len(decision_q)
  noise_idx <- if (common_q + decision_q < d) seq.int(common_q + decision_q + 1L, d) else integer(0)

  contrast <- matrix(0, nrow = K, ncol = decision_q)
  for (g in seq_len(K)) {
    idx <- ((g - 1L) * per_k) + seq_len(per_k)
    contrast[, idx] <- -1 / (K - 1)
    contrast[g, idx] <- 1
  }
  v_norm2 <- sum(contrast[1, ]^2)
  v_dot <- sum(contrast[1, ] * contrast[2, ])
  H2 <- calibrate_common_norm2(kappa, cfg$target_angle_deg, v_norm2, v_dot)
  common_value <- sqrt(H2 / common_q)
  scale_by_k <- sqrt(pmax(kappa^2 - H2, 0) / v_norm2)

  eta <- matrix(0, nrow = K, ncol = d)
  eta[, common_idx] <- common_value
  for (k in seq_len(K)) eta[k, decision_idx] <- scale_by_k[k] * contrast[k, ]

  mu <- sweep(eta, 1, sqrt(rowSums(eta^2)), "/")
  kappa_actual <- sqrt(rowSums(eta^2))
  support <- matrix(FALSE, nrow = K, ncol = d)
  support[, decision_idx] <- TRUE

  pair_cos <- tcrossprod(mu)
  pair_angle <- acos(pmin(pmax(pair_cos[upper.tri(pair_cos)], -1), 1)) * 180 / pi

  list(
    alpha = rep(1 / K, K),
    mu = mu,
    kappa = kappa_actual,
    eta = eta,
    support = support,
    common_idx = common_idx,
    decision_idx = decision_idx,
    noise_idx = noise_idx,
    decision_blocks = split(decision_idx, rep(seq_len(K), each = per_k)),
    common_value = common_value,
    contrast_scale = scale_by_k,
    mu_pairwise_angle_mean = mean(pair_angle),
    mu_pairwise_angle_min = min(pair_angle),
    mu_pairwise_angle_max = max(pair_angle),
    mu_pairwise_cos_mean = mean(pair_cos[upper.tri(pair_cos)]),
    target_angle_deg = cfg$target_angle_deg
  )
}

simulate_from_params <- function(n, params) {
  K <- nrow(params$mu)
  z <- sample.int(K, size = n, replace = TRUE, prob = params$alpha)
  X <- matrix(0, nrow = n, ncol = ncol(params$mu))
  for (k in seq_len(K)) {
    idx <- which(z == k)
    if (length(idx) > 0) X[idx, ] <- rvMF(length(idx), params$mu[k, ], params$kappa[k])
  }
  list(X = X, z = z, params = params)
}

selection_type_metrics <- function(active, params) {
  data.frame(
    common_false_selection_rate = if (length(params$common_idx)) mean(active[params$common_idx]) else NA_real_,
    decision_selection_rate = if (length(params$decision_idx)) mean(active[params$decision_idx]) else NA_real_,
    noise_false_selection_rate = if (length(params$noise_idx)) mean(active[params$noise_idx]) else NA_real_
  )
}

append_type_metrics <- function(row, active, params) cbind(row, selection_type_metrics(active, params))

add_runtime_columns <- function(row, refit_status = "support_refit") {
  defaults <- list(
    objective = NA_real_, n_decrease = NA_integer_, min_objective_diff = NA_real_,
    line_search_halving = NA_integer_, line_search_accepted = NA,
    adaptive_penalty = NA_integer_, adaptive_gamma = NA_real_, adaptive_eps = NA_real_,
    adaptive_weight_min = NA_real_, adaptive_weight_median = NA_real_, adaptive_weight_max = NA_real_,
    refit_status = refit_status
  )
  for (nm in names(defaults)) if (!nm %in% names(row)) row[[nm]] <- defaults[[nm]]
  row$refit_status <- refit_status
  row
}

fit_zero_refit_row <- function(method, base_row, active) {
  out <- base_row
  out$method <- method
  for (nm in c("ARI", "loglik", "pen_loglik", "converged", "iter",
               "MSE_mu", "MSE_kappa", "MSE_centered_eta", "kappa_hat_mean",
               "df", "BIC", "EBIC")) {
    if (nm %in% names(out)) out[[nm]] <- NA_real_
  }
  out$selected_q <- sum(active)
  out$refit_status <- "zero_active_support"
  out
}

safe_best <- function(tab, select_ic) which.min(tab[[select_ic]])

method_row <- function(name, fit, X, z, params, active, lambda_mu = NA_real_,
                       lambda_eta = NA_real_, beta = NA_real_, ic = NULL,
                       support_entry = NULL) {
  row <- eval_method(
    name, fit, X, z, params, active, support_entry,
    lambda_mu = lambda_mu, lambda_eta = lambda_eta, beta = beta, ic = ic
  )
  row
}

fit_d_l <- function(X, z, params, cfg) {
  path <- fit_svMF_path(
    X = X, K = cfg$K, labels_true = z, mu_true = params$mu,
    support_true = params$support, nstart = cfg$nstart,
    max_path_steps = cfg$d_l_steps, max_iter = cfg$max_iter,
    gamma = 0.5, verbose = FALSE
  )
  idx <- safe_best(path$path, cfg$select_ic)
  fit <- path$fits[[idx]]
  prow <- path$path[idx, , drop = FALSE]
  active <- active_mu_coord(fit)
  refit <- fit_support_refit(X, cfg$K, active, fit, max_iter = cfg$max_iter)
  row <- method_row("D-L", refit, X, z, params, active, beta = prow$beta)
  row$penalty_target <- "mu_entry"
  row$penalty_group <- 0L
  row$penalty_adaptive <- 0L
  add_runtime_columns(append_type_metrics(row, active, params))
}

active_mu_group <- function(theta, zero_eps = 1e-8) sqrt(colSums(theta$mu^2)) > zero_eps

mu_group_penalty_value <- function(mu, weights = NULL) {
  norms <- sqrt(colSums(mu * mu))
  if (is.null(weights)) weights <- rep(1, length(norms))
  sum(weights * norms)
}

prox_mu_group <- function(mu, lambda, weights = NULL) {
  norms <- sqrt(colSums(mu * mu))
  if (is.null(weights)) weights <- rep(1, length(norms))
  scale <- ifelse(norms > 0, pmax(1 - lambda * weights / norms, 0), 0)
  sweep(mu, 2, scale, "*")
}

normalize_rows_masked <- function(mu, active, fallback_mu = NULL, zero_eps = 1e-12) {
  if (any(active)) mu[, !active] <- 0
  for (k in seq_len(nrow(mu))) {
    nr <- l2_norm(mu[k, ])
    if (nr > zero_eps) {
      mu[k, ] <- mu[k, ] / nr
    } else if (any(active) && !is.null(fallback_mu)) {
      tmp <- fallback_mu[k, ]
      tmp[!active] <- 0
      tmp_norm <- l2_norm(tmp)
      if (tmp_norm > zero_eps) {
        mu[k, ] <- tmp / tmp_norm
      } else {
        mu[k, ] <- 0
        mu[k, which(active)[1]] <- 1
      }
    } else if (any(active)) {
      mu[k, ] <- 0
      mu[k, which(active)[1]] <- 1
    } else if (!is.null(fallback_mu)) {
      mu[k, ] <- fallback_mu[k, ]
    } else {
      mu[k, ] <- 0
      mu[k, 1] <- 1
    }
  }
  mu
}

fit_d_group_em <- function(X, K, lambda, init, weights = NULL, max_iter = 80, tol = 1e-7) {
  theta <- init
  theta$mu <- normalize_rows(theta$mu)
  theta$alpha <- pmax(theta$alpha, 1e-12)
  theta$alpha <- theta$alpha / sum(theta$alpha)
  if (length(theta$kappa) == 1) theta$kappa <- rep(theta$kappa, K)
  if (is.null(weights)) weights <- rep(1, ncol(X))

  prev_obj <- -Inf
  last_e <- NULL
  n_decrease <- 0L
  min_objective_diff <- Inf
  max_line_search_halving <- 0L
  line_search_accepted_all <- TRUE

  for (iter in seq_len(max_iter)) {
    e <- e_step_vmf(X, theta)
    mstep <- unpenalized_eta_mstep(X, e$tau)
    theta_target_raw <- eta_to_theta(mstep$alpha, mstep$eta, fallback_mu = theta$mu)
    mu_target_raw <- prox_mu_group(theta_target_raw$mu, lambda, weights)
    active_target <- sqrt(colSums(mu_target_raw^2)) > 1e-12
    mu_target <- normalize_rows_masked(mu_target_raw, active_target, fallback_mu = theta$mu)
    mu_old <- normalize_rows_masked(theta$mu, active_target, fallback_mu = theta$mu)

    step_size <- 1
    halving <- 0L
    accepted <- FALSE
    repeat {
      alpha_new <- theta$alpha + step_size * (theta_target_raw$alpha - theta$alpha)
      kappa_new <- theta$kappa + step_size * (theta_target_raw$kappa - theta$kappa)
      mu_new <- mu_old + step_size * (mu_target - mu_old)
      mu_new <- normalize_rows_masked(mu_new, active_target, fallback_mu = theta$mu)
      theta_try <- list(
        alpha = pmax(alpha_new, 1e-12) / sum(pmax(alpha_new, 1e-12)),
        mu = mu_new,
        kappa = pmax(kappa_new, 1e-10)
      )
      e_try <- e_step_vmf(X, theta_try)
      obj_try <- e_try$loglik - lambda * mu_group_penalty_value(theta_try$mu, weights)
      if (!is.finite(prev_obj) || obj_try >= prev_obj - 1e-8 || halving >= 25L) {
        theta_new <- theta_try
        e_new <- e_try
        obj <- obj_try
        accepted <- !is.finite(prev_obj) || obj_try >= prev_obj - 1e-8
        break
      }
      step_size <- step_size / 2
      halving <- halving + 1L
    }

    max_line_search_halving <- max(max_line_search_halving, halving)
    line_search_accepted_all <- line_search_accepted_all && accepted
    if (is.finite(prev_obj)) {
      objective_diff <- obj - prev_obj
      min_objective_diff <- min(min_objective_diff, objective_diff)
      if (objective_diff < -1e-8) n_decrease <- n_decrease + 1L
    }
    theta <- theta_new
    last_e <- e_new
    if (is.finite(prev_obj) && abs(obj - prev_obj) / max(1, abs(prev_obj)) < tol) {
      return(c(theta, list(
        failed = FALSE, converged = TRUE, iter = iter, loglik = e_new$loglik,
        pen_loglik = obj, tau = e_new$tau, objective = obj,
        n_decrease = n_decrease,
        min_objective_diff = ifelse(is.infinite(min_objective_diff), NA_real_, min_objective_diff),
        line_search_halving = max_line_search_halving,
        line_search_accepted = line_search_accepted_all
      )))
    }
    prev_obj <- obj
  }
  if (is.null(last_e)) last_e <- e_step_vmf(X, theta)
  final_obj <- last_e$loglik - lambda * mu_group_penalty_value(theta$mu, weights)
  c(theta, list(
    failed = FALSE, converged = FALSE, iter = max_iter, loglik = last_e$loglik,
    pen_loglik = final_obj, tau = last_e$tau, objective = final_obj,
    n_decrease = n_decrease,
    min_objective_diff = ifelse(is.infinite(min_objective_diff), NA_real_, min_objective_diff),
    line_search_halving = max_line_search_halving,
    line_search_accepted = line_search_accepted_all
  ))
}

group_weights_from_norm <- function(norms, gamma, eps) {
  w <- (norms + eps)^(-gamma)
  med <- median(w[is.finite(w) & w > 0])
  if (is.finite(med) && med > 0) w <- w / med
  w
}

fit_d_group_path <- function(X, z, params, cfg, adaptive = FALSE) {
  dense <- fit_svMF_multistart(X, cfg$K, beta = 0, nstart = cfg$nstart, max_iter = cfg$max_iter)
  weights <- rep(1, ncol(X))
  if (adaptive) {
    weights <- group_weights_from_norm(
      sqrt(colSums(dense$mu^2)), cfg$adaptive_gamma, cfg$adaptive_eps
    )
  }
  lambda <- 0
  fit <- fit_d_group_em(X, cfg$K, lambda, init = dense, weights = weights, max_iter = cfg$max_iter)
  rows <- list()
  fits <- list()
  add_row <- function(fit, lambda) {
    active <- active_mu_group(fit)
    ic <- support_ic(fit$loglik, nrow(X), ncol(X), cfg$K, sum(active))
    row <- method_row(if (adaptive) "D-AGL path" else "D-GL path",
                      fit, X, z, params, active, lambda_mu = lambda, ic = ic)
    row$objective <- ifelse(is.null(fit$objective), fit$pen_loglik, fit$objective)
    row$n_decrease <- ifelse(is.null(fit$n_decrease), NA_integer_, fit$n_decrease)
    row$min_objective_diff <- ifelse(is.null(fit$min_objective_diff), NA_real_, fit$min_objective_diff)
    row$line_search_halving <- ifelse(is.null(fit$line_search_halving), NA_integer_, fit$line_search_halving)
    row$line_search_accepted <- ifelse(is.null(fit$line_search_accepted), NA, fit$line_search_accepted)
    row
  }
  rows[[1L]] <- add_row(fit, lambda)
  fits[[1L]] <- fit
  for (step in 2:cfg$group_steps) {
    e <- e_step_vmf(X, fit)
    mstep <- unpenalized_eta_mstep(X, e$tau)
    theta_target_raw <- eta_to_theta(mstep$alpha, mstep$eta, fallback_mu = fit$mu)
    norms <- sqrt(colSums(theta_target_raw$mu^2))
    thresholds <- norms / pmax(weights, 1e-12)
    candidates <- thresholds[thresholds > lambda + 1e-10]
    if (length(candidates) == 0) break
    lambda_next <- min(candidates)
    if (lambda > 0) lambda_next <- max(lambda_next, lambda * (1 + cfg$min_rel_lambda))
    if (!is.finite(lambda_next) || lambda_next <= lambda) break
    fit_next <- tryCatch(
      fit_d_group_em(X, cfg$K, lambda_next, init = fit, weights = weights, max_iter = cfg$max_iter),
      error = function(e) NULL
    )
    if (is.null(fit_next) || isTRUE(fit_next$failed)) break
    fit <- fit_next
    lambda <- lambda_next
    rows[[length(rows) + 1L]] <- add_row(fit, lambda)
    fits[[length(fits) + 1L]] <- fit
    if (sum(active_mu_group(fit)) <= 1) break
  }
  tab <- do.call(rbind, rows)
  idx <- safe_best(tab, cfg$select_ic)
  fit_best <- fits[[idx]]
  active <- active_mu_group(fit_best)
  base <- tab[idx, , drop = FALSE]
  name <- if (adaptive) "D-AGL" else "D-GL"
  if (!any(active)) {
    out <- append_type_metrics(fit_zero_refit_row(name, base, active), active, params)
  } else {
    refit <- fit_support_refit(X, cfg$K, active, fit_best, max_iter = cfg$max_iter)
    out <- method_row(name, refit, X, z, params, active, lambda_mu = base$lambda_mu)
    out <- append_type_metrics(out, active, params)
  }
  out$penalty_target <- "mu_group"
  out$penalty_group <- 1L
  out$penalty_adaptive <- as.integer(adaptive)
  out$adaptive_penalty <- as.integer(adaptive)
  out$adaptive_gamma <- ifelse(adaptive, cfg$adaptive_gamma, NA_real_)
  out$adaptive_eps <- ifelse(adaptive, cfg$adaptive_eps, NA_real_)
  out$adaptive_weight_min <- ifelse(adaptive, min(weights), NA_real_)
  out$adaptive_weight_median <- ifelse(adaptive, median(weights), NA_real_)
  out$adaptive_weight_max <- ifelse(adaptive, max(weights), NA_real_)
  add_runtime_columns(out)
}

soft_threshold <- function(x, lambda) sign(x) * pmax(abs(x) - lambda, 0)

prox_eta_l1_col <- function(y, lambda_eta) {
  mu0 <- mean(y)
  z <- y - mu0
  f <- function(tau) sum(soft_threshold(z - tau, lambda_eta))
  if (abs(f(0)) < 1e-10) return(mu0 + soft_threshold(z, lambda_eta))
  lower <- min(z) - lambda_eta - 10
  upper <- max(z) + lambda_eta + 10
  tau <- uniroot(f, lower = lower, upper = upper, tol = 1e-10)$root
  mu0 + soft_threshold(z - tau, lambda_eta)
}

prox_eta_l1 <- function(eta, lambda_eta) apply(eta, 2, prox_eta_l1_col, lambda_eta = lambda_eta)

active_eta_l1 <- function(theta, zero_eps = 1e-8) {
  colSums(abs(center_eta(eta_matrix(theta))) > zero_eps) > 0
}

eta_l1_ic <- function(theta, n, d, loglik, gamma = 0.5, zero_eps = 1e-8) {
  centered <- center_eta(eta_matrix(theta))
  nonzero_by_coord <- colSums(abs(centered) > zero_eps)
  df <- (nrow(theta$mu) - 1) + ncol(theta$mu) + sum(pmax(nonzero_by_coord - 1, 0))
  data.frame(df = df, BIC = log(n) * df - 2 * loglik,
             EBIC = (log(n) + 2 * gamma * log(d)) * df - 2 * loglik)
}

fit_eta_l1_em <- function(X, K, lambda_eta, init, max_iter = 80, tol = 1e-7) {
  theta <- init
  theta$mu <- normalize_rows(theta$mu)
  theta$alpha <- pmax(theta$alpha, 1e-12)
  theta$alpha <- theta$alpha / sum(theta$alpha)
  if (length(theta$kappa) == 1) theta$kappa <- rep(theta$kappa, K)
  prev_obj <- -Inf
  last_e <- NULL
  for (iter in seq_len(max_iter)) {
    e <- e_step_vmf(X, theta)
    mstep <- unpenalized_eta_mstep(X, e$tau)
    eta_new <- prox_eta_l1(mstep$eta, lambda_eta)
    theta <- eta_to_theta(mstep$alpha, eta_new, fallback_mu = theta$mu)
    e_new <- e_step_vmf(X, theta)
    pen_value <- sum(abs(center_eta(eta_matrix(theta))))
    obj <- e_new$loglik - lambda_eta * pen_value
    last_e <- e_new
    if (is.finite(prev_obj) && abs(obj - prev_obj) / max(1, abs(prev_obj)) < tol) {
      return(c(theta, list(failed = FALSE, converged = TRUE, iter = iter,
                           loglik = e_new$loglik, pen_loglik = obj, tau = e_new$tau)))
    }
    prev_obj <- obj
  }
  if (is.null(last_e)) last_e <- e_step_vmf(X, theta)
  pen_value <- sum(abs(center_eta(eta_matrix(theta))))
  c(theta, list(failed = FALSE, converged = FALSE, iter = max_iter,
                loglik = last_e$loglik,
                pen_loglik = last_e$loglik - lambda_eta * pen_value,
                tau = last_e$tau))
}

fit_eta_path <- function(X, z, params, cfg, method = c("E-L", "E-GL", "E-AGL")) {
  method <- match.arg(method)
  dense <- fit_svMF_multistart(X, cfg$K, beta = 0, nstart = cfg$nstart, max_iter = cfg$max_iter)
  weights <- rep(1, ncol(X))
  if (method == "E-AGL") {
    eta0 <- eta_matrix(dense)
    weights <- group_weights_from_norm(
      sqrt(colSums(center_eta(eta0)^2)), cfg$adaptive_gamma, cfg$adaptive_eps
    )
  }
  lambda <- 0
  fit <- if (method == "E-L") {
    fit_eta_l1_em(X, cfg$K, lambda, init = dense, max_iter = cfg$max_iter)
  } else {
    fit_eta_centered_em(
      X, cfg$K, lambda, init = dense, max_iter = cfg$max_iter,
      adaptive_weights = weights
    )
  }
  rows <- list()
  fits <- list()
  add_row <- function(fit, lambda) {
    active <- if (method == "E-L") active_eta_l1(fit) else active_eta_centered(fit)
    ic <- if (method == "E-L") {
      eta_l1_ic(fit, nrow(X), ncol(X), fit$loglik)
    } else {
      eta_centered_ic(fit, nrow(X), ncol(X), fit$loglik)
    }
    row <- method_row(paste(method, "path"), fit, X, z, params, active,
                      lambda_eta = lambda, ic = ic)
    row$objective <- ifelse(is.null(fit$objective), fit$pen_loglik, fit$objective)
    row$n_decrease <- ifelse(is.null(fit$n_decrease), NA_integer_, fit$n_decrease)
    row$min_objective_diff <- ifelse(is.null(fit$min_objective_diff), NA_real_, fit$min_objective_diff)
    row$line_search_halving <- ifelse(is.null(fit$line_search_halving), NA_integer_, fit$line_search_halving)
    row$line_search_accepted <- ifelse(is.null(fit$line_search_accepted), NA, fit$line_search_accepted)
    row
  }
  rows[[1L]] <- add_row(fit, lambda)
  fits[[1L]] <- fit
  for (step in 2:cfg$eta_steps) {
    e <- e_step_vmf(X, fit)
    mstep <- unpenalized_eta_mstep(X, e$tau)
    centered <- center_eta(mstep$eta)
    thresholds <- if (method == "E-L") abs(as.vector(centered)) else sqrt(colSums(centered^2)) / pmax(weights, 1e-12)
    candidates <- thresholds[thresholds > lambda + 1e-10]
    if (length(candidates) == 0) break
    lambda_next <- min(candidates)
    if (lambda > 0) lambda_next <- max(lambda_next, lambda * (1 + cfg$min_rel_lambda))
    if (!is.finite(lambda_next) || lambda_next <= lambda) break
    fit_next <- tryCatch(
      if (method == "E-L") {
        fit_eta_l1_em(X, cfg$K, lambda_next, init = fit, max_iter = cfg$max_iter)
      } else {
        fit_eta_centered_em(
          X, cfg$K, lambda_next, init = fit, max_iter = cfg$max_iter,
          adaptive_weights = weights
        )
      },
      error = function(e) NULL
    )
    if (is.null(fit_next) || isTRUE(fit_next$failed)) break
    fit <- fit_next
    lambda <- lambda_next
    rows[[length(rows) + 1L]] <- add_row(fit, lambda)
    fits[[length(fits) + 1L]] <- fit
    active_count <- if (method == "E-L") sum(active_eta_l1(fit)) else sum(active_eta_centered(fit))
    if (active_count <= 1) break
  }
  tab <- do.call(rbind, rows)
  idx <- safe_best(tab, cfg$select_ic)
  fit_best <- fits[[idx]]
  active <- if (method == "E-L") active_eta_l1(fit_best) else active_eta_centered(fit_best)
  base <- tab[idx, , drop = FALSE]
  if (!any(active)) {
    out <- append_type_metrics(fit_zero_refit_row(method, base, active), active, params)
  } else {
    refit <- fit_support_refit(X, cfg$K, active, fit_best, max_iter = cfg$max_iter)
    out <- method_row(method, refit, X, z, params, active, lambda_eta = base$lambda_eta)
    out <- append_type_metrics(out, active, params)
  }
  out$penalty_target <- ifelse(method == "E-L", "eta_entry", "eta_group")
  out$penalty_group <- as.integer(method != "E-L")
  out$penalty_adaptive <- as.integer(method == "E-AGL")
  out$adaptive_penalty <- as.integer(method == "E-AGL")
  out$adaptive_gamma <- ifelse(method == "E-AGL", cfg$adaptive_gamma, NA_real_)
  out$adaptive_eps <- ifelse(method == "E-AGL", cfg$adaptive_eps, NA_real_)
  out$adaptive_weight_min <- ifelse(method == "E-AGL", min(weights), NA_real_)
  out$adaptive_weight_median <- ifelse(method == "E-AGL", median(weights), NA_real_)
  out$adaptive_weight_max <- ifelse(method == "E-AGL", max(weights), NA_real_)
  add_runtime_columns(out)
}

run_one <- function(rep_id, cfg) {
  set.seed(cfg$base_seed + rep_id)
  params <- make_eta_first_s1_params(cfg)
  dat <- simulate_from_params(cfg$n, params)
  cat(sprintf(
    "[%s] rep %d/%d: target angle=%.1f, angle mean/min/max=%.2f/%.2f/%.2f, kappa=(%s), true decision q=%d\n",
    cfg$scenario_id, rep_id, cfg$n_rep, cfg$target_angle_deg,
    params$mu_pairwise_angle_mean, params$mu_pairwise_angle_min,
    params$mu_pairwise_angle_max, paste(round(params$kappa, 3), collapse = ","),
    length(params$decision_idx)
  ))
  rows <- list(
    fit_d_l(dat$X, dat$z, dat$params, cfg),
    fit_d_group_path(dat$X, dat$z, dat$params, cfg, adaptive = FALSE),
    fit_d_group_path(dat$X, dat$z, dat$params, cfg, adaptive = TRUE),
    fit_eta_path(dat$X, dat$z, dat$params, cfg, "E-L"),
    fit_eta_path(dat$X, dat$z, dat$params, cfg, "E-GL"),
    fit_eta_path(dat$X, dat$z, dat$params, cfg, "E-AGL")
  )
  out <- do.call(rbind, rows)
  out$scenario <- cfg$scenario_id
  out$rep <- rep_id
  out$n <- cfg$n
  out$d <- cfg$d
  out$K_true <- cfg$K
  out$common_q <- cfg$common_q
  out$decision_q <- length(params$decision_idx)
  out$decision_q_per_component <- cfg$decision_per_component
  out$noise_q <- length(params$noise_idx)
  out$true_q <- length(params$decision_idx)
  out$target_angle_deg <- cfg$target_angle_deg
  out$mu_pairwise_angle_mean <- params$mu_pairwise_angle_mean
  out$mu_pairwise_angle_min <- params$mu_pairwise_angle_min
  out$mu_pairwise_angle_max <- params$mu_pairwise_angle_max
  out$kappa_true_min <- min(params$kappa)
  out$kappa_true_max <- max(params$kappa)
  out$kappa_true_mean <- mean(params$kappa)
  out$kappa_true_ratio <- max(params$kappa) / min(params$kappa)
  out$common_eta_value <- params$common_value
  out
}

cat(sprintf(
  "Running paper %s eta-first simulation: reps=%d, n=%d, d=%d, K=%d, nstart=%d, max_iter=%d, select=%s, Rcpp=%s\n",
  cfg$scenario_id,
  cfg$n_rep, cfg$n, cfg$d, cfg$K, cfg$nstart, cfg$max_iter,
  cfg$select_ic, ifelse(cfg$use_rcpp, "ON", "OFF")
))

row_list <- vector("list", cfg$n_rep)
for (rep_id in seq_len(cfg$n_rep)) {
  row_list[[rep_id]] <- tryCatch(
    run_one(rep_id, cfg),
    error = function(e) {
      message(sprintf("[ERROR] rep %d: %s", rep_id, conditionMessage(e)))
      data.frame(
        method = "ERROR", K_fit = NA_real_, beta = NA_real_,
        lambda_mu = NA_real_, lambda_kappa = NA_real_, lambda_eta = NA_real_,
        ARI = NA_real_, loglik = NA_real_, pen_loglik = NA_real_,
        converged = NA, iter = NA_real_, true_union_q = NA_real_,
        selected_q = NA_real_, TPR = NA_real_, FPR = NA_real_,
        Precision = NA_real_, F1 = NA_real_, entry_TPR = NA_real_,
        entry_FPR = NA_real_, entry_Precision = NA_real_, entry_F1 = NA_real_,
        MSE_mu = NA_real_, MSE_kappa = NA_real_, MSE_centered_eta = NA_real_,
        kappa_hat_mean = NA_real_, df = NA_real_, BIC = NA_real_,
        EBIC = NA_real_, common_false_selection_rate = NA_real_,
        decision_selection_rate = NA_real_, noise_false_selection_rate = NA_real_,
        objective = NA_real_, n_decrease = NA_real_, min_objective_diff = NA_real_,
        line_search_halving = NA_real_, line_search_accepted = NA,
        adaptive_penalty = NA_integer_, adaptive_gamma = NA_real_,
        adaptive_eps = NA_real_, adaptive_weight_min = NA_real_,
        adaptive_weight_median = NA_real_, adaptive_weight_max = NA_real_,
        refit_status = conditionMessage(e), penalty_target = NA_character_,
        penalty_group = NA_integer_, penalty_adaptive = NA_integer_,
        scenario = cfg$scenario_id, rep = rep_id,
        n = cfg$n, d = cfg$d, K_true = cfg$K, common_q = cfg$common_q,
        decision_q = cfg$K * cfg$decision_per_component,
        decision_q_per_component = cfg$decision_per_component,
        noise_q = cfg$d - cfg$common_q - cfg$K * cfg$decision_per_component,
        true_q = cfg$K * cfg$decision_per_component,
        target_angle_deg = cfg$target_angle_deg,
        mu_pairwise_angle_mean = NA_real_, mu_pairwise_angle_min = NA_real_,
        mu_pairwise_angle_max = NA_real_, kappa_true_min = min(cfg$kappa),
        kappa_true_max = max(cfg$kappa), kappa_true_mean = mean(cfg$kappa),
        kappa_true_ratio = max(cfg$kappa) / min(cfg$kappa),
        common_eta_value = NA_real_
      )
    }
  )
}

raw <- do.call(rbind, row_list)

safe_mean <- function(x) if (sum(!is.na(x)) == 0) NA_real_ else mean(x, na.rm = TRUE)
num_cols <- names(raw)[vapply(raw, is.numeric, logical(1))]
groups <- unique(raw[, c("scenario", "method")])
summary <- do.call(rbind, lapply(seq_len(nrow(groups)), function(i) {
  sub <- raw[raw$scenario == groups$scenario[i] & raw$method == groups$method[i], ]
  means <- as.data.frame(as.list(vapply(sub[, num_cols, drop = FALSE], safe_mean, numeric(1))))
  data.frame(
    scenario = groups$scenario[i],
    method = groups$method[i],
    reps = length(unique(sub$rep)),
    valid_reps = sum(!is.na(sub$ARI)),
    error_reps = sum(sub$method == "ERROR"),
    zero_support_refit_reps = sum(sub$refit_status == "zero_active_support", na.rm = TRUE),
    means,
    row.names = NULL
  )
}))

method_order <- c("D-L", "D-GL", "D-AGL", "E-L", "E-GL", "E-AGL", "ERROR")
summary$method <- factor(summary$method, levels = method_order)
summary <- summary[order(summary$method), ]
summary$method <- as.character(summary$method)

raw_path <- file.path(cfg$out_dir, sprintf("%s_raw.csv", cfg$run_label))
summary_path <- file.path(cfg$out_dir, sprintf("%s_summary.csv", cfg$run_label))
notes_path <- file.path(cfg$out_dir, sprintf("%s_notes.md", cfg$run_label))
write.csv(raw, raw_path, row.names = FALSE)
write.csv(summary, summary_path, row.names = FALSE)

fmt <- function(x, digits = 3) ifelse(is.na(x), "NA", formatC(as.numeric(x), digits = digits, format = "f"))

notes <- c(
  sprintf("# Paper simulation %s eta-first run", cfg$scenario_id),
  "",
  sprintf("- Date: %s", Sys.Date()),
  sprintf("- Scenario: %s", cfg$scenario_desc),
  sprintf("- Setting: K=%d, n=%d, d=%d, common q=%d, decision q=%d (%d per component), noise q=%d.",
          cfg$K, cfg$n, cfg$d, cfg$common_q, cfg$K * cfg$decision_per_component,
          cfg$decision_per_component, cfg$d - cfg$common_q - cfg$K * cfg$decision_per_component),
  sprintf("- Target pairwise direction angle: %.1f degrees.", cfg$target_angle_deg),
  sprintf("- Kappa: (%s).", paste(cfg$kappa, collapse = ", ")),
  sprintf("- Repetitions: %d.", cfg$n_rep),
  sprintf("- Tuning: %s, all rows are support-refit results.", cfg$select_ic),
  sprintf("- Rcpp helpers: %s.", ifelse(cfg$use_rcpp, "ON", "OFF")),
  "",
  "## Summary",
  "",
  "| method | reps | valid | ARI | true q | selected q | TPR | FPR | Precision | F1 | MSE_mu | MSE_kappa | MSE_centered_eta | common false | noise FPR |",
  "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
)
for (i in seq_len(nrow(summary))) {
  notes <- c(notes, sprintf(
    "| %s | %d | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |",
    summary$method[i], summary$reps[i], summary$valid_reps[i],
    fmt(summary$ARI[i]), fmt(summary$true_q[i], 0), fmt(summary$selected_q[i], 2),
    fmt(summary$TPR[i]), fmt(summary$FPR[i]), fmt(summary$Precision[i]),
    fmt(summary$F1[i]), fmt(summary$MSE_mu[i]), fmt(summary$MSE_kappa[i]),
    fmt(summary$MSE_centered_eta[i]), fmt(summary$common_false_selection_rate[i]),
    fmt(summary$noise_false_selection_rate[i])
  ))
}

notes <- c(
  notes,
  "",
  "## Notes",
  "",
  "- True q is the posterior decision support size, not the number of common signal coordinates.",
  "- Common coordinates have equal eta values across components and should not be selected as decision coordinates.",
  "- This run is a paper-simulation candidate; check smoke output before treating larger repetitions as final."
)
writeLines(notes, notes_path)

cat("Wrote:\n")
cat("  ", normalizePath(raw_path, winslash = "/"), "\n", sep = "")
cat("  ", normalizePath(summary_path, winslash = "/"), "\n", sep = "")
cat("  ", normalizePath(notes_path, winslash = "/"), "\n", sep = "")
print(summary[, c(
  "method", "reps", "valid_reps", "error_reps", "ARI", "true_q",
  "selected_q", "TPR", "FPR", "Precision", "F1", "MSE_mu",
  "MSE_kappa", "MSE_centered_eta", "common_false_selection_rate",
  "noise_false_selection_rate"
)])
