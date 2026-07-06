# ==============================================================================
# Centering/parameterization ablation diagnostic
# ------------------------------------------------------------------------------
# Diagnostic-only runner for separating three effects:
#   1. mu-space vs eta-space parameterization,
#   2. raw vs centered contrast,
#   3. entry-wise vs coordinate group penalty.
#
# This file does not change the official simulation runners.
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
  lines <- gsub(
    'source(file.path("r", "rossi_barbaro_2022_reproduction.r"))',
    "",
    lines,
    fixed = TRUE
  )
  idx <- grep("fit_rossi_pair <-", lines, fixed = TRUE)[1]
  if (is.na(idx)) stop("Could not find helper boundary.")
  eval(parse(text = lines[seq_len(idx - 1L)]), envir = .GlobalEnv)
}

source_k4_helpers()

parse_num_grid <- function(x) {
  as.numeric(strsplit(x, ",", fixed = TRUE)[[1]])
}

cfg <- list(
  run_label = Sys.getenv("ETA_CENTER_ABL_LABEL", "eta_centering_ablation_all_models_pilot5_260708"),
  n_rep = as.integer(Sys.getenv("ETA_CENTER_ABL_N_REP", "5")),
  n = as.integer(Sys.getenv("ETA_CENTER_ABL_N", "1000")),
  d = as.integer(Sys.getenv("ETA_CENTER_ABL_D", "100")),
  K = as.integer(Sys.getenv("ETA_CENTER_ABL_K", "4")),
  nstart = as.integer(Sys.getenv("ETA_CENTER_ABL_NSTART", "5")),
  max_iter = as.integer(Sys.getenv("ETA_CENTER_ABL_MAX_ITER", "80")),
  path_steps = as.integer(Sys.getenv("ETA_CENTER_ABL_PATH_STEPS", "50")),
  min_rel_lambda = as.numeric(Sys.getenv("ETA_CENTER_ABL_MIN_REL_LAMBDA", "1e-3")),
  select_ic = toupper(Sys.getenv("ETA_CENTER_ABL_SELECT_IC", "BIC")),
  base_seed = as.integer(Sys.getenv("ETA_CENTER_ABL_BASE_SEED", "20260708")),
  adaptive_gamma = as.numeric(Sys.getenv("ETA_CENTER_ABL_ADAPTIVE_GAMMA", "1")),
  adaptive_eps = as.numeric(Sys.getenv("ETA_CENTER_ABL_ADAPTIVE_EPS", "1e-6")),
  out_dir = Sys.getenv(
    "ETA_CENTER_ABL_OUT_DIR",
    "results/eta_centering_ablation_all_models_pilot5_260708"
  )
)

if (!cfg$select_ic %in% c("BIC", "EBIC")) {
  stop("ETA_CENTER_ABL_SELECT_IC must be BIC or EBIC.")
}

common_q <- as.integer(Sys.getenv("ETA_CENTER_ABL_COMMON_Q", "6"))
specific_q <- as.integer(Sys.getenv("ETA_CENTER_ABL_SPECIFIC_Q", "4"))
specific_weight <- as.numeric(Sys.getenv("ETA_CENTER_ABL_WEIGHT", "0.5"))
kappa_vec <- parse_num_grid(Sys.getenv("ETA_CENTER_ABL_KAPPA", "30,45,65,90"))

dir.create(cfg$out_dir, recursive = TRUE, showWarnings = FALSE)

make_specific_effect_params_diag <- function(d, K, common_q, specific_q,
                                             specific_weight, kappa) {
  if (length(kappa) != K) stop("length(kappa) must match K.")
  if (common_q + K * specific_q > d) {
    stop("common_q + K * specific_q must be <= d.")
  }

  mu_raw <- matrix(0, nrow = K, ncol = d)
  support <- matrix(FALSE, nrow = K, ncol = d)
  common_idx <- seq_len(common_q)
  mu_raw[, common_idx] <- 1.0
  support[, common_idx] <- TRUE

  start <- common_q + 1L
  specific_index <- vector("list", K)
  for (k in seq_len(K)) {
    idx <- start + ((k - 1L) * specific_q) + seq_len(specific_q) - 1L
    specific_index[[k]] <- idx
    mu_raw[k, idx] <- specific_weight
    support[k, idx] <- TRUE
  }

  list(
    alpha = rep(1 / K, K),
    mu = normalize_rows(mu_raw),
    kappa = kappa,
    support = support,
    common_idx = common_idx,
    specific_idx = unlist(specific_index, use.names = FALSE),
    noise_idx = seq.int(common_q + K * specific_q + 1L, d)
  )
}

simulate_from_params_diag <- function(n, params) {
  K <- nrow(params$mu)
  z <- sample.int(K, size = n, replace = TRUE, prob = params$alpha)
  X <- matrix(0, nrow = n, ncol = ncol(params$mu))
  for (k in seq_len(K)) {
    idx <- which(z == k)
    if (length(idx) > 0) {
      X[idx, ] <- rvMF(length(idx), params$mu[k, ], params$kappa[k])
    }
  }
  list(X = X, z = z, params = params)
}

soft_threshold <- function(x, lambda) {
  sign(x) * pmax(abs(x) - lambda, 0)
}

center_cols <- function(mat) {
  sweep(mat, 2, colMeans(mat), "-")
}

prox_raw_group <- function(mat, lambda_vec) {
  if (length(lambda_vec) == 1L) lambda_vec <- rep(lambda_vec, ncol(mat))
  norms <- sqrt(colSums(mat * mat))
  scale <- ifelse(norms > 0, pmax(1 - lambda_vec / norms, 0), 0)
  sweep(mat, 2, scale, "*")
}

prox_centered_group_mat <- function(mat, lambda_vec) {
  if (length(lambda_vec) == 1L) lambda_vec <- rep(lambda_vec, ncol(mat))
  means <- colMeans(mat)
  centered <- sweep(mat, 2, means, "-")
  shrunk <- prox_raw_group(centered, lambda_vec)
  sweep(shrunk, 2, means, "+")
}

prox_centered_l1_col <- function(y, lambda) {
  mu <- mean(y)
  z <- y - mu
  f <- function(tau) sum(soft_threshold(z - tau, lambda))
  if (abs(f(0)) < 1e-10) {
    return(mu + soft_threshold(z, lambda))
  }
  lower <- min(z) - lambda - 10
  upper <- max(z) + lambda + 10
  tau <- uniroot(f, lower = lower, upper = upper, tol = 1e-10)$root
  mu + soft_threshold(z - tau, lambda)
}

prox_centered_l1 <- function(mat, lambda) {
  apply(mat, 2, prox_centered_l1_col, lambda = lambda)
}

normalize_rows_masked <- function(mu, active = rep(TRUE, ncol(mu)),
                                  fallback_mu = NULL, zero_eps = 1e-12) {
  if (any(!active)) mu[, !active] <- 0
  for (k in seq_len(nrow(mu))) {
    nr <- l2_norm(mu[k, ])
    if (nr > zero_eps) {
      mu[k, ] <- mu[k, ] / nr
    } else if (!is.null(fallback_mu) && any(active)) {
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

variant_specs <- list(
  list(method = "M-L", family = "mu", penalty = "raw_l1", adaptive = FALSE),
  list(method = "M-GL", family = "mu", penalty = "raw_group", adaptive = FALSE),
  list(method = "M-AGL", family = "mu", penalty = "raw_group", adaptive = TRUE),
  list(method = "M-CGL", family = "mu", penalty = "centered_group", adaptive = FALSE),
  list(method = "E-L", family = "eta", penalty = "raw_l1", adaptive = FALSE),
  list(method = "E-GL", family = "eta", penalty = "raw_group", adaptive = FALSE),
  list(method = "E-CL", family = "eta", penalty = "centered_l1", adaptive = FALSE),
  list(method = "E-CGL", family = "eta", penalty = "centered_group", adaptive = FALSE),
  list(method = "E-CAGL", family = "eta", penalty = "centered_group", adaptive = TRUE)
)

variant_name <- function(spec, refit = FALSE) {
  paste0(spec$method, " ", cfg$select_ic, if (refit) " + refit" else "")
}

feature_matrix <- function(theta, spec) {
  eta <- eta_matrix(theta)
  if (spec$family == "mu") {
    mat <- theta$mu
  } else {
    mat <- eta
  }
  if (spec$penalty %in% c("centered_l1", "centered_group")) {
    mat <- center_cols(mat)
  }
  mat
}

active_variant <- function(theta, spec, zero_eps = 1e-8) {
  mat <- feature_matrix(theta, spec)
  if (spec$penalty %in% c("raw_l1", "centered_l1")) {
    colSums(abs(mat) > zero_eps) > 0
  } else {
    sqrt(colSums(mat * mat)) > zero_eps
  }
}

entry_count_variant <- function(theta, spec, zero_eps = 1e-8) {
  mat <- feature_matrix(theta, spec)
  sum(abs(mat) > zero_eps)
}

penalty_value_variant <- function(theta, spec, weights = NULL) {
  mat <- feature_matrix(theta, spec)
  if (spec$penalty %in% c("raw_l1", "centered_l1")) {
    sum(abs(mat))
  } else {
    norms <- sqrt(colSums(mat * mat))
    if (!is.null(weights)) norms <- weights * norms
    sum(norms)
  }
}

ic_variant <- function(theta, spec, n, d, loglik, gamma = 0.5) {
  K <- nrow(theta$mu)
  if (spec$penalty %in% c("raw_l1", "centered_l1")) {
    nnz <- entry_count_variant(theta, spec)
    df <- (K - 1) + K + max(nnz - K, 1)
  } else if (spec$family == "eta" && spec$penalty == "centered_group") {
    df <- eta_centered_df(theta)
  } else {
    df <- support_df(K, sum(active_variant(theta, spec)))
  }
  data.frame(
    df = df,
    BIC = log(n) * df - 2 * loglik,
    EBIC = (log(n) + 2 * gamma * log(d)) * df - 2 * loglik
  )
}

adaptive_weights_variant <- function(dense, spec) {
  if (!isTRUE(spec$adaptive)) return(NULL)
  mat <- feature_matrix(dense, spec)
  norms <- sqrt(colSums(mat * mat))
  w <- (norms + cfg$adaptive_eps)^(-cfg$adaptive_gamma)
  med <- median(w[is.finite(w) & w > 0])
  if (is.finite(med) && med > 0) w <- w / med
  w
}

lambda_upper_from_target <- function(target_eta, theta_target, spec, weights = NULL) {
  if (spec$family == "mu") {
    mat <- theta_target$mu
  } else {
    mat <- target_eta
  }
  if (spec$penalty %in% c("centered_l1", "centered_group")) {
    mat <- center_cols(mat)
  }
  if (spec$penalty %in% c("raw_l1", "centered_l1")) {
    val <- max(abs(mat), na.rm = TRUE)
  } else {
    norms <- sqrt(colSums(mat * mat))
    if (!is.null(weights)) {
      idx <- which(weights > 0)
      val <- if (length(idx) == 0) 0 else max(norms[idx] / weights[idx], na.rm = TRUE)
    } else {
      val <- max(norms, na.rm = TRUE)
    }
  }
  if (!is.finite(val) || val <= 0) val <- 1
  val
}

apply_penalty_variant <- function(alpha, target_eta, current_theta, spec,
                                  lambda, weights = NULL) {
  theta_target <- eta_to_theta(alpha, target_eta, fallback_mu = current_theta$mu)
  lam <- lambda
  if (spec$adaptive && !is.null(weights) &&
      spec$penalty %in% c("raw_group", "centered_group")) {
    lam <- lambda * weights
  }

  if (spec$family == "mu") {
    mu_raw <- theta_target$mu
    mu_new <- switch(
      spec$penalty,
      raw_l1 = soft_threshold(mu_raw, lambda),
      raw_group = prox_raw_group(mu_raw, lam),
      centered_group = prox_centered_group_mat(mu_raw, lam),
      stop("Unsupported mu penalty.")
    )
    active <- if (spec$penalty == "centered_group") {
      sqrt(colSums(center_cols(mu_new)^2)) > 1e-12
    } else {
      sqrt(colSums(mu_new^2)) > 1e-12
    }
    mu_new <- normalize_rows_masked(mu_new, active, fallback_mu = current_theta$mu)
    list(
      alpha = alpha,
      mu = mu_new,
      kappa = pmax(theta_target$kappa, 1e-10)
    )
  } else {
    eta_new <- switch(
      spec$penalty,
      raw_l1 = soft_threshold(target_eta, lambda),
      raw_group = prox_raw_group(target_eta, lam),
      centered_l1 = prox_centered_l1(target_eta, lambda),
      centered_group = prox_eta_centered(target_eta, lambda, if (spec$adaptive) weights else NULL),
      stop("Unsupported eta penalty.")
    )
    eta_to_theta(alpha, eta_new, fallback_mu = current_theta$mu)
  }
}

fit_variant_em <- function(X, K, lambda, init, spec, weights = NULL,
                           max_iter = 80, tol = 1e-7) {
  theta <- init
  theta$mu <- normalize_rows(theta$mu)
  theta$alpha <- pmax(theta$alpha, 1e-12)
  theta$alpha <- theta$alpha / sum(theta$alpha)
  if (length(theta$kappa) == 1) theta$kappa <- rep(theta$kappa, K)

  prev_obj <- -Inf
  last_e <- NULL
  n_decrease <- 0L
  min_objective_diff <- Inf
  max_line_search_halving <- 0L
  line_search_accepted_all <- TRUE

  for (iter in seq_len(max_iter)) {
    e <- e_step_vmf(X, theta)
    mstep <- unpenalized_eta_mstep(X, e$tau)
    target <- apply_penalty_variant(
      mstep$alpha, mstep$eta, theta, spec, lambda, weights
    )

    step_size <- 1
    halving <- 0L
    accepted <- FALSE
    theta_new <- NULL
    e_new <- NULL
    obj <- NA_real_

    repeat {
      alpha_new <- theta$alpha + step_size * (target$alpha - theta$alpha)
      kappa_new <- theta$kappa + step_size * (target$kappa - theta$kappa)
      mu_new <- theta$mu + step_size * (target$mu - theta$mu)
      mu_new <- normalize_rows(mu_new)
      theta_try <- list(
        alpha = pmax(alpha_new, 1e-12) / sum(pmax(alpha_new, 1e-12)),
        mu = mu_new,
        kappa = pmax(kappa_new, 1e-10)
      )
      e_try <- e_step_vmf(X, theta_try)
      pen_value <- penalty_value_variant(theta_try, spec, weights)
      obj_try <- e_try$loglik - lambda * pen_value
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
    if (is.finite(prev_obj) &&
        abs(obj - prev_obj) / max(1, abs(prev_obj)) < tol) {
      return(c(theta, list(
        failed = FALSE, converged = TRUE, iter = iter,
        loglik = e_new$loglik, pen_loglik = obj, tau = e_new$tau,
        objective = obj, n_decrease = n_decrease,
        min_objective_diff = ifelse(is.infinite(min_objective_diff), NA_real_, min_objective_diff),
        line_search_halving = max_line_search_halving,
        line_search_accepted = line_search_accepted_all
      )))
    }
    prev_obj <- obj
  }

  if (is.null(last_e)) last_e <- e_step_vmf(X, theta)
  final_obj <- last_e$loglik - lambda * penalty_value_variant(theta, spec, weights)
  c(theta, list(
    failed = FALSE, converged = FALSE, iter = max_iter,
    loglik = last_e$loglik, pen_loglik = final_obj, tau = last_e$tau,
    objective = final_obj, n_decrease = n_decrease,
    min_objective_diff = ifelse(is.infinite(min_objective_diff), NA_real_, min_objective_diff),
    line_search_halving = max_line_search_halving,
    line_search_accepted = line_search_accepted_all
  ))
}

selection_type_metrics <- function(active, params) {
  data.frame(
    common_selection_rate = mean(active[params$common_idx]),
    specific_selection_rate = mean(active[params$specific_idx]),
    noise_selection_rate = mean(active[params$noise_idx])
  )
}

append_type_metrics <- function(row, active, params) {
  cbind(row, selection_type_metrics(active, params))
}

fit_zero_refit_row <- function(method, base_row, active) {
  out <- base_row
  out$method <- method
  out$ARI <- NA_real_
  out$loglik <- NA_real_
  out$pen_loglik <- NA_real_
  out$converged <- NA
  out$iter <- NA_real_
  out$selected_q <- sum(active)
  out$TPR <- NA_real_
  out$FPR <- NA_real_
  out$Precision <- NA_real_
  out$F1 <- NA_real_
  out$entry_TPR <- NA_real_
  out$entry_FPR <- NA_real_
  out$entry_Precision <- NA_real_
  out$entry_F1 <- NA_real_
  out$MSE_mu <- NA_real_
  out$MSE_kappa <- NA_real_
  out$MSE_centered_eta <- NA_real_
  out$kappa_hat_mean <- NA_real_
  out$df <- NA_real_
  out$BIC <- NA_real_
  out$EBIC <- NA_real_
  out$refit_status <- "zero_active_support"
  out
}

add_runtime_cols <- function(row, fit, spec, lambda, weights) {
  if (!"refit_status" %in% names(row)) row$refit_status <- "not_refit"
  row$penalty_family <- spec$family
  row$penalty_type <- spec$penalty
  row$adaptive <- isTRUE(spec$adaptive)
  row$objective <- if (!is.null(fit$objective)) fit$objective else NA_real_
  row$n_decrease <- if (!is.null(fit$n_decrease)) fit$n_decrease else NA_integer_
  row$min_objective_diff <- if (!is.null(fit$min_objective_diff)) fit$min_objective_diff else NA_real_
  row$line_search_halving <- if (!is.null(fit$line_search_halving)) fit$line_search_halving else NA_integer_
  row$line_search_accepted <- if (!is.null(fit$line_search_accepted)) fit$line_search_accepted else NA
  row$lambda <- lambda
  row$weight_min <- if (is.null(weights)) NA_real_ else min(weights)
  row$weight_max <- if (is.null(weights)) NA_real_ else max(weights)
  row
}

fit_variant_path_pair <- function(X, z, params, cfg, spec) {
  dense <- fit_svMF_multistart(
    X, cfg$K, beta = 0, nstart = cfg$nstart, max_iter = cfg$max_iter
  )
  weights <- adaptive_weights_variant(dense, spec)
  e_dense <- e_step_vmf(X, dense)
  mstep0 <- unpenalized_eta_mstep(X, e_dense$tau)
  theta_target0 <- eta_to_theta(mstep0$alpha, mstep0$eta, fallback_mu = dense$mu)
  lambda_max <- lambda_upper_from_target(mstep0$eta, theta_target0, spec, weights)
  lambda_grid <- unique(c(
    0,
    exp(seq(
      log(max(lambda_max * cfg$min_rel_lambda, 1e-12)),
      log(lambda_max * 1.05),
      length.out = max(2L, cfg$path_steps - 1L)
    ))
  ))

  fits <- list()
  rows <- list()
  path_rows <- list()
  fit <- dense

  for (i in seq_along(lambda_grid)) {
    lambda <- lambda_grid[i]
    fit_next <- tryCatch(
      fit_variant_em(
        X, cfg$K, lambda, init = fit, spec = spec, weights = weights,
        max_iter = cfg$max_iter
      ),
      error = function(e) NULL
    )
    if (is.null(fit_next) || isTRUE(fit_next$failed)) next
    fit <- fit_next
    fits[[length(fits) + 1L]] <- fit
    active <- active_variant(fit, spec)
    ic <- ic_variant(fit, spec, nrow(X), ncol(X), fit$loglik)
    row <- eval_method(
      variant_name(spec, refit = FALSE),
      fit, X, z, params, active, NULL,
      lambda_eta = if (spec$family == "eta") lambda else NA_real_,
      ic = ic
    )
    row$lambda_mu <- if (spec$family == "mu") lambda else NA_real_
    row <- add_runtime_cols(row, fit, spec, lambda, weights)
    rows[[length(rows) + 1L]] <- row
    path_rows[[length(path_rows) + 1L]] <- row
  }

  if (length(rows) == 0) stop(sprintf("No path rows for %s.", spec$method))
  tab <- do.call(rbind, rows)
  best <- which.min(tab[[cfg$select_ic]])
  fit_best <- fits[[best]]
  active <- active_variant(fit_best, spec)
  out <- tab[best, , drop = FALSE]

  refit_method <- variant_name(spec, refit = TRUE)
  if (!any(active)) {
    out_refit <- fit_zero_refit_row(refit_method, out, active)
  } else {
    refit <- fit_support_refit(X, cfg$K, active, fit_best, max_iter = cfg$max_iter)
    out_refit <- eval_method(
      refit_method, refit, X, z, params, active, NULL,
      lambda_eta = out$lambda_eta
    )
    out_refit$lambda_mu <- out$lambda_mu
    out_refit <- add_runtime_cols(out_refit, refit, spec, out$lambda, weights)
    out_refit$refit_status <- "support_refit"
  }

  list(
    rows = rbind(
      append_type_metrics(out, active, params),
      append_type_metrics(out_refit, active, params)
    ),
    path = do.call(rbind, path_rows)
  )
}

run_one <- function(rep_id, cfg) {
  set.seed(cfg$base_seed + rep_id)
  params <- make_specific_effect_params_diag(
    d = cfg$d, K = cfg$K, common_q = common_q,
    specific_q = specific_q, specific_weight = specific_weight,
    kappa = kappa_vec
  )
  dat <- simulate_from_params_diag(cfg$n, params)
  cat(sprintf(
    "[ablation] rep %d/%d: true_union_q=%d, K=%d, n=%d, d=%d\n",
    rep_id, cfg$n_rep, sum(colSums(params$support) > 0), cfg$K, cfg$n, cfg$d
  ))

  rows <- list()
  paths <- list()
  for (spec in variant_specs) {
    cat(sprintf("  - %s\n", spec$method))
    fit <- fit_variant_path_pair(dat$X, dat$z, dat$params, cfg, spec)
    rows[[length(rows) + 1L]] <- fit$rows
    paths[[length(paths) + 1L]] <- fit$path
  }

  out <- do.call(rbind, rows)
  path <- do.call(rbind, paths)
  for (obj_name in c("out", "path")) {
    obj <- get(obj_name)
    obj$scenario <- "strong_common_specific"
    obj$rep <- rep_id
    obj$n <- cfg$n
    obj$d <- cfg$d
    obj$K_true <- cfg$K
    obj$common_q <- common_q
    obj$specific_q_per_component <- specific_q
    obj$specific_weight <- specific_weight
    obj$true_entry_q <- sum(dat$params$support)
    obj$true_union_q <- sum(colSums(dat$params$support) > 0)
    obj$kappa_true_min <- min(kappa_vec)
    obj$kappa_true_max <- max(kappa_vec)
    obj$kappa_true_ratio <- max(kappa_vec) / min(kappa_vec)
    assign(obj_name, obj)
  }
  list(rows = out, path = path)
}

all_rows <- list()
all_paths <- list()
for (rep_id in seq_len(cfg$n_rep)) {
  res <- tryCatch(
    run_one(rep_id, cfg),
    error = function(e) {
      data.frame(
        method = "ERROR", K_fit = NA_real_, beta = NA_real_,
        lambda_mu = NA_real_, lambda_kappa = NA_real_, lambda_eta = NA_real_,
        ARI = NA_real_, loglik = NA_real_, pen_loglik = NA_real_,
        converged = NA, iter = NA_real_, true_union_q = NA_real_,
        selected_q = NA_real_, TPR = NA_real_, FPR = NA_real_,
        Precision = NA_real_, F1 = NA_real_, entry_TPR = NA_real_,
        entry_FPR = NA_real_, entry_Precision = NA_real_, entry_F1 = NA_real_,
        MSE_mu = NA_real_, MSE_kappa = NA_real_, MSE_centered_eta = NA_real_,
        kappa_hat_mean = NA_real_, df = NA_real_, BIC = NA_real_, EBIC = NA_real_,
        penalty_family = NA_character_, penalty_type = conditionMessage(e),
        adaptive = NA, objective = NA_real_, n_decrease = NA_integer_,
        min_objective_diff = NA_real_, line_search_halving = NA_integer_,
        line_search_accepted = NA, lambda = NA_real_, weight_min = NA_real_,
        weight_max = NA_real_, common_selection_rate = NA_real_,
        specific_selection_rate = NA_real_, noise_selection_rate = NA_real_,
        scenario = "ERROR", rep = rep_id, n = cfg$n, d = cfg$d,
        K_true = cfg$K, common_q = common_q,
        specific_q_per_component = specific_q, specific_weight = specific_weight,
        true_entry_q = NA_real_, kappa_true_min = min(kappa_vec),
        kappa_true_max = max(kappa_vec), kappa_true_ratio = max(kappa_vec) / min(kappa_vec)
      )
    }
  )
  if (is.list(res) && !is.data.frame(res)) {
    all_rows[[length(all_rows) + 1L]] <- res$rows
    all_paths[[length(all_paths) + 1L]] <- res$path
  } else {
    all_rows[[length(all_rows) + 1L]] <- res
  }
}

raw <- do.call(rbind, all_rows)
paths <- if (length(all_paths) > 0) do.call(rbind, all_paths) else data.frame()

safe_mean <- function(x) {
  if (all(is.na(x))) return(NA_real_)
  mean(x, na.rm = TRUE)
}

num_cols <- names(raw)[vapply(raw, is.numeric, logical(1))]
groups <- unique(raw[, c("scenario", "method")])
summary <- do.call(rbind, lapply(seq_len(nrow(groups)), function(i) {
  sub <- raw[raw$scenario == groups$scenario[i] & raw$method == groups$method[i], ]
  means <- as.data.frame(lapply(sub[, num_cols, drop = FALSE], safe_mean))
  data.frame(
    scenario = groups$scenario[i],
    method = groups$method[i],
    reps = nrow(sub),
    valid_reps = sum(sub$method != "ERROR"),
    zero_support_reps = sum(sub$selected_q == 0, na.rm = TRUE),
    means,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}))

method_order <- unlist(lapply(variant_specs, function(spec) {
  c(variant_name(spec, FALSE), variant_name(spec, TRUE))
}), use.names = FALSE)
summary$method <- factor(summary$method, levels = c(method_order, "ERROR"))
summary <- summary[order(summary$scenario, summary$method), ]
summary$method <- as.character(summary$method)

raw_path <- file.path(cfg$out_dir, sprintf("%s_raw.csv", cfg$run_label))
path_path <- file.path(cfg$out_dir, sprintf("%s_path_candidates.csv", cfg$run_label))
summary_path <- file.path(cfg$out_dir, sprintf("%s_summary.csv", cfg$run_label))
report_path <- file.path(cfg$out_dir, sprintf("%s_report.md", cfg$run_label))

write.csv(raw, raw_path, row.names = FALSE)
write.csv(paths, path_path, row.names = FALSE)
write.csv(summary, summary_path, row.names = FALSE)

fmt <- function(x, digits = 3) {
  ifelse(is.na(x), "NA", formatC(x, format = "f", digits = digits))
}

report <- c(
  "# Eta centering ablation all-model diagnostic",
  "",
  "Diagnostic-only run. These rows separate mu/eta parameterization, raw/centered contrast, and entry-wise/group penalties.",
  "",
  sprintf("- reps: %d", cfg$n_rep),
  sprintf("- K=%d, n=%d, d=%d", cfg$K, cfg$n, cfg$d),
  sprintf("- common q=%d, specific q/component=%d, true union q=%d",
          common_q, specific_q, common_q + cfg$K * specific_q),
  sprintf("- kappa=(%s)", paste(kappa_vec, collapse = ",")),
  "",
  "| method | reps | valid | selected q | ARI | TPR | FPR | Precision | F1 | MSE_mu | MSE_kappa | MSE_eta | common q rate | specific q rate | noise q rate |",
  "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
)

for (i in seq_len(nrow(summary))) {
  report <- c(report, sprintf(
    "| %s | %d | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |",
    summary$method[i], summary$reps[i], summary$valid_reps[i],
    fmt(summary$selected_q[i], 2), fmt(summary$ARI[i]),
    fmt(summary$TPR[i]), fmt(summary$FPR[i]), fmt(summary$Precision[i]),
    fmt(summary$F1[i]), fmt(summary$MSE_mu[i]), fmt(summary$MSE_kappa[i]),
    fmt(summary$MSE_centered_eta[i]),
    fmt(summary$common_selection_rate[i]), fmt(summary$specific_selection_rate[i]),
    fmt(summary$noise_selection_rate[i])
  ))
}

writeLines(report, report_path, useBytes = TRUE)

cat("\nWrote:\n")
cat("  ", normalizePath(summary_path, winslash = "/"), "\n", sep = "")
cat("  ", normalizePath(report_path, winslash = "/"), "\n", sep = "")
print(summary[, c(
  "scenario", "method", "reps", "valid_reps", "selected_q", "ARI",
  "TPR", "FPR", "Precision", "F1", "MSE_mu", "MSE_kappa",
  "MSE_centered_eta", "common_selection_rate", "specific_selection_rate",
  "noise_selection_rate"
)])
