# ==============================================================================
# Rossi mu-group diagnostic
# ------------------------------------------------------------------------------
# Diagnostic-only ablation baseline. This is NOT the Rossi and Barbaro (2022)
# official sparse vMF baseline. It asks whether coordinate-wise group shrinkage
# on the Rossi direction parameter mu can mimic the proposed centered Eta-group
# behavior in the current K=4 common+specific setting.
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
    '',
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
  run_label = Sys.getenv("ROSSI_MU_GROUP_LABEL", "rossi_mu_group_diagnostic_rep20_w050_260708"),
  n_rep = as.integer(Sys.getenv("ROSSI_MU_GROUP_N_REP", "20")),
  n = as.integer(Sys.getenv("ROSSI_MU_GROUP_N", "1000")),
  d = as.integer(Sys.getenv("ROSSI_MU_GROUP_D", "100")),
  K = as.integer(Sys.getenv("ROSSI_MU_GROUP_K", "4")),
  nstart = as.integer(Sys.getenv("ROSSI_MU_GROUP_NSTART", "5")),
  max_iter = as.integer(Sys.getenv("ROSSI_MU_GROUP_MAX_ITER", "100")),
  path_steps = as.integer(Sys.getenv("ROSSI_MU_GROUP_PATH_STEPS", "80")),
  min_rel_lambda = as.numeric(Sys.getenv("ROSSI_MU_GROUP_MIN_REL_LAMBDA", "1e-3")),
  select_ic = toupper(Sys.getenv("ROSSI_MU_GROUP_SELECT_IC", "BIC")),
  base_seed = as.integer(Sys.getenv("ROSSI_MU_GROUP_BASE_SEED", "20260708")),
  out_dir = Sys.getenv(
    "ROSSI_MU_GROUP_OUT_DIR",
    "results/rossi_mu_group_diagnostic_rep20_w050_260708"
  )
)

if (!cfg$select_ic %in% c("BIC", "EBIC")) {
  stop("ROSSI_MU_GROUP_SELECT_IC must be BIC or EBIC.")
}

common_q <- as.integer(Sys.getenv("ROSSI_MU_GROUP_COMMON_Q", "6"))
specific_q <- as.integer(Sys.getenv("ROSSI_MU_GROUP_SPECIFIC_Q", "4"))
specific_weight <- as.numeric(Sys.getenv("ROSSI_MU_GROUP_WEIGHT", "0.5"))
kappa_vec <- parse_num_grid(Sys.getenv("ROSSI_MU_GROUP_KAPPA", "30,45,65,90"))

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

active_mu_group_diag <- function(theta, zero_eps = 1e-8) {
  sqrt(colSums(theta$mu^2)) > zero_eps
}

mu_group_penalty_value <- function(mu) {
  sum(sqrt(colSums(mu * mu)))
}

prox_mu_group <- function(mu, lambda) {
  norms <- sqrt(colSums(mu * mu))
  scale <- ifelse(norms > 0, pmax(1 - lambda / norms, 0), 0)
  sweep(mu, 2, scale, "*")
}

normalize_rows_masked <- function(mu, active, fallback_mu = NULL, zero_eps = 1e-12) {
  if (any(active)) {
    mu[, !active] <- 0
  }
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

mu_group_ic <- function(theta, n, d, loglik, gamma = 0.5) {
  support_ic(loglik, n, d, nrow(theta$mu), sum(active_mu_group_diag(theta)), gamma = gamma)
}

fit_rossi_mu_group_em <- function(X, K, lambda, init = NULL,
                                  max_iter = 100, tol = 1e-7) {
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
  n_decrease <- 0L
  min_objective_diff <- Inf
  max_line_search_halving <- 0L
  line_search_accepted_all <- TRUE

  for (iter in seq_len(max_iter)) {
    e <- e_step_vmf(X, theta)
    mstep <- unpenalized_eta_mstep(X, e$tau)
    theta_target_raw <- eta_to_theta(mstep$alpha, mstep$eta, fallback_mu = theta$mu)
    mu_target_raw <- prox_mu_group(theta_target_raw$mu, lambda)
    active_target <- sqrt(colSums(mu_target_raw^2)) > 1e-12
    mu_target <- normalize_rows_masked(mu_target_raw, active_target, fallback_mu = theta$mu)
    mu_old <- normalize_rows_masked(theta$mu, active_target, fallback_mu = theta$mu)

    step_size <- 1
    halving <- 0L
    accepted <- FALSE
    theta_new <- NULL
    e_new <- NULL
    obj <- NA_real_
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
      obj_try <- e_try$loglik - lambda * mu_group_penalty_value(theta_try$mu)
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
  final_obj <- last_e$loglik - lambda * mu_group_penalty_value(theta$mu)
  c(theta, list(
    failed = FALSE, converged = FALSE, iter = max_iter,
    loglik = last_e$loglik, pen_loglik = final_obj, tau = last_e$tau,
    objective = final_obj, n_decrease = n_decrease,
    min_objective_diff = ifelse(is.infinite(min_objective_diff), NA_real_, min_objective_diff),
    line_search_halving = max_line_search_halving,
    line_search_accepted = line_search_accepted_all
  ))
}

method_name_diag <- function(refit = FALSE) {
  paste0(
    "Rossi mu-group diagnostic ",
    cfg$select_ic,
    if (refit) " + refit" else ""
  )
}

fit_zero_refit_row <- function(base_row) {
  out <- base_row
  out$method <- method_name_diag(refit = TRUE)
  zero_cols <- c(
    "ARI", "loglik", "pen_loglik", "converged", "iter",
    "MSE_mu", "MSE_kappa", "MSE_centered_eta", "kappa_hat_mean",
    "df", "BIC", "EBIC"
  )
  for (nm in intersect(zero_cols, names(out))) out[[nm]] <- NA_real_
  out$refit_status <- "zero_active_support"
  out
}

add_runtime_cols_diag <- function(row, refit_status = "not_refit") {
  row$refit_status <- refit_status
  row$diagnostic_variant <- "rossi_mu_group"
  row$official_baseline <- 0L
  row
}

add_row_diag <- function(fit, X, z, params, lambda) {
  active <- active_mu_group_diag(fit)
  ic <- mu_group_ic(fit, nrow(X), ncol(X), fit$loglik)
  row <- eval_method(
    method_name_diag(refit = FALSE),
    fit, X, z, params, active, NULL,
    lambda_mu = lambda,
    ic = ic
  )
  row$objective <- ifelse(is.null(fit$objective), fit$pen_loglik, fit$objective)
  row$n_decrease <- ifelse(is.null(fit$n_decrease), NA_integer_, fit$n_decrease)
  row$min_objective_diff <- ifelse(is.null(fit$min_objective_diff), NA_real_, fit$min_objective_diff)
  row$line_search_halving <- ifelse(is.null(fit$line_search_halving), NA_integer_, fit$line_search_halving)
  row$line_search_accepted <- ifelse(is.null(fit$line_search_accepted), NA, fit$line_search_accepted)
  add_runtime_cols_diag(row)
}

fit_rossi_mu_group_path <- function(X, z, params, cfg) {
  dense <- fit_svMF_multistart(
    X, cfg$K, beta = 0, nstart = cfg$nstart, max_iter = cfg$max_iter
  )
  lambda <- 0
  fit <- fit_rossi_mu_group_em(
    X, cfg$K, lambda = lambda, init = dense, max_iter = cfg$max_iter
  )

  rows <- list(add_row_diag(fit, X, z, params, lambda))
  fits <- list(fit)

  for (step in 2:cfg$path_steps) {
    e <- e_step_vmf(X, fit)
    mstep <- unpenalized_eta_mstep(X, e$tau)
    theta_target_raw <- eta_to_theta(mstep$alpha, mstep$eta, fallback_mu = fit$mu)
    thresholds <- sqrt(colSums(theta_target_raw$mu^2))
    candidates <- thresholds[thresholds > lambda + 1e-10]
    if (length(candidates) == 0) break
    lambda_next <- min(candidates)
    if (lambda > 0) lambda_next <- max(lambda_next, lambda * (1 + cfg$min_rel_lambda))
    if (!is.finite(lambda_next) || lambda_next <= lambda) break

    fit_next <- tryCatch(
      fit_rossi_mu_group_em(
        X, cfg$K, lambda = lambda_next, init = fit, max_iter = cfg$max_iter
      ),
      error = function(e) NULL
    )
    if (is.null(fit_next) || isTRUE(fit_next$failed)) break

    fit <- fit_next
    lambda <- lambda_next
    fits[[length(fits) + 1L]] <- fit
    rows[[length(rows) + 1L]] <- add_row_diag(fit, X, z, params, lambda)
    if (sum(active_mu_group_diag(fit)) <= 1) break
  }

  candidates <- do.call(rbind, rows)
  best <- which.min(candidates[[cfg$select_ic]])
  fit_best <- fits[[best]]
  active <- active_mu_group_diag(fit_best)
  out <- candidates[best, , drop = FALSE]
  out$method <- method_name_diag(refit = FALSE)
  out <- add_runtime_cols_diag(out)

  if (!any(active)) {
    out_refit <- fit_zero_refit_row(out)
  } else {
    refit <- fit_support_refit(X, cfg$K, active, fit_best, max_iter = cfg$max_iter)
    out_refit <- eval_method(
      method_name_diag(refit = TRUE),
      refit, X, z, params, active, NULL,
      lambda_mu = out$lambda_mu
    )
    out_refit$objective <- NA_real_
    out_refit$n_decrease <- NA_integer_
    out_refit$min_objective_diff <- NA_real_
    out_refit$line_search_halving <- NA_integer_
    out_refit$line_search_accepted <- NA
    out_refit <- add_runtime_cols_diag(out_refit, refit_status = "support_refit")
  }

  list(rows = rbind(out, out_refit), candidates = candidates)
}

run_one <- function(rep_id, cfg) {
  set.seed(cfg$base_seed + rep_id)
  params <- make_specific_effect_params_diag(
    cfg$d, cfg$K, common_q, specific_q, specific_weight, kappa_vec
  )
  dat <- simulate_from_params_diag(cfg$n, params)
  pairwise_cos <- tcrossprod(params$mu)
  diag(pairwise_cos) <- NA_real_
  cat(sprintf(
    "[strong] rep %d/%d: union_q=%d, kappa=(%s)\n",
    rep_id, cfg$n_rep, sum(colSums(params$support) > 0),
    paste(kappa_vec, collapse = ",")
  ))
  fit <- fit_rossi_mu_group_path(dat$X, dat$z, dat$params, cfg)
  out <- fit$rows
  out$scenario <- "strong_common_specific"
  out$rep <- rep_id
  out$n <- cfg$n
  out$d <- cfg$d
  out$K_true <- cfg$K
  out$common_q <- common_q
  out$specific_q_per_component <- specific_q
  out$specific_weight <- specific_weight
  out$true_union_q <- sum(colSums(params$support) > 0)
  out$true_entry_q <- sum(params$support)
  out$mu_pairwise_cos_mean <- mean(pairwise_cos, na.rm = TRUE)
  out$kappa_true_min <- min(kappa_vec)
  out$kappa_true_max <- max(kappa_vec)
  out$kappa_true_ratio <- max(kappa_vec) / min(kappa_vec)

  candidates <- fit$candidates
  candidates$scenario <- "strong_common_specific"
  candidates$rep <- rep_id
  candidates$n <- cfg$n
  candidates$d <- cfg$d
  candidates$K_true <- cfg$K
  candidates$common_q <- common_q
  candidates$specific_q_per_component <- specific_q
  candidates$specific_weight <- specific_weight
  candidates$true_union_q <- sum(colSums(params$support) > 0)
  candidates$true_entry_q <- sum(params$support)
  candidates$kappa_true_min <- min(kappa_vec)
  candidates$kappa_true_max <- max(kappa_vec)
  candidates$kappa_true_ratio <- max(kappa_vec) / min(kappa_vec)

  list(rows = out, candidates = candidates)
}

cat(sprintf(
  "Running Rossi mu-group diagnostic: reps=%d, n=%d, d=%d, nstart=%d, max_iter=%d, select=%s\n",
  cfg$n_rep, cfg$n, cfg$d, cfg$nstart, cfg$max_iter, cfg$select_ic
))

row_list <- list()
candidate_list <- list()
for (rep_id in seq_len(cfg$n_rep)) {
  res <- tryCatch(
    run_one(rep_id, cfg),
    error = function(e) {
      message(sprintf("[ERROR] rep %d: %s", rep_id, conditionMessage(e)))
      err <- data.frame(
        method = "ERROR", K_fit = NA_real_, beta = NA_real_,
        lambda_mu = NA_real_, lambda_kappa = NA_real_,
        lambda_eta = NA_real_, ARI = NA_real_, loglik = NA_real_,
        pen_loglik = NA_real_, converged = NA, iter = NA_real_,
        true_union_q = NA_real_, selected_q = NA_real_,
        TPR = NA_real_, FPR = NA_real_, Precision = NA_real_,
        F1 = NA_real_, entry_TPR = NA_real_, entry_FPR = NA_real_,
        entry_Precision = NA_real_, entry_F1 = NA_real_,
        MSE_mu = NA_real_, MSE_kappa = NA_real_,
        MSE_centered_eta = NA_real_, kappa_hat_mean = NA_real_,
        df = NA_real_, BIC = NA_real_, EBIC = NA_real_,
        objective = NA_real_, n_decrease = NA_real_,
        min_objective_diff = NA_real_, line_search_halving = NA_real_,
        line_search_accepted = NA,
        refit_status = conditionMessage(e),
        diagnostic_variant = "rossi_mu_group",
        official_baseline = 0L,
        scenario = "strong_common_specific", rep = rep_id,
        n = cfg$n, d = cfg$d, K_true = cfg$K,
        common_q = common_q, specific_q_per_component = specific_q,
        specific_weight = specific_weight, true_entry_q = NA_real_,
        mu_pairwise_cos_mean = NA_real_, kappa_true_min = min(kappa_vec),
        kappa_true_max = max(kappa_vec),
        kappa_true_ratio = max(kappa_vec) / min(kappa_vec)
      )
      list(rows = err, candidates = NULL)
    }
  )
  row_list[[rep_id]] <- res$rows
  candidate_list[[rep_id]] <- res$candidates
}

raw <- do.call(rbind, row_list)
candidates <- do.call(rbind, candidate_list[!vapply(candidate_list, is.null, logical(1))])

safe_mean <- function(x) {
  if (sum(!is.na(x)) == 0) NA_real_ else mean(x, na.rm = TRUE)
}

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
    zero_support_refit_reps = sum(sub$refit_status == "zero_active_support", na.rm = TRUE),
    means,
    row.names = NULL
  )
}))

method_order <- c(method_name_diag(FALSE), method_name_diag(TRUE), "ERROR")
summary$method <- factor(summary$method, levels = method_order)
summary <- summary[order(summary$scenario, summary$method), ]
summary$method <- as.character(summary$method)

raw_path <- file.path(cfg$out_dir, sprintf("%s_raw.csv", cfg$run_label))
candidate_path <- file.path(cfg$out_dir, sprintf("%s_path_candidates.csv", cfg$run_label))
summary_path <- file.path(cfg$out_dir, sprintf("%s_summary.csv", cfg$run_label))
report_path <- file.path(cfg$out_dir, sprintf("%s_report.md", cfg$run_label))
write.csv(raw, raw_path, row.names = FALSE)
write.csv(candidates, candidate_path, row.names = FALSE)
write.csv(summary, summary_path, row.names = FALSE)

fmt <- function(x, digits = 3) {
  ifelse(is.na(x), "NA", formatC(as.numeric(x), digits = digits, format = "f"))
}

report_lines <- c(
  "# Rossi mu-group diagnostic",
  "",
  sprintf("- Date: %s", Sys.Date()),
  "- Status: ablation diagnostic only; not Rossi 2022 official baseline.",
  sprintf("- Setting: K=%d, n=%d, d=%d, common q=%d, component-specific q=%d each, true union q=%d.",
          cfg$K, cfg$n, cfg$d, common_q, specific_q, common_q + cfg$K * specific_q),
  sprintf("- Kappa: (%s), specific weight: %.3f.", paste(kappa_vec, collapse = ", "), specific_weight),
  sprintf("- Repetitions: %d.", cfg$n_rep),
  sprintf("- Tuning: mu-space group path + %s.", cfg$select_ic),
  "",
  "## Summary",
  "",
  "| method | reps | valid_reps | zero_refit | ARI | selected_q | TPR | FPR | Precision | F1 | MSE_mu | MSE_kappa | MSE_centered_eta | lambda_mu |",
  "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
)

for (i in seq_len(nrow(summary))) {
  report_lines <- c(report_lines, sprintf(
    "| %s | %d | %d | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |",
    summary$method[i], summary$reps[i], summary$valid_reps[i],
    summary$zero_support_refit_reps[i], fmt(summary$ARI[i]), fmt(summary$selected_q[i], 2),
    fmt(summary$TPR[i]), fmt(summary$FPR[i]), fmt(summary$Precision[i]),
    fmt(summary$F1[i]), fmt(summary$MSE_mu[i]), fmt(summary$MSE_kappa[i]),
    fmt(summary$MSE_centered_eta[i]), fmt(summary$lambda_mu[i])
  ))
}

report_lines <- c(
  report_lines,
  "",
  "## Notes",
  "",
  "- This variant applies coordinate-wise group shrinkage to the Rossi direction matrix mu.",
  "- It is a diagnostic approximation for separating mu-space group-penalty effects from eta-centered contrast effects.",
  "- Unit-norm rows are restored after shrinkage while preserving the selected coordinate mask.",
  "- It should not be described as Rossi and Barbaro (2022) official baseline."
)

writeLines(report_lines, report_path)

cat("Wrote:\n")
cat("  ", normalizePath(raw_path, winslash = "/"), "\n", sep = "")
cat("  ", normalizePath(candidate_path, winslash = "/"), "\n", sep = "")
cat("  ", normalizePath(summary_path, winslash = "/"), "\n", sep = "")
cat("  ", normalizePath(report_path, winslash = "/"), "\n", sep = "")
print(summary[, c(
  "scenario", "method", "reps", "valid_reps", "zero_support_refit_reps",
  "ARI", "selected_q", "TPR", "FPR", "Precision", "F1",
  "MSE_mu", "MSE_kappa", "MSE_centered_eta", "lambda_mu",
  "n_decrease", "min_objective_diff", "line_search_halving"
)])
