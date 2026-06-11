# ==============================================================================
# Robustness grid for the kappa-contrast limitation scenario
# ------------------------------------------------------------------------------
# Compares four methods across concentration ratios and directional overlap:
#
#   1. Rossi & Barbaro (2022) sparse vMF path, BIC
#   2. Separate mu/kappa penalty EM, BIC + support refit
#   3. Eta penalty EM, BIC + support refit
#   4. Eta contrast screening, BIC + support refit
#
# The goal is to check whether the eta-based interpretation remains stable
# beyond the single extreme case mu_1 = mu_2, kappa_2 / kappa_1 = 10.
# ==============================================================================

source(file.path("r", "rossi_barbaro_2022_reproduction.r"))

source_until_marker <- function(path, marker = "Simulation driver") {
  lines <- readLines(path, warn = FALSE)
  idx <- grep(marker, lines, fixed = TRUE)[1]
  if (is.na(idx) || idx <= 1) {
    stop("Could not find marker in ", path)
  }
  eval(parse(text = paste(lines[seq_len(idx - 1)], collapse = "\n")),
       envir = .GlobalEnv)
}

# Reuse the already-checked implementations without running their simulations.
source_until_marker(file.path("r", "separate_penalty_vmf_run.r"))
sep_fit_grid <- fit_separate_penalty_grid
sep_support_refit <- fit_support_constrained_vmf

source_until_marker(file.path("r", "eta_penalty_vmf_run.r"))
eta_fit_grid <- fit_eta_penalty_grid
eta_penalty_active <- eta_contrast_active
eta_support_refit <- fit_support_constrained_vmf

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

parse_num_grid <- function(x) as.numeric(strsplit(x, ",")[[1]])
parse_int_grid <- function(x) as.integer(strsplit(x, ",")[[1]])

rob_cfg <- list(
  run_label = Sys.getenv("ROB_RUN_LABEL", "robustness_kappa_mu_grid"),
  n_rep = as.integer(Sys.getenv("ROB_N_REP", "20")),
  n = as.integer(Sys.getenv("ROB_N", "1000")),
  d = as.integer(Sys.getenv("ROB_D", "100")),
  q = as.integer(Sys.getenv("ROB_Q", "10")),
  kappa_low = as.numeric(Sys.getenv("ROB_KAPPA_LOW", "20")),
  kappa_ratio_grid = parse_num_grid(Sys.getenv("ROB_KAPPA_RATIO_GRID", "2,5,10")),
  mu_cos_grid = parse_num_grid(Sys.getenv("ROB_MU_COS_GRID", "1,0.99,0.95")),
  K_fit = 2,
  nstart = as.integer(Sys.getenv("ROB_NSTART", "3")),
  max_path_steps = as.integer(Sys.getenv("ROB_MAX_PATH_STEPS", "120")),
  max_active = as.integer(Sys.getenv("ROB_MAX_ACTIVE", "35")),
  max_iter = as.integer(Sys.getenv("ROB_MAX_ITER", "160")),
  inner_max_iter = as.integer(Sys.getenv("ROB_INNER_MAX_ITER", "60")),
  workers = as.integer(Sys.getenv("ROB_WORKERS", "1")),
  base_seed = as.integer(Sys.getenv("ROB_BASE_SEED", "20260604")),
  lambda_mu_grid = parse_num_grid(Sys.getenv(
    "ROB_LAMBDA_MU_GRID",
    "0,200,400,600"
  )),
  lambda_kappa_grid = parse_num_grid(Sys.getenv(
    "ROB_LAMBDA_KAPPA_GRID",
    "0,25,50"
  )),
  lambda_eta_grid = parse_num_grid(Sys.getenv(
    "ROB_LAMBDA_ETA_GRID",
    "0,0.5,1,2,5,10"
  )),
  out_dir = Sys.getenv("ROB_OUT_DIR", "results/robustness_kappa_mu_grid_260604")
)

if (!dir.exists(rob_cfg$out_dir)) dir.create(rob_cfg$out_dir, recursive = TRUE)
cell_dir <- file.path(rob_cfg$out_dir, "cells")
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
    selected_q = sum(active),
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

eta_contrast_score <- function(theta) {
  ord <- order(theta$kappa)
  eta <- sweep(theta$mu[ord, , drop = FALSE], 1, theta$kappa[ord], "*")
  abs(eta[2, ] - eta[1, ])
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

method_row <- function(method,
                       fit,
                       X,
                       z,
                       support_true,
                       q,
                       active,
                       tuning = NA_real_,
                       failed = FALSE,
                       error = NA_character_,
                       BIC = NA_real_,
                       EBIC = NA_real_,
                       df = NA_real_) {
  if (failed || is.null(fit)) {
    return(data.frame(
      method = method,
      failed = TRUE,
      error = error,
      ARI = NA_real_,
      tuning = tuning,
      loglik = NA_real_,
      BIC = BIC,
      EBIC = EBIC,
      df = df,
      selected_q = NA_real_,
      TPR = NA_real_,
      FPR = NA_real_,
      precision = NA_real_,
      F1 = NA_real_,
      mu_contrast_norm = NA_real_,
      eta_contrast_norm = NA_real_,
      mu_topq_recall = NA_real_,
      eta_topq_recall = NA_real_,
      kappa_low_hat = NA_real_,
      kappa_high_hat = NA_real_,
      kappa_ratio_hat = NA_real_
    ))
  }

  cluster <- max.col(fit$tau, ties.method = "first")
  sm <- support_metrics(active, support_true)
  cm <- contrast_metrics(fit, support_true, q)

  data.frame(
    method = method,
    failed = FALSE,
    error = NA_character_,
    ARI = adjusted_rand_index(z, cluster),
    tuning = tuning,
    loglik = fit$loglik,
    BIC = BIC,
    EBIC = EBIC,
    df = df,
    sm,
    cm
  )
}

failure_row <- function(method, err) {
  msg <- if (inherits(err, "error")) conditionMessage(err) else as.character(err)
  method_row(method, NULL, NULL, NULL, NULL, NA, rep(FALSE, 1),
             failed = TRUE, error = msg)
}

# ------------------------------------------------------------------------------
# Method wrappers
# ------------------------------------------------------------------------------

fit_rossi_sparse_bic_row <- function(X, z, support_true, cfg) {
  path <- fit_svMF_path(
    X,
    K = cfg$K_fit,
    labels_true = z,
    nstart = cfg$nstart,
    max_path_steps = cfg$max_path_steps
  )
  idx <- which.min(path$path$BIC)
  fit <- path$fits[[idx]]
  active <- colSums(abs(fit$mu) > 1e-8) > 0

  method_row(
    method = "rossi_sparse_vmf_BIC",
    fit = fit,
    X = X,
    z = z,
    support_true = support_true,
    q = cfg$q,
    active = active,
    tuning = path$path$beta[idx],
    BIC = path$path$BIC[idx],
    EBIC = path$path$EBIC[idx],
    df = path$path$df[idx]
  )
}

fit_separate_refit_row <- function(X, z, support_true, cfg) {
  sep_cfg <- list(
    q = cfg$q,
    K_fit_grid = cfg$K_fit,
    lambda_mu_grid = cfg$lambda_mu_grid,
    lambda_kappa_grid = cfg$lambda_kappa_grid,
    nstart = cfg$nstart,
    max_iter = cfg$max_iter,
    inner_max_iter = cfg$inner_max_iter
  )

  fit <- sep_fit_grid(X, z, support_true, sep_cfg)
  active <- colSums(abs(fit$fit$mu) > 1e-8) > 0
  refit <- sep_support_refit(
    X,
    K = cfg$K_fit,
    active = active,
    init = fit$fit,
    max_iter = cfg$max_iter
  )
  if (isTRUE(refit$failed)) stop("Separate support refit failed.")

  ic <- support_ic(refit$loglik, nrow(X), ncol(X), cfg$K_fit, sum(active))
  method_row(
    method = "separate_penalty_EM_BIC_refit",
    fit = refit,
    X = X,
    z = z,
    support_true = support_true,
    q = cfg$q,
    active = active,
    tuning = fit$row$lambda_mu,
    BIC = ic$BIC,
    EBIC = ic$EBIC,
    df = ic$df
  )
}

fit_eta_penalty_refit_row <- function(X, z, support_true, cfg) {
  eta_cfg <- list(
    q = cfg$q,
    K_fit = cfg$K_fit,
    lambda_eta_grid = cfg$lambda_eta_grid,
    nstart = cfg$nstart,
    max_iter = cfg$max_iter
  )

  fit <- eta_fit_grid(X, z, support_true, eta_cfg)
  active <- eta_penalty_active(fit$fit)
  if (!any(active)) stop("Eta penalty selected no active coordinates.")

  refit <- eta_support_refit(
    X,
    K = cfg$K_fit,
    active = active,
    init = fit$fit,
    max_iter = cfg$max_iter
  )
  if (isTRUE(refit$failed)) stop("Eta penalty support refit failed.")

  ic <- support_ic(refit$loglik, nrow(X), ncol(X), cfg$K_fit, sum(active))
  method_row(
    method = "eta_penalty_EM_BIC_refit",
    fit = refit,
    X = X,
    z = z,
    support_true = support_true,
    q = cfg$q,
    active = active,
    tuning = fit$row$lambda_eta,
    BIC = ic$BIC,
    EBIC = ic$EBIC,
    df = ic$df
  )
}

fit_eta_screening_refit_row <- function(X, z, support_true, cfg) {
  n <- nrow(X)
  d <- ncol(X)
  dense <- fit_svMF_multistart(X, cfg$K_fit, beta = 0, nstart = cfg$nstart)
  score <- eta_contrast_score(dense)
  ord <- order(score, decreasing = TRUE)
  m_grid <- seq_len(min(cfg$max_active, d))

  rows <- list()
  fits <- list()

  for (m in m_grid) {
    active <- rep(FALSE, d)
    active[ord[seq_len(m)]] <- TRUE
    fit <- eta_support_refit(
      X,
      K = cfg$K_fit,
      active = active,
      init = dense,
      max_iter = cfg$max_iter
    )
    if (isTRUE(fit$failed)) next

    ic <- support_ic(fit$loglik, n, d, cfg$K_fit, sum(active))
    rows[[length(rows) + 1L]] <- data.frame(m = m, ic)
    fits[[length(fits) + 1L]] <- list(fit = fit, active = active, ic = ic)
  }

  if (length(rows) == 0L) stop("All eta screening refits failed.")
  path <- do.call(rbind, rows)
  idx <- which.min(path$BIC)
  selected <- fits[[idx]]

  method_row(
    method = "eta_screening_BIC_refit",
    fit = selected$fit,
    X = X,
    z = z,
    support_true = support_true,
    q = cfg$q,
    active = selected$active,
    tuning = path$m[idx],
    BIC = selected$ic$BIC,
    EBIC = selected$ic$EBIC,
    df = selected$ic$df
  )
}

# ------------------------------------------------------------------------------
# Simulation driver
# ------------------------------------------------------------------------------

scenario_grid <- expand.grid(
  kappa_ratio = rob_cfg$kappa_ratio_grid,
  mu_cos = rob_cfg$mu_cos_grid,
  rep = seq_len(rob_cfg$n_rep)
)
scenario_grid$scenario_id <- seq_len(nrow(scenario_grid))

run_one <- function(i, cfg, cell_dir, scenario_grid) {
  sc <- scenario_grid[i, ]
  out_file <- file.path(
    cell_dir,
    sprintf("cell_ratio%s_mucos%s_rep%03d.csv",
            gsub("\\.", "p", as.character(sc$kappa_ratio)),
            gsub("\\.", "p", as.character(sc$mu_cos)),
            sc$rep)
  )
  if (file.exists(out_file)) return(out_file)

  set.seed(cfg$base_seed + i)
  kappa_high <- cfg$kappa_low * sc$kappa_ratio
  dat <- simulate_kappa_contrast_data(
    n = cfg$n,
    d = cfg$d,
    q = cfg$q,
    kappa_low = cfg$kappa_low,
    kappa_high = kappa_high,
    mu_cos = sc$mu_cos
  )

  X <- dat$X
  z <- dat$z
  support_true <- dat$params$support[1, ]

  rows <- list(
    tryCatch(
      fit_rossi_sparse_bic_row(X, z, support_true, cfg),
      error = function(e) failure_row("rossi_sparse_vmf_BIC", e)
    ),
    tryCatch(
      fit_separate_refit_row(X, z, support_true, cfg),
      error = function(e) failure_row("separate_penalty_EM_BIC_refit", e)
    ),
    tryCatch(
      fit_eta_penalty_refit_row(X, z, support_true, cfg),
      error = function(e) failure_row("eta_penalty_EM_BIC_refit", e)
    ),
    tryCatch(
      fit_eta_screening_refit_row(X, z, support_true, cfg),
      error = function(e) failure_row("eta_screening_BIC_refit", e)
    )
  )

  out <- do.call(rbind, rows)
  out$rep <- sc$rep
  out$n <- cfg$n
  out$d <- cfg$d
  out$q <- cfg$q
  out$mu_cos <- sc$mu_cos
  out$kappa_low <- cfg$kappa_low
  out$kappa_high <- kappa_high
  out$kappa_ratio <- sc$kappa_ratio
  out$true_mu_contrast_norm <- l2_norm(dat$params$mu[2, ] - dat$params$mu[1, ])
  out$true_eta_contrast_norm <- l2_norm(
    dat$params$kappa[2] * dat$params$mu[2, ] -
      dat$params$kappa[1] * dat$params$mu[1, ]
  )

  write.csv(out, out_file, row.names = FALSE)
  out_file
}

cat(sprintf(
  "Running robustness grid: scenarios=%d, reps/scenario=%d, n=%d, d=%d, q=%d\n",
  length(rob_cfg$kappa_ratio_grid) * length(rob_cfg$mu_cos_grid),
  rob_cfg$n_rep,
  rob_cfg$n,
  rob_cfg$d,
  rob_cfg$q
))
cat("kappa ratios: ", paste(rob_cfg$kappa_ratio_grid, collapse = ","), "\n", sep = "")
cat("mu_cos grid: ", paste(rob_cfg$mu_cos_grid, collapse = ","), "\n", sep = "")

tasks <- seq_len(nrow(scenario_grid))
workers <- max(1L, min(rob_cfg$workers, length(tasks)))

if (workers == 1L) {
  files <- character(length(tasks))
  for (ii in tasks) {
    cat(sprintf("[%03d/%03d]\n", ii, length(tasks)))
    files[ii] <- run_one(ii, rob_cfg, cell_dir, scenario_grid)
  }
} else {
  cl <- parallel::makeCluster(workers)
  on.exit(parallel::stopCluster(cl), add = TRUE)
  parallel::clusterEvalQ(cl, source(file.path("r", "rossi_barbaro_2022_reproduction.r")))
  parallel::clusterExport(cl, setdiff(ls(), "cl"), envir = environment())
  files <- parallel::parLapplyLB(
    cl,
    tasks,
    function(ii) run_one(ii, rob_cfg, cell_dir, scenario_grid)
  )
}

cell_files <- list.files(cell_dir, pattern = "^cell_.*\\.csv$", full.names = TRUE)
cell_tables <- lapply(cell_files, read.csv)
all_cols <- Reduce(union, lapply(cell_tables, names))
cell_tables <- lapply(cell_tables, function(tab) {
  missing <- setdiff(all_cols, names(tab))
  for (col in missing) tab[[col]] <- NA
  tab[, all_cols]
})
raw <- do.call(rbind, cell_tables)

raw_path <- file.path(rob_cfg$out_dir, sprintf("%s_raw.csv", rob_cfg$run_label))
summary_path <- file.path(rob_cfg$out_dir, sprintf("%s_summary.csv", rob_cfg$run_label))
write.csv(raw, raw_path, row.names = FALSE)

safe_mean <- function(x) if (all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE)
safe_se <- function(x) {
  nn <- sum(!is.na(x))
  if (nn > 1) sd(x, na.rm = TRUE) / sqrt(nn) else NA_real_
}

raw$failed_num <- as.numeric(raw$failed)
raw$valid_num <- as.numeric(!raw$failed)

keys <- c("method", "n", "d", "q", "mu_cos", "kappa_low", "kappa_high", "kappa_ratio")
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
  "ARI", "tuning", "loglik", "BIC", "EBIC", "df",
  "selected_q", "TPR", "FPR", "precision", "F1",
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

summary <- summary[order(summary$kappa_ratio, summary$mu_cos, summary$method), ]
write.csv(summary, summary_path, row.names = FALSE)

shown <- summary[, c(
  "kappa_ratio", "mu_cos", "method", "fail_rate", "valid_reps",
  "ARI_mean", "selected_q_mean", "TPR_mean", "FPR_mean",
  "precision_mean", "F1_mean", "eta_contrast_norm_mean",
  "kappa_ratio_hat_mean"
)]
num <- vapply(shown, is.numeric, logical(1))
shown[num] <- lapply(shown[num], round, 3)

cat("\nDone.\n")
cat("Raw: ", normalizePath(raw_path, winslash = "/"), "\n", sep = "")
cat("Summary: ", normalizePath(summary_path, winslash = "/"), "\n", sep = "")
print(shown, row.names = FALSE)
