# ==============================================================================
# Proposed eta-contrast screening/refit under the kappa-contrast limitation case
# ------------------------------------------------------------------------------
# Same data-generating scenario as rb2022_kappa_contrast_limit_run.r:
#   mu_1 = mu_2, kappa_1 << kappa_2
#
# Proposed simplified workflow:
#   1. Fit dense vMF with K = 2.
#   2. Score variables by |eta_2j - eta_1j|, eta_k = kappa_k * mu_k.
#   3. Select active coordinates along a top-m path by support-constrained BIC.
#   4. Refit an unpenalized vMF mixture with mu constrained to the selected support.
# ==============================================================================

source(file.path("r", "rossi_barbaro_2022_reproduction.r"))

cfg <- list(
  run_label = Sys.getenv("ETA_RUN_LABEL", "eta_contrast_proposed_limit"),
  n_rep = as.integer(Sys.getenv("ETA_N_REP", "30")),
  n = as.integer(Sys.getenv("ETA_N", "1000")),
  d = as.integer(Sys.getenv("ETA_D", "100")),
  q = as.integer(Sys.getenv("ETA_Q", "10")),
  K_true = 2,
  kappa_low = as.numeric(Sys.getenv("ETA_KAPPA_LOW", "20")),
  kappa_high = as.numeric(Sys.getenv("ETA_KAPPA_HIGH", "200")),
  mu_cos = as.numeric(Sys.getenv("ETA_MU_COS", "1")),
  nstart = as.integer(Sys.getenv("ETA_NSTART", "5")),
  max_active = as.integer(Sys.getenv("ETA_MAX_ACTIVE", "35")),
  score_type = Sys.getenv("ETA_SCORE", "eta"),
  workers = as.integer(Sys.getenv("ETA_WORKERS", "6")),
  base_seed = 20260602,
  out_dir = Sys.getenv("ETA_OUT_DIR", "results/eta_contrast_proposed_limit_260602")
)

if (!dir.exists(cfg$out_dir)) dir.create(cfg$out_dir, recursive = TRUE)
cell_dir <- file.path(cfg$out_dir, "cells")
plot_dir <- file.path(cfg$out_dir, "plots")
if (!dir.exists(cell_dir)) dir.create(cell_dir, recursive = TRUE)
if (!dir.exists(plot_dir)) dir.create(plot_dir, recursive = TRUE)

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

mask_and_normalize_mu <- function(mu, active, fallback = NULL) {
  K <- nrow(mu)
  d <- ncol(mu)
  out <- matrix(0, nrow = K, ncol = d)
  out[, active] <- mu[, active, drop = FALSE]
  nr <- sqrt(rowSums(out * out))
  for (k in seq_len(K)) {
    if (nr[k] < 1e-10) {
      if (!is.null(fallback)) {
        out[k, active] <- fallback[k, active]
      } else {
        out[k, active] <- rnorm(sum(active))
      }
      nr[k] <- l2_norm(out[k, ])
    }
    out[k, ] <- out[k, ] / nr[k]
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
        failed = TRUE, converged = FALSE, iter = iter,
        loglik = e$loglik, tau = tau
      )))
    }

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
        failed = FALSE, converged = TRUE, iter = iter,
        loglik = e_new$loglik, tau = e_new$tau
      )))
    }
    prev <- e_new$loglik
  }

  if (is.null(last_e)) last_e <- e_step_vmf(X, theta)
  c(theta, list(
    failed = FALSE, converged = FALSE, iter = max_iter,
    loglik = last_e$loglik,
    tau = last_e$tau
  ))
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

eta_contrast_score <- function(theta) {
  ord <- order(theta$kappa)
  eta <- sweep(theta$mu[ord, , drop = FALSE], 1, theta$kappa[ord], "*")
  abs(eta[2, ] - eta[1, ])
}

mu_contrast_score <- function(theta) {
  ord <- order(theta$kappa)
  mu <- theta$mu[ord, , drop = FALSE]
  abs(mu[2, ] - mu[1, ])
}

contrast_score <- function(theta, score_type) {
  if (score_type == "eta") return(eta_contrast_score(theta))
  if (score_type == "mu_kappa_separate") return(mu_contrast_score(theta))
  stop("Unknown ETA_SCORE: ", score_type)
}

score_method_prefix <- function(score_type) {
  if (score_type == "eta") return("eta_contrast")
  if (score_type == "mu_kappa_separate") return("mu_kappa_separate")
  stop("Unknown ETA_SCORE: ", score_type)
}

fit_eta_contrast_refit <- function(X, z, support_true, nstart, max_active,
                                   score_type = "eta") {
  n <- nrow(X)
  d <- ncol(X)
  K <- 2

  dense <- fit_svMF_multistart(X, K, beta = 0, nstart = nstart)
  score <- contrast_score(dense, score_type)
  ord <- order(score, decreasing = TRUE)
  m_grid <- seq_len(min(max_active, d))

  rows <- list()
  fits <- list()
  for (m in m_grid) {
    active <- rep(FALSE, d)
    active[ord[seq_len(m)]] <- TRUE
    fit <- fit_support_constrained_vmf(X, K, active, dense)
    if (isTRUE(fit$failed)) next
    ic <- support_ic(fit$loglik, n, d, K, m)
    cl <- max.col(fit$tau, ties.method = "first")
    sm <- support_metrics(active, support_true)
    row <- data.frame(
      m = m,
      ARI = adjusted_rand_index(z, cl),
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
  bic_idx <- which.min(path$BIC)
  ebic_idx <- which.min(path$EBIC)

  make_selected <- function(idx, criterion) {
    one <- fits[[idx]]
    cm <- contrast_from_fit(one$fit, support_true)
    cbind(
      data.frame(method = paste0(score_method_prefix(score_type), "_", criterion, "_refit")),
      one$row,
      cm
    )
  }

  list(
    path = path,
    BIC = make_selected(bic_idx, "BIC"),
    EBIC = make_selected(ebic_idx, "EBIC"),
    dense_init = dense
  )
}

contrast_from_fit <- function(theta, support_true) {
  score <- eta_contrast_score(theta)
  active_top <- rep(FALSE, length(score))
  active_top[order(score, decreasing = TRUE)[seq_len(sum(support_true))]] <- TRUE
  sm <- support_metrics(active_top, support_true)
  ord <- order(theta$kappa)
  mu <- theta$mu[ord, , drop = FALSE]
  kappa <- theta$kappa[ord]
  mu_delta <- mu[2, ] - mu[1, ]
  eta_delta <- kappa[2] * mu[2, ] - kappa[1] * mu[1, ]
  data.frame(
    eta_topq_TPR = sm$TPR,
    eta_topq_FPR = sm$FPR,
    mu_contrast_norm = l2_norm(mu_delta),
    eta_contrast_norm = l2_norm(eta_delta),
    kappa_low_hat = kappa[1],
    kappa_high_hat = kappa[2],
    kappa_ratio_hat = kappa[2] / max(kappa[1], 1e-12)
  )
}

run_one <- function(rep_id, cfg, cell_dir) {
  out_file <- file.path(cell_dir, sprintf("cell_%03d.csv", rep_id))
  path_file <- file.path(cell_dir, sprintf("path_%03d.csv", rep_id))
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
  support <- dat$params$support[1, ]

  fit <- tryCatch(
    fit_eta_contrast_refit(
      X, z, support,
      cfg$nstart,
      cfg$max_active,
      score_type = cfg$score_type
    ),
    error = function(e) e
  )

  if (inherits(fit, "error")) {
    out <- data.frame(
      method = paste0(score_method_prefix(cfg$score_type), c("_BIC_refit", "_EBIC_refit")),
      m = NA_real_,
      ARI = NA_real_,
      loglik = NA_real_,
      df = NA_real_,
      BIC = NA_real_,
      EBIC = NA_real_,
      shat = NA_real_,
      TPR = NA_real_,
      FPR = NA_real_,
      precision = NA_real_,
      F1 = NA_real_,
      eta_topq_TPR = NA_real_,
      eta_topq_FPR = NA_real_,
      mu_contrast_norm = NA_real_,
      eta_contrast_norm = NA_real_,
      kappa_low_hat = NA_real_,
      kappa_high_hat = NA_real_,
      kappa_ratio_hat = NA_real_,
      error = conditionMessage(fit)
    )
  } else {
    out <- rbind(fit$BIC, fit$EBIC)
    out$error <- NA_character_
  }
  out$rep <- rep_id
  out$n <- cfg$n
  out$d <- cfg$d
  out$q <- cfg$q
  out$mu_cos <- cfg$mu_cos
  out$kappa_low <- cfg$kappa_low
  out$kappa_high <- cfg$kappa_high
  out$true_mu_contrast_norm <- l2_norm(dat$params$mu[2, ] - dat$params$mu[1, ])
  out$true_eta_contrast_norm <- l2_norm(
    dat$params$kappa[2] * dat$params$mu[2, ] -
      dat$params$kappa[1] * dat$params$mu[1, ]
  )

  if (!inherits(fit, "error")) {
    path <- fit$path
    path$rep <- rep_id
    write.csv(path, path_file, row.names = FALSE)
  }
  write.csv(out, out_file, row.names = FALSE)
  out_file
}

cat(sprintf(
  "Running proposed eta-contrast refit: reps=%d, n=%d, d=%d, q=%d, kappa=(%.1f, %.1f), mu_cos=%.3f\n",
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
  parallel::clusterExport(
    cl,
    c(
      "cfg", "cell_dir", "make_kappa_contrast_params",
      "simulate_kappa_contrast_data", "mask_and_normalize_mu",
      "fit_support_constrained_vmf", "support_df", "support_ic",
      "support_metrics", "eta_contrast_score", "mu_contrast_score",
      "contrast_score", "score_method_prefix", "fit_eta_contrast_refit",
      "contrast_from_fit", "run_one"
    ),
    envir = environment()
  )
  files <- parallel::parLapplyLB(cl, tasks, function(i) run_one(i, cfg, cell_dir))
}

cell_files <- list.files(cell_dir, pattern = "^cell_[0-9]+\\.csv$", full.names = TRUE)
cell_tables <- lapply(cell_files, read.csv)
all_cols <- Reduce(union, lapply(cell_tables, names))
cell_tables <- lapply(cell_tables, function(tab) {
  missing <- setdiff(all_cols, names(tab))
  for (col in missing) tab[[col]] <- NA
  tab[, all_cols]
})
raw <- do.call(rbind, cell_tables)
if (!"error" %in% names(raw)) raw$error <- NA_character_
raw_path <- file.path(cfg$out_dir, sprintf("%s_raw.csv", cfg$run_label))
summary_path <- file.path(cfg$out_dir, sprintf("%s_summary.csv", cfg$run_label))
write.csv(raw, raw_path, row.names = FALSE)

safe_mean <- function(x) if (all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE)
safe_se <- function(x) {
  nn <- sum(!is.na(x))
  if (nn > 1) sd(x, na.rm = TRUE) / sqrt(nn) else NA_real_
}
raw$run_failed <- !is.na(raw$error) & nzchar(raw$error)

metrics <- c(
  "m", "ARI", "df", "BIC", "EBIC", "shat", "TPR", "FPR",
  "precision", "F1", "eta_topq_TPR", "eta_topq_FPR",
  "mu_contrast_norm", "eta_contrast_norm",
  "kappa_low_hat", "kappa_high_hat", "kappa_ratio_hat",
  "true_mu_contrast_norm", "true_eta_contrast_norm"
)

summary <- unique(raw[, c("method", "n", "d", "q", "mu_cos", "kappa_low", "kappa_high")])
fail_rate <- aggregate(
  raw[, "run_failed", drop = FALSE],
  raw[, "method", drop = FALSE],
  mean
)
names(fail_rate)[2] <- "fail_rate"
valid_reps <- aggregate(
  !raw$run_failed,
  raw[, "method", drop = FALSE],
  sum
)
names(valid_reps)[2] <- "valid_reps"
total_reps <- aggregate(
  raw[, "rep", drop = FALSE],
  raw[, "method", drop = FALSE],
  function(x) length(unique(x))
)
names(total_reps)[2] <- "total_reps"
summary <- merge(summary, fail_rate, by = "method", all.x = TRUE)
summary <- merge(summary, valid_reps, by = "method", all.x = TRUE)
summary <- merge(summary, total_reps, by = "method", all.x = TRUE)
for (m in metrics) {
  agg_mean <- aggregate(raw[, m, drop = FALSE], raw[, "method", drop = FALSE], safe_mean)
  agg_se <- aggregate(raw[, m, drop = FALSE], raw[, "method", drop = FALSE], safe_se)
  names(agg_mean)[2] <- paste0(m, "_mean")
  names(agg_se)[2] <- paste0(m, "_se")
  summary <- merge(summary, agg_mean, by = "method", all.x = TRUE)
  summary <- merge(summary, agg_se, by = "method", all.x = TRUE)
}
summary <- summary[order(summary$method), ]
write.csv(summary, summary_path, row.names = FALSE)

plot_bars <- function(summary, metric, file_name, main, ylim = NULL) {
  vals <- summary[[paste0(metric, "_mean")]]
  ses <- summary[[paste0(metric, "_se")]]
  names(vals) <- summary$method
  if (is.null(ylim)) ylim <- c(0, max(vals + ses, na.rm = TRUE) * 1.15)
  png(file.path(plot_dir, file_name), width = 1200, height = 760, res = 140)
  old <- par(mar = c(7, 4, 3, 1))
  bp <- barplot(vals, ylim = ylim, col = "seagreen", ylab = metric,
                main = main, las = 2)
  arrows(bp, vals - ses, bp, vals + ses, angle = 90, code = 3, length = 0.04)
  par(old)
  dev.off()
}

plot_bars(summary, "ARI", "fig_eta_proposed_ari.png",
          "Eta-contrast refit clustering", ylim = c(0, 1))
plot_bars(summary, "shat", "fig_eta_proposed_shat.png",
          "Selected active coordinates", ylim = c(0, cfg$max_active))
plot_bars(summary, "F1", "fig_eta_proposed_support_f1.png",
          "Support recovery F1", ylim = c(0, 1))

path_files <- list.files(cell_dir, pattern = "^path_[0-9]+\\.csv$", full.names = TRUE)
if (length(path_files) > 0L) {
  paths <- do.call(rbind, lapply(path_files, read.csv))
  path_summary <- aggregate(
    paths[, c("BIC", "EBIC", "ARI", "TPR", "FPR", "F1")],
    paths[, "m", drop = FALSE],
    safe_mean
  )
  write.csv(path_summary, file.path(cfg$out_dir, sprintf("%s_path_summary.csv", cfg$run_label)),
            row.names = FALSE)

  png(file.path(plot_dir, "fig_eta_proposed_bic_path.png"),
      width = 1200, height = 760, res = 140)
  old <- par(mar = c(4, 4, 3, 1))
  plot(path_summary$m, path_summary$BIC, type = "b", pch = 16, col = "seagreen",
       xlab = "number of selected coordinates", ylab = "mean BIC",
       main = "Eta-contrast support path")
  abline(v = cfg$q, lty = 2, col = "tomato")
  legend("topright", legend = "true q", lty = 2, col = "tomato", bty = "n")
  par(old)
  dev.off()
} else {
  path_summary <- data.frame()
  write.csv(path_summary, file.path(cfg$out_dir, sprintf("%s_path_summary.csv", cfg$run_label)),
            row.names = FALSE)
  cat("No successful path files were produced; skipped BIC path plot.\n")
}

shown <- summary
num <- vapply(shown, is.numeric, logical(1))
shown[num] <- lapply(shown[num], round, 3)
cat("\nDone.\n")
cat("Raw: ", normalizePath(raw_path, winslash = "/"), "\n", sep = "")
cat("Summary: ", normalizePath(summary_path, winslash = "/"), "\n", sep = "")
cat("Plots: ", normalizePath(plot_dir, winslash = "/"), "\n", sep = "")
print(shown, row.names = FALSE)
