# ==============================================================================
# Limitation scenario for Rossi & Barbaro (2022)
# ------------------------------------------------------------------------------
# The component directions are identical or almost identical, while the
# concentrations differ strongly. This creates a cluster contrast in
# eta_k = kappa_k * mu_k, but not in the directional prototype mu_k itself.
# ==============================================================================

source(file.path("r", "rossi_barbaro_2022_reproduction.r"))

cfg <- list(
  run_label = Sys.getenv("KC_RUN_LABEL", "rb2022_kappa_contrast_limit"),
  n_rep = as.integer(Sys.getenv("KC_N_REP", "30")),
  n = as.integer(Sys.getenv("KC_N", "1000")),
  d = as.integer(Sys.getenv("KC_D", "100")),
  q = as.integer(Sys.getenv("KC_Q", "10")),
  K_true = 2,
  K_fit_grid = 1:3,
  kappa_low = as.numeric(Sys.getenv("KC_KAPPA_LOW", "20")),
  kappa_high = as.numeric(Sys.getenv("KC_KAPPA_HIGH", "200")),
  mu_cos = as.numeric(Sys.getenv("KC_MU_COS", "1")),
  nstart = as.integer(Sys.getenv("KC_NSTART", "5")),
  max_path_steps = as.integer(Sys.getenv("KC_MAX_PATH_STEPS", "250")),
  workers = as.integer(Sys.getenv("KC_WORKERS", "6")),
  base_seed = 20260602,
  out_dir = Sys.getenv("KC_OUT_DIR", "results/rb2022_kappa_contrast_limit_260602")
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

safe_topq_recall <- function(score, support, q) {
  if (all(!is.finite(score)) || max(abs(score), na.rm = TRUE) < 1e-12) {
    return(NA_real_)
  }
  top <- order(abs(score), decreasing = TRUE)[seq_len(q)]
  mean(support[top])
}

contrast_metrics <- function(theta, support, q) {
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
  mu_delta <- mu[2, ] - mu[1, ]
  eta_delta <- kappa[2] * mu[2, ] - kappa[1] * mu[1, ]

  data.frame(
    mu_contrast_norm = l2_norm(mu_delta),
    eta_contrast_norm = l2_norm(eta_delta),
    mu_topq_recall = safe_topq_recall(mu_delta, support, q),
    eta_topq_recall = safe_topq_recall(eta_delta, support, q),
    kappa_low_hat = kappa[1],
    kappa_high_hat = kappa[2],
    kappa_ratio_hat = kappa[2] / max(kappa[1], 1e-12)
  )
}

select_dense_bic <- function(X, z, K_grid, nstart) {
  rows <- list()
  fits <- list()
  for (K in K_grid) {
    fit <- fit_svMF_multistart(X, K, beta = 0, nstart = nstart)
    row <- evaluate_fit(fit, X, beta = 0, labels_true = z)
    row$K_fit <- K
    rows[[length(rows) + 1L]] <- row
    fits[[as.character(K)]] <- fit
  }
  tab <- do.call(rbind, rows)
  best <- tab[which.min(tab$BIC), , drop = FALSE]
  list(row = best, fit = fits[[as.character(best$K_fit)]])
}

select_sparse_bic <- function(X, z, K_grid, nstart, max_path_steps) {
  rows <- list()
  fits <- list()
  for (K in K_grid) {
    path <- fit_svMF_path(
      X, K,
      labels_true = z,
      nstart = nstart,
      max_path_steps = max_path_steps
    )
    ptab <- path$path
    idx <- which.min(ptab$BIC)
    best <- ptab[idx, , drop = FALSE]
    best$K_fit <- K
    rows[[length(rows) + 1L]] <- best
    fits[[as.character(K)]] <- path$fits[[idx]]
  }
  tab <- do.call(rbind, rows)
  best <- tab[which.min(tab$BIC), , drop = FALSE]
  list(row = best, fit = fits[[as.character(best$K_fit)]])
}

run_one <- function(rep_id, cfg, cell_dir) {
  out_file <- file.path(cell_dir, sprintf("cell_%03d.csv", rep_id))
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
  params <- dat$params
  support <- params$support[1, ]

  true_theta <- list(alpha = params$alpha, mu = params$mu, kappa = params$kappa)
  true_e <- e_step_vmf(X, true_theta)
  oracle_cluster <- max.col(true_e$tau, ties.method = "first")

  true_contrast <- contrast_metrics(true_theta, support, cfg$q)
  true_contrast$method <- "true_parameters"
  true_contrast$ARI <- adjusted_rand_index(z, oracle_cluster)
  true_contrast$K_fit <- 2
  true_contrast$BIC <- NA_real_
  true_contrast$nnz_fraction <- mean(params$support)
  true_contrast$beta <- NA_real_

  skm <- spherical_kmeans(X, 2, nstart = min(cfg$nstart, 10))
  skm_row <- data.frame(
    method = "spherical_kmeans",
    ARI = adjusted_rand_index(z, skm$cluster),
    K_fit = 2,
    BIC = NA_real_,
    nnz_fraction = NA_real_,
    beta = NA_real_,
    contrast_metrics(
      list(mu = skm$mu, kappa = c(1, 1)),
      support,
      cfg$q
    )
  )

  dense <- select_dense_bic(X, z, cfg$K_fit_grid, cfg$nstart)
  dense_cm <- contrast_metrics(dense$fit, support, cfg$q)
  dense_row <- data.frame(
    method = "dense_vmf_BIC",
    ARI = dense$row$ARI,
    K_fit = dense$row$K_fit,
    BIC = dense$row$BIC,
    nnz_fraction = dense$row$nnz_fraction,
    beta = 0,
    dense_cm
  )

  sparse <- select_sparse_bic(
    X, z, cfg$K_fit_grid,
    cfg$nstart,
    cfg$max_path_steps
  )
  sparse_cm <- contrast_metrics(sparse$fit, support, cfg$q)
  sparse_row <- data.frame(
    method = "rossi_sparse_vmf_BIC",
    ARI = sparse$row$ARI,
    K_fit = sparse$row$K_fit,
    BIC = sparse$row$BIC,
    nnz_fraction = sparse$row$nnz_fraction,
    beta = sparse$row$beta,
    sparse_cm
  )

  rows <- rbind(true_contrast, skm_row, dense_row, sparse_row)
  rows$rep <- rep_id
  rows$n <- cfg$n
  rows$d <- cfg$d
  rows$q <- cfg$q
  rows$mu_cos <- cfg$mu_cos
  rows$kappa_low <- cfg$kappa_low
  rows$kappa_high <- cfg$kappa_high
  rows$true_mu_contrast_norm <- l2_norm(params$mu[2, ] - params$mu[1, ])
  rows$true_eta_contrast_norm <- l2_norm(
    params$kappa[2] * params$mu[2, ] -
      params$kappa[1] * params$mu[1, ]
  )

  write.csv(rows, out_file, row.names = FALSE)
  out_file
}

tasks <- seq_len(cfg$n_rep)
cat(sprintf(
  "Running kappa-contrast limitation simulation: reps=%d, n=%d, d=%d, q=%d, kappa=(%.1f, %.1f), mu_cos=%.3f\n",
  cfg$n_rep, cfg$n, cfg$d, cfg$q, cfg$kappa_low, cfg$kappa_high, cfg$mu_cos
))

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
      "simulate_kappa_contrast_data", "safe_topq_recall",
      "contrast_metrics", "select_dense_bic", "select_sparse_bic", "run_one"
    ),
    envir = environment()
  )
  files <- parallel::parLapplyLB(cl, tasks, function(i) run_one(i, cfg, cell_dir))
}

cell_files <- list.files(cell_dir, pattern = "^cell_[0-9]+\\.csv$", full.names = TRUE)
raw <- do.call(rbind, lapply(cell_files, read.csv))
raw_path <- file.path(cfg$out_dir, sprintf("%s_raw.csv", cfg$run_label))
summary_path <- file.path(cfg$out_dir, sprintf("%s_summary.csv", cfg$run_label))
write.csv(raw, raw_path, row.names = FALSE)

safe_mean <- function(x) if (all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE)
safe_se <- function(x) {
  nn <- sum(!is.na(x))
  if (nn > 1) sd(x, na.rm = TRUE) / sqrt(nn) else NA_real_
}

metrics <- c(
  "ARI", "K_fit", "nnz_fraction", "beta",
  "mu_contrast_norm", "eta_contrast_norm",
  "mu_topq_recall", "eta_topq_recall",
  "kappa_low_hat", "kappa_high_hat", "kappa_ratio_hat",
  "true_mu_contrast_norm", "true_eta_contrast_norm"
)

summary <- unique(raw[, c("method", "n", "d", "q", "mu_cos", "kappa_low", "kappa_high")])
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

plot_method_bars <- function(summary, metric, file_name, main, ylim = NULL) {
  vals <- summary[[paste0(metric, "_mean")]]
  ses <- summary[[paste0(metric, "_se")]]
  names(vals) <- summary$method
  if (is.null(ylim)) ylim <- c(0, max(vals + ses, na.rm = TRUE) * 1.15)
  png(file.path(plot_dir, file_name), width = 1200, height = 760, res = 140)
  old <- par(mar = c(7, 4, 3, 1))
  bp <- barplot(vals, ylim = ylim, col = "steelblue", ylab = metric,
                main = main, las = 2)
  arrows(bp, vals - ses, bp, vals + ses, angle = 90, code = 3, length = 0.04)
  par(old)
  dev.off()
}

plot_method_bars(summary, "ARI", "fig_kappa_contrast_ari.png",
                 "Clustering under same mu, different kappa", ylim = c(0, 1))
plot_method_bars(summary, "mu_topq_recall", "fig_kappa_contrast_mu_topq.png",
                 "Top-q active recall from mu contrast", ylim = c(0, 1))
plot_method_bars(summary, "eta_topq_recall", "fig_kappa_contrast_eta_topq.png",
                 "Top-q active recall from eta contrast", ylim = c(0, 1))

params <- make_kappa_contrast_params(
  cfg$d, cfg$q, cfg$kappa_low, cfg$kappa_high, cfg$mu_cos
)
mu_delta <- params$mu[2, ] - params$mu[1, ]
eta_delta <- params$kappa[2] * params$mu[2, ] - params$kappa[1] * params$mu[1, ]
png(file.path(plot_dir, "fig_true_mu_vs_eta_contrast.png"),
    width = 1400, height = 720, res = 140)
old <- par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
barplot(abs(mu_delta), col = ifelse(params$support[1, ], "tomato", "gray80"),
        main = "|mu2 - mu1|", xlab = "coordinate", ylab = "absolute contrast")
barplot(abs(eta_delta), col = ifelse(params$support[1, ], "tomato", "gray80"),
        main = "|kappa2 mu2 - kappa1 mu1|", xlab = "coordinate",
        ylab = "absolute contrast")
par(old)
dev.off()

shown <- summary
num <- vapply(shown, is.numeric, logical(1))
shown[num] <- lapply(shown[num], round, 3)
cat("\nDone.\n")
cat("Raw: ", normalizePath(raw_path, winslash = "/"), "\n", sep = "")
cat("Summary: ", normalizePath(summary_path, winslash = "/"), "\n", sep = "")
cat("Plots: ", normalizePath(plot_dir, winslash = "/"), "\n", sep = "")
print(shown, row.names = FALSE)
