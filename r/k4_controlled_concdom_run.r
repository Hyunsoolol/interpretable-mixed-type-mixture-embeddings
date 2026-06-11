# ==============================================================================
# K=4 controlled concentration-dominant simulation
# ------------------------------------------------------------------------------
# This is a revised version of the previous realistic concentration-dominant
# setting.  The active variable structure is kept identical to the K=4 stress
# setting: all components share the same q active coordinates.  Only the mean
# directions and concentrations are relaxed to a more realistic configuration.
#
# Tuning remains the official 260622 rule:
#   path-based candidates + BIC selection + optional support refit.
# ==============================================================================

source_until_before <- function(file, marker, back = 1L) {
  lines <- readLines(file, encoding = "UTF-8", warn = FALSE)
  idx <- tail(grep(marker, lines, fixed = TRUE), 1L)
  if (is.na(idx)) stop(sprintf("Marker not found in %s: %s", file, marker))
  keep <- seq_len(max(1L, idx - back))
  eval(parse(text = lines[keep]), envir = .GlobalEnv)
}

source_until_before(file.path("r", "k4_path_tuning_compare_run.r"), "rows <- list()", back = 1L)

cfg$run_label <- Sys.getenv("K4_CONTROLLED_LABEL", "k4_controlled_concdom_260622")
cfg$n_rep <- as.integer(Sys.getenv("K4_CONTROLLED_N_REP", "20"))
cfg$n <- as.integer(Sys.getenv("K4_CONTROLLED_N", "1000"))
cfg$d <- as.integer(Sys.getenv("K4_CONTROLLED_D", "100"))
cfg$K <- as.integer(Sys.getenv("K4_CONTROLLED_K", "4"))
cfg$nstart <- as.integer(Sys.getenv("K4_CONTROLLED_NSTART", "5"))
cfg$max_iter <- as.integer(Sys.getenv("K4_CONTROLLED_MAX_ITER", "100"))
cfg$max_path_steps <- as.integer(Sys.getenv("K4_CONTROLLED_ROSSI_STEPS", "220"))
cfg$sep_mu_path_steps <- as.integer(Sys.getenv("K4_CONTROLLED_SEP_MU_STEPS", "300"))
cfg$eta_path_steps <- as.integer(Sys.getenv("K4_CONTROLLED_ETA_STEPS", "120"))
cfg$base_seed <- as.integer(Sys.getenv("K4_CONTROLLED_BASE_SEED", "20260622"))
cfg$out_dir <- Sys.getenv(
  "K4_CONTROLLED_OUT_DIR",
  "results/k4_controlled_concdom_260622"
)

q_active <- as.integer(Sys.getenv("K4_CONTROLLED_Q", "10"))
target_cos <- as.numeric(Sys.getenv("K4_CONTROLLED_MU_COS", "0.95"))
kappa_vec <- as.numeric(strsplit(
  Sys.getenv("K4_CONTROLLED_KAPPA", "25,40,65,100"),
  ",",
  fixed = TRUE
)[[1]])

dir.create(cfg$out_dir, recursive = TRUE, showWarnings = FALSE)

simplex_rows <- function(K) {
  centered <- diag(K) - matrix(1 / K, nrow = K, ncol = K)
  eig <- eigen(centered, symmetric = TRUE)
  basis <- eig$vectors[, seq_len(K - 1), drop = FALSE]
  coords <- centered %*% basis
  coords * sqrt(K / (K - 1))
}

orthogonal_basis_to <- function(u, m) {
  q <- length(u)
  A <- cbind(u, diag(q)[, seq_len(m), drop = FALSE])
  Q <- qr.Q(qr(A), complete = FALSE)
  Q[, 2:(m + 1), drop = FALSE]
}

make_controlled_concdom_params <- function(d, q, K, target_cos, kappa) {
  if (q < K) stop("q must be at least K for simplex mean construction.")
  if (length(kappa) != K) stop("length(kappa) must match K.")
  if (target_cos <= -1 / (K - 1) || target_cos >= 1) {
    stop("target_cos must be in (-1/(K-1), 1).")
  }

  support <- matrix(FALSE, nrow = K, ncol = d)
  support[, seq_len(q)] <- TRUE

  base <- rep(1 / sqrt(q), q)
  basis <- orthogonal_basis_to(base, K - 1)
  simplex <- simplex_rows(K)
  v <- simplex %*% t(basis)

  a2 <- (target_cos * (K - 1) + 1) / K
  a <- sqrt(a2)
  b <- sqrt(1 - a2)

  mu_active <- sweep(v, 2, b, "*") + matrix(a * base, nrow = K, ncol = q, byrow = TRUE)
  mu <- matrix(0, nrow = K, ncol = d)
  mu[, seq_len(q)] <- mu_active
  mu <- normalize_rows(mu)

  list(
    alpha = rep(1 / K, K),
    mu = mu,
    kappa = kappa,
    support = support
  )
}

run_one_controlled <- function(rep_id, cfg) {
  set.seed(cfg$base_seed + rep_id)
  params <- make_controlled_concdom_params(
    d = cfg$d, q = q_active, K = cfg$K,
    target_cos = target_cos, kappa = kappa_vec
  )
  dat <- simulate_from_params(cfg$n, params)
  pairwise_cos <- as.numeric(crossprod(t(params$mu))[upper.tri(diag(cfg$K))])
  cat(sprintf(
    "[controlled_concdom] rep %d/%d: q=%d, mean pairwise cos=%.3f, kappa=(%s)\n",
    rep_id, cfg$n_rep, q_active, mean(pairwise_cos),
    paste(kappa_vec, collapse = ",")
  ))

  out <- rbind(
    fit_rossi_path_pair(dat$X, dat$z, dat$params, cfg),
    fit_separate_path_grid_pair(dat$X, dat$z, dat$params, cfg),
    fit_eta_centered_path_pair(dat$X, dat$z, dat$params, cfg)
  )
  out$scenario <- "controlled_concdom_common_support"
  out$rep <- rep_id
  out$n <- cfg$n
  out$d <- cfg$d
  out$K_true <- cfg$K
  out$q_active_common <- q_active
  out$mu_pairwise_cos_target <- target_cos
  out$mu_pairwise_cos_mean <- mean(pairwise_cos)
  out$kappa_true_min <- min(kappa_vec)
  out$kappa_true_max <- max(kappa_vec)
  out$kappa_true_ratio <- max(kappa_vec) / min(kappa_vec)
  out
}

cat(sprintf(
  "Running K=4 controlled concentration-dominant simulation: reps=%d, n=%d, d=%d, q=%d, target_cos=%.3f, nstart=%d\n",
  cfg$n_rep, cfg$n, cfg$d, q_active, target_cos, cfg$nstart
))

rows <- list()
for (rep_id in seq_len(cfg$n_rep)) {
  rows[[rep_id]] <- tryCatch(
    run_one_controlled(rep_id, cfg),
    error = function(e) {
      data.frame(
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
        scenario = "controlled_concdom_common_support",
        rep = rep_id, n = cfg$n, d = cfg$d, K_true = cfg$K,
        q_active_common = q_active,
        mu_pairwise_cos_target = target_cos,
        mu_pairwise_cos_mean = NA_real_,
        kappa_true_min = min(kappa_vec),
        kappa_true_max = max(kappa_vec),
        kappa_true_ratio = max(kappa_vec) / min(kappa_vec),
        error = conditionMessage(e)
      )
    }
  )
}

raw <- do.call(rbind, rows)
raw_path <- file.path(cfg$out_dir, sprintf("%s_raw.csv", cfg$run_label))
write.csv(raw, raw_path, row.names = FALSE)

num_cols <- names(raw)[vapply(raw, is.numeric, logical(1))]
groups <- unique(raw[, c("scenario", "method")])
summary <- do.call(rbind, lapply(seq_len(nrow(groups)), function(i) {
  sub <- raw[raw$scenario == groups$scenario[i] & raw$method == groups$method[i], ]
  means <- aggregate(
    sub[, num_cols, drop = FALSE],
    list(dummy = rep(1, nrow(sub))),
    mean,
    na.rm = TRUE
  )
  means$dummy <- NULL
  data.frame(
    scenario = groups$scenario[i],
    method = groups$method[i],
    reps = length(unique(sub$rep)),
    means,
    row.names = NULL
  )
}))

method_order <- c(
  "Rossi path BIC", "Rossi path BIC + refit",
  "Separate 2D path/grid BIC", "Separate 2D path/grid BIC + refit",
  "Eta centered path BIC", "Eta centered path BIC + refit",
  "ERROR"
)
summary$method <- factor(summary$method, levels = method_order)
summary <- summary[order(summary$scenario, summary$method), ]
summary$method <- as.character(summary$method)

summary_path <- file.path(cfg$out_dir, sprintf("%s_summary.csv", cfg$run_label))
write.csv(summary, summary_path, row.names = FALSE)

cat("\nWrote:\n")
cat("  ", normalizePath(raw_path, winslash = "/"), "\n", sep = "")
cat("  ", normalizePath(summary_path, winslash = "/"), "\n", sep = "")
print(summary[, c(
  "scenario", "method", "reps", "ARI", "true_union_q", "selected_q",
  "TPR", "FPR", "Precision", "F1", "MSE_mu", "MSE_kappa",
  "MSE_centered_eta", "kappa_hat_mean", "BIC", "EBIC",
  "q_active_common", "mu_pairwise_cos_mean", "kappa_true_ratio"
)])
