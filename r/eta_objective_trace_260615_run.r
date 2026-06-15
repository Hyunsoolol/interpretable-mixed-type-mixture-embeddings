# Eta centered penalty objective trace smoke test
# This script does not change the official fitting functions. It wraps the
# current proximal EM-type update and records the penalized objective path.

options(stringsAsFactors = FALSE)
set.seed(20260615)

read_r_bom_safe <- function(path) {
  txt <- readLines(path, warn = FALSE, encoding = "UTF-8")
  if (length(txt) > 0) txt[1] <- sub("^\\ufeff", "", txt[1])
  txt
}

source_until_before <- function(path, marker = NULL, skip_source_rossi = FALSE) {
  txt <- read_r_bom_safe(path)
  if (!is.null(marker)) {
    idx <- grep(marker, txt, fixed = TRUE)
    if (length(idx) > 0) txt <- txt[seq_len(idx[1] - 1L)]
  }
  if (skip_source_rossi) {
    txt <- txt[!grepl("rossi_barbaro_2022_reproduction", txt, fixed = TRUE)]
  }
  eval(parse(text = txt), envir = .GlobalEnv)
}

source_until_before("r/rossi_barbaro_2022_reproduction.r", marker = "# Script entry point")
source_until_before("r/rb2022_k4_pilot_compare_run.r", marker = "run_one <- function", skip_source_rossi = TRUE)

make_specific_effect_params <- function(d, K, common_q, specific_q,
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

out_dir <- "results/eta_objective_trace_260615"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cfg <- list(
  K = 4,
  n = 300,
  d = 100,
  common_q = 6,
  specific_q = 4,
  specific_weight = 0.5,
  kappa = c(40, 50, 60, 70),
  nstart = 5,
  max_iter = 80,
  tol = 1e-7,
  lambda_candidates = NA_real_
)

simulate_specific <- function(cfg) {
  params <- make_specific_effect_params(
    d = cfg$d,
    K = cfg$K,
    common_q = cfg$common_q,
    specific_q = cfg$specific_q,
    specific_weight = cfg$specific_weight,
    kappa = cfg$kappa
  )
  z <- rep(seq_len(cfg$K), length.out = cfg$n)
  z <- sample(z)
  X <- matrix(0, cfg$n, cfg$d)
  for (k in seq_len(cfg$K)) {
    idx <- which(z == k)
    X[idx, ] <- rvMF(length(idx), params$mu[k, ], params$kappa[k])
  }
  list(X = X, z = z, params = params)
}

fit_eta_centered_em_trace <- function(X, K, lambda_eta, init,
                                      max_iter = 80, tol = 1e-7) {
  theta <- init
  theta$mu <- normalize_rows(theta$mu)
  theta$alpha <- pmax(theta$alpha, 1e-12)
  theta$alpha <- theta$alpha / sum(theta$alpha)
  if (length(theta$kappa) == 1) theta$kappa <- rep(theta$kappa, K)

  trace <- data.frame()
  prev_obj <- NA_real_
  last_e <- NULL

  for (iter in seq_len(max_iter)) {
    e <- e_step_vmf(X, theta)
    mstep <- unpenalized_eta_mstep(X, e$tau)
    eta_new <- prox_eta_centered(mstep$eta, lambda_eta)
    theta_new <- eta_to_theta(mstep$alpha, eta_new, fallback_mu = theta$mu)
    e_new <- e_step_vmf(X, theta_new)
    penalty <- sum(sqrt(colSums(center_eta(eta_matrix(theta_new))^2)))
    obj <- e_new$loglik - lambda_eta * penalty
    diff <- if (is.na(prev_obj)) NA_real_ else obj - prev_obj

    trace <- rbind(trace, data.frame(
      lambda_eta = lambda_eta,
      iter = iter,
      loglik = e_new$loglik,
      penalty = penalty,
      objective = obj,
      objective_diff = diff,
      decreased = ifelse(is.na(diff), FALSE, diff < -1e-8)
    ))

    theta <- theta_new
    last_e <- e_new
    if (!is.na(prev_obj) && abs(obj - prev_obj) / max(1, abs(prev_obj)) < tol) {
      theta$converged <- TRUE
      theta$iter <- iter
      theta$loglik <- e_new$loglik
      theta$pen_loglik <- obj
      theta$tau <- e_new$tau
      theta$trace <- trace
      return(theta)
    }
    prev_obj <- obj
  }

  if (is.null(last_e)) last_e <- e_step_vmf(X, theta)
  penalty <- sum(sqrt(colSums(center_eta(eta_matrix(theta))^2)))
  theta$converged <- FALSE
  theta$iter <- max_iter
  theta$loglik <- last_e$loglik
  theta$pen_loglik <- last_e$loglik - lambda_eta * penalty
  theta$tau <- last_e$tau
  theta$trace <- trace
  theta
}

dat <- simulate_specific(cfg)
X <- dat$X
params <- dat$params

# Dense initialization, then take a small number of path-like lambda candidates.
dense <- fit_svMF_multistart(X, cfg$K, beta = 0, nstart = cfg$nstart, max_iter = cfg$max_iter)
e0 <- e_step_vmf(X, dense)
mstep0 <- unpenalized_eta_mstep(X, e0$tau)
thresholds <- sort(unique(round(sqrt(colSums(center_eta(mstep0$eta)^2)), 10)))
thresholds <- thresholds[is.finite(thresholds) & thresholds > 1e-10]
if (length(thresholds) < 4) {
  lambda_grid <- c(0, thresholds)
} else {
  idx <- unique(pmax(1, pmin(length(thresholds), round(seq(1, length(thresholds), length.out = 4)))))
  lambda_grid <- c(0, thresholds[idx])
}

all_trace <- list()
summary_rows <- list()
init <- dense
for (lambda_eta in lambda_grid) {
  fit <- fit_eta_centered_em_trace(
    X, cfg$K, lambda_eta = lambda_eta, init = init,
    max_iter = cfg$max_iter, tol = cfg$tol
  )
  tr <- fit$trace
  all_trace[[length(all_trace) + 1L]] <- tr
  summary_rows[[length(summary_rows) + 1L]] <- data.frame(
    lambda_eta = lambda_eta,
    iterations = nrow(tr),
    converged = fit$converged,
    n_decrease = sum(tr$decreased, na.rm = TRUE),
    min_objective_diff = suppressWarnings(min(tr$objective_diff, na.rm = TRUE)),
    first_objective = tr$objective[1],
    last_objective = tr$objective[nrow(tr)],
    final_loglik = fit$loglik,
    final_penalty = tail(tr$penalty, 1),
    active_q = sum(active_eta_centered(fit))
  )
  init <- fit
}

trace_df <- do.call(rbind, all_trace)
summary_df <- do.call(rbind, summary_rows)
summary_df$has_decrease <- summary_df$n_decrease > 0

write.csv(trace_df, file.path(out_dir, "eta_objective_trace_raw.csv"), row.names = FALSE)
write.csv(summary_df, file.path(out_dir, "eta_objective_trace_summary.csv"), row.names = FALSE)

md <- c(
  "# Eta Objective Trace Smoke Test 260615",
  "",
  "## Setting",
  "",
  sprintf("- K = %d", cfg$K),
  sprintf("- n = %d", cfg$n),
  sprintf("- d = %d", cfg$d),
  sprintf("- common variables = %d", cfg$common_q),
  sprintf("- component-specific variables = %d per component", cfg$specific_q),
  sprintf("- w = %.2f", cfg$specific_weight),
  sprintf("- kappa = (%s)", paste(cfg$kappa, collapse = ", ")),
  sprintf("- random start = %d", cfg$nstart),
  "- update = unpenalized eta M-step followed by centered eta proximal shrinkage",
  "",
  "## Summary",
  "",
  "| lambda_eta | iter | converged | active q | decreases | min objective diff | first objective | last objective |",
  "|---:|---:|:---:|---:|---:|---:|---:|---:|"
)
for (i in seq_len(nrow(summary_df))) {
  md <- c(md, sprintf(
    "| %.6g | %d | %s | %d | %d | %.6g | %.6f | %.6f |",
    summary_df$lambda_eta[i], summary_df$iterations[i],
    ifelse(summary_df$converged[i], "yes", "no"),
    summary_df$active_q[i], summary_df$n_decrease[i],
    summary_df$min_objective_diff[i], summary_df$first_objective[i],
    summary_df$last_objective[i]
  ))
}

if (any(summary_df$has_decrease)) {
  md <- c(md, "", "## Note", "", "At least one lambda path candidate had a decreasing penalized objective step. The current update should be described as a proximal EM-type heuristic, not a guaranteed monotone EM algorithm. A line-search or MM safeguard is recommended for a paper-grade algorithm.")
} else {
  md <- c(md, "", "## Note", "", "No objective decrease was observed in this smoke test. This is only an empirical check, not a proof of monotonicity. The method should still be described as a proximal EM-type update unless a monotonicity safeguard/proof is added.")
}

writeLines(md, file.path(out_dir, "eta_objective_trace_summary.md"), useBytes = TRUE)
cat(paste(md, collapse = "\n"))
