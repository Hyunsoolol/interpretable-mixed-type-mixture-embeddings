#!/usr/bin/env Rscript

# Summarize the Classic3 K-selection panel without using labels for selection.
# Reference labels are attached only to report ARI/NMI after each fit is fixed.

options(stringsAsFactors = FALSE)

if (!requireNamespace("Matrix", quietly = TRUE)) stop("Matrix is required.")
if (!requireNamespace("ggplot2", quietly = TRUE)) stop("ggplot2 is required.")
suppressPackageStartupMessages(library(Matrix))

train_path <- file.path(
  "data", "classic3", "processed",
  "classic3_splade_holdout_train_top2000_260711.rds"
)
test_path <- file.path(
  "data", "classic3", "processed",
  "classic3_splade_holdout_test_top2000_260711.rds"
)
panel_dir <- file.path("results", "classic3_k_selection_panel_b10_inbag_260714")
out_dir <- file.path("results", "classic3_k_selection_panel_final_260714")
figure_dir <- file.path("docs", "manuscript", "figures")
K_values <- c(3L, 7L, 8L, 10L)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

source_no_bom <- function(path, envir = .GlobalEnv) {
  txt <- readLines(path, warn = FALSE, encoding = "UTF-8")
  if (length(txt) > 0L) txt[1L] <- sub("^\ufeff", "", txt[1L])
  eval(parse(text = txt), envir = envir)
}
source_no_bom(file.path("r", "realdata", "exact_centered_refit_helpers_260711.r"))

train <- readRDS(train_path)
test <- readRDS(test_path)
X_train <- train$X
X_test <- test$X
if (!inherits(X_train, "sparseMatrix")) X_train <- Matrix(X_train, sparse = TRUE)
if (!inherits(X_test, "sparseMatrix")) X_test <- Matrix(X_test, sparse = TRUE)
y_train <- as.integer(train$y)
y_test <- as.integer(test$y)
tokens <- as.character(train$vocab)
d <- ncol(X_train)

row_logsumexp <- function(scores) {
  maxima <- apply(scores, 1L, max)
  maxima + log(rowSums(exp(scores - maxima)))
}

evaluate_fit <- function(X, fit) {
  K <- nrow(fit$mu)
  kappa <- rep(as.numeric(fit$kappa), length.out = K)
  alpha <- pmax(as.numeric(fit$alpha), 1e-15)
  alpha <- alpha / sum(alpha)
  linear <- as.matrix(X %*% t(fit$mu))
  linear <- sweep(linear, 2L, kappa, "*")
  scores <- sweep(
    linear, 2L,
    log(alpha) + exact_refit_log_vmf_const(kappa, ncol(X)), "+"
  )
  log_norm <- row_logsumexp(scores)
  tau <- exp(scores - log_norm)
  cluster <- max.col(scores, ties.method = "first")
  size <- as.integer(table(factor(cluster, levels = seq_len(K))))
  list(
    loglik = sum(log_norm),
    NLL_per_doc = -mean(log_norm),
    tau = tau,
    cluster = cluster,
    cluster_size = size,
    min_cluster_n = min(size),
    min_cluster_prop = min(size) / nrow(X),
    normalized_entropy = -sum(tau * log(pmax(tau, 1e-15))) /
      (nrow(X) * log(K))
  )
}

adjusted_rand_index <- function(truth, cluster) {
  tab <- table(truth, cluster)
  choose2 <- function(x) x * (x - 1) / 2
  n <- sum(tab)
  if (n < 2L) return(NA_real_)
  index <- sum(choose2(tab))
  row_index <- sum(choose2(rowSums(tab)))
  col_index <- sum(choose2(colSums(tab)))
  expected <- row_index * col_index / choose2(n)
  maximum <- (row_index + col_index) / 2
  if (abs(maximum - expected) < .Machine$double.eps) return(1)
  (index - expected) / (maximum - expected)
}

normalized_mutual_information <- function(truth, cluster) {
  tab <- table(truth, cluster)
  p <- tab / sum(tab)
  pi <- rowSums(p)
  pj <- colSums(p)
  expected <- outer(pi, pj)
  keep <- p > 0 & expected > 0
  mi <- sum(p[keep] * log(p[keep] / expected[keep]))
  h1 <- -sum(pi[pi > 0] * log(pi[pi > 0]))
  h2 <- -sum(pj[pj > 0] * log(pj[pj > 0]))
  if (h1 + h2 <= 0) return(1)
  2 * mi / (h1 + h2)
}

external_partition_scores <- function(truth, cluster) {
  tab <- table(truth, cluster)
  p <- tab / sum(tab)
  pi <- rowSums(p)
  pj <- colSums(p)
  expected <- outer(pi, pj)
  keep <- p > 0 & expected > 0
  mi <- sum(p[keep] * log(p[keep] / expected[keep]))
  h_truth <- -sum(pi[pi > 0] * log(pi[pi > 0]))
  h_cluster <- -sum(pj[pj > 0] * log(pj[pj > 0]))
  c(
    purity = sum(apply(tab, 2L, max)) / sum(tab),
    homogeneity = if (h_truth <= 0) 1 else mi / h_truth,
    completeness = if (h_cluster <= 0) 1 else mi / h_cluster
  )
}

npmi_for_tokens <- function(X, ids) {
  ids <- unique(as.integer(ids))
  if (length(ids) < 2L) return(NA_real_)
  present <- X[, ids, drop = FALSE] > 0
  p <- Matrix::colSums(present) / nrow(X)
  joint <- as.matrix(Matrix::crossprod(present)) / nrow(X)
  pairs <- which(upper.tri(joint), arr.ind = TRUE)
  values <- vapply(seq_len(nrow(pairs)), function(i) {
    a <- pairs[i, 1L]
    b <- pairs[i, 2L]
    pxy <- joint[a, b]
    if (!is.finite(pxy) || pxy <= 0) return(-1)
    if (pxy >= 1 - 1e-15) return(1)
    log(pxy / max(p[a] * p[b], 1e-15)) / (-log(pxy))
  }, numeric(1))
  mean(values[is.finite(values)])
}

fit_coherence <- function(fit, top_n = 10L) {
  K <- nrow(fit$mu)
  eta <- sweep(fit$mu, 1L, rep(as.numeric(fit$kappa), length.out = K), "*")
  centered <- sweep(eta, 2L, colMeans(eta), "-")
  active <- sqrt(colSums(centered * centered)) > 1e-8
  values <- vapply(seq_len(K), function(k) {
    eligible <- which(active & is.finite(centered[k, ]) & centered[k, ] > 0)
    if (length(eligible) < top_n) eligible <- which(active & is.finite(centered[k, ]))
    top <- head(eligible[order(centered[k, eligible], decreasing = TRUE)], top_n)
    npmi_for_tokens(X_train, top)
  }, numeric(1))
  mean(values, na.rm = TRUE)
}

selection_rows <- list()
for (K in K_values) {
  prefix <- sprintf("classic3_ecgl_k%d_exact_panel", K)
  exact_dir <- file.path("results", sprintf("classic3_ecgl_k%d_exact_panel_260714", K))
  selection_path <- file.path(exact_dir, paste0(prefix, "_selection.csv"))
  fit_path <- file.path(exact_dir, paste0(prefix, "_selected_fits.rds"))
  if (!file.exists(selection_path) || !file.exists(fit_path)) {
    stop("Missing exact E-CGL result for K=", K)
  }
  selection <- utils::read.csv(selection_path, check.names = FALSE)
  selected <- selection[selection$criterion == "BIC", , drop = FALSE]
  if (nrow(selected) != 1L) stop("Expected one BIC row for K=", K)
  fits <- readRDS(fit_path)
  fit <- fits[["E-CGL__BIC"]]
  if (is.null(fit)) stop("Missing E-CGL__BIC fit for K=", K)
  train_eval <- evaluate_fit(X_train, fit)
  test_eval <- evaluate_fit(X_test, fit)
  external <- external_partition_scores(y_test, test_eval$cluster)
  eta <- sweep(
    fit$mu, 1L,
    rep(as.numeric(fit$kappa), length.out = K), "*"
  )
  centered <- sweep(eta, 2L, colMeans(eta), "-")
  selected_q_recomputed <- sum(sqrt(colSums(centered * centered)) > 1e-8)
  selection_rows[[length(selection_rows) + 1L]] <- data.frame(
    K = K,
    selector = "exact BIC after refit",
    selected_q = as.integer(selected$selected_q),
    support_prop = as.numeric(selected$selected_q) / d,
    train_NLL_per_doc = train_eval$NLL_per_doc,
    test_NLL_per_doc = test_eval$NLL_per_doc,
    train_ARI = adjusted_rand_index(y_train, train_eval$cluster),
    test_ARI = adjusted_rand_index(y_test, test_eval$cluster),
    train_NMI = normalized_mutual_information(y_train, train_eval$cluster),
    test_NMI = normalized_mutual_information(y_test, test_eval$cluster),
    test_purity = unname(external["purity"]),
    test_homogeneity = unname(external["homogeneity"]),
    test_completeness = unname(external["completeness"]),
    train_min_cluster_n = train_eval$min_cluster_n,
    train_min_cluster_prop = train_eval$min_cluster_prop,
    test_min_cluster_n = test_eval$min_cluster_n,
    test_min_cluster_prop = test_eval$min_cluster_prop,
    normalized_entropy = train_eval$normalized_entropy,
    mean_component_NPMI = fit_coherence(fit),
    train_cluster_size = paste(train_eval$cluster_size, collapse = ";"),
    test_cluster_size = paste(test_eval$cluster_size, collapse = ";"),
    selected_q_recomputed = selected_q_recomputed,
    support_count_identical = selected_q_recomputed == as.integer(selected$selected_q),
    train_tau_row_sum_error = max(abs(rowSums(train_eval$tau) - 1)),
    test_tau_row_sum_error = max(abs(rowSums(test_eval$tau) - 1)),
    converged = isTRUE(selected$converged),
    labels_used_for_selection = FALSE,
    stringsAsFactors = FALSE
  )
}
ecgl <- do.call(rbind, selection_rows)

panel <- utils::read.csv(
  file.path(panel_dir, "classic3_k_panel_summary.csv"), check.names = FALSE
)
criteria <- utils::read.csv(
  file.path(panel_dir, "classic3_k_selection_by_criterion.csv"),
  check.names = FALSE
)

utils::write.csv(
  ecgl,
  file.path(out_dir, "classic3_ecgl_exact_bic_k_comparison.csv"),
  row.names = FALSE
)
utils::write.csv(
  criteria,
  file.path(out_dir, "classic3_dense_k_selection_by_criterion.csv"),
  row.names = FALSE
)

panel_audit <- utils::read.csv(
  file.path(panel_dir, "classic3_k_panel_audit.csv"), check.names = FALSE
)
audit <- data.frame(
  check = c(
    "all_dense_panel_checks_pass",
    "all_ecgl_exact_fits_converged",
    "all_ecgl_support_counts_match",
    "all_train_tau_rows_sum_to_one",
    "all_test_tau_rows_sum_to_one",
    "labels_excluded_from_selection"
  ),
  pass = c(
    all(panel_audit$pass),
    all(ecgl$converged),
    all(ecgl$support_count_identical),
    max(ecgl$train_tau_row_sum_error) < 1e-10,
    max(ecgl$test_tau_row_sum_error) < 1e-10,
    all(!ecgl$labels_used_for_selection)
  ),
  stringsAsFactors = FALSE
)
utils::write.csv(
  audit,
  file.path(out_dir, "classic3_k_selection_panel_audit.csv"),
  row.names = FALSE
)

plot_data <- rbind(
  data.frame(
    K = panel$K,
    kappa_model = panel$kappa_model,
    metric = "OOB NLL per document: mean +/- SE (lower is better)",
    value = panel$OOB_NLL_mean,
    lower = panel$OOB_NLL_mean - panel$OOB_NLL_se,
    upper = panel$OOB_NLL_mean + panel$OOB_NLL_se
  ),
  data.frame(
    K = panel$K,
    kappa_model = panel$kappa_model,
    metric = "Bootstrap pairwise ARI: mean +/- SD (higher is better)",
    value = panel$pairwise_stability_mean,
    lower = pmax(0, panel$pairwise_stability_mean - panel$pairwise_stability_sd),
    upper = pmin(1, panel$pairwise_stability_mean + panel$pairwise_stability_sd)
  )
)
plot_data$metric <- factor(
  plot_data$metric,
  levels = c(
    "OOB NLL per document: mean +/- SE (lower is better)",
    "Bootstrap pairwise ARI: mean +/- SD (higher is better)"
  )
)
plot_data$kappa_model <- factor(
  plot_data$kappa_model, levels = c("shared", "free"),
  labels = c("Shared kappa", "Free kappa")
)

p <- ggplot2::ggplot(
  plot_data,
  ggplot2::aes(x = K, y = value, color = kappa_model, shape = kappa_model)
) +
  ggplot2::geom_vline(xintercept = 3, color = "grey70", linewidth = 0.5) +
  ggplot2::geom_errorbar(
    ggplot2::aes(ymin = lower, ymax = upper), width = 0.15, linewidth = 0.45
  ) +
  ggplot2::geom_point(size = 2.4) +
  ggplot2::facet_wrap(~metric, ncol = 1, scales = "free_y") +
  ggplot2::scale_x_continuous(breaks = 2:10) +
  ggplot2::scale_color_manual(values = c("#C45A32", "#167D88")) +
  ggplot2::labs(x = "Number of components K", y = NULL, color = NULL, shape = NULL) +
  ggplot2::theme_minimal(base_size = 11) +
  ggplot2::theme(
    panel.grid.minor = ggplot2::element_blank(),
    strip.text = ggplot2::element_text(face = "bold"),
    legend.position = "top",
    axis.title.x = ggplot2::element_text(margin = ggplot2::margin(t = 8))
  )

figure_path <- file.path(figure_dir, "classic3_k_selection_diagnostics_260714.png")
ggplot2::ggsave(figure_path, p, width = 7.4, height = 6.0, dpi = 180)

fmt <- function(x, digits = 4L) formatC(x, format = "f", digits = digits)
dense_selected <- criteria[
  criteria$criterion %in% c(
    "BIC", "RICc", "EBIC_g0.5", "EBIC_g1",
    "bootstrap_OOB_NLL_1SE", "bootstrap_pairwise_stability"
  ),
  c("kappa_model", "criterion", "selected_K", "labels_used"),
  drop = FALSE
]

notes <- c(
  "# Classic3 K-selection panel",
  "",
  "- Selection diagnostics are label-free; labels are used only for post-fit ARI/NMI.",
  "- Bootstrap results use B=10 and are an exploratory stability diagnostic.",
  "- Bootstrap fits use in-bag-only initializations. The SPLADE top-2000 representation remains fixed from the full training split.",
  "- An earlier full-train-initialized bootstrap diagnostic is superseded and is not used here.",
  "- Dense-vMF likelihood/IC and out-of-bag density criteria favor finer partitions near the upper K boundary.",
  "- Bootstrap partition stability peaks at K=3 for both shared- and free-kappa dense vMF fits.",
  "- E-CGL is fitted separately at K=3, 7, 8, and 10; support is selected by exact BIC after fixed-support refit.",
  "",
  "## Dense K choices",
  "",
  "| kappa model | criterion | selected K | labels used |",
  "|---|---|---:|---|",
  apply(dense_selected, 1L, function(z) {
    paste0("| ", z[[1]], " | ", z[[2]], " | ", z[[3]], " | ", z[[4]], " |")
  }),
  "",
  "## E-CGL exact-BIC comparison",
  "",
  "| K | selected q | test NLL/doc | test ARI | purity | homogeneity | completeness |",
  "|---:|---:|---:|---:|---:|---:|---:|",
  vapply(seq_len(nrow(ecgl)), function(i) {
    paste0(
      "| ", ecgl$K[i], " | ", ecgl$selected_q[i], " | ",
      fmt(ecgl$test_NLL_per_doc[i]), " | ", fmt(ecgl$test_ARI[i]), " | ",
      fmt(ecgl$test_purity[i]), " | ", fmt(ecgl$test_homogeneity[i]), " | ",
      fmt(ecgl$test_completeness[i]), " |"
    )
  }, character(1)),
  "",
  "## Interpretation",
  "",
  "The primary Classic3 benchmark is fixed at the three externally supplied topic categories. Exploratory in-bag bootstrap stability is highest at K=3, whereas likelihood-based criteria continue to reward finer partitions near the upper K boundary. E-CGL is therefore interpreted as conditional support selection at a fixed K rather than a method for selecting K itself.",
  "",
  "Selected q is not interpreted as a cross-K sparsity ranking because the active-coordinate degrees-of-freedom cost changes with K."
)
writeLines(
  notes,
  file.path(out_dir, "classic3_k_selection_panel_notes.md"),
  useBytes = TRUE
)

cat("Saved: ", normalizePath(out_dir, winslash = "/"), "\n", sep = "")
cat("Figure: ", normalizePath(figure_path, winslash = "/"), "\n", sep = "")
print(ecgl, row.names = FALSE)
