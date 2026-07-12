#!/usr/bin/env Rscript

# Post-fit interpretation of the exact E-CGL solutions at K=3 and K=10.
# Supplied Classic3 labels are used only after fitting for evaluation and
# component naming. They are not used for K or support selection.

options(stringsAsFactors = FALSE)

if (!requireNamespace("Matrix", quietly = TRUE)) stop("Matrix is required.")
if (!requireNamespace("ggplot2", quietly = TRUE)) stop("ggplot2 is required.")
suppressPackageStartupMessages(library(Matrix))

test_path <- file.path(
  "data", "classic3", "processed",
  "classic3_splade_holdout_test_top2000_260711.rds"
)
fit_path <- function(K) file.path(
  "results", sprintf("classic3_ecgl_k%d_exact_panel_260714", K),
  sprintf("classic3_ecgl_k%d_exact_panel_selected_fits.rds", K)
)
comparison_path <- file.path(
  "results", "classic3_k_selection_panel_final_b20_260712",
  "classic3_ecgl_exact_bic_k_comparison.csv"
)
out_dir <- file.path(
  "results", "classic3_k_component_interpretation_260712"
)
figure_path <- file.path(
  "docs", "manuscript", "figures",
  "classic3_k3_k10_label_component_heatmap_260712.png"
)

for (path in c(test_path, fit_path(3L), fit_path(10L), comparison_path)) {
  if (!file.exists(path)) stop("Missing input: ", path)
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(figure_path), recursive = TRUE, showWarnings = FALSE)

source_no_bom <- function(path, envir = .GlobalEnv) {
  txt <- readLines(path, warn = FALSE, encoding = "UTF-8")
  if (length(txt) > 0L) txt[1L] <- sub("^\ufeff", "", txt[1L])
  eval(parse(text = txt), envir = envir)
}
source_no_bom(file.path(
  "r", "realdata", "exact_centered_refit_helpers_260711.r"
))

test <- readRDS(test_path)
X_test <- test$X
if (!inherits(X_test, "sparseMatrix")) {
  X_test <- Matrix::Matrix(X_test, sparse = TRUE)
}
truth <- as.integer(test$y)
class_names <- as.character(test$categories)
tokens <- as.character(test$vocab)
n_test <- nrow(X_test)
d <- ncol(X_test)

row_logsumexp <- function(scores) {
  maxima <- apply(scores, 1L, max)
  maxima + log(rowSums(exp(scores - maxima)))
}

evaluate_fit <- function(fit) {
  K <- nrow(fit$mu)
  kappa <- rep(as.numeric(fit$kappa), length.out = K)
  alpha <- pmax(as.numeric(fit$alpha), 1e-15)
  alpha <- alpha / sum(alpha)
  linear <- as.matrix(X_test %*% t(fit$mu))
  linear <- sweep(linear, 2L, kappa, "*")
  scores <- sweep(
    linear, 2L,
    log(alpha) + exact_refit_log_vmf_const(kappa, d), "+"
  )
  log_norm <- row_logsumexp(scores)
  list(
    cluster = max.col(scores, ties.method = "first"),
    NLL_per_doc = -mean(log_norm)
  )
}

adjusted_rand_index <- function(truth, cluster) {
  tab <- table(truth, cluster)
  choose2 <- function(x) x * (x - 1) / 2
  n <- sum(tab)
  index <- sum(choose2(tab))
  row_index <- sum(choose2(rowSums(tab)))
  col_index <- sum(choose2(colSums(tab)))
  expected <- row_index * col_index / choose2(n)
  maximum <- (row_index + col_index) / 2
  if (abs(maximum - expected) < .Machine$double.eps) return(1)
  (index - expected) / (maximum - expected)
}

make_component_profile <- function(K, fit, predicted) {
  counts <- table(
    factor(truth, levels = seq_along(class_names), labels = class_names),
    factor(predicted, levels = seq_len(K))
  )
  majority_index <- apply(counts, 2L, which.max)
  majority <- class_names[majority_index]
  component_n <- colSums(counts)
  purity <- apply(counts, 2L, max) / pmax(component_n, 1L)
  order_index <- order(match(majority, class_names), -component_n, seq_len(K))
  within <- integer(K)
  for (name in class_names) {
    ids <- order_index[majority[order_index] == name]
    if (length(ids)) within[ids] <- seq_along(ids)
  }
  display <- paste0(majority, "-", within)

  component <- data.frame(
    K = K, original_component = seq_len(K), component = display,
    majority_class = majority, test_n = component_n, purity = purity,
    alpha = as.numeric(fit$alpha),
    kappa = rep(as.numeric(fit$kappa), length.out = K),
    display_order = match(seq_len(K), order_index),
    stringsAsFactors = FALSE
  )
  for (i in seq_along(class_names)) {
    component[[paste0("n_", class_names[i])]] <- as.integer(counts[i, ])
  }

  count_rows <- do.call(rbind, lapply(seq_len(K), function(k) {
    data.frame(
      K = K, original_component = k, component = display[k],
      true_class = class_names, count = as.integer(counts[, k]),
      class_total = as.integer(rowSums(counts)),
      row_prop = as.numeric(counts[, k] / rowSums(counts)),
      component_prop = as.numeric(counts[, k] / pmax(component_n[k], 1L)),
      display_order = component$display_order[k],
      stringsAsFactors = FALSE
    )
  }))

  eta <- sweep(
    fit$mu, 1L, rep(as.numeric(fit$kappa), length.out = K), "*"
  )
  centered <- sweep(eta, 2L, colMeans(eta), "-")
  active <- if (!is.null(fit$active) && length(fit$active) == d) {
    as.logical(fit$active)
  } else {
    sqrt(colSums(centered * centered)) > 1e-8
  }
  top_n <- 5L
  token_rows <- do.call(rbind, lapply(seq_len(K), function(k) {
    eligible <- which(active & is.finite(centered[k, ]))
    positive <- head(eligible[order(centered[k, eligible], decreasing = TRUE)], top_n)
    negative <- head(eligible[order(centered[k, eligible], decreasing = FALSE)], top_n)
    rbind(
      data.frame(
        K = K, original_component = k, component = display[k],
        majority_class = majority[k], direction = "positive",
        rank = seq_along(positive), feature_id = positive,
        token = tokens[positive], contrast = centered[k, positive],
        stringsAsFactors = FALSE
      ),
      data.frame(
        K = K, original_component = k, component = display[k],
        majority_class = majority[k], direction = "negative",
        rank = seq_along(negative), feature_id = negative,
        token = tokens[negative], contrast = centered[k, negative],
        stringsAsFactors = FALSE
      )
    )
  }))

  list(component = component, counts = count_rows, tokens = token_rows)
}

profiles <- list()
fit_store <- list()
score_rows <- list()
for (K in c(3L, 10L)) {
  fits <- readRDS(fit_path(K))
  fit <- fits[["E-CGL__BIC"]]
  if (is.null(fit)) stop("Missing E-CGL__BIC fit for K=", K)
  evaluated <- evaluate_fit(fit)
  profiles[[as.character(K)]] <- make_component_profile(
    K, fit, evaluated$cluster
  )
  fit_store[[as.character(K)]] <- fit
  score_rows[[as.character(K)]] <- data.frame(
    K = K, test_NLL_per_doc = evaluated$NLL_per_doc,
    test_ARI = adjusted_rand_index(truth, evaluated$cluster),
    selected_q = sum(fit$active),
    stringsAsFactors = FALSE
  )
}

component <- do.call(rbind, lapply(profiles, `[[`, "component"))
counts <- do.call(rbind, lapply(profiles, `[[`, "counts"))
token_profile <- do.call(rbind, lapply(profiles, `[[`, "tokens"))
scores <- do.call(rbind, score_rows)

component <- component[order(component$K, component$display_order), ]
counts <- counts[order(counts$K, counts$display_order, counts$true_class), ]
token_profile <- token_profile[
  order(token_profile$K, match(token_profile$component, component$component),
        token_profile$direction, token_profile$rank),
]

k10_tokens <- token_profile[token_profile$K == 10L, , drop = FALSE]
token_summary <- do.call(rbind, lapply(
  component$component[component$K == 10L], function(name) {
    z <- k10_tokens[k10_tokens$component == name, , drop = FALSE]
    pos <- z[z$direction == "positive" & z$rank <= 3L, ]
    neg <- z[z$direction == "negative" & z$rank <= 3L, ]
    meta <- component[component$K == 10L & component$component == name, ]
    data.frame(
      component = name, majority_class = meta$majority_class,
      test_n = meta$test_n, purity = meta$purity,
      alpha = meta$alpha, kappa = meta$kappa,
      positive_tokens = paste(
        sprintf("%s (%+.1f)", pos$token, pos$contrast), collapse = "; "
      ),
      negative_tokens = paste(
        sprintf("%s (%+.1f)", neg$token, neg$contrast), collapse = "; "
      ),
      stringsAsFactors = FALSE
    )
  }
))

comparison <- read.csv(comparison_path, stringsAsFactors = FALSE)
for (K in c(3L, 10L)) {
  expected <- comparison[comparison$K == K, , drop = FALSE]
  observed <- scores[scores$K == K, , drop = FALSE]
  if (nrow(expected) != 1L || nrow(observed) != 1L) {
    stop("Missing unique comparison row for K=", K)
  }
  if (abs(expected$test_NLL_per_doc - observed$test_NLL_per_doc) > 1e-8 ||
      abs(expected$test_ARI - observed$test_ARI) > 1e-10 ||
      expected$selected_q != observed$selected_q) {
    stop("Comparison reproduction failed for K=", K)
  }
}

utils::write.csv(
  component, file.path(out_dir, "classic3_k_component_summary.csv"),
  row.names = FALSE
)
utils::write.csv(
  counts, file.path(out_dir, "classic3_k_label_component_counts.csv"),
  row.names = FALSE
)
utils::write.csv(
  token_profile, file.path(out_dir, "classic3_k_component_top_tokens.csv"),
  row.names = FALSE
)
utils::write.csv(
  token_summary, file.path(out_dir, "classic3_k10_component_token_summary.csv"),
  row.names = FALSE
)
utils::write.csv(
  scores, file.path(out_dir, "classic3_k_component_score_audit.csv"),
  row.names = FALSE
)

plot_data <- counts
plot_data$K_label <- factor(
  paste0("K = ", plot_data$K), levels = c("K = 3", "K = 10")
)
ordered_keys <- unlist(lapply(c(3L, 10L), function(K) {
  z <- component[component$K == K, ]
  paste0("K", K, "__", z$component)
}))
plot_data$component_key <- factor(
  paste0("K", plot_data$K, "__", plot_data$component),
  levels = ordered_keys
)
plot_data$true_class <- factor(plot_data$true_class, levels = rev(class_names))
plot_data$label <- ifelse(
  plot_data$count > 0,
  sprintf("%d\n%.1f%%", plot_data$count, 100 * plot_data$row_prop), ""
)

p <- ggplot2::ggplot(
  plot_data,
  ggplot2::aes(x = component_key, y = true_class, fill = row_prop)
) +
  ggplot2::geom_tile(color = "white", linewidth = 0.6) +
  ggplot2::geom_text(ggplot2::aes(label = label), size = 3.0) +
  ggplot2::facet_grid(. ~ K_label, scales = "free_x", space = "free_x") +
  ggplot2::scale_x_discrete(labels = function(x) sub("^K[0-9]+__", "", x)) +
  ggplot2::scale_fill_gradient(
    low = "#F7F8FA", high = "#176B87", limits = c(0, 1),
    name = "P(component | class)"
  ) +
  ggplot2::labs(
    x = "Post-hoc component name", y = "Supplied broad topic",
    title = "Classic3 E-CGL: broad topics and fitted components",
    subtitle = "Cell labels are test count and row percentage; labels were not used for fitting"
  ) +
  ggplot2::theme_minimal(base_size = 11) +
  ggplot2::theme(
    panel.grid = ggplot2::element_blank(),
    strip.text = ggplot2::element_text(face = "bold"),
    axis.text.x = ggplot2::element_text(angle = 45, hjust = 1),
    plot.title = ggplot2::element_text(face = "bold"),
    legend.position = "right"
  )
ggplot2::ggsave(figure_path, p, width = 12.0, height = 4.2, dpi = 180)

fmt <- function(x, digits = 3L) {
  formatC(as.numeric(x), format = "f", digits = digits)
}
notes <- c(
  "# Classic3 K=3 versus K=10 component interpretation",
  "",
  "- Exact E-CGL BIC-after-refit fits are reused; no model is refitted.",
  "- Supplied labels are used only for post-fit evaluation and component naming.",
  sprintf("- Test documents: %d; vocabulary coordinates: %d.", n_test, d),
  "",
  "## K=10 component profile",
  "",
  "| component | majority topic | test n | purity | positive centered-Eta tokens |",
  "|---|---|---:|---:|---|"
)
for (i in seq_len(nrow(token_summary))) {
  notes <- c(notes, sprintf(
    "| %s | %s | %d | %s | %s |",
    token_summary$component[i], token_summary$majority_class[i],
    token_summary$test_n[i], fmt(token_summary$purity[i]),
    token_summary$positive_tokens[i]
  ))
}
notes <- c(
  notes, "", "## Interpretation boundary", "",
  "- K=10 has high component purity but divides each supplied broad topic into multiple components.",
  "- Post-hoc component names do not imply supervised fitting.",
  "- Token contrasts describe relative component scores, not absolute word preferences.",
  "- K=3 remains the externally defined broad-topic benchmark; K=10 is a finer density resolution."
)
writeLines(
  notes, file.path(out_dir, "classic3_k_component_interpretation_notes.md"),
  useBytes = TRUE
)

cat("Saved: ", normalizePath(out_dir, winslash = "/"), "\n", sep = "")
cat("Figure: ", normalizePath(figure_path, winslash = "/"), "\n", sep = "")
print(component[component$K == 10L, ], row.names = FALSE)
