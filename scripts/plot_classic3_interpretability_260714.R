#!/usr/bin/env Rscript

# Create a paper-facing interpretability profile from the final Classic3
# held-out E-CGL exact-BIC fit. Supplied labels are used only to name fitted
# components after estimation.

options(stringsAsFactors = FALSE)

if (!requireNamespace("ggplot2", quietly = TRUE)) {
  stop("ggplot2 is required.")
}

data_path <- file.path(
  "data", "classic3", "processed",
  "classic3_splade_holdout_train_top2000_260711.rds"
)
fit_path <- file.path(
  "results", "classic3_splade_holdout_train_exact_ic_260711",
  "classic3_exact_after_refit_ic_selected_fits.rds"
)
out_csv <- file.path(
  "results", "realdata_final_validation_260711",
  "classic3_ecgl_top_token_contrasts.csv"
)
signed_out_csv <- file.path(
  "results", "realdata_final_validation_260711",
  "classic3_ecgl_signed_token_contrasts.csv"
)
out_png <- file.path(
  "docs", "manuscript", "figures",
  "classic3_ecgl_centered_eta_heatmap_260714.png"
)

for (path in c(data_path, fit_path)) {
  if (!file.exists(path)) stop("Missing input: ", path)
}
dir.create(dirname(out_csv), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(out_png), recursive = TRUE, showWarnings = FALSE)

payload <- readRDS(data_path)
fits <- readRDS(fit_path)
fit <- fits[["E-CGL__BIC"]]
if (is.null(fit)) stop("Missing E-CGL__BIC fit.")

class_names <- as.character(payload$categories)
truth <- as.integer(payload$y)
tokens <- as.character(payload$vocab)
K <- length(class_names)
d <- length(tokens)

if (!identical(dim(fit$mu), c(K, d))) stop("Fit dimension mismatch.")
if (length(fit$active) != d) stop("Support dimension mismatch.")

permutations <- function(values) {
  if (length(values) == 1L) return(matrix(values, nrow = 1L))
  do.call(rbind, lapply(seq_along(values), function(i) {
    rest <- permutations(values[-i])
    cbind(values[i], rest)
  }))
}

match_components <- function(truth, predicted, K) {
  tab <- table(
    factor(predicted, levels = seq_len(K)),
    factor(truth, levels = seq_len(K))
  )
  perms <- permutations(seq_len(K))
  scores <- apply(perms, 1L, function(p) {
    sum(tab[cbind(seq_len(K), p)])
  })
  as.integer(perms[which.max(scores), ])
}

component_to_class <- match_components(truth, fit$cluster, K)
component_for_class <- match(seq_len(K), component_to_class)

eta <- sweep(fit$mu, 1L, fit$kappa, "*")
centered <- sweep(eta, 2L, colMeans(eta), "-")
centered_by_class <- centered[component_for_class, , drop = FALSE]
rownames(centered_by_class) <- class_names
centered_norm <- sqrt(colSums(centered * centered))

top_n <- 5L
top_rows <- do.call(rbind, lapply(seq_len(K), function(k) {
  values <- centered_by_class[k, ]
  eligible <- which(fit$active & is.finite(values))
  ord <- head(eligible[order(values[eligible], decreasing = TRUE)], top_n)
  row <- data.frame(
    target_class = class_names[k],
    rank = seq_along(ord),
    feature_id = ord,
    token = tokens[ord],
    centered_norm = centered_norm[ord],
    stringsAsFactors = FALSE
  )
  for (h in seq_len(K)) {
    row[[paste0("contrast_", class_names[h])]] <- centered_by_class[h, ord]
  }
  row
}))

utils::write.csv(top_rows, out_csv, row.names = FALSE)

signed_rows <- do.call(rbind, lapply(seq_len(K), function(k) {
  values <- centered_by_class[k, ]
  eligible <- which(fit$active & is.finite(values))
  positive <- head(eligible[order(values[eligible], decreasing = TRUE)], top_n)
  negative <- head(eligible[order(values[eligible], decreasing = FALSE)], top_n)
  rbind(
    data.frame(
      target_class = class_names[k], direction = "positive",
      rank = seq_along(positive), feature_id = positive,
      token = tokens[positive], contrast = values[positive],
      stringsAsFactors = FALSE
    ),
    data.frame(
      target_class = class_names[k], direction = "negative",
      rank = seq_along(negative), feature_id = negative,
      token = tokens[negative], contrast = values[negative],
      stringsAsFactors = FALSE
    )
  )
}))
utils::write.csv(signed_rows, signed_out_csv, row.names = FALSE)

plot_rows <- do.call(rbind, lapply(seq_len(nrow(top_rows)), function(i) {
  do.call(rbind, lapply(seq_len(K), function(k) {
    data.frame(
      target_class = top_rows$target_class[i],
      rank = top_rows$rank[i],
      token = top_rows$token[i],
      display_class = class_names[k],
      contrast = top_rows[[paste0("contrast_", class_names[k])]][i],
      stringsAsFactors = FALSE
    )
  }))
}))

row_order <- paste(top_rows$target_class, top_rows$rank, top_rows$token, sep = "__")
plot_rows$row_id <- paste(
  plot_rows$target_class, plot_rows$rank, plot_rows$token, sep = "__"
)
plot_rows$row_label <- paste0(plot_rows$target_class, "  |  ", plot_rows$token)
label_map <- setNames(
  paste0(top_rows$target_class, "  |  ", top_rows$token),
  row_order
)
plot_rows$row_id <- factor(plot_rows$row_id, levels = rev(row_order))
plot_rows$display_class <- factor(plot_rows$display_class, levels = class_names)

max_abs <- max(abs(plot_rows$contrast), na.rm = TRUE)
plot_rows$text_color <- ifelse(
  abs(plot_rows$contrast) >= 0.55 * max_abs, "white", "#17202A"
)

p <- ggplot2::ggplot(
  plot_rows,
  ggplot2::aes(x = display_class, y = row_id, fill = contrast)
) +
  ggplot2::geom_tile(color = "#D8DEE6", linewidth = 0.45) +
  ggplot2::geom_text(
    ggplot2::aes(label = sprintf("%.1f", contrast), color = text_color),
    size = 3.35,
    family = "sans"
  ) +
  ggplot2::scale_color_identity() +
  ggplot2::scale_fill_gradient2(
    low = "#D97706",
    mid = "#F7F8FA",
    high = "#2563A6",
    midpoint = 0,
    limits = c(-max_abs, max_abs),
    name = expression(hat(c)[kj])
  ) +
  ggplot2::scale_y_discrete(labels = function(x) unname(label_map[x])) +
  ggplot2::labs(
    title = "Classic3 E-CGL 선택 token의 centered-Eta contrast",
    subtitle = paste0(
      "Class별 양의 contrast 상위 5개; exact BIC after refit, ",
      "selected q = ", sum(fit$active), " / ", d
    ),
    x = "대응 class",
    y = "기준 class  |  SPLADE token",
    caption = paste0(
      "양수는 해당 class의 선형 score가 component 평균보다 큰 방향을 뜻한다. ",
      "Class label은 적합 후 component 명명에만 사용했다."
    )
  ) +
  ggplot2::theme_minimal(base_size = 12, base_family = "Malgun Gothic") +
  ggplot2::theme(
    plot.background = ggplot2::element_rect(fill = "white", color = NA),
    panel.background = ggplot2::element_rect(fill = "white", color = NA),
    panel.grid = ggplot2::element_blank(),
    plot.title = ggplot2::element_text(
      face = "bold", size = 16, color = "#17202A", margin = ggplot2::margin(b = 6)
    ),
    plot.subtitle = ggplot2::element_text(
      size = 11.5, color = "#46515C", margin = ggplot2::margin(b = 12)
    ),
    plot.caption = ggplot2::element_text(
      size = 9.5, color = "#5F6B76", hjust = 0, margin = ggplot2::margin(t = 12)
    ),
    axis.title = ggplot2::element_text(face = "bold", color = "#303943"),
    axis.text.x = ggplot2::element_text(face = "bold", color = "#303943"),
    axis.text.y = ggplot2::element_text(size = 10.2, color = "#303943"),
    legend.position = "right",
    legend.title = ggplot2::element_text(face = "bold"),
    plot.margin = ggplot2::margin(18, 22, 18, 18)
  )

ggplot2::ggsave(
  filename = out_png,
  plot = p,
  width = 11.2,
  height = 8.4,
  units = "in",
  dpi = 180,
  bg = "white"
)

cat("component-to-class mapping:\n")
print(data.frame(
  fitted_component = seq_len(K),
  matched_class = class_names[component_to_class]
), row.names = FALSE)
cat("\ntop tokens:\n")
print(top_rows[, c("target_class", "rank", "token", "centered_norm")], row.names = FALSE)
cat("\nSaved CSV:", normalizePath(out_csv, winslash = "/"), "\n")
cat("Saved signed CSV:", normalizePath(signed_out_csv, winslash = "/"), "\n")
cat("Saved PNG:", normalizePath(out_png, winslash = "/"), "\n")
