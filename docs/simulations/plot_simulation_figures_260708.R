suppressPackageStartupMessages({
  library(ggplot2)
})

fig_dir <- file.path("docs", "simulations", "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

method_order <- c("M-L", "M-GL", "M-AGL", "E-CL", "E-CGL", "E-CAGL")
family_colors <- c("M-series" = "#5B8DEF", "E-series" = "#F39C6B")

num <- function(x) suppressWarnings(as.numeric(x))

recode_method_names <- function(method) {
  method <- as.character(method)
  method[method == "D-L"] <- "M-L"
  method[method == "D-GL"] <- "M-GL"
  method[method == "D-AGL"] <- "M-AGL"
  method[method == "E-L"] <- "E-CL"
  method[method == "E-GL"] <- "E-CGL"
  method[method == "E-AGL"] <- "E-CAGL"
  method
}

bind_rows_base <- function(rows) {
  all_names <- unique(unlist(lapply(rows, names), use.names = FALSE))
  rows <- lapply(rows, function(x) {
    missing <- setdiff(all_names, names(x))
    for (nm in missing) x[[nm]] <- NA
    x[, all_names, drop = FALSE]
  })
  do.call(rbind, rows)
}

read_one <- function(path, scenario_label) {
  if (!file.exists(path)) {
    stop("Missing input file: ", path)
  }
  x <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  x$scenario_label <- scenario_label
  if ("method" %in% names(x)) {
    x$method <- recode_method_names(x$method)
  }
  x$method <- factor(x$method, levels = method_order)
  x$method_family <- ifelse(grepl("^M-", as.character(x$method)), "M-series", "E-series")
  x$method_family <- factor(x$method_family, levels = c("M-series", "E-series"))
  if (!("MSE_eta" %in% names(x)) && "MSE_centered_eta" %in% names(x)) {
    x$MSE_eta <- x$MSE_centered_eta
  }
  if (!("common_selected_q" %in% names(x)) && "common_false_selection_rate" %in% names(x)) {
    x$common_selected_q <- num(x$common_false_selection_rate) * num(x$common_q)
  }
  if (!("decision_selected_q" %in% names(x)) && "decision_selection_rate" %in% names(x)) {
    x$decision_selected_q <- num(x$decision_selection_rate) * num(x$decision_q)
  }
  if (!("noise_selected_q" %in% names(x)) && "noise_false_selection_rate" %in% names(x)) {
    x$noise_selected_q <- num(x$noise_false_selection_rate) * num(x$noise_q)
  }
  x
}

make_metric_long <- function(x) {
  metrics <- c("ARI", "selected_q", "F1", "MSE_eta")
  do.call(rbind, lapply(metrics, function(metric) {
    data.frame(
      scenario_label = x$scenario_label,
      method = x$method,
      method_family = x$method_family,
      metric = metric,
      value = num(x[[metric]]),
      stringsAsFactors = FALSE
    )
  }))
}

make_support_long <- function(x) {
  do.call(rbind, list(
    data.frame(
      scenario_label = x$scenario_label,
      method = x$method,
      method_family = x$method_family,
      support_type = "common q",
      value = num(x$common_selected_q),
      stringsAsFactors = FALSE
    ),
    data.frame(
      scenario_label = x$scenario_label,
      method = x$method,
      method_family = x$method_family,
      support_type = "specific q",
      value = num(x$decision_selected_q),
      stringsAsFactors = FALSE
    ),
    data.frame(
      scenario_label = x$scenario_label,
      method = x$method,
      method_family = x$method_family,
      support_type = "noise q",
      value = num(x$noise_selected_q),
      stringsAsFactors = FALSE
    )
  ))
}

save_metric_plot <- function(x, outfile, title) {
  long <- make_metric_long(x)
  long$scenario_label <- factor(long$scenario_label, levels = unique(x$scenario_label))
  long <- long[!is.na(long$value), , drop = FALSE]
  p <- ggplot(long, aes(x = method, y = value, fill = method_family)) +
    geom_boxplot(width = 0.72, linewidth = 0.35, outlier.size = 0.55, outlier.alpha = 0.65, na.rm = TRUE) +
    facet_grid(metric ~ scenario_label, scales = "free_y") +
    scale_fill_manual(values = family_colors, drop = FALSE)
  p <- p +
    labs(
      title = title,
      subtitle = "Boxplots use replicate-level raw results.",
      x = NULL,
      y = NULL,
      fill = NULL
    ) +
    theme_bw(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", size = 13),
      plot.subtitle = element_text(size = 9, color = "grey25"),
      strip.background = element_rect(fill = "grey92", color = "grey70"),
      strip.text = element_text(face = "bold", size = 8),
      axis.text.x = element_text(angle = 35, hjust = 1),
      axis.text.y = element_text(size = 8),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      legend.position = "bottom"
    )
  ggsave(outfile, p, width = 14, height = 8.4, dpi = 220)
}

save_support_plot <- function(x, outfile, title) {
  long <- make_support_long(x)
  long$scenario_label <- factor(long$scenario_label, levels = unique(x$scenario_label))
  long$support_type <- factor(long$support_type, levels = c("common q", "specific q", "noise q"))
  long <- long[!is.na(long$value), , drop = FALSE]
  p <- ggplot(long, aes(x = method, y = value, fill = method_family)) +
    geom_boxplot(width = 0.72, linewidth = 0.35, outlier.size = 0.55, outlier.alpha = 0.65, na.rm = TRUE) +
    facet_grid(support_type ~ scenario_label, scales = "free_y") +
    scale_fill_manual(values = family_colors, drop = FALSE) +
    labs(
      title = title,
      subtitle = "Boxplots use replicate-level raw results.",
      x = NULL,
      y = "selected coordinates",
      fill = NULL
    ) +
    theme_bw(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", size = 13),
      plot.subtitle = element_text(size = 9, color = "grey25"),
      strip.background = element_rect(fill = "grey92", color = "grey70"),
      strip.text = element_text(face = "bold", size = 8),
      axis.text.x = element_text(angle = 35, hjust = 1),
      axis.text.y = element_text(size = 8),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      legend.position = "bottom"
    )
  height <- ifelse(length(unique(x$scenario_label)) == 1, 6.0, 7.8)
  ggsave(outfile, p, width = 14, height = height, dpi = 220)
}

basic_paths <- c(
  S1 = "results/paper_eta_first_s1_angle90_kappa30_60_rep50_260702/paper_eta_first_s1_angle90_kappa30_60_rep50_260702_raw.csv",
  S2 = "results/paper_eta_first_s2_angle90_kappa45_equal_rep50_260702/paper_eta_first_s2_angle90_kappa45_equal_rep50_260702_raw.csv",
  S3 = "results/paper_eta_first_s3_angle60_kappa30_60_rep50_260702/paper_eta_first_s3_angle60_kappa30_60_rep50_260702_raw.csv",
  S4 = "results/paper_eta_first_s4_angle60_kappa45_equal_rep50_260702/paper_eta_first_s4_angle60_kappa45_equal_rep50_260702_raw.csv",
  S5 = "results/paper_eta_first_s5_angle30_kappa43_47_rep50_260702/paper_eta_first_s5_angle30_kappa43_47_rep50_260702_raw.csv",
  S6 = "results/paper_eta_first_s6_angle30_kappa45_equal_rep50_260702/paper_eta_first_s6_angle30_kappa45_equal_rep50_260702_raw.csv"
)
basic <- bind_rows_base(Map(read_one, basic_paths, names(basic_paths)))

negative_paths <- c(
  "S1-N" = "results/paper_eta_neg_s1n_angle90_kappa30_60_rep50_260702/paper_eta_neg_s1n_angle90_kappa30_60_rep50_260702_raw.csv",
  "S2-N" = "results/paper_eta_neg_s2n_angle90_kappa45_equal_rep50_260702/paper_eta_neg_s2n_angle90_kappa45_equal_rep50_260702_raw.csv",
  "S3-N" = "results/paper_eta_neg_s3n_angle60_kappa30_60_rep50_260702/paper_eta_neg_s3n_angle60_kappa30_60_rep50_260702_raw.csv",
  "S4-N" = "results/paper_eta_neg_s4n_angle60_kappa45_equal_rep50_260702/paper_eta_neg_s4n_angle60_kappa45_equal_rep50_260702_raw.csv",
  "S5-N" = "results/paper_eta_neg_s5n_angle30_kappa43_47_rep50_260702/paper_eta_neg_s5n_angle30_kappa43_47_rep50_260702_raw.csv",
  "S6-N" = "results/paper_eta_neg_s6n_angle30_kappa45_equal_rep50_260702/paper_eta_neg_s6n_angle30_kappa45_equal_rep50_260702_raw.csv"
)
negative <- bind_rows_base(Map(read_one, negative_paths, names(negative_paths)))

shared <- read_one(
  "results/paper_eta_shared_background_e1c_b3_delta8_14_rep50_260706/paper_eta_shared_background_e1c_b3_delta8_14_rep50_260706_raw.csv",
  "Shared"
)

save_metric_plot(
  basic,
  file.path(fig_dir, "simulation_basic_metrics_260708.png"),
  "Basic simulations: metrics by scenario"
)
save_support_plot(
  basic,
  file.path(fig_dir, "simulation_basic_support_260708.png"),
  "Basic simulations: selected support composition"
)
save_metric_plot(
  negative,
  file.path(fig_dir, "simulation_negative_metrics_260708.png"),
  "Negative-control simulations: metrics by scenario"
)
save_support_plot(
  negative,
  file.path(fig_dir, "simulation_negative_support_260708.png"),
  "Negative-control simulations: selected support composition"
)
save_metric_plot(
  shared,
  file.path(fig_dir, "simulation_shared_metrics_260708.png"),
  "Shared-background simulation: metrics by method"
)
save_support_plot(
  shared,
  file.path(fig_dir, "simulation_shared_support_260708.png"),
  "Shared-background simulation: selected support composition"
)

cat("Created figures in", fig_dir, "\n")
