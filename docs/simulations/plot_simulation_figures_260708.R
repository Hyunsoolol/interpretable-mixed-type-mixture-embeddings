library(ggplot2)

fig_dir <- file.path("docs", "simulations", "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

method_order <- c("M-L", "M-GL", "M-AGL", "E-CL", "E-CGL", "E-CAGL")
method_colors <- c(
  "M-L" = "#A6C8E5",
  "M-GL" = "#4E79A7",
  "M-AGL" = "#0B3C68",
  "E-CL" = "#F7C59F",
  "E-CGL" = "#F28E2B",
  "E-CAGL" = "#B13A16"
)
support_colors <- c(
  "common q" = "#A0CBE8",
  "specific q" = "#59A14F",
  "noise q" = "#E15759"
)

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
  x <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  x$scenario_label <- scenario_label
  if ("method" %in% names(x)) {
    x$method <- recode_method_names(x$method)
  }
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
      method = factor(x$method, levels = method_order),
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
      method = factor(x$method, levels = method_order),
      support_type = "common q",
      value = num(x$common_selected_q),
      stringsAsFactors = FALSE
    ),
    data.frame(
      scenario_label = x$scenario_label,
      method = factor(x$method, levels = method_order),
      support_type = "specific q",
      value = num(x$decision_selected_q),
      stringsAsFactors = FALSE
    ),
    data.frame(
      scenario_label = x$scenario_label,
      method = factor(x$method, levels = method_order),
      support_type = "noise q",
      value = num(x$noise_selected_q),
      stringsAsFactors = FALSE
    )
  ))
}

save_metric_plot <- function(x, outfile, title) {
  long <- make_metric_long(x)
  long$scenario_label <- factor(long$scenario_label, levels = unique(x$scenario_label))
  p <- ggplot(long, aes(x = scenario_label, y = value, fill = method)) +
    geom_col(position = position_dodge(width = 0.78), width = 0.68, na.rm = TRUE) +
    facet_wrap(~metric, scales = "free_y", ncol = 2) +
    scale_fill_manual(values = method_colors, drop = FALSE)
  p <- p +
    labs(title = title, x = NULL, y = NULL, fill = "method") +
    theme_minimal(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      axis.text.x = element_text(angle = 35, hjust = 1),
      panel.grid.minor = element_blank(),
      legend.position = "bottom"
    )
  ggsave(outfile, p, width = 10.5, height = 6.2, dpi = 180)
}

save_support_plot <- function(x, outfile, title) {
  long <- make_support_long(x)
  long$scenario_label <- factor(long$scenario_label, levels = unique(x$scenario_label))
  long$support_type <- factor(long$support_type, levels = c("common q", "specific q", "noise q"))
  p <- ggplot(long, aes(x = method, y = value, fill = support_type)) +
    geom_col(width = 0.72, na.rm = TRUE) +
    facet_wrap(~scenario_label, scales = "free_y", ncol = ifelse(length(unique(x$scenario_label)) == 1, 1, 3)) +
    scale_fill_manual(values = support_colors, drop = FALSE) +
    labs(title = title, x = NULL, y = "selected coordinates", fill = NULL) +
    theme_minimal(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      axis.text.x = element_text(angle = 35, hjust = 1),
      panel.grid.minor = element_blank(),
      legend.position = "bottom"
    )
  height <- ifelse(length(unique(x$scenario_label)) == 1, 4.6, 7.2)
  ggsave(outfile, p, width = 10.5, height = height, dpi = 180)
}

basic_paths <- c(
  S1 = "results/paper_eta_first_s1_angle90_kappa30_60_rep50_260702/paper_eta_first_s1_angle90_kappa30_60_rep50_260702_summary.csv",
  S2 = "results/paper_eta_first_s2_angle90_kappa45_equal_rep50_260702/paper_eta_first_s2_angle90_kappa45_equal_rep50_260702_summary.csv",
  S3 = "results/paper_eta_first_s3_angle60_kappa30_60_rep50_260702/paper_eta_first_s3_angle60_kappa30_60_rep50_260702_summary.csv",
  S4 = "results/paper_eta_first_s4_angle60_kappa45_equal_rep50_260702/paper_eta_first_s4_angle60_kappa45_equal_rep50_260702_summary.csv",
  S5 = "results/paper_eta_first_s5_angle30_kappa43_47_rep50_260702/paper_eta_first_s5_angle30_kappa43_47_rep50_260702_summary.csv",
  S6 = "results/paper_eta_first_s6_angle30_kappa45_equal_rep50_260702/paper_eta_first_s6_angle30_kappa45_equal_rep50_260702_summary.csv"
)
basic <- bind_rows_base(Map(read_one, basic_paths, names(basic_paths)))

negative <- read.csv(
  "results/paper_eta_negative_control_s1n_s4n_rep50_260702/paper_eta_negative_control_s1n_s4n_rep50_summary.csv",
  stringsAsFactors = FALSE,
  check.names = FALSE
)
negative$scenario_label <- negative$scenario
negative$method <- recode_method_names(negative$method)
negative$MSE_eta <- negative$MSE_eta
negative$common_selected_q <- num(negative$common_selected_q)
negative$decision_selected_q <- num(negative$decision_selected_q)
negative$noise_selected_q <- num(negative$noise_selected_q)
negative <- bind_rows_base(list(
  negative,
  read_one(
    "results/paper_eta_neg_s5n_angle30_kappa43_47_rep50_260702/paper_eta_neg_s5n_angle30_kappa43_47_rep50_260702_summary.csv",
    "S5-N"
  ),
  read_one(
    "results/paper_eta_neg_s6n_angle30_kappa45_equal_rep50_260702/paper_eta_neg_s6n_angle30_kappa45_equal_rep50_260702_summary.csv",
    "S6-N"
  )
))

shared <- read_one(
  "results/paper_eta_shared_background_e1c_b3_delta8_14_rep50_260706/paper_eta_shared_background_e1c_b3_delta8_14_rep50_260706_summary.csv",
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
