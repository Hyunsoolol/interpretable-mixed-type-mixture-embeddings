#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
})

raw_dirs <- list.dirs("results", recursive = FALSE, full.names = TRUE)
raw_dirs <- raw_dirs[grepl("^paper_eta_oracle_bayes_studyb_.*rep100_path240_260714$", basename(raw_dirs))]
raw_files <- unlist(lapply(raw_dirs, function(path) {
  list.files(path, pattern = "_raw\\.csv$", full.names = TRUE)
}), use.names = FALSE)

if (length(raw_files) == 0) {
  stop("No Study B rep=100 raw files found.")
}

method_levels_raw <- c("D-L", "D-GL", "D-AGL", "E-L", "E-GL", "E-AGL")
method_levels_plot <- c("D-L", "D-GL", "D-AGL", "E-CL", "E-CGL", "E-CAGL")

read_one <- function(path) {
  out <- read.csv(path, stringsAsFactors = FALSE)
  out$source_file <- basename(path)
  out
}

df <- bind_rows(lapply(raw_files, read_one)) %>%
  filter(
    method %in% method_levels_raw,
    n %in% c(300, 1000),
    target_oracle_error %in% c(0.025, 0.05, 0.10)
  ) %>%
  mutate(
    method_label = recode(method, "E-L" = "E-CL", "E-GL" = "E-CGL", "E-AGL" = "E-CAGL"),
    method_label = factor(method_label, levels = method_levels_plot),
    family = if_else(grepl("^D-", method_label), "D-series", "E-series"),
    family = factor(family, levels = c("D-series", "E-series")),
    target_label = case_when(
      target_oracle_error == 0.025 ~ "e_B = 2.5%",
      target_oracle_error == 0.05 ~ "e_B = 5.0%",
      target_oracle_error == 0.10 ~ "e_B = 10.0%",
      TRUE ~ paste0("e_B = ", target_oracle_error)
    ),
    target_label = factor(target_label, levels = c("e_B = 2.5%", "e_B = 5.0%", "e_B = 10.0%")),
    n_label = factor(paste0("n = ", n), levels = c("n = 300", "n = 1000")),
    selected_common_q = common_false_selection_rate * common_q,
    selected_decision_q = decision_selection_rate * decision_q,
    selected_noise_q = noise_false_selection_rate * noise_q,
    MSE_eta = MSE_centered_eta,
    log_MSE_eta = log(MSE_eta + 1e-12)
  )

if (nrow(df) == 0) {
  stop("Study B raw files were found, but no rows matched the plotting filter.")
}

fig_dir <- file.path("docs", "simulations", "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

make_boxplot <- function(data, y_col, y_label, title, filename, y_limits = NULL, hline = NULL) {
  p <- ggplot(data, aes(x = method_label, y = .data[[y_col]], fill = family)) +
    geom_boxplot(width = 0.72, linewidth = 0.35, outlier.size = 0.55, outlier.alpha = 0.65) +
    facet_grid(n_label ~ target_label) +
    scale_fill_manual(values = c("D-series" = "#5B8DEF", "E-series" = "#F39C6B")) +
    labs(
      title = title,
      subtitle = "Boxplots use rep=100 raw results; equal and heterogeneous kappa settings are pooled within each panel.",
      x = NULL,
      y = y_label,
      fill = NULL
    ) +
    theme_bw(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", size = 13),
      plot.subtitle = element_text(size = 9, color = "grey25"),
      strip.background = element_rect(fill = "grey92", color = "grey70"),
      strip.text = element_text(face = "bold", size = 9),
      axis.text.x = element_text(angle = 35, hjust = 1, vjust = 1, size = 8),
      axis.text.y = element_text(size = 8),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      legend.position = "bottom",
      plot.margin = margin(8, 12, 8, 8)
    )

  if (!is.null(y_limits)) {
    p <- p + coord_cartesian(ylim = y_limits)
  }
  if (!is.null(hline)) {
    p <- p + geom_hline(yintercept = hline, linetype = "dashed", linewidth = 0.35, color = "grey35")
  }

  ggsave(file.path(fig_dir, filename), p, width = 13.5, height = 7.2, dpi = 220)
}

make_boxplot(
  df,
  y_col = "ARI",
  y_label = "ARI",
  title = "Study B clustering accuracy by oracle error level and sample size",
  filename = "studyb_boxplot_ari_by_eb_n_260714.png",
  y_limits = c(0, 1)
)

make_boxplot(
  df,
  y_col = "F1",
  y_label = "Decision support F1",
  title = "Study B decision-support recovery by oracle error level and sample size",
  filename = "studyb_boxplot_f1_by_eb_n_260714.png",
  y_limits = c(0, 1)
)

make_boxplot(
  df,
  y_col = "selected_q",
  y_label = "Selected q",
  title = "Study B selected support size by oracle error level and sample size",
  filename = "studyb_boxplot_selectedq_by_eb_n_260714.png",
  y_limits = c(0, 205),
  hline = 16
)

make_boxplot(
  df,
  y_col = "selected_noise_q",
  y_label = "Selected noise q",
  title = "Study B selected noise coordinates by oracle error level and sample size",
  filename = "studyb_boxplot_noiseq_by_eb_n_260714.png",
  y_limits = c(0, 185),
  hline = 0
)

make_boxplot(
  df,
  y_col = "log_MSE_eta",
  y_label = "log(MSE_eta)",
  title = "Study B centered eta estimation error by oracle error level and sample size",
  filename = "studyb_boxplot_logmse_eta_by_eb_n_260714.png"
)

cat("Created Study B boxplots in ", normalizePath(fig_dir), "\n", sep = "")
cat("Rows used: ", nrow(df), "\n", sep = "")
cat("Raw files used:\n")
cat(paste0(" - ", raw_files, collapse = "\n"), "\n", sep = "")
