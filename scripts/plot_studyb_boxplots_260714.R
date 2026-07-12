#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
})

raw_dirs <- list.dirs("results", recursive = FALSE, full.names = TRUE)
raw_dirs <- raw_dirs[grepl(
  "^paper_eta_studyb_v2_refitB_guard40_all6_(equal|hetero)_eb(025|05|10)_n(300|1000)_rep100_path240_260712$",
  basename(raw_dirs)
)]
raw_files <- unlist(lapply(raw_dirs, function(path) {
  list.files(path, pattern = "_raw\\.csv$", full.names = TRUE)
}), use.names = FALSE)

if (length(raw_files) == 0) {
  stop("No Study B rep=100 raw files found.")
}

method_levels_raw <- c("D-L", "D-GL", "D-AGL", "E-L", "E-CGL", "E-ACGL")
method_levels_plot <- c("M-L", "M-GL", "M-AGL", "E-CL", "E-CGL", "E-ACGL")

read_one <- function(path) {
  out <- read.csv(path, stringsAsFactors = FALSE)
  out$source_file <- basename(path)
  out
}

df <- bind_rows(lapply(raw_files, read_one)) %>%
  filter(
    method %in% method_levels_raw,
    (method %in% c("D-L", "D-GL", "D-AGL", "E-L") & rule == "current_BIC_before_support_refit") |
      (method %in% c("E-CGL", "E-ACGL") & rule == "BIC_after_exact_refit"),
    n %in% c(300, 1000),
    target_oracle_error %in% c(0.025, 0.05, 0.10)
  ) %>%
  mutate(
    method_label = recode(
      method,
      "D-L" = "M-L", "D-GL" = "M-GL", "D-AGL" = "M-AGL",
      "E-L" = "E-CL", "E-CGL" = "E-CGL", "E-ACGL" = "E-ACGL"
    ),
    method_label = factor(method_label, levels = method_levels_plot),
    family = if_else(grepl("^M-", method_label), "M-series", "E-series"),
    family = factor(family, levels = c("M-series", "E-series")),
    target_label = case_when(
      target_oracle_error == 0.025 ~ "e_B = 2.5%",
      target_oracle_error == 0.05 ~ "e_B = 5.0%",
      target_oracle_error == 0.10 ~ "e_B = 10.0%",
      TRUE ~ paste0("e_B = ", target_oracle_error)
    ),
    target_label = factor(target_label, levels = c("e_B = 2.5%", "e_B = 5.0%", "e_B = 10.0%")),
    n_label = factor(paste0("n = ", n), levels = c("n = 300", "n = 1000")),
    kappa_pattern = if_else(grepl("hetero", cell), "heterogeneous", "equal"),
    selected_common_q = common_q_selected,
    selected_decision_q = decision_q_selected,
    selected_noise_q = noise_q_selected,
    F1_plot = F1,
    MSE_eta = MSE_centered_eta,
    log_MSE_eta = log(MSE_eta + 1e-12)
  )

if (nrow(df) == 0) {
  stop("Study B raw files were found, but no rows matched the plotting filter.")
}

fig_dir <- file.path("docs", "simulations", "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

make_boxplot <- function(data, y_col, y_label, title, filename, y_limits = NULL,
                         hline = NULL,
                         subtitle = "반복 100회 결과이며, 각 패널은 등분산·이분산 집중도 조건을 함께 포함한다.") {
  p <- ggplot(data, aes(x = method_label, y = .data[[y_col]], fill = family)) +
    geom_boxplot(width = 0.72, linewidth = 0.35, outlier.size = 0.55, outlier.alpha = 0.65) +
    facet_grid(n_label ~ target_label) +
    scale_fill_manual(values = c("M-series" = "#356AA0", "E-series" = "#D97732")) +
    labs(
      title = title,
      subtitle = subtitle,
      x = NULL,
      y = y_label,
      fill = NULL
    ) +
    theme_bw(base_size = 11) +
    theme(
      text = element_text(family = "Malgun Gothic"),
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
  title = "Study B: 군집 복원 정확도",
  filename = "studyb_boxplot_ari_by_eb_n_260714.png",
  y_limits = c(0, 1),
  subtitle = "반복 100회 결과이며, 계산 실패로 ARI가 정의되지 않은 1개 M-GL 반복은 제외한다."
)

make_boxplot(
  df,
  y_col = "F1_plot",
  y_label = "Decision-support F1",
  title = "Study B: posterior decision support 복원",
  filename = "studyb_boxplot_f1_by_eb_n_260714.png",
  y_limits = c(0, 1),
  subtitle = "반복 100회 결과이며, 등분산·이분산 집중도 조건을 함께 포함한다."
)

make_boxplot(
  df,
  y_col = "selected_q",
  y_label = "Selected q",
  title = "Study B: 선택된 support 크기",
  filename = "studyb_boxplot_selectedq_by_eb_n_260714.png",
  y_limits = c(0, 205),
  hline = 16
)

make_boxplot(
  df,
  y_col = "selected_noise_q",
  y_label = "Selected noise q",
  title = "Study B: 선택된 noise 좌표 수",
  filename = "studyb_boxplot_noiseq_by_eb_n_260714.png",
  y_limits = c(0, 185),
  hline = 0
)

make_boxplot(
  df,
  y_col = "log_MSE_eta",
  y_label = "log(MSE_eta)",
  title = "Study B: centered eta 추정 오차",
  filename = "studyb_boxplot_logmse_eta_by_eb_n_260714.png",
  subtitle = "반복 100회 결과이며, 계산 실패로 MSE_eta가 정의되지 않은 1개 M-GL 반복은 제외한다."
)

cat("Created Study B boxplots in ", normalizePath(fig_dir), "\n", sep = "")
cat("Rows used: ", nrow(df), "\n", sep = "")
cat("Raw files used:\n")
cat(paste0(" - ", raw_files, collapse = "\n"), "\n", sep = "")
