# ==============================================================================
# Post-processing for the full Rossi & Barbaro (2022) reproduction run
# ==============================================================================

out_dir <- "results/rb2022_paperlike_full_260602"
plot_dir <- file.path(out_dir, "plots")
by_k_path <- file.path(out_dir, "rb2022_paperlike_full_by_k.csv")

by_k <- read.csv(by_k_path)

zero_pr <- subset(
  by_k,
  n == 1000 & K_fit == 4 & selection %in% c("AIC", "BIC")
)

zero_pr$true_nonzero <- round(zero_pr$nonzero_fraction * zero_pr$d) * zero_pr$K_fit
zero_pr$total <- zero_pr$d * zero_pr$K_fit
zero_pr$tp <- zero_pr$entry_recall * zero_pr$true_nonzero
zero_pr$pred_nonzero <- zero_pr$nnz_fraction * zero_pr$total
zero_pr$fp <- pmax(0, zero_pr$pred_nonzero - zero_pr$tp)
zero_pr$fn <- pmax(0, zero_pr$true_nonzero - zero_pr$tp)
zero_pr$tn <- pmax(0, zero_pr$total - zero_pr$true_nonzero - zero_pr$fp)

# Treat zero entries as the positive class, matching the paper's Figure 16
# interpretation of zero-component recovery.
zero_pr$zero_precision <- zero_pr$tn / (zero_pr$tn + zero_pr$fn)
zero_pr$zero_recall <- zero_pr$tn / (zero_pr$tn + zero_pr$fp)

key <- c("overlap", "nonzero_fraction", "selection")
zero_agg <- aggregate(
  zero_pr[, c(
    "zero_precision", "zero_recall",
    "entry_precision", "entry_recall",
    "nnz_fraction", "ARI"
  )],
  zero_pr[, key],
  function(x) mean(x, na.rm = TRUE)
)
zero_agg <- zero_agg[order(
  zero_agg$overlap,
  zero_agg$nonzero_fraction,
  zero_agg$selection
), ]

write.csv(
  zero_agg,
  file.path(out_dir, "rb2022_paperlike_full_zero_precision_recall_K4_N1000.csv"),
  row.names = FALSE
)

png(
  file.path(plot_dir, "fig_zero_precision_recall_K4_N1000.png"),
  width = 1300,
  height = 800,
  res = 140
)
old <- par(mfrow = c(2, 3), mar = c(4, 4, 3, 1), oma = c(0, 0, 3, 0))
for (ov in sort(unique(zero_agg$overlap))) {
  for (sp in sort(unique(zero_agg$nonzero_fraction))) {
    panel <- subset(zero_agg, overlap == ov & nonzero_fraction == sp)
    panel <- panel[match(c("AIC", "BIC"), panel$selection), ]
    mat <- rbind(
      precision = panel$zero_precision,
      recall = panel$zero_recall
    )
    colnames(mat) <- panel$selection
    barplot(
      mat,
      beside = TRUE,
      ylim = c(0, 1),
      col = c("steelblue", "orange"),
      main = sprintf("overlap=%.3f, nonzero=%.2f", ov, sp),
      ylab = "score"
    )
    legend(
      "bottomleft",
      legend = rownames(mat),
      fill = c("steelblue", "orange"),
      cex = 0.75,
      bty = "n"
    )
  }
}
mtext(
  "Precision/Recall of zero entries, K=4, N=1000",
  outer = TRUE,
  cex = 1.2,
  font = 2
)
par(old)

shown <- zero_agg
num <- vapply(shown, is.numeric, logical(1))
shown[num] <- lapply(shown[num], round, 3)
print(shown, row.names = FALSE)
