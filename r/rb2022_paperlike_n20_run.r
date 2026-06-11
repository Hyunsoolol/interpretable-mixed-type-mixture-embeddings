# ==============================================================================
# Paper-like reproduction for Rossi & Barbaro (2022), medium budget
# ------------------------------------------------------------------------------
# Matches the artificial simulation grid in the paper as closely as is practical
# for an interactive run:
#   - d = 100, true K = 4
#   - N in {200, 1000}
#   - overlap in {2.5%, 5%}
#   - true nonzero fractions in {5%, 10%, 15%}
#   - candidate K in {1, ..., 6}
#   - component-specific kappa
#
# Reduced relative to the paper:
#   - 20 data sets per condition instead of 100
#   - 5 random starts instead of 10
#
# The script writes one CSV per simulation cell before combining results. It can
# be rerun after interruption and will skip completed cells.
# ==============================================================================

source(file.path("r", "rossi_barbaro_2022_reproduction.r"))

cfg <- list(
  run_label = Sys.getenv("RB2022_RUN_LABEL", "rb2022_paperlike_n20"),
  n_rep = as.integer(Sys.getenv("RB2022_N_REP", "20")),
  n_grid = c(200, 1000),
  d = 100,
  K_true = 4,
  K_fit_grid = 1:6,
  overlap_grid = c(0.025, 0.05),
  nonzero_fraction_grid = c(0.05, 0.10, 0.15),
  nstart = as.integer(Sys.getenv("RB2022_NSTART", "5")),
  max_path_steps = as.integer(Sys.getenv("RB2022_MAX_PATH_STEPS", "400")),
  base_seed = 20260602,
  out_dir = Sys.getenv("RB2022_OUT_DIR", "results/rb2022_paperlike_n20_260602"),
  workers = as.integer(Sys.getenv("RB2022_WORKERS", "4"))
)

if (!dir.exists(cfg$out_dir)) dir.create(cfg$out_dir, recursive = TRUE)
cell_dir <- file.path(cfg$out_dir, "cells")
plot_dir <- file.path(cfg$out_dir, "plots")
if (!dir.exists(cell_dir)) dir.create(cell_dir, recursive = TRUE)
if (!dir.exists(plot_dir)) dir.create(plot_dir, recursive = TRUE)

tasks <- expand.grid(
  rep_id = seq_len(cfg$n_rep),
  n = cfg$n_grid,
  overlap = cfg$overlap_grid,
  nonzero_fraction = cfg$nonzero_fraction_grid,
  KEEP.OUT.ATTRS = FALSE,
  stringsAsFactors = FALSE
)
tasks$cell_id <- seq_len(nrow(tasks))

run_one_task <- function(task_row, cfg, cell_dir) {
  cell_id <- task_row$cell_id
  out_file <- file.path(cell_dir, sprintf("cell_%03d.csv", cell_id))
  err_file <- file.path(cell_dir, sprintf("cell_%03d_error.txt", cell_id))
  if (file.exists(out_file)) {
    return(list(ok = TRUE, skipped = TRUE, file = out_file, cell_id = cell_id))
  }

  tryCatch({
    seed <- cfg$base_seed + 100000L * task_row$rep_id +
      1000L * as.integer(1000 * task_row$overlap) +
      as.integer(1000 * task_row$nonzero_fraction) + task_row$n

    one <- run_rb2022_one(
      rep_id = task_row$rep_id,
      n = task_row$n,
      d = cfg$d,
      K_true = cfg$K_true,
      K_fit_grid = cfg$K_fit_grid,
      overlap = task_row$overlap,
      nonzero_fraction = task_row$nonzero_fraction,
      nstart = cfg$nstart,
      max_path_steps = cfg$max_path_steps,
      seed = seed,
      verbose = FALSE
    )

    write.csv(one, out_file, row.names = FALSE)
    list(ok = TRUE, skipped = FALSE, file = out_file, cell_id = cell_id)
  }, error = function(e) {
    writeLines(conditionMessage(e), err_file)
    list(ok = FALSE, skipped = FALSE, file = err_file, cell_id = cell_id,
         error = conditionMessage(e))
  })
}

cat(sprintf(
  "Running %d cells: n_rep=%d, nstart=%d, workers=%d, max_path_steps=%d\n",
  nrow(tasks), cfg$n_rep, cfg$nstart, cfg$workers, cfg$max_path_steps
))

workers <- max(1L, min(cfg$workers, nrow(tasks)))
if (workers == 1L) {
  task_results <- vector("list", nrow(tasks))
  for (i in seq_len(nrow(tasks))) {
    cat(sprintf("[%03d/%03d] rep=%d n=%d overlap=%.3f nonzero=%.2f\n",
                i, nrow(tasks), tasks$rep_id[i], tasks$n[i],
                tasks$overlap[i], tasks$nonzero_fraction[i]))
    task_results[[i]] <- run_one_task(tasks[i, ], cfg, cell_dir)
  }
} else {
  cl <- parallel::makeCluster(workers)
  on.exit(parallel::stopCluster(cl), add = TRUE)
  parallel::clusterEvalQ(cl, {
    source(file.path("r", "rossi_barbaro_2022_reproduction.r"))
    NULL
  })
  parallel::clusterExport(cl, c("cfg", "cell_dir", "run_one_task"), envir = environment())
  task_results <- parallel::parLapplyLB(
    cl,
    split(tasks, seq_len(nrow(tasks))),
    function(row) run_one_task(row[1, ], cfg, cell_dir)
  )
}

failed <- vapply(task_results, function(x) !isTRUE(x$ok), logical(1))
if (any(failed)) {
  print(task_results[failed])
  stop("Some cells failed. See *_error.txt files in the cells directory.")
}

cell_files <- list.files(cell_dir, pattern = "^cell_[0-9]+\\.csv$", full.names = TRUE)
raw <- do.call(rbind, lapply(cell_files, read.csv))
raw_path <- file.path(cfg$out_dir, sprintf("%s_raw.csv", cfg$run_label))
summary_path <- file.path(cfg$out_dir, sprintf("%s_summary.csv", cfg$run_label))
write.csv(raw, raw_path, row.names = FALSE)

summary <- summarize_rb2022(raw)
write.csv(summary, summary_path, row.names = FALSE)

safe_mean <- function(x) if (all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE)
safe_sd <- function(x) if (sum(!is.na(x)) > 1) sd(x, na.rm = TRUE) else NA_real_

make_compact <- function(raw, crit_name = "BIC") {
  key <- c("n", "overlap", "nonzero_fraction")
  skm <- subset(raw, method == "spherical_kmeans" & selected == TRUE)
  skm_agg <- aggregate(skm[, c("ARI")], skm[, key], safe_mean)
  names(skm_agg)[4] <- "skm_ARI"

  dense <- subset(raw, method == "dense_vmf_selected_K" &
                    selected == TRUE & criterion == crit_name)
  dense_agg <- aggregate(dense[, c("ARI", "K_fit")], dense[, key], safe_mean)
  names(dense_agg)[4:5] <- c("dense_ARI", "dense_K")

  sparse <- subset(raw, method == "sparse_vmf_selected_K_beta" &
                     selected == TRUE & criterion == crit_name)
  sparse_agg <- aggregate(
    sparse[, c("ARI", "K_fit", "nnz_fraction", "beta")],
    sparse[, key],
    safe_mean
  )
  names(sparse_agg)[4:7] <- c("sparse_ARI", "sparse_K", "sparse_nnz", "sparse_beta")

  tab <- Reduce(function(x, y) merge(x, y, by = key, all = TRUE),
                list(skm_agg, dense_agg, sparse_agg))
  tab <- tab[order(tab$n, tab$overlap, tab$nonzero_fraction), ]
  tab
}

bic_compact <- make_compact(raw, "BIC")
ebic_compact <- make_compact(raw, "EBIC")
write.csv(bic_compact, file.path(cfg$out_dir, sprintf("%s_BIC_compact.csv", cfg$run_label)),
          row.names = FALSE)
write.csv(ebic_compact, file.path(cfg$out_dir, sprintf("%s_EBIC_compact.csv", cfg$run_label)),
          row.names = FALSE)

path <- subset(raw, method == "sparse_vmf_path")
group_cols <- c("rep", "n", "overlap", "nonzero_fraction", "K_fit")

pick_best_by <- function(tab, crit) tab[which.min(tab[[crit]]), , drop = FALSE]
split_key <- interaction(path[, group_cols], drop = TRUE)
dense_by_k <- subset(path, beta == 0)
aic_by_k <- do.call(rbind, lapply(split(path, split_key), pick_best_by, crit = "AIC"))
bic_by_k <- do.call(rbind, lapply(split(path, split_key), pick_best_by, crit = "BIC"))
dense_by_k$selection <- "Dense"
aic_by_k$selection <- "AIC"
bic_by_k$selection <- "BIC"
by_k <- rbind(dense_by_k, aic_by_k, bic_by_k)
write.csv(by_k, file.path(cfg$out_dir, sprintf("%s_by_k.csv", cfg$run_label)),
          row.names = FALSE)

aggregate_metric <- function(dat, metric) {
  key <- c("n", "overlap", "nonzero_fraction", "K_fit", "selection")
  agg <- aggregate(dat[, metric, drop = FALSE], dat[, key], safe_mean)
  names(agg)[ncol(agg)] <- metric
  agg
}

plot_metric_by_k <- function(dat, metric, n_value, file_name, ylab, ylim = NULL) {
  sub <- subset(dat, n == n_value)
  png(file.path(plot_dir, file_name), width = 1500, height = 900, res = 140)
  old <- par(mfrow = c(2, 3), mar = c(4, 4, 3, 1), oma = c(0, 0, 3, 0))
  on.exit(par(old), add = TRUE)
  cols <- c(Dense = "firebrick", AIC = "darkgreen", BIC = "royalblue")
  pchs <- c(Dense = 16, AIC = 17, BIC = 15)
  for (ov in sort(unique(sub$overlap))) {
    for (sp in sort(unique(sub$nonzero_fraction))) {
      panel <- subset(sub, overlap == ov & nonzero_fraction == sp)
      if (is.null(ylim)) {
        yy <- panel[[metric]]
        yr <- range(yy, na.rm = TRUE)
        pad <- diff(yr) * 0.12 + 1e-6
        y_lim <- c(max(0, yr[1] - pad), min(1, yr[2] + pad))
      } else {
        y_lim <- ylim
      }
      plot(NA, xlim = range(cfg$K_fit_grid), ylim = y_lim,
           xlab = "K", ylab = ylab,
           main = sprintf("overlap=%.3f, nonzero=%.2f", ov, sp),
           xaxt = "n")
      axis(1, at = cfg$K_fit_grid)
      for (sel in c("Dense", "AIC", "BIC")) {
        line <- subset(panel, selection == sel)
        line <- line[order(line$K_fit), ]
        if (nrow(line) > 0) {
          lines(line$K_fit, line[[metric]], type = "b",
                col = cols[sel], pch = pchs[sel], lwd = 2)
        }
      }
      legend("bottomright", legend = c("Dense", "AIC", "BIC"),
             col = cols, pch = pchs, lwd = 2, cex = 0.75, bty = "n")
    }
  }
  mtext(sprintf("%s by K, N=%d", ylab, n_value), outer = TRUE, cex = 1.2, font = 2)
}

ari_by_k <- aggregate_metric(by_k, "ARI")
nnz_by_k <- aggregate_metric(by_k, "nnz_fraction")
plot_metric_by_k(ari_by_k, "ARI", 200, "fig_ari_by_k_N200.png", "ARI", ylim = c(0, 1))
plot_metric_by_k(ari_by_k, "ARI", 1000, "fig_ari_by_k_N1000.png", "ARI", ylim = c(0, 1))
plot_metric_by_k(nnz_by_k, "nnz_fraction", 200, "fig_sparsity_by_k_N200.png", "Nonzero fraction", ylim = c(0, 1))
plot_metric_by_k(nnz_by_k, "nnz_fraction", 1000, "fig_sparsity_by_k_N1000.png", "Nonzero fraction", ylim = c(0, 1))

plot_k_selection <- function(raw, n_value, file_name) {
  selected <- subset(raw, selected == TRUE &
                       criterion %in% c("AIC", "BIC") &
                       method %in% c("dense_vmf_selected_K", "sparse_vmf_selected_K_beta") &
                       n == n_value)
  selected$model <- ifelse(selected$method == "dense_vmf_selected_K", "Dense", "Sparse")
  png(file.path(plot_dir, file_name), width = 1500, height = 900, res = 140)
  old <- par(mfrow = c(2, 3), mar = c(4, 4, 3, 1), oma = c(0, 0, 3, 0))
  on.exit(par(old), add = TRUE)
  for (ov in sort(unique(selected$overlap))) {
    for (sp in sort(unique(selected$nonzero_fraction))) {
      panel <- subset(selected, overlap == ov & nonzero_fraction == sp)
      tab <- table(factor(panel$K_fit, levels = cfg$K_fit_grid),
                   interaction(panel$model, panel$criterion, sep = "-"))
      barplot(t(tab), beside = TRUE,
              col = c("mistyrose", "tomato", "lightblue", "royalblue"),
              ylim = c(0, cfg$n_rep),
              main = sprintf("overlap=%.3f, nonzero=%.2f", ov, sp),
              xlab = "K", ylab = "count")
      legend("topright", legend = colnames(tab),
             fill = c("mistyrose", "tomato", "lightblue", "royalblue"),
             cex = 0.65, bty = "n")
    }
  }
  mtext(sprintf("Selected K counts, N=%d", n_value), outer = TRUE, cex = 1.2, font = 2)
}

plot_k_selection(raw, 200, "fig_selected_k_counts_N200.png")
plot_k_selection(raw, 1000, "fig_selected_k_counts_N1000.png")

pr <- subset(by_k, n == 1000 & K_fit == 4 & selection %in% c("AIC", "BIC"))
pr_key <- c("overlap", "nonzero_fraction", "selection")
pr_agg <- aggregate(pr[, c("entry_precision", "entry_recall")], pr[, pr_key], safe_mean)
write.csv(pr_agg, file.path(cfg$out_dir, sprintf("%s_precision_recall_K4_N1000.csv", cfg$run_label)),
          row.names = FALSE)

png(file.path(plot_dir, "fig_precision_recall_K4_N1000.png"), width = 1300, height = 800, res = 140)
old <- par(mfrow = c(2, 3), mar = c(4, 4, 3, 1), oma = c(0, 0, 3, 0))
for (ov in sort(unique(pr_agg$overlap))) {
  for (sp in sort(unique(pr_agg$nonzero_fraction))) {
    panel <- subset(pr_agg, overlap == ov & nonzero_fraction == sp)
    mat <- rbind(precision = panel$entry_precision, recall = panel$entry_recall)
    colnames(mat) <- panel$selection
    barplot(mat, beside = TRUE, ylim = c(0, 1),
            col = c("steelblue", "orange"),
            main = sprintf("overlap=%.3f, nonzero=%.2f", ov, sp),
            ylab = "score")
    legend("topleft", legend = rownames(mat), fill = c("steelblue", "orange"),
           cex = 0.75, bty = "n")
  }
}
mtext("Precision/Recall of nonzero entries, K=4, N=1000", outer = TRUE, cex = 1.2, font = 2)
par(old)

cat("\nDone.\n")
cat("Raw: ", normalizePath(raw_path, winslash = "/"), "\n", sep = "")
cat("Summary: ", normalizePath(summary_path, winslash = "/"), "\n", sep = "")
cat("BIC compact: ", normalizePath(file.path(cfg$out_dir, sprintf("%s_BIC_compact.csv", cfg$run_label)), winslash = "/"), "\n", sep = "")
cat("Plots: ", normalizePath(plot_dir, winslash = "/"), "\n", sep = "")
print(round(bic_compact, 3), row.names = FALSE)
