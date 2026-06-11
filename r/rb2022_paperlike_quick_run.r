# ==============================================================================
# Quick paper-like reproduction run for Rossi & Barbaro (2022)
# ------------------------------------------------------------------------------
# This driver keeps the paper's main simulation grid, but uses fewer repetitions
# and fewer random starts so it can finish in an interactive research meeting.
#
# Paper-like parts:
#   - N in {200, 1000}
#   - d = 100
#   - true K = 4
#   - candidate K in {1, ..., 6}
#   - overlap in {2.5%, 5%}
#   - nonzero fraction in {5%, 10%, 15%}
#
# Shortened parts:
#   - n_rep = 3 instead of 100
#   - nstart = 2 instead of a larger multi-start budget
#   - max_path_steps = 350
#
# Results are appended after every cell, so an interrupted run still leaves usable
# partial output.
# ==============================================================================

source(file.path("r", "rossi_barbaro_2022_reproduction.r"))

cfg <- list(
  n_rep = 3,
  n_grid = c(200, 1000),
  d = 100,
  K_true = 4,
  K_fit_grid = 1:6,
  overlap_grid = c(0.025, 0.05),
  nonzero_fraction_grid = c(0.05, 0.10, 0.15),
  nstart = 2,
  max_path_steps = 350,
  base_seed = 20260601,
  out_dir = "results/rb2022_paperlike_quick_260601"
)

if (!dir.exists(cfg$out_dir)) dir.create(cfg$out_dir, recursive = TRUE)

raw_path <- file.path(cfg$out_dir, "rb2022_paperlike_quick_raw.csv")
summary_path <- file.path(cfg$out_dir, "rb2022_paperlike_quick_summary.csv")

if (file.exists(raw_path)) file.remove(raw_path)

cell_id <- 1L
total_cells <- cfg$n_rep * length(cfg$n_grid) *
  length(cfg$overlap_grid) * length(cfg$nonzero_fraction_grid)

for (rep_id in seq_len(cfg$n_rep)) {
  for (n in cfg$n_grid) {
    for (overlap in cfg$overlap_grid) {
      for (nonzero_fraction in cfg$nonzero_fraction_grid) {
        seed <- cfg$base_seed + 100000L * rep_id +
          1000L * as.integer(1000 * overlap) +
          as.integer(1000 * nonzero_fraction) + n

        cat(sprintf(
          "\n[%03d/%03d] rep=%d n=%d overlap=%.3f nonzero=%.2f\n",
          cell_id, total_cells, rep_id, n, overlap, nonzero_fraction
        ))

        one <- run_rb2022_one(
          rep_id = rep_id,
          n = n,
          d = cfg$d,
          K_true = cfg$K_true,
          K_fit_grid = cfg$K_fit_grid,
          overlap = overlap,
          nonzero_fraction = nonzero_fraction,
          nstart = cfg$nstart,
          max_path_steps = cfg$max_path_steps,
          seed = seed,
          verbose = FALSE
        )

        write.table(
          one,
          raw_path,
          sep = ",",
          row.names = FALSE,
          col.names = !file.exists(raw_path),
          append = file.exists(raw_path)
        )

        cell_id <- cell_id + 1L
      }
    }
  }
}

raw <- read.csv(raw_path)
summary <- summarize_rb2022(raw)
write.csv(summary, summary_path, row.names = FALSE)

cat("\nDone.\n")
cat("Raw: ", normalizePath(raw_path, winslash = "/"), "\n", sep = "")
cat("Summary: ", normalizePath(summary_path, winslash = "/"), "\n", sep = "")

focus <- subset(
  summary,
  selected == TRUE &
    (method == "spherical_kmeans" |
       criterion %in% c("BIC", "EBIC"))
)

print(
  focus[, c(
    "n", "overlap", "nonzero_fraction", "method", "criterion", "reps",
    "beta_mean", "ARI_mean", "nnz_fraction_mean", "K_fit",
    "BIC_mean", "EBIC_mean"
  )],
  row.names = FALSE
)
