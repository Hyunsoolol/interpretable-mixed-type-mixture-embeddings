# ==============================================================================
# Rossi & Barbaro (2022) paper-sparsity corrected reproduction
# ------------------------------------------------------------------------------
# Paper Figure 13/15/16 use "sparsity" as the fraction of zero coordinates in
# the directional means. The existing reproduction driver uses nonzero_fraction.
#
# Therefore paper sparsity = 0.10 corresponds to nonzero_fraction = 0.90.
# This script runs the single comparison cell needed for thesis-meeting_260622:
#   N = 1000, d = 100, K = 4, overlap = 0.05, paper sparsity = 0.10.
# ==============================================================================

source(file.path("r", "rossi_barbaro_2022_reproduction.r"))

run_label <- Sys.getenv(
  "RB2022_CORRECTED_LABEL",
  "rb2022_paper_sparsity_corrected_ov05_sp010_rep100_260610"
)
out_dir <- Sys.getenv(
  "RB2022_CORRECTED_OUT_DIR",
  file.path("results", run_label)
)

n_rep <- as.integer(Sys.getenv("RB2022_CORRECTED_N_REP", "100"))
nstart <- as.integer(Sys.getenv("RB2022_CORRECTED_NSTART", "10"))
max_path_steps <- as.integer(Sys.getenv("RB2022_CORRECTED_MAX_PATH_STEPS", "700"))

paper_sparsity <- 0.10
nonzero_fraction <- 1 - paper_sparsity

cat(sprintf(
  "Corrected paper-sparsity run: reps=%d, nstart=%d, max_path_steps=%d, paper_sparsity=%.2f, nonzero_fraction=%.2f\n",
  n_rep, nstart, max_path_steps, paper_sparsity, nonzero_fraction
))

res <- run_rb2022_reproduction(
  n_rep = n_rep,
  n_grid = c(1000),
  d = 100,
  K_true = 4,
  K_fit_grid = c(4),
  overlap_grid = c(0.05),
  nonzero_fraction_grid = c(nonzero_fraction),
  nstart = nstart,
  max_path_steps = max_path_steps,
  base_seed = 20260610,
  out_dir = out_dir,
  verbose = TRUE
)

raw <- res$results

target <- subset(
  raw,
  method == "sparse_vmf_selected_K_beta" &
    selected == TRUE &
    criterion == "BIC" &
    K_fit == 4
)

total_entries <- 4 * 100
true_nonzero_entries <- 4 * round(nonzero_fraction * 100)
true_zero_entries <- total_entries - true_nonzero_entries

tp_nonzero <- target$entry_recall * true_nonzero_entries
pred_nonzero <- tp_nonzero / target$entry_precision
fp_nonzero <- pred_nonzero - tp_nonzero
tp_zero <- true_zero_entries - fp_nonzero
fn_zero <- true_nonzero_entries - tp_nonzero
pred_zero <- tp_zero + fn_zero

zero_precision <- tp_zero / pred_zero
zero_recall <- tp_zero / true_zero_entries

zero_metrics <- data.frame(
  overlap = 0.05,
  paper_sparsity = paper_sparsity,
  nonzero_fraction = nonzero_fraction,
  selection = "BIC",
  ARI = mean(target$ARI, na.rm = TRUE),
  beta = mean(target$beta, na.rm = TRUE),
  achieved_sparsity = mean(1 - target$nnz_fraction, na.rm = TRUE),
  nnz_fraction = mean(target$nnz_fraction, na.rm = TRUE),
  zero_precision = mean(zero_precision, na.rm = TRUE),
  zero_recall = mean(zero_recall, na.rm = TRUE),
  entry_precision = mean(target$entry_precision, na.rm = TRUE),
  entry_recall = mean(target$entry_recall, na.rm = TRUE)
)

write.csv(
  zero_metrics,
  file.path(out_dir, sprintf("%s_paper_compare_summary.csv", run_label)),
  row.names = FALSE
)

print(zero_metrics)
