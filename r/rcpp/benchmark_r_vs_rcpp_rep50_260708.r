#!/usr/bin/env Rscript

# Diagnostic-only R-only vs Rcpp-helper runtime benchmark.
# This script does not modify official R files. It creates temporary runner
# copies under results/ and patches only those copies to use sourceCpp(cacheDir).

cmd_file <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_file <- if (length(cmd_file)) sub("^--file=", "", cmd_file[[1]]) else
  file.path("r", "rcpp", "benchmark_r_vs_rcpp_rep50_260708.r")
repo_root <- normalizePath(file.path(dirname(script_file), "..", ".."),
                           winslash = "/", mustWork = FALSE)
if (!file.exists(file.path(repo_root, "r", "methods", "rb2022_k4_pilot_compare_run.r"))) {
  repo_root <- normalizePath(getwd(), winslash = "/", mustWork = FALSE)
}
setwd(repo_root)

out_root <- file.path("results", "rcpp_vs_r_runtime_benchmark_rep50_260708")
temp_dir <- file.path(out_root, "temp_methods")
cache_dir <- file.path(out_root, "sourcecpp_cache")
dir.create(temp_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)

rossi_src <- file.path("r", "methods", "rossi_barbaro_2022_reproduction.r")
pilot_src <- file.path("r", "methods", "rb2022_k4_pilot_compare_run.r")
rossi_tmp <- file.path(temp_dir, "rossi_barbaro_2022_reproduction.r")
pilot_tmp <- file.path(temp_dir, "rb2022_k4_pilot_compare_run.r")

cache_dir_norm <- normalizePath(cache_dir, winslash = "/", mustWork = FALSE)
rossi_lines <- readLines(rossi_src, warn = FALSE)
rossi_lines <- sub(
  "Rcpp::sourceCpp\\(path\\)",
  sprintf(
    "Rcpp::sourceCpp(path, rebuild = FALSE, cacheDir = %s)",
    deparse(cache_dir_norm)
  ),
  rossi_lines
)
writeLines(rossi_lines, rossi_tmp, useBytes = TRUE)

pilot_lines <- readLines(pilot_src, warn = FALSE)
pilot_lines <- sub(
  "source\\(file.path\\(\"r\", \"methods\", \"rossi_barbaro_2022_reproduction.r\"\\)\\)",
  sprintf("source(%s)", deparse(normalizePath(rossi_tmp, winslash = "/", mustWork = FALSE))),
  pilot_lines
)
writeLines(pilot_lines, pilot_tmp, useBytes = TRUE)

if (!requireNamespace("Rcpp", quietly = TRUE)) {
  stop("Rcpp is not available.")
}

message("[runtime-benchmark] Warming sourceCpp cache...")
warm_time <- system.time({
  Rcpp::sourceCpp(file.path("r", "rcpp", "eta_helpers.cpp"),
                  rebuild = FALSE, cacheDir = cache_dir, showOutput = FALSE)
  Rcpp::sourceCpp(file.path("r", "rcpp", "vmf_e_step_helpers.cpp"),
                  rebuild = FALSE, cacheDir = cache_dir, showOutput = FALSE)
})

common_env <- c(
  "RB2022_K4_PILOT_N_REP=50",
  "RB2022_K4_PILOT_N=300",
  "RB2022_K4_PILOT_D=60",
  "RB2022_K4_PILOT_K=4",
  "RB2022_K4_PILOT_NSTART=3",
  "RB2022_K4_PILOT_MAX_ITER=50",
  "RB2022_K4_PILOT_MAX_PATH_STEPS=40",
  "RB2022_K4_PILOT_BASE_SEED=20260708",
  "RB2022_K4_PILOT_LAMBDA_MU=0,50,100,200",
  "RB2022_K4_PILOT_LAMBDA_KAPPA=0,5,10",
  "RB2022_K4_PILOT_LAMBDA_ETA=0,0.5,1,2,4,8",
  sprintf("RCPP_HELPERS_CACHE_DIR=%s", cache_dir_norm)
)

set_env_pairs <- function(pairs) {
  values <- strsplit(pairs, "=", fixed = TRUE)
  names <- vapply(values, `[`, character(1), 1)
  vals <- vapply(values, function(x) paste(x[-1], collapse = "="), character(1))
  old <- Sys.getenv(names, unset = NA_character_)
  names(old) <- names
  do.call(Sys.setenv, as.list(stats::setNames(vals, names)))
  old
}

restore_env_pairs <- function(old) {
  for (nm in names(old)) {
    if (is.na(old[[nm]])) {
      Sys.unsetenv(nm)
    } else {
      do.call(Sys.setenv, as.list(stats::setNames(old[[nm]], nm)))
    }
  }
}

run_in_process <- function(repeat_id, mode) {
  label <- sprintf("runtime_rep50_%s_r%02d_260708", tolower(mode), repeat_id)
  subdir <- file.path(out_root, sprintf("repeat_%02d", repeat_id), tolower(mode))
  dir.create(subdir, recursive = TRUE, showWarnings = FALSE)
  log_path <- file.path(subdir, paste0(label, ".log"))
  env_pairs <- c(
    common_env,
    sprintf("USE_RCPP_HELPERS=%s", if (mode == "ON") "1" else "0"),
    sprintf("RB2022_K4_PILOT_LABEL=%s", label),
    sprintf("RB2022_K4_PILOT_OUT_DIR=%s", normalizePath(subdir, winslash = "/", mustWork = FALSE))
  )
  old_env <- set_env_pairs(env_pairs)
  status <- 0L
  err <- NULL
  elapsed <- system.time({
    con <- file(log_path, open = "wt")
    sink(con, type = "output")
    sink(con, type = "message")
    tryCatch(
      source(pilot_tmp, local = FALSE),
      error = function(e) {
        status <<- 1L
        err <<- conditionMessage(e)
      }
    )
    sink(type = "message")
    sink(type = "output")
    close(con)
  })
  restore_env_pairs(old_env)
  if (!is.null(err)) {
    write(err, file = log_path, append = TRUE)
  }
  data.frame(
    repeat_id = repeat_id,
    mode = mode,
    label = label,
    out_dir = normalizePath(subdir, winslash = "/", mustWork = FALSE),
    status = status,
    elapsed_sec = unname(elapsed[["elapsed"]]),
    user_sec = unname(elapsed[["user.self"]]),
    system_sec = unname(elapsed[["sys.self"]]),
    log_path = normalizePath(log_path, winslash = "/", mustWork = FALSE),
    error = if (is.null(err)) "" else err
  )
}

compare_df <- function(x, y, name, repeat_id) {
  stopifnot(identical(names(x), names(y)))
  num_cols <- names(x)[vapply(x, is.numeric, logical(1))]
  char_cols <- setdiff(names(x), num_cols)
  diffs <- unlist(lapply(num_cols, function(v) abs(x[[v]] - y[[v]])), use.names = FALSE)
  data.frame(
    repeat_id = repeat_id,
    file = name,
    nrow_off = nrow(x),
    nrow_on = nrow(y),
    numeric_max_diff = if (length(diffs)) max(diffs, na.rm = TRUE) else 0,
    n_num_diff_gt_1e8 = sum(diffs > 1e-8, na.rm = TRUE),
    n_num_diff_gt_1e10 = sum(diffs > 1e-10, na.rm = TRUE),
    all_char_same = all(vapply(
      char_cols,
      function(v) identical(as.character(x[[v]]), as.character(y[[v]])),
      logical(1)
    ))
  )
}

read_output <- function(row, suffix) {
  read.csv(file.path(row$out_dir, paste0(row$label, "_", suffix, ".csv")), check.names = FALSE)
}

timing_rows <- list()
comparison_rows <- list()
idx <- 0L
for (repeat_id in seq_len(3)) {
  message("[runtime-benchmark] Repeat ", repeat_id, " OFF...")
  off <- run_in_process(repeat_id, "OFF")
  message("[runtime-benchmark] Repeat ", repeat_id, " ON...")
  on <- run_in_process(repeat_id, "ON")
  timing_rows[[length(timing_rows) + 1L]] <- off
  timing_rows[[length(timing_rows) + 1L]] <- on

  raw_off <- read_output(off, "raw")
  raw_on <- read_output(on, "raw")
  summary_off <- read_output(off, "summary")
  summary_on <- read_output(on, "summary")
  idx <- idx + 1L
  comparison_rows[[idx]] <- compare_df(raw_off, raw_on, "raw", repeat_id)
  idx <- idx + 1L
  comparison_rows[[idx]] <- compare_df(summary_off, summary_on, "summary", repeat_id)
}

timing <- do.call(rbind, timing_rows)
comparison <- do.call(rbind, comparison_rows)

timing_path <- file.path(out_root, "runtime_timing_repeats.csv")
comparison_path <- file.path(out_root, "runtime_off_on_comparison_repeats.csv")
write.csv(timing, timing_path, row.names = FALSE)
write.csv(comparison, comparison_path, row.names = FALSE)

timing_summary <- do.call(rbind, lapply(split(timing, timing$mode), function(z) {
  data.frame(
    mode = z$mode[[1]],
    repeats = nrow(z),
    mean_elapsed_sec = mean(z$elapsed_sec),
    sd_elapsed_sec = stats::sd(z$elapsed_sec),
    median_elapsed_sec = stats::median(z$elapsed_sec),
    min_elapsed_sec = min(z$elapsed_sec),
    max_elapsed_sec = max(z$elapsed_sec)
  )
}))
timing_summary_path <- file.path(out_root, "runtime_timing_summary.csv")
write.csv(timing_summary, timing_summary_path, row.names = FALSE)

off_med <- timing_summary$median_elapsed_sec[timing_summary$mode == "OFF"]
on_med <- timing_summary$median_elapsed_sec[timing_summary$mode == "ON"]
all_pass <- all(timing$status == 0) &&
  all(comparison$n_num_diff_gt_1e8 == 0) &&
  all(comparison$all_char_same)

notes <- c(
  "# R-only vs Rcpp-helper rep=50 runtime diagnostic",
  "",
  "- Scope: diagnostic-only runtime variability check.",
  "- Comparison: same algorithm, low-level helper replacement only.",
  "- Official source files modified: NO.",
  "- Temporary runner copies: `results/rcpp_vs_r_runtime_benchmark_rep50_260708/temp_methods`.",
  "- Loading route: `Rcpp::sourceCpp(cacheDir=..., rebuild=FALSE)` in the temporary copies.",
  "- Full rep=100 simulation: NO.",
  "- Setting: K=4, n=300, d=60, rep=50, nstart=3, max_iter=50, max_path_steps=40.",
  sprintf("- Cache warm-up elapsed: %.3f sec.", unname(warm_time[["elapsed"]])),
  sprintf("- Equality result: %s at tolerance `1e-8`.", if (all_pass) "PASS" else "FAIL"),
  sprintf("- Median OFF elapsed: %.3f sec.", off_med),
  sprintf("- Median ON elapsed: %.3f sec.", on_med),
  sprintf("- Median OFF/ON ratio: %.3f.", off_med / on_med),
  "",
  "## Timing summary",
  "",
  paste(capture.output(print(timing_summary, row.names = FALSE)), collapse = "\n"),
  "",
  "## Interpretation",
  "",
  "- The repeated benchmark separates sourceCpp cache warm-up from the timed rep=50 runs.",
  "- Results are diagnostic and should not be used as a publication speed claim.",
  "- The R-only path remains the reference implementation and can be restored with `USE_RCPP_HELPERS=0`."
)
notes_path <- file.path(out_root, "runtime_benchmark_notes.md")
writeLines(notes, notes_path, useBytes = TRUE)

print(timing_summary)
print(comparison)
message("[runtime-benchmark] Wrote: ", timing_path)
message("[runtime-benchmark] Wrote: ", timing_summary_path)
message("[runtime-benchmark] Wrote: ", comparison_path)
message("[runtime-benchmark] Wrote: ", notes_path)

if (!all_pass) {
  stop("R-only vs Rcpp-helper runtime benchmark equality failed.")
}
