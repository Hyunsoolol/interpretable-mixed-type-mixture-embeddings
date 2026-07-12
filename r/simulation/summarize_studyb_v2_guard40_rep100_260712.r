options(stringsAsFactors = FALSE)

results_root <- "results"
input_pattern <- paste0(
  "^paper_eta_studyb_v2_refitB_guard40_all6_",
  "(equal|hetero)_eb(025|05|10)_n(300|1000)_rep100_path240_260712$"
)
output_dir <- file.path(
  results_root, "paper_eta_studyb_v2_refitB_guard40_all6_rep100_260712"
)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

input_dirs <- list.dirs(results_root, recursive = FALSE, full.names = TRUE)
input_dirs <- input_dirs[grepl(input_pattern, basename(input_dirs))]
input_dirs <- sort(input_dirs)
if (length(input_dirs) != 12L) {
  stop(sprintf("Expected 12 completed Study B directories; found %d.", length(input_dirs)))
}

one_file <- function(directory, suffix) {
  files <- list.files(
    directory, pattern = paste0(suffix, "$"), full.names = TRUE
  )
  if (length(files) != 1L) {
    stop(sprintf("Expected one *%s in %s; found %d.", suffix, directory, length(files)))
  }
  files
}

read_one <- function(directory, suffix) {
  read.csv(one_file(directory, suffix), check.names = FALSE)
}

safe_mean <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  if (any(is.finite(x))) mean(x[is.finite(x)]) else NA_real_
}

safe_sd <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x)]
  if (length(x) > 1L) sd(x) else NA_real_
}

rbind_fill <- function(xs) {
  all_names <- unique(unlist(lapply(xs, names), use.names = FALSE))
  xs <- lapply(xs, function(x) {
    missing <- setdiff(all_names, names(x))
    for (nm in missing) x[[nm]] <- NA
    x[, all_names, drop = FALSE]
  })
  do.call(rbind, xs)
}

all_raw <- list()
all_final <- list()
all_candidates <- list()
qa_rows <- list()
calibration_rows <- list()

standard_methods <- c("D-L", "D-GL", "D-AGL", "E-L")
group_methods <- c("E-CGL", "E-ACGL")

for (i in seq_along(input_dirs)) {
  directory <- input_dirs[i]
  label <- basename(directory)
  raw <- read_one(directory, "_raw\\.csv")
  candidates <- read_one(directory, "_candidate_supports\\.csv")
  status <- read_one(directory, "_status\\.csv")
  calibration <- read_one(directory, "_calibration\\.csv")
  marker <- list.files(directory, pattern = "_complete\\.ok$", full.names = TRUE)

  identity_match <- regmatches(
    label,
    regexec(
      "_(equal|hetero)_eb(025|05|10)_n(300|1000)_rep100_path240_260712$",
      label
    )
  )[[1L]]
  if (length(identity_match) != 4L) stop("Cannot parse result label: ", label)
  kappa_tag <- identity_match[2L]
  eb_tag <- identity_match[3L]
  expected_n <- as.integer(identity_match[4L])
  expected_eB <- switch(eb_tag, "025" = 0.025, "05" = 0.05, "10" = 0.10)
  kappa_pattern <- if (kappa_tag == "equal") {
    "equal"
  } else {
    "heterogeneous"
  }
  expected_cell <- sprintf(
    "eB%03d_n%d_%s", round(1000 * expected_eB), expected_n, kappa_pattern
  )
  raw$result_label <- label
  raw$kappa_pattern <- kappa_pattern
  candidates$result_label <- label
  candidates$kappa_pattern <- kappa_pattern
  calibration$result_label <- label
  calibration$kappa_pattern <- kappa_pattern

  final <- raw[
    (raw$method %in% standard_methods &
       raw$rule == "current_BIC_before_support_refit") |
      (raw$method %in% group_methods & raw$rule == "BIC_after_exact_refit"),
    , drop = FALSE
  ]
  zero_support <- is.finite(final$selected_q) & final$selected_q == 0
  final$F1_all <- ifelse(
    zero_support, 0, ifelse(is.finite(final$F1), final$F1, NA_real_)
  )
  final$ARI_all <- ifelse(
    zero_support, 0, ifelse(is.finite(final$ARI), final$ARI, NA_real_)
  )
  final$computational_failure <- grepl("^ERROR:", final$refit_status)

  eligible <- candidates$exact_eligible %in% TRUE &
    is.finite(candidates$BIC_after_selection)
  candidate_groups <- split(
    seq_len(nrow(candidates)), paste(candidates$method, candidates$rep, sep = "|")
  )
  winners <- do.call(rbind, lapply(candidate_groups, function(indices) {
    one <- candidates[indices, , drop = FALSE]
    ok <- one$exact_eligible %in% TRUE & is.finite(one$BIC_after_selection)
    if (!any(ok)) return(NULL)
    one[which.min(one$BIC_after_selection), , drop = FALSE]
  }))
  expected_candidate_keys <- as.vector(outer(
    seq_len(100L), group_methods,
    function(rep, method) paste(rep, method, sep = "|")
  ))
  actual_candidate_keys <- paste(candidates$rep, candidates$method, sep = "|")
  candidate_key_counts <- table(
    factor(unique(actual_candidate_keys), levels = expected_candidate_keys)
  )
  candidate_keys_exact <-
    length(unique(actual_candidate_keys)) == length(expected_candidate_keys) &&
    all(candidate_key_counts == 1L) &&
    all(actual_candidate_keys %in% expected_candidate_keys)

  selected_group_raw <- final[final$method %in% group_methods, , drop = FALSE]
  winner_match <- merge(
    winners[, c("method", "rep", "source_path_index", "BIC_after_exact")],
    selected_group_raw[, c("method", "rep", "selected_path_index", "BIC")],
    by = c("method", "rep"), suffixes = c("_candidate", "_raw"), sort = TRUE
  )
  winner_matches_raw <- nrow(winner_match) == 200L &&
    all(winner_match$source_path_index == winner_match$selected_path_index) &&
    all(abs(winner_match$BIC_after_exact - winner_match$BIC) <= 1e-8)

  fallback_guard_ok <- all(vapply(candidate_groups, function(indices) {
    one <- candidates[indices, , drop = FALSE]
    ok <- one$exact_eligible %in% TRUE & is.finite(one$BIC_after_selection)
    if (!any(ok)) return(FALSE)
    winner <- one[which.min(one$BIC_after_selection), , drop = FALSE]
    fallback <- any(one$shortlist_fallback_full %in% TRUE)
    shortlist_applied <- any(one$shortlist_applied %in% TRUE)
    unique_count <- unique(as.integer(one$unique_support_count))
    if (length(unique_count) != 1L || !is.finite(unique_count)) return(FALSE)
    support_keys_unique <- length(unique(one$support_key)) == nrow(one)
    if (!support_keys_unique) return(FALSE)
    full_evaluated <- nrow(one) == unique_count &&
      length(unique(one$support_key)) == unique_count
    ranks <- as.integer(one$rank_before_main)
    if (anyNA(ranks) || length(unique(ranks)) != nrow(one)) return(FALSE)
    expected_ranks <- if (full_evaluated) {
      seq_len(unique_count)
    } else {
      seq_len(min(40L, unique_count))
    }
    if (!identical(sort(ranks), expected_ranks)) return(FALSE)
    if (fallback && !full_evaluated) return(FALSE)
    if (shortlist_applied && !full_evaluated) {
      if (winner$rank_before_main >= 38L && !fallback) return(FALSE)
    }
    TRUE
  }, logical(1L)))

  expected_raw_rows <- 800L
  expected_final_rows <- 600L
  expected_keys <- c(
    as.vector(outer(
      seq_len(100L), standard_methods,
      function(rep, method) paste(rep, method, "current_BIC_before_support_refit", sep = "|")
    )),
    as.vector(outer(
      seq_len(100L), group_methods,
      function(rep, method) paste(rep, method, "current_BIC_before_exact_refit", sep = "|")
    )),
    as.vector(outer(
      seq_len(100L), group_methods,
      function(rep, method) paste(rep, method, "BIC_after_exact_refit", sep = "|")
    ))
  )
  actual_keys <- paste(raw$rep, raw$method, raw$rule, sep = "|")
  key_counts <- table(factor(actual_keys, levels = expected_keys))
  group_selected <- final[final$method %in% group_methods, , drop = FALSE]
  fallback_keys <- unique(paste(
    candidates$method[candidates$shortlist_fallback_full %in% TRUE],
    candidates$rep[candidates$shortlist_fallback_full %in% TRUE], sep = "|"
  ))
  fallback_keys <- fallback_keys[nzchar(fallback_keys)]

  target <- unique(suppressWarnings(as.numeric(calibration$target_oracle_error)))
  achieved <- unique(suppressWarnings(as.numeric(calibration$achieved_oracle_error)))
  ci_low <- unique(suppressWarnings(as.numeric(calibration$achieved_ci_low)))
  ci_high <- unique(suppressWarnings(as.numeric(calibration$achieved_ci_high)))
  calibration_identity_ok <- nrow(calibration) == 1L && length(target) == 1L &&
    abs(target - expected_eB) <= 1e-12
  calibration_abs_error <- if (length(achieved) == 1L) {
    abs(achieved - expected_eB)
  } else {
    Inf
  }
  calibration_tolerance <- max(0.0025, 0.10 * expected_eB)
  calibration_target_in_ci <- length(ci_low) == 1L && length(ci_high) == 1L &&
    is.finite(ci_low) && is.finite(ci_high) &&
    expected_eB >= ci_low && expected_eB <= ci_high
  calibration_ok <- calibration_identity_ok && length(achieved) == 1L &&
    length(ci_low) == 1L && length(ci_high) == 1L &&
    is.finite(target) && is.finite(achieved) &&
    calibration_abs_error <= calibration_tolerance
  cell_identity_ok <-
    identical(unique(raw$cell), expected_cell) &&
    identical(unique(candidates$cell), expected_cell) &&
    nrow(status) == 1L && identical(status$cell[1L], expected_cell) &&
    length(unique(raw$n)) == 1L && as.integer(unique(raw$n)) == expected_n &&
    length(unique(raw$target_oracle_error)) == 1L &&
    abs(as.numeric(unique(raw$target_oracle_error)) - expected_eB) <= 1e-12 &&
    all(abs(as.numeric(raw$achieved_oracle_error) - achieved) <= 1e-12)
  final_failures <- final$computational_failure %in% TRUE
  standard_failure_rows <- sum(
    final_failures & final$method %in% standard_methods, na.rm = TRUE
  )
  unexpected_failure_rows <- sum(
    final_failures & !final$method %in% standard_methods, na.rm = TRUE
  )
  terminal_artifact_ok <- if (standard_failure_rows == 0L) {
    length(marker) == 1L && nrow(status) == 1L &&
      status$status[1L] == "complete" && status$error_rows[1L] == 0L &&
      file.info(file.path(directory, "run.err"))$size == 0
  } else {
    length(marker) == 0L && nrow(status) == 1L &&
      status$status[1L] == "failed_validation" &&
      as.integer(status$error_rows[1L]) == standard_failure_rows
  }

  qa_rows[[i]] <- data.frame(
    result_label = label,
    expected_cell = expected_cell,
    cell_identity_ok = cell_identity_ok,
    complete_marker = length(marker) == 1L,
    status_complete = nrow(status) == 1L && status$status[1L] == "complete",
    terminal_artifact_ok = terminal_artifact_ok,
    completed_reps = if (nrow(status)) as.integer(status$completed_reps[1L]) else NA_integer_,
    raw_rows = nrow(raw), expected_raw_rows = expected_raw_rows,
    final_rows = nrow(final), expected_final_rows = expected_final_rows,
    row_keys_exact = length(actual_keys) == length(expected_keys) && all(key_counts == 1L),
    standard_failure_rows = standard_failure_rows,
    unexpected_failure_rows = unexpected_failure_rows,
    invalid_candidate_refits = sum(!eligible),
    candidate_method_reps = length(candidate_groups),
    candidate_keys_exact = candidate_keys_exact,
    winner_rows = if (is.null(winners)) 0L else nrow(winners),
    winner_matches_raw = winner_matches_raw,
    winner_rank_max = if (is.null(winners)) NA_real_ else max(winners$rank_before_main),
    fallback_method_reps = length(fallback_keys),
    fallback_guard_ok = fallback_guard_ok,
    group_selected_converged = nrow(group_selected) == 200L &&
      all(group_selected$converged %in% TRUE),
    max_group_constraint_error = if (nrow(group_selected)) {
      max(group_selected$constraint_error, na.rm = TRUE)
    } else {
      NA_real_
    },
    zero_support_final_rows = sum(final$selected_q == 0, na.rm = TRUE),
    calibration_abs_error = calibration_abs_error,
    calibration_tolerance = calibration_tolerance,
    calibration_target_in_ci = calibration_target_in_ci,
    calibration_ok = calibration_ok,
    error_log_bytes = file.info(file.path(directory, "run.err"))$size,
    stringsAsFactors = FALSE
  )

  all_raw[[i]] <- raw
  all_final[[i]] <- final
  all_candidates[[i]] <- candidates
  calibration_rows[[i]] <- calibration
}

raw <- rbind_fill(all_raw)
final <- rbind_fill(all_final)
candidates <- rbind_fill(all_candidates)
qa <- do.call(rbind, qa_rows)
calibration <- do.call(rbind, calibration_rows)

summary_metrics <- c(
  "selected_q", "common_q_selected", "decision_q_selected", "noise_q_selected",
  "TPR", "FPR", "Precision", "F1_all", "ARI_all", "MSE_mu", "MSE_kappa",
  "MSE_centered_eta", "loglik", "test_NLL"
)
summary_groups <- split(
  seq_len(nrow(final)),
  interaction(
    final$target_oracle_error, final$n, final$kappa_pattern, final$method,
    drop = TRUE, lex.order = TRUE
  )
)
summary <- do.call(rbind, lapply(summary_groups, function(indices) {
  one <- final[indices, , drop = FALSE]
  out <- data.frame(
    target_eB = one$target_oracle_error[1L], n = one$n[1L],
    kappa_pattern = one$kappa_pattern[1L], method = one$method[1L],
    reps = nrow(one),
    successful_reps = sum(!(one$computational_failure %in% TRUE)),
    computational_failure_reps = sum(one$computational_failure %in% TRUE),
    zero_support_reps = sum(one$selected_q == 0, na.rm = TRUE),
    stringsAsFactors = FALSE
  )
  for (metric in summary_metrics) {
    values <- one[[metric]]
    valid_n <- sum(is.finite(suppressWarnings(as.numeric(values))))
    out[[paste0(metric, "_valid_n")]] <- valid_n
    out[[paste0(metric, "_mean")]] <- safe_mean(values)
    out[[paste0(metric, "_sd")]] <- safe_sd(values)
    out[[paste0(metric, "_mcse")]] <- if (valid_n > 1L) {
      safe_sd(values) / sqrt(valid_n)
    } else {
      NA_real_
    }
  }
  out
}))
summary <- summary[order(
  summary$target_eB, summary$n, summary$kappa_pattern,
  match(summary$method, c(standard_methods, group_methods))
), ]

e_cgl <- final[final$method == "E-CGL", , drop = FALSE]
e_acgl <- final[final$method == "E-ACGL", , drop = FALSE]
paired <- merge(
  e_cgl, e_acgl,
  by = c("result_label", "target_oracle_error", "n", "kappa_pattern", "rep"),
  suffixes = c("_CGL", "_ACGL"), sort = TRUE
)
paired_metrics <- c(
  "selected_q", "common_q_selected", "decision_q_selected", "noise_q_selected",
  "F1_all", "ARI_all", "MSE_centered_eta", "MSE_kappa"
)
paired_groups <- split(
  seq_len(nrow(paired)),
  interaction(
    paired$target_oracle_error, paired$n, paired$kappa_pattern,
    drop = TRUE, lex.order = TRUE
  )
)
paired_summary <- do.call(rbind, lapply(paired_groups, function(indices) {
  one <- paired[indices, , drop = FALSE]
  out <- data.frame(
    target_eB = one$target_oracle_error[1L], n = one$n[1L],
    kappa_pattern = one$kappa_pattern[1L], paired_reps = nrow(one),
    contrast = "E-ACGL minus E-CGL", stringsAsFactors = FALSE
  )
  for (metric in paired_metrics) {
    delta <- one[[paste0(metric, "_ACGL")]] - one[[paste0(metric, "_CGL")]]
    delta <- delta[is.finite(delta)]
    mean_delta <- if (length(delta)) mean(delta) else NA_real_
    se_delta <- if (length(delta) > 1L) sd(delta) / sqrt(length(delta)) else NA_real_
    critical <- if (length(delta) > 1L) qt(0.975, df = length(delta) - 1L) else NA_real_
    out[[paste0(metric, "_n")]] <- length(delta)
    out[[paste0(metric, "_mean_delta")]] <- mean_delta
    out[[paste0(metric, "_mcse")]] <- se_delta
    out[[paste0(metric, "_ci_low")]] <- mean_delta - critical * se_delta
    out[[paste0(metric, "_ci_high")]] <- mean_delta + critical * se_delta
  }
  out
}))

summary_groups_ok <- nrow(summary) == 72L && all(summary$reps == 100L)
core_valid_counts_ok <- all(
  summary$selected_q_valid_n == summary$successful_reps &
    summary$common_q_selected_valid_n == summary$successful_reps &
    summary$decision_q_selected_valid_n == summary$successful_reps &
    summary$noise_q_selected_valid_n == summary$successful_reps &
    summary$F1_all_valid_n == summary$successful_reps &
    summary$ARI_all_valid_n == summary$successful_reps
) && all(summary$successful_reps + summary$computational_failure_reps == 100L)
paired_groups_ok <- nrow(paired_summary) == 12L &&
  all(paired_summary$paired_reps == 100L) && nrow(paired) == 1200L
paired_n_columns <- grep("_n$", names(paired_summary), value = TRUE)
paired_metric_counts_ok <- length(paired_n_columns) == length(paired_metrics) &&
  all(vapply(paired_summary[, paired_n_columns, drop = FALSE], function(x) {
    all(as.integer(x) == 100L)
  }, logical(1L)))
calibration_groups <- split(
  seq_len(nrow(calibration)),
  interaction(
    calibration$target_oracle_error, calibration$kappa_pattern,
    drop = TRUE, lex.order = TRUE
  )
)
calibration_cross_n_ok <- length(calibration_groups) == 6L &&
  all(vapply(calibration_groups, function(indices) {
    one <- calibration[indices, , drop = FALSE]
    if (nrow(one) != 2L) return(FALSE)
    compare_names <- setdiff(names(one), c("result_label"))
    all(vapply(one[, compare_names, drop = FALSE], function(x) {
      if (is.numeric(x)) {
        all(is.finite(x)) && diff(range(x)) <= 1e-12
      } else {
        length(unique(as.character(x))) == 1L
      }
    }, logical(1L)))
  }, logical(1L)))
exact_retry_candidates <- if ("exact_retry_used" %in% names(candidates)) {
  sum(candidates$exact_retry_used %in% TRUE, na.rm = TRUE)
} else {
  0L
}
max_exact_total_iter <- if ("exact_iter" %in% names(candidates)) {
  max(suppressWarnings(as.numeric(candidates$exact_iter)), na.rm = TRUE)
} else {
  NA_real_
}

qa$pass <- with(
  qa,
  cell_identity_ok & terminal_artifact_ok & completed_reps == 100L &
    raw_rows == expected_raw_rows & final_rows == expected_final_rows &
    row_keys_exact & unexpected_failure_rows == 0L &
    invalid_candidate_refits == 0L &
    candidate_method_reps == 200L & candidate_keys_exact &
    winner_rows == 200L & winner_matches_raw & fallback_guard_ok &
    group_selected_converged & is.finite(max_group_constraint_error) &
    max_group_constraint_error <= 1e-8 & calibration_ok
)
qa$summary_groups_ok <- summary_groups_ok
qa$core_valid_counts_ok <- core_valid_counts_ok
qa$paired_groups_ok <- paired_groups_ok
qa$paired_metric_counts_ok <- paired_metric_counts_ok
qa$calibration_cross_n_ok <- calibration_cross_n_ok
global_pass <- all(qa$pass) && summary_groups_ok && core_valid_counts_ok &&
  paired_groups_ok && paired_metric_counts_ok && calibration_cross_n_ok

summary_path <- file.path(output_dir, "studyb_guard40_all6_rep100_summary.csv")
paired_path <- file.path(output_dir, "studyb_guard40_all6_rep100_paired_e_methods.csv")
qa_path <- file.path(output_dir, "studyb_guard40_all6_rep100_qa.csv")
calibration_path <- file.path(output_dir, "studyb_guard40_all6_rep100_calibration.csv")
notes_path <- file.path(output_dir, "studyb_guard40_all6_rep100_notes.md")
write.csv(summary, summary_path, row.names = FALSE)
write.csv(paired_summary, paired_path, row.names = FALSE)
write.csv(qa, qa_path, row.names = FALSE)
write.csv(calibration, calibration_path, row.names = FALSE)

notes <- c(
  "# Study B guarded exact-refit rep=100 QA",
  "",
  "- All six methods use the same generated data within each `(cell, rep)`.",
  "- E-CGL is the main method; E-ACGL is the adaptive extension.",
  "- E-CGL/E-ACGL use BIC-after exact centered-support refits.",
  "- Shortlist40 is guarded: a winner at rank 38 or higher triggers full refitting.",
  "- Failed or non-converged exact refits are ineligible, and any unresolved exact candidate fails QA.",
  "- Standard-method computational failures remain explicit; metric means are conditional on successful fits and retain their valid counts.",
  "- Zero-support outcomes remain in unconditional F1/ARI summaries as zero rather than being dropped.",
  "",
  sprintf("- Audited cells passing integrity checks: %d/12.", sum(qa$pass)),
  sprintf("- Standard-method computational failures: %d/4,800 attempts.", sum(qa$standard_failure_rows)),
  sprintf("- Exact candidates requiring deterministic continuation: %d.", exact_retry_candidates),
  sprintf("- Maximum exact-refit outer iterations after continuation: %s.", max_exact_total_iter),
  sprintf("- Total final method-replicate rows: %d (expected 7200).", nrow(final)),
  sprintf("- Maximum winner BIC-before rank: %s.", max(qa$winner_rank_max)),
  sprintf("- Full-fallback method-replicates: %d.", sum(qa$fallback_method_reps)),
  sprintf("- Zero-support final rows: %d.", sum(qa$zero_support_final_rows)),
  sprintf("- Maximum selected group constraint error: %.3e.", max(qa$max_group_constraint_error)),
  sprintf("- Summary groups complete: %s (72 expected).", summary_groups_ok),
  sprintf("- Paired E-method groups complete: %s (1,200 pairs expected).", paired_groups_ok),
  sprintf("- Paired metric counts complete: %s (100 per cell expected).", paired_metric_counts_ok),
  sprintf("- Calibration identical across n within each design: %s.", calibration_cross_n_ok),
  sprintf("- Calibration targets inside independent MC intervals: %d/12 cells.", sum(qa$calibration_target_in_ci)),
  sprintf("- Maximum absolute target-achieved error: %.4f.", max(qa$calibration_abs_error)),
  "",
  "This output is a computational audit artifact. Interpretation and document updates require a separate review."
)
writeLines(notes, notes_path)

cat(sprintf("Study B QA passed cells: %d/12\n", sum(qa$pass)))
cat(sprintf("Summary: %s\n", normalizePath(summary_path, winslash = "/")))
cat(sprintf("QA: %s\n", normalizePath(qa_path, winslash = "/")))
if (!global_pass) quit(status = 1L)
