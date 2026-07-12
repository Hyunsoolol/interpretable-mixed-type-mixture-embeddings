#!/usr/bin/env Rscript

# Paired Stage-2 diagnostic for the Study B two-step procedure.
# Stage 1 selects K by independent test NLL on a dense vMF fit. Stage 2 fixes
# that K and selects the E-CGL support by BIC after an exact centered-Eta refit.
# Existing method files and final Study B results are not modified.

options(stringsAsFactors = FALSE)

getenv_int <- function(name, default) {
  as.integer(Sys.getenv(name, as.character(default)))
}

parse_names <- function(x) {
  trimws(strsplit(x, ",", fixed = TRUE)[[1L]])
}

source_v2_helpers_without_running <- function() {
  path <- file.path(
    "r", "simulation", "paper_eta_studyb_confirmatory_v2_260711.r"
  )
  lines <- readLines(path, warn = FALSE, encoding = "UTF-8")
  if (length(lines) > 0L) lines[1L] <- sub("^\ufeff", "", lines[1L])
  stop_idx <- grep("^raw_rows <- list\\(\\)", lines)[1L]
  if (is.na(stop_idx) || stop_idx <= 1L) {
    stop("Cannot find the Study B v2 execution boundary.")
  }
  eval(parse(text = lines[seq_len(stop_idx - 1L)]), envir = .GlobalEnv)
}

Sys.setenv(
  USE_RCPP_HELPERS = Sys.getenv("USE_RCPP_HELPERS", "1"),
  V2_N_VALUES = "1000",
  V2_EB_VALUES = "0.05",
  V2_METHODS = "E-CGL",
  V2_N_REP = "1",
  V2_D = "200",
  V2_NSTART = Sys.getenv("K_STAGE2_NSTART", "10"),
  V2_MAX_ITER = Sys.getenv("K_STAGE2_MAX_ITER", "100"),
  V2_ETA_STEPS = Sys.getenv("K_STAGE2_ETA_STEPS", "240"),
  V2_TEST_N = Sys.getenv("K_STAGE2_TEST_N", "5000"),
  V2_REFIT_MAX_ITER = Sys.getenv("K_STAGE2_REFIT_MAX_ITER", "160"),
  V2_REFIT_RETRY_MAX_ITER = Sys.getenv("K_STAGE2_REFIT_RETRY_MAX_ITER", "840"),
  V2_REFIT_SHORTLIST = Sys.getenv("K_STAGE2_REFIT_SHORTLIST", "40"),
  V2_REFIT_GUARD_RANK = Sys.getenv("K_STAGE2_REFIT_GUARD_RANK", "38"),
  V2_BASE_SEED = Sys.getenv("K_STAGE2_BASE_SEED", "20260712")
)
source_v2_helpers_without_running()

stage2 <- list(
  label = Sys.getenv(
    "K_STAGE2_LABEL", "paper_eta_k_twostep_stage2_ecgl_rep20_260712"
  ),
  out_dir = Sys.getenv(
    "K_STAGE2_OUT_DIR",
    "results/paper_eta_k_twostep_stage2_ecgl_rep20_260712"
  ),
  n_rep = getenv_int("K_STAGE2_N_REP", 20L),
  n = getenv_int("K_STAGE2_N", 1000L),
  test_n = getenv_int("K_STAGE2_TEST_N", 5000L),
  d = getenv_int("K_STAGE2_D", 200L),
  base_seed = getenv_int("K_STAGE2_BASE_SEED", 20260712L),
  scenarios = parse_names(Sys.getenv(
    "K_STAGE2_SCENARIOS", "equal,heterogeneous"
  )),
  stage1_dir = Sys.getenv(
    "K_STAGE2_STAGE1_DIR", "results/paper_eta_k_stage1_rep20_260712"
  ),
  calibration_path = Sys.getenv(
    "K_STAGE2_CALIBRATION_PATH",
    paste0(
      "results/paper_eta_studyb_v2_refitB_guard40_all6_rep100_260712/",
      "studyb_guard40_all6_rep100_calibration.csv"
    )
  )
)

valid_scenarios <- c("equal", "heterogeneous")
if (!length(stage2$scenarios) || any(!stage2$scenarios %in% valid_scenarios)) {
  stop("K_STAGE2_SCENARIOS must contain equal and/or heterogeneous.")
}
if (stage2$n != 1000L || stage2$d != 200L) {
  stop("This paired diagnostic is fixed to the Stage-1 n=1000, d=200 data.")
}

cfg$n <- stage2$n
cfg$d <- stage2$d
cfg$K <- 4L
cfg$common_q <- 4L
cfg$decision_per_component <- 4L
cfg$target_oracle_error <- 0.05
cfg$base_seed <- stage2$base_seed
cfg$test_n <- stage2$test_n
cfg$eta_steps <- getenv_int("K_STAGE2_ETA_STEPS", 240L)
cfg$nstart <- getenv_int("K_STAGE2_NSTART", 10L)
cfg$max_iter <- getenv_int("K_STAGE2_MAX_ITER", 100L)
cfg$exact_max_iter <- getenv_int("K_STAGE2_REFIT_MAX_ITER", 160L)
cfg$exact_retry_max_iter <- getenv_int(
  "K_STAGE2_REFIT_RETRY_MAX_ITER", 840L
)
cfg$refit_shortlist <- getenv_int("K_STAGE2_REFIT_SHORTLIST", 40L)
cfg$refit_guard_rank <- getenv_int("K_STAGE2_REFIT_GUARD_RANK", 38L)

dir.create(stage2$out_dir, recursive = TRUE, showWarnings = FALSE)

stage1_selection_path <- file.path(
  stage2$stage1_dir, "k_stage1_selection_by_rep.csv"
)
stage1_dense_path <- file.path(stage2$stage1_dir, "k_stage1_dense_final.csv")
if (!file.exists(stage1_selection_path) || !file.exists(stage1_dense_path)) {
  stop("Stage-1 selection or dense-fit audit file is missing.")
}

stage1_selection <- read.csv(stage1_selection_path, stringsAsFactors = FALSE)
stage1_selection <- stage1_selection[
  stage1_selection$criterion == "test_NLL", , drop = FALSE
]
stage1_dense <- read.csv(stage1_dense_path, stringsAsFactors = FALSE)

calibration <- read.csv(stage2$calibration_path, stringsAsFactors = FALSE)
calibration <- calibration[
  abs(calibration$target_oracle_error - 0.05) < 1e-12, , drop = FALSE
]
calibration <- calibration[!duplicated(calibration$kappa_pattern), , drop = FALSE]

scenario_defs <- list(
  equal = list(id = "OBE5_equal_kappa", kappa = rep(45, 4L)),
  heterogeneous = list(
    id = "OBE5_heterogeneous_kappa", kappa = c(30, 40, 50, 60)
  )
)
for (name in names(scenario_defs)) {
  row <- calibration[calibration$kappa_pattern == name, , drop = FALSE]
  if (nrow(row) != 1L) stop("Missing unique calibration row for ", name)
  scenario_defs[[name]]$A <- row$common_norm
  scenario_defs[[name]]$achieved_eB <- row$achieved_oracle_error
}

make_data <- function(scenario_name, rep_id) {
  scenario <- scenario_defs[[scenario_name]]
  params <- make_oracle_eta_params_for_A(cfg, scenario$kappa, scenario$A)
  scenario_index <- match(scenario_name, names(scenario_defs))
  set.seed(stage2$base_seed + scenario_index * 1000003L + rep_id * 1009L)
  train <- simulate_from_params(stage2$n, params)
  set.seed(stage2$base_seed + scenario_index * 2000003L + rep_id * 2017L)
  test <- simulate_from_params(stage2$test_n, params)
  list(params = params, train = train, test = test)
}

stage1_fit_seed <- function(scenario_name, rep_id, K) {
  stage2$base_seed + match(scenario_name, names(scenario_defs)) * 3000001L +
    rep_id * 10007L + K * 101L
}

raw_path <- file.path(stage2$out_dir, "k_twostep_stage2_ecgl_raw.csv")
candidate_path <- file.path(
  stage2$out_dir, "k_twostep_stage2_ecgl_candidate_supports.csv"
)
summary_path <- file.path(stage2$out_dir, "k_twostep_stage2_ecgl_summary.csv")
qa_path <- file.path(stage2$out_dir, "k_twostep_stage2_ecgl_qa.csv")
notes_path <- file.path(stage2$out_dir, "k_twostep_stage2_ecgl_notes.md")
complete_path <- file.path(stage2$out_dir, "k_twostep_stage2_ecgl_complete.ok")

raw <- if (file.exists(raw_path)) {
  read.csv(raw_path, stringsAsFactors = FALSE)
} else {
  data.frame()
}
candidates <- if (file.exists(candidate_path)) {
  read.csv(candidate_path, stringsAsFactors = FALSE)
} else {
  data.frame()
}

append_fill <- function(existing, row) {
  if (is.null(existing) || !nrow(existing)) return(row)
  if (is.null(row) || !nrow(row)) return(existing)
  rbind_fill(list(existing, row))
}

summarize_stage2 <- function(x) {
  if (!nrow(x)) return(data.frame())
  metrics <- c(
    "selected_q", "common_q_selected", "decision_q_selected",
    "noise_q_selected", "TPR", "FPR", "Precision", "F1", "ARI",
    "MSE_centered_eta", "MSE_kappa", "test_NLL", "method_elapsed_sec",
    "dense_loglik_abs_diff", "dense_test_NLL_abs_diff"
  )
  out <- lapply(unique(x$scenario_name), function(name) {
    z <- x[x$scenario_name == name, , drop = FALSE]
    failed <- if ("failed" %in% names(z)) {
      z$failed %in% TRUE
    } else {
      rep(FALSE, nrow(z))
    }
    one <- data.frame(
      scenario = name,
      reps = length(unique(z$rep)),
      K4_selection_rate = mean(z$stage1_selected_K == 4L),
      successful_reps = sum(z$converged %in% TRUE & !failed),
      support_exact_rate = mean(
        z$common_q_selected == 0 & z$decision_q_selected == 16 &
          z$noise_q_selected == 0
      ),
      exact_shortlist_fallback_reps = sum(
        z$exact_shortlist_fallback_full %in% TRUE
      ),
      exact_invalid_candidate_total = sum(
        z$exact_invalid_candidate_count, na.rm = TRUE
      )
    )
    for (metric in metrics) {
      value <- suppressWarnings(as.numeric(z[[metric]]))
      good <- is.finite(value)
      one[[paste0(metric, "_mean")]] <- if (any(good)) mean(value[good]) else NA_real_
      one[[paste0(metric, "_sd")]] <- if (sum(good) > 1L) sd(value[good]) else NA_real_
      one[[paste0(metric, "_mcse")]] <- if (sum(good) > 1L) {
        sd(value[good]) / sqrt(sum(good))
      } else {
        NA_real_
      }
    }
    one
  })
  do.call(rbind, out)
}

write_checkpoint <- function() {
  if (nrow(raw)) {
    write.csv(raw, raw_path, row.names = FALSE)
    write.csv(summarize_stage2(raw), summary_path, row.names = FALSE)
  }
  if (nrow(candidates)) write.csv(candidates, candidate_path, row.names = FALSE)
}

run_started <- proc.time()[["elapsed"]]
for (scenario_name in stage2$scenarios) {
  scenario <- scenario_defs[[scenario_name]]
  for (rep_id in seq_len(stage2$n_rep)) {
    raw_done <- nrow(raw) && any(
      raw$scenario_name == scenario_name & raw$rep == rep_id
    )
    candidate_done <- nrow(candidates) && any(
      candidates$scenario_name == scenario_name & candidates$rep == rep_id
    )
    if (raw_done && candidate_done) next
    if (raw_done != candidate_done) {
      if (nrow(raw)) {
        raw <- raw[!(raw$scenario_name == scenario_name & raw$rep == rep_id),
                   , drop = FALSE]
      }
      if (nrow(candidates)) {
        candidates <- candidates[!(
          candidates$scenario_name == scenario_name & candidates$rep == rep_id
        ), , drop = FALSE]
      }
    }

    selected <- stage1_selection[
      stage1_selection$scenario == scenario$id &
        stage1_selection$rep == rep_id, , drop = FALSE
    ]
    if (nrow(selected) != 1L) {
      stop("Missing unique Stage-1 test-NLL selection for ", scenario$id,
           " rep ", rep_id)
    }
    selected_K <- as.integer(selected$selected_K)
    if (selected_K != 4L) {
      stop("Stage-1 did not select K=4 for ", scenario$id, " rep ", rep_id)
    }

    dat <- make_data(scenario_name, rep_id)
    dense_reference <- stage1_dense[
      stage1_dense$scenario == scenario$id & stage1_dense$rep == rep_id &
        stage1_dense$K_fit == selected_K, , drop = FALSE
    ]
    if (nrow(dense_reference) != 1L) {
      stop("Missing unique Stage-1 dense K=4 fit audit row.")
    }

    dense_seed <- stage1_fit_seed(scenario_name, rep_id, selected_K)
    dense <- fit_svMF_multistart(
      dat$train$X, selected_K, beta = 0, nstart = cfg$nstart,
      shared_kappa = FALSE, seed = dense_seed, max_iter = cfg$max_iter
    )
    dense_train <- e_step_vmf(dat$train$X, dense)
    dense_test <- e_step_vmf(dat$test$X, dense)
    dense_loglik_diff <- abs(dense_train$loglik - dense_reference$loglik)
    dense_test_diff <- abs(
      -dense_test$loglik / nrow(dat$test$X) - dense_reference$test_NLL
    )
    if (!is.finite(dense_loglik_diff) || dense_loglik_diff > 1e-8 ||
        !is.finite(dense_test_diff) || dense_test_diff > 1e-8) {
      stop(sprintf(
        "Stage-1 data/fit reproduction failed for %s rep %d: %.3e, %.3e",
        scenario_name, rep_id, dense_loglik_diff, dense_test_diff
      ))
    }

    cfg$current_init_seed <- dense_seed
    result <- fit_e_group_rules(
      dat$train$X, dat$test$X, dat$train$z, dat$params, dense, "E-CGL",
      scenario$id, rep_id, scenario$achieved_eB
    )
    before <- result$rows[
      result$rows$rule == "current_BIC_before_exact_refit", , drop = FALSE
    ]
    after <- result$rows[
      result$rows$rule == "BIC_after_exact_refit", , drop = FALSE
    ]
    if (nrow(before) != 1L || nrow(after) != 1L) {
      stop("Unexpected E-CGL selector row structure.")
    }

    after$scenario_name <- scenario_name
    after$stage1_criterion <- "independent_test_NLL"
    after$stage1_selected_K <- selected_K
    after$stage1_selection_value <- selected$value
    after$dense_loglik_stage1 <- dense_reference$loglik
    after$dense_loglik_stage2 <- dense_train$loglik
    after$dense_loglik_abs_diff <- dense_loglik_diff
    after$dense_test_NLL_stage1 <- dense_reference$test_NLL
    after$dense_test_NLL_stage2 <- -dense_test$loglik / nrow(dat$test$X)
    after$dense_test_NLL_abs_diff <- dense_test_diff
    after$BIC_before_selected_q <- before$selected_q
    after$BIC_before_F1 <- before$F1
    after$BIC_before_ARI <- before$ARI
    after$BIC_after_minus_before_q <- after$selected_q - before$selected_q
    after$BIC_after_minus_before_F1 <- after$F1 - before$F1
    raw <- append_fill(raw, after)

    cand <- result$candidates
    cand$scenario_name <- scenario_name
    cand$stage1_selected_K <- selected_K
    candidates <- append_fill(candidates, cand)
    write_checkpoint()
    cat(sprintf(
      "[stage2] %s rep %d/%d: K=%d q=%d F1=%.3f ARI=%.3f\n",
      scenario_name, rep_id, stage2$n_rep, selected_K, after$selected_q,
      after$F1, after$ARI
    ))
  }
}

summary <- summarize_stage2(raw)
expected_n <- length(stage2$scenarios) * stage2$n_rep
raw_failed <- if ("failed" %in% names(raw)) {
  raw$failed %in% TRUE
} else {
  rep(FALSE, nrow(raw))
}
qa <- data.frame(
  check = c(
    "final_row_count", "unique_scenario_rep", "stage1_selected_K4",
    "dense_loglik_reproduction", "dense_test_NLL_reproduction",
    "exact_refit_converged", "exact_constraint", "finite_core_metrics",
    "candidate_refits_eligible"
  ),
  observed = c(
    nrow(raw),
    length(unique(paste(raw$scenario_name, raw$rep, sep = "|"))),
    sum(raw$stage1_selected_K == 4L),
    max(raw$dense_loglik_abs_diff, na.rm = TRUE),
    max(raw$dense_test_NLL_abs_diff, na.rm = TRUE),
    sum(raw$converged %in% TRUE & !raw_failed),
    max(raw$constraint_error, na.rm = TRUE),
    sum(apply(
      raw[, c("selected_q", "F1", "ARI", "MSE_centered_eta")],
      1L, function(x) all(is.finite(as.numeric(x)))
    )),
    sum(candidates$exact_eligible %in% TRUE)
  ),
  expected = c(
    expected_n, expected_n, expected_n, "<=1e-8", "<=1e-8",
    expected_n, "<=1e-8", expected_n, nrow(candidates)
  ),
  pass = c(
    nrow(raw) == expected_n,
    length(unique(paste(raw$scenario_name, raw$rep, sep = "|"))) == expected_n,
    all(raw$stage1_selected_K == 4L),
    max(raw$dense_loglik_abs_diff, na.rm = TRUE) <= 1e-8,
    max(raw$dense_test_NLL_abs_diff, na.rm = TRUE) <= 1e-8,
    all(raw$converged %in% TRUE & !raw_failed),
    all(is.finite(raw$constraint_error) & raw$constraint_error <= 1e-8),
    all(apply(
      raw[, c("selected_q", "F1", "ARI", "MSE_centered_eta")],
      1L, function(x) all(is.finite(as.numeric(x)))
    )),
    all(candidates$exact_eligible %in% TRUE)
  ),
  stringsAsFactors = FALSE
)

write.csv(summary, summary_path, row.names = FALSE)
write.csv(qa, qa_path, row.names = FALSE)

fmt <- function(x, digits = 3L) {
  ifelse(is.na(x), "NA", formatC(as.numeric(x), format = "f", digits = digits))
}
notes <- c(
  "# Study B paired two-step E-CGL diagnostic",
  "",
  "## Procedure",
  "",
  "- Stage 1: dense free-kappa vMF; K selected by independent test NLL over K=2,...,8.",
  "- Stage 2: selected K fixed; E-CGL path followed by BIC-after-exact-centered-refit.",
  sprintf(
    "- n=%d, d=%d, target eB=0.05, reps=%d per scenario, Eta path=%d.",
    stage2$n, stage2$d, stage2$n_rep, cfg$eta_steps
  ),
  "- The independent test selector is a simulation diagnostic, not a deployable real-data rule.",
  "",
  "## Results",
  "",
  "| scenario | K=4 rate | selected q | common q | decision q | noise q | F1 | ARI | MSE eta | exact support rate |",
  "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
)
for (i in seq_len(nrow(summary))) {
  notes <- c(notes, sprintf(
    "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |",
    summary$scenario[i], fmt(summary$K4_selection_rate[i]),
    fmt(summary$selected_q_mean[i]), fmt(summary$common_q_selected_mean[i]),
    fmt(summary$decision_q_selected_mean[i]),
    fmt(summary$noise_q_selected_mean[i]), fmt(summary$F1_mean[i]),
    fmt(summary$ARI_mean[i]), fmt(summary$MSE_centered_eta_mean[i]),
    fmt(summary$support_exact_rate[i])
  ))
}
notes <- c(
  notes, "", "## QA", "",
  sprintf("- Final rows: %d/%d.", nrow(raw), expected_n),
  sprintf("- QA checks passed: %d/%d.", sum(qa$pass), nrow(qa)),
  sprintf(
    "- Maximum Stage-1 dense log-likelihood reproduction difference: %.3e.",
    max(raw$dense_loglik_abs_diff, na.rm = TRUE)
  ),
  sprintf(
    "- Exact candidate refits: %d; ineligible: %d.", nrow(candidates),
    sum(!(candidates$exact_eligible %in% TRUE))
  ),
  sprintf(
    "- Full-shortlist fallback reps: %d.",
    sum(raw$exact_shortlist_fallback_full %in% TRUE)
  ),
  sprintf("- Total elapsed time: %.1f seconds.", proc.time()[["elapsed"]] - run_started),
  "",
  "## Interpretation boundary",
  "",
  "- Labels are used only for ARI and support-recovery evaluation.",
  "- Stage-1 independent test NLL uses the known simulation distribution through a held-out sample.",
  "- Candidate-support output is an audit artifact and is not a commit candidate."
)
writeLines(notes, notes_path, useBytes = TRUE)

if (!all(qa$pass)) stop("Stage-2 QA failed; inspect the QA CSV.")
writeLines(
  c(stage2$label, paste0("completed_at=", format(Sys.time(), tz = "UTC", usetz = TRUE))),
  complete_path, useBytes = TRUE
)

cat("Wrote paired Stage-2 outputs to ", normalizePath(stage2$out_dir), "\n", sep = "")
print(summary)
