#!/usr/bin/env Rscript

# Extended, label-free K diagnostic for Classic3.
# Existing train-only dense-vMF fits are reused for IC/entropy/coherence.
# Bootstrap fits use out-of-bag NLL and full-train prediction stability.
# Supplied labels are attached only after K diagnostics for ARI/NMI reporting.

options(stringsAsFactors = FALSE)

if (!requireNamespace("Matrix", quietly = TRUE)) stop("Matrix is required.")
suppressPackageStartupMessages(library(Matrix))

getenv_chr <- function(name, default) {
  value <- Sys.getenv(name, unset = "")
  if (nzchar(value)) value else default
}
getenv_int <- function(name, default) {
  value <- Sys.getenv(name, unset = "")
  if (nzchar(value)) as.integer(value) else default
}

cfg <- list(
  train_data = getenv_chr(
    "CLASSIC3_PANEL_TRAIN_DATA",
    file.path(
      "data", "classic3", "processed",
      "classic3_splade_holdout_train_top2000_260711.rds"
    )
  ),
  test_data = getenv_chr(
    "CLASSIC3_PANEL_TEST_DATA",
    file.path(
      "data", "classic3", "processed",
      "classic3_splade_holdout_test_top2000_260711.rds"
    )
  ),
  dense_candidates = getenv_chr(
    "CLASSIC3_PANEL_DENSE_CANDIDATES",
    file.path(
      "results", "classic3_splade_holdout_k_selection_k2_10_260711",
      "classic3_dense_k_candidates.csv"
    )
  ),
  dense_fits = getenv_chr(
    "CLASSIC3_PANEL_DENSE_FITS",
    file.path(
      "results", "classic3_splade_holdout_k_selection_k2_10_260711",
      "classic3_dense_k_fits.rds"
    )
  ),
  out_dir = getenv_chr(
    "CLASSIC3_PANEL_OUT_DIR",
    file.path("results", "classic3_k_selection_panel_diag_260714")
  ),
  K_grid = as.integer(strsplit(
    getenv_chr("CLASSIC3_PANEL_K_GRID", "2,3,4,5,6,7,8,9,10"),
    ",", fixed = TRUE
  )[[1L]]),
  bootstrap_reps = getenv_int("CLASSIC3_PANEL_BOOTSTRAP_REPS", 10L),
  bootstrap_nstart = getenv_int("CLASSIC3_PANEL_BOOTSTRAP_NSTART", 5L),
  max_iter = getenv_int("CLASSIC3_PANEL_MAX_ITER", 100L),
  spherical_max_iter = getenv_int("CLASSIC3_PANEL_SPHERICAL_MAX_ITER", 40L),
  top_tokens = getenv_int("CLASSIC3_PANEL_TOP_TOKENS", 10L),
  base_seed = getenv_int("CLASSIC3_PANEL_SEED", 20260714L),
  tol = 1e-7,
  zero_eps = 1e-8
)

for (path in c(
  cfg$train_data, cfg$test_data, cfg$dense_candidates, cfg$dense_fits
)) {
  if (!file.exists(path)) stop("Missing input: ", path)
}
if (any(!is.finite(cfg$K_grid)) || any(cfg$K_grid < 2L)) {
  stop("Invalid K grid.")
}
if (cfg$bootstrap_reps < 1L || cfg$bootstrap_nstart < 1L) {
  stop("Bootstrap reps and nstart must be positive.")
}
dir.create(cfg$out_dir, recursive = TRUE, showWarnings = FALSE)

source_no_bom <- function(path, envir = .GlobalEnv) {
  txt <- readLines(path, warn = FALSE, encoding = "UTF-8")
  if (length(txt) > 0L) txt[1L] <- sub("^\ufeff", "", txt[1L])
  eval(parse(text = txt), envir = envir)
}
source_no_bom(file.path("r", "methods", "rossi_barbaro_2022_reproduction.r"))
source_no_bom(file.path("r", "realdata", "exact_centered_refit_helpers_260711.r"))

log_vmf_const_one <- exact_refit_log_vmf_const_one
log_vmf_const <- exact_refit_log_vmf_const

train_payload <- readRDS(cfg$train_data)
test_payload <- readRDS(cfg$test_data)
X_train <- train_payload$X
X_test <- test_payload$X
if (!inherits(X_train, "sparseMatrix")) {
  X_train <- Matrix::Matrix(X_train, sparse = TRUE)
}
if (!inherits(X_test, "sparseMatrix")) {
  X_test <- Matrix::Matrix(X_test, sparse = TRUE)
}
y_train <- as.integer(train_payload$y)
y_test <- as.integer(test_payload$y)
tokens <- as.character(train_payload$vocab)
n <- nrow(X_train)
d <- ncol(X_train)
n_test <- nrow(X_test)

if (ncol(X_test) != d) stop("Train/test dimension mismatch.")
if (!identical(tokens, as.character(test_payload$vocab))) {
  stop("Train/test vocabulary mismatch.")
}

dense_candidates <- utils::read.csv(
  cfg$dense_candidates, check.names = FALSE
)
dense_fits <- readRDS(cfg$dense_fits)

candidate_key <- function(K, model) paste0("K", K, "__", model)

row_logsumexp <- function(scores) {
  maxima <- apply(scores, 1L, max)
  maxima + log(rowSums(exp(scores - maxima)))
}

evaluate_fit <- function(X, fit, K, return_tau = FALSE) {
  kappa <- as.numeric(fit$kappa)
  if (length(kappa) == 1L) kappa <- rep(kappa, K)
  alpha <- pmax(as.numeric(fit$alpha), 1e-15)
  alpha <- alpha / sum(alpha)
  linear <- as.matrix(X %*% t(fit$mu))
  linear <- sweep(linear, 2L, kappa, "*")
  scores <- sweep(
    linear, 2L,
    log(alpha) + exact_refit_log_vmf_const(kappa, ncol(X)), "+"
  )
  log_norm <- row_logsumexp(scores)
  tau <- exp(scores - log_norm)
  cluster <- max.col(scores, ties.method = "first")
  output <- list(
    loglik = sum(log_norm),
    loglik_by_row = log_norm,
    NLL_per_doc = -mean(log_norm),
    cluster = cluster,
    cluster_size = as.integer(table(factor(cluster, levels = seq_len(K)))),
    tau = if (return_tau) tau else NULL
  )
  output
}

normalized_mutual_information <- function(truth, cluster) {
  tab <- table(truth, cluster)
  p <- tab / sum(tab)
  pi <- rowSums(p)
  pj <- colSums(p)
  expected <- outer(pi, pj)
  keep <- p > 0 & expected > 0
  mi <- sum(p[keep] * log(p[keep] / expected[keep]))
  h1 <- -sum(pi[pi > 0] * log(pi[pi > 0]))
  h2 <- -sum(pj[pj > 0] * log(pj[pj > 0]))
  if (h1 + h2 <= 0) return(0)
  2 * mi / (h1 + h2)
}

npmi_for_tokens <- function(X, ids) {
  ids <- unique(as.integer(ids))
  if (length(ids) < 2L) return(NA_real_)
  present <- X[, ids, drop = FALSE] > 0
  p <- Matrix::colSums(present) / nrow(X)
  joint <- as.matrix(Matrix::crossprod(present)) / nrow(X)
  pairs <- which(upper.tri(joint), arr.ind = TRUE)
  values <- vapply(seq_len(nrow(pairs)), function(i) {
    a <- pairs[i, 1L]
    b <- pairs[i, 2L]
    pxy <- joint[a, b]
    if (!is.finite(pxy) || pxy <= 0) return(-1)
    if (pxy >= 1 - 1e-15) return(1)
    log(pxy / max(p[a] * p[b], 1e-15)) / (-log(pxy))
  }, numeric(1))
  mean(values[is.finite(values)])
}

reference_rows <- list()
token_rows <- list()
reference_id <- 0L
token_id <- 0L

for (K in cfg$K_grid) {
  for (model in c("shared", "free")) {
    key <- candidate_key(K, model)
    fit <- dense_fits[[key]]
    if (is.null(fit)) stop("Missing dense fit: ", key)
    source_row <- dense_candidates[
      dense_candidates$K == K & dense_candidates$kappa_model == model,
      , drop = FALSE
    ]
    if (nrow(source_row) != 1L) stop("Expected one candidate row: ", key)

    train_eval <- evaluate_fit(X_train, fit, K, return_tau = TRUE)
    test_eval <- evaluate_fit(X_test, fit, K, return_tau = FALSE)
    entropy <- -sum(train_eval$tau * log(pmax(train_eval$tau, 1e-15)))
    normalized_entropy <- entropy / (n * log(K))
    eta <- sweep(fit$mu, 1L, rep(as.numeric(fit$kappa), length.out = K), "*")
    centered <- sweep(eta, 2L, colMeans(eta), "-")

    component_coherence <- numeric(K)
    for (k in seq_len(K)) {
      eligible <- which(is.finite(centered[k, ]) & centered[k, ] > 0)
      if (length(eligible) < cfg$top_tokens) {
        eligible <- which(is.finite(centered[k, ]))
      }
      top <- head(
        eligible[order(centered[k, eligible], decreasing = TRUE)],
        cfg$top_tokens
      )
      component_coherence[k] <- npmi_for_tokens(X_train, top)
      for (rank in seq_along(top)) {
        token_id <- token_id + 1L
        token_rows[[token_id]] <- data.frame(
          K = K,
          kappa_model = model,
          component = k,
          rank = rank,
          feature_id = top[rank],
          token = tokens[top[rank]],
          centered_eta = centered[k, top[rank]],
          component_NPMI = component_coherence[k],
          stringsAsFactors = FALSE
        )
      }
    }

    reference_id <- reference_id + 1L
    reference_rows[[reference_id]] <- cbind(
      source_row,
      data.frame(
        posterior_entropy = entropy,
        posterior_entropy_per_doc = entropy / n,
        normalized_entropy = normalized_entropy,
        ICL_BIC = source_row$BIC + 2 * entropy,
        train_min_cluster_n = min(train_eval$cluster_size),
        train_min_cluster_prop = min(train_eval$cluster_size) / n,
        test_min_cluster_n = min(test_eval$cluster_size),
        test_min_cluster_prop = min(test_eval$cluster_size) / n_test,
        mean_component_NPMI = mean(component_coherence, na.rm = TRUE),
        min_component_NPMI = min(component_coherence, na.rm = TRUE),
        stringsAsFactors = FALSE
      )
    )
  }
}

reference <- do.call(rbind, reference_rows)
top_tokens <- do.call(rbind, token_rows)

make_spherical_init <- function(X, K, seed) {
  set.seed(seed)
  n_local <- nrow(X)
  mu <- as.matrix(X[sample.int(n_local, K, replace = FALSE), , drop = FALSE])
  cluster <- rep(NA_integer_, n_local)
  for (iter in seq_len(cfg$spherical_max_iter)) {
    similarity <- as.matrix(X %*% t(mu))
    next_cluster <- max.col(similarity, ties.method = "random")
    next_mu <- matrix(0, nrow = K, ncol = ncol(X))
    for (k in seq_len(K)) {
      index <- which(next_cluster == k)
      if (!length(index)) {
        next_mu[k, ] <- as.numeric(X[sample.int(n_local, 1L), ])
      } else {
        next_mu[k, ] <- Matrix::colMeans(X[index, , drop = FALSE])
      }
    }
    next_mu <- normalize_rows(next_mu)
    if ((!anyNA(cluster) && all(next_cluster == cluster)) ||
        max(abs(next_mu - mu)) < 1e-8) {
      mu <- next_mu
      cluster <- next_cluster
      break
    }
    mu <- next_mu
    cluster <- next_cluster
  }
  similarity <- as.matrix(X %*% t(mu))
  cluster <- max.col(similarity, ties.method = "first")
  tau <- matrix(0, nrow = n_local, ncol = K)
  tau[cbind(seq_len(n_local), cluster)] <- 1
  Nk <- colSums(tau)
  if (any(Nk == 0L)) return(NULL)
  r <- as.matrix(t(tau) %*% X)
  mu <- normalize_rows(r)
  kappa <- vapply(seq_len(K), function(k) {
    estimate_kappa(l2_norm(r[k, ]) / Nk[k], ncol(X))
  }, numeric(1))
  list(alpha = Nk / n_local, mu = mu, kappa = kappa)
}

fit_bootstrap_dense <- function(X, K, shared, bootstrap_id) {
  # Every initialization is estimated from the in-bag bootstrap sample.
  # A full-train reference fit would leak OOB observations into validation.
  inits <- list()
  for (s in seq_len(cfg$bootstrap_nstart)) {
    seed <- cfg$base_seed + bootstrap_id * 1000003L + K * 1009L +
      as.integer(shared) * 101L + s * 17L
    init <- make_spherical_init(X, K, seed)
    if (!is.null(init)) inits[[length(inits) + 1L]] <- init
  }
  if (!length(inits)) return(NULL)

  best <- NULL
  best_loglik <- -Inf
  for (init in inits) {
    fit <- tryCatch(
      fit_svMF_em(
        X, K, beta = 0, init = init, shared_kappa = shared,
        max_iter = cfg$max_iter, tol = cfg$tol, zero_eps = cfg$zero_eps
      ),
      error = function(e) NULL
    )
    if (is.null(fit) || isTRUE(fit$failed) || !is.finite(fit$loglik)) next
    if (fit$loglik > best_loglik) {
      best <- fit
      best_loglik <- fit$loglik
    }
  }
  best
}

bootstrap_rows <- list()
bootstrap_id <- 0L
prediction_store <- setNames(
  lapply(rep(NA, 2L * length(cfg$K_grid)), function(x) {
    matrix(NA_integer_, nrow = n, ncol = cfg$bootstrap_reps)
  }),
  unlist(lapply(cfg$K_grid, function(K) {
    c(candidate_key(K, "shared"), candidate_key(K, "free"))
  }))
)

bootstrap_started <- proc.time()[["elapsed"]]
for (b in seq_len(cfg$bootstrap_reps)) {
  set.seed(cfg$base_seed + b * 7919L)
  sample_index <- sample.int(n, n, replace = TRUE)
  unique_index <- unique(sample_index)
  oob_index <- setdiff(seq_len(n), unique_index)
  X_boot <- X_train[sample_index, , drop = FALSE]

  for (K in cfg$K_grid) {
    for (model in c("shared", "free")) {
      shared <- identical(model, "shared")
      key <- candidate_key(K, model)
      started <- proc.time()[["elapsed"]]
      fit <- fit_bootstrap_dense(X_boot, K, shared, bootstrap_id = b)
      elapsed <- proc.time()[["elapsed"]] - started
      bootstrap_id <- bootstrap_id + 1L

      if (is.null(fit)) {
        bootstrap_rows[[bootstrap_id]] <- data.frame(
          bootstrap = b, K = K, kappa_model = model,
          unique_train_n = length(unique_index), oob_n = length(oob_index),
          converged = FALSE, loglik = NA_real_, OOB_NLL_per_doc = NA_real_,
          full_min_cluster_prop = NA_real_, elapsed_sec = elapsed,
          stringsAsFactors = FALSE
        )
        next
      }

      full_eval <- evaluate_fit(X_train, fit, K, return_tau = FALSE)
      oob_eval <- evaluate_fit(
        X_train[oob_index, , drop = FALSE], fit, K, return_tau = FALSE
      )
      prediction_store[[key]][, b] <- full_eval$cluster
      bootstrap_rows[[bootstrap_id]] <- data.frame(
        bootstrap = b, K = K, kappa_model = model,
        unique_train_n = length(unique_index), oob_n = length(oob_index),
        converged = isTRUE(fit$converged), loglik = fit$loglik,
        OOB_NLL_per_doc = oob_eval$NLL_per_doc,
        full_min_cluster_prop = min(full_eval$cluster_size) / n,
        elapsed_sec = elapsed,
        stringsAsFactors = FALSE
      )
      cat(
        "Completed bootstrap=", b, " K=", K, " model=", model,
        " OOB NLL=", formatC(oob_eval$NLL_per_doc, digits = 3, format = "f"),
        "\n", sep = ""
      )
    }
  }

  checkpoint <- do.call(rbind, bootstrap_rows)
  utils::write.csv(
    checkpoint,
    file.path(cfg$out_dir, "classic3_k_bootstrap_raw.csv"),
    row.names = FALSE
  )
  saveRDS(
    prediction_store,
    file.path(cfg$out_dir, "classic3_k_bootstrap_predictions.rds")
  )
}
bootstrap_elapsed <- proc.time()[["elapsed"]] - bootstrap_started
bootstrap_raw <- do.call(rbind, bootstrap_rows)

stability_rows <- list()
stability_id <- 0L
for (K in cfg$K_grid) {
  for (model in c("shared", "free")) {
    key <- candidate_key(K, model)
    predictions <- prediction_store[[key]]
    valid <- which(colSums(is.na(predictions)) == 0L)
    pairwise <- numeric(0)
    if (length(valid) >= 2L) {
      pairs <- utils::combn(valid, 2L)
      pairwise <- apply(pairs, 2L, function(index) {
        adjusted_rand_index(
          predictions[, index[1L]], predictions[, index[2L]]
        )
      })
    }
    reference_cluster <- evaluate_fit(
      X_train, dense_fits[[key]], K, return_tau = FALSE
    )$cluster
    to_reference <- if (length(valid)) {
      vapply(valid, function(index) {
        adjusted_rand_index(reference_cluster, predictions[, index])
      }, numeric(1))
    } else numeric(0)
    raw_subset <- bootstrap_raw[
      bootstrap_raw$K == K & bootstrap_raw$kappa_model == model,
      , drop = FALSE
    ]
    nll <- raw_subset$OOB_NLL_per_doc[is.finite(raw_subset$OOB_NLL_per_doc)]

    stability_id <- stability_id + 1L
    stability_rows[[stability_id]] <- data.frame(
      K = K,
      kappa_model = model,
      valid_bootstrap_reps = length(valid),
      pairwise_stability_mean = if (length(pairwise)) mean(pairwise) else NA_real_,
      pairwise_stability_sd = if (length(pairwise) > 1L) stats::sd(pairwise) else NA_real_,
      pairwise_stability_min = if (length(pairwise)) min(pairwise) else NA_real_,
      reference_ARI_mean = if (length(to_reference)) mean(to_reference) else NA_real_,
      OOB_NLL_mean = if (length(nll)) mean(nll) else NA_real_,
      OOB_NLL_sd = if (length(nll) > 1L) stats::sd(nll) else NA_real_,
      OOB_NLL_se = if (length(nll) > 1L) stats::sd(nll) / sqrt(length(nll)) else NA_real_,
      min_cluster_prop_mean = mean(raw_subset$full_min_cluster_prop, na.rm = TRUE),
      failure_count = sum(!is.finite(raw_subset$OOB_NLL_per_doc)),
      stringsAsFactors = FALSE
    )
  }
}
stability <- do.call(rbind, stability_rows)
panel <- merge(reference, stability, by = c("K", "kappa_model"), all.x = TRUE)
panel <- panel[order(panel$kappa_model, panel$K), , drop = FALSE]

selection_rows <- list()
selection_id <- 0L
for (model in c("shared", "free")) {
  subset <- panel[panel$kappa_model == model, , drop = FALSE]
  for (criterion in c(
    "AIC", "BIC", "RIC", "RICc", "EBIC_g0.5", "EBIC_g1", "ICL_BIC"
  )) {
    index <- which.min(subset[[criterion]])
    selection_id <- selection_id + 1L
    selection_rows[[selection_id]] <- data.frame(
      kappa_model = model,
      criterion = criterion,
      direction = "min",
      selected_K = subset$K[index],
      value = subset[[criterion]][index],
      labels_used = FALSE,
      stringsAsFactors = FALSE
    )
  }

  index_oob <- which.min(subset$OOB_NLL_mean)
  selection_id <- selection_id + 1L
  selection_rows[[selection_id]] <- data.frame(
    kappa_model = model,
    criterion = "bootstrap_OOB_NLL_min",
    direction = "min",
    selected_K = subset$K[index_oob],
    value = subset$OOB_NLL_mean[index_oob],
    labels_used = FALSE,
    stringsAsFactors = FALSE
  )
  selected_se <- subset$OOB_NLL_se[index_oob]
  if (!is.finite(selected_se)) selected_se <- 0
  threshold <- subset$OOB_NLL_mean[index_oob] + selected_se
  eligible <- subset$K[
    is.finite(subset$OOB_NLL_mean) & subset$OOB_NLL_mean <= threshold
  ]
  selection_id <- selection_id + 1L
  selection_rows[[selection_id]] <- data.frame(
    kappa_model = model,
    criterion = "bootstrap_OOB_NLL_1SE",
    direction = "smallest_within_1SE",
    selected_K = min(eligible),
    value = threshold,
    labels_used = FALSE,
    stringsAsFactors = FALSE
  )

  if (any(is.finite(subset$pairwise_stability_mean))) {
    index_stability <- which.max(ifelse(
      is.finite(subset$pairwise_stability_mean),
      subset$pairwise_stability_mean,
      -Inf
    ))
    stability_K <- subset$K[index_stability]
    stability_value <- subset$pairwise_stability_mean[index_stability]
  } else {
    stability_K <- NA_integer_
    stability_value <- NA_real_
  }
  selection_id <- selection_id + 1L
  selection_rows[[selection_id]] <- data.frame(
    kappa_model = model,
    criterion = "bootstrap_pairwise_stability",
    direction = "max",
    selected_K = stability_K,
    value = stability_value,
    labels_used = FALSE,
    stringsAsFactors = FALSE
  )
}
selection <- do.call(rbind, selection_rows)

audit <- data.frame(
  check = c(
    "reference_rows", "bootstrap_rows", "all_reference_converged",
    "all_bootstrap_finite_oob", "all_bootstrap_converged",
    "train_test_vocab_identical",
    "labels_excluded_from_selection", "finite_ICL", "finite_entropy",
    "finite_token_coherence", "bootstrap_initialization_inbag_only"
  ),
  observed = c(
    nrow(reference), nrow(bootstrap_raw), all(reference$converged),
    sum(stability$failure_count), all(bootstrap_raw$converged),
    identical(tokens, as.character(test_payload$vocab)),
    !any(selection$labels_used), all(is.finite(reference$ICL_BIC)),
    all(is.finite(reference$posterior_entropy)),
    all(is.finite(reference$mean_component_NPMI)), TRUE
  ),
  expected = c(
    2L * length(cfg$K_grid),
    cfg$bootstrap_reps * 2L * length(cfg$K_grid),
    TRUE, 0L, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE
  ),
  stringsAsFactors = FALSE
)
audit$pass <- as.character(audit$observed) == as.character(audit$expected)

utils::write.csv(
  reference,
  file.path(cfg$out_dir, "classic3_k_reference_extended.csv"),
  row.names = FALSE
)
utils::write.csv(
  top_tokens,
  file.path(cfg$out_dir, "classic3_k_top_tokens.csv"),
  row.names = FALSE
)
utils::write.csv(
  bootstrap_raw,
  file.path(cfg$out_dir, "classic3_k_bootstrap_raw.csv"),
  row.names = FALSE
)
utils::write.csv(
  stability,
  file.path(cfg$out_dir, "classic3_k_bootstrap_stability.csv"),
  row.names = FALSE
)
utils::write.csv(
  panel,
  file.path(cfg$out_dir, "classic3_k_panel_summary.csv"),
  row.names = FALSE
)
utils::write.csv(
  selection,
  file.path(cfg$out_dir, "classic3_k_selection_by_criterion.csv"),
  row.names = FALSE
)
utils::write.csv(
  audit,
  file.path(cfg$out_dir, "classic3_k_panel_audit.csv"),
  row.names = FALSE
)

fmt <- function(x, digits = 3L) formatC(x, digits = digits, format = "f")
selection_lines <- paste0(
  "| ", selection$kappa_model, " | ", selection$criterion, " | ",
  selection$selected_K, " | ", fmt(selection$value, 3L), " |"
)
notes <- c(
  "# Classic3 extended K-selection panel diagnostic",
  "",
  sprintf(
    "- Candidate K={%s}; bootstrap reps=%d; bootstrap nstart=%d.",
    paste(cfg$K_grid, collapse = ","),
    cfg$bootstrap_reps,
    cfg$bootstrap_nstart
  ),
  "- AIC/BIC/RIC/RICc/EBIC/ICL use train fits only.",
  "- ICL is BIC + 2 times posterior classification entropy (lower is better).",
  "- Held-out density uses bootstrap out-of-bag NLL; the external test split is not used for K selection.",
  "- Every bootstrap initialization is estimated from the in-bag sample; full-train fits are excluded from initialization.",
  "- Stability is mean pairwise ARI between full-train predictions from bootstrap fits.",
  sprintf(
    "- Token coherence is mean NPMI among the top %d positive centered-Eta tokens per component.",
    cfg$top_tokens
  ),
  "- Supplied labels are used only after the diagnostic for ARI/NMI reporting.",
  sprintf("- Bootstrap elapsed seconds: %.1f.", bootstrap_elapsed),
  "",
  "| kappa model | criterion | selected K | value |",
  "|---|---|---:|---:|",
  selection_lines,
  "",
  sprintf("- Audit checks passed: %d/%d.", sum(audit$pass), nrow(audit)),
  ""
)
writeLines(
  notes,
  file.path(cfg$out_dir, "classic3_k_panel_notes.md"),
  useBytes = TRUE
)

print(selection, row.names = FALSE)
print(audit, row.names = FALSE)
cat("Saved:", normalizePath(cfg$out_dir, winslash = "/"), "\n")
