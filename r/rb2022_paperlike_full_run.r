# ==============================================================================
# Full paper-scale reproduction for Rossi & Barbaro (2022)
# ------------------------------------------------------------------------------
# Artificial simulation settings:
#   - 100 data sets per condition
#   - 10 random starts
#   - d = 100, true K = 4
#   - N in {200, 1000}
#   - overlap in {2.5%, 5%}
#   - true nonzero fractions in {5%, 10%, 15%}
#   - candidate K in {1, ..., 6}
#
# The shared driver writes one CSV per simulation cell and skips completed cells,
# so this script can be rerun after interruption.
# ==============================================================================

set_default_env <- function(name, value) {
  if (identical(Sys.getenv(name), "")) {
    args <- list(value)
    names(args) <- name
    do.call(Sys.setenv, args)
  }
}

set_default_env("RB2022_RUN_LABEL", "rb2022_paperlike_full")
set_default_env("RB2022_N_REP", "100")
set_default_env("RB2022_NSTART", "10")
set_default_env("RB2022_MAX_PATH_STEPS", "700")
set_default_env("RB2022_WORKERS", "8")
set_default_env("RB2022_OUT_DIR", "results/rb2022_paperlike_full_260602")

source(file.path("r", "rb2022_paperlike_n20_run.r"))
