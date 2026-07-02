// Standalone Rcpp prototypes for Eta-group low-level numeric helpers.
// These functions are not wired into the official R fitting pipeline.

#include <Rcpp.h>
using namespace Rcpp;

namespace {

NumericVector make_weights(int d, Nullable<NumericVector> adaptive_weights) {
  if (adaptive_weights.isNull()) {
    return NumericVector(d, 1.0);
  }
  NumericVector weights(adaptive_weights);
  if (weights.size() != d) {
    stop("adaptive_weights must have length ncol(eta).");
  }
  return weights;
}

void copy_dimnames(SEXP from, SEXP to) {
  SEXP dimnames = Rf_getAttrib(from, R_DimNamesSymbol);
  if (!Rf_isNull(dimnames)) {
    Rf_setAttrib(to, R_DimNamesSymbol, dimnames);
  }
}

} // namespace

// [[Rcpp::export]]
double eta_centered_penalty_value_cpp(
    NumericMatrix eta,
    Nullable<NumericVector> adaptive_weights = R_NilValue) {
  const int K = eta.nrow();
  const int d = eta.ncol();
  NumericVector weights = make_weights(d, adaptive_weights);

  double total = 0.0;
  for (int j = 0; j < d; ++j) {
    double mean_j = 0.0;
    for (int k = 0; k < K; ++k) {
      mean_j += eta(k, j);
    }
    mean_j /= static_cast<double>(K);

    double norm_sq = 0.0;
    for (int k = 0; k < K; ++k) {
      const double centered = eta(k, j) - mean_j;
      norm_sq += centered * centered;
    }
    total += weights[j] * std::sqrt(norm_sq);
  }
  return total;
}

// [[Rcpp::export]]
NumericMatrix prox_eta_centered_cpp(
    NumericMatrix eta,
    double lambda_eta,
    Nullable<NumericVector> adaptive_weights = R_NilValue) {
  const int K = eta.nrow();
  const int d = eta.ncol();
  NumericVector weights = make_weights(d, adaptive_weights);
  NumericMatrix out(K, d);

  for (int j = 0; j < d; ++j) {
    double mean_j = 0.0;
    for (int k = 0; k < K; ++k) {
      mean_j += eta(k, j);
    }
    mean_j /= static_cast<double>(K);

    double norm_sq = 0.0;
    for (int k = 0; k < K; ++k) {
      const double centered = eta(k, j) - mean_j;
      norm_sq += centered * centered;
    }
    const double norm_j = std::sqrt(norm_sq);
    const double threshold = lambda_eta * weights[j];
    const double scale = (norm_j > 0.0)
      ? std::max(1.0 - threshold / norm_j, 0.0)
      : 0.0;

    for (int k = 0; k < K; ++k) {
      out(k, j) = mean_j + (eta(k, j) - mean_j) * scale;
    }
  }

  copy_dimnames(eta, out);
  return out;
}

// [[Rcpp::export]]
NumericMatrix normalize_rows_cpp(NumericMatrix X, double eps = 1e-12) {
  const int n = X.nrow();
  const int d = X.ncol();
  NumericMatrix out(n, d);

  for (int i = 0; i < n; ++i) {
    double norm_sq = 0.0;
    for (int j = 0; j < d; ++j) {
      norm_sq += X(i, j) * X(i, j);
    }
    double norm_i = std::sqrt(norm_sq);
    if (norm_i < eps) {
      norm_i = 1.0;
    }
    for (int j = 0; j < d; ++j) {
      out(i, j) = X(i, j) / norm_i;
    }
  }

  copy_dimnames(X, out);
  return out;
}
