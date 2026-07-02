// Standalone Rcpp prototypes for vMF E-step numeric helpers.
// These functions are not wired into the official fitting pipeline.

#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericVector row_logsumexp_cpp(NumericMatrix M) {
  const int n = M.nrow();
  const int K = M.ncol();
  NumericVector out(n);

  for (int i = 0; i < n; ++i) {
    double row_max = R_NegInf;
    for (int k = 0; k < K; ++k) {
      if (M(i, k) > row_max) row_max = M(i, k);
    }

    double sum_exp = 0.0;
    for (int k = 0; k < K; ++k) {
      sum_exp += std::exp(M(i, k) - row_max);
    }
    out[i] = row_max + std::log(sum_exp);
  }

  return out;
}

// [[Rcpp::export]]
List e_step_vmf_cpp(
    NumericMatrix X,
    NumericVector alpha,
    NumericMatrix mu,
    NumericVector kappa,
    NumericVector log_const) {
  const int n = X.nrow();
  const int d = X.ncol();
  const int K = mu.nrow();

  if (mu.ncol() != d) stop("ncol(mu) must match ncol(X).");
  if (alpha.size() != K) stop("alpha length must equal nrow(mu).");
  if (kappa.size() != K) stop("kappa length must equal nrow(mu).");
  if (log_const.size() != K) stop("log_const length must equal nrow(mu).");

  NumericMatrix logdens(n, K);
  NumericMatrix tau(n, K);
  NumericVector lse(n);
  double loglik = 0.0;

  for (int i = 0; i < n; ++i) {
    double row_max = R_NegInf;
    for (int k = 0; k < K; ++k) {
      double dot = 0.0;
      for (int j = 0; j < d; ++j) {
        dot += X(i, j) * mu(k, j);
      }
      const double alpha_k = std::max(static_cast<double>(alpha[k]), 1e-300);
      const double val = dot * kappa[k] + log_const[k] + std::log(alpha_k);
      logdens(i, k) = val;
      if (val > row_max) row_max = val;
    }

    double sum_exp = 0.0;
    for (int k = 0; k < K; ++k) {
      sum_exp += std::exp(logdens(i, k) - row_max);
    }
    lse[i] = row_max + std::log(sum_exp);
    loglik += lse[i];

    for (int k = 0; k < K; ++k) {
      tau(i, k) = std::exp(logdens(i, k) - lse[i]);
    }
  }

  return List::create(
    _["tau"] = tau,
    _["loglik"] = loglik,
    _["logdens"] = logdens,
    _["lse"] = lse
  );
}
