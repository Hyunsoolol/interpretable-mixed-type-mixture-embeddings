# Rossi 모형과 Eta 제안 모형 R 코드 구현 설명

업데이트: 2026-06-15

이 문서는 현재 thesis-hp-clustering에서 사용한 R 코드 기준으로, Rossi & Barbaro (2022) sparse vMF 모형과 제안하는 Eta-group 모형이 어떻게 구현되어 있는지 정리한 것이다.

목적은 연구미팅에서 다음 질문에 답할 수 있도록 하는 것이다.

- Rossi는 코드에서 어떻게 추정했는가?
- penalty가 있을 때 EM update는 어떻게 구현했는가?
- 제안 Eta-group은 Rossi와 무엇이 다른가?
- lambda path와 BIC tuning은 코드에서 어떻게 처리했는가?
- refit은 정확히 무엇을 다시 추정하는가?

## 1. 관련 R 파일

| 파일 | 역할 |
|---|---|
| `r/rossi_barbaro_2022_reproduction.r` | vMF density, vMF sampling, Rossi sparse vMF EM, beta path, BIC/EBIC 계산 |
| `r/rb2022_k4_pilot_compare_run.r` | 공통 평가 함수, refit, Eta-group EM prototype, K=4 비교용 기본 함수 |
| `r/k4_path_tuning_compare_run.r` | K=4에서 Rossi / Separate / Eta path tuning 비교 |
| `r/k4_specific_effect_run.r` | common variable + component-specific variable 시뮬레이션 실행 코드 |

가장 핵심적인 모형 추정 함수는 다음이다.

| 기능 | 함수 |
|---|---|
| vMF E-step | `e_step_vmf()` |
| kappa 근사 추정 | `estimate_kappa()` |
| Rossi coordinate update | `update_mu_kappa_one()` |
| Rossi EM | `fit_svMF_em()` |
| Rossi multi-start | `fit_svMF_multistart()` |
| Rossi beta path | `fit_svMF_path()` |
| support refit | `fit_support_refit()` |
| Eta centered proximal step | `prox_eta_centered()` |
| Eta centered EM | `fit_eta_centered_em()` |
| Eta path tuning | `fit_eta_centered_path_pair()`, `fit_eta_specific_pair()` |

## 2. 공통 vMF mixture 구조

관측값 `X`는 row-wise unit norm을 갖는 방향자료로 둔다.

```text
||x_i||_2 = 1
```

K개 component를 갖는 vMF mixture는 다음 구조다.

```text
f(x_i) = sum_k alpha_k C_d(kappa_k) exp(kappa_k mu_k^T x_i)
```

여기서

| 기호 | 코드 객체 | 의미 |
|---|---|---|
| alpha_k | `theta$alpha[k]` | mixing proportion |
| mu_k | `theta$mu[k, ]` | component direction |
| kappa_k | `theta$kappa[k]` | concentration |
| tau_ik | `e$tau[i, k]` | posterior responsibility |

코드에서는 parameter를 다음 list로 저장한다.

```r
theta <- list(
  alpha = alpha,
  mu = mu,
  kappa = kappa
)
```

## 3. E-step 구현

E-step은 `e_step_vmf()`에서 구현한다.

코드 흐름은 다음과 같다.

```r
logdens <- X %*% t(theta$mu)
logdens <- sweep(logdens, 2, kappa, "*")
logdens <- sweep(logdens, 2, log_vmf_const(kappa, d), "+")
logdens <- sweep(logdens, 2, log(pmax(theta$alpha, 1e-300)), "+")
```

즉 각 i, k에 대해 다음 값을 log scale로 계산한다.

```text
log alpha_k + log C_d(kappa_k) + kappa_k mu_k^T x_i
```

그 다음 `row_logsumexp()`를 이용해 posterior를 계산한다.

```text
tau_ik = exp(logdens_ik - logsumexp_k(logdens_ik))
```

log scale로 계산하는 이유는 고차원에서 `exp(kappa * mu^T x)`가 매우 커지거나 작아질 수 있기 때문이다.

## 4. kappa 추정 구현

vMF mixture의 M-step에서 concentration parameter는 정확히는 다음 방정식을 풀어야 한다.

```text
A_d(kappa_k) = rho_k
```

하지만 inverse를 매번 수치적으로 푸는 대신, 현재 코드는 널리 쓰이는 근사식을 사용한다.

```text
kappa ≈ (d * rho - rho^3) / (1 - rho^2)
```

구현 위치는 `estimate_kappa()`다.

```r
estimate_kappa <- function(rho, d, kappa_cap = 1e6) {
  rho <- min(max(rho, 1e-10), 1 - 1e-8)
  kappa <- (d * rho - rho^3) / (1 - rho^2)
  min(max(kappa, 1e-10), kappa_cap)
}
```

주의할 점은 `rho`가 1에 가까워지면 kappa가 발산할 수 있다는 것이다. 그래서 코드에서는 `rho`를 `1 - 1e-8`보다 작게 제한하고, `kappa_cap`도 둔다.

이 근사식은 vMF mixture EM에서 자주 쓰이는 Banerjee et al. (2005) 계열의 update approximation이다.

> Reference note. 이 $\kappa$ update는 Banerjee et al. (2005)의 vMF mixture EM 문헌에서 사용하는 concentration approximation과 같은 계열이다. vMF M-step은 mean resultant length $\bar{\rho}$에 대해 $A_d(\kappa)=\bar{\rho}$를 풀어야 하지만, $A_d$는 modified Bessel function ratio라 inverse를 매번 닫힌형으로 계산하기 어렵다. 따라서 현재 구현의
> $$
> \hat{\kappa}\approx\frac{d\bar{\rho}-\bar{\rho}^3}{1-\bar{\rho}^2}
> $$
> 는 closed-form solution이 아니라 계산 효율을 위한 approximation으로 해석한다.

## 5. Rossi 구현

### 5.1 모형

Rossi & Barbaro (2022) 방법은 component direction `mu_k`에 L1 penalty를 둔다.

```text
penalized loglik = loglik - beta * sum_k ||mu_k||_1
```

즉 penalty는 concentration `kappa_k`가 아니라 direction vector `mu_k`에 직접 들어간다.

코드에서는 `fit_svMF_em()`의 argument `beta`가 이 penalty parameter다.

> Reference note. Rossi and Barbaro (2022)는 본 연구의 sparse vMF mixture baseline이다. 해당 논문은 vMF mixture를 $L_1$-penalized likelihood로 추정해 sparse prototype/direction을 얻는 접근을 제안한다. 코드 비교에서 Rossi 방식은 direction parameter $\mu_k$에 sparsity penalty를 주는 방법으로 구현했고, 본 연구의 Eta-group은 posterior decision에 직접 들어가는 natural parameter $\eta_k=\kappa_k\mu_k$의 component contrast를 penalize한다는 점에서 구분된다.

```r
fit_svMF_em(X, K, beta = beta, ...)
```

### 5.2 Rossi M-step의 핵심

E-step 이후 sufficient statistic은 다음이다.

```text
N_k = sum_i tau_ik
r_k = sum_i tau_ik x_i
```

Rossi update는 component별로 `update_mu_kappa_one()`에서 수행한다.

핵심 코드는 다음이다.

```r
shrink <- pmax(kappa * abs(r_k) - beta, 0)
mu_new <- sign(r_k) * shrink / l2_norm(shrink)
rho <- crossprod(mu_new, r_k) / Nk
kappa_new <- estimate_kappa(rho, d, kappa_cap)
```

수식으로 쓰면 다음과 같다.

```text
z_kj = sign(r_kj) * (kappa_k |r_kj| - beta)_+
mu_k = z_k / ||z_k||_2
rho_k = mu_k^T r_k / N_k
kappa_k = estimate_kappa(rho_k, d)
```

여기서 중요한 점은 `mu_k`와 `kappa_k`가 서로 의존한다는 것이다.

- `mu_k` update에는 현재 `kappa_k`가 들어간다.
- `kappa_k` update에는 새로 얻은 `mu_k`가 들어간다.

따라서 코드에서는 `inner_max_iter` 동안 component 내부 update를 반복한다.

### 5.3 shared kappa 옵션

`fit_svMF_em()`에는 `shared_kappa` 옵션이 있다.

```r
fit_svMF_em(..., shared_kappa = FALSE)
```

현재 시뮬레이션의 주 비교에서는 component별 `kappa_k`를 허용하는 `shared_kappa = FALSE`를 주로 사용한다.

만약 `shared_kappa = TRUE`이면 모든 component가 같은 kappa를 공유한다. 이 경우 code는 component별 `mu_k`를 업데이트한 뒤, 전체 rho를 이용해 하나의 shared kappa를 다시 추정한다.

### 5.4 Rossi EM loop

`fit_svMF_em()`의 전체 반복 구조는 다음과 같다.

```text
1. 초기 theta 설정
2. E-step: tau 계산
3. M-step:
   - alpha 업데이트
   - 각 component의 mu, kappa 업데이트
4. penalized objective 계산
5. objective 변화가 tol보다 작으면 수렴
```

objective는 다음처럼 계산된다.

```r
obj <- e_new$loglik - beta * sum(abs(theta_new$mu))
```

수렴 기준은 상대 변화량이다.

```text
abs(obj_new - obj_old) / max(1, abs(obj_old)) < tol
```

## 6. Rossi beta path tuning

### 6.1 왜 path tuning을 쓰는가

고정 grid를 임의로 잡으면 beta 후보가 너무 작거나 너무 클 수 있다. Rossi 논문 흐름에 맞추기 위해 현재 코드는 data-adaptive path를 사용한다.

구현 위치는 `fit_svMF_path()`다.

### 6.2 path 생성 방식

첫 단계는 beta = 0, 즉 dense vMF mixture다.

```r
beta <- 0
fit <- fit_svMF_multistart(X, K, beta = 0, ...)
```

그 다음 현재 fit에서 각 coordinate가 다음에 0이 될 수 있는 threshold를 계산한다.

```r
margin[k, ] <- kappa[k] * abs(r[k, ]) - beta
candidates <- margin[margin > beta_eps]
beta_next <- beta + min(candidates)
```

의미는 다음과 같다.

```text
현재 beta보다 큰 다음 threshold = min positive kappa_k |r_kj|
```

새 beta에서는 이전 fit을 warm start로 사용한다.

```r
fit_next <- fit_svMF_em(X, K, beta = beta_next, init = fit, ...)
```

이 과정을 `max_path_steps`까지 반복한다.

### 6.3 BIC 선택

path 위의 각 후보 fit에 대해 BIC를 계산하고, 가장 작은 BIC를 선택한다.

```r
idx <- which.min(path$path$BIC)
fit <- path$fits[[idx]]
```

BIC 계산은 `model_ic()` 또는 각 driver의 IC 함수에서 수행한다.

기본 구조는 다음이다.

```text
BIC = log(n) * df - 2 * loglik
```

Rossi의 자유도는 대략 다음 구조로 계산된다.

```text
df = alpha/kappa 자유도 + active mu 자유도
```

`mu_k`는 unit norm 제약이 있으므로 active coordinate 수가 m이면 방향 자유도는 대략 `m - 1`이다.

## 7. 제안 Eta-group 구현

### 7.1 왜 eta를 쓰는가

vMF component는 다음 자연모수로 쓸 수 있다.

```text
eta_k = kappa_k * mu_k
```

posterior decision에는 `mu_k`와 `kappa_k`가 따로 들어가는 것이 아니라, 실제로는 `eta_k^T x_i`가 직접 들어간다.

```text
log posterior score for component k
= log alpha_k + log C_d(kappa_k) + eta_k^T x_i
```

따라서 군집 구분에 중요한 변수는 `mu_k`의 sparsity보다 `eta_k`의 component 간 차이에서 더 직접적으로 나타난다.

이 점이 Rossi 방식과 제안 방식의 핵심 차이다.

## 8. K=4 centered eta penalty

현재 K=4 시뮬레이션에서 사용하는 제안 모형은 centered eta penalty다.

각 coordinate j에 대해 component 평균 eta를 계산한다.

```text
eta_bar_j = mean_k eta_kj
```

그리고 centered eta를 만든다.

```text
c_kj = eta_kj - eta_bar_j
```

coordinate j가 군집 구분에 기여하는 정도는 다음 group norm으로 측정한다.

```text
contrast_j = sqrt(sum_k c_kj^2)
```

penalty는 coordinate별 contrast norm에 들어간다.

```text
penalized loglik = loglik - lambda_eta * sum_j contrast_j
```

이 penalty는 모든 component에 공통으로 나타나는 eta 성분은 보존하고, component 간 차이를 만드는 coordinate만 선택하도록 유도한다.

> Reference note. Eta-group의 coordinate-wise group penalty는 Yuan and Lin (2006)의 group lasso 아이디어를 centered eta contrast에 맞춘 것이다. 원래 group lasso는 미리 정의된 변수 group을 함께 선택하거나 제거하기 위한 penalty이고, 여기서는 coordinate $j$마다 $K$개 component의 centered eta vector $c_{\cdot j}$를 하나의 group으로 본다.

## 9. Eta proximal M-step

Eta-group의 exact M-step은 닫힌형으로 바로 풀기 어렵다. 이유는 vMF normalizing constant가 `||eta_k||_2`에 의존하고, penalty는 component 간 centered eta에 걸려 있기 때문이다.

현재 구현은 proximal EM-type update다. 즉 exact closed-form penalized M-step을 푸는 것이 아니라, unpenalized eta M-step 후보에 centered eta group soft-thresholding을 적용한다.

구현 위치는 다음 함수들이다.

| 함수 | 역할 |
|---|---|
| `unpenalized_eta_mstep()` | penalty 없는 vMF M-step으로 eta 후보 계산 |
| `prox_eta_centered()` | centered eta에 group soft-thresholding 적용 |
| `eta_to_theta()` | eta에서 mu, kappa 복원 |
| `fit_eta_centered_em()` | Eta-group proximal EM-type 반복 |

### 9.1 unpenalized eta M-step

먼저 일반 vMF M-step처럼 `mu_k`, `kappa_k`를 계산한 뒤 eta로 변환한다.

```r
mu <- normalize_rows(r)
kappa[k] <- estimate_kappa(rho, ncol(X), kappa_cap)
eta <- sweep(mu, 1, kappa, "*")
```

즉

```text
eta_k = kappa_k * mu_k
```

### 9.2 centered eta proximal update

`prox_eta_centered()`의 핵심 코드는 다음이다.

```r
mean_eta <- colMeans(eta)
centered <- sweep(eta, 2, mean_eta, "-")
norms <- sqrt(colSums(centered * centered))
scale <- ifelse(norms > 0, pmax(1 - lambda_eta / norms, 0), 0)
eta_new <- sweep(sweep(centered, 2, scale, "*"), 2, mean_eta, "+")
```

즉 coordinate별로 다음 group soft-thresholding을 한다.

```text
if ||c_.j||_2 > lambda_eta:
    c_.j_new = (1 - lambda_eta / ||c_.j||_2) * c_.j
else:
    c_.j_new = 0

eta_kj_new = eta_bar_j + c_kj_new
```

수식으로 쓰면 다음 block soft-thresholding이다.

$$
c_{\cdot j}^{new}
=
\left(
1-\frac{\lambda_\eta}{\|c_{\cdot j}^{0}\|_2}
\right)_+
c_{\cdot j}^{0}.
$$

> Reference note. 이 식은 group lasso penalty의 proximal operator, 즉 block/group soft-thresholding이다. Yuan and Lin (2006)의 group lasso penalty와 Parikh and Boyd (2014)의 proximal algorithms 문헌에서 표준적으로 쓰이는 shrinkage 형태와 같은 계열이다. 본 연구에서는 coordinate $j$별 centered eta vector $c_{\cdot j}$에 적용한다. 이는 전체 mixture objective의 convexity 보장이 아니라, proximal EM-type update 내부의 shrinkage step이다.

여기서 중요한 해석은 다음이다.

- `eta_bar_j`는 component 공통 성분이다.
- `c_kj`는 component 간 차이 성분이다.
- penalty는 공통 성분 자체를 제거하는 것이 아니라, 군집 구분 contrast를 shrink한다.

따라서 common variable과 component-specific variable이 섞인 환경에서 Rossi보다 해석이 자연스럽다.

### 9.3 eta에서 mu, kappa 복원

proximal update 후에는 다시 vMF parameter로 바꾼다.

```text
kappa_k = ||eta_k||_2
mu_k = eta_k / ||eta_k||_2
```

구현 함수는 `eta_to_theta()`다.

만약 `eta_k` norm이 너무 작아지는 경우에는 이전 `mu_k`를 fallback으로 사용해 수치적 실패를 줄인다.

## 10. Eta proximal EM-type loop

`fit_eta_centered_em()`의 반복 구조는 다음과 같다. 함수명은 기존 코드와의 호환을 위해 `em`으로 남아 있지만, 논문 표현에서는 proximal EM-type update로 부르는 것이 더 정확하다.

```text
1. dense vMF 또는 이전 fit으로 초기화
2. E-step: tau 계산
3. unpenalized eta M-step 계산
4. centered eta proximal update
5. eta -> mu, kappa 복원
6. loglik와 penalized objective 계산
7. objective 변화가 작으면 수렴
```

objective는 다음이다.

```r
penalty <- sum(sqrt(colSums(center_eta(eta_matrix(theta_new))^2)))
obj <- e_new$loglik - lambda_eta * penalty
```

즉 코드에서 사용하는 penalty는 다음이다.

```text
lambda_eta * sum_j sqrt(sum_k (eta_kj - eta_bar_j)^2)
```

## 11. Eta lambda path tuning

K=4 시뮬레이션에서는 fixed lambda grid가 아니라 path tuning을 사용한다.

구현 위치는 `fit_eta_specific_pair()` 또는 `fit_eta_centered_path_pair()`다.

### 11.1 path 시작

먼저 dense vMF fit을 만든다.

```r
dense <- fit_svMF_multistart(X, K, beta = 0, ...)
lambda_eta <- 0
fit <- fit_eta_centered_em(X, K, lambda_eta = 0, init = dense, ...)
```

### 11.2 다음 lambda 후보

현재 fit에서 E-step을 한 뒤, unpenalized eta M-step을 계산한다.

```r
e <- e_step_vmf(X, fit)
mstep <- unpenalized_eta_mstep(X, e$tau)
thresholds <- sqrt(colSums(center_eta(mstep$eta)^2))
```

`thresholds[j]`는 coordinate j의 centered eta contrast 크기다.

다음 lambda는 현재 lambda보다 큰 threshold 중 가장 작은 값으로 둔다.

```r
candidates <- thresholds[thresholds > lambda_eta + 1e-10]
lambda_next <- min(candidates)
```

이것은 Rossi beta path와 같은 철학이다.

- Rossi는 `kappa_k |r_kj|` threshold를 따라 beta를 증가시킨다.
- Eta는 `||centered eta_.j||_2` threshold를 따라 lambda_eta를 증가시킨다.

### 11.3 BIC 선택

각 lambda 후보마다 fit을 저장하고, BIC가 가장 작은 후보를 선택한다.

```r
best <- best_ic_index(tab, cfg)
fit <- fits[[best]]
```

현재 공식 비교에서는 `path tuning + BIC`를 기본 기준으로 둔다.

## 12. Refit 구현

Refit은 선택된 변수 support를 고정한 뒤 penalty 없이 vMF mixture를 다시 추정하는 단계다.

구현 함수는 `fit_support_refit()`이다.

### 12.1 Rossi refit

Rossi에서는 active coordinate를 다음처럼 정의한다.

```r
active <- active_mu_coord(fit)
```

즉 어느 component에서든 `mu_kj`가 0이 아니면 coordinate j를 active로 본다.

그 support만 남기고, penalty 없이 다시 EM을 수행한다.

### 12.2 Eta refit

Eta에서는 active coordinate를 다음처럼 정의한다.

```r
active <- active_eta_centered(fit)
```

즉 coordinate j의 centered eta contrast가 0이 아니면 active로 본다.

이 active support를 고정한 뒤 `fit_support_refit()`으로 penalty 없이 다시 vMF mixture를 추정한다.

해석은 다음과 같다.

- penalty 단계: 변수 선택
- refit 단계: 선택된 변수만 사용해 bias를 줄인 모수 재추정

따라서 Eta-group + refit 결과에서 clustering 성능과 kappa 추정이 개선되는 경우가 많다.

## 13. Evaluation code

각 방법의 결과는 `eval_method()`에서 같은 형식으로 정리한다.

주요 출력은 다음이다.

| 지표 | 의미 |
|---|---|
| `ARI` | true label과 estimated cluster 비교 |
| `selected_q` | 선택된 coordinate 수 |
| `TPR` | true active variable 중 선택 비율 |
| `FPR` | noise variable 중 잘못 선택한 비율 |
| `Precision` | 선택된 변수 중 true active 비율 |
| `F1` | 변수 선택 precision과 recall의 조화평균 |
| `MSE_mu` | true mu와 estimated mu 차이 |
| `MSE_kappa` | true kappa와 estimated kappa 차이 |
| `MSE_centered_eta` | centered eta 차이 |

시뮬레이션에서는 label switching 문제가 있으므로, `parameter_metrics()`에서 true mu와 estimated mu의 cosine similarity가 최대가 되도록 component permutation을 맞춘 뒤 MSE를 계산한다.

## 14. 시뮬레이션 driver 코드

최근 사용한 핵심 시뮬레이션은 `r/k4_specific_effect_run.r`이다.

이 파일은 다음 환경을 만든다.

```text
common variables:
v_kj = 1.0 for all components

component-specific variables:
v_kj = w only for component k

noise variables:
v_kj = 0

mu_k = v_k / ||v_k||
```

관련 함수는 `make_specific_effect_params()`다.

그 다음 `simulate_from_params()`에서 각 관측치의 component label을 뽑고, `rvMF()`로 방향자료를 생성한다.

비교는 한 replication 안에서 같은 simulated data에 대해 다음 순서로 수행한다.

```text
1. fit_rossi_specific_pair()
2. fit_separate_specific_pair()
3. fit_eta_specific_pair()
```

이 중 thesis의 핵심 비교는 Rossi와 Eta-group이다. Separate는 교수님 제안 baseline으로 함께 둔 것이다.

## 15. 코드상 중요한 해석 포인트

### 15.1 Rossi penalty의 대상

Rossi는 `mu_k`의 sparsity를 만든다.

```text
penalty target: mu_k
```

하지만 posterior decision에는 `kappa_k * mu_k`가 들어간다. 따라서 concentration 차이가 큰 상황에서는 `mu_k` sparsity만으로 군집 구분 변수의 중요도를 직접 설명하기 어렵다.

### 15.2 Eta-group의 대상

Eta-group은 posterior decision에 직접 들어가는 자연모수 contrast를 shrink한다.

```text
penalty target: centered eta contrast
eta_k = kappa_k * mu_k
```

그래서 common variable이 많고 component-specific variable이 일부만 존재하는 환경에서, Eta-group은 공통 성분보다 component 간 차이를 만드는 변수에 더 직접적으로 반응한다.

### 15.3 Refit의 의미

Refit은 새로운 변수 선택 방법이 아니다. 이미 선택된 support를 고정한 후 penalty 없이 다시 추정하는 bias correction 단계다.

```text
penalized fit: support 선택
refit: 선택된 support에서 alpha, mu, kappa 재추정
```

따라서 결과표에서는 penalty fit과 penalty + refit을 함께 보는 것이 좋다.

## 16. 현재 코드의 한계와 검증 포인트

현재 Eta 구현은 exact closed-form penalized M-step이 아니라 proximal EM-type update다. 따라서 논문 작성 시 표현은 다음처럼 조심하는 것이 좋다.

```text
We implement a proximal EM-type update for the centered eta-contrast penalty.
```

Objective trace smoke test는 `results/eta_objective_trace_260615/`에 저장했다. 해당 smoke test에서는 일부 lambda 후보에서 penalized objective 감소가 관찰되었다. 따라서 현재 버전은 monotone EM algorithm이라고 주장하면 안 되고, 논문 버전에서는 line search 또는 MM safeguard를 추가하는 것이 필요하다.

BIC 자유도 역시 구현상 근사다. Eta-group에서는 `df = (K - 1) + d + (K - 1) * m`을 사용한다. 여기서 `m`은 active centered eta coordinate 수, `d`는 coordinate별 공통 eta baseline, `(K - 1) * m`은 선택된 coordinate의 centered contrast 자유도를 나타낸다.

> Reference note. Path+BIC는 penalized mixture model에서 tuning parameter를 고르는 실용적 기준으로 사용한다. 다만 Eta-group의 자유도는 엄밀한 effective degrees of freedom이 아니라 구현 수준의 근사다. 현재 BIC 계산에서는
> $$
> df_{\eta}=(K-1)+d+(K-1)m
> $$
> 를 사용하며, 여기서 $m$은 선택된 centered eta coordinate 수다. 논문에서는 이 값을 implementation-level approximation으로 표기하고, EBIC/RIC-like 또는 path diagnostic을 sensitivity로 함께 제시하는 편이 안전하다.

또한 tuning은 현재 공식 비교에서 path tuning + BIC로 통일했지만, 고차원에서는 BIC가 dense한 model을 선호하거나 반대로 과도하게 sparse한 선택을 할 수 있다. 따라서 본문에서는 BIC 기준 결과를 제시하고, EBIC/RICc는 sensitivity analysis 또는 appendix로 두는 것이 적절하다.

## 17. 연구미팅에서 답할 핵심 문장

Rossi 방법은 component direction mu에 L1 penalty를 둔다. 하지만 vMF posterior decision에는 kappa와 mu의 곱인 eta가 직접 들어간다. 그래서 평균 방향의 sparsity보다 eta contrast의 sparsity가 군집 구분 변수 선택에는 더 직접적인 기준이다.

현재 R 구현에서는 Rossi의 beta path와 유사하게, Eta도 centered eta contrast norm을 기준으로 lambda path를 만들고 BIC로 선택한다. 선택된 변수 support에 대해서는 penalty 없이 refit하여 shrinkage bias를 줄였다.

따라서 제안 방법의 장점은 ARI를 크게 올리는 것보다, ARI를 유지하면서 selected q와 FPR을 크게 줄이고 posterior decision에 직접 관련된 sparse contrast를 회복하는 데 있다.

## References Mentioned

- Banerjee, A., Dhillon, I. S., Ghosh, J., and Sra, S. (2005). *Clustering on the Unit Hypersphere using von Mises-Fisher Distributions*. Journal of Machine Learning Research, 6(46), 1345-1382. Used for vMF mixture EM and concentration approximation based on mean resultant length.
- Rossi, F. and Barbaro, F. (2022). *Mixture of von Mises-Fisher distribution with sparse prototypes*. Neurocomputing, 501, 41-74. DOI: 10.1016/j.neucom.2022.05.118. Used as sparse vMF mixture baseline with $L_1$ sparsity on direction/prototype parameters.
- Yuan, M. and Lin, Y. (2006). *Model Selection and Estimation in Regression with Grouped Variables*. Journal of the Royal Statistical Society: Series B, 68(1), 49-67. DOI: 10.1111/j.1467-9868.2005.00532.x. Used for group lasso and group-level selection.
- Parikh, N. and Boyd, S. (2014). *Proximal Algorithms*. Foundations and Trends in Optimization, 1(3), 123-231. Used for proximal-operator interpretation of the group soft-thresholding step.
