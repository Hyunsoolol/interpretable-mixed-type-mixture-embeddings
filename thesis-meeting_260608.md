# 연구미팅 핵심 정리: Rossi 2022 재현 및 eta 기반 방법 비교

## 1. 연구 질문

Rossi & Barbaro (2022)의 sparse vMF mixture를 R로 재현한 뒤, concentration 차이로 군집이 구분되는 상황에서 기존 `mu` 기반 sparsity와 `eta` 기반 sparsity를 비교했다.

vMF mixture는 다음과 같다.

```math
p(x_i \mid \Theta)
=
\sum_{k=1}^{K}
\alpha_k C_d(\kappa_k)
\exp(\kappa_k \mu_k^\top x_i)
```

posterior decision에 직접 들어가는 natural parameter는 다음이다.

```math
\eta_k = \kappa_k \mu_k
```

K=2에서는 군집 구분의 coordinate-level effect가 다음 contrast로 나타난다.

```math
\eta_2-\eta_1
```

따라서 concentration 차이가 중요한 상황에서는 `mu`보다 `eta` 기준 variable selection이 더 직접적인 해석을 제공한다.

## 2. 비교 방법

비교한 penalty 구조는 다음 세 가지다.

```math
\ell_{\mathrm{Rossi}}
=
\ell(\Theta)
-
\beta
\sum_k \|\mu_k\|_1
```

```math
\ell_{\mathrm{sep}}
=
\ell(\Theta)
-
\lambda_\mu
\sum_k \|\mu_k\|_1
-
\lambda_\kappa
\sum_k \kappa_k
```

```math
\ell_{\eta}
=
\ell(\Theta)
-
\lambda_\eta
\|\eta_2-\eta_1\|_1
```

최종 비교는 refit 포함 6가지 방법으로 수행했다.

| 방법 | penalty 기준 | refit |
|---|---|---|
| Rossi | `mu_k` | 없음 |
| Rossi + refit | Rossi support | 있음 |
| 분리 패널티 | `mu_k`, `kappa_k` | 없음 |
| 분리 패널티 + refit | 분리 패널티 support | 있음 |
| 에타 패널티 | `eta_2 - eta_1` | 없음 |
| 에타 패널티 + refit | 에타 패널티 support | 있음 |

## 3. Main Simulation: concentration-driven setting

기존 `mu` 기반 sparsity의 한계를 보기 위해 평균 방향은 같고 concentration만 다른 상황을 구성했다.

```text
replication = 30
valid replication = 30

K = 2
n = 1000
d = 100
true active q = 10
mixing proportion = (0.5, 0.5)

mu_1 = mu_2
kappa_1 = 20
kappa_2 = 200
kappa ratio = 10
```

true signal은 다음과 같다.

```text
||mu_2 - mu_1|| = 0
||eta_2 - eta_1|| = 180
```

tuning은 다음 기준으로 선택했다.

```text
Rossi:
K = 2
nstart = 5
beta path에서 BIC로 선택

분리 패널티:
lambda_mu = {0, 100, 200, 300, 400, 500, 600}
lambda_kappa = {0, 10, 25, 50, 75}
BIC로 선택

에타 패널티:
lambda_eta = {0, 1, 2, 5, 10, 15, 20, 30, 40, 50}
BIC로 선택
```

Clustering 및 variable selection 결과는 다음과 같다.

| method | ARI | selected q | TPR | FPR | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Rossi | 1.000 | 21.933 | 1.000 | 0.133 | 0.470 | 0.635 |
| Rossi + refit | 1.000 | 21.933 | 1.000 | 0.133 | 0.470 | 0.635 |
| 분리 패널티 | 1.000 | 24.300 | 1.000 | 0.159 | 0.422 | 0.590 |
| 분리 패널티 + refit | 1.000 | 24.300 | 1.000 | 0.159 | 0.422 | 0.590 |
| 에타 패널티 | 1.000 | 11.800 | 1.000 | 0.020 | 0.856 | 0.920 |
| 에타 패널티 + refit | 1.000 | 11.800 | 1.000 | 0.020 | 0.856 | 0.920 |

Parameter estimation 결과는 다음과 같다.

| method | MSE_mu | MSE_kappa | MSE_eta_contrast | kappa ratio | eta contrast norm |
|---|---:|---:|---:|---:|---:|
| Rossi | 1.68e-4 | 1.327 | 0.247 | 10.149 | 181.370 |
| Rossi + refit | 5.57e-5 | 1.406 | 0.362 | 10.040 | 181.025 |
| 분리 패널티 | 1.60e-4 | 1.330 | 0.241 | 10.143 | 181.342 |
| 분리 패널티 + refit | 6.25e-5 | 1.422 | 0.391 | 10.034 | 181.052 |
| 에타 패널티 | 1.64e-4 | 9.254 | 0.360 | 8.428 | 174.767 |
| 에타 패널티 + refit | 3.36e-5 | 1.249 | 0.179 | 10.052 | 180.828 |

해석:

- 모든 방법의 `ARI = 1.000`이므로, main setting에서는 clustering 차이가 핵심이 아니다.
- 모든 방법의 `TPR = 1.000`으로 true active coordinate는 모두 찾았다.
- 차이는 false positive control에서 나타났다.
- 에타 패널티는 `selected q = 11.800`, `FPR = 0.020`, `F1 = 0.920`으로 가장 안정적이었다.
- 에타 패널티는 refit 전에는 concentration을 shrink하지만, refit 후 `kappa ratio`와 `eta contrast norm`이 true value에 가까워졌다.

## 4. 추가 Simulation 1: concentration 차이가 약한 경우

```text
K = 2
n = 1000
d = 100
true active q = 10
replication = 30

mu_1 = mu_2
kappa_1 = 20
kappa_2 = 40
true kappa ratio = 2
true eta contrast norm = 20
```

Clustering 및 variable selection 결과는 다음과 같다.

| method | ARI | selected q | FPR | F1 |
|---|---:|---:|---:|---:|
| Rossi | 0.363 | 41.133 | 0.346 | 0.502 |
| Rossi + refit | 0.349 | 41.133 | 0.346 | 0.502 |
| 분리 패널티 | 0.360 | 35.133 | 0.279 | 0.585 |
| 분리 패널티 + refit | 0.356 | 35.133 | 0.279 | 0.585 |
| 에타 패널티 | 0.343 | 11.467 | 0.016 | 0.935 |
| 에타 패널티 + refit | 0.367 | 11.467 | 0.016 | 0.935 |

Parameter estimation 결과는 다음과 같다.

| method | MSE_mu | MSE_kappa | MSE_eta_contrast | kappa ratio |
|---|---:|---:|---:|---:|
| Rossi | 1.43e-4 | 1.826 | 0.252 | 2.023 |
| Rossi + refit | 2.85e-4 | 1.558 | 0.523 | 1.988 |
| 분리 패널티 | 2.25e-4 | 1.412 | 0.355 | 2.008 |
| 분리 패널티 + refit | 2.67e-4 | 1.217 | 0.457 | 1.993 |
| 에타 패널티 | 1.82e-4 | 10.544 | 0.421 | 1.623 |
| 에타 패널티 + refit | 8.51e-5 | 1.618 | 0.188 | 2.001 |

concentration 차이가 약하면 clustering 자체가 어렵다. 그러나 variable selection에서는 에타 패널티가 `selected q`, `FPR`, `F1`에서 가장 안정적이었다. Refit 후에는 `kappa ratio = 2.001`로 true value에 가까워졌다.

## 5. 추가 Simulation 2: 평균과 집중도 차이가 모두 있는 경우

```text
K = 2
n = 1000
d = 100
true active q = 10
replication = 30

mu_cos = 0.95
kappa_1 = 20
kappa_2 = 100
true mu contrast norm = 0.316
true kappa ratio = 5
true eta contrast norm = 81.240
```

Clustering 및 variable selection 결과는 다음과 같다.

| method | ARI | selected q | FPR | F1 |
|---|---:|---:|---:|---:|
| Rossi | 0.995 | 12.467 | 0.027 | 0.895 |
| Rossi + refit | 0.995 | 12.467 | 0.027 | 0.895 |
| 분리 패널티 | 0.995 | 13.700 | 0.041 | 0.855 |
| 분리 패널티 + refit | 0.995 | 13.700 | 0.041 | 0.855 |
| 에타 패널티 | 0.994 | 11.633 | 0.018 | 0.944 |
| 에타 패널티 + refit | 0.995 | 11.633 | 0.018 | 0.944 |

Parameter estimation 결과는 다음과 같다.

| method | MSE_mu | MSE_kappa | MSE_eta_contrast | kappa ratio |
|---|---:|---:|---:|---:|
| Rossi | 1.16e-4 | 0.336 | 0.157 | 5.057 |
| Rossi + refit | 3.15e-5 | 0.343 | 0.110 | 5.034 |
| 분리 패널티 | 1.25e-4 | 0.346 | 0.165 | 5.061 |
| 분리 패널티 + refit | 3.65e-5 | 0.353 | 0.131 | 5.033 |
| 에타 패널티 | 1.34e-4 | 8.147 | 0.364 | 4.210 |
| 에타 패널티 + refit | 3.62e-5 | 0.336 | 0.107 | 5.029 |

평균 차이까지 있으면 모든 방법의 ARI가 거의 1에 가깝다. Variable selection에서는 에타 패널티가 가장 안정적이고, parameter estimation에서는 에타 패널티 + refit이 `MSE_eta_contrast = 0.107`로 가장 낮았다.

## 6. 현재 결론

1. Rossi 2022 방법은 논문과 같은 정성적 패턴으로 재현되었다.
2. Rossi 방법은 clustering은 잘하지만 `mu_k` 중심 sparsity라 concentration-driven variable effect를 직접 설명하기 어렵다.
3. 분리 패널티는 `mu_k`와 `kappa_k`를 분리해서 penalize하지만, `kappa_k`가 scalar라 coordinate-level concentration effect를 직접 선택하지 못한다.
4. `eta_k = kappa_k mu_k` 기준 penalty는 posterior decision에 직접 들어가는 coordinate effect를 선택한다.
5. 에타 패널티는 TPR을 유지하면서 FPR을 크게 낮췄다.
6. 에타 패널티 + refit은 support를 유지하면서 `MSE_mu`, `MSE_kappa`, `MSE_eta_contrast`, `kappa ratio`, `eta norm`을 true value에 가깝게 개선했다.

## 부록 A. Rossi & Barbaro (2022) 재현

논문 artificial simulation 구조를 R로 재현했다. 계산 시간 때문에 medium-budget setting으로 먼저 확인했다.

```text
d = 100
K* = 4
N = 200, 1000
overlap = 2.5%, 5%
true nonzero fraction = 5%, 10%, 15%
candidate K = 1,...,6
replication = 20
random starts = 5
```

주요 결과는 다음과 같다.

| setting | dense vMF | sparse vMF BIC | 해석 |
|---|---|---|---|
| `N = 1000` | true `K=4` 근처 회복 | `K=4` 안정적 선택 | 논문과 같은 패턴 |
| `N = 200` | `K=1` 또는 `K=2` underfit | `K=3` 또는 `K=4`로 개선 | sparsity가 effective dimension을 줄임 |
| variable recovery | recall 높음 | precision은 조건별 차이 | true active는 많이 찾지만 false positive 존재 |

`N=1000, K=4`에서 BIC 기준 nonzero entry recovery는 다음 범위였다.

```text
recall:    0.863 - 0.960
precision: 0.444 - 0.891
```

Rossi 방법은 논문 결과와 같은 정성적 패턴을 재현했다. 다만 variable recovery에서는 false positive가 남았다.

## 부록 B. Eta screening + refit 확인

계산적으로 단순한 eta screening + refit도 추가로 확인했다.

```text
1. dense vMF mixture 적합
2. |eta_2j - eta_1j| score 계산
3. support path를 만들고 BIC로 support size 선택
4. 선택된 support에서 penalty 없이 refit
```

결과는 다음과 같다.

```text
ARI = 1.000
selected q = 10.000
TPR = 1.000
FPR = 0.000
Precision = 1.000
F1 = 1.000
eta contrast norm = 180.766
kappa ratio = 10.063
```

`eta` contrast score 자체가 concentration-driven active coordinate를 잘 포착한다는 근거다.
