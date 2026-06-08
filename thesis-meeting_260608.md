# 연구미팅 핵심 정리: Rossi 2022 재현 및 eta 기반 방법 비교

## 1. 목적

Rossi & Barbaro (2022)의 sparse vMF mixture를 R로 재현한 뒤, concentration 차이로 군집이 구분되는 상황에서 기존 `mu` 기반 sparsity와 제안하는 `eta` 기반 sparsity를 비교했다.

핵심 질문은 다음이다.

```text
vMF mixture에서 concentration 차이가 군집 구분을 만들 때,
어떤 coordinate가 군집 구분에 기여했는지를 무엇을 기준으로 선택해야 하는가?
```

## 2. 기본 모형

vMF density는 다음과 같다.

```math
f(x \mid \mu, \kappa)
= C_d(\kappa)\exp(\kappa \mu^\top x),
\qquad x,\mu \in S^{d-1}.
```

vMF mixture는 다음과 같다.

```math
p(x_i \mid \Theta)
= \sum_{k=1}^{K}
\alpha_k C_d(\kappa_k)
\exp(\kappa_k \mu_k^\top x_i).
```

posterior decision에 직접 들어가는 자연모수는

```math
\eta_k = \kappa_k \mu_k
```

이다. 따라서 component 간 구분은 `mu_k`만이 아니라 `eta_k`의 차이에 의해 결정된다.

## 3. Rossi & Barbaro (2022) sparse vMF

Rossi & Barbaro (2022)는 component direction `mu_k`에 L1 penalty를 둔다.

```math
\ell_p(\Theta)
= \ell(\Theta)
- \beta \sum_{k=1}^{K}\|\mu_k\|_1.
```

EM update의 주요 형태는 다음과 같다.

```math
\tau_{ik}
=
\frac{
\alpha_k C_d(\kappa_k)\exp(\kappa_k\mu_k^\top x_i)
}{
\sum_{\ell=1}^{K}
\alpha_\ell C_d(\kappa_\ell)\exp(\kappa_\ell\mu_\ell^\top x_i)
}.
```

```math
N_k = \sum_i \tau_{ik},
\qquad
r_k = \sum_i \tau_{ik}x_i.
```

```math
\mu_{kj}
\propto
\mathrm{sign}(r_{kj})
\left(\kappa_k |r_{kj}| - \beta\right)_+.
```

```math
\rho_k = \frac{\mu_k^\top r_k}{N_k},
\qquad
\kappa_k \approx
\frac{d\rho_k - \rho_k^3}{1-\rho_k^2}.
```

이 방법은 sparse prototype을 제공하지만, penalty와 해석 대상이 `mu_k`에 있다. 그러나 posterior decision에는 `eta_k = kappa_k mu_k`가 직접 들어간다.

## 4. 2022 논문 재현 결과

논문 artificial simulation 구조를 R로 재현했다. 계산 시간 때문에 먼저 medium-budget setting으로 수행했다.

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
| `N = 1000` | true `K=4` 근처 회복 | `K=4`를 안정적으로 선택 | 논문과 같은 패턴 |
| `N = 200` | `K=1` 또는 `K=2`로 underfit 경향 | `K=3` 또는 `K=4`로 개선 | sparsity가 effective dimension을 줄임 |
| variable recovery | recall 높음 | precision은 조건에 따라 차이 | true active를 많이 찾지만 false positive 존재 |

`N=1000, K=4`에서 BIC 기준 nonzero entry recovery는 다음 범위였다.

```text
recall:    0.863 - 0.960
precision: 0.444 - 0.891
```

정리하면, Rossi 방법은 논문 결과와 같은 정성적 패턴을 재현했다. 즉, clustering과 `K` 선택은 잘 수행하지만, variable recovery에서는 false positive가 남는다.

## 5. concentration-driven 한계 setting

기존 `mu` 기반 sparsity의 한계를 보기 위해 다음 setting을 구성했다.

```text
K = 2
n = 1000
d = 100
true active q = 10
replication = 30

mu_1 = mu_2
kappa_1 = 20
kappa_2 = 200
```

이 경우 평균 방향은 같다.

```math
\|\mu_2 - \mu_1\| = 0.
```

하지만 concentration은 크게 다르다.

```text
kappa ratio = 10
```

따라서 true signal은 `mu` 차이가 아니라 `eta` 차이에 있다.

```math
\|\eta_2 - \eta_1\| = 180.
```

즉, 군집은 concentration 차이로 구분되지만 `mu_1 = mu_2`이므로, `mu` 중심 variable selection은 군집 구분에 기여한 coordinate를 직접 설명하기 어렵다.

## 6. 비교 방법

같은 setting에서 세 가지 penalized 방법과 각 refit 버전을 비교했다.

| 방법 | penalty 대상 | support 선택 기준 | refit |
|---|---|---|---|
| Rossi | `mu_k` | `mu_k`의 nonzero coordinate | 없음 |
| Rossi + refit | Rossi support 고정 | Rossi support | 있음 |
| 분리 패널티 | `mu_k`, `kappa_k` | `mu_k`의 nonzero coordinate | 없음 |
| 분리 패널티 + refit | 분리 패널티 support 고정 | 분리 패널티 support | 있음 |
| 에타 패널티 | `eta_2 - eta_1` | `eta_2j - eta_1j`의 nonzero coordinate | 없음 |
| 에타 패널티 + refit | 에타 패널티 support 고정 | 에타 패널티 support | 있음 |

Refit은 다음 절차를 의미한다.

```text
1. penalized model에서 support S 선택
2. S를 고정
3. j not in S인 coordinate는 0으로 고정
4. penalty 없이 vMF mixture 재추정
```

즉,

```math
\mu_{kj} = 0 \quad \text{for } j \notin S
```

라는 제약 아래 unpenalized EM을 다시 수행한다.

## 7. 평가지표

6가지 방법은 다음 세 범주의 지표로 비교한다.

```text
Clustering:
ARI

Variable selection:
selected q
TPR
FPR
Precision
F1

Parameter estimation:
MSE_mu
MSE_kappa
MSE_eta_contrast
kappa_1_hat
kappa_2_hat
kappa ratio
```

`ARI`는 clustering label이 true label과 얼마나 일치하는지를 본다.

`selected q`, `TPR`, `FPR`, `Precision`, `F1`은 active coordinate recovery를 평가한다. 이 연구의 핵심은 clustering 자체뿐 아니라 군집 차이를 설명하는 coordinate를 얼마나 정확히 선택하는지에 있다.

Parameter estimation은 다음 MSE로 평가한다.

```math
\mathrm{MSE}_{\mu}
=
\frac{1}{Kd}
\sum_{k=1}^{K}
\|\hat{\mu}_k-\mu_k\|_2^2.
```

```math
\mathrm{MSE}_{\kappa}
=
\frac{1}{K}
\sum_{k=1}^{K}
(\hat{\kappa}_k-\kappa_k)^2.
```

```math
\mathrm{MSE}_{\Delta\eta}
=
\frac{1}{d}
\sum_{j=1}^{d}
\left[
(\hat{\eta}_{2j}-\hat{\eta}_{1j})
-
(\eta_{2j}-\eta_{1j})
\right]^2.
```

여기서 `eta_k = kappa_k mu_k`이고, `MSE_eta_contrast`는 `eta_2 - eta_1`에 대한 MSE를 의미한다. Posterior decision에서 직접 비교되는 값이 `eta_2 - eta_1`이므로, 이 지표가 concentration-driven setting에서 가장 중요한 parameter estimation 지표다.

MSE 계산 전에는 label switching을 정리해야 한다. 현재 simulation은 `kappa_1 < kappa_2` 구조이므로, 추정 component도 `kappa`가 작은 component와 큰 component 순서로 정렬한 뒤 true parameter와 비교한다.

## 8. 분리 패널티 EM

교수님 제안에 따라 `mu`와 `kappa`에 penalty를 분리해서 두는 EM을 구현했다.

```math
Q_{\text{pen}}
= \ell(\Theta)
- \lambda_\mu \sum_k \|\mu_k\|_1
- \lambda_\kappa \sum_k \kappa_k.
```

주요 update는 다음과 같다.

```math
z_{kj}
=
\mathrm{sign}(r_{kj})
\left(\kappa_k |r_{kj}| - \lambda_\mu\right)_+,
\qquad
\mu_k = z_k / \|z_k\|_2.
```

```math
s_k = \mu_k^\top r_k,
\qquad
\rho_k = \frac{s_k-\lambda_\kappa}{N_k},
\qquad
\kappa_k = A_d^{-1}(\rho_k).
```

하지만 `kappa_k`는 coordinate-specific parameter가 아니라 component-level scalar이다. 따라서 `kappa_k`에 별도 penalty를 두어도 어떤 coordinate가 concentration 차이에 기여했는지를 직접 선택하기 어렵다.

## 9. 에타 패널티 EM

제안 방향은 자연모수 `eta_k`를 기준으로 variable selection을 수행하는 것이다.

```math
\eta_k = \kappa_k \mu_k.
```

K = 2에서 prototype objective는 다음과 같다.

```math
\ell(\Theta)
- \lambda_\eta \sum_j |\eta_{2j} - \eta_{1j}|.
```

구현은 practical proximal EM 형태로 수행했다.

```text
1. unpenalized vMF M-step으로 eta_k 계산
2. delta = eta_2 - eta_1
3. delta_j <- sign(delta_j)(|delta_j| - lambda_eta)_+
4. eta_k를 다시 mu_k, kappa_k로 변환
```

이 방식은 posterior decision에 직접 들어가는 coordinate effect `eta_2j - eta_1j`를 직접 shrink하고 선택한다.

## 10. 6가지 방법 비교 결과

아래 표는 다음 기본 setting에서 30회 반복한 평균 결과이다.

```text
replication = 30
valid replication = 30

K_true = 2
n = 1000
d = 100
true active q = 10
mixing proportion = (0.5, 0.5)

mu_1 = mu_2
kappa_1 = 20
kappa_2 = 200
kappa ratio = 10
```

각 반복에서는 같은 생성 구조의 데이터를 새로 만들고, 여섯 가지 방법을 같은 concentration-driven setting에서 비교했다.

```text
Rossi:
K grid = {1, 2, 3}
nstart = 5
beta path에서 BIC로 선택

분리 패널티:
K = 2
lambda_mu = {0, 100, 200, 300, 400, 500, 600}
lambda_kappa = {0, 10, 25, 50, 75}
BIC로 tuning 선택

에타 패널티:
K = 2
lambda_eta = {0, 1, 2, 5, 10, 15, 20, 30, 40, 50}
BIC로 tuning 선택
```

True 값은 다음과 같다.

```text
selected q = 10
mu contrast norm = 0
eta contrast norm = 180
kappa_1 = 20
kappa_2 = 200
kappa ratio = 10
```

Clustering과 variable selection 결과는 다음과 같다.

| method | ARI | selected q | TPR | FPR | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Rossi | 1.000 | 21.933 | 1.000 | 0.133 | 0.470 | 0.635 |
| Rossi + refit | 1.000 | 21.933 | 1.000 | 0.133 | 0.470 | 0.635 |
| 분리 패널티 | 1.000 | 24.300 | 1.000 | 0.159 | 0.422 | 0.590 |
| 분리 패널티 + refit | 1.000 | 24.300 | 1.000 | 0.159 | 0.422 | 0.590 |
| 에타 패널티 | 1.000 | 11.800 | 1.000 | 0.020 | 0.856 | 0.920 |
| 에타 패널티 + refit | 1.000 | 11.800 | 1.000 | 0.020 | 0.856 | 0.920 |

Parameter estimation 결과는 다음과 같다.

| method | MSE_mu | MSE_kappa | MSE_eta_contrast | kappa_1_hat | kappa_2_hat | kappa ratio | eta contrast norm |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rossi | 1.68e-4 | 1.327 | 0.247 | 19.795 | 200.796 | 10.149 | 181.370 |
| Rossi + refit | 5.57e-5 | 1.406 | 0.362 | 20.021 | 200.920 | 10.040 | 181.025 |
| 분리 패널티 | 1.60e-4 | 1.330 | 0.241 | 19.808 | 200.800 | 10.143 | 181.342 |
| 분리 패널티 + refit | 6.25e-5 | 1.422 | 0.391 | 20.035 | 200.946 | 10.034 | 181.052 |
| 에타 패널티 | 1.64e-4 | 9.254 | 0.360 | 23.489 | 197.912 | 8.428 | 174.767 |
| 에타 패널티 + refit | 3.36e-5 | 1.249 | 0.179 | 19.978 | 200.722 | 10.052 | 180.828 |

MSE 지표를 포함해 같은 setting에서 30회 재실행했으며, 모든 방법에서 실패한 반복은 없었다.

## 11. 결과 해석

모든 방법의 ARI가 1.000이므로, 이 setting에서는 clustering 성능 차이가 핵심이 아니다.

또한 모든 방법의 TPR이 1.000이므로, true active coordinate는 모두 찾았다.

차이는 false positive control에서 나타난다.

```text
Rossi:
selected q = 21.933
FPR = 0.133
F1 = 0.635

분리 패널티:
selected q = 24.300
FPR = 0.159
F1 = 0.590

에타 패널티:
selected q = 11.800
FPR = 0.020
F1 = 0.920
```

Rossi와 분리 패널티는 refit 후에도 support 지표가 변하지 않았다.

```text
Rossi F1: 0.635 -> 0.635
분리 패널티 F1: 0.590 -> 0.590
```

따라서 두 방법의 한계는 refit 부재가 아니라 variable selection target에 있다.

에타 패널티는 TPR을 유지하면서 FPR을 크게 낮췄다. 다만 refit 전에는 penalty 때문에 `eta` contrast와 kappa ratio가 shrink되었다.

```text
에타 패널티:
eta contrast norm = 174.767
kappa ratio = 8.428
```

refit 후에는 support는 유지하면서 추정량이 true value에 가까워졌다.

```text
에타 패널티 + refit:
eta contrast norm = 180.828
kappa ratio = 10.052
```

MSE 기준으로도 에타 패널티 + refit이 가장 좋은 parameter estimation 결과를 보였다.

```text
에타 패널티:
MSE_kappa = 9.254
MSE_eta_contrast = 0.360

에타 패널티 + refit:
MSE_kappa = 1.249
MSE_eta_contrast = 0.179
```

즉, 에타 패널티는 support selection에서 가장 좋고, refit을 추가하면 penalty shrinkage가 줄어들어 concentration과 eta contrast 추정도 개선된다.

## 12. Eta screening + refit 확인

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

이는 `eta` contrast score 자체가 concentration-driven active coordinate를 잘 포착한다는 근거다. 최종 방법은 penalized likelihood 안에서 eta contrast를 직접 다루는 에타 패널티 EM으로 정리하고, eta screening + refit은 practical comparison으로 둘 수 있다.

## 13. 추가 simulation

Main setting 외에 두 가지 상황을 추가로 확인했다. 모든 setting에서 `n = 1000`, `d = 100`, `true active q = 10`, `replication = 30`으로 두었다. 모든 방법에서 실패한 반복은 없었다.

### 13.1 집중도 차이가 더 작은 경우

평균 방향 차이는 없고, concentration 차이만 약하게 둔 경우다.

```text
mu_1 = mu_2
kappa_1 = 20
kappa_2 = 40
true kappa ratio = 2
true eta contrast norm = 20
```

Clustering과 variable selection 결과는 다음과 같다.

| method | ARI | selected q | TPR | FPR | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Rossi | 0.142 | 14.467 | 1.000 | 0.050 | 0.856 | 0.893 |
| Rossi + refit | 0.140 | 14.467 | 1.000 | 0.050 | 0.856 | 0.893 |
| 분리 패널티 | 0.360 | 35.133 | 1.000 | 0.279 | 0.463 | 0.585 |
| 분리 패널티 + refit | 0.356 | 35.133 | 1.000 | 0.279 | 0.463 | 0.585 |
| 에타 패널티 | 0.343 | 11.467 | 1.000 | 0.016 | 0.882 | 0.935 |
| 에타 패널티 + refit | 0.367 | 11.467 | 1.000 | 0.016 | 0.882 | 0.935 |

Parameter estimation 결과는 다음과 같다.

| method | MSE_mu | MSE_kappa | MSE_eta_contrast | kappa_1_hat | kappa_2_hat | kappa ratio | eta contrast norm |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rossi | 1.06e-4 | 0.932 | 0.190 | 19.870 | 40.332 | 2.032 | 21.002 |
| Rossi + refit | 1.36e-4 | 0.959 | 0.292 | 20.206 | 40.705 | 2.017 | 21.163 |
| 분리 패널티 | 2.25e-4 | 1.412 | 0.355 | 19.704 | 39.503 | 2.008 | 20.889 |
| 분리 패널티 + refit | 2.67e-4 | 1.217 | 0.457 | 20.013 | 39.841 | 1.993 | 21.086 |
| 에타 패널티 | 1.82e-4 | 10.544 | 0.421 | 23.375 | 37.893 | 1.623 | 15.030 |
| 에타 패널티 + refit | 8.51e-5 | 1.618 | 0.188 | 20.171 | 40.286 | 2.001 | 20.623 |

이 setting에서는 concentration 차이가 약해 clustering 자체가 어렵다. ARI가 전체적으로 낮아졌고, 특히 Rossi는 BIC가 `K = 1`을 선택하는 반복이 있어 ARI가 낮게 나왔다. 그러나 variable selection 기준에서는 에타 패널티가 가장 안정적이다. `selected q`가 true value 10에 가장 가깝고, FPR과 F1도 가장 좋다.

Refit 전 에타 패널티는 concentration을 shrink해서 `kappa ratio = 1.623`으로 작게 추정하지만, refit 후에는 `kappa ratio = 2.001`로 true value에 가까워진다. `MSE_eta_contrast`도 refit 후 0.188로 가장 낮다.

### 13.2 평균과 집중도 차이가 모두 있는 경우

평균 방향 차이와 concentration 차이가 동시에 있는 경우다.

```text
mu_cos = 0.95
kappa_1 = 20
kappa_2 = 100
true mu contrast norm = 0.316
true kappa ratio = 5
true eta contrast norm = 81.240
```

Clustering과 variable selection 결과는 다음과 같다.

| method | ARI | selected q | TPR | FPR | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Rossi | 0.995 | 12.467 | 1.000 | 0.027 | 0.817 | 0.895 |
| Rossi + refit | 0.995 | 12.467 | 1.000 | 0.027 | 0.817 | 0.895 |
| 분리 패널티 | 0.995 | 13.700 | 1.000 | 0.041 | 0.760 | 0.855 |
| 분리 패널티 + refit | 0.995 | 13.700 | 1.000 | 0.041 | 0.760 | 0.855 |
| 에타 패널티 | 0.994 | 11.633 | 1.000 | 0.018 | 0.912 | 0.944 |
| 에타 패널티 + refit | 0.995 | 11.633 | 1.000 | 0.018 | 0.912 | 0.944 |

Parameter estimation 결과는 다음과 같다.

| method | MSE_mu | MSE_kappa | MSE_eta_contrast | kappa_1_hat | kappa_2_hat | kappa ratio | eta contrast norm |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rossi | 1.16e-4 | 0.336 | 0.157 | 19.867 | 100.422 | 5.057 | 82.178 |
| Rossi + refit | 3.15e-5 | 0.343 | 0.110 | 19.967 | 100.477 | 5.034 | 81.832 |
| 분리 패널티 | 1.25e-4 | 0.346 | 0.165 | 19.852 | 100.424 | 5.061 | 82.217 |
| 분리 패널티 + refit | 3.65e-5 | 0.353 | 0.131 | 19.976 | 100.493 | 5.033 | 81.852 |
| 에타 패널티 | 1.34e-4 | 8.147 | 0.364 | 23.267 | 97.849 | 4.210 | 76.422 |
| 에타 패널티 + refit | 3.62e-5 | 0.336 | 0.107 | 19.982 | 100.444 | 5.029 | 81.811 |

평균 차이까지 있으면 모든 방법의 ARI가 거의 1에 가깝다. 하지만 variable selection에서는 여전히 에타 패널티가 가장 좋다. Rossi와 분리 패널티는 true active q보다 넓게 선택하는 경향이 있고, 에타 패널티는 selected q, FPR, Precision, F1에서 가장 안정적이다.

Parameter estimation에서도 에타 패널티 + refit이 가장 좋은 결과를 보였다. `MSE_eta_contrast = 0.107`로 가장 낮고, `kappa ratio = 5.029`로 true value 5에 가깝다.

## 14. 현재 결론

현재까지의 결과는 다음과 같이 정리된다.

```text
1. Rossi 2022 방법은 논문과 같은 정성적 패턴으로 재현되었다.

2. Rossi 방법은 clustering은 잘하지만,
   mu_k 중심 sparsity라 concentration-driven variable effect를 직접 설명하기 어렵다.

3. Rossi + refit과 분리 패널티 + refit을 해도 support recovery는 개선되지 않았다.

4. 분리 패널티의 한계는 scalar kappa와 mu를 분리해서 penalize하는 구조가
   coordinate-level concentration effect를 직접 선택하지 못한다는 점이다.

5. eta_k = kappa_k mu_k를 기준으로 penalty를 두면,
   posterior decision에 직접 들어가는 coordinate effect를 선택할 수 있다.

6. 에타 패널티는 TPR을 유지하면서 FPR을 크게 낮췄다.

7. 에타 패널티 + refit은 support를 유지하면서
   MSE_kappa와 MSE_eta_contrast를 줄였고,
   kappa ratio와 eta norm도 true value에 가까워졌다.

8. concentration 차이가 매우 약한 경우에는 clustering 자체가 어려워지지만,
   에타 패널티는 support recovery에서 가장 안정적이었다.

9. 평균 차이와 concentration 차이가 모두 있는 경우에도
   에타 패널티는 variable selection에서 가장 좋고,
   에타 패널티 + refit은 MSE_eta_contrast가 가장 낮았다.
```
