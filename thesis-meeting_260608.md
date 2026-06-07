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

## 7. 분리 패널티 EM

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

## 8. 에타 패널티 EM

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

## 9. 6가지 방법 비교 결과

True 값은 다음과 같다.

```text
selected q = 10
mu contrast norm = 0
eta contrast norm = 180
kappa ratio = 10
```

| method | ARI | selected q | TPR | FPR | Precision | F1 | eta contrast norm | kappa ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Rossi | 1.000 | 21.933 | 1.000 | 0.133 | 0.470 | 0.635 | 181.370 | 10.149 |
| Rossi + refit | 1.000 | 21.933 | 1.000 | 0.133 | 0.470 | 0.635 | 181.025 | 10.040 |
| 분리 패널티 | 1.000 | 24.300 | 1.000 | 0.159 | 0.422 | 0.590 | 181.342 | 10.143 |
| 분리 패널티 + refit | 1.000 | 24.300 | 1.000 | 0.159 | 0.422 | 0.590 | 181.052 | 10.034 |
| 에타 패널티 | 1.000 | 11.800 | 1.000 | 0.020 | 0.856 | 0.920 | 174.767 | 8.428 |
| 에타 패널티 + refit | 1.000 | 11.800 | 1.000 | 0.020 | 0.856 | 0.920 | 180.828 | 10.052 |

## 10. 결과 해석

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

## 11. Eta screening + refit 확인

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

## 12. Robustness pilot

한계 setting 외에도 다음 grid에서 pilot simulation을 수행했다.

```text
kappa ratio = {2, 5, 10}
mu_cos = {1.00, 0.99, 0.95}
replication = 10
```

요약은 다음과 같다.

| setting | 결과 |
|---|---|
| `kappa ratio = 2` | signal이 약해 모든 방법의 ARI가 낮음. 그래도 eta 기반 방법이 support recovery에서 상대적으로 안정적 |
| `kappa ratio = 5` | 모든 방법의 ARI가 약 0.99. eta penalty/refit과 eta screening/refit의 support F1이 가장 높음 |
| `kappa ratio = 10` | 모든 방법의 ARI가 1.00. Rossi와 분리 패널티는 selected q를 약 22-25로 넓게 선택, eta penalty/refit은 약 11-12로 true q에 가까움 |

## 13. 현재 결론

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

6. 에타 패널티는 TPR을 유지하면서 FPR을 크게 낮췄고,
   refit 후 eta norm과 kappa ratio도 true value에 가까워졌다.
```

## 14. 다음 단계

pilot 수준의 robustness simulation을 final simulation으로 확장한다.

```text
replication = 30 or 50
kappa ratio = {2, 5, 10}
mu_cos = {1.00, 0.99, 0.95}
```

최종 결과는 다음 지표 중심으로 정리한다.

```text
ARI
selected q
TPR
FPR
Precision
F1
eta contrast norm
kappa ratio
```

실제 directional data 예제를 추가하면, eta 기반 support가 실제 clustering 해석에도 도움이 된다는 점을 보일 수 있다.
