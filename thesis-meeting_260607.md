# 연구미팅 정리: 2022 sparse vMF 재현 이후 진행 내용

## 1. 진행 배경

지난 회의에서 제안 모형의 큰 구조는 확인받았다. 이후 바로 제안 모형 시뮬레이션으로 들어가기보다, 먼저 Rossi & Barbaro (2022)의 sparse vMF mixture 방법을 R로 재현하고, 그 방법이 어떤 상황에서 해석상 한계를 가지는지 확인했다.

핵심 질문은 다음과 같다.

```text
vMF mixture에서 군집 차이가 concentration 차이로 나타나는 경우,
어떤 coordinate가 군집 구분에 기여했는지를 어떻게 선택하고 해석할 것인가?
```

vMF mixture에서 posterior decision에 직접 들어가는 자연모수는 다음이다.

```text
eta_k = kappa_k mu_k
```

따라서 component 간 decision boundary는 `mu_k`만이 아니라 `eta_k`의 차이에 의해 결정된다.

## 2. Rossi & Barbaro (2022) 재현

Rossi & Barbaro (2022)는 sparse vMF mixture에서 component별 평균 방향 `mu_k`에 L1 penalty를 둔다.

```text
log L - beta sum_k ||mu_k||_1
```

R로 2022 논문의 시뮬레이션 구조를 재현했고, 정성적인 패턴은 논문과 일치했다.

다만 한계는 다음이다.

```text
Rossi 방법의 penalty와 해석 대상은 mu_k이다.
그러나 vMF mixture의 posterior decision에는 eta_k = kappa_k mu_k가 직접 들어간다.
따라서 concentration 차이가 중요한 상황에서는 mu_k 중심의 sparsity가 해석상 한계를 가질 수 있다.
```

## 3. 한계 시뮬레이션 설정

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

이 설정에서는 평균 방향은 같다.

```text
||mu_2 - mu_1|| = 0
```

하지만 concentration은 크게 다르다.

```text
kappa ratio = 10
```

따라서 true signal은 `mu` 차이가 아니라 `eta` 차이에 있다.

```text
||eta_2 - eta_1|| = 180
```

즉, 군집은 concentration 차이로 구분되지만 `mu_1`과 `mu_2`는 같으므로, `mu` 중심 variable selection은 군집 구분에 기여한 coordinate를 직접 설명하기 어렵다.

## 4. 비교한 방법

이번에는 세 가지 penalized 방법과 각 refit 버전을 비교했다.

```text
1. Rossi
2. Rossi + refit
3. 분리 패널티
4. 분리 패널티 + refit
5. 에타 패널티
6. 에타 패널티 + refit
```

Refit은 모든 방법에서 같은 의미다.

```text
1. penalized model에서 active support S 선택
2. S를 고정
3. 선택되지 않은 coordinate는 0으로 고정
4. penalty 없이 vMF mixture를 다시 추정
```

즉,

```text
mu_kj = 0 for j not in S
```

라는 제약 아래 unpenalized EM을 다시 수행한다. Refit은 support를 새로 선택하는 단계가 아니라, 선택된 support 안에서 shrinkage bias를 줄이는 단계다.

## 5. 분리 패널티 EM

교수님 제안에 따라 `mu`와 `kappa`에 penalty를 분리해서 두는 EM을 구현했다.

```text
Q_pen
= log L
- lambda_mu sum_k ||mu_k||_1
- lambda_kappa sum_k kappa_k
```

업데이트 구조는 다음과 같다.

```text
z_kj = sign(r_kj) (kappa_k |r_kj| - lambda_mu)_+
mu_k = z_k / ||z_k||_2
```

그리고 `kappa_k`는 다음으로 업데이트한다.

```text
s_k = mu_k^T r_k
rho_k = (s_k - lambda_kappa) / N_k
kappa_k = A_d^{-1}(rho_k)
```

중요한 점은 `kappa_k`가 coordinate별 parameter가 아니라 component-level scalar라는 것이다. 따라서 `lambda_kappa`를 따로 두어도 어떤 coordinate가 concentration-driven separation에 기여했는지를 직접 선택하기 어렵다.

## 6. 에타 패널티 EM

제안 방향은 자연모수 `eta_k`를 기준으로 variable selection을 수행하는 것이다.

```text
eta_k = kappa_k mu_k
```

K = 2에서 에타 패널티 prototype은 다음 목적함수를 사용한다.

```text
log L - lambda_eta sum_j |eta_2j - eta_1j|
```

즉 posterior decision에 직접 들어가는 coordinate effect,

```text
eta_2j - eta_1j
```

를 직접 shrink하고 선택한다.

구현은 practical proximal EM 형태로 진행했다.

```text
1. unpenalized vMF M-step으로 eta_k 계산
2. delta = eta_2 - eta_1
3. delta_j <- sign(delta_j)(|delta_j| - lambda_eta)_+
4. eta_k를 다시 mu_k, kappa_k로 변환
```

이후 에타 패널티가 선택한 support를 고정하고 penalty 없이 refit을 수행했다.

## 7. 6가지 방법 비교 결과

```text
True:
selected q = 10
||mu_2 - mu_1|| = 0
||eta_2 - eta_1|| = 180
kappa ratio = 10
```

| method | ARI | selected q | TPR | FPR | Precision | F1 | `||mu2-mu1||` | `||eta2-eta1||` | kappa ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Rossi | 1.000 | 21.933 | 1.000 | 0.133 | 0.470 | 0.635 | 0.181 | 181.370 | 10.149 |
| Rossi + refit | 1.000 | 21.933 | 1.000 | 0.133 | 0.470 | 0.635 | 0.105 | 181.025 | 10.040 |
| 분리 패널티 | 1.000 | 24.300 | 1.000 | 0.159 | 0.422 | 0.590 | 0.175 | 181.342 | 10.143 |
| 분리 패널티 + refit | 1.000 | 24.300 | 1.000 | 0.159 | 0.422 | 0.590 | 0.110 | 181.052 | 10.034 |
| 에타 패널티 | 1.000 | 11.800 | 1.000 | 0.020 | 0.856 | 0.920 | 0.161 | 174.767 | 8.428 |
| 에타 패널티 + refit | 1.000 | 11.800 | 1.000 | 0.020 | 0.856 | 0.920 | 0.085 | 180.828 | 10.052 |

## 8. 결과 해석

모든 방법의 ARI가 1.000이다.

```text
즉, 이 setting에서는 clustering 성능이 핵심 차이가 아니다.
핵심 차이는 active coordinate를 얼마나 정확히 선택하는가이다.
```

또한 모든 방법의 TPR이 1.000이다.

```text
즉, 모든 방법이 true active coordinate는 모두 찾았다.
차이는 inactive coordinate를 얼마나 추가로 잘못 선택했는가에서 나타났다.
```

Rossi 방법은 true active q = 10보다 넓은 support를 선택했다.

```text
Rossi selected q = 21.933
TPR = 1.000
FPR = 0.133
F1 = 0.635
```

Rossi + refit에서도 support 지표는 변하지 않았다.

```text
selected q = 21.933
TPR = 1.000
F1 = 0.635
```

분리 패널티도 support를 넓게 선택했다.

```text
분리 패널티 selected q = 24.300
TPR = 1.000
FPR = 0.159
F1 = 0.590
```

분리 패널티 + refit에서도 support 지표는 그대로였다.

```text
selected q = 24.300
TPR = 1.000
F1 = 0.590
```

따라서 Rossi와 분리 패널티의 한계는 refit 부재가 아니라 variable selection target에 있다.

반면 에타 패널티는 true q = 10에 가까운 support를 선택했다.

```text
에타 패널티 selected q = 11.800
TPR = 1.000
FPR = 0.020
F1 = 0.920
```

즉, 에타 패널티는 true active coordinate를 놓치지 않으면서 불필요한 inactive coordinate 선택을 크게 줄였다.

다만 refit 전에는 penalty 때문에 `eta` contrast와 kappa ratio가 shrink되었다.

```text
||eta_2 - eta_1|| = 174.767
kappa ratio = 8.428
```

에타 패널티 + refit은 support는 유지하면서 추정량을 true value에 가깝게 회복했다.

```text
||eta_2 - eta_1|| = 180.828
kappa ratio = 10.052
```

## 9. Eta screening + refit 추가 확인

에타 패널티 EM과 별도로, 계산적으로 단순한 eta screening + refit 절차도 확인했다.

```text
1. dense vMF mixture 적합
2. |eta_2j - eta_1j| 점수 계산
3. support path를 만들고 BIC로 support size 선택
4. 선택된 support에서 penalty 없이 refit
```

한계 setting에서 eta screening + refit은 다음 결과를 보였다.

```text
ARI = 1.000
selected q = 10.000
TPR = 1.000
FPR = 0.000
Precision = 1.000
F1 = 1.000
||eta_2 - eta_1|| = 180.766
kappa ratio = 10.063
```

이는 eta contrast score 자체가 concentration-driven active coordinate를 잘 포착한다는 진단적 근거로 볼 수 있다.

다만 최종 제안 방법은 penalized likelihood 안에서 eta contrast를 직접 다루는 에타 패널티 EM으로 정리하고, eta screening + refit은 계산적으로 안정적인 practical version 또는 비교 방법으로 제시할 수 있다.

## 10. Robustness pilot

한 가지 극단 setting만이 아니라 다음 grid에서도 pilot simulation을 수행했다.

```text
kappa ratio = {2, 5, 10}
mu_cos = {1.00, 0.99, 0.95}
replication = 10
```

### kappa ratio = 2

Signal이 약해서 모든 방법의 ARI가 낮았다. 그래도 eta 기반 방법이 support recovery에서는 상대적으로 안정적이었다.

### kappa ratio = 5

모든 방법의 ARI가 약 0.99로 높았다. 이때 eta penalty/refit과 eta screening/refit의 support F1이 가장 높았다.

### kappa ratio = 10

모든 방법의 ARI가 1.00이었다. 하지만 Rossi와 분리 패널티는 selected q를 약 22-25로 넓게 선택했다. 반면 eta penalty/refit은 selected q가 약 11-12로 true q = 10에 가까웠다.

따라서 concentration 차이가 충분하면 기존 방법도 clustering은 잘하지만, 해석 가능한 active coordinate recovery에서는 eta 기반 방법이 더 안정적이었다.

## 11. 현재까지의 결론

```text
1. Rossi & Barbaro (2022) 방법은 clustering은 잘하지만,
   mu_k 중심의 sparsity라 concentration-driven variable effect를 직접 설명하기 어렵다.

2. Rossi + refit을 해도 support recovery는 개선되지 않았다.
   이는 refit이 support를 새로 선택하지 않기 때문이다.

3. 교수님 제안인 분리 패널티 EM도 구현하고 refit까지 확인했다.
   그러나 support recovery 지표는 개선되지 않았다.

4. 분리 패널티의 한계는 refit 부재가 아니라,
   scalar kappa와 mu를 분리해서 penalize하는 구조가
   coordinate-level concentration effect를 직접 선택하지 못한다는 점이다.

5. 반면 eta_k = kappa_k mu_k를 기준으로 penalty를 두면,
   posterior decision에 실제로 들어가는 coordinate effect를 직접 선택할 수 있다.

6. 시뮬레이션에서도 에타 패널티가 TPR을 유지하면서 FPR을 크게 낮췄고,
   refit을 붙이면 eta norm과 kappa ratio도 true value에 가까워졌다.
```

## 12. 교수님께 보고할 핵심 문장

```text
기존 Rossi 방법과 분리 패널티 방법은 clustering은 잘하고,
true active coordinate도 놓치지는 않았습니다.

하지만 concentration-driven setting에서 inactive coordinate를 함께 넓게 선택하는 문제가 남았습니다.

두 방법 모두 refit을 붙여도 support recovery는 개선되지 않았기 때문에,
문제는 refit 부재가 아니라 variable selection target에 있다고 볼 수 있습니다.

반면 eta_k = kappa_k mu_k는 posterior decision에 직접 들어가는 자연모수이므로,
eta contrast에 penalty를 두는 방식이 concentration-driven variable effect를 더 직접적으로 선택합니다.

현재 시뮬레이션에서는 에타 패널티 + refit이 TPR을 유지하면서 FPR을 낮추고,
최종 추정량 해석 측면에서도 가장 자연스러운 방향으로 보입니다.
```

## 13. 다음 단계

다음 단계는 pilot 수준의 robustness simulation을 final simulation으로 확장하는 것이다.

추천 설정은 다음과 같다.

```text
replication = 30 or 50
kappa ratio = {2, 5, 10}
mu_cos = {1.00, 0.99, 0.95}
```

최종 결과에서는 다음 두 표를 중심으로 정리한다.

```text
1. ARI 비교
2. selected q, TPR, FPR, Precision, F1 비교
```

추가로 실제 데이터 예제를 붙이면, eta 기반 support가 단순한 시뮬레이션 결과가 아니라 실제 directional clustering 해석에도 도움이 된다는 점을 보여줄 수 있다.

## 14. 관련 파일

```text
rossi_barbaro_2022_reproduction.r
rossi_refit_limit_run.r
separate_penalty_vmf_run.r
eta_penalty_vmf_run.r
robustness_kappa_mu_grid_run.r

results/rossi_refit_limit_260604/rossi_refit_limit_summary.csv
results/separate_penalty_refit_260604/separate_penalty_refit_summary.csv
results/eta_penalty_refit_260604/eta_penalty_refit_summary.csv
results/robustness_pilot_fair_260604/robustness_pilot_fair_summary.csv
```
