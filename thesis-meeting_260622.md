# Thesis Meeting 260622

업데이트: 2026-06-11
상세 시뮬레이션 결과는 `thesis-simulation_260611.md`, 구현 및 추정 방법은 `thesis-implementation_260622.md`에 분리했다.

## 1. 핵심 메시지

Rossi & Barbaro (2022)의 sparse vMF mixture는 component direction $\mu_k$에 sparsity penalty를 둔다. 그러나 vMF mixture의 posterior classification에는 $\mu_k$ 자체보다 natural parameter

$$\eta_k = \kappa_k \mu_k$$

가 직접 들어간다. 따라서 평균 방향 차이가 작고 집중도 차이가 중요한 setting에서는 $\eta_k$ 또는 component 간 $\eta_k$ contrast를 sparse하게 만드는 방식이 더 자연스럽다.

이번 연구미팅에서 확인할 내용은 다음이다.

| 구분 | 확인 내용 |
|:---|:---|
| 기존 방법 재현 | Rossi & Barbaro (2022) setting에서 sparse vMF가 정상적으로 재현되는지 |
| 한계 setting | 평균 방향 차이가 작고 집중도 차이가 중요한 경우 Rossi 방식이 노이즈 변수를 과도하게 선택하는지 |
| 제안 방향 | $\eta_k=\kappa_k\mu_k$ penalty가 변수 선택과 해석 가능성을 개선하는지 |
| refit | 선택된 support에서 penalty 없이 재추정하면 모수 추정 bias가 줄어드는지 |
| 고차원 | 차원이 커질 때 BIC 기준이 충분한지 |

## 2. 방법 비교

비교한 방법은 6가지다.

| 방법 | penalty 기준 | 최종 추정 |
|:---|:---|:---|
| Rossi | $\mu_k$ L1 penalty | penalized EM |
| Rossi + refit | Rossi와 동일 | 선택 support 고정 후 unpenalized EM |
| 분리 패널티 | $\mu_k$ L1 penalty + $\kappa_k$ penalty | penalized EM |
| 분리 패널티 + refit | 분리 패널티와 동일 | 선택 support 고정 후 unpenalized EM |
| 에타 패널티 | $\eta_k$ contrast 또는 centered $\eta_k$ penalty | proximal EM-type update |
| 에타 패널티 + refit | 에타 패널티와 동일 | proximal update로 선택한 support 고정 후 unpenalized EM |

공식 비교에서는 각 방법의 path 기반 후보를 만들고 BIC가 최소인 지점을 선택했다. EBIC는 고차원 setting에서 보조 지표로만 확인했다.

## 3. 주요 수식

vMF mixture의 posterior responsibility는 다음과 같다.

$$\tau_{ik} = \frac{\alpha_k C_d(\kappa_k)\exp(\eta_k^T x_i)}{\sum_{\ell=1}^K \alpha_\ell C_d(\kappa_\ell)\exp(\eta_\ell^T x_i)}.$$

두 component의 posterior decision boundary는 다음처럼 쓸 수 있다.

$$\log\frac{\tau_{i2}}{\tau_{i1}} = \mathrm{const} + (\eta_2-\eta_1)^T x_i.$$

따라서 coordinate $j$가 군집 구분에 기여하는지는 $\mu_{kj}$보다 $\eta_{kj}$의 component 간 차이를 보는 것이 더 직접적이다.

K=4에서는 한 쌍의 contrast만으로 전체 component 차이를 표현하기 어렵기 때문에 coordinate별 centered eta를 사용한다.

$$\bar{\eta}_j = K^{-1}\sum_{k=1}^K \eta_{kj}, \qquad c_{kj} = \eta_{kj}-\bar{\eta}_j.$$

에타 패널티는 다음 형태로 구현했다.

$$Q_{\eta} = \ell(\Theta) - \lambda_\eta \sum_{j=1}^d \|c_{\cdot j}\|_2.$$

현재 구현은 exact M-step이 아니라 unpenalized eta M-step 후 centered eta contrast shrinkage를 적용하는 proximal EM-type update다. 따라서 monotone EM 보장은 아직 없고, 논문 버전에서는 objective trace 확인과 line search/MM 보강을 검토한다.

Refit은 penalized fit에서 선택된 support를 고정하고 penalty 없이 vMF mixture를 다시 추정하는 단계다. Support는 바꾸지 않고, $\alpha_k$, $\mu_k$, $\kappa_k$만 다시 추정한다.

## 4. Rossi 2022 재현

먼저 Rossi & Barbaro (2022)의 sparse vMF가 원 논문 setting에서 정상적으로 작동하는지 확인했다.

| 항목 | 논문 기준 | 구현 결과 |
|:---|:---|---:|
| ARI | 약 0.80-0.90 | 0.871 |
| 정성적 패턴 | Figure와 유사 | 재현 가능 |

이 재현에서는 Rossi 방법이 잘 작동한다. 따라서 이후 결과는 “Rossi 구현이 틀렸다”가 아니라, “Rossi 방법이 유리한 sparse direction setting과 concentration-dominant setting이 다르다”는 점을 보여주는 비교로 해석한다.

## 5. Concentration-dominant setting

평균 방향이 같거나 매우 유사하고, 집중도 차이가 군집 차이를 만드는 setting을 구성했다.

### 5.1. K=4 stress setting

```text
K = 4
n = 1000
d = 100
반복수 = 20
random start = 5
공통 active 변수 = 10개
군집별 특정 변수 = 없음
kappa = (20, 35, 60, 100)
```

| Method | ARI | Selected q | TPR | FPR | Precision | F1 |
|:---|---:|---:|---:|---:|---:|---:|
| Rossi | 0.432 | 85.650 | 1.000 | 0.841 | 0.128 | 0.223 |
| 분리 패널티 | 0.429 | 92.550 | 1.000 | 0.917 | 0.109 | 0.196 |
| 에타 패널티 + refit | 0.440 | 27.950 | 1.000 | 0.199 | 0.553 | 0.661 |

이 setting에서는 모든 방법이 true active variable은 찾는다. 차이는 노이즈 변수 선택이다. Rossi와 분리 패널티는 대부분의 노이즈 변수를 함께 선택하지만, 에타 패널티 + refit은 ARI를 유지하면서 FPR을 크게 낮춘다.

### 5.2. Controlled concentration-dominant setting

Stress setting보다 완화된 상황으로, 평균 방향을 완전히 같게 두지 않고 pairwise cosine을 0.95로 설정했다. 변수 구조는 stress setting과 동일하게 고정했다.

```text
K = 4
n = 1000
d = 100
반복수 = 20
random start = 5
공통 active 변수 = 10개
kappa = (25, 40, 65, 100)
```

| Method | ARI | True q | Selected q | TPR | FPR | Precision | F1 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Rossi | 0.513 | 10.000 | 98.500 | 1.000 | 0.983 | 0.102 | 0.184 |
| 분리 패널티 | 0.525 | 10.000 | 95.600 | 1.000 | 0.951 | 0.105 | 0.190 |
| 에타 패널티 + refit | 0.523 | 10.000 | 28.800 | 1.000 | 0.209 | 0.443 | 0.586 |

ARI는 세 방법이 비슷하다. 그러나 ARI는 군집 라벨만 평가하므로 변수 선택의 질을 보여주지 않는다. Rossi와 분리 패널티는 거의 모든 변수를 선택하기 때문에 해석 가능한 sparse structure를 제공하지 못한다. 에타 패널티 + refit은 군집화 성능을 유지하면서 selected q와 FPR을 줄인다.

## 6. 공통 변수 + 군집별 특정 변수 setting

현실적인 sparse structure를 보기 위해 공통 변수와 component-specific 변수를 함께 둔 setting을 구성했다.

```text
K = 4
n = 1000
d = 100
반복수 = 100
random start = 10
공통 변수 = 6개
군집별 특정 변수 = component마다 4개, 총 16개
true union q = 22
w = 0.50
kappa = (30, 45, 65, 90)
```

| Method | ARI | True q | Selected q | TPR | FPR | Precision | F1 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Rossi | 0.680 | 22.000 | 98.520 | 1.000 | 0.981 | 0.223 | 0.365 |
| 분리 패널티 | 0.684 | 22.000 | 86.460 | 1.000 | 0.826 | 0.258 | 0.409 |
| 에타 패널티 + refit | 0.686 | 22.000 | 24.750 | 0.994 | 0.037 | 0.890 | 0.937 |

Rossi는 ARI가 낮지 않지만 거의 모든 변수를 선택한다. 즉 군집 라벨은 어느 정도 맞추지만 sparse variable selection에는 실패한다. 에타 패널티 + refit은 true union q=22에 가까운 24.750개를 선택하고, F1을 0.937까지 높인다.

변수 유형별 선택률은 다음과 같다.

| Method | Common selection | Specific selection | Noise selection |
|:---|---:|---:|---:|
| Rossi | 1.000 | 1.000 | 0.981 |
| 분리 패널티 | 1.000 | 1.000 | 0.826 |
| 에타 패널티 + refit | 1.000 | 0.992 | 0.037 |

에타 패널티는 공통 변수뿐 아니라 군집별 특정 변수도 거의 유지하면서 노이즈 변수를 제거한다.

## 7. Robustness 확인

### 7.1. 군집별 특정 변수 weight 변화

군집별 특정 변수의 신호 세기 $w$를 변화시켰다.

| w | mean cosine | Method | ARI | Selected q | FPR | Precision | F1 | Specific selection |
|---:|---:|:---|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.960 | 에타 패널티 + refit | 0.399 | 32.130 | 0.183 | 0.621 | 0.682 | 0.743 |
| 0.35 | 0.924 | 에타 패널티 + refit | 0.505 | 29.810 | 0.123 | 0.723 | 0.794 | 0.888 |
| 0.50 | 0.857 | 에타 패널티 + refit | 0.686 | 24.750 | 0.037 | 0.890 | 0.937 | 0.992 |

$w$가 작아질수록 평균 방향이 더 비슷해져 군집별 특정 변수를 찾기 어렵다. 그래도 에타 패널티는 노이즈 선택률을 낮게 유지한다.

### 7.2. 약한 집중도 차이

집중도 차이를 약하게 하기 위해 $\kappa=(40,50,60,70)$로 설정했다.

| Method | ARI | True q | Selected q | TPR | FPR | Precision | F1 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Rossi | 0.563 | 22.000 | 97.580 | 1.000 | 0.969 | 0.226 | 0.368 |
| 분리 패널티 | 0.567 | 22.000 | 78.150 | 1.000 | 0.720 | 0.294 | 0.450 |
| 에타 패널티 + refit | 0.575 | 22.000 | 23.980 | 1.000 | 0.025 | 0.921 | 0.958 |

집중도 차이가 약해져도 에타 패널티 + refit은 true union q=22에 가까운 변수 수를 선택하고, FPR을 0.025로 유지한다.

## 8. 고차원 결과

공통 변수 + 군집별 특정 변수 구조를 유지하고 차원 $d$를 증가시켰다.

### 8.1. 차원에 따라 kappa를 조정한 setting

| d | Method | ARI | Selected q | TPR | FPR | Precision | F1 |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 100 | 에타 패널티 + refit | 0.658 | 25.700 | 0.945 | 0.063 | 0.842 | 0.884 |
| 200 | 에타 패널티 + refit | 0.872 | 33.200 | 1.000 | 0.063 | 0.865 | 0.911 |
| 500 | 에타 패널티 + refit | 0.850 | 269.550 | 1.000 | 0.518 | 0.327 | 0.414 |

$d=200$까지는 에타 패널티 + refit이 안정적이다. 그러나 $d=500$에서는 BIC 기준이 느슨해져 노이즈 변수를 많이 선택한다.

### 8.2. kappa를 고정한 stress setting

모든 차원에서 $\kappa=(30,45,65,90)$로 고정했다.

| d | Method | 성공 reps | ARI | Selected q | TPR | FPR | Precision | F1 |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| 100 | 에타 패널티 + refit | 20 | 0.658 | 25.700 | 0.945 | 0.063 | 0.842 | 0.884 |
| 200 | 에타 패널티 + refit | 19 | 0.422 | 34.105 | 0.773 | 0.096 | 0.559 | 0.627 |
| 400 | 에타 패널티 + refit | 7 | 0.234 | 13.143 | 0.494 | 0.006 | 0.847 | 0.619 |

$\kappa$를 고정하고 차원만 늘리면 concentration signal이 상대적으로 약해진다. $d=400$에서는 active coordinate를 선택하지 못한 반복이 많으므로 stress setting으로 해석해야 한다.

## 9. 현재 결론

| 내용 | 결론 |
|:---|:---|
| 2022 논문 재현 | Rossi 방법은 원 논문 setting에서 정상적으로 재현된다 |
| Rossi의 한계 | concentration-dominant setting에서는 노이즈 변수를 과도하게 선택한다 |
| 에타 패널티 장점 | ARI를 유지하면서 selected q, FPR, Precision, F1을 개선한다 |
| refit 역할 | penalty shrinkage를 줄이고 선택된 support 위에서 모수를 재추정한다 |
| 고차원 이슈 | BIC만으로는 약할 수 있어 EBIC 검토가 필요하다 |
