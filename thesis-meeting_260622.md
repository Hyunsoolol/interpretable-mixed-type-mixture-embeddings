# Thesis Meeting 260622

업데이트: 2026-06-11
상세 시뮬레이션 결과는 `thesis-simulation_260611.md`에 분리하여 정리했다. 이 문서는 연구미팅에서 논의할 핵심 내용만 요약한다.

## 1. 연구 질문

Rossi & Barbaro (2022)의 sparse vMF mixture는 component direction $\mu_k$에 sparsity penalty를 둔다. 그러나 vMF mixture에서 posterior classification에 직접 들어가는 항은 $\mu_k$ 자체가 아니라

$$\eta_k = \kappa_k \mu_k$$

이다. 따라서 평균 방향 차이보다 집중도 차이가 군집을 만드는 상황에서는 $\eta_k$에 penalty를 두는 방식이 더 자연스럽다.

핵심 질문은 다음과 같다.

* 평균 방향 차이가 작고 집중도 차이가 중요한 경우, 기존 sparse vMF가 불필요한 변수를 많이 선택하는가?
* $\eta_k=\kappa_k\mu_k$에 직접 penalty를 두면 해석 가능한 변수 선택이 개선되는가?
* 선택된 변수로 refit을 하면 penalty로 인한 모수 추정 bias가 줄어드는가?

## 2. 비교 방법

6가지 방법을 비교했다.

| 구분 | 방법 | 설명 |
|:---|:---|:---|
| 1 | Rossi | Rossi & Barbaro (2022) 방식. $\mu_k$에 L1 penalty |
| 2 | Rossi + refit | Rossi에서 선택된 변수로 penalty 없이 재추정 |
| 3 | 분리 패널티 | $\mu_k$와 $\kappa_k$에 각각 penalty |
| 4 | 분리 패널티 + refit | 분리 패널티에서 선택된 변수로 재추정 |
| 5 | 에타 패널티 | $\eta_k=\kappa_k\mu_k$에 직접 penalty |
| 6 | 에타 패널티 + refit | 에타 패널티에서 선택된 변수로 재추정 |

Tuning은 기본적으로 path 기반 후보를 만들고 BIC가 최소인 지점을 선택했다. EBIC는 고차원 상황에서 보조 지표로만 확인했다.

## 3. 주요 수식

vMF mixture의 posterior probability는 다음 항에 의해 결정된다.

$$\tau_{ik} = \frac{\alpha_k C_d(\kappa_k)\exp(\eta_k^T x_i)}{\sum_{\ell=1}^K \alpha_\ell C_d(\kappa_\ell)\exp(\eta_\ell^T x_i)}.$$

두 component의 posterior decision boundary는 다음처럼 쓸 수 있다.

$$\log\frac{\tau_{i2}}{\tau_{i1}} = \mathrm{const} + (\eta_2-\eta_1)^T x_i.$$

따라서 변수 선택 관점에서는 $\mu_k$의 sparsity보다 $\eta_k$ 또는 $\eta_k$ contrast의 sparsity가 posterior decision에 더 직접적으로 연결된다.

에타 패널티의 기본 형태는 다음과 같이 둔다.

$$Q_{\eta} = \ell(\Theta) - \lambda_\eta \sum_k \|\eta_k-\bar{\eta}\|_1.$$

여기서 $\bar{\eta}$는 component별 $\eta_k$의 중심값이다. 이 방식은 component 간 decision contrast에 기여하지 않는 좌표를 줄이는 방향으로 작동한다.

Refit은 penalized fit에서 선택된 support를 고정하고 penalty 없이 vMF mixture를 다시 추정하는 단계다. 목적은 L1 penalty로 인한 $\kappa$와 $\eta$의 수축 편향을 줄이는 것이다.

## 4. 2022 논문 재현

먼저 Rossi & Barbaro (2022)의 sparse vMF가 논문에서 제시한 artificial simulation 결과와 유사하게 재현되는지 확인했다.

| 항목 | 논문 기준 | 구현 결과 |
|:---|:---|---:|
| ARI | 약 0.80-0.90 | 0.871 |
| 해석 | 논문 Figure와 유사한 수준 | 재현 가능 |

중요한 점은 논문에서의 sparsity가 nonzero 비율이 아니라 zero coordinate 비율로 해석되어야 한다는 것이다. 이 기준으로 보면 Rossi 방법은 원 논문 setting에서는 정상적으로 작동한다.

## 5. Concentration-dominant setting

평균 방향이 같거나 매우 유사하고, 집중도 차이가 군집 차이를 만드는 상황을 구성했다. 이 setting은 제안 방법이 필요한 핵심 상황이다.

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

평균 방향이 같고 집중도만 다른 경우 Rossi와 분리 패널티는 true active variable은 모두 선택하지만, 노이즈 변수도 대부분 선택한다. 에타 패널티는 ARI를 유지하면서 FPR을 크게 줄인다.

### 5.2. Controlled concentration-dominant setting

Stress setting보다 현실적으로 만들기 위해 평균 방향을 완전히 같게 두지는 않고 pairwise cosine을 0.95로 설정했다. 변수 구조는 동일하게 유지했다.

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

ARI만 보면 차이가 크지 않다. 그러나 Rossi와 분리 패널티는 거의 모든 변수를 선택하므로 변수 선택 및 해석 가능성 측면에서는 약하다. 에타 패널티는 군집화 성능을 유지하면서 선택 변수 수와 FPR을 크게 줄인다.

## 6. 공통 변수 + 군집별 특정 변수 setting

더 현실적인 sparse structure를 보기 위해 공통 변수와 component-specific 변수를 함께 둔 setting을 구성했다.

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

이 setting에서 Rossi는 ARI가 낮지 않지만 거의 모든 변수를 선택한다. 즉, 군집 라벨은 어느 정도 맞추지만 sparse variable selection에는 실패한다. 에타 패널티 + refit은 true union q=22에 가까운 24.750개를 선택하고, 노이즈 선택률을 0.037로 낮춘다.

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

$w$가 작아질수록 평균 방향이 더 비슷해져 군집별 특정 변수를 찾기 어려워진다. 그래도 Rossi와 분리 패널티보다 노이즈 선택률은 낮게 유지된다.

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

$\kappa$를 고정하고 차원만 늘리면 concentration signal이 상대적으로 약해진다. $d=400$에서는 active coordinate를 선택하지 못한 반복이 많아져 stress setting으로 해석해야 한다.

## 9. 현재 결론

* Rossi 2022 방법은 원 논문 setting에서는 잘 재현된다.
* 그러나 concentration-dominant setting에서는 Rossi와 분리 패널티가 true active variable을 포함하면서도 노이즈 변수를 과도하게 선택한다.
* 에타 패널티는 posterior decision에 직접 들어가는 $\eta_k=\kappa_k\mu_k$를 sparse하게 만드는 방식이므로, 집중도 차이가 중요한 상황에서 더 해석 가능한 변수 선택을 제공한다.
* Refit은 에타 패널티의 수축 bias를 줄이고, 선택된 support 위에서 모수 추정을 안정화하는 역할을 한다.
* 고차원에서는 BIC만으로는 충분하지 않을 수 있으며, EBIC 또는 더 강한 tuning 기준을 추가로 검토할 필요가 있다.

## 10. 연구미팅에서 논의할 점

1. 제안 방법의 핵심을 $\eta_k$ penalty로 고정해도 되는지
2. 분리 패널티는 baseline으로 유지하고, 주 방법은 에타 패널티 + refit으로 가져갈지
3. 고차원 setting에서 EBIC 또는 다른 tuning 기준을 추가할지
4. real data 분석을 어떤 자료로 진행할지
5. 논문 목표 저널을 Journal of Classification 또는 CSDA 수준으로 둘지
