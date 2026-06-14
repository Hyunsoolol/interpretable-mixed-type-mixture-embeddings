# Thesis Simulation 260615

## 1. 목적

Rossi & Barbaro (2022)의 sparse vMF 방법과 제안하는 eta contrast penalty 방법을 비교하기 위해, 평균 방향에 공통 변수와 군집별 특정 변수가 함께 존재하는 시뮬레이션 환경을 구성하였다.

핵심 관심은 단순히 ARI만 높이는 것이 아니라, 군집 성능을 유지하면서 군집 구분에 필요한 변수만 선택하는지 확인하는 것이다.

## 2. 공통 시뮬레이션 설정

| 항목 | 값 |
|---|---:|
| 군집 수 K | 4 |
| 표본 수 n | 1000 |
| 변수 수 d | 100 |
| 반복 수 | 100 |
| random start | 10 |
| tuning | path tuning + BIC |
| common variables | 6 |
| component-specific variables | component마다 4개 |
| true union q | 22 |
| specific weight w | 0.50 |
| 평균 방향 평균 cosine | 0.857 |

변수 생성 방식은 다음과 같다.

```text
common variables:
v_kj = 1.0 for all components

component-specific variables:
v_kj = 0.5 only for component k

noise variables:
v_kj = 0

mu_k = v_k / ||v_k||
```

비교 방법은 다음 6가지이다.

| 번호 | 방법 |
|---:|---|
| 1 | Rossi sparse vMF |
| 2 | Rossi sparse vMF + refit |
| 3 | Separate penalty EM |
| 4 | Separate penalty EM + refit |
| 5 | Eta contrast penalty |
| 6 | Eta contrast penalty + refit |

평가 지표는 ARI, NMI, purity, selected q, TPR, FPR, Precision, F1, MSE_mu, MSE_kappa, MSE_centered_eta를 사용하였다.

## 3. Main setting 1: strong concentration difference

설정은 다음과 같다.

| 항목 | 값 |
|---|---:|
| kappa | (30, 45, 65, 90) |
| kappa ratio | 3.00 |
| 반복 수 | 100 |

| method | ARI | NMI | purity | selected q | TPR | FPR | Precision | F1 | MSE_mu | MSE_kappa | MSE_eta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Rossi | 0.685 | 0.657 | 0.865 | 95.98 | 1.000 | 0.948 | 0.230 | 0.374 | 0.00012 | 2.653 | 0.254 |
| Rossi + refit | 0.655 | 0.633 | 0.848 | 95.98 | 1.000 | 0.948 | 0.230 | 0.374 | 0.00032 | 3.305 | 0.585 |
| Separate penalty | 0.686 | 0.661 | 0.865 | 70.83 | 1.000 | 0.626 | 0.328 | 0.488 | 0.00008 | 9.817 | 0.152 |
| Separate penalty + refit | 0.665 | 0.641 | 0.854 | 70.83 | 1.000 | 0.626 | 0.328 | 0.488 | 0.00025 | 2.613 | 0.477 |
| Eta contrast | 0.624 | 0.637 | 0.819 | 24.74 | 0.994 | 0.037 | 0.890 | 0.938 | 0.00029 | 14.576 | 0.425 |
| Eta contrast + refit | 0.686 | 0.660 | 0.865 | 24.74 | 0.994 | 0.037 | 0.890 | 0.938 | 0.00010 | 1.905 | 0.185 |

### 해석

Strong concentration setting에서는 Rossi, Separate, Eta+refit의 ARI가 거의 비슷하다. 그러나 변수 선택에서는 차이가 매우 크다.

Rossi는 평균적으로 약 96개 변수를 선택하여 거의 dense하게 작동한다. Separate penalty도 약 71개 변수를 선택하여 false positive가 여전히 크다. 반면 Eta+refit은 약 25개 변수만 선택하면서 true union q = 22에 가장 가깝고, FPR도 0.037로 낮다.

따라서 이 setting은 Eta penalty의 장점을 "ARI를 유지하면서 해석 가능한 sparse contrast를 회복한다"는 방향으로 보여주기에 적합해보임.

## 4. Main setting 2: weak concentration difference

설정은 다음과 같다.

| 항목 | 값 |
|---|---:|
| kappa | (40, 50, 60, 70) |
| kappa ratio | 1.75 |
| 반복 수 | 100 |

| method | ARI | NMI | purity | selected q | TPR | FPR | Precision | F1 | MSE_mu | MSE_kappa | MSE_eta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Rossi | 0.570 | 0.523 | 0.814 | 94.04 | 1.000 | 0.924 | 0.235 | 0.380 | 0.00011 | 2.536 | 0.256 |
| Rossi + refit | 0.529 | 0.489 | 0.792 | 94.04 | 1.000 | 0.924 | 0.235 | 0.380 | 0.00028 | 3.433 | 0.646 |
| Separate penalty | 0.572 | 0.529 | 0.815 | 56.79 | 1.000 | 0.446 | 0.456 | 0.602 | 0.00006 | 12.282 | 0.150 |
| Separate penalty + refit | 0.553 | 0.509 | 0.805 | 56.79 | 1.000 | 0.446 | 0.456 | 0.602 | 0.00018 | 2.463 | 0.431 |
| Eta contrast | 0.564 | 0.525 | 0.810 | 23.98 | 1.000 | 0.025 | 0.921 | 0.958 | 0.00018 | 7.333 | 0.355 |
| Eta contrast + refit | 0.575 | 0.529 | 0.817 | 23.98 | 1.000 | 0.025 | 0.921 | 0.958 | 0.00007 | 1.801 | 0.181 |

### 해석

Weak concentration setting은 concentration 차이가 강하지 않아 더 어려운 환경이다. 이 경우에도 Eta+refit은 Rossi보다 ARI, NMI, purity가 모두 약간 높고, 변수 선택 성능은 크게 개선된다.

Rossi는 selected q가 94.04, FPR이 0.924로 대부분의 noise 변수를 선택한다. Eta+refit은 selected q가 23.98로 true union q = 22에 가깝고, FPR은 0.025로 크게 낮다. Precision과 F1도 각각 0.921, 0.958로 가장 높다.

이 setting은 본문에서 가장 설득력 있는 결과로 볼 수 있다. concentration 차이가 약해져도 Eta contrast penalty가 군집 성능을 유지하면서 불필요한 변수를 제거하기 때문이다.

## 5. Rossi 대비 Eta+refit 요약

| setting | ARI Rossi | ARI Eta+refit | ARI gain | q reduction | FPR reduction | F1 gain |
|---|---:|---:|---:|---:|---:|---:|
| strong kappa | 0.685 | 0.686 | 0.001 | 0.742 | 0.912 | 0.564 |
| weak kappa | 0.570 | 0.575 | 0.005 | 0.745 | 0.898 | 0.578 |

Eta+refit의 핵심 장점은 ARI의 큰 증가라기보다, ARI를 유지하거나 소폭 개선하면서 selected q와 FPR을 크게 줄이는 데 있다.

## 6. Pilot robustness 결과

### 6.1 Specific weight 변화

| setting | Rossi ARI | Eta+refit ARI | Rossi q | Eta+refit q | Rossi FPR | Eta+refit FPR |
|---|---:|---:|---:|---:|---:|---:|
| w = 0.25, strong kappa | 0.358 | 0.374 | 97.90 | 33.00 | 0.973 | 0.218 |
| w = 0.50, strong kappa | 0.650 | 0.662 | 99.35 | 27.00 | 0.992 | 0.079 |
| w = 0.75, strong kappa | 0.837 | 0.837 | 98.35 | 23.80 | 0.979 | 0.023 |

w가 커질수록 군집 구분이 쉬워져 Rossi의 ARI도 높아진다. 그러나 Rossi는 모든 경우에서 거의 모든 변수를 선택한다. Eta+refit은 w 변화에도 selected q와 FPR을 낮게 유지한다.

### 6.2 High-dimensional robustness pilot

| setting | Rossi ARI | Eta+refit ARI | Rossi q | Eta+refit q | Rossi FPR | Eta+refit FPR |
|---|---:|---:|---:|---:|---:|---:|
| d = 200, strong kappa | 0.366 | 0.428 | 199.10 | 28.60 | 0.995 | 0.070 |
| d = 400, strong kappa | 0.126 | 0.192 | 396.00 | 17.67 | 0.989 | 0.021 |

d = 200에서는 Eta+refit이 ARI와 변수 선택 모두에서 Rossi보다 좋다. d = 400에서는 Eta가 상대적으로 낫지만 전체 ARI가 낮고 일부 replication에서 실패가 있어 본문 주력 결과보다는 부록 또는 한계로 두는 것이 적절해보임

## 7. 현재 결론

- Rossi sparse vMF는 ARI는 괜찮지만 공통 변수와 군집별 특정 변수가 섞인 환경에서 noise 변수를 많이 포함한다.
- Separate penalty는 Rossi보다 변수 선택이 나아지지만 여전히 selected q와 FPR이 크다.
- Eta contrast + refit은 군집 성능을 유지하면서 true union q에 가까운 변수를 선택한다.
