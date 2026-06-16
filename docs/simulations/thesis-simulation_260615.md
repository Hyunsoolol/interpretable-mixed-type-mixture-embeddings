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

주의: Eta fitting은 현재 exact EM이 아니라 proximal EM-type update로 구현되어 있다. 즉 unpenalized eta M-step 후 centered eta contrast shrinkage를 적용한다. Line-search safeguard 버전의 objective trace smoke test는 통과했지만, 이것은 exact penalized EM 이론을 의미하지 않으며 weak setting의 path/tuning instability는 별도 한계로 남아 있다.

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

단, 아래 2026-06-16 추가 진단 이후에는 이 weak setting을 본문 주력 성공 사례로 그대로 쓰는 것은 위험하다. 위 표는 초기 path/BIC 결과에서는 설득력 있어 보였지만, line-search safeguard, zero-support 처리, path 후보 저장 후 재검증에서 Eta BIC가 null support 또는 dense support로 불안정하게 이동하는 문제가 확인되었다.

### 4.1. 2026-06-16 weak path construction 진단

weak setting에서 Eta BIC가 null support를 자주 선택하거나, positive-support sensitivity가 dense support로 튀는 원인을 확인하기 위해 Eta path 후보 전체를 저장하고 여러 refinement를 비교했다. 이 비교는 공식 알고리즘을 바꾸기 위한 결과가 아니라, path construction failure mode를 진단하기 위한 결과다.

| path construction | scope | near22 후보율 | BIC null 선택률 | positive-support dense 선택률 | 비고 |
|:---|:---|---:|---:|---:|:---|
| no refinement | weak100 | 0.23 | 0.73 | 0.72 | 기본 path가 true union q=22 근처 후보를 충분히 만들지 못함 |
| oracle target-refine | weak100 | 0.89 | 0.14 | 0.09 | q=17-27 true-support 주변 정보를 사용하므로 공식 알고리즘 불가 |
| adaptive v1 support-jump | weak100 | 0.73 | 0.24 | 0.24 | target-free지만 oracle에는 못 미침 |
| adaptive v2 priority midpoint | smoke10 | 0.50 | 0.50 | 0.50 | priority refinement가 충분한 후보 다양성을 만들지 못함 |
| adaptive v2.1 duplicate endpoint | smoke10 | 0.50 | 0.40 | 0.40 | evaluated 254, saved unique support 6, duplicate endpoint 248 |
| adaptive v3 multi-point | smoke10 | 0.50 | 0.60 | 0.40 | evaluated 990, saved unique support 6, duplicate endpoint 984 |

핵심 결론은 weak setting의 실패가 단순히 lambda grid가 성기기 때문이라고 보기 어렵다는 것이다. v3에서는 log-lambda interval 안에서 multi-point refinement를 수행하여 990개 후보를 평가했지만, saved unique support는 6개에 그쳤고 near22 후보율은 개선되지 않았으며 BIC null 선택률은 v2.1보다 악화되었다.

따라서 weak setting에서 Eta BIC의 불안정성은 "path grid density 부족"보다는 현재 proximal EM-type update가 만드는 support plateau와 BIC tuning instability 문제로 보는 것이 더 타당하다. 논문 본문에서는 strong/common+specific setting을 중심 결과로 두고, weak setting은 appendix에서 limitation과 diagnostic study로 제시하는 편이 안전하다.

추가로 target-free stability selection smoke diagnostic도 수행했다. 설정은 B=20, subsample 비율 0.7, threshold 0.6, weak setting 5회 반복이었다. ERROR row는 없었고 `Eta centered stability`와 `Eta centered stability + refit` row가 모두 저장되었지만, stability support는 `[0, 0, 22, 0, 0]`으로 5회 중 4회가 zero support였다. `Eta centered stability + refit`은 valid_reps가 1/5이고 zero_support_refit_reps가 4/5였다. 따라서 현재 threshold 0.6 stability selection은 zero-support 문제를 해결했다는 증거가 아니라, threshold/subsample/B sensitivity가 필요한 다음 진단 후보로만 해석한다.

Threshold sweep smoke에서는 threshold 0.2, 0.3, 0.4, 0.5, 0.6을 비교했다. 모든 threshold가 ERROR 없이 실행됐지만 zero-support stability selection은 모두 4/5였다. Nonzero repetition의 selected q만 25, 23, 22, 22, 22로 변했고 dense-support failure는 나타나지 않았다. 따라서 threshold를 낮추는 것만으로는 weak setting의 zero-support instability를 해결하지 못한다.

Subsample-level stability diagnostic을 추가로 저장한 결과, zero-support repetitions 1, 2, 4, 5에서는 각 20개 subsample이 모두 selected q=0을 선택했다. 모든 subsample fit은 성공했으므로 frequency=0의 직접 원인은 fit failure나 support가 서로 흩어져 frequency가 분산되는 현상이 아니라, subsample별 BIC가 반복적으로 null support를 고르는 현상으로 보인다. Rep 3에서는 20개 subsample 중 12개가 q=17-27 근처를 선택했고 full stability support도 q=22였다.

이 결과는 full-data Eta path에서도 zero-support repetitions가 q=17-27 후보를 만들지 못하고 q=0 또는 dense q>=75 후보만 갖는다는 이전 진단과 일치한다. 따라서 weak setting의 stability failure는 threshold만 낮추면 해결되는 문제가 아니라 path/BIC selection instability의 연장선으로 보는 것이 타당하다. 다음 보강은 stability threshold 조정보다 alternative IC 또는 Eta proximal/MM/coordinate update 개선 쪽이 우선이다.

IC/df 관점에서 추가 확인하면, 현재 subsample diagnostic은 각 subsample의 selected candidate만 저장하므로 subsample 내부에서 alternative IC/df가 어떤 lambda 후보를 골랐을지는 직접 재계산할 수 없다. Zero-support repetitions의 selected candidate df는 모두 null-support baseline df=103이었고, full-data path도 q=0 또는 dense 후보로만 구성되었다. 따라서 weak instability를 BIC penalty 단독 문제로 단정하기보다, 중간 support 후보가 부족한 path/BIC instability로 해석하는 편이 안전하다. 다음 IC 진단은 subsample별 전체 lambda path 저장이 필요하다.

Subsample full-path diagnostic smoke 3회에서는 alternative df를 직접 재선택했다. `df_current`, `df_no_d`, `df_support_only`, `df_half_d`는 모두 동일한 선택을 냈다. 이들은 selected q에 대한 penalty slope가 `(K-1)m`으로 같고 d 항 차이는 후보 간 상수이기 때문이다. `df_direction_only = (K-1)+m`만 zero rate를 0.783에서 0.667로 낮췄지만 dense75 rate가 0.117로 생기고 FPR이 증가했다. 따라서 weak setting의 문제는 d 상수항 penalty만의 문제가 아니라, support path 후보와 IC slope가 함께 만드는 instability로 해석해야 한다.

추가 slope sensitivity에서는 `df = c0 + gamma * selected_q`를 확인했다. c0는 후보 간 상수항이므로 어떤 값으로 두어도 선택을 바꾸지 않았다. Gamma를 3에서 1-1.5로 낮추면 zero rate는 0.783에서 0.667로 줄지만 near22 rate는 0.20에 머물고 dense75 rate가 0.117로 증가했다. Gamma=0.5에서는 dense75 rate가 0.333, FPR이 0.332로 커졌다. 즉 usable gamma 구간은 아직 명확하지 않고, weak setting 보강은 단순 df 상수항 수정이 아니라 slope/selection rule 또는 path update 개선 문제다.

## 5. Rossi 대비 Eta+refit 요약

| setting | ARI Rossi | ARI Eta+refit | ARI gain | q reduction | FPR reduction | F1 gain |
|---|---:|---:|---:|---:|---:|---:|
| strong kappa | 0.685 | 0.686 | 0.001 | 0.742 | 0.912 | 0.564 |
| weak kappa | 0.570 | 0.575 | 0.005 | 0.745 | 0.898 | 0.578 |

Eta+refit의 핵심 장점은 ARI의 큰 증가라기보다, ARI를 유지하거나 소폭 개선하면서 selected q와 FPR을 크게 줄이는 데 있다. 다만 2026-06-16 path 후보 진단 이후에는 이 문장을 weak setting 전체에 무조건 적용하면 안 된다. Weak setting에서는 Eta path/BIC 선택이 support plateau와 tuning instability에 민감하므로, 본문 주장은 strong 및 안정적으로 재현되는 setting 중심으로 제한하고 weak 결과는 sensitivity/appendix로 낮추는 것이 적절하다.

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
- 단, weak concentration setting의 Eta BIC 결과는 path construction과 tuning에 불안정하다. Adaptive refinement v1-v3 진단 결과를 보면 단순 grid refinement로 해결되는 문제가 아니므로, weak setting은 본문 핵심 성공 사례가 아니라 appendix의 failure-mode/sensitivity 결과로 두는 것이 안전하다.
