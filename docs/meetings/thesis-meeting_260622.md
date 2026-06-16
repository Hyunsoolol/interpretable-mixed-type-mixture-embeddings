# Thesis Meeting 260622

업데이트: 2026-06-11
상세 시뮬레이션 결과는 `thesis-simulation_260611.md`, 구현 및 추정 방법은 `thesis-implementation_260622.md`에 분리했다.

## 0. 교수님께 먼저 보여드릴 1페이지 요약

현재 가능한 논문 주장은 다음 한 문장으로 제한하는 것이 안전하다.

> Eta penalty의 주장은 ARI를 크게 올리는 것이 아니라, vMF mixture 안에서 posterior decision parameter인 eta contrast를 sparse하게 만들어 ARI를 유지하면서 해석 가능한 변수 선택을 제공하는 것이다.

현재 본문 후보는 strong/common+specific simulation이다. PBMC lymphoid3는 real-data 보조 사례로 둘 수 있지만, 전체 baseline 최고라고 주장하면 안 된다. Weak concentration setting, high-dimensional stress setting, BBC News text는 부록 또는 한계 사례가 더 적절하다.

공식 알고리즘에는 oracle target-refine과 adaptive v2/v2.1/v3 refinement를 넣지 않는다. Oracle target-refine은 true q 주변 정보를 사용하므로 방법론적으로 방어할 수 없다. Adaptive v2/v2.1/v3는 target-free이지만 unique support 다양성을 충분히 만들지 못했다. 특히 adaptive v3는 990개 후보를 평가했는데 saved unique support가 6개뿐이었다.

### 표 A. 결과 배치 판단

| 결과/setting | 현재 판단 | 이유 | 문서 위치 |
|:---|:---|:---|:---|
| strong common+specific | 본문 핵심 후보 | ARI를 유지하면서 selected q와 FPR을 크게 줄이는 주장이 가장 안정적임 | 본문 simulation |
| weak concentration | 부록/한계 후보 | Eta BIC가 null/dense support로 불안정하며 path refinement로 해결되지 않음 | appendix diagnostic |
| high-dimensional stress | 부록/한계 후보 | signal 약화와 BIC 불안정성이 섞여 있어 본문 성공 사례로 방어하기 어려움 | appendix robustness |
| PBMC lymphoid3 | real-data 보조 사례 | vMF mixture 계열에서는 Eta가 해석 가능한 marker selection을 제공하지만 sparse k-means보다 ARI가 높다고 주장할 수 없음 | real-data section 또는 appendix |
| BBC News text | 부록/보조 사례 | 개선폭과 kappa 구조가 본문 핵심 claim을 지지하기에는 약함 | supplementary |
| adaptive refinement diagnostics | appendix diagnostic | v1은 개선됐지만 v2/v2.1/v3가 unique support plateau 문제를 해결하지 못함 | appendix method diagnostic |

### 표 B. Weak path diagnostic 요약

| path construction | scope | near22 후보율 | BIC null률 | positive dense률 | 판단 |
|:---|:---|---:|---:|---:|:---|
| no refinement | weak100 | 0.23 | 0.73 | 0.72 | 기본 path가 q=22 근처 후보를 충분히 만들지 못함 |
| oracle target-refine | weak100 | 0.89 | 0.14 | 0.09 | 성능은 좋지만 true q 정보를 사용하므로 공식 알고리즘 불가 |
| adaptive v1 | weak100 | 0.73 | 0.24 | 0.24 | target-free 개선은 있으나 oracle 수준은 아님 |
| adaptive v2 | smoke10 | 0.50 | 0.50 | 0.50 | priority midpoint만으로는 불충분 |
| adaptive v2.1 | smoke10 | 0.50 | 0.40 | 0.40 | duplicate endpoint를 써도 support 다양성 부족 |
| adaptive v3 | smoke10 | 0.50 | 0.60 | 0.40 | multi-point 990회 평가에도 saved unique support 6개뿐 |

현재 방어하기 어려운 주장은 다음이다.

1. Eta가 weak concentration setting에서도 항상 안정적으로 true support를 회복한다.
2. Lambda grid를 더 촘촘히 만들면 weak setting 문제가 해결된다.
3. Oracle target-refine 또는 adaptive v2/v3를 공식 path construction으로 쓰면 된다.
4. Eta proximal update를 exact penalized EM으로 표현할 수 있다.

다음 보강 후보는 stability selection, EBIC/RICc/alternative IC, 그리고 MM safeguard 또는 coordinate/proximal update 개선이다. 이 중 stability selection은 tuning instability를 직접 겨냥하지만, 5회 smoke diagnostic에서는 threshold 0.6 기준으로 5회 중 4회가 zero support를 선택했다. 따라서 아직 공식 tuning 후보가 아니라 threshold/subsample sensitivity를 더 봐야 하는 진단 후보로만 둔다.

미팅에서 결정해야 할 것은 다음이다.

1. Weak setting을 appendix로 낮출지 결정한다.
2. Official tuning을 path+BIC로 유지할지 결정한다.
3. Stability selection을 추가 실험으로 할지 결정한다.
4. Eta proximal update를 MM/coordinate 방식으로 보강할지 결정한다.
5. 논문 타깃을 top-tier 방법론에서 중상위/응용방법론으로 조정할지 결정한다.

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

현재 구현은 exact M-step이 아니라 unpenalized eta M-step 후 centered eta contrast shrinkage를 적용하는 proximal EM-type update다. Line-search safeguard 버전의 objective trace smoke test는 통과했지만, exact penalized EM 이론은 아니므로 논문에서는 proximal EM-type update로 제한해 표현한다.

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

초기 summary 기준으로는 집중도 차이가 약해져도 에타 패널티 + refit이 true union q=22에 가까운 변수 수를 선택하고 FPR을 0.025로 유지하는 것처럼 보였다. 그러나 2026-06-16 추가 진단에서 line-search safeguard, zero-support 처리, path 후보 전체 저장 후 weak setting의 Eta BIC가 null support 또는 dense support로 불안정하게 튀는 문제가 확인되었다.

따라서 이 결과는 본문 주력 성공 사례가 아니라, 현재 알고리즘의 failure mode와 tuning sensitivity를 보여주는 appendix 후보로 낮추는 것이 안전하다.

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
| weak setting 이슈 | Eta BIC가 null/dense support로 불안정해질 수 있으며, 단순 path grid refinement로 해결되지 않았다 |

## 10. Real data 결과의 위치

현재 real data 결과는 본문 핵심 주장보다는 보조 근거로 조심스럽게 배치한다.

| 데이터 | 현재 해석 |
|:---|:---|
| PBMC 3K lymphoid3 | vMF mixture 계열에서는 Eta가 Rossi보다 높고 sparse marker selection을 제공한다. 다만 sparse k-means가 ARI만 보면 더 높으므로 전체 baseline 최고라고 주장하지 않는다. |
| BBC News text | 높은 ARI와 selected q 감소를 보이는 보조 사례다. 그러나 kappa ratio가 1에 가깝고 Rossi 대비 개선폭이 작으므로 본문 핵심 real-data benchmark보다는 appendix 또는 supplementary 사례가 적절하다. |

## 11. 2026-06-16 추가 진단: weak path instability

weak concentration setting에서 Eta path construction이 단순히 성긴 grid 때문에 실패하는지 확인하기 위해 path 후보 전체를 저장하고 oracle/adaptive refinement를 비교했다.

| path construction | scope | near22 후보율 | BIC null 선택률 | positive-support dense 선택률 |
|:---|:---|---:|---:|---:|
| no refinement | weak100 | 0.23 | 0.73 | 0.72 |
| oracle target-refine | weak100 | 0.89 | 0.14 | 0.09 |
| adaptive v1 support-jump | weak100 | 0.73 | 0.24 | 0.24 |
| adaptive v2 priority midpoint | smoke10 | 0.50 | 0.50 | 0.50 |
| adaptive v2.1 duplicate endpoint | smoke10 | 0.50 | 0.40 | 0.40 |
| adaptive v3 multi-point | smoke10 | 0.50 | 0.60 | 0.40 |

v3에서는 multi-point refinement로 990개 후보를 평가했지만 saved unique support는 6개뿐이었다. 즉 weak setting의 문제는 단순히 lambda grid를 더 촘촘히 찍으면 해결되는 문제가 아니라, proximal path가 같은 support plateau에 머물고 BIC 선택이 null/dense 쪽으로 불안정해지는 문제로 보는 것이 타당하다.

2026-06-16에 target-free stability selection smoke diagnostic도 확인했다. 설정은 subsample 비율 0.7, B=20, threshold=0.6, weak setting 5회 반복이었다. 모든 반복에서 ERROR row는 없었고 stability row는 정상 저장되었지만, selected support는 `[0, 0, 22, 0, 0]`이었다. 즉 5회 중 4회가 zero support였고, stability + refit은 valid_reps가 1/5, zero_support_refit_reps가 4/5였다.

| method | scope | selected q 평균 | refit valid_reps | zero-support refit | 판단 |
|:---|:---|---:|---:|---:|:---|
| Eta centered path BIC | smoke5 | 4.2 | 5/5 | 0/5 | null support가 여전히 자주 선택됨 |
| Eta centered path BIC + refit | smoke5 | 4.2 | 1/5 | 4/5 | zero-support refit 문제가 그대로 남음 |
| Eta centered positive-support BIC | smoke5 | 83.6 | 5/5 | 0/5 | zero는 피하지만 dense support와 높은 FPR 문제가 큼 |
| Eta centered stability | smoke5 | 4.4 | 5/5 | 0/5 | threshold 0.6에서 대부분 zero support |
| Eta centered stability + refit | smoke5 | 4.4 | 1/5 | 4/5 | 현재 설정만으로는 zero-support 문제를 줄이지 못함 |

따라서 stability selection은 아직 성능 개선 결과가 아니라 다음 tuning instability 보강 후보의 첫 smoke test로만 해석한다. Threshold sweep, subsample 비율 변화, B 증가, 그리고 weak100 재실행 전에는 본문 claim에 넣으면 안 된다.

Stability threshold sweep smoke도 수행했다. Threshold 0.2, 0.3, 0.4, 0.5, 0.6 모두 ERROR 없이 실행됐지만, 모든 threshold에서 zero-support stability selection이 4/5로 동일했다. Nonzero인 한 반복만 selected q가 25, 23, 22, 22, 22로 변했고 dense support failure는 없었다. 즉 threshold만 낮추는 방식은 zero-support 문제를 해결하지 못했고, 현재 evidence로는 stability selection도 공식 tuning으로 올리기 어렵다.

Subsample-level diagnostic을 추가로 저장한 결과, zero-support 반복 1, 2, 4, 5에서는 20개 subsample 모두가 selected q=0을 골랐다. 모든 subsample fit은 성공했으므로 frequency=0의 직접 원인은 fit failure나 non-overlapping support dispersion이 아니라, subsample별 BIC가 반복적으로 null support를 선택하는 현상으로 보인다. Rep 3만 예외적으로 subsample 20개 중 12개가 q=17-27 근처를 선택했고 full stability support도 q=22였다.

이 결과는 full-data Eta path 진단과도 일관적이다. Zero-support 반복 1, 2, 4, 5의 full-data path는 q=17-27 후보를 만들지 못하고 q=0 또는 q>=75 후보로만 구성되었다. 따라서 weak100 no-refinement의 near22 후보율 0.23, BIC null률 0.73, positive-support dense률 0.72는 단순 threshold calibration 문제가 아니라 path/BIC selection instability로 해석하는 편이 더 안전하다. 다음 보강은 stability threshold 조정보다 alternative IC 또는 Eta proximal/MM/coordinate update 개선 쪽이 우선이다.

IC/df 관점에서 추가 확인한 결과, 현재 subsample diagnostic은 각 subsample의 selected candidate만 저장하므로 subsample 내부에서 alternative IC/df가 어떤 lambda 후보를 골랐을지는 직접 재계산할 수 없다. 다만 zero-support 반복의 selected candidate df가 모두 null-support baseline df=103이고, full-data path에도 중간 support 후보가 없다는 점을 함께 보면, weak instability는 BIC penalty 단독 문제라기보다 중간 support 후보가 부족한 path/BIC instability로 해석하는 것이 안전하다. 다음 IC 진단이 필요하다면 subsample별 전체 lambda path를 저장해야 한다.

Subsample full-path diagnostic smoke 3회에서는 이 분해를 직접 확인했다. `df_current`, `df_no_d`, `df_support_only`, `df_half_d`는 모두 같은 선택을 냈다. 이 기준들은 후보 간 상수항만 다르고 selected q에 대한 penalty slope가 `(K-1)m`으로 같기 때문이다. `df_direction_only = (K-1)+m`처럼 slope를 낮추면 zero rate는 0.783에서 0.667로 조금 줄지만 dense75 rate가 0.117로 생기고 FPR도 올라갔다. 따라서 현재 evidence로는 d 상수항을 빼는 식의 단순 df 수정이 해결책이 아니라, path 후보 형성과 IC slope/selection rule을 같이 봐야 한다.

더 촘촘한 slope sensitivity에서도 같은 결론이 나왔다. `df = c0 + gamma * selected_q`에서 c0는 후보 간 상수라 선택을 전혀 바꾸지 않았고, 선택은 gamma에 의해 결정되었다. Gamma를 3에서 1-1.5로 낮추면 zero rate는 0.783에서 0.667로 줄지만 near22 rate는 0.20에 머물고 dense75 rate가 0.117로 생겼다. Gamma=0.5는 dense75 rate가 0.333, FPR이 0.332로 커졌다. 따라서 weak setting 보강은 단순 df 상수항 수정이 아니라 IC slope/selection rule 또는 path update 개선 문제다.

교수님께 가져갈 결론은 다음이다.

1. Strong/common+specific setting은 Eta penalty의 장점을 보여주는 본문 후보로 유지한다.
2. Weak setting은 현재 상태에서 본문 성공 사례로 쓰면 위험하며, appendix diagnostic 또는 limitation으로 낮춘다.
3. Oracle target-refine은 true q 주변 정보를 사용하므로 공식 알고리즘이 될 수 없다.
4. Adaptive v2/v2.1/v3도 target-free이지만, unique support 다양성을 충분히 만들지 못해 공식 path construction으로 채택하기 어렵다.
5. 다음 보강은 grid refinement가 아니라 stability selection, alternative IC, 또는 MM/coordinate update 개선 쪽이 더 합리적이다.

교수님께 물어볼 질문은 다음이다.

1. Weak setting을 본문에서 제외하고 appendix failure-mode로 낮추는 데 동의하시는가?
2. 공식 tuning은 일단 path+BIC로 유지하되, EBIC/RICc/stability selection을 sensitivity로 추가하는 방향이 적절한가?
3. Eta proximal update를 더 이론적으로 방어하기 위해 MM safeguard 또는 coordinate descent 쪽으로 구현을 바꾸는 것이 논문 기여에 필요한가?
4. 논문 주장을 "Eta가 ARI를 크게 올린다"가 아니라 "ARI를 유지하면서 sparse and interpretable eta contrast를 회복한다"로 제한하는 데 동의하시는가?
5. Weak setting의 null/dense instability를 한계로 공개하는 것이 좋은가, 아니면 방법론 보강 후 다시 본문 후보로 올릴지 결정해야 하는가?
