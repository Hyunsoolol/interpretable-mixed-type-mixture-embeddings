# Thesis Meeting 260622

업데이트: 2026-06-16

## 1. 핵심 결론

현재 연구의 가장 안전한 주장은 다음이다.

> Eta penalty는 ARI를 크게 올리는 방법이라기보다, vMF mixture 안에서 posterior decision parameter인 `eta = kappa * mu`의 component contrast를 sparse하게 만들어, ARI를 유지하면서 해석 가능한 변수 선택을 제공하는 방법이다.

따라서 본문 주장은 “clustering accuracy 개선”보다 “model-based sparse interpretation”에 맞추는 것이 안전하다.

## 2. 현재 논문 배치안

| 결과/setting | 현재 판단 | 이유 |
|:---|:---|:---|
| strong common+specific simulation | 본문 핵심 후보 | ARI는 유지되고 selected q, FPR, F1이 가장 안정적으로 개선됨 |
| concentration-dominant simulation | 본문 보조 후보 | Rossi sparse vMF의 한계를 설명하기 좋음 |
| weak concentration simulation | appendix/limitation | Eta BIC가 null/dense support로 불안정함 |
| high-dimensional stress | appendix | BIC와 signal strength 문제가 섞여 본문 성공 사례로는 위험함 |
| PBMC lymphoid3 | real-data 보조 사례 | vMF mixture 계열의 marker selection 사례로는 가능하나 baseline 최고 주장은 불가 |
| BBC News/text | appendix 또는 삭제 후보 | 본문 핵심 claim을 강하게 지지하기에는 약함 |
| adaptive refinement diagnostics | appendix diagnostic | grid refinement로 weak instability가 해결되지 않음을 보여주는 진단 자료 |

## 3. 모형 아이디어

Rossi & Barbaro (2022)는 component direction `mu_k`에 sparsity penalty를 둔다. 그러나 vMF mixture의 posterior classification에는 `mu_k` 자체보다

```text
eta_k = kappa_k * mu_k
```

가 직접 들어간다. 두 component의 posterior boundary도 `eta_2 - eta_1` contrast로 표현된다. 따라서 concentration 차이가 중요한 setting에서는 `mu`가 아니라 `eta` contrast를 sparse하게 만드는 것이 더 자연스럽다.

K>2에서는 coordinate별 centered eta contrast를 사용한다.

```text
c_kj = eta_kj - mean_k(eta_kj)
penalty = lambda_eta * sum_j ||c_.j||_2
```

현재 구현은 exact penalized EM이 아니라, unpenalized eta update 뒤에 centered eta contrast shrinkage를 적용하는 proximal EM-type update다. 논문에서도 이 표현을 유지해야 한다.

## 4. 핵심 simulation 결과

### 4.1 Strong common+specific setting

설정:

```text
K = 4, n = 1000, d = 100, rep = 100
common variables = 6
component-specific variables = 16
true union q = 22
kappa = (30, 45, 65, 90)
```

| Method | ARI | Selected q | TPR | FPR | Precision | F1 |
|:---|---:|---:|---:|---:|---:|---:|
| Rossi | 0.680 | 98.520 | 1.000 | 0.981 | 0.223 | 0.365 |
| Separate penalty | 0.684 | 86.460 | 1.000 | 0.826 | 0.258 | 0.409 |
| Eta penalty + refit | 0.686 | 24.750 | 0.994 | 0.037 | 0.890 | 0.937 |

해석:

Eta penalty + refit은 ARI를 유지하면서 true union q=22에 가까운 support를 선택한다. 이 결과가 현재 본문에 가장 적합하다.

### 4.2 Concentration-dominant setting

| Method | ARI | Selected q | TPR | FPR | Precision | F1 |
|:---|---:|---:|---:|---:|---:|---:|
| Rossi | 0.513 | 98.500 | 1.000 | 0.983 | 0.102 | 0.184 |
| Separate penalty | 0.525 | 95.600 | 1.000 | 0.951 | 0.105 | 0.190 |
| Eta penalty + refit | 0.523 | 28.800 | 1.000 | 0.209 | 0.443 | 0.586 |

해석:

Rossi와 separate penalty는 clustering은 어느 정도 되지만 거의 모든 변수를 선택한다. Eta penalty는 ARI를 비슷하게 유지하면서 noise selection을 줄인다.

## 5. Weak setting 진단

Weak concentration setting에서는 처음 summary만 보면 Eta penalty가 잘 작동하는 것처럼 보였다. 하지만 line-search safeguard, zero-support 처리, path candidates 저장 후 다시 확인하니 Eta BIC가 null support 또는 dense support로 불안정하게 튀는 문제가 확인됐다.

| path construction | scope | near22 후보율 | BIC null률 | positive dense률 | 판단 |
|:---|:---|---:|---:|---:|:---|
| no refinement | weak100 | 0.23 | 0.73 | 0.72 | 기본 path가 중간 support를 충분히 만들지 못함 |
| oracle target-refine | weak100 | 0.89 | 0.14 | 0.09 | true q 주변 정보를 쓰므로 공식 알고리즘 불가 |
| adaptive v1 | weak100 | 0.73 | 0.24 | 0.24 | 개선은 있으나 oracle 수준은 아님 |
| adaptive v2 | smoke10 | 0.50 | 0.50 | 0.50 | midpoint refinement만으로 부족 |
| adaptive v2.1 | smoke10 | 0.50 | 0.40 | 0.40 | duplicate endpoint를 써도 support 다양성 부족 |
| adaptive v3 | smoke10 | 0.50 | 0.60 | 0.40 | 990개 평가에도 saved unique support 6개뿐 |

결론:

Weak setting의 문제는 단순히 lambda grid가 성긴 문제가 아니다. Proximal path가 같은 support plateau에 머물고, BIC가 null/dense 쪽으로 불안정하게 선택하는 문제로 보는 것이 더 타당하다. 따라서 weak setting은 본문 성공 사례가 아니라 appendix diagnostic 또는 limitation으로 낮추는 것이 안전하다.

## 6. Stability selection 진단

Stability selection도 바로 해결책이 되지는 않았다.

| Diagnostic | 결과 | 해석 |
|:---|:---|:---|
| threshold 0.6 smoke5 | 5회 중 4회 zero support | 공식 tuning 후보로 보기 어려움 |
| threshold sweep 0.2-0.6 | 모든 threshold에서 4/5 zero support | threshold만 낮춰도 해결 안 됨 |
| subsample diagnostic | zero-support reps에서는 20개 subsample 모두 q=0 선택 | fit failure가 아니라 subsample BIC가 반복적으로 null을 선택 |
| IC slope sensitivity | gamma를 낮추면 zero는 줄지만 dense/FPR 증가 | 단순 df 상수항 수정은 해결책 아님 |

다음 보강은 stability threshold 조정보다 alternative IC, selection rule, 또는 Eta update 자체의 개선 쪽이 우선이다.

## 7. Real data 위치

PBMC lymphoid3는 real-data 보조 사례로 사용할 수 있다. 다만 sparse k-means보다 ARI가 높다는 식의 주장은 피해야 한다. Eta의 장점은 vMF mixture 안에서 marker support를 더 해석 가능하게 준다는 점으로 제한한다.

BBC News/text 결과는 본문 핵심 근거로는 약하다. appendix 또는 supplementary 사례로 두는 것이 안전하다.

## 8. 현재 방어 가능한 주장과 어려운 주장

방어 가능한 주장:

1. Eta contrast는 posterior decision parameter에 직접 연결된다.
2. Strong/common+specific setting에서 Eta penalty + refit은 ARI를 유지하면서 selected q와 FPR을 크게 줄인다.
3. Rossi sparse vMF는 concentration-dominant setting에서 noise variable을 과도하게 선택할 수 있다.
4. Weak setting에서는 path/BIC instability가 존재하며, 이를 limitation으로 명확히 제시할 수 있다.

아직 방어하기 어려운 주장:

1. Eta penalty가 모든 setting에서 ARI를 크게 개선한다.
2. Weak concentration setting에서도 Eta BIC가 안정적으로 true support를 회복한다.
3. Lambda grid refinement만으로 weak instability가 해결된다.
4. 현재 Eta update가 exact penalized EM이다.
5. Oracle target-refine 또는 adaptive v2/v3를 공식 알고리즘으로 쓸 수 있다.

## 9. 교수님께 여쭤볼 결정 사항

1. 논문 주장을 “ARI 개선”이 아니라 “ARI 유지 + sparse interpretable eta contrast”로 제한해도 되는가?
2. Strong/common+specific setting을 본문 핵심 simulation으로 두는 데 동의하시는가?
3. Weak setting은 appendix diagnostic 또는 limitation으로 낮춰도 되는가?
4. Official tuning은 일단 path+BIC로 유지하고, EBIC/RICc/stability selection은 sensitivity로 둘지?
5. Eta proximal update를 MM safeguard 또는 coordinate/proximal update로 보강해야 논문 기여가 충분해지는가?
6. 논문 타깃을 top-tier 방법론보다 중상위/응용방법론 쪽으로 조정하는 것이 현실적인가?

## 10. 미팅 전 남은 작업

1. Strong/common+specific 결과 표를 최종 본문 후보로 정리한다.
2. Weak setting은 failure-mode appendix 문장으로 정리한다.
3. PBMC 결과는 “best ARI”가 아니라 “model-based marker interpretation” 중심으로 다시 쓴다.
4. Eta update를 proximal EM-type update로 일관되게 표현한다.
5. 교수님 의견에 따라 다음 보강을 alternative IC로 갈지, MM/coordinate update로 갈지 결정한다.
