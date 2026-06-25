# Negative-control Simulation Summary 260708

## 1. 목적

이 문서는 Eta-group penalty가 Rossi 또는 Separate penalty보다 불리해질 수 있는 상황을 확인하기 위한 negative-control diagnostic 결과를 정리한다. 여기서의 실행은 official algorithm 변경이 아니라, 방법의 적용 범위와 한계를 확인하기 위한 진단용 실험이다.

현재 official method는 그대로 `Eta-group path+BIC + selected support fixed unpenalized refit`으로 둔다.

## 2. 기존 smoke 결과 요약

| Setting | 주요 설계 | 주요 결과 | 해석 |
|:---|:---|:---|:---|
| A: direction-sparse smoke | $K=4$, $d=100$, true union $q=20$, $\kappa=(60,60,60,60)$ | Eta + refit ARI=0.976, selected q=28.8, F1=0.676; Rossi/Separate ARI=0.975, selected q=100, F1=0.333 | Rossi/Separate가 뚜렷하게 유리한 setting은 아니었다. Eta가 support recovery는 더 좋았지만, Eta refit의 MSE_kappa가 컸다. |
| B: dense eta smoke | $K=4$, $d=100$, true union $q=80$, $\kappa=(30,45,65,90)$ | Eta + refit ARI=0.377, selected q=47.4, F1=0.702; Separate + refit ARI=0.396, selected q=100, F1=0.889 | 좋은 negative-control 후보. True support가 dense하면 Eta-group이 과도하게 shrink하여 ARI/F1 손실이 생긴다. |
| C: weak signal smoke | $K=4$, $d=100$, $w=0.20$, $\kappa=(25,30,35,40)$ | Eta BIC selected q=0, refit invalid; Rossi/Separate ARI near 0 | 너무 어려운 low-signal stress setting이다. 거의 모든 방법이 실패한다. |

## 3. Setting B rep50: dense eta / 약한 sparsity truth

설계:

| 항목 | 값 |
|:---|:---|
| K | 4 |
| n | 1000 |
| d | 100 |
| rep | 50 |
| common q | 20 |
| specific q per component | 15 |
| true union q | 80 |
| specific weight | 0.25 |
| kappa | $(30,45,65,90)$ |
| selection | BIC |

결과:

| Method | ARI | Selected q | TPR | FPR | Precision | F1 | MSE_mu | MSE_kappa | MSE_centered_eta |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Rossi BIC | 0.384 | 99.90 | 1.000 | 0.996 | 0.801 | 0.889 | 0.000886 | 70.526 | 2.236 |
| Rossi BIC + refit | 0.380 | 99.90 | 1.000 | 0.996 | 0.801 | 0.889 | 0.000993 | 78.229 | 2.544 |
| Separate BIC | 0.381 | 99.74 | 1.000 | 0.988 | 0.802 | 0.890 | 0.000956 | 52.943 | 1.753 |
| Separate BIC + refit | 0.378 | 99.74 | 1.000 | 0.988 | 0.802 | 0.890 | 0.001029 | 50.549 | 2.150 |
| Eta-group BIC | 0.324 | 52.82 | 0.615 | 0.180 | 0.944 | 0.726 | 0.000557 | 186.886 | 2.895 |
| Eta-group BIC + refit | 0.368 | 52.82 | 0.615 | 0.180 | 0.944 | 0.726 | 0.000897 | 93.913 | 2.721 |

해석:

- Setting B는 현재까지 가장 명확한 Eta-group failure mode다.
- True decision support가 dense한데 Eta-group은 selected q=52.82만 선택해 true union q=80보다 훨씬 작은 support를 고른다.
- 이로 인해 FPR과 Precision은 좋아지지만, TPR, F1, ARI, MSE_centered_eta에서는 Separate보다 손해를 본다.
- 따라서 이 결과는 Eta-group이 dense weak-sparsity truth에 맞는 방법이 아님을 보여주는 유용한 negative-control 결과다.

## 4. Setting C2 rep50: 완화된 weak signal

설계:

| 항목 | 값 |
|:---|:---|
| K | 4 |
| n | 1000 |
| d | 100 |
| rep | 50 |
| common q | 6 |
| specific q per component | 4 |
| true union q | 22 |
| specific weight | 0.25 |
| kappa | $(35,45,55,65)$ |
| selection | BIC |

결과:

| Method | ARI | Selected q | TPR | FPR | Precision | F1 | MSE_mu | MSE_kappa | MSE_centered_eta |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Rossi BIC | 0.140 | 99.80 | 1.000 | 0.997 | 0.220 | 0.361 | 0.001168 | 88.023 | 3.275 |
| Rossi BIC + refit | 0.131 | 99.80 | 1.000 | 0.997 | 0.220 | 0.361 | 0.001498 | 93.711 | 3.990 |
| Separate BIC | 0.141 | 98.46 | 1.000 | 0.980 | 0.224 | 0.365 | 0.000895 | 78.332 | 2.386 |
| Separate BIC + refit | 0.130 | 98.46 | 1.000 | 0.980 | 0.224 | 0.365 | 0.001376 | 93.758 | 3.740 |
| Eta-group BIC | 0.012 | 2.68 | 0.058 | 0.018 | 0.706 | 0.462 | 0.000408 | 126.285 | 2.050 |
| Eta-group BIC + refit | 0.121 | 2.68 | 0.058 | 0.018 | 0.706 | 0.462 | 0.002925 | 247.896 | 4.330 |
| Eta-group positive-support | 0.078 | 15.52 | 0.448 | 0.073 | 0.800 | 0.531 | 0.000508 | 150.646 | 2.198 |
| Eta-group positive-support + refit | 0.137 | 15.52 | 0.448 | 0.073 | 0.800 | 0.531 | 0.001265 | 172.028 | 3.118 |

해석:

- C2 rep50에서도 weak-signal 구조는 여전히 어렵다. Rossi/Separate는 ARI가 약 0.13-0.14 수준이고 support는 거의 full support로 선택된다.
- Standard Eta BIC는 평균 selected q=2.68로 매우 sparse하며, 50회 중 43회에서 selected q=0을 선택했다. 따라서 BIC-selected refit의 valid replicate는 7회뿐이다.
- Positive-support diagnostic을 쓰면 zero support는 피하고 FPR=0.073, Precision=0.800, F1=0.531로 support metric은 좋아지지만, ARI=0.137로 Rossi/Separate와 비슷하거나 낮다.
- 따라서 C2는 Rossi/Separate가 깨끗하게 유리한 setting이라기보다, Eta BIC의 zero-support tuning failure와 weak-signal clustering difficulty를 보여주는 diagnostic으로 보는 것이 맞다.

## 5. Setting A 재설계 필요성

첫 번째 Setting A는 Rossi/Separate가 유리한 setting으로 충분하지 않았다. 이유는 현재 support metric이 coordinate union support이기 때문이다. 이 metric은 Eta-group이 공통 non-discriminating coordinate를 제거할 때 자연스럽게 유리해진다.

Rossi와 Separate는 dense support를 선택하는 반면, Eta-group은 더 작은 decision-contrast support를 유지한다. 따라서 Rossi가 자연스럽게 유리한 상황을 평가하려면 목표를 분리해야 한다.

- 관심 대상이 posterior decision support이면 Eta-group이 자연스럽게 유리하다.
- 관심 대상이 prototype 또는 direction support이면 Rossi가 더 적절한 비교 대상일 수 있다.

A2 설계 방향:

1. Concentration을 같거나 거의 같게 두어 $\eta$ contrast가 $\mu$ contrast와 가깝게 만든다.
2. Component-specific direction pattern을 명확히 만든다.
3. Prototype support recovery와 decision support recovery를 분리해서 평가한다.
4. Coordinate union support뿐 아니라 entry-level $\mu$ support metric도 함께 보고한다.

## 6. Setting A2 smoke: direction-sparse / equal concentration

설계:

| 항목 | 값 |
|:---|:---|
| K | 4 |
| n | 1000 |
| d | 100 |
| rep | 5 |
| common q | 1 |
| specific q per component | 5 |
| true union q | 21 |
| specific weight | 1.0 |
| kappa | $(60,60,60,60)$ |
| selection | BIC |

결과:

| Method | ARI | Selected q | TPR | FPR | Precision | F1 | entry_TPR | entry_FPR | MSE_mu | MSE_kappa | MSE_centered_eta |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Rossi BIC | 0.999 | 100.00 | 1.000 | 1.000 | 0.210 | 0.347 | 1.000 | 0.813 | 0.000091 | 1.344 | 0.262 |
| Rossi BIC + refit | 0.999 | 100.00 | 1.000 | 1.000 | 0.210 | 0.347 | NA | NA | 0.000130 | 1.383 | 0.373 |
| Separate BIC | 0.999 | 100.00 | 1.000 | 1.000 | 0.210 | 0.347 | 1.000 | 0.725 | 0.000077 | 1.306 | 0.222 |
| Separate BIC + refit | 0.999 | 100.00 | 1.000 | 1.000 | 0.210 | 0.347 | NA | NA | 0.000130 | 1.383 | 0.373 |
| Eta-group BIC | 0.998 | 40.60 | 0.962 | 0.258 | 0.502 | 0.658 | NA | NA | 0.000067 | 4.256 | 0.160 |
| Eta-group BIC + refit | 0.998 | 40.60 | 0.962 | 0.258 | 0.502 | 0.658 | NA | NA | 0.001473 | 43.610 | 0.250 |

해석:

- A2도 현재 coordinate union support 기준에서는 Rossi/Separate가 명확히 유리한 결과를 만들지 못했다.
- Rossi와 Separate는 true coordinate를 모두 잡지만 noise coordinate도 거의 모두 선택해 selected q=100, FPR=1.000이 된다.
- Eta-group은 ARI를 거의 유지하면서 selected q=40.60으로 줄이고 union-support F1도 더 높다.
- 다만 Rossi/Separate는 entry-level prototype support에서 entry_TPR=1.000을 보인다. 이 부분은 Rossi류 방법의 자연스러운 목표와 관련이 있다.
- 따라서 A2는 방법의 명확한 실패라기보다 metric mismatch를 보여준다. Rossi-style direction sparsity를 공정하게 평가하려면 prototype/entry-level support recovery와 posterior decision support를 분리해야 한다.

### 6.1 Setting A4 redesign smoke: entry-sparse / union-dense

A4는 기존 generator를 수정하지 않고 만들 수 있는 가장 직접적인 entry-sparse / union-dense redesign이다. 공통 좌표를 제거하고, 각 component가 서로 겹치지 않는 20개 좌표만 사용하도록 했다.

설계:

| 항목 | 값 |
|:---|:---|
| K | 4 |
| n | 1000 |
| d | 100 |
| rep | 5 |
| common q | 0 |
| specific q per component | 20 |
| true union q | 80 |
| true entry q | 80 |
| specific weight | 1.0 |
| kappa | $(60,60,60,60)$ |
| selection | BIC |

결과:

| Method | ARI | Selected q | FPR | Precision | F1 | entry_TPR | entry_FPR | entry_Precision | entry_F1 | MSE_mu | MSE_kappa | MSE_centered_eta |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Rossi BIC | 0.999 | 100.00 | 1.000 | 0.800 | 0.889 | 1.000 | 0.754 | 0.249 | 0.399 | 0.000090 | 1.286 | 0.257 |
| Rossi BIC + refit | 0.999 | 100.00 | 1.000 | 0.800 | 0.889 | NA | NA | NA | NA | 0.000130 | 1.339 | 0.370 |
| Separate BIC | 0.999 | 99.60 | 0.980 | 0.803 | 0.891 | 1.000 | 0.643 | 0.280 | 0.438 | 0.000075 | 1.239 | 0.215 |
| Separate BIC + refit | 0.999 | 99.60 | 0.980 | 0.803 | 0.891 | NA | NA | NA | NA | 0.000130 | 1.339 | 0.370 |
| Eta-group BIC | 0.999 | 91.00 | 0.550 | 0.881 | 0.936 | NA | NA | NA | NA | 0.000130 | 8.791 | 0.417 |
| Eta-group BIC + refit | 0.999 | 91.00 | 0.550 | 0.881 | 0.936 | NA | NA | NA | NA | 0.000123 | 1.281 | 0.355 |

해석:

- A4에서도 ARI는 모든 방법에서 거의 동일하게 높다.
- Coordinate union support 기준으로는 Eta-group이 여전히 selected q=91.00, F1=0.936으로 Rossi/Separate보다 낫다.
- Prototype entry support 기준에서는 Separate BIC가 Rossi BIC보다 entry_F1이 높다. Eta-group은 coordinate-level contrast를 선택하는 방법이므로 동일한 entry-level prototype metric이 직접 정의되지 않는다.
- 따라서 A4는 Rossi/Separate가 전체적으로 우월한 setting이라기보다, posterior decision support와 prototype entry support를 분리해서 보고해야 한다는 점을 더 분명하게 보여준다.
- Full rep50은 가능하지만, "Eta-group의 깨끗한 실패"를 확인하기 위한 full run으로는 아직 충분히 강한 후보가 아니다. 교수님께 prototype-support metric을 먼저 확인받는 것이 더 안전하다.

## 7. Fragmented block-like smoke: 공유 좌표 없는 setting

Rossi/Separate가 유리할 수 있는 setting으로, 공통 support를 0으로 두고 component-specific support만 사용하는 fragmented block-like setting을 현재 generator 안에서 확인했다. 완전한 binary block-diagonal generator는 아니지만, R core simulation code를 수정하지 않고 만들 수 있는 가장 가까운 diagnostic이다.

### 7.1 저차원 fragmented smoke

설계: $K=4$, $n=1000$, $d=60$, rep=5, common q=0, specific q=10 per component, true union q=40, $\kappa=(60,60,60,60)$.

| Method | ARI | Selected q | FPR | Precision | F1 | MSE_mu | MSE_kappa | MSE_centered_eta |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| Rossi BIC | 1.000 | 59.00 | 0.950 | 0.678 | 0.808 | 0.000049 | 1.481 | 0.156 |
| Separate BIC | 1.000 | 56.80 | 0.840 | 0.705 | 0.827 | 0.000034 | 1.394 | 0.112 |
| Eta-group BIC | 0.999 | 42.80 | 0.140 | 0.936 | 0.967 | 0.000109 | 11.101 | 0.458 |
| Eta-group BIC + refit | 0.999 | 42.80 | 0.140 | 0.936 | 0.967 | 0.000079 | 1.490 | 0.244 |

해석:

- 이 low-dimensional fragmented setting에서도 Rossi/Separate가 전체적으로 명확히 우월하다고 보기는 어렵다.
- Rossi/Separate는 ARI가 1에 가깝고 Separate BIC의 raw prototype-parameter MSE는 낮지만, 여전히 dense support를 선택한다.
- Eta-group은 ARI를 거의 유지하면서 decision support를 더 sparse하게 만든다.
- 따라서 prototype-parameter accuracy와 decision-support sparsity는 별도 지표로 보고해야 한다.

### 7.2 고차원 fragmented smoke

설계: $K=4$, $n=1000$, $d=400$, rep=3, common q=0, specific q=20 per component, true union q=80, $\kappa=(60,60,60,60)$.

| Method | ARI | Selected q | FPR | Precision | F1 | MSE_mu | MSE_kappa | MSE_centered_eta |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| Rossi BIC | 0.826 | 400.00 | 1.000 | 0.200 | 0.333 | 0.000425 | 48.931 | 1.417 |
| Separate BIC | 0.827 | 400.00 | 1.000 | 0.200 | 0.333 | 0.000425 | 25.462 | 1.329 |
| Eta-group BIC | 0.851 | 368.33 | 0.901 | 0.217 | 0.357 | 0.000310 | 4.283 | 0.736 |
| Eta-group BIC + refit | 0.832 | 368.33 | 0.901 | 0.217 | 0.357 | 0.000419 | 46.142 | 1.408 |

해석:

- High-dimensional version에서는 모든 방법이 dense해진다.
- Rossi/Separate가 기대했던 명확한 우위를 얻지는 못했고, Eta-group도 dense하지만 조금 덜 dense하다.
- 따라서 현재 generator는 "block-diagonal fragmented data에서 Rossi가 지배적으로 유리하다"는 setting을 재현하지 못한다.
- 이 질문을 제대로 보려면 dedicated block-diagonal 또는 binary-style generator와 prototype-support metric이 필요하다.

### 7.3 d=800 attempt

d=800 fragmented smoke도 common q=0, specific q=20 per component로 시도했다. Replicate 계산은 진행됐지만 summary binding 단계에서 column mismatch error가 발생했다. 이번 diagnostic에서는 R core algorithm을 수정하지 않기로 했으므로 이 결과는 해석에 사용하지 않는다. 대신 very high-dimensional diagnostic에서 script robustness issue가 있음을 기록한다.

## 8. 현재 결론

현재까지 가장 유용한 negative-control 결과는 Setting B rep50이다.

Setting B는 Eta-group이 보편적으로 더 좋은 방법이 아님을 보여준다. True separation이 dense하고 많은 weak coordinate가 함께 작동하는 경우 Eta-group은 support를 과도하게 shrink하여 Rossi 또는 Separate보다 ARI/F1이 낮아질 수 있다. 이 결과는 논문에서 limitation 또는 negative-control diagnostic으로 포함할 가치가 있다.

C2 rep50은 weak signal에서 Eta BIC가 zero support를 자주 선택할 수 있음을 보여준다. 50회 중 43회에서 selected q=0이었고, BIC-selected refit valid replicate는 7회뿐이었다. 따라서 positive-support 또는 alternative tuning은 diagnostic으로는 유용하지만 official tuning으로 제시하면 안 된다.

A2는 equal-concentration direction-sparse design에서도 coordinate union support 기준으로는 Eta-group이 여전히 유리하게 보인다는 점을 보여준다. 이는 Rossi/Separate가 더 나을 수 없다는 뜻이 아니라, prototype-level target과 decision-level target을 분리해야 한다는 뜻이다.

새로 확인한 A4 smoke도 같은 방향이다. Union-dense / entry-sparse 구조를 만들었지만 coordinate union support 기준에서는 Eta-group이 여전히 더 좋고, prototype entry support에서는 Separate가 더 나아 보인다. 따라서 이 setting은 clean failure라기보다 metric separation diagnostic으로 보는 것이 맞다.

Fragmented low-dimensional/high-dimensional smoke도 현재 generator에서는 clean Rossi/Separate-dominant result를 만들지 못했다. 모든 방법이 clustering을 잘하거나 dense support로 가며, prototype-sparsity advantage가 분리되지 않는다. Rossi/Separate가 구조적으로 유리한 setting을 보이려면 dedicated block-diagonal generator가 필요하다.

## 9. Full simulation 권장 사항

| 후보 | 권장 여부 | 이유 |
|:---|:---|:---|
| Setting B rep100 | 선택 사항, 급하지 않음 | rep50만으로도 안정적인 negative-control evidence가 있음 |
| Setting C2 rep50 | 완료 | Eta BIC zero-support tuning failure를 확인했지만, Rossi/Separate도 dense support이므로 cleaner weak-signal negative-control은 별도 설계가 필요함 |
| Setting A4 rep50 | 조건부 보류 | entry-level에서는 Separate가 유리해 보이나 union support에서는 Eta-group이 여전히 유리함. 먼저 prototype-support metric 정의가 필요함 |
| Prototype-support metric | 다음 우선 작업 | Rossi/Separate의 자연스러운 목표를 평가하기 위해 필요함 |
| Dedicated block-diagonal generator | metric 정의 이후 권장 | Rossi/Separate가 구조적으로 유리한 fragmented-data scenario를 깨끗하게 테스트하기 위해 필요함 |
