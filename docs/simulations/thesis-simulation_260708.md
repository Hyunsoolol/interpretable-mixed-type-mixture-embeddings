# Thesis Simulation 260708

## 1. 목적

7월 8일 연구미팅 공유용으로, 논문용 후보 simulation S1-S4 결과를 정리한다.
이번 simulation의 핵심 질문은 clustering accuracy 자체보다 posterior decision support recovery이다.

제안 모형의 목표 support는 centered eta contrast
\[
c_{kj}=\eta_{kj}-\bar{\eta}_j,\qquad \eta_k=\kappa_k\mu_k
\]
에서 0이 아닌 coordinate이다. 따라서 공통 변수는 \(\mu\)에는 존재할 수 있지만 모든 component에 같은 방향으로 들어가므로 decision support에는 포함하지 않는다.

## 2. Simulation Setting

공통 설정:

| 항목 | 값 |
|---|---:|
| \(K\) | 4 |
| \(n\) | 1000 |
| \(d\) | 200 |
| 공통 변수 | 4 |
| 군집특성 변수 | 16 |
| Noise 변수 | 180 |
| True decision \(q\) | 16 |
| Initialization | nstart = 10 |
| Path length | 240 |
| Selection criterion | BIC |
| Refit | yes, all methods |
| Repetitions | 50 |

Scenario design:

| Scenario | 평균 방향 차이 | 집중도 차이 | Target angle | Kappa |
|---|---|---|---:|---|
| S1 | 보통 | 있음 | 90 deg | (30, 40, 50, 60) |
| S2 | 보통 | 없음 | 90 deg | (45, 45, 45, 45) |
| S3 | 작음 | 있음 | 60 deg | (30, 40, 50, 60) |
| S4 | 작음 | 없음 | 60 deg | (45, 45, 45, 45) |

Methods:

| Method | Penalty target | Group penalty | Adaptive |
|---|---|---:|---:|
| D-L | direction \(\mu_{kj}\) | no | no |
| D-GL | direction \(\mu_{\cdot j}\) | yes | no |
| D-AGL | direction \(\mu_{\cdot j}\) | yes | yes |
| E-L | centered eta entry \(c_{kj}\) | no | no |
| E-GL | centered eta coordinate \(c_{\cdot j}\) | yes | no |
| E-AGL | centered eta coordinate \(c_{\cdot j}\) | yes | yes |

평가 지표:

- `selected q`: total selected coordinates.
- `공통 선택`: 선택된 공통 변수 수. 이상적인 값은 0.
- `군집특성 선택`: 선택된 군집특성 변수 수. 이상적인 값은 16.
- `noise 선택`: 선택된 noise 변수 수. 이상적인 값은 0.
- `MSE_eta`: MSE for centered eta contrast.

## 3. S1: Moderate Mean Difference + Heterogeneous Concentration

Setting: target angle 90 deg, \(\kappa=(30,40,50,60)\).

| Method | ARI | selected q | 공통 선택 | 군집특성 선택 | noise 선택 | TPR | FPR | Precision | F1 | MSE_mu | MSE_kappa | MSE_eta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D-L | 0.837 | 199.80 | 4.00 | 16.00 | 179.80 | 1.000 | 0.999 | 0.080 | 0.148 | 0.000526 | 6.230 | 0.738 |
| D-GL | 0.863 | 20.00 | 4.00 | 16.00 | 0.00 | 1.000 | 0.022 | 0.800 | 0.889 | 0.000052 | 1.039 | 0.069 |
| D-AGL | 0.863 | 20.00 | 4.00 | 16.00 | 0.00 | 1.000 | 0.022 | 0.800 | 0.889 | 0.000052 | 1.039 | 0.069 |
| E-L | 0.837 | 198.10 | 3.96 | 16.00 | 178.14 | 1.000 | 0.990 | 0.081 | 0.149 | 0.000539 | 6.115 | 0.737 |
| E-GL | 0.865 | 17.44 | 0.02 | 16.00 | 1.42 | 1.000 | 0.008 | 0.924 | 0.959 | 0.001528 | 39.821 | 0.078 |
| E-AGL | 0.865 | 16.06 | 0.00 | 16.00 | 0.06 | 1.000 | 0.000 | 0.996 | 0.998 | 0.001520 | 41.107 | 0.057 |

해석:

- D-L and E-L select almost all variables.
- D-GL and D-AGL remove noise, but keep all 4 common variables.
- E-AGL is closest to the true decision support: selected q = 16.06, common selected = 0.00, noise selected = 0.06.

## 4. S2: Moderate Mean Difference + Equal Concentration

Setting: target angle 90 deg, \(\kappa=(45,45,45,45)\).

| Method | ARI | selected q | 공통 선택 | 군집특성 선택 | noise 선택 | TPR | FPR | Precision | F1 | MSE_mu | MSE_kappa | MSE_eta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D-L | 0.881 | 199.68 | 4.00 | 16.00 | 179.68 | 1.000 | 0.998 | 0.080 | 0.148 | 0.000425 | 5.326 | 0.708 |
| D-GL | 0.903 | 20.00 | 4.00 | 16.00 | 0.00 | 1.000 | 0.022 | 0.800 | 0.889 | 0.000042 | 1.204 | 0.070 |
| D-AGL | 0.903 | 20.00 | 4.00 | 16.00 | 0.00 | 1.000 | 0.022 | 0.800 | 0.889 | 0.000042 | 1.203 | 0.070 |
| E-L | 0.881 | 197.50 | 3.92 | 16.00 | 177.58 | 1.000 | 0.986 | 0.081 | 0.150 | 0.000450 | 5.039 | 0.707 |
| E-GL | 0.904 | 17.06 | 0.00 | 16.00 | 1.06 | 1.000 | 0.006 | 0.941 | 0.969 | 0.001387 | 41.603 | 0.072 |
| E-AGL | 0.904 | 16.12 | 0.00 | 16.00 | 0.12 | 1.000 | 0.001 | 0.993 | 0.996 | 0.001378 | 42.199 | 0.057 |

해석:

- Equal concentration does not remove the pattern seen in S1.
- Direction group methods still keep the 4 common variables.
- E-AGL again recovers the decision support almost exactly.

## 5. S3: Small Mean Difference + Heterogeneous Concentration

Setting: target angle 60 deg, \(\kappa=(30,40,50,60)\).
The actual pairwise direction angle mean/min/max is 66.02/42.07/86.06 deg, so this is the hardest setting among S1-S4.

| Method | ARI | selected q | 공통 선택 | 군집특성 선택 | noise 선택 | TPR | FPR | Precision | F1 | MSE_mu | MSE_kappa | MSE_eta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D-L | 0.546 | 199.94 | 4.00 | 16.00 | 179.94 | 1.000 | 1.000 | 0.080 | 0.148 | 0.001309 | 120.607 | 2.377 |
| D-GL | 0.613 | 37.24 | 4.00 | 16.00 | 17.24 | 1.000 | 0.115 | 0.549 | 0.677 | 0.000506 | 143.148 | 1.308 |
| D-AGL | 0.587 | 57.26 | 4.00 | 16.00 | 37.26 | 1.000 | 0.224 | 0.396 | 0.528 | 0.000598 | 110.669 | 1.299 |
| E-L | 0.544 | 199.20 | 3.94 | 16.00 | 179.26 | 1.000 | 0.996 | 0.080 | 0.149 | 0.001308 | 82.063 | 2.124 |
| E-GL | 0.609 | 44.70 | 0.60 | 15.94 | 28.16 | 0.996 | 0.156 | 0.495 | 0.618 | 0.003631 | 157.113 | 0.755 |
| E-AGL | 0.631 | 21.22 | 0.12 | 15.02 | 6.08 | 0.939 | 0.034 | 0.877 | 0.881 | 0.004147 | 234.696 | 0.250 |

해석:

- S3 is a stress-test setting; all methods lose clustering and support quality.
- E-AGL does not perfectly recover all decision variables, but it is much less dense than the alternatives.
- E-AGL gives the best ARI, F1, FPR, and MSE_eta in this setting.

## 6. S4: Small Mean Difference + Equal Concentration

Setting: target angle 60 deg, \(\kappa=(45,45,45,45)\).

| Method | ARI | selected q | 공통 선택 | 군집특성 선택 | noise 선택 | TPR | FPR | Precision | F1 | MSE_mu | MSE_kappa | MSE_eta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D-L | 0.564 | 199.80 | 4.00 | 16.00 | 179.80 | 1.000 | 0.999 | 0.080 | 0.148 | 0.000555 | 8.878 | 1.011 |
| D-GL | 0.647 | 20.24 | 4.00 | 16.00 | 0.24 | 1.000 | 0.023 | 0.791 | 0.883 | 0.000056 | 1.602 | 0.097 |
| D-AGL | 0.648 | 20.00 | 4.00 | 16.00 | 0.00 | 1.000 | 0.022 | 0.800 | 0.889 | 0.000053 | 1.608 | 0.093 |
| E-L | 0.563 | 198.88 | 4.00 | 16.00 | 178.88 | 1.000 | 0.994 | 0.080 | 0.149 | 0.000555 | 8.833 | 1.010 |
| E-GL | 0.648 | 17.76 | 0.06 | 16.00 | 1.70 | 1.000 | 0.010 | 0.908 | 0.950 | 0.003898 | 314.002 | 0.103 |
| E-AGL | 0.651 | 16.32 | 0.02 | 16.00 | 0.30 | 1.000 | 0.002 | 0.982 | 0.990 | 0.003924 | 323.765 | 0.079 |

해석:

- Even under small mean separation, equal concentration gives a cleaner pattern than S3.
- D-GL/D-AGL again retain all common variables.
- E-AGL is closest to the target decision support.

## 7. Overall Summary

| Scenario | Best decision-support method | Main pattern |
|---|---|---|
| S1 | E-AGL | Nearly exact recovery of decision support. |
| S2 | E-AGL | Same pattern as S1 under equal concentration. |
| S3 | E-AGL | Hard stress setting; E-AGL remains least dense and best in F1/MSE_eta. |
| S4 | E-AGL | Small mean difference but equal concentration; E-AGL remains closest to true q. |

핵심 해석:

- Entry-wise penalties, D-L and E-L, tend to keep almost all coordinates.
- Direction group penalties, D-GL and D-AGL, remove most noise but keep common coordinates.
- Eta group penalties, E-GL and E-AGL, directly target posterior decision support and therefore remove common coordinates.
- E-AGL is the most stable method across S1-S4 for decision-support recovery.

주의할 점:

- E-GL and E-AGL have larger MSE_kappa because they shrink the full eta norm when common coordinates are removed.
- This is expected under a centered eta contrast target.
- For this paper, MSE_eta and decision support recovery are the primary estimation targets; MSE_kappa is reported as a secondary diagnostic.

## 8. Result Files

The summary values above are taken from:

- `results/paper_eta_first_s1_angle90_kappa30_60_rep50_260702/paper_eta_first_s1_angle90_kappa30_60_rep50_260702_summary.csv`
- `results/paper_eta_first_s2_angle90_kappa45_equal_rep50_260702/paper_eta_first_s2_angle90_kappa45_equal_rep50_260702_summary.csv`
- `results/paper_eta_first_s3_angle60_kappa30_60_rep50_260702/paper_eta_first_s3_angle60_kappa30_60_rep50_260702_summary.csv`
- `results/paper_eta_first_s4_angle60_kappa45_equal_rep50_260702/paper_eta_first_s4_angle60_kappa45_equal_rep50_260702_summary.csv`

Raw result files와 run log는 연구미팅 문서 commit 대상이 아니다.
