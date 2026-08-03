# CSDA 최종 시뮬레이션 결과 (rep=100)

## 1. 종합 결과

2026년 8월 3일 기준 최종 시뮬레이션, selector audit 및 fixed-support
oracle benchmark를 완료하였다.

| 검증 항목 | 결과 | 판정 |
|---|---:|---|
| Main jobs | 236/236 | PASS |
| Main method-repetition rows | 15,500/15,500 | PASS |
| Selector groups | 5,900/5,900 | PASS |
| Oracle jobs | 32/32 | PASS |
| Oracle paired rows | 1,600/1,600 | PASS |
| Missing / duplicate / unexpected key | 0 / 0 / 0 | PASS |
| ERROR row | 0 | PASS |

핵심 결과는 다음과 같다.

1. E-CGL은 대부분의 sparse posterior-score support 조건에서 참
   $q_{\eta}=16$에 가까운 support를 선택하였다.
2. $e_B=0.10$, $n=300$, heterogeneous $\kappa$에서는 E-CGL의
   $F_{1,\eta}$가 0.768로 낮아졌다. E-ACGL은 같은 조건에서 0.948이었다.
3. Common-$\kappa$에서는 M-CGL과 E-CGL의 target-specific 결과가
   유사하였다. Heterogeneous-$\kappa$에서는 두 방법이 서로 다른
   $S_{\mu}$와 $S_{\eta}$를 선택하였다.
4. 표본크기가 증가할수록 target-specific $F_1$, exact-support rate 및
   oracle benchmark gap이 개선되었다.
5. Dense support와 high-dimensional small-sample 조건에서는 exact support
   recovery가 성립하지 않았다. 이 결과는 sparse-support 방법의 적용 범위로
   유지한다.

## 2. 설계와 평가 기준

주요 DGP는 $K=4$, $d=200$, $(q_C,q_D,q_N)=(4,16,180)$으로 두었다.
Oracle Bayes error는 $e_B\in\{0.025,0.05,0.10\}$, 표본크기는
$n\in\{300,1000\}$이며, concentration 구조를 분리하여 평가하였다.

$$
\boldsymbol{\kappa}_{\mathrm{equal}}=(45,45,45,45)
$$

$$
\boldsymbol{\kappa}_{\mathrm{heterogeneous}}=(30,40,50,60)
$$

M-L은 prototype support $S_P$, M-CGL은 directional support $S_{\mu}$,
E-CGL과 E-ACGL은 posterior-score contrast support $S_{\eta}$를 기준으로
평가하였다. 서로 다른 estimand의 $F_1$은 직접적인 승패 지표로 사용하지
않았다.

모든 표의 값은 100회 평균이다. `exact`는 각 방법의 참 target support를
정확히 선택한 반복 비율이다. Path oracle은 simulation에서만 사용하는
상한 기준이며 구현 가능한 selector가 아니다.

## 3. Main posterior-score recovery

| $e_B$ | $n$ | $\kappa$ | E-CGL $q$ | E-CGL $F_{1,\eta}$ | E-CGL exact | E-CGL ARI | E-ACGL $q$ | E-ACGL $F_{1,\eta}$ | E-ACGL exact | E-ACGL ARI |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.5% | 300 | equal | 16.14 | 0.996 | 0.86 | 0.927 | 16.14 | 0.996 | 0.86 | 0.927 |
| 2.5% | 1000 | equal | 16.02 | 0.999 | 0.98 | 0.933 | 16.02 | 0.999 | 0.98 | 0.933 |
| 2.5% | 300 | heterogeneous | 16.15 | 0.996 | 0.88 | 0.925 | 16.15 | 0.996 | 0.88 | 0.925 |
| 2.5% | 1000 | heterogeneous | 16.02 | 0.999 | 0.98 | 0.928 | 16.02 | 0.999 | 0.98 | 0.928 |
| 5.0% | 300 | equal | 16.26 | 0.992 | 0.77 | 0.856 | 16.23 | 0.993 | 0.79 | 0.856 |
| 5.0% | 1000 | equal | 16.06 | 0.998 | 0.95 | 0.867 | 16.06 | 0.998 | 0.95 | 0.867 |
| 5.0% | 300 | heterogeneous | 16.60 | 0.983 | 0.69 | 0.857 | 16.18 | 0.995 | 0.85 | 0.858 |
| 5.0% | 1000 | heterogeneous | 16.02 | 0.999 | 0.98 | 0.869 | 16.02 | 0.999 | 0.98 | 0.869 |
| 10.0% | 300 | equal | 16.28 | 0.992 | 0.76 | 0.724 | 16.33 | 0.987 | 0.77 | 0.721 |
| 10.0% | 1000 | equal | 16.02 | 0.999 | 0.98 | 0.747 | 16.02 | 0.999 | 0.98 | 0.747 |
| 10.0% | 300 | heterogeneous | 24.34 | 0.768 | 0.01 | 0.638 | 15.23 | 0.948 | 0.22 | 0.702 |
| 10.0% | 1000 | heterogeneous | 16.47 | 0.986 | 0.72 | 0.753 | 16.01 | 1.000 | 0.99 | 0.753 |

E-CGL의 주 specification은 단순성과 초기 가중치 비의존성을 기준으로
유지한다. 다만 small-sample, high-overlap, heterogeneous-$\kappa$ 조건은
adaptive weighting의 이득이 확인된 대표 조건으로 별도 보고한다.

### 3.1 대표 비교 모형 결과

다음 표는 $e_B=0.05$, heterogeneous $\kappa$ 조건의 전체 비교 모형
결과이다. Target $F_1$은 M-L에 $F_{1,P}$, M-CGL에 $F_{1,\mu}$,
E-CGL과 E-ACGL에 $F_{1,\eta}$를 적용한 값이다. 서로 다른 target $F_1$의
수치만으로 방법 간 우열을 정하지 않는다. Test NLL은 작을수록 좋다.

| $n$ | 방법 | selected $q$ | target $F_1$ | ARI | test NLL | 중앙값 초/rep |
|---:|---|---:|---:|---:|---:|---:|
| 300 | Spherical $k$-means | - | - | 0.580 | - | 0.06 |
| 300 | Dense-shared | - | - | 0.649 | -245.860 | 1.20 |
| 300 | Dense-free | - | - | 0.709 | -246.082 | 2.82 |
| 300 | M-L | 199.70 | 0.182 | 0.718 | -246.122 | 9.13 |
| 300 | M-CGL | 20.73 | 0.897 | 0.812 | -247.060 | 434.47 |
| 300 | E-CGL | 16.60 | 0.983 | 0.857 | -247.206 | 50.26 |
| 300 | E-ACGL | 16.18 | 0.995 | 0.858 | -247.215 | 43.65 |
| 1000 | Spherical $k$-means | - | - | 0.773 | - | 0.25 |
| 1000 | Dense-shared | - | - | 0.765 | -247.004 | 1.59 |
| 1000 | Dense-free | - | - | 0.835 | -247.228 | 4.54 |
| 1000 | M-L | 199.67 | 0.182 | 0.836 | -247.232 | 29.05 |
| 1000 | M-CGL | 20.04 | 0.999 | 0.867 | -247.510 | 496.21 |
| 1000 | E-CGL | 16.02 | 0.999 | 0.869 | -247.517 | 46.88 |
| 1000 | E-ACGL | 16.02 | 0.999 | 0.869 | -247.517 | 42.36 |

M-L은 prototype support가 거의 dense한 이 DGP에서 평균 약 200개 좌표를
유지하였다. M-CGL과 E-CGL은 각 target에 가까운 20개와 16개 좌표를
선택하였다. 이 결과는 기존 방법의 실패가 아니라 prototype,
directional 및 posterior-score support의 차이를 반영한다.

## 4. M-CGL과 E-CGL의 estimand 분리

| 진단 | 방법 | selected $q$ | $\eta$-only | $\mu$-only | both | null | $F_{1,\mu}$ | $F_{1,\eta}$ | exact target | ARI |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Common $\kappa$ | M-CGL | 16.04 | 0.00 | 0.00 | 16.00 | 0.04 | 0.999 | 0.999 | 0.96 | 0.867 |
| Common $\kappa$ | E-CGL | 16.06 | 0.00 | 0.00 | 16.00 | 0.06 | 0.998 | 0.998 | 0.95 | 0.867 |
| Pure concentration | M-CGL | 0.77 | 0.03 | 0.00 | 0.00 | 0.74 | 0.240 | 0.004 | 0.24 | 0.674 |
| Pure concentration | E-CGL | 16.33 | 16.00 | 0.00 | 0.00 | 0.33 | 0.000 | 0.990 | 0.77 | 0.635 |
| Shared canonical | M-CGL | 21.16 | 0.00 | 1.10 | 19.99 | 0.07 | 0.348 | 0.973 | 0.00 | 0.859 |
| Shared canonical | E-CGL | 20.02 | 0.00 | 0.01 | 20.00 | 0.01 | 0.333 | 1.000 | 0.98 | 0.870 |
| Crossed support | M-CGL | 11.67 | 0.01 | 3.64 | 8.00 | 0.02 | 0.983 | 0.677 | 0.64 | 0.996 |
| Crossed support | E-CGL | 11.87 | 3.70 | 0.12 | 8.00 | 0.05 | 0.681 | 0.980 | 0.69 | 0.996 |

Common-$\kappa$에서는 $S_{\mu}=S_{\eta}$이므로 두 방법의 결과가
일치하였다. Crossed-support에서는 각 방법이 자신의 target을 회복하고
상대 target에 대한 $F_1$은 낮아졌다. 이는 두 방법의 estimand가 다름을
직접 확인한다.

Pure-concentration에서 M-CGL path는 매 반복 empty support를 포함했지만,
BIC의 exact-empty rate는 0.24였다. $\mathrm{EBIC}_{1}$은 평균
selected $q=0.03$, exact-empty rate 0.97이었다. Shared-canonical에서는
M-CGL이 참 $\mu$-only 좌표 80개 중 평균 1.10개만 선택하였다. 두 결과는
M-CGL의 path 생성과 selector가 DGP에 따라 민감할 수 있음을 보여준다.

## 5. 표본크기와 oracle benchmark

| 방법 | $\kappa$ | $n$ | selected $q$ | target $F_1$ | exact | path-oracle $F_1$ | selector gap | oracle NLL gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E-CGL | equal | 300 | 16.26 | 0.992 | 0.77 | 1.000 | 0.008 | 0.0086 |
| E-CGL | equal | 600 | 16.02 | 0.999 | 0.98 | 1.000 | 0.001 | 0.0004 |
| E-CGL | equal | 1000 | 16.06 | 0.998 | 0.95 | 1.000 | 0.002 | 0.0007 |
| E-CGL | equal | 2000 | 16.00 | 1.000 | 1.00 | 1.000 | 0.000 | 0.0000 |
| E-CGL | heterogeneous | 300 | 16.60 | 0.983 | 0.69 | 0.990 | 0.007 | 0.0139 |
| E-CGL | heterogeneous | 600 | 16.06 | 0.998 | 0.95 | 1.000 | 0.002 | 0.0011 |
| E-CGL | heterogeneous | 1000 | 16.02 | 0.999 | 0.98 | 1.000 | 0.001 | 0.0003 |
| E-CGL | heterogeneous | 2000 | 16.00 | 1.000 | 1.00 | 1.000 | 0.000 | 0.0000 |
| M-CGL | equal | 300 | 16.23 | 0.993 | 0.79 | 1.000 | 0.007 | 0.0079 |
| M-CGL | equal | 600 | 16.03 | 0.999 | 0.97 | 1.000 | 0.001 | 0.0007 |
| M-CGL | equal | 1000 | 16.04 | 0.999 | 0.96 | 1.000 | 0.001 | 0.0005 |
| M-CGL | equal | 2000 | 16.00 | 1.000 | 1.00 | 1.000 | 0.000 | 0.0000 |
| M-CGL | heterogeneous | 300 | 20.73 | 0.897 | 0.08 | 0.912 | 0.014 | 0.1366 |
| M-CGL | heterogeneous | 600 | 20.39 | 0.990 | 0.72 | 0.991 | 0.001 | 0.0044 |
| M-CGL | heterogeneous | 1000 | 20.04 | 0.999 | 0.96 | 1.000 | 0.000 | 0.0004 |
| M-CGL | heterogeneous | 2000 | 20.00 | 1.000 | 1.00 | 1.000 | 0.000 | 0.0000 |

표본크기 증가에 따라 두 방법 모두 target-specific support recovery와
oracle gap이 개선되었다. $n=2000$에서는 모든 조건에서 target $F_1=1$과
exact-support rate 1을 기록하였다. 이는 empirical recovery behavior이며
selection consistency 또는 oracle property의 이론적 증명을 의미하지 않는다.

## 6. Selector 진단

| 조건과 방법 | BIC $q$ | BIC target $F_1$ | $\mathrm{EBIC}_{1}$ $q$ | $\mathrm{EBIC}_{1}$ target $F_1$ | path-oracle $q$ | path-oracle $F_1$ |
|---|---:|---:|---:|---:|---:|---:|
| Pure concentration, M-CGL | 0.77 | 0.240 | 0.03 | 0.970 | 0.00 | 1.000 |
| $e_B=0.10$, $n=300$, heterogeneous, E-CGL | 24.34 | 0.768 | 22.99 | 0.789 | 22.07 | 0.814 |
| $e_B=0.10$, $n=300$, heterogeneous, E-ACGL | 15.23 | 0.948 | 14.53 | 0.937 | 15.45 | 0.956 |

Pure-concentration M-CGL의 문제는 path가 empty support를 만들지 못한 것이
아니라 BIC가 해당 support를 선택하지 않은 데 있다. 반면 어려운 E-CGL
cell에서는 $\mathrm{EBIC}_{1}$과 path oracle도 개선 폭이 제한적이므로
selector만의 문제로 설명할 수 없다.

## 7. 적용 범위와 한계

| 조건 | 방법 | selected $q$ | $F_{1,\eta}$ | exact | ARI |
|---|---|---:|---:|---:|---:|
| Weak-signal beta-min, $n=1000$ | E-CGL | 15.99 | 0.998 | 0.93 | 0.869 |
| Weak-signal beta-min, $n=1000$ | E-ACGL | 15.80 | 0.994 | 0.81 | 0.869 |
| High-dimensional, $n=300,d=500$ | E-CGL | 55.71 | 0.804 | 0.01 | 0.769 |
| High-dimensional, $n=300,d=500$ | E-ACGL | 36.71 | 0.889 | 0.00 | 0.767 |
| Moderately dense, $n=300$ | E-CGL | 88.40 | 0.792 | 0.00 | 0.553 |
| Moderately dense, $n=300$ | E-ACGL | 57.97 | 0.797 | 0.00 | 0.544 |
| Moderately dense, $n=1000$ | E-CGL | 87.84 | 0.890 | 0.00 | 0.674 |
| Moderately dense, $n=1000$ | E-ACGL | 67.47 | 0.913 | 0.00 | 0.679 |
| Strongly dense, $n=1000$ | E-CGL | 153.96 | 0.924 | 0.00 | 0.635 |
| Strongly dense, $n=1000$ | E-ACGL | 125.69 | 0.878 | 0.00 | 0.613 |

Dense-support 조건에서는 평균 $F_1$이 높더라도 exact-support rate는 0이었다.
E-ACGL은 high-dimensional 및 moderately dense 조건에서 false-positive
축소에 도움이 되었으나 strongly dense 조건에서는 E-CGL보다 낮았다.
Adaptive extension의 일률적 우월성은 확인되지 않았다.

## 8. 계산시간과 수렴

| 방법 | 반복 수 | 중앙값 초/rep | 평균 초/rep | 총 CPU 시간 | 최대 초/rep |
|---|---:|---:|---:|---:|---:|
| Spherical $k$-means | 2,400 | 0.18 | 0.18 | 0.1 h | 0.7 |
| Dense-shared | 2,400 | 1.41 | 1.88 | 1.3 h | 7.6 |
| Dense-free | 2,400 | 4.00 | 5.13 | 3.4 h | 20.9 |
| M-L | 2,400 | 26.06 | 24.54 | 16.4 h | 112.9 |
| E-ACGL | 2,400 | 41.20 | 48.25 | 32.2 h | 158.6 |
| E-CGL | 2,400 | 45.86 | 54.01 | 36.0 h | 200.2 |
| M-CGL | 1,100 | 312.18 | 313.57 | 95.8 h | 2,188.3 |

E-CGL, E-ACGL, M-L 및 M-CGL의 기록된 convergence rate는 1이었다.
Dense-free와 Dense-shared의 평균 convergence rate는 각각 0.9983과
0.9996이었으며 모든 결과 행은 유효하였다. Pure-concentration M-CGL의
세 반복은 최초 1,800초 제한을 넘었으나 7,200초 제한 재실행에서 모두
완료되었다.

M-CGL과 E-CGL의 실행시간은 알고리즘 구조, path 후보 수 및 estimand가
다르므로 자유도만으로 해석하지 않는다.

## 9. 논문 반영 범위

본문에는 다음 결과를 우선 배치한다.

- Main recovery에서 E-CGL의 $F_{1,\eta}$와 exact-support rate
- Common/heterogeneous $\kappa$에서 M-CGL과 E-CGL의 estimand 차이
- 표본크기별 target-specific $F_1$과 oracle NLL gap
- Dense/high-dimensional 조건의 limitation

E-ACGL 전체 결과, selector 기준 전체표, 반복별 boxplot, runtime 분포 및
전체 method-cell 결과는 보충자료에 배치한다.

수치 원본은 다음 파일에서 확인한다.

- `tables/csda_final_rep100_summary_260803.csv`
- `tables/csda_final_selector_summary_260803.csv`
- `tables/csda_final_oracle_summary_260803.csv`

대용량 raw, path candidate 및 worker log는 로컬에만 유지한다.
