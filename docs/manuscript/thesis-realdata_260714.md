# 실자료 분석 결과: Classic3 주 분석과 적용 범위

## 1. 핵심 결과

실자료의 주 분석은 **Classic3 SPLADE top-2000**으로 구성하였다. E-CGL은 2,000개 좌표 중 1,347개를 선택하여 32.7%를 제거하면서 dense free-$\kappa_k$ vMF와 같은 test ARI 0.9927과 NMI 0.9863을 유지하였다. 반복 재선택에서 Nogueira stability는 0.884였다.

따라서 Classic3 결과는 E-CGL이 군집 배정을 유지하면서 posterior decision coordinate를 안정적으로 축약한 사례로 보고한다. 선택 비율은 67.4%이므로 극희소 support가 아니라 **중간 밀도의 decision support**로 해석한다. E-ACGL은 E-CGL과 거의 같은 결과를 보여 adaptive 보조 모형으로 둔다.

| 자료 | 문서 내 역할 | 핵심 결과 |
|---|---|---|
| **Classic3** | **주 실자료 분석** | E-CGL이 좌표 32.7%를 제거하고 dense vMF와 같은 test ARI 유지 |
| BBC5 | 부록: 강건성 및 희소화 비용 | 좌표 축약과 함께 ARI와 test density가 감소 |
| CSTR | 부록: 문헌 재현 및 적용 한계 | Rossi M-L이 E-CGL/E-ACGL보다 높은 ARI 기록 |

## 2. Classic3 주 분석

### 2.1 분석 설계

Classic3는 CISI, CRAN, MED 초록 3,890건으로 구성된다. SPLADE lexical expansion 좌표 중 train 자료에서 분산이 큰 2,000개를 선택하고, 층화 분할한 train 3,111건과 test 779건에 같은 vocabulary를 적용하였다. 모든 문서는 단위 $L_2$ 노름으로 정규화하였다.

제공된 주제 라벨은 train/test 분할과 ARI/NMI 평가에만 사용하였으며, 좌표 선택·초기화·모수 추정에는 사용하지 않았다. E-CGL과 E-ACGL은 train path의 각 고유 support를 exact fixed-support refit한 후 BIC로 선택하였다.

### 2.2 Held-out 성능

| 방법 | 선택 기준 | selected $q$ | support 비율 | test NLL/document | test ARI | test NMI |
|---|---|---:|---:|---:|---:|---:|
| Spherical $k$-means | cosine objective | 2,000 | 1.000 | NA | 0.9856 | 0.9710 |
| Dense vMF, shared $\kappa$ | unpenalized | 2,000 | 1.000 | -4871.6918 | 0.9856 | 0.9710 |
| Dense vMF, free $\kappa_k$ | unpenalized | 2,000 | 1.000 | **-4872.9015** | **0.9927** | **0.9863** |
| M-L | BIC before refit | 2,000 | 1.000 | -4871.0937 | 0.9892 | 0.9787 |
| **E-CGL** | exact BIC after refit | **1,347** | **0.674** | -4872.2942 | **0.9927** | **0.9863** |
| E-ACGL | exact BIC after refit | 1,348 | 0.674 | -4872.2981 | 0.9927 | 0.9863 |

E-CGL의 test NLL은 dense free-$\kappa_k$ vMF보다 문서당 0.6073 높았다. 이는 predictive density의 최댓값보다 군집 배정을 유지하는 좌표 축약에 해당한다. E-ACGL은 support 크기와 held-out 성능에서 E-CGL을 개선하지 않았다.

Held-out negative log-likelihood는 $-n_{\mathrm{test}}^{-1}\sum_i\log \widehat p(x_i)$로 계산하였다. vMF는 연속 밀도이므로 값이 음수일 수 있으며, 작은 값이 더 높은 test density를 뜻한다.

### 2.3 Support 안정성

Train 자료의 80%를 다시 추출하고 전체 path에서 support를 재선택하는 절차를 20회 반복하였다.

| 방법 | selected $q$, 평균 (SD) | ARI 평균 | Nogueira stability | 평균 support Jaccard |
|---|---:|---:|---:|---:|
| **E-CGL** | 1343.9 (16.8) | 0.9740 | 0.884 | 0.927 |
| E-ACGL | 1345.4 (10.2) | 0.9744 | 0.887 | 0.928 |

두 방법 모두 support 크기와 선택 좌표가 반복 표본에서 유사하게 유지되었다. Classic3에서는 E-CGL을 주 결과로 보고하고 E-ACGL은 adaptive sensitivity로 제시한다.

## 3. 부록 A. 적용 범위 점검

### A.1 BBC5: 좌표 축약에 따른 성능 비용

BBC5는 중복 기사 101건을 분할 전에 제거한 2,124건의 5개 주제 기사 자료다. Train 1,697건에서 선택한 SPLADE top-1,000 좌표를 test 427건에 적용하였다.

| 방법 | selected $q$ | test NLL/document | test ARI | test NMI |
|---|---:|---:|---:|---:|
| Dense vMF, free $\kappa_k$ | 1,000 | **-2126.2275** | **0.8959** | **0.8736** |
| M-L | 1,000 | -2124.8031 | 0.8889 | 0.8667 |
| E-CGL | **679** | -2124.7525 | 0.8849 | 0.8615 |
| E-ACGL | 691 | -2124.8671 | 0.8849 | 0.8615 |

E-CGL은 좌표의 32.1%를 제거했지만 dense free-$\kappa_k$ vMF보다 test ARI가 0.0109 낮고 test NLL이 문서당 1.4750 높았다. BBC5에서는 군집 구분 정보가 선택된 centered-$\eta$ support에 충분히 집중되지 않아, 좌표 축약이 군집·밀도 적합의 손실을 동반한 것으로 해석된다. E-ACGL도 이 손실을 줄이지 못했다.

### A.2 CSTR: prototype-oriented support가 유리한 사례

CSTR은 475개 컴퓨터과학 기술보고서 초록을 1,000개 이진 단어 좌표로 표현한 자료다. Rossi and Barbaro (2022)의 분석 조건에 맞춰 50회 반복하였다.

| 방법 | 단계 | selected $q$ 평균 | ARI 평균 (SD) | NMI 평균 |
|---|---|---:|---:|---:|
| Dense shared-$\kappa$ vMF | dense | 1,000.0 | 0.8023 (0.0087) | 0.7650 |
| **Rossi M-L** | penalized | 888.7 | **0.8083 (0.0079)** | **0.7703** |
| E-CGL | refit | 311.1 | 0.6153 (0.0065) | 0.6449 |
| E-ACGL | refit | 313.3 | 0.6066 (0.0109) | 0.6401 |

현재 구현은 논문의 dense shared-$\kappa$ vMF ARI 0.804와 M-L ARI 0.808을 각각 0.8023과 0.8083으로 근접하게 재현하였다. 반면 E-CGL은 평균 311.1개 좌표를 선택한 뒤 ARI가 0.6153으로 감소하였다.

이 결과는 CSTR의 이진 어휘 구조가 sparse posterior decision support보다 dense 또는 prototype-oriented support와 더 잘 맞는다는 해석과 일관된다. 또한 $n=475$에 비해 $d=1{,}000$인 조건은 centered contrast support 추정에 불리하게 작용할 수 있다.

### A.3 해석 범위

| 자료 | 관찰된 한계 | 결과가 시사하는 구조 |
|---|---|---|
| BBC5 | 좌표 축약 후 ARI·NLL 감소 | decision 정보가 여러 좌표에 분산된 중간·조밀 support 가능성 |
| CSTR | E-CGL/E-ACGL의 강한 축약과 ARI 감소 | sparse prototype 또는 dense 어휘 support가 더 적합할 가능성 |

두 자료 모두 true feature support를 제공하지 않는다. 따라서 위 설명은 관찰된 성능 차이에 대한 구조적 해석이며, 자료 생성 원인을 식별한 결과는 아니다.

## 4. 부록 B. $K$ 선택 진단

주 support 비교에서는 자료에 제공된 $K$를 고정하였다. Classic3 train 자료에서 dense vMF의 후보를 $K=2,\ldots,10$으로 확장한 결과는 다음과 같다.

| $\kappa$ 모형 | AIC | BIC | RIC | RICc | EBIC$_{0.5}$ | EBIC$_1$ |
|---|---:|---:|---:|---:|---:|---:|
| shared $\kappa$ | 10 | 10 | 9 | 8 | 8 | 8 |
| free $\kappa_k$ | 10 | 10 | 9 | 7 | 9 | 7 |

외부 주제 라벨 기준 test ARI는 $K=3$에서 가장 높았지만 likelihood 기반 정보지수는 더 큰 $K$를 선택하였다. 이에 따라 실자료 분석에서는 $K$ 선택과 support 선택을 구분하며, E-CGL의 기여 범위에 $K$ 선택을 포함하지 않는다.

## 5. 보고 범위와 근거

실자료 결과가 뒷받침하는 범위는 다음과 같다.

1. Classic3에서 E-CGL은 held-out 군집 성능을 유지하면서 좌표를 안정적으로 축약하였다.
2. BBC5와 CSTR에서는 좌표 축약이 군집 또는 밀도 적합의 손실을 동반하였다.
3. E-ACGL은 세 자료에서 E-CGL을 일관되게 개선하지 않아 adaptive 보조 모형으로 둔다.
4. 실제 feature support의 정답이 없으므로 TPR, FPR, Precision, F1은 보고하지 않는다.
5. M-L과 E-CGL의 selected $q$는 각각 prototype-union support와 posterior-decision support를 나타낸다.

근거 파일:

- `results/realdata_final_validation_260711/realdata_final_validation_summary.csv`
- `results/realdata_final_validation_260711/realdata_final_irene_audit.csv`
- `results/classic3_exact_bic_reselection_stability_b20_nstart30_260711/classic3_reselection_summary.csv`
- `results/classic3_splade_holdout_k_selection_k2_10_260711/classic3_dense_k_selection.csv`
- `results/bbc5_splade_holdout_train_exact_ic_named_260711/bbc5_exact_after_refit_ic_selection.csv`
