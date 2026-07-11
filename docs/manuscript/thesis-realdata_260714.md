# 실자료 분석

## 1. 분석 목적과 자료의 역할

실자료 분석은 centered natural-parameter contrast에 기반한 posterior decision support가 문서 군집화에서 어떤 좌표를 남기는지 평가한다. E-CGL을 주 제안법으로 두고 E-ACGL은 adaptive 확장으로 보고한다.

| 자료 | 역할 | 분석 단위 | $K$ | 전처리 후 $n$ | $d$ | 평가 방식 |
|---|---|---|---:|---:|---:|---|
| Classic3 | 주 실자료 분석 | CISI, CRAN, MED 초록 | 3 | 3,890 | 2,000 | train/test = 3,111/779 |
| BBC5 | 강건성 및 희소화 비용 | BBC 5개 주제 기사 | 5 | 2,124 | 1,000 | train/test = 1,697/427 |
| CSTR | Rossi-style 문헌 재현 및 적용 한계 | 컴퓨터과학 기술보고서 초록 | 4 | 475 | 1,000 | transductive, 50회 반복 |

Classic3와 BBC5에는 SPLADE lexical expansion 좌표를 사용하였다. 층화 분할 후 SPLADE 좌표의 분산 순위는 train 문서에서만 계산하고, 고정된 train vocabulary를 test 문서에 적용하였다. BBC5는 정규화된 본문이 같은 기사 101건을 분할 전에 제거하였다. 모든 문서는 단위 $L_2$ 노름으로 정규화하였다. 제공된 주제 라벨은 분할과 ARI/NMI 계산에만 사용하였으며, 좌표 선택·초기화·모수 추정에는 사용하지 않았다.

E-CGL과 E-ACGL의 support는 train path의 각 고유 support를 exact fixed-support refit한 후 BIC로 선택하였다. M-L은 train path의 BIC-selected fit을 사용하였다. 따라서 표의 `selected q`는 M-L에서는 prototype-union support, E-CGL/E-ACGL에서는 posterior-decision support를 의미한다.

Held-out negative log-likelihood는 $-n_{\mathrm{test}}^{-1}\sum_i\log \widehat p(x_i)$로 계산하였다. vMF는 연속 밀도이므로 이 값은 음수가 될 수 있으며, 작은 값이 더 높은 test density를 뜻한다.

## 2. Classic3 주 분석

### 2.1 Held-out 성능

| 방법 | 선택 기준 | selected $q$ | support 비율 | test NLL/document | test ARI | test NMI |
|---|---|---:|---:|---:|---:|---:|
| Spherical $k$-means | cosine objective | 2,000 | 1.000 | NA | 0.9856 | 0.9710 |
| Dense vMF, shared $\kappa$ | unpenalized | 2,000 | 1.000 | -4871.6918 | 0.9856 | 0.9710 |
| Dense vMF, free $\kappa_k$ | unpenalized | 2,000 | 1.000 | **-4872.9015** | **0.9927** | **0.9863** |
| M-L | BIC before refit | 2,000 | 1.000 | -4871.0937 | 0.9892 | 0.9787 |
| **E-CGL** | exact BIC after refit | **1,347** | **0.674** | -4872.2942 | **0.9927** | **0.9863** |
| E-ACGL | exact BIC after refit | 1,348 | 0.674 | -4872.2981 | 0.9927 | 0.9863 |

E-CGL은 2,000개 후보 좌표 중 1,347개를 선택하여 32.7%를 제거하였다. Test ARI와 NMI는 dense free-$\kappa_k$ vMF와 같았고, test NLL은 문서당 0.6073 높았다. 따라서 이 결과는 predictive density의 최댓값보다 군집 배정을 유지하는 좌표 축약에 해당한다.

E-ACGL은 E-CGL과 거의 같은 support 크기와 held-out 성능을 보였다. Classic3 결과에서는 adaptive weighting의 추가 이득이 확인되지 않았으므로 E-CGL을 주 결과로 보고한다.

### 2.2 Support 안정성

Train 자료의 80%를 다시 추출하고 전체 path에서 support를 재선택하는 절차를 20회 반복하였다.

| 방법 | selected $q$, 평균 (SD) | ARI 평균 | Nogueira stability | 평균 support Jaccard |
|---|---:|---:|---:|---:|
| **E-CGL** | 1343.9 (16.8) | 0.9740 | 0.884 | 0.927 |
| E-ACGL | 1345.4 (10.2) | 0.9744 | 0.887 | 0.928 |

두 방법 모두 support 크기와 선택 좌표가 반복 표본에서 유사하게 유지되었다. 다만 선택 비율이 약 0.67이므로 Classic3 결과는 극희소 support가 아니라 안정적인 중간 밀도의 decision support로 해석한다.

## 3. 강건성 및 적용 한계

### 3.1 BBC5 held-out 분석

| 방법 | 선택 기준 | selected $q$ | support 비율 | test NLL/document | test ARI | test NMI |
|---|---|---:|---:|---:|---:|---:|
| Spherical $k$-means | cosine objective | 1,000 | 1.000 | NA | 0.8942 | 0.8719 |
| Dense vMF, shared $\kappa$ | unpenalized | 1,000 | 1.000 | -2125.6229 | 0.8942 | 0.8719 |
| Dense vMF, free $\kappa_k$ | unpenalized | 1,000 | 1.000 | **-2126.2275** | **0.8959** | **0.8736** |
| M-L | BIC before refit | 1,000 | 1.000 | -2124.8031 | 0.8889 | 0.8667 |
| **E-CGL** | exact BIC after refit | **679** | **0.679** | -2124.7525 | 0.8849 | 0.8615 |
| E-ACGL | exact BIC after refit | 691 | 0.691 | -2124.8671 | 0.8849 | 0.8615 |

E-CGL은 좌표의 32.1%를 제거했지만 dense free-$\kappa_k$ vMF보다 test ARI가 0.0109 낮고 test NLL이 문서당 1.4750 높았다. BBC5에서는 좌표 축약과 군집·밀도 적합 사이의 손실이 관찰되었다.

E-CGL과 E-ACGL의 exact-BIC margin은 각각 16.689와 1.623이었다. E-ACGL의 support 선택은 E-CGL보다 인접 후보에 민감했으며, held-out ARI 개선은 없었다.

### 3.2 CSTR 문헌 재현 및 한계

| 방법 | 단계 | selected $q$ 평균 | ARI 평균 (SD) | NMI 평균 |
|---|---|---:|---:|---:|
| Dense shared-$\kappa$ vMF | dense | 1,000.0 | 0.8023 (0.0087) | 0.7650 |
| **Rossi M-L** | penalized | 888.7 | **0.8083 (0.0079)** | **0.7703** |
| E-CGL | penalized | 311.1 | 0.6344 (0.0144) | 0.6496 |
| E-CGL | refit | 311.1 | 0.6153 (0.0065) | 0.6449 |
| E-ACGL | penalized | 313.3 | 0.6095 (0.0097) | 0.6381 |
| E-ACGL | refit | 313.3 | 0.6066 (0.0109) | 0.6401 |

현재 구현은 Rossi and Barbaro (2022)의 CSTR 결과를 근접하게 재현하였다. 논문에 보고된 ARI는 dense shared-$\kappa$ vMF 0.804, M-L 0.808이며, 현재 구현의 평균은 각각 0.8023과 0.8083이었다.

CSTR에서는 M-L이 E-CGL/E-ACGL보다 높은 ARI를 보였다. M-L은 prototype sparsity를, E-CGL은 centered-$\eta$ decision support를 추정하므로 support 수 자체는 직접적인 우열 기준이 아니다. 이 결과는 CSTR의 어휘 구조가 sparse posterior decision support보다 dense 또는 prototype-oriented support에 더 가까울 가능성을 보여준다.

## 4. $K$ 선택 진단

주 support 비교에서는 자료에 제공된 $K$를 고정하였다. Classic3 train 자료에서 dense vMF의 후보를 $K=2,\ldots,10$으로 확장한 별도 진단 결과는 다음과 같다.

| $\kappa$ 모형 | AIC | BIC | RIC | RICc | EBIC$_{0.5}$ | EBIC$_1$ |
|---|---:|---:|---:|---:|---:|---:|
| shared $\kappa$ | 10 | 10 | 9 | 8 | 8 | 8 |
| free $\kappa_k$ | 10 | 10 | 9 | 7 | 9 | 7 |

외부 주제 라벨 기준 test ARI는 $K=3$에서 가장 높았다. 반면 likelihood 기반 정보지수는 더 세분된 문서 밀도 구조를 선택하였다. 따라서 실자료에서 $K$ 선택과 support 선택을 분리하며, E-CGL이 $K$ 선택 문제를 해결한다는 주장은 하지 않는다.

## 5. 실자료 결과의 범위

Classic3에서는 E-CGL이 dense vMF와 같은 held-out 군집 성능을 유지하면서 약 3분의 1의 좌표를 제거하였다. BBC5에서는 유사한 축약이 ARI와 NLL의 감소를 동반했고, CSTR에서는 prototype-oriented M-L이 더 높은 ARI를 보였다.

따라서 실자료 결과가 뒷받침하는 범위는 다음과 같다.

1. E-CGL은 posterior decision support가 중간 수준으로 희소한 자료에서 군집 배정을 유지하며 좌표를 축약할 수 있다.
2. Predictive density와 외부 라벨 일치도에 대한 E-CGL의 효과는 자료에 따라 달랐다.
3. E-ACGL은 현재 세 실자료에서 E-CGL을 일관되게 개선하지 않았으므로 adaptive 보조 모형으로 둔다.
4. 실제 feature support의 정답이 없으므로 TPR, FPR, Precision, F1은 보고하지 않는다.
5. M-L과 E-CGL의 selected $q$는 서로 다른 estimand를 나타낸다.

## 6. 본문과 보충자료 배치

본문에는 Classic3의 held-out 결과와 support 안정성, BBC5/CSTR 핵심 비교를 제시한다. 전체 IC path, exact-refit convergence audit, BBC5 중복 제거 기록, Classic3 $K$-grid 전체 결과는 보충자료에 둔다.

근거 파일:

- `results/realdata_final_validation_260711/realdata_final_validation_summary.csv`
- `results/realdata_final_validation_260711/realdata_final_irene_audit.csv`
- `results/classic3_exact_bic_reselection_stability_b20_nstart30_260711/classic3_reselection_summary.csv`
- `results/classic3_splade_holdout_k_selection_k2_10_260711/classic3_dense_k_selection.csv`
- `results/bbc5_splade_holdout_train_exact_ic_named_260711/bbc5_exact_after_refit_ic_selection.csv`
