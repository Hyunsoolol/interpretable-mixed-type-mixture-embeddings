# Eta Centered Group vs ANOVA L1 Pilot

## 목적

현재 K=4 common + component-specific 시뮬레이션 구조에서 `eta` penalty를 두 방식으로 비교했다.

- `Eta centered group`: 좌표별 centered eta contrast에 group lasso penalty를 적용한다.
- `Eta ANOVA L1`: centered ANOVA effect `alpha_kj = eta_kj - mean_k eta_kj`에 component-wise L1 penalty를 적용한다.

이 실험은 전체 논문용 100회 benchmark가 아니라, 현재 방법론 선택을 위한 rep=20 pilot이다.

## 설정

| 항목 | 값 |
|---|---:|
| K | 4 |
| n | 1000 |
| d | 100 |
| common variables | 6 |
| component-specific variables | 4 per component |
| true union q | 22 |
| repetitions | 20 per scenario |
| tuning | path candidates + BIC |
| refit | selected coordinate support 고정 후 unpenalized refit |

Scenarios:

- strong: `kappa = (30, 45, 65, 90)`
- weak: `kappa = (40, 50, 60, 70)`

## 핵심 결과

| scenario | method | ARI | selected_q | FPR | Precision | F1 | MSE_kappa | MSE_centered_eta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| strong | Eta centered group BIC | 0.641 | 25.45 | 0.046 | 0.867 | 0.925 | 11.228 | 0.379 |
| strong | Eta centered group BIC + refit | 0.684 | 25.45 | 0.046 | 0.867 | 0.925 | 1.992 | 0.191 |
| strong | Eta ANOVA L1 BIC | 0.658 | 99.90 | 0.999 | 0.220 | 0.361 | 3.311 | 0.491 |
| strong | Eta ANOVA L1 BIC + refit | 0.652 | 99.90 | 0.999 | 0.220 | 0.361 | 3.604 | 0.581 |
| weak | Eta centered group BIC | 0.552 | 23.90 | 0.024 | 0.924 | 0.960 | 6.364 | 0.346 |
| weak | Eta centered group BIC + refit | 0.565 | 23.90 | 0.024 | 0.924 | 0.960 | 1.614 | 0.172 |
| weak | Eta ANOVA L1 BIC | 0.523 | 99.50 | 0.994 | 0.221 | 0.362 | 2.530 | 0.538 |
| weak | Eta ANOVA L1 BIC + refit | 0.515 | 99.50 | 0.994 | 0.221 | 0.362 | 2.989 | 0.659 |

## 판단

현재 full EM + path BIC 조건에서는 ANOVA L1이 더 좋은 선택이라고 보기 어렵다. ANOVA L1은 null support로 무너지는 문제는 피하지만, 반대로 거의 모든 좌표를 선택한다. 이 때문에 FPR이 1에 가깝고 F1이 약 0.36 수준으로 낮다.

반대로 centered group penalty는 strong과 weak 모두에서 selected_q가 true union q=22 근처에 머물렀고, FPR은 낮으며 F1은 높았다. 특히 weak setting에서도 selected_q=23.90, FPR=0.024, F1=0.960으로 안정적이었다.

## 결론

현재 연구의 주 penalty는 `Eta centered group`으로 유지하는 것이 맞다.

ANOVA-type L1은 component-wise effect를 해석하는 대안 penalty로는 의미가 있지만, 현재 BIC path tuning과 결합하면 support가 지나치게 dense해지는 문제가 크다. 따라서 지금 단계에서 주 모형으로 교체할 근거는 없다.

## Git 반영 범위

- 요약 결과 CSV: `docs/simulations/tables/eta_group_vs_anova_l1_current_rep20_260616_summary.csv`
- 상세 raw/path candidates와 R 실행 코드는 로컬에만 둔다.
