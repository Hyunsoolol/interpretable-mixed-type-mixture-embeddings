# Implementation Note 260624

업데이트: 2026-06-17  
기준 문서: `docs/meetings/thesis-meeting_260624.md`

이 문서는 연구미팅 260624 기준으로 구현 상태를 정리한 것이다. 목적은 교수님께 현재 구현이 어디까지 안정화됐고, 어떤 부분은 아직 diagnostic 단계인지 명확히 보여주는 것이다.

## 1. 현재 구현 요약

제안 방법 이름은 `Eta-group`으로 통일한다.

| 구분 | 현재 구현 |
|:---|:---|
| 핵심 parameter | `eta = kappa * mu` |
| 해석 | posterior decision parameter에 직접 sparsity를 부여 |
| K = 2 | eta contrast penalty |
| K > 2 | centered eta group lasso penalty |
| 구현 성격 | proximal EM-type update |

Eta-group의 목적은 ARI를 크게 올리는 것이 아니다. vMF mixture 안에서 component 간 eta contrast가 있는 coordinate를 선택해, clustering 성능을 유지하면서 해석 가능한 sparse support를 얻는 것이다.

## 2. 현재 공식 후보 알고리즘

현재 연구미팅 자료의 공식 후보는 다음 기준이다.

| 항목 | 기준 |
|:---|:---|
| 방법명 | Eta-group |
| penalty | centered eta group lasso |
| path 생성 | eta penalty path |
| tuning | path+BIC |
| refit | selected support 고정 후 unpenalized refit |
| zero support | rep ERROR가 아니라 `zero_active_support` 상태로 기록 |

`Eta-group path+BIC + refit`을 현재 main candidate로 둔다. Positive-support BIC, target refinement, adaptive refinement, stability selection은 공식 알고리즘이 아니라 diagnostic 또는 sensitivity로만 둔다.

## 3. 구현 안정화 완료 사항

완료된 안정화 작업은 다음과 같다.

| 항목 | 상태 |
|:---|:---|
| UTF-8 BOM 정리 | 완료 |
| objective trace smoke test | 완료 |
| monotone line-search safeguard | 추가 |
| objective decrease 점검 | `n_decrease = 0` 기준으로 확인 |
| 미세 objective diff | numerical tolerance 수준으로 해석 |
| zero-support refit 처리 | `zero_active_support`로 기록 |
| summary 구분 | `valid_reps`, `n_nonmissing_ARI`, `zero_support_refit_reps` 추가 |

중요한 점은 현재 update가 closed-form penalized M-step이 아니라는 것이다. 따라서 EM 단조성 자체를 이론적으로 주장하지 않고, 구현 수준에서는 line-search safeguard와 objective trace로 점검한다.

## 4. Path Candidates와 Diagnostic 저장

Eta path 후보는 별도 CSV로 저장한다.

주요 저장 컬럼은 다음과 같다.

| 범주 | 컬럼 |
|:---|:---|
| tuning | `lambda_eta`, `selected_q`, `df`, `BIC`, `EBIC` |
| 성능 | `ARI`, `TPR`, `FPR`, `Precision`, `F1` |
| 모수 | `MSE_mu`, `MSE_kappa`, `MSE_centered_eta` |
| objective trace | `objective`, `n_decrease`, `min_objective_diff`, `line_search_halving`, `line_search_accepted` |
| provenance | `path_source`, parent lambda, refinement iteration 관련 컬럼 |

Target/adaptive refinement와 stability selection은 별도 옵션으로만 사용한다. 이 결과는 path 구조를 이해하기 위한 진단 자료이며, 현재 공식 알고리즘으로 쓰지 않는다.

## 5. Tuning 관련 현재 결론

현재 결과 기준 결론은 다음과 같다.

| setting | 판단 |
|:---|:---|
| d = 100 strong | official path+BIC가 main result로 사용 가능 |
| d = 100 weak | path+BIC 결과는 안정적이나 robustness evidence로 해석 |
| d = 200 기본 path | dense support로 가기 쉬움 |
| d = 400 기본 path | high-dimensional stress limitation |
| EBIC/RIC-like 재선택 | 기본 path 후보가 부족하면 효과가 거의 없음 |
| long path 240 | d=200/d=400에서 selected q, FPR, Precision, F1 개선 |

Long path 240 diagnostic은 고차원에서 path density/range가 중요하다는 증거를 제공한다. 다만 long path만으로 true union q=22 근처 support를 안정적으로 회복하지는 못한다. 따라서 high-dimensional 보강은 path construction, MM/coordinate update, screening 쪽을 우선 후보로 둔다.

## 6. 남은 구현상 위험

아직 남은 위험은 다음과 같다.

| 위험 | 현재 해석 |
|:---|:---|
| update 이론 | proximal EM-type update라 global convergence 보장은 없음 |
| line search | objective decrease를 줄였지만 이론 보장은 아님 |
| kappa update | approximation 기반이므로 고차원에서 불안정 가능 |
| eta 복원 | `eta -> mu/kappa` 복원 시 작은 norm과 큰 kappa에 주의 필요 |
| support plateau | weak/high-dimensional path에서 중간 support 후보가 부족할 수 있음 |
| diagnostic methods | positive-support/adaptive/stability는 appendix 후보 |

특히 high-dimensional setting에서는 기본 path가 sparse support 후보를 충분히 만들지 못하는 경우가 있다. 이 문제는 단순히 BIC penalty를 강하게 하는 것만으로는 해결되지 않았다.

## 7. 연구미팅 260624에서 결정할 것

미팅에서 결정해야 할 질문은 다음이다.

1. 공식 알고리즘을 `Eta-group path+BIC + refit`으로 유지할지.
2. positive-support/adaptive/stability 결과를 appendix diagnostic으로만 둘지.
3. high-dimensional 보강을 path construction, update 개선, screening 중 어디부터 갈지.
4. 논문 본문 claim을 strong common+specific setting 중심으로 둘지.
5. high-dimensional 결과를 robustness/stress limitation으로 둘지.

현재 가장 안전한 논문 주장은 다음이다.

> Eta-group은 vMF mixture에서 posterior decision parameter인 eta의 component contrast를 sparse하게 만들어, clustering 성능을 유지하면서 해석 가능한 coordinate support를 제공한다.

