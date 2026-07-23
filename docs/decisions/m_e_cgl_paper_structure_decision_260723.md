---
title: M/E-CGL Paper Structure Decision
type: decision
status: accepted
date_created: 2026-07-23
date_updated: 2026-07-23
source_of_truth: true
tags:
  - methods
  - estimand
  - simulation
  - manuscript
---

# M/E-CGL 논문 구조 최종 결정

## 1. 최종 결정

현재 논문의 방법론 구조는 다음으로 확정한다.

| 방법 | 최종 역할 | 원고 위치 |
|---|---|---|
| **E-CGL** | **주 제안 모형** | 본문 방법론·이론·주 시뮬레이션·주 실자료 |
| E-ACGL | adaptive 보조 확장 | 본문 민감도 요약, 전체 결과는 Supplement |
| **M-CGL** | centered directional-support 대응 모형 | 본문 이론 관계와 matched diagnostic |
| M-ACGL | M-CGL의 adaptive 보조 확장 | Supplement |

따라서 현재 원고에서는 다음 두 구성을 채택하지 않는다.

- E-CGL/E-ACGL만 제시하는 E-only 구조
- M-CGL과 E-CGL을 동일 비중의 공동 주제안법으로 두는 co-primary 구조

최종 구조는 다음과 같다.

$$\boxed{\text{E-CGL primary}+\text{M-CGL directional companion}+\text{adaptive extensions in secondary analyses}}$$

M-CGL은 수학적으로 유효한 모형이며 제외하지 않는다. 다만 E-CGL과 같은
추정 대상을 갖지 않고, 구현·반복실험·실자료 근거도 아직 E-CGL과 같은
수준이 아니므로 공동 주방법으로 두지 않는다.

## 2. 논문의 중심 연구 질문

vMF mixture에서

$$x_{i}\mid z_{i}=k\sim\mathrm{vMF}(\mu_{k},\kappa_{k}),\qquad \|\mu_{k}\|_{2}=1,\qquad \eta_{k}=\kappa_{k}\mu_{k}$$

이고 pairwise posterior log-odds는

$$\log\frac{\tau_{k}(x)}{\tau_{\ell}(x)}=\log\frac{\pi_{k}C_{d}(\kappa_{k})}{\pi_{\ell}C_{d}(\kappa_{\ell})}+(\eta_{k}-\eta_{\ell})^{\top}x$$

이다. 현재 논문의 중심 질문은 component 방향 자체의 희소성이 아니라,
posterior linear decision score의 component 차이를 만드는 coordinate를
선택하는 것이다.

Centering matrix를

$$H=I_{K}-\frac{1}{K}\mathbf{1}\mathbf{1}^{\top}$$

로 두면 주 추정 대상은

$$S_{E}^{*}=\{j:\|(H\eta^{*})_{\cdot j}\|_{2}>0\}$$

이다. 따라서 이 추정 대상을 직접 정규화하는

$$\widehat{\Theta}_{E\text{-CGL}}=\arg\max_{\Theta}\{\ell(\Theta)-\lambda_{\eta}\sum_{j=1}^{d}\|(H\eta)_{\cdot j}\|_{2}\}$$

를 주 제안 모형으로 둔다.

## 3. M-CGL의 수학적 위치

M-CGL은 다음의 유효한 제약 최적화 문제이다.

$$\widehat{\Theta}_{M\text{-CGL}}=\arg\max_{\Theta}\{\ell(\Theta)-\lambda_{\mu}\sum_{j=1}^{d}\|(H\mu)_{\cdot j}\|_{2}\},\qquad \mu_{k}\in\mathbb{S}^{d-1}$$

그 추정 대상은

$$S_{M}^{*}=\{j:\|(H\mu^{*})_{\cdot j}\|_{2}>0\}$$

즉 component 간 **방향 이질성 support**이다. $H\mu$는 새로운 방향 모수가
아니라 구면에 매립된 방향들의 extrinsic contrast이므로, $H\mu$ 자체가
단위구면 위에 있을 필요는 없다. 구면 제약은 M-CGL을 배제하지 않지만,
centered contrast와 행별 단위노름 제약을 결합하여 최적화를 비분리적으로
만든다.

M-CGL은 Rossi 2022 official baseline이 아니다. Rossi 계열 M-L은
component-entry prototype sparsity를 추정하며, M-CGL은 본 연구에서
정의한 centered directional-support 대응 모형이다.

## 4. M-CGL과 E-CGL의 관계

### 4.1 공통 concentration

$$\kappa_{1}=\cdots=\kappa_{K}=\kappa$$

이면

$$H\eta=H(\kappa\mu)=\kappa H\mu$$

이므로

$$\boxed{S_{M}^{*}=S_{E}^{*}}$$

이 경우 M-CGL과 E-CGL은 같은 population support를 목표로 한다. 따라서
공통 $\kappa$ 실험은 두 수치 구현의 support equivalence를 확인하는
matched diagnostic이다.

### 4.2 서로 다른 concentration

$$\eta=D_{\kappa}\mu,\qquad D_{\kappa}=\mathrm{diag}(\kappa_{1},\ldots,\kappa_{K})$$

이면 일반적으로

$$H\eta=HD_{\kappa}\mu\neq D_{\kappa}H\mu$$

따라서

$$S_{M}^{*}\neq S_{E}^{*}$$

일 수 있다. 이는 어느 한 방법의 실패가 아니라 추정 대상의 차이이다.

특히

$$\mu_{1}=\cdots=\mu_{K}=\mu,\qquad \kappa_{k}\neq\kappa_{\ell}$$

이면

$$H\mu=0$$

이지만, $\mu_{j}\neq0$인 coordinate에서는

$$(H\eta)_{\cdot j}=\mu_{j}H(\kappa_{1},\ldots,\kappa_{K})^{\top}\neq0$$

일 수 있다. M-CGL은 방향 차이가 없다고 판단하고, E-CGL은 concentration
차이가 posterior linear score에 만든 coordinate 차이를 선택한다.

## 5. 세 가지 원고 구조의 판정

| 후보 | 장점 | 핵심 문제 | 판정 |
|---|---|---|---|
| E-CGL + E-ACGL만 제시 | 가장 간결하고 기존 결과가 충분함 | centered $\mu$를 왜 비교하지 않았는지 설명이 약함 | 채택하지 않음 |
| **E-CGL 주 방법 + M-CGL 대응 모형** | 연구 질문, 교수님 피드백, 현재 증거 수준이 일치함 | 두 estimand와 평가 지표를 분리해야 함 | **채택** |
| M-CGL/E-CGL co-primary | 하나의 병렬 framework로 제시 가능 | M 계열의 수치·반복·실자료 근거가 부족하고 원고 초점이 분산됨 | 현재 원고에서는 채택하지 않음 |

E-only 구조는 유효한 $H\mu$ 대안을 생략한다. 반면 co-primary 구조는 서로
다른 두 추정 목표를 같은 문제의 경쟁 해법처럼 보이게 하고, 현재 근거의
비대칭을 가린다. 채택한 구조는 E-CGL의 연구 질문을 유지하면서 M-CGL로
parameterization 차이를 직접 검증한다.

## 6. 실제 근거

### 6.1 M/E matched diagnostic

| 설정 | 저장 결과 | 해석 |
|---|---|---|
| common $\kappa=(32,32,32,32)$, rep=5 | 네 방법 모두 selected $q=8$, target F1=1.000 | $S_{M}^{*}=S_{E}^{*}$ 확인 |
| heterogeneous $\kappa=(22,28,34,40)$, rep=5 | M-CGL은 $q=8$, $F_{1,M}=1.000$; E-CGL은 $q=10.6$, $F_{1,E}=0.934$ | 서로 다른 estimand를 회복하는 방향 확인 |
| Study-B-aligned $\kappa=(30,40,50,60)$, rep=5 | M-CGL은 $q=8$, E-CGL은 $q=12$; 각 target F1=1.000 | 방향 좌표 8개와 concentration-driven 좌표 4개가 분리됨 |
| same $\mu$/different $\kappa$, rep=3 | truth-blind BIC에서 M 계열 $q=0$--1, E 계열 $q=10$--11 | concentration-only 차이의 방향 확인; 최종 반복 근거는 아님 |

모든 matched 결과는 각 방법을 자기 population target으로 평가한다.
$F_{1,M}$과 $F_{1,E}$의 교차 비교는 방법의 우열이 아니라 estimand 차이를
설명하는 보조 지표이다.

### 6.2 E-CGL의 확정 근거

Study B의 12개 cell은 각 100회, 총 7,200개 방법별 fit으로 완료되었고
오류 행은 없었다.

| 표본크기 | E-CGL F1 범위 | E-CGL selected $q$ 범위 | E-CGL noise $q$ 범위 |
|---:|---:|---:|---:|
| $n=300$ | 0.838--0.998 | 16.07--19.34 | 0.06--4.56 |
| $n=1000$ | 0.991--1.000 | 16.00--16.29 | 0.00--0.26 |

E-ACGL의 전체 F1 범위는 0.951--1.000이었다. 어려운
$n=300$, heterogeneous $\kappa$, oracle Bayes error 10% cell에서는
E-CGL보다 잡음 선택을 줄였지만, 다른 cell과 Classic3에서 일관된 개선은
없었다. 따라서 E-ACGL은 adaptive 보조 확장으로 유지한다.

Shared-background rep=50에서는 true decision $q=20$에 대해 E-CGL이
평균 selected $q=20.02$, common $q=0$, noise $q=0.02$,
F1=0.9995를 기록하였다. 반면 dense-support negative control에서는
E-CGL F1이 0.244--0.994로 변하여 sparse decision-support 가정의 한계도
확인되었다.

### 6.3 실자료와 계산 근거

Classic3에서 E-CGL은 2,000개 좌표 중 1,347개를 선택하여 32.7%를
제거하면서 dense free-$\kappa_{k}$ vMF와 같은 test ARI 0.9927을
유지하였다. 반복 재선택의 Nogueira stability는 0.884였다. E-ACGL은
support 크기와 held-out 성능을 개선하지 않았다.

M-CGL/M-ACGL의 Classic3 matched 결과는 아직 없다. 현재 M 계열은
R-only constrained diagnostic이고 E 계열은 package Rcpp backend이므로,
matched rep=5에서 관찰된 약 9--24배의 실행시간 차이는 알고리즘 자체의
속도 비교가 아니라 현재 구현 비용으로만 해석한다.

## 7. 수치 검증 수준

Truth-blind fixed-support retry audit에서는 다음이 확인되었다.

- candidate 94개, 고유 support 78개
- refit 369회 중 359회가 수치 적격
- 78개 support 모두 적어도 하나의 적격 refit 확보
- M row-norm 오차 $2.22\times10^{-16}$
- M inactive equality 오차 0
- E constraint 오차 $1.11\times10^{-16}$
- IC 재계산 오차 $9\times10^{-11}$ 이하

이는 후보별 refit 가능성과 IC 재계산에 대한 조건부 PASS이다. 그러나
같은 support의 적격 로그우도 범위는 M에서 25.90, E에서 37.44였고,
truth-blind retry 후 12개 BIC 선택 중 6개가 바뀌었다. 따라서 이 감사가
global optimum, selector consistency 또는 일반적 수렴을 의미하지는 않는다.

현재 증거 수준은 다음과 같이 비대칭적이다.

| 항목 | E-CGL/E-ACGL | M-CGL/M-ACGL |
|---|---|---|
| 구현 | package production 경로 | constrained diagnostic |
| 반복 simulation | Study B 12-cell rep=100 | matched rep=5, 경계 rep=3 |
| 고차원·negative control | 완료 | 제한적 |
| 실자료 | Classic3, BBC5, CSTR | matched 분석 없음 |
| Rcpp | production helper와 R fallback | sphere prototype 미연결 |
| 수치 검증 | package test와 final validation | Gate 0 조건부 PASS |

이 비대칭이 M/E co-primary 구조를 채택하지 않는 직접적인 실증 근거이다.

## 8. 검토 결과

| 검토 관점 | 판정 |
|---|---|
| 이론·수학 | posterior decision support에는 E-CGL이 직접 대응하며 M-CGL은 별도 directional estimand |
| 시뮬레이션 | E-CGL의 rep=100 근거와 M-CGL의 rep=5/3 근거를 동일 비중으로 둘 수 없음 |
| 계산·최적화 | M-CGL은 유효하지만 비분리 제약 최적화와 selector 민감성 때문에 현재 diagnostic 단계 |
| 저널 편집·리뷰 | E-only는 비교 누락 위험, co-primary는 초점과 근거 불균형 위험; companion 구조가 가장 명확함 |

각 기준은 `E-CGL primary + M-CGL companion` 구조를 지지한다.

외부 구성안의 세부 제안은 다음과 같이 반영한다.

| 제안 | 판정 | 원고 반영 |
|---|---|---|
| estimand를 estimator보다 먼저 정의 | 수용 | $S_{E}^{*}$와 $S_{M}^{*}$를 방법론보다 먼저 정의 |
| E-CGL 주방법, M-CGL directional companion | 수용 | 최종 원고 구조로 유지 |
| common-$\kappa$ equivalence와 heterogeneous-$\kappa$ divergence | 수용 | 독립 이론 절로 구성 |
| exact centered-support refit 강조 | 수용 | support 제약과 공통 baseline을 구분하여 기술 |
| adaptive 방법을 Supplement 중심으로 배치 | 수용 | 본문에는 정의와 핵심 민감도만 제시 |
| EBIC를 주 선택 기준으로 사용 | 수정 | 현재 구현과 확정 결과에 맞춰 BIC-after를 주 기준, EBIC를 민감도로 사용 |
| proximal EM의 단조증가·수렴을 보장 | 수정 | proximal EM-type update와 step-halving safeguard로 기술; global convergence는 주장하지 않음 |
| M-CGL이 본질적으로 매우 불안정 | 기각 | 유효한 제약 모형이며, 현재 구현에서 결합 최적화와 selector 민감성이 관찰되었다고 기술 |
| posterior decision 전체를 직접 선택 | 수정 | posterior log-odds의 **선형 좌표항**을 만드는 support로 한정 |
| 새로운 estimand라고 단정 | 보류 | 문헌 검토 전에는 “정의하고 직접 추정한다”로 기술 |
| 완벽한 filtering 또는 압도적 성능 | 기각 | cell별 수치와 적용 조건을 함께 보고 |
| 현재 상태로 상위 저널 준비가 완료됨 | 기각 | M 반복 검증, matched 실자료, IC 민감도와 원고화가 남아 있음 |

## 9. 최종 원고 구성

### 9.1 가제

**Centered Natural-Parameter Group Penalties for Posterior Decision Support in
High-Dimensional vMF Mixtures**

이 제목은 가제이며, “posterior decision support”의 선행 정의와 사용례를
문헌 검토한 뒤 신규성 표현의 범위를 확정한다.

### 9.2 본문

1. **Introduction**
   - 고차원 방향자료의 model-based clustering 문제
   - sparse prototype support와 posterior linear decision-score support의 구분
   - E-CGL의 연구 질문과 기여 범위
2. **vMF mixture and estimands**
   - vMF mixture와 pairwise posterior log-odds
   - 주 추정 대상 $S_{E}^{*}$
   - 방향성 대응 추정 대상 $S_{M}^{*}$
3. **E-CGL methodology**
   - centered natural-parameter group penalty
   - proximal EM-type update와 step-halving safeguard
   - exact centered-support refit
   - BIC-after 주 선택 규칙, EBIC 민감도, practical df approximation
4. **Relation to M-CGL**
   - M-CGL의 유효한 구면 제약 목적함수
   - common-$\kappa$ support equivalence
   - heterogeneous-$\kappa$ estimand divergence
   - 결합 제약이 만드는 계산상의 차이
5. **Simulation studies**
   - oracle Bayes error 기반 Study B
   - M/E matched estimand diagnostic
   - shared-background와 dense/weak negative control
6. **Real-data analysis**
   - Classic3 E-CGL 주 분석
   - 선택 좌표와 signed centered-$\eta$ contrast 해석
   - M-CGL은 matched 검증 완료 후 보조 결과로 포함
7. **Discussion**
   - estimand별 적용 범위
   - adaptive 확장과 tuning 민감도
   - dense support, weak signal, $K$ 선택의 한계

E-CGL의 최적화는 자연모수의 유클리드 공간에서 수행되지만 mixture
likelihood 자체는 비볼록이다. 따라서 계산상 장점은 “제약 없는 proximal
구조”로 기술하며 global optimum이나 일반적 수렴을 주장하지 않는다.

### 9.3 Supplement

- E-ACGL과 M-ACGL의 전체 결과
- M-CGL ADMM·product-of-spheres 계산 상세와 수치 잔차
- M/E matched simulation 전체 표
- same-$\mu$/different-$\kappa$ 경계 진단
- runtime과 selector sensitivity
- dense/weak negative-control 전체 결과

### 9.4 모형 선택과 자유도 서술

주 선택 규칙은 현재 확정 결과와 일치하는 BIC-after-refit이다. EBIC는
고차원 선택 민감도로 보고하며, 자료 근거 없이 주 선택 규칙으로 승격하지
않는다. 자유도는 exact effective degrees of freedom가 아니라 비정규
penalized mixture에서 사용하는 practical model-selection approximation으로
명시한다. 엄밀한 자유도를 일반적인 open problem이라고 단정하지 않고,
본 연구에서 도출하지 않았다는 범위로 한정한다.

## 10. 원고에서 사용할 주장

사용 가능한 주장은 다음과 같다.

1. E-CGL은 centered natural-parameter contrast를 통해 posterior log-odds의
   선형 좌표항을 만드는 support를 직접 대상으로 한다.
2. M-CGL은 수학적으로 유효한 centered directional-support 대응 모형이다.
3. 공통 $\kappa$에서는 두 population support가 일치한다.
4. 서로 다른 $\kappa_{k}$에서는 두 estimand가 달라질 수 있다.
5. E-CGL은 지정된 sparse-decision Study B와 Classic3에서 support 축약
   근거를 보였으며, dense/weak setting에서는 한계가 확인되었다.
6. Adaptive 방법은 기본 방법을 일관되게 개선하지 않아 보조 확장으로 둔다.

다음 주장은 사용하지 않는다.

- M-CGL은 수학적으로 성립하지 않는다.
- E-CGL이 모든 환경에서 M 계열보다 우월하다.
- M-CGL과 E-CGL은 같은 support를 추정한다.
- M/E co-primary 구조가 실증적으로 확립되었다.
- adaptive penalty가 항상 기본 penalty보다 우월하다.
- 현재 optimizer가 global optimum 또는 일반적 수렴을 보장한다.
- 현재 BIC 자유도가 exact effective degrees of freedom이다.
- M-CGL이 구면 제약 때문에 본질적으로 불안정하거나 성립하지 않는다.
- posterior decision support가 prototype support보다 일반적으로 우월하다.
- 문헌 검토 없이 posterior decision support를 새로운 estimand라고 단정한다.

## 11. 남은 필수 검증

현재 논문 구조는 이 결정으로 고정한다. 아래 작업은 M-CGL을 공동 주방법으로
승격하기 위한 조건이 아니라, 본문의 directional companion 결과를
신뢰할 수 있게 보고하기 위한 최소 검증이다.

1. M fit별 명시적 seed와 실행 순서 독립성 확보
2. KKT/score residual, outer·ADMM·sphere convergence,
   $\min_{k}\widehat{\pi}_{k}$, $\min_{k}N_{k}$ 기록
3. common-$\kappa$, heterogeneous-$\kappa$,
   same-$\mu$/different-$\kappa$ 각각 rep=20
4. 모든 후보 support에 동일한 truth-blind multistart refit 적용
5. BIC/EBIC 및 $m=0$ df 민감도 확인
6. Classic3에서 동일 split·초기값을 사용한 M-CGL/E-CGL matched pilot
7. 동일 backend 또는 명시된 계산 예산에서 runtime 재측정

위 검증 결과가 부정적이면 M-CGL의 본문 결과 비중을 줄이고 Supplement
diagnostic으로 이동한다. 그렇더라도 E-CGL을 주 제안 모형으로 두는 현재
논문 구조는 유지한다.

## 12. 근거 파일

- [M/E parallel plan](../planning/m_e_cgl_parallel_paper_plan_260722.md)
- [M/E implementation inventory](../planning/m_e_model_implementation_inventory_260722.md)
- [Matched rep=5 notes](../../results/m_e_cgl_matched_rep5_combined_260723/m_e_cgl_matched_rep5_notes.md)
- [Study-B-aligned heterogeneous notes](../../results/m_e_cgl_matched_studyb_heterogeneous_kappa_rep5_260723/m_e_cgl_matched_notes.md)
- [Truth-blind retry audit](../../results/m_e_cgl_truth_blind_retry_ordered_v2_260723/truth_blind_retry_notes.md)
- [Study B rep=100 validation](../../results/studyb_all_model_final_rep100_260717/studyb_final_all12_final_validation_notes.md)
- [Study B rep=100 summary](../../results/studyb_all_model_final_rep100_260717/studyb_final_all12_summary.csv)
- [Real-data results](../manuscript/thesis-realdata_260714.md)
- [Claim-evidence matrix](../manuscript/claim_evidence_matrix_260714.md)
