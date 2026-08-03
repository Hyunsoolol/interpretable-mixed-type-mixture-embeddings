# CSDA 투고 논문 구성안

업데이트: 2026-08-03

이 문서는 논문 본문이 아니라 장별 목적, 포함 결과, 본문과 보충자료의
경계를 정한 목차 문서이다.

## 논문 범위

| 방법 | 추정 대상 | 논문 내 역할 |
|---|---|---|
| E-CGL | posterior-score contrast support $S_{\eta}$ | 주 제안 방법 |
| E-ACGL | adaptive centered-$\eta$ support | 보조 확장 |
| M-CGL | centered directional support $S_{\mu}$ | matched directional comparator |
| M-L | prototype support $S_P$ | Rossi 문헌 비교 |
| Dense vMF | 전체 좌표를 사용하는 density model | 비정규화 기준 |

주 연구 질문은 다음과 같다.

$$
S_{\eta} = \{j:\|\boldsymbol{\eta}_{\cdot j}-\bar{\eta}_{j}\mathbf{1}_{K}\|_2>0\}
$$

즉, 모든 성분쌍의 posterior log-score 비교에 실제로 들어가는 좌표를
추정하는 것이 논문의 중심이다. M-CGL은 별도 주방법이 아니라
$S_{\mu}$와 $S_{\eta}$가 달라지는 원인을 확인하는 비교 방법으로 둔다.

## 구성 근거

### 참고한 CSDA 논문

| 논문 | 본문 전개 |
|---|---|
| Park and Seo (2025), *Penalized maximum likelihood estimation with nonparametric Gaussian scale mixture errors*, CSDA 211, 108206 | Introduction → Model and methodology → Asymptotic properties → Numerical examples → Discussion |
| Yuan, Jin, and Li (2024), *FDR control for linear log-contrast models with high-dimensional compositional covariates*, CSDA 197, 107973 | Introduction → Methodology → Theoretical properties → Numerical simulations → Applications → Conclusion and discussion |

- Park and Seo의 구성처럼 모형, penalty, 추정 알고리즘을 하나의 방법론
  절에서 연결한다.
- Yuan, Jin, and Gaorong Li의 구성처럼 이론, 시뮬레이션, 실자료를
  분리하여 각 절의 검증 목적을 명확히 한다.
- CSDA Guide for Authors에 따라 본문과 하위 절을 번호화하고, 표와 그림은
  처음 인용되는 내용 가까이에 배치한다.
- 알고리즘 언어, tolerance, 전체 path 결과와 반복별 진단은 보충자료로
  이동하고 본문에는 재현에 필요한 계산 원리만 둔다.

참고 문헌:

- Park and Seo (2025): <https://doi.org/10.1016/j.csda.2025.108206>
- Yuan, Jin, and Li (2024): <https://doi.org/10.1016/j.csda.2024.107973>
- CSDA Guide for Authors:
  <https://www.sciencedirect.com/journal/computational-statistics-and-data-analysis/publish/guide-for-authors>

### 채택한 전체 흐름

$$
\text{문제와 estimand}
\rightarrow
\text{방법론과 계산}
\rightarrow
\text{이론적 성질}
\rightarrow
\text{수치실험}
\rightarrow
\text{실자료}
\rightarrow
\text{토의}
$$

본문은 6개 절로 구성한다. Introduction에는 세부 절을 두지 않고,
방법론 이후부터 필요한 범위에서만 2단계 하위 절을 사용한다.

## Front matter

### Title

- 방법과 추정 대상을 함께 드러내는 서술형 제목
- 약어와 수식을 제목에서 사용하지 않음
- 현재 작업 제목:
  *Centered Natural-Parameter Regularization for Posterior-Score
  Coordinate Selection in von Mises-Fisher Mixtures*

### Abstract

초록은 다음 다섯 요소만 포함한다.

1. 고차원 방향성 군집화의 좌표 선택 문제
2. prototype support와 posterior-score support의 차이
3. centered natural-parameter group penalty와 추정 알고리즘
4. 핵심 이론 결과
5. 시뮬레이션과 실자료의 대표 결과

초록에는 세부 tuning 값, 전체 비교 방법 목록, 보충 진단을 넣지 않는다.

### Keywords and highlights

- Keywords: directional data, finite mixture, natural parameter,
  group regularization, posterior score, variable selection
- Highlights: 3--5개, 각 85자 이내
- Highlights는 문제, estimand, 알고리즘, 대표 검증 결과 순으로 구성

## 1. Introduction

Introduction은 하위 절 없이 다음 문단 순서로 작성한다.

1. 방향성 자료와 vMF mixture의 활용 범위
2. 고차원 환경에서 좌표 축약과 해석이 필요한 이유
3. 기존 sparse vMF가 주로 prototype presence를 선택한다는 점
4. posterior component comparison은
   $\boldsymbol{\eta}_{k}=\kappa_k\boldsymbol{\mu}_{k}$의 차이에 의해
   결정된다는 문제
5. equality- 또는 heterogeneity-based selection 문헌과의 연결
6. 기존 문헌에서 남는 통계적·계산적 공백
7. 본 논문의 기여 3--4개
8. 논문 구성

서론에서는 $S_P$, $S_{\mu}$, $S_{\eta}$를 공식 정의하지 않는다.
E-CGL을 주방법으로 제시하며 M-CGL의 알고리즘을 독립 기여로 열거하지
않는다.

## 2. Model and methodology

### 2.1 von Mises-Fisher mixtures in natural parameters

- vMF density와 mixture likelihood
- $\|\boldsymbol{\mu}_{k}\|_2=1$, $\kappa_k\geq0$
- 자연모수 관계

$$
\boldsymbol{\eta}_{k} = \kappa_k\boldsymbol{\mu}_{k}
$$

### 2.2 Posterior-score contrast support

- pairwise posterior log-score
- feature-dependent 선형항과 intercept의 구분
- posterior-score contrast support의 공식 정의

$$
\log\frac{\Pr(Z=k\mid\boldsymbol{x})}{\Pr(Z=\ell\mid\boldsymbol{x})} = a_{k\ell}+(\boldsymbol{\eta}_{k}-\boldsymbol{\eta}_{\ell})^{\mathsf T}\boldsymbol{x}
$$

$$
c_{kj}^{(\eta)} = \eta_{kj}-\bar{\eta}_{j}
$$

$$
S_{\eta} = \{j:\|\boldsymbol{c}_{j}^{(\eta)}\|_2>0\}
$$

### 2.3 Centered-$\eta$ coordinate group lasso

- centered-$\eta$ parameterization
- coordinate-wise group penalty
- 공통 baseline은 penalty 대상에서 제외

$$
P_{\mathrm{E-CGL}}(\boldsymbol{\eta}) = \lambda_{\eta}\sum_{j=1}^{d}\|\boldsymbol{c}_{j}^{(\eta)}\|_2
$$

- E-CGL 목적함수
- 선택 좌표의 의미
- nonzero prototype 선택과의 차이

### 2.4 Adaptive extension

- adaptive weight 정의
- $\gamma=1$과 median normalization
- E-CGL의 보조 확장으로만 제시
- E-ACGL이 항상 개선된다는 주장을 두지 않음

### 2.5 Prototype and directional comparators

세 비교 대상의 estimand를 한 표에서 구분한다.

$$
S_P = \{j:\|\boldsymbol{\mu}_{\cdot j}\|_2>0\}
$$

$$
c_{kj}^{(\mu)} = \mu_{kj}-\bar{\mu}_{j}
$$

$$
S_{\mu} = \{j:\|\boldsymbol{c}_{j}^{(\mu)}\|_2>0\}
$$

- M-L: prototype support
- M-CGL: directional heterogeneity support
- E-CGL/E-ACGL: posterior-score contrast support
- M-CGL의 목적함수와 단위구면 제약만 본문에 제시
- M-CGL의 ADMM 및 manifold 반복식은 보충자료에 배치

### 2.6 Guarded proximal generalized-EM estimation

본문 Algorithm 1은 E-CGL/E-ACGL만 대상으로 한다.

1. finite-concentration dense multiple-start initialization
2. E-step
3. guarded proximal generalized M-step
4. centered group proximal update
5. step-halving과 accepted-objective 검사
6. warm-start regularization path
7. candidate support 저장

Rcpp는 구현 세부로만 언급한다. 알고리즘 정의에 특정 프로그래밍 언어를
포함하지 않는다.

### 2.7 Unpenalized refit under selected contrast constraints

- 선택되지 않은 좌표에서 centered contrast만 0으로 제한
- 공통 natural-parameter baseline 유지
- support-constrained refit
- BIC-after-refit
- 명목 모형 차원은 exact effective degrees of freedom이 아님을 명시
- `exact`는 선택 support의 equality constraint를 정확히 적용한다는 뜻이며,
  비볼록 mixture likelihood의 전역 최적점을 뜻하지 않음

$$
\mathrm{df}_{\eta}(S) = d+(K-1)|S|+(K-1)\mathbf{1}(|S|>0)
$$

- 같은 support를 생성하는 여러 $\lambda_{\eta}$의 처리 규칙
- EBIC와 path sensitivity는 보충자료

## 3. Properties of the support target and algorithm

### 3.1 Posterior-score cancellation and label invariance

$$
j\notin S_{\eta}
\quad\Longleftrightarrow\quad
\eta_{1j}=\cdots=\eta_{Kj}
$$

해당 좌표가 모든 pairwise posterior log-score의 feature-dependent
선형항에서 소거됨을 보인다.

**Pairwise-dispersion identity.**

- centered group norm과 모든 성분쌍 차이의 관계
- 성분 label permutation에 대한 목적함수와 support의 불변성

### 3.2 Directional and natural-parameter heterogeneity

공통 concentration에서는 두 support가 일치한다.

$$
\kappa_1=\cdots=\kappa_K=\kappa\quad\Longrightarrow\quad\boldsymbol{c}_{j}^{(\eta)} = \kappa\boldsymbol{c}_{j}^{(\mu)}
$$

이질적 concentration에서는 방향 차이, concentration 차이, 두 요인의
상호작용으로 $S_{\mu}$와 $S_{\eta}$가 달라질 수 있음을 제시한다.

### 3.3 Conditional optimization properties

- fixed-responsibility smooth block의 convexity
- centered group penalty의 closed-form proximal map
- majorization 조건
- accepted generalized-EM update의 objective 비감소

전체 mixture likelihood의 전역 최적점 또는 모든 parameter iterate의
전역 수렴은 주장하지 않는다.

**Parameter-space and claim boundaries.**

- 모든 accepted path/refit iterate에 적용하는 finite concentration bound
- near-empty component 처리
- mixture nonregularity
- BIC의 practical model-selection approximation
- 증명하지 않은 oracle property, selection consistency, global convergence는
  본문 주장에 포함하지 않음

## 4. Numerical studies

### 4.1 Design and evaluation criteria

- 주 support-recovery 실험에서는 $K=K^\ast=4$로 고정한다.
- 기본 혼합비율은 $\pi_k=0.25$이고 기본 차원은 $d=200$이다.
- 각 고유 DGP cell은 $R=100$회 반복한다.
- 각 반복에서 공통 학습자료와 독립 검증자료
  $n_{\mathrm{test}}=5000$을 모든 방법에 동일하게 사용한다.
- M-L과 E 계열의 penalized path length는 240, dense initialization은
  `nstart=10`으로 고정한다.
- 비볼록 M-CGL은 60점과 120점 warm-start path에서 얻은 support의 합집합을
  refit한 뒤 BIC로 선택한다. 별도 path 240은 민감도 분석에 사용한다.
- 모든 likelihood 방법에 같은 concentration bound
  $0\leq\kappa_k\leq10^6$과 초기값 예산을 적용하고, 경계 도달률을
  기록한다. E 계열은 초기값, accepted proximal-gradient proposal 및
  support-constrained refit에서 이 조건을 검사한다.
- M-L, M-CGL 및 E-CGL은 각 estimand를 보존하는 support-constrained
  refit과 방법별 nominal dimension을 사용하여 BIC-after-refit으로
  support를 선택한다.
- E-CGL refit에서는 비선택 좌표의 centered contrast만 0으로 제한하고
  공통 natural-parameter baseline은 유지한다.
- equal 및 heterogeneous concentration 결과는 합치지 않는다.
- 평균과 함께 Monte Carlo standard error를 보고한다.

현재 완료된 24개 DGP cell에 적용한 공통 패널은 다음과 같다.

| 계열 | 방법 |
|---|---|
| External clustering | Spherical $k$-means |
| Density baseline | Dense vMF, shared/free $\kappa$ |
| Published sparse vMF | M-L |
| Proposed | E-CGL |
| Adaptive sensitivity | E-ACGL |

추가 예정인 $e_B=0.10$, $n=600,2000$ 두 cell은 4.4의 표본크기
회복 진단이므로 E-CGL을 주 분석으로 하고 E-ACGL만 민감도로 저장한다.

M-CGL은 $S_{\mu}$와 $S_{\eta}$의 관계를 확인하는 directional companion으로
다음 11개 고유 cell에 적용한다.

| M-CGL 적용 범위 | 고유 cell 수 |
|---|---:|
| $e_B=0.05$, $n=300,1000$, equal/heterogeneous $\kappa$ | 4 |
| Pure concentration, shared canonical background, crossed support | 3 |
| $e_B=0.05$, $n=600,2000$, equal/heterogeneous $\kappa$ | 4 |
| **합계** | **11** |

이 범위는 common-$\kappa$ equivalence, heterogeneous-$\kappa$ divergence와
표본크기 변화를 모두 포함한다. 최종 반복 전체의 중앙 실행시간은
M-CGL 312.18초/rep, E-CGL 45.86초/rep, E-ACGL 41.20초/rep였다.
M-CGL은 1,100회, E 계열은 각각 2,400회로 적용 범위가 다르므로 단순한
paired runtime 비교로 해석하지 않는다. M-CGL의 60·120 support 합집합은
세 대표 셀에서 별도 path 240과 같거나 더 낮은 refit BIC 후보를 포함했다.
M-CGL은 기존 문헌
baseline이 아니라 별도 estimand를 갖는 비교 방법이므로, 난이도 및 한계
24개 cell 전체가 아닌 위의 사전 지정된 estimand panel에서 평가한다.

E-ACGL은 E-CGL의 adaptive sensitivity로만 평가하고 전체 결과는
보충자료에 둔다. M-ACGL은 논문 비교군과 최종 simulation에서 제외한다.
Oracle-$S_{\mu}$와 Oracle-$S_{\eta}$는 경쟁 방법이 아니라
support가 알려진 기준 적합으로 사용한다. Path oracle은 표본크기 진단에서만
사용한다.

평가 지표:

$$
F_{1,P},\quad F_{1,\mu},\quad F_{1,\eta},\quad
\mathrm{ARI},\quad \mathrm{NMI}
$$

$$
\mathrm{NLL}_{\mathrm{test}},\quad
\mathrm{MSE}_{\mu},\quad \mathrm{MSE}_{\kappa},\quad
\mathrm{MSE}_{\eta}
$$

M-L은 $S_P$, M-CGL은 $S_{\mu}$, E-CGL과 E-ACGL은 $S_{\eta}$에
대한 target-specific recovery를 우선 보고한다. 모든 sparse 방법의
$S_{\eta}$ 기준 결과는 cross-target 성능으로 별도 표시한다. Dense vMF와
spherical $k$-means에는 support $F_1$을 부여하지 않는다. 공통 지표에는
test NLL, Bayes excess classification error, ARI, 실행시간 및 실패율을
포함한다.

### 4.2 Posterior-score support recovery

참 좌표 구조는 다음과 같다.

$$
(q_C,q_D,q_N)=(4,16,180)
$$

| 요인 | 수준 |
|---|---|
| 표본크기 | $n\in\{300,1000\}$ |
| oracle Bayes error | $e_B\in\{0.025,0.05,0.10\}$ |
| equal concentration | $\boldsymbol{\kappa}=(45,45,45,45)$ |
| heterogeneous concentration | $\boldsymbol{\kappa}=(30,40,50,60)$ |

총 12개 고유 cell이다. 각 cell의 참 모수는 반복 전에 고정하고,
oracle Bayes error는 독립 Monte Carlo 표본으로 재검증한다. 목표값과
achieved value의 차이는 0.002 이내로 제한하고 Monte Carlo standard
error를 함께 저장한다. 본문에서는
E-CGL의 $F_{1,\eta}$, exact-support rate, selected $q$, ARI, test NLL 및
$\mathrm{MSE}_{\eta}$를 중심으로 보고한다.

### 4.3 Directional versus posterior-score estimands

| 진단 | 설정 | 참 support 관계 |
|---|---|---|
| Common $\kappa$ | 4.2의 $n=1000$, $e_B=0.05$, equal $\kappa$ cell 재사용 | $S_{\mu}=S_{\eta}$ |
| Pure concentration | $n=1000$, $d=200$, $\mu_1=\cdots=\mu_4$, $\boldsymbol{\kappa}=(10,30,80,200)$, $\lvert\operatorname{supp}(\mu)\rvert=16$ | $S_{\mu}=\varnothing$, $S_{\eta}\neq\varnothing$ |
| Shared canonical background | $n=1000$, $(q_C,q_D,q_N)=(80,20,100)$, heterogeneous $\kappa$, $e_B=0.05$ | $S_{\mu}\supset S_{\eta}$ 가능 |
| Crossed support | $n=400$, $d=24$, $(q_{\eta\text{-only}},q_{\mu\text{-only}},q_{\mathrm{both}},q_0)=(4,4,8,8)$ | 두 support의 교차 구조 |

이 절은 M-CGL과 E-CGL의 승패가 아니라 서로 다른 참 support를 복원하는지
검증한다. Common-$\kappa$ cell은 4.2에서 재사용하므로 새로운 고유 cell은
3개이다. $S_{\mu}=\varnothing$인 pure-concentration cell에서는
$F_{1,\mu}$ 대신 exact-empty rate와 false-positive coordinate 수를
보고한다.

Crossed-support cell은 estimand 분리만 확인하는 stylized diagnostic이다.
eta-only 좌표에는 공통 방향계수 $\mu_{kj}=0.20$, mu-only 좌표에는
공통 자연모수 $\eta_{kj}=8$을 사용한다. 이 cell의 achieved oracle error는
약 0.0013이며 난이도별 성능 비교에는 사용하지 않는다.

### 4.4 Sample size, oracle benchmarks, and selector sensitivity

기본 표본크기 trajectory는 다음 조건을 고정한다.

$$
d=200,\qquad e_B=0.05,\qquad(q_C,q_D,q_N)=(4,16,180)
$$

표본크기는 다음 네 수준으로 둔다.

$$
n\in\{300,600,1000,2000\}
$$

Equal 및 heterogeneous $\kappa$를 각각 실행한다. $n=300,1000$의 4개
cell은 4.2에서 재사용하고, $n=600,2000$의 4개 cell을 추가한다.

E-CGL의 어려운 조건에서 표본크기 회복을 확인하기 위해 다음 trajectory를
추가한다.

$$
d=200,\qquad e_B=0.10,\qquad(q_C,q_D,q_N)=(4,16,180)
$$

$$
\boldsymbol{\kappa}=(30,40,50,60),\qquad
n\in\{300,600,1000,2000\}
$$

$n=300,1000$은 4.2의 기존 cell을 재사용하고, $n=600,2000$ 두 cell을
rep=100으로 추가한다. 두 새 cell은 E-CGL을 주 분석으로 실행하며 E-ACGL은
adaptive sensitivity로 함께 저장한다. M-CGL의 기존 $e_B=0.05$
표본크기 결과는 Supplement에 제시하고, $e_B=0.10$ 확장에는 추가하지 않는다.

4.4 본문 표는 M-CGL 행을 제외하고 E-CGL만 제시한다. 표는
$e_B=0.05$의 equal/heterogeneous $\kappa$ 8행과 $e_B=0.10$의
heterogeneous $\kappa$ 4행을 합한 12행으로 구성한다. 각 표본크기에서
BIC-selected support, path oracle 및 oracle-support refit을 비교하여 path
생성 오차와 selector 오차를 분리한다.

핵심 표본크기 지표는 다음과 같다.

$$
P(\widehat S_{\eta}=S_{\eta}),\quad F_{1,\eta},\quad
\mathrm{MSE}_{\eta},\quad \Delta_{\mathrm{NLL}}
$$

`oracle property` 대신 `oracle-support comparison` 또는
`oracle benchmark gap`을 사용한다. 이 절은 selection consistency의
증명이 아니라 표본크기 증가에 따른 empirical recovery behavior를
평가한다.

### 4.5 Stress conditions and computation

적용 범위와 한계는 다음 5개 cell에서 평가한다.

| 진단 | $n,d$ | support | 난이도 및 concentration |
|---|---|---|---|
| Moderately dense 1 | $300,200$ | $(q_C,q_D,q_N)=(4,80,116)$ | $e_B=0.10$, heterogeneous $\kappa$ |
| Moderately dense 2 | $1000,200$ | $(q_C,q_D,q_N)=(4,80,116)$ | $e_B=0.10$, heterogeneous $\kappa$ |
| Strongly dense | $1000,200$ | $(q_C,q_D,q_N)=(4,160,36)$ | $e_B=0.10$, heterogeneous $\kappa$ |
| High dimensional | $300,500$ | $(q_C,q_D,q_N)=(10,40,450)$ | $e_B=0.05$, $\boldsymbol{\kappa}=(45,60,75,90)$ |
| Weak-signal beta-min | $1000,200$ | $q_D=16$: strong 4, weak 12 | weak/strong contrast norm ratio 0.25, $e_B=0.05$ |

각 cell에서 수렴 성공률, 실패율, path endpoint 선택률, 실행시간 및
support recovery를 함께 보고한다. Weak-signal cell은 전체 Bayes 난이도와
개별 좌표의 최소 신호를 구분하기 위한 진단이다.

**Final execution scope and DGP validation.**

| 연구 | 최종 계획의 새 고유 cell 수 |
|---|---:|
| Main posterior-score recovery | 12 |
| Estimand separation | 3 |
| Sample-size behavior | 6 |
| Computation and limitations | 5 |
| **합계** | **26** |

최종 원고 계획은 26개 고유 DGP cell과 cell당 100회 반복으로 구성한다.
2026년 8월 3일 현재 24개 cell은 완료되었고, $e_B=0.10$,
heterogeneous $\kappa$의 $n=600,2000$ 두 cell은 실행 전이다. 기존 결과는
DGP, 알고리즘, refit 및 selector가 모두 동일할 때만 재사용한다.

각 cell 생성 직후 다음 invariant와 참 support를 자동 검증한다.

$$
\|\boldsymbol{\mu}_k\|_2=1,\qquad
\|\boldsymbol{\eta}_k\|_2=\kappa_k
$$

$$
S_P,\qquad S_{\mu},\qquad S_{\eta}
$$

Achieved oracle Bayes error, active centered contrast의 최소 norm, pairwise
KL divergence 및 oracle posterior confusion matrix를 함께 저장한다.
Support를 구성한 뒤 전체 행을 다시 scaling하여 참 support가 달라지는
생성 방식은 사용하지 않는다.

objective trace, Rcpp equality, 전체 runtime 표, EBIC/df/path 민감도는
보충자료에 둔다.

**Current rep=100 evidence freeze before the planned extension.**

2026년 8월 3일 기준 24개 DGP cell과 M-CGL estimand panel의 최종 실행,
selector audit 및 fixed-support oracle benchmark를 완료하였다.

| gate | 결과 | 판정 |
|---|---:|---|
| Main jobs | 236/236 | PASS |
| Main method-repetition rows | 15,500/15,500 | PASS |
| Selector groups | 5,900/5,900 | PASS |
| Oracle paired rows | 1,600/1,600 | PASS |
| Missing / duplicate / ERROR row | 0 / 0 / 0 | PASS |

Main sparse-support 조건에서 E-CGL은 대부분 참 $q_{\eta}=16$에 가까운
support를 선택하였다. $e_B=0.10$, $n=300$, heterogeneous $\kappa$에서는
$F_{1,\eta}=0.768$로 낮아졌고 E-ACGL은 0.948이었다. 이 조건은 adaptive
weighting의 이득과 E-CGL의 small-sample limitation을 함께 보여준다.

표본크기 진단에서 E-CGL은 equal 및 heterogeneous $\kappa$ 모두
$n=600$부터 $F_{1,\eta}\geq0.998$이었고, $n=2000$에서 exact-support
rate 1을 기록하였다. M-CGL도 $n=2000$에서 두 concentration 구조 모두
$F_{1,\mu}=1$과 exact-support rate 1을 기록하였다.

Pure-concentration M-CGL path는 매 반복 empty support를 포함했으나 BIC의
exact-empty rate는 0.24였다. $\mathrm{EBIC}_{1}$에서는 0.97이었다.
Shared-canonical cell에서 M-CGL의 $F_{1,\mu}$는 0.348이었으며, E-CGL의
$F_{1,\eta}$는 1이었다. 두 결과는 estimand와 selector에 따른 차이를
분리하여 보고한다.

Dense/high-dimensional 조건에서는 exact-support rate가 대부분 0이었다.
E-ACGL은 일부 조건에서 false-positive를 줄였으나 strongly dense cell에서는
E-CGL보다 낮았다. Adaptive extension의 일률적 우월성은 주장하지 않는다.

$e_B=0.10$, heterogeneous $\kappa$의 $n=600,2000$ 표본크기 확장은
위 evidence freeze에 아직 포함하지 않는다. 두 cell이 rep=100 무결성
검증을 통과한 뒤 4.4의 12행 E-CGL 표와 표본크기 Figure를 확정한다.

전체 결과와 수치표는
[`csda-final-simulation-results_260803.md`](../simulations/csda-final-simulation-results_260803.md)에 정리하였다.

## 5. Real-data applications

### 5.1 Classic3 and the primary analysis protocol

- Classic3를 본문 주 적용 사례로 사용한다.
- 완전·근접 중복을 제거한 전체 자료
  $n=3{,}883$, $d=2{,}000$, $K=3$을 한 번 적합한다.
- 문서는 고정된 SPLADE vocabulary-aligned representation으로 변환하고,
  label을 사용하지 않은 전체 payload의 분산 기준 상위 2,000개 좌표를
  선택한 뒤 row-wise unit normalization을 적용한다.
- benchmark label은 fitting, tuning 및 support 선택에 사용하지 않는다.
  $K=3$ 확인과 적합 후 ARI/NMI, component 명명 및 시각화에만 사용한다.
- 기존 five locked 80/20 splits는 삭제하지 않고 Supplement의
  prespecified stability and held-out assessment로 유지한다.
- 이전 계산 경로의 전체 자료 수치는 참고 기록으로만 보존하고 최종 본문
  수치로 재사용하지 않는다.

### 5.2 Methods and evaluation

- spherical $k$-means
- sparse $k$-means
- Dense vMF shared/free $\kappa$
- M-L
- M-CGL: 대표 centered directional comparator
- E-CGL: 주 방법
- E-ACGL: adaptive extension

전체 자료 적합은 $K=3$, `nstart=30`, centered path 240을 사용하고,
각 희소 방법의 추정 대상을 보존하는 exact support-constrained refit 후
방법별 명목 차원으로 BIC를 계산한다. M-L은 해당 방법의 path 및 refit
규칙을 유지한다.

본문 평가 지표는 다음과 같다.

$$
\mathrm{ARI},\quad \mathrm{NMI},\quad q,\quad q/d,
\quad \text{cluster sizes},\quad
\ell(\widehat\Theta),\quad \mathrm{BIC}^{\mathrm{refit}}
$$

희소 방법은 동일한 concentration 구조의 dense model과 matched comparison을
구성하고, 전체 자료 1회 적합의 차이는 기술적으로 보고한다. 실자료에는
참 support가 없으므로 TPR, FPR, precision 및 support $F_1$을 보고하지
않는다.

### 5.3 Classic3 full-data results and interpretation

- 전체 비교 방법의 ARI, NMI, selected $q$, $q/d$, cluster size,
  observed log-likelihood 및 BIC-after-refit
- E-CGL의 좌표 유지율
- matched dense model과의 기술적 비교
- 전체 자료 E-CGL centered-$\eta$ token contrast heatmap
- component별 양·음의 주요 contrast token
- 선택되지 않았지만 공통 baseline이 큰 token의 해석
- convergence, numerical warning 및 path-boundary diagnostic

최종 표와 그림은 final guarded proximal-gradient E path와 현재
BIC-after-refit 규칙으로 전체 자료 적합을 다시 완료한 뒤 확정한다.

### 5.4 Supplementary stability and contrast analyses

- Classic3 five locked splits에서 ARI/NMI, $q/d$, held-out NLL,
  support Jaccard 및 selection frequency를 보고한다.
- 분할이 서로 겹치므로 평균과 표준편차는 기술통계로만 해석한다.
- BBCSport는 정보가 조밀하게 분포한 contrast application으로 Supplement에
  배치하고 희소화에 따른 held-out NLL 손실을 그대로 보고한다.
- CSTR은 Rossi 방식 구현의 문헌 재현과 prototype-oriented limitation을
  확인하는 보조 사례로만 사용한다.
- E-CGL의 일률적 군집·예측 우월성을 주장하지 않는다.

### 5.5 Reproducibility statement

- 데이터 출처, 중복 제거 규칙, SPLADE model/revision, vocabulary ranking,
  seed, software 및 hardware를 고정한다.
- 본문에는 전체 자료에서 선택된 모든 적합의 완료 여부와 핵심 수치
  안정성만 기술한다.
- five-split 전체 표, support stability, BBCSport, CSTR 문헌 재현,
  Bessel-ratio audit 및 path-resolution audit는 Supplement에 둔다.

## 6. Discussion

Discussion은 별도 Conclusion 절로 나누지 않고 다음 순서로 구성한다.

1. E-CGL이 추정하는 posterior-score contrast support
2. prototype, directional, posterior-score support의 차이
3. common/heterogeneous concentration에서의 적용 범위
4. dense support와 약한 신호에서의 한계
5. practical BIC와 고정 $K$ 조건
6. 회전 비불변성 및 vocabulary-aligned 좌표 해석의 범위
7. pairwise fusion, data-driven $K$, 안정화 이론에 관한 후속 과제

M-CGL을 실패한 방법으로 서술하지 않는다. M-CGL은 방향 이질성,
E-CGL은 posterior-score 이질성을 추정한다는 차이로 정리한다.

## Main tables and figures

| 번호 | 내용 |
|---|---|
| Table 1 | 방법별 estimand, penalty, concentration 구조 |
| Table 2 | 시뮬레이션 설계와 참 support |
| Table 3 | 주요 support recovery, estimand separation 및 두 난이도의 표본크기 결과 |
| Table 4 | Classic3 전체 자료의 방법별 적합·좌표 선택 결과 |
| Figure 1 | prototype, directional, posterior-score support 개념도 |
| Figure 2 | concentration 구조별 target-specific $F_1$ |
| Figure 3 | $e_B=0.05,0.10$의 표본 크기별 recovery 및 oracle benchmark gap |
| Figure 4 | Classic3 전체 자료 E-CGL centered-$\eta$ token contrast |

본문에서는 동일 수치를 표와 그림에 중복하여 제시하지 않는다.

## Supplement

### S1. Additional notation and proofs

- Proposition의 전체 증명
- pairwise-dispersion identity
- label invariance
- proximal map과 accepted-update 결과

### S2. Complete algorithms

- E-CGL/E-ACGL 상세 pseudocode
- M-CGL ADMM 및 manifold update
- Banerjee concentration approximation과 수치적 root solving
- stopping rule과 failure handling

### S3. Full simulation design and results

- 전체 cell 결과
- adaptive extension
- target-specific 및 cross-target 지표
- Monte Carlo uncertainty

### S4. Model-selection sensitivity

- BIC-before versus BIC-after
- EBIC
- nominal df 대안
- path length와 path oracle
- $K$ 선택 및 misspecification

### S5. Numerical validation

- objective trace
- proximal 및 ADMM residual
- sphere constraint error
- R과 Rcpp 결과 일치성
- 초기값, warning, failure, runtime

### S6. Additional real-data results

- Classic3 five-split 전체 표와 held-out 결과
- support Jaccard 및 coordinate selection frequency
- 추가 token 및 common-baseline 표
- BBCSport 전체 결과와 path diagnostic
- CSTR 문헌 재현

### S7. Reproducibility

- 소프트웨어와 하드웨어
- seed와 환경
- 데이터 전처리 규약
- 코드와 데이터 접근 경로

## 본문 주장 경계

| 본문에서 주장 | 본문에서 주장하지 않음 |
|---|---|
| $S_{\eta}$의 posterior-score cancellation 의미 | 모든 형태의 군집 관련 변수 선택 |
| E-CGL의 centered group regularization | E-CGL의 일률적 우월성 |
| common $\kappa$에서 $S_{\mu}$와 $S_{\eta}$의 관계 | 이질적 $\kappa$에서 두 support의 일반적 동일성 |
| accepted update의 제한적 계산 성질 | 전역 최적점 또는 전역 수렴 |
| practical BIC-after support selection | exact effective df 또는 BIC 일관성 |
| oracle-support benchmark와의 경험적 차이 | 증명되지 않은 oracle property |
| 고정 $K$에서의 support recovery | joint $(K,\lambda)$ 선택의 해결 |

## 분량 기준

아래 분량은 작업 기준이며 CSDA의 공식 제한이 아니다.

| 절 | 목표 분량 |
|---|---:|
| Introduction | 2--3쪽 |
| Model and methodology | 5--6쪽 |
| Theoretical properties | 3--4쪽 |
| Numerical studies | 5--6쪽 |
| Real-data applications | 3--4쪽 |
| Discussion | 1--2쪽 |

본문은 참고문헌을 제외하고 약 20--25쪽을 목표로 하며, 전체 반복 결과와
구현 진단은 보충자료에 둔다.
