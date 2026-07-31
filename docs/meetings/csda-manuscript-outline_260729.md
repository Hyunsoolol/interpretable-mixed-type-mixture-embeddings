# CSDA 투고 논문 구성안

업데이트: 2026-07-31

이 문서는 논문 본문이 아니라 장별 목적, 포함 결과, 본문과 보충자료의
경계를 정한 목차 문서이다.

## 논문 범위

| 방법 | 추정 대상 | 논문 내 역할 |
|---|---|---|
| E-CGL | posterior-score contrast support $S_{\eta}$ | 주 제안 방법 |
| E-ACGL | adaptive centered-$\eta$ support | 보조 확장 |
| M-CGL | centered directional support $S_{\mu}$ | matched directional comparator |
| M-ACGL | adaptive centered-$\mu$ support | 보조 비교 |
| M-L | prototype support $S_P$ | Rossi 문헌 비교 |
| Dense vMF | 전체 좌표를 사용하는 density model | 비정규화 기준 |

주 연구 질문은 다음과 같다.

$$
S_{\eta} = \{j:\|\boldsymbol{\eta}_{\cdot j}-\bar{\eta}_{j}\mathbf{1}_{K}\|_2>0\}
$$

즉, 모든 성분쌍의 posterior log-score 비교에 실제로 들어가는 좌표를
추정하는 것이 논문의 중심이다. M-CGL과 M-ACGL은 별도 주방법이 아니라
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
  Heterogeneity in von Mises-Fisher Mixtures*

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

### 2.1 von Mises-Fisher mixture

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

### 2.3 E-CGL

- centered-$\eta$ parameterization
- coordinate-wise group penalty
- 공통 baseline은 penalty 대상에서 제외

$$
P_{\mathrm{E-CGL}}(\boldsymbol{\eta}) = \lambda_{\eta}\sum_{j=1}^{d}\|\boldsymbol{c}_{j}^{(\eta)}\|_2
$$

- E-CGL 목적함수
- 선택 좌표의 의미
- nonzero prototype 선택과의 차이

### 2.4 E-ACGL

- adaptive weight 정의
- $\gamma=1$과 median normalization
- E-CGL의 보조 확장으로만 제시
- E-ACGL이 항상 개선된다는 주장을 두지 않음

### 2.5 Directional and prototype comparators

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
- M-CGL/M-ACGL: directional heterogeneity support
- E-CGL/E-ACGL: posterior-score contrast support
- M-CGL의 목적함수와 단위구면 제약만 본문에 제시
- M-CGL의 ADMM 및 manifold 반복식은 보충자료에 배치

### 2.6 Estimation algorithm

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

### 2.7 Support-constrained refit and selection

- 선택되지 않은 좌표에서 centered contrast만 0으로 제한
- 공통 natural-parameter baseline 유지
- support-constrained refit
- BIC-after-refit
- 명목 모형 차원은 exact effective degrees of freedom이 아님을 명시

$$
\mathrm{df}_{\eta}(S) = d+(K-1)|S|+(K-1)\mathbf{1}(|S|>0)
$$

- 같은 support를 생성하는 여러 $\lambda_{\eta}$의 처리 규칙
- EBIC와 path sensitivity는 보충자료

## 3. Theoretical properties

### 3.1 Posterior-score cancellation

$$
j\notin S_{\eta}
\quad\Longleftrightarrow\quad
\eta_{1j}=\cdots=\eta_{Kj}
$$

해당 좌표가 모든 pairwise posterior log-score의 feature-dependent
선형항에서 소거됨을 보인다.

### 3.2 Pairwise-dispersion identity and label invariance

- centered group norm과 모든 성분쌍 차이의 관계
- 성분 label permutation에 대한 목적함수와 support의 불변성

### 3.3 Relation between directional and posterior-score supports

공통 concentration에서는 두 support가 일치한다.

$$
\kappa_1=\cdots=\kappa_K=\kappa\quad\Longrightarrow\quad\boldsymbol{c}_{j}^{(\eta)} = \kappa\boldsymbol{c}_{j}^{(\mu)}
$$

이질적 concentration에서는 방향 차이, concentration 차이, 두 요인의
상호작용으로 $S_{\mu}$와 $S_{\eta}$가 달라질 수 있음을 제시한다.

### 3.4 Proximal and accepted-update properties

- fixed-responsibility smooth block의 convexity
- centered group penalty의 closed-form proximal map
- majorization 조건
- accepted generalized-EM update의 objective 비감소

전체 mixture likelihood의 전역 최적점 또는 모든 parameter iterate의
전역 수렴은 주장하지 않는다.

### 3.5 Parameter-space and claim boundaries

- finite concentration parameter space
- near-empty component 처리
- mixture nonregularity
- BIC의 practical model-selection approximation
- 증명하지 않은 oracle property, selection consistency, global convergence는
  본문 주장에 포함하지 않음

## 4. Numerical studies

### 4.1 Design and evaluation criteria

- true $K$는 support-recovery 실험에서 고정
- oracle Bayes error로 군집 난이도 설정
- equal/heterogeneous concentration 결과를 분리
- 반복 수와 Monte Carlo standard error 보고

주 비교 방법:

| 계열 | 방법 |
|---|---|
| Density baseline | Dense vMF, shared/free $\kappa$ |
| Published sparse vMF | M-L |
| Directional contrast | M-CGL |
| Proposed | E-CGL |
| Oracle reference | Oracle-$S_{\mu}$, Oracle-$S_{\eta}$ |

E-ACGL과 M-ACGL의 전체 결과는 보충자료에 둔다.

평가 지표:

$$
F_{1,\eta},\quad F_{1,\mu},\quad \mathrm{ARI},\quad
\mathrm{NLL}_{\mathrm{test}},\quad
\mathrm{MSE}_{\eta},\quad \mathrm{MSE}_{\mu}
$$

각 방법은 자신의 estimand에 대한 target-specific 성능과
$S_{\eta}$에 대한 cross-target 성능을 구분하여 보고한다.

### 4.2 Main posterior-score support recovery

- $n\in\{300,1000\}$
- $d=200$
- oracle Bayes error $e_B\in\{0.025,0.05,0.10\}$
- common, decision, noise 좌표 분해
- equal/heterogeneous $\kappa$를 별도 표 또는 panel로 제시
- E-CGL의 $F_{1,\eta}$, selected $q$, ARI, MSE를 중심으로 보고

### 4.3 Estimand-separation diagnostics

1. Common $\kappa$: $S_{\mu}=S_{\eta}$
2. Pure concentration heterogeneity:
   $S_{\mu}=\varnothing$, $S_{\eta}\neq\varnothing$
3. Shared canonical background:
   $S_{\mu}\supset S_{\eta}$가 가능한 경우
4. Crossed support:
   $\eta$-only, $\mu$-only, both, null

이 절은 M-CGL과 E-CGL의 승패가 아니라 서로 다른 참 support를 복원하는지
검증한다.

### 4.4 Oracle benchmark and sample-size behavior

- oracle-support refit
- path oracle
- BIC-selected support
- path와 selector에서 발생하는 오차 분리
- 표본 크기에 따른 support recovery와 estimation error

`oracle property` 대신 `oracle-support comparison` 또는
`oracle benchmark gap`을 사용한다.

### 4.5 Computation and limitations

- 수렴 성공률과 실패율
- E-CGL 실행시간
- path endpoint 선택률
- dense weak-support negative control
- 초기값 및 path sensitivity의 핵심 결과

objective trace, Rcpp equality, 전체 runtime 표, EBIC/df/path 민감도는
보충자료에 둔다.

## 5. Real-data applications

### 5.1 Datasets and analysis protocol

- Classic3: 주 적용 사례
- BBCSport: 대조 사례
- SPLADE vocabulary-aligned representation
- 중복 제거, train-only vocabulary ranking, unit-norm normalization
- 고정 반복 홀드아웃
- benchmark label은 적합 후 평가와 성분 이름 부여에만 사용

### 5.2 Methods and evaluation

- spherical $k$-means
- Dense vMF shared/free $\kappa$
- M-L
- M-CGL/M-ACGL
- E-CGL/E-ACGL
- held-out ARI, NMI, NLL/doc, selected $q$, support stability, runtime

희소 방법은 동일한 concentration 구조의 dense model과 paired comparison을
수행한다. 실자료에는 참 support가 없으므로 TPR, FPR, precision, $F_1$을
보고하지 않는다.

### 5.3 Classic3: primary application

- 8개 방법의 평균과 표준편차
- E-CGL의 좌표 유지율
- dense free-$\kappa_k$ 대비 paired ARI와 NLL 차이
- 분할 간 support Jaccard
- centered-$\eta$ token contrast heatmap
- 선택되지 않았지만 공통 baseline이 큰 token의 해석

### 5.4 BBCSport: contrast application

- 동일한 평가표
- 희소화에 따른 held-out NLL 손실
- 정보가 조밀하게 분포할 때의 제한
- E-CGL이 모든 자료에서 우월하지 않음을 명시

### 5.5 Reproducibility statement

본문에는 모든 선택 적합의 완료 여부와 핵심 수치 안정성만 기술한다.
CSTR 문헌 재현, Bessel-ratio audit, path-resolution audit는 보충자료로
이동한다.

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
| Table 3 | 주요 posterior-score support recovery 결과 |
| Table 4 | Classic3와 BBCSport 반복 홀드아웃 결과 |
| Figure 1 | prototype, directional, posterior-score support 개념도 |
| Figure 2 | concentration 구조별 target-specific $F_1$ |
| Figure 3 | 표본 크기별 oracle benchmark gap |
| Figure 4 | Classic3 centered-$\eta$ token contrast |

본문에서는 동일 수치를 표와 그림에 중복하여 제시하지 않는다.

## Supplement

### S1. Additional notation and proofs

- Proposition의 전체 증명
- pairwise-dispersion identity
- label invariance
- proximal map과 accepted-update 결과

### S2. Complete algorithms

- E-CGL/E-ACGL 상세 pseudocode
- M-CGL/M-ACGL ADMM 및 manifold update
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

- split-level 전체 표
- support stability
- 추가 token 표
- BBCSport path diagnostic
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
