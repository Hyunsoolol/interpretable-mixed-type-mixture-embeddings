# 연구미팅 상세 백업자료: E-CGL 방법론, 근거, 검증 및 한계 (2026-07-14)

이 문서는 2026년 7월 14일 연구미팅의 질문 대응용 기술 부록이다. 주 모형은
**E-CGL**이며, **E-ACGL**은 adaptive 보조 확장이다. 본문에서 생략한
모형 정의, 이론적 동기, 기하학적 해석, 최적화, refit, 정보지수,
시뮬레이션, 실자료, 계산 검증과 한계를 한 문서에 정리한다.

## 0. 한 문장 요약과 주장 범위

관측치 $x\in\mathbb S^{d-1}$에 대한 vMF mixture에서 posterior score의
coordinate coefficient는 방향모수 $\mu_k$ 단독이 아니라 자연모수
$\eta_k=\kappa_k\mu_k$이다. E-CGL은 component 공통 부분을 제거한

$$
c_{kj}=\eta_{kj}-\bar\eta_j,
\qquad
\bar\eta_j=K^{-1}\sum_{k=1}^K\eta_{kj}
$$

를 coordinate별 group으로 묶어

$$
\lambda_\eta\sum_{j=1}^d\lVert c_{\cdot j}\rVert_2
$$

로 penalize한다. 선택 대상은 sparse prototype이 아니라 **component 간
선형 posterior score 차이를 만드는 coordinate support**다.

현재 근거가 뒷받침하는 범위는 다음과 같다.

$$
\boxed{
\text{E-CGL은 sparse posterior decision support를 직접 추정하는 모형이다.}
}
$$

모든 clustering 환경에서의 위험 우위, selection consistency, 전역 최적성,
정보지수의 일관성은 현재 주장 범위에 포함하지 않는다.

## 1. 연구 동기와 estimand

### 1.1 기존 sparse prototype과 연구 질문의 차이

Rossi and Barbaro (2022)의 sparse vMF mixture는 방향 prototype
$\mu_k$의 nonzero entry를 선택한다. 해당 support는

$$
S_{\mu}
=\left\{j:\exists k,\ \mu_{kj}\neq0\right\}
$$

로 표현할 수 있다. 이 support는 coordinate가 어느 component의 prototype에
존재하는지를 나타낸다.

본 연구의 질문은 다르다.

> Coordinate $j$가 존재하는가가 아니라, coordinate $j$가 component 간
> posterior score 차이를 만드는가?

이에 따라 estimand를

$$
S_{\mathrm{dec}}
=\left\{j:\lVert c_{\cdot j}\rVert_2>0\right\}
$$

로 정의한다. $S_\mu$와 $S_{\mathrm{dec}}$는 공통 좌표가 없을 때 가까울 수
있지만, 모든 component에 공통으로 존재하는 coordinate가 많으면 서로
달라진다.

### 1.2 세 종류의 coordinate

| coordinate 유형 | $\bar\eta_j$ | $c_{\cdot j}$ | prototype에 존재 | decision support |
|:---|:---:|:---:|:---:|:---:|
| 공통 영향 coordinate | 비영일 수 있음 | $0$ | 예 | 아니오 |
| 군집 구분 coordinate | 제한 없음 | 비영 | 대체로 예 | 예 |
| 전역 null coordinate | $0$ | $0$ | 아니오 | 아니오 |

공통 영향 coordinate와 전역 null coordinate는 둘 다 decision support에서
제외되지만 같은 모수 상태는 아니다. 전자는 공통 baseline $\bar\eta_j$가
남고, 후자는 baseline과 contrast가 모두 0이다.

## 2. vMF mixture와 posterior score

### 2.1 vMF density

$x_i\in\mathbb S^{d-1}$일 때 $K$-component vMF mixture는

$$
f(x_i;\Theta)
=\sum_{k=1}^K\pi_k C_d(\kappa_k)
\exp\{\kappa_k\mu_k^\top x_i\},
$$

$$
\pi_k>0,
\qquad
\sum_{k=1}^K\pi_k=1,
\qquad
\lVert\mu_k\rVert_2=1,
\qquad
\kappa_k\ge0.
$$

정규화 상수는

$$
C_d(\kappa)
=\frac{\kappa^{d/2-1}}
{(2\pi)^{d/2}I_{d/2-1}(\kappa)}
$$

이며 $I_\nu$는 제1종 modified Bessel function이다.

### 2.2 자연모수 표현

$$
\eta_k=\kappa_k\mu_k\in\mathbb R^d.
$$

Observed log-likelihood는

$$
\ell(\Theta)
=\sum_{i=1}^n
\log\left[
\sum_{k=1}^K
\pi_k C_d(\lVert\eta_k\rVert_2)
\exp(\eta_k^\top x_i)
\right].
$$

Posterior responsibility는

$$
\tau_{ik}
=\frac{
\pi_k C_d(\lVert\eta_k\rVert_2)\exp(\eta_k^\top x_i)
}{
\sum_{\ell=1}^K
\pi_\ell C_d(\lVert\eta_\ell\rVert_2)
\exp(\eta_\ell^\top x_i)
}.
$$

Component $k$의 posterior classification score는

$$
s_k(x)
=\log\pi_k+\log C_d(\lVert\eta_k\rVert_2)+\eta_k^\top x.
$$

## 3. 왜 $\mu$가 아니라 $\eta$인가

### 3.1 Posterior score의 직접 계수

두 component $k$와 $\ell$의 score 차이는

$$
s_k(x)-s_\ell(x)
=a_{k\ell}+(\eta_k-\eta_\ell)^\top x,
$$

$$
a_{k\ell}
=\log\frac{\pi_k}{\pi_\ell}
+\log\frac{C_d(\lVert\eta_k\rVert_2)}
{C_d(\lVert\eta_\ell\rVert_2)}.
$$

따라서 coordinate $x_j$의 선형 계수는
$\eta_{kj}-\eta_{\ell j}$이다. $\mu_k$만 penalize하면
$\kappa_k$ 차이가 coordinate effect에 미치는 영향을 선택 기준에 직접
포함하지 못한다.

### 3.2 $\eta$에서 $\mu$와 $\kappa$의 복원

$\eta_k\neq0$이면

$$
\kappa_k=\lVert\eta_k\rVert_2,
\qquad
\mu_k=\frac{\eta_k}{\lVert\eta_k\rVert_2}.
$$

증명은 $\lVert\eta_k\rVert_2
=\kappa_k\lVert\mu_k\rVert_2=\kappa_k$에서 바로 따른다. $\eta_k=0$이면
$\kappa_k=0$이고 방향 $\mu_k$는 식별되지 않는다. 이는 mixture component
label의 permutation invariance와 별개의 문제다.

### 3.3 같은 $\mu$, 다른 $\kappa$의 예

$\mu_1=\mu_2=\mu$이고 $\kappa_1\neq\kappa_2$이면

$$
\eta_1-\eta_2=(\kappa_1-\kappa_2)\mu\neq0.
$$

Score 차이는

$$
s_1(x)-s_2(x)
=\log\frac{\pi_1}{\pi_2}
+\log\frac{C_d(\kappa_1)}{C_d(\kappa_2)}
+(\kappa_1-\kappa_2)\mu^\top x.
$$

따라서 방향이 같아도 concentration 차이로 posterior가 달라질 수 있다.
$\mu$ support만으로는 이 차이를 표현할 수 없지만 $\eta$ contrast에는 직접
나타난다.

### 3.4 $\eta$가 설명하지 않는 부분

$\eta$의 coordinate contrast는 posterior score의 선형 $x$-dependent
부분을 설명한다. Mixing proportion과 normalizing constant가 만드는
절편 $a_{k\ell}$도 posterior decision에 남는다. 따라서 E-CGL은
posterior decision의 모든 원인을 coordinate support 하나로 환원하는
모형이 아니라, **coordinate-dependent linear decision support**를
추정하는 모형이다.

## 4. 왜 raw $\eta$가 아니라 centered $\eta$인가

### 4.1 Orthogonal decomposition

Coordinate $j$의 component vector를

$$
\eta_{\cdot j}
=\bar\eta_j\mathbf1_K+c_{\cdot j},
\qquad
\bar\eta_j=K^{-1}\mathbf1_K^\top\eta_{\cdot j},
$$

$$
c_{\cdot j}=H_K\eta_{\cdot j},
\qquad
H_K=I_K-K^{-1}\mathbf1_K\mathbf1_K^\top,
\qquad
\mathbf1_K^\top c_{\cdot j}=0
$$

로 분해한다. $\bar\eta_j\mathbf1_K$는 $\operatorname{span}(\mathbf1_K)$에
대한 projection이고, $c_{\cdot j}$는 그 직교공간
$\mathbf1_K^\perp$에 대한 projection이다.

### 4.2 Pairwise decision coefficient의 보존

$$
\eta_{kj}-\eta_{\ell j}
=c_{kj}-c_{\ell j}.
$$

따라서 공통 baseline은 pairwise linear score coefficient에서 상쇄되고,
centered contrast가 모든 pairwise slope 차이를 보존한다.

다음 네 조건은 동치다.

$$
c_{\cdot j}=0
\Longleftrightarrow
\eta_{1j}=\cdots=\eta_{Kj}
\Longleftrightarrow
\eta_{kj}-\eta_{\ell j}=0\ \forall k,\ell
$$

$$
\Longleftrightarrow
x_j\text{의 계수가 모든 pairwise linear score 차이에서 0이다.}
$$

### 4.3 Raw $\eta$ penalty의 target 불일치

Raw group penalty

$$
\lambda_\eta\sum_j\lVert\eta_{\cdot j}\rVert_2
$$

는 common baseline과 component contrast를 함께 줄인다. 모든 component에
같은 값으로 존재하는 좌표도 $\lVert\eta_{\cdot j}\rVert_2>0$이므로
prototype presence와 decision heterogeneity를 구분하지 못한다.

Centered penalty는

$$
\lambda_\eta\sum_j\lVert H_K\eta_{\cdot j}\rVert_2
$$

로서 component 간 차이만 penalize하고 공통 baseline은 penalty 대상에서
제외한다.

### 4.4 공통 coordinate가 완전히 사라지는 것은 아니다

$c_{\cdot j}=0$이면 $x_j$의 pairwise linear coefficient는 사라지지만
$\bar\eta_j$는 모형에 남는다. 또한

$$
\kappa_k=\lVert\eta_k\rVert_2
$$

이므로 공통 baseline은 $C_d(\kappa_k)$가 만드는 score 절편에 간접적으로
영향을 줄 수 있다. 이 때문에 최종 refit에서 비선택 coordinate의
$\eta$ 전체를 0으로 만들지 않고 contrast만 0으로 둔다.

### 4.5 Centered $\mu$를 주 estimand로 두지 않은 이유

$\mu_{\cdot j}-\bar\mu_j\mathbf1$은 수학적으로 계산할 수 있다. 단순히
구면 제약 때문에 정의할 수 없는 양은 아니다. 다만 다음 이유로 주
estimand가 아니다.

| 이유 | 설명 |
|:---|:---|
| score target 불일치 | posterior의 선형 계수는 $\mu_k$가 아니라 $\kappa_k\mu_k$다. |
| concentration 누락 | $\mu$ centering은 component별 $\kappa_k$ 차이를 제거한다. |
| parameter geometry | centered $\mu$는 unit sphere 위의 방향모수 자체가 아니라 tangent-like Euclidean contrast다. |

Centered $\mu$는 보조 진단량으로 사용할 수 있으나 posterior decision
coefficient의 직접 support는 centered $\eta$로 정의된다.

## 5. 왜 coordinate-wise group $L_2$인가

### 5.1 선택 단위와 penalty 단위

선택 여부는

$$
j\in S_{\mathrm{dec}}
\Longleftrightarrow
\lVert c_{\cdot j}\rVert_2>0
$$

로 coordinate 전체에서 정의된다. 이에 대응하는 E-CGL penalty는

$$
P_{\mathrm{CGL}}(\eta)
=\lambda_\eta\sum_{j=1}^d\lVert c_{\cdot j}\rVert_2.
$$

각 group은 coordinate $j$에 대한 $K$개 component contrast entry다.

### 5.2 Pairwise 차이의 기하학적 의미

다음 항등식이 성립한다.

$$
\lVert c_{\cdot j}\rVert_2^2
=\frac1K\sum_{1\le k<\ell\le K}
(\eta_{kj}-\eta_{\ell j})^2.
$$

즉 group norm은 coordinate $j$에서 모든 component pair의 posterior-score
slope 차이를 모은 root-sum-square다. $\lVert c_{\cdot j}\rVert_2=0$이면
모든 pair의 slope가 같고, 비영이면 적어도 한 pair가 다르다.

### 5.3 Proximal map

Centered target $z\in\mathbf1_K^\perp$에 대해

$$
\min_{c:\mathbf1_K^\top c=0}
\left\{
\frac12\lVert c-z\rVert_2^2+t\lambda\lVert c\rVert_2
\right\}
$$

의 해는

$$
\operatorname{prox}_{t\lambda\lVert\cdot\rVert_2}(z)
=\left(1-\frac{t\lambda}{\lVert z\rVert_2}\right)_+z.
$$

해가 $z$의 scalar multiple이므로 centering 제약을 그대로 만족한다. 한
coordinate의 contrast vector는 전체가 0이 되거나 방향을 유지한 채 함께
축소된다.

### 5.4 Entry-wise $L_1$과의 차이

E-CL은

$$
P_{\mathrm{CL}}(\eta)
=\lambda_\eta\sum_{k=1}^K\sum_{j=1}^d|c_{kj}|
$$

를 사용하며 각 entry는

$$
\operatorname{sign}(z_{kj})(|z_{kj}|-t\lambda_\eta)_+
$$

로 따로 threshold된다.

| 항목 | E-CL | E-CGL |
|:---|:---|:---|
| 선택 단위 | component-coordinate entry | coordinate의 contrast vector |
| 일부 component만 0 가능 | 예 | 비영 group에서는 상대 방향 유지 |
| 직접 대응하는 estimand | entry support | coordinate decision support |
| label permutation invariance | 전체 penalty는 불변 | group별 norm과 support가 불변 |

Group penalty가 음수 값을 새로 만들거나 특정 부호를 강제하는 것은 아니다.
선택된 group 안에서는 기존 contrast vector의 상대 방향을 유지하므로 양·음의
상대 score contribution이 entry-wise threshold보다 덜 파편화될 수 있다.

### 5.5 성능 근거의 범위

현재 matched ablation에서는 쉬운 조건에서 E-CL과 E-CGL의 결과가 거의
같았고, 어려운 조건에서도 평균 차이가 Monte Carlo 변동과 비슷했다.
따라서 group penalty의 보편적 성능 우위를 주장하지 않는다. E-CGL의
주 근거는 **coordinate-level estimand와 penalty 단위의 일치**다.

과거 `Eta ANOVA L1` 진단은 selector, path, update와 refit이 현재 matched
E-CL과 달랐다. 당시 dense support 결과는 구현 전체의 결합 효과이며,
현재 E-CL 대 E-CGL의 단독 penalty 효과로 해석하지 않는다.

## 6. Adaptive 확장 E-ACGL

E-ACGL의 penalty는

$$
P_{\mathrm{ACGL}}(\eta)
=\lambda_\eta\sum_{j=1}^d w_j\lVert c_{\cdot j}\rVert_2,
$$

$$
w_j^{\mathrm{raw}}
=\left(\lVert c_{\cdot j}^{\mathrm{init}}\rVert_2+\epsilon\right)^{-\gamma},
\qquad
w_j=\frac{w_j^{\mathrm{raw}}}
{\operatorname{median}_h w_h^{\mathrm{raw}}}.
$$

현재 설정은

$$
\gamma=1,
\qquad
\epsilon=10^{-6}.
$$

Median normalization은 상대 weight 순서를 바꾸지 않고 공통 scale만
$\lambda_\eta$에 흡수될 수 있게 한다. 초기 contrast가 큰 coordinate는
작은 weight를 받아 덜 축소되고, 작은 coordinate는 더 크게 축소된다.

E-ACGL은 다음 이유로 보조 확장으로 둔다.

| 항목 | E-CGL | E-ACGL |
|:---|:---|:---|
| weight | $w_j=1$ | 초기 적합에서 추정 |
| 추가 의존성 | 없음 | 초기 contrast, $\gamma$, $\epsilon$ |
| 현재 성능 | 전반적으로 안정 | 조건별 개선·악화가 혼재 |
| 논문 역할 | 주 specification | adaptive sensitivity |

## 7. 이론적 성질

### 7.1 Component-level 복원 가능성

$\eta_k\neq0$에서 $(\mu_k,\kappa_k)$는 $\eta_k$로부터 유일하게 복원된다.
$\eta_k=0$에서는 $\mu_k$가 식별되지 않는다.

### 7.2 Pairwise score 보존

$$
(\eta_k-\eta_\ell)^\top x
=(c_k-c_\ell)^\top x.
$$

Centering은 모든 pairwise linear posterior-score difference를 보존한다.

### 7.3 Decision support의 필요충분조건

$$
j\notin S_{\mathrm{dec}}
\Longleftrightarrow
c_{\cdot j}=0
\Longleftrightarrow
x_j\text{의 모든 pairwise slope가 0}.
$$

### 7.4 Label invariance

Component permutation matrix $P$에 대해

$$
\lVert(Pc)_{\cdot j}\rVert_2
=\lVert c_{\cdot j}\rVert_2.
$$

따라서 E-CGL penalty와 $S_{\mathrm{dec}}$는 label switching에 불변이다.
Adaptive weight도 group norm으로 만들기 때문에 같은 성질을 갖는다.

### 7.5 현재 이론이 포함하지 않는 것

현재 명제는 estimand, posterior-score geometry와 penalty의 정렬을 보인다.
다음 항목은 아직 formal theorem으로 확립하지 않았다.

- 고차원 asymptotic selection consistency
- Oracle property
- 모든 signal regime에서의 risk dominance
- Proximal path fit의 stationary-point convergence
- Non-convex mixture likelihood의 전역 최적성
- Practical BIC 자유도의 일관성

## 8. Penalized estimation과 최적화

### 8.1 Path-generating criterion

고정된 $K$와 $\lambda_\eta$에서

$$
\mathcal L_{\lambda_\eta}(\Theta)
=\ell(\Theta)
-\lambda_\eta\sum_{j=1}^d w_j\lVert c_{\cdot j}\rVert_2
$$

를 path 생성 기준으로 사용한다. $w_j=1$이면 E-CGL이다. Penalty와 vMF
normalizer가 $\eta_k$를 통해 결합되므로 penalized M-step 전체의 closed-form
해는 없다.

### 8.2 E-step과 unpenalized target

Iteration $t$에서

$$
N_k^{(t)}=\sum_{i=1}^n\tau_{ik}^{(t)},
\qquad
r_k^{(t)}=\sum_{i=1}^n\tau_{ik}^{(t)}x_i,
$$

$$
\widetilde\pi_k=\frac{N_k^{(t)}}n,
\qquad
\widetilde\mu_k=\frac{r_k^{(t)}}{\lVert r_k^{(t)}\rVert_2},
\qquad
\widetilde\rho_k=\frac{\lVert r_k^{(t)}\rVert_2}{N_k^{(t)}}.
$$

Concentration target은 mean-resultant-length approximation

$$
\widetilde\kappa_k
\approx
\frac{d\widetilde\rho_k-\widetilde\rho_k^3}
{1-\widetilde\rho_k^2}
$$

으로 계산하고

$$
\widetilde\eta_k=\widetilde\kappa_k\widetilde\mu_k
$$

를 만든다.

### 8.3 Centered proximal working update

$\widetilde C$를 $\widetilde\eta$의 centered contrast matrix라 하면

$$
C^\star
=\arg\min_{C:\mathbf1^\top C=0}
\left\{
\frac12\lVert C-\widetilde C\rVert_F^2
+\lambda_\eta\sum_{j=1}^d w_j\lVert c_{\cdot j}\rVert_2
\right\}.
$$

Coordinate별 해는

$$
c_{\cdot j}^\star
=\left(1-\frac{\lambda_\eta w_j}
{\lVert\widetilde c_{\cdot j}\rVert_2}\right)_+
\widetilde c_{\cdot j},
$$

$$
\eta_{kj}^\star
=\widetilde{\bar\eta}_j+c_{kj}^\star.
$$

### 8.4 Step-halving safeguard

위 quadratic working problem이 exact vMF penalized M-step의 majorizer임을
가정하지 않는다. Candidate에서 observed penalized criterion이 감소하면

$$
\eta^{(t+1)}
=\eta^{(t)}+\rho_t(\eta^\star-\eta^{(t)}),
\qquad
\rho_t\in\{1,2^{-1},2^{-2},\ldots\}
$$

로 update를 줄인다. 최대 25회 halving을 허용하고 acceptance 여부를
기록한다.

따라서 알고리즘 명칭은 **proximal EM-type update with step-halving
safeguard**다. 표준 EM의 closed-form M-step이나 일반적인 단조수렴 정리를
주장하지 않는다.

## 9. 선택 후 exact centered-support refit

### 9.1 선택 support

$$
\widehat S_{\mathrm{dec}}
=\left\{j:\lVert\widehat c_{\cdot j}\rVert_2>0\right\}.
$$

Penalty가 만든 shrinkage bias를 줄이기 위해 support를 고정하고 penalty 없이
refit한다.

### 9.2 A refit과 B refit

| refit | $j\notin S$의 제약 | 공통 baseline | 해석 | 현재 역할 |
|:---|:---|:---|:---|:---|
| A: active-only | $\eta_{1j}=\cdots=\eta_{Kj}=0$ | 제거 | contrast와 common effect를 함께 제거 | 과거 결과와의 ablation |
| **B: centered fixed-support** | $c_{\cdot j}=0$ | 유지·재추정 | decision contrast만 제거 | **주 분석** |

B refit의 parameterization은

$$
\eta_{kj}=b_j+c_{kj},
\qquad
\sum_{k=1}^Kc_{kj}=0,
\qquad
c_{kj}=0\quad(j\notin S).
$$

비선택 coordinate도 $b_j=\bar\eta_j$는 추정된다. 따라서 선택되지 않은
coordinate가 data density에서 사라진다는 뜻이 아니라 component 구분
contrast가 없다는 뜻이다.

### 9.3 Exact support 제약 하의 M-step

Responsibilities가 고정되면

$$
Q_S(\eta)
=\sum_{k=1}^K
\left[
\eta_k^\top r_k
+N_k\log C_d(\lVert\eta_k\rVert_2)
\right]
$$

를 $b$와 active contrast에 대해 최대화한다. $A_d(\kappa)$를 mean resultant
function이라 하면

$$
\frac{\partial}{\partial\eta_k}
\log C_d(\lVert\eta_k\rVert_2)
=-A_d(\kappa_k)\mu_k,
$$

따라서 unconstrained row gradient는

$$
\nabla_{\eta_k}Q_S
=r_k-N_kA_d(\kappa_k)\mu_k.
$$

구현은 support 제약을 parameterization으로 정확히 강제하고 L-BFGS-B와
step-halving을 사용한다. 여기서 `exact`는 **support constraint를 정확히
만족한다**는 뜻이며 non-convex mixture likelihood의 전역 maximizer를
뜻하지 않는다.

## 10. Penalty path와 support 선택

### 10.1 Threshold path

Path는 $\lambda_\eta=0$에서 시작하고 warm start를 사용한다. 현재 target에서

$$
u_j=\frac{\lVert\widetilde c_{\cdot j}\rVert_2}{w_j}
$$

를 계산한 뒤 현재 $\lambda_h$보다 큰 가장 가까운 threshold를

$$
\lambda_{h+1}
=\min\{u_j:u_j>\lambda_h+\delta\}
$$

로 선택한다. Support가 거의 비거나 더 큰 threshold가 없거나 최대 path
길이에 도달하면 종료한다.

주 Study B는 path length 240을 사용했다. $d=500$ 고차원 진단은 path 600을
사용했다. 240 대 300 sensitivity에서는 논문 결과를 바꿀 추가 이득이
확인되지 않아 240을 주 길이로 유지했다.

### 10.2 E-ACGL optional path extension

로컬 패키지에는 E-ACGL의 selected support가 path 경계에 놓일 때만
240에서 최대 480으로 확장하는 optional guard를 구현했다. 기본값은 OFF다.
알려진 경계 반복에서 자동 240→480 결과는 수동 path 480과 support,
log-likelihood, $\eta$가 모두 동일했고 최종 경계에서 벗어났다.

이 기능은 path coverage만 바꾸며 penalty와 selector는 바꾸지 않는다.

## 11. BIC, EBIC와 자유도

### 11.1 BIC-after-refit

Path의 고유 support $S$마다 B refit을 수행하고

$$
\operatorname{BIC}^{\mathrm{refit}}(S)
=-2\ell(\widehat\Theta_S^{\mathrm{refit}})
+\log(n)\operatorname{df}(S)
$$

를 계산한다.

$$
\widehat S
=\arg\min_{S\in\mathcal S_{\mathrm{path}}}
\operatorname{BIC}^{\mathrm{refit}}(S).
$$

### 11.2 Practical degrees of freedom

$m=|S|$일 때 주 자유도는

$$
\operatorname{df}_A(S)
=d+(K-1)m+(K-1)\mathbf1(m>0).
$$

| 항 | 의미 |
|:---|:---|
| $d$ | 모든 coordinate의 common natural-parameter baseline $b_j$ |
| $(K-1)m$ | active coordinate의 sum-to-zero centered contrast |
| $(K-1)\mathbf1(m>0)$ | 비퇴화 mixture에서 mixing proportion |

이는 penalized nonregular mixture의 exact effective degrees of freedom가
아니라 support별 비교를 위한 practical approximation이다.

민감도 규칙은

$$
\operatorname{df}_B=Km+(K-1)\mathbf1(m>0),
$$

$$
\operatorname{df}_C=d+Km+(K-1)\mathbf1(m>0)
$$

이다.

### 11.3 EBIC sensitivity

$$
\operatorname{EBIC}_\gamma(S)
=\operatorname{BIC}^{\mathrm{refit}}(S)
+2\gamma\log{d\choose m},
\qquad
\gamma\in\{0.25,0.5,1\}.
$$

Adaptive weight exponent $\gamma=1$과 EBIC의 $\gamma$는 서로 다른
매개변수다.

### 11.4 BIC-before와 BIC-after

$$
\operatorname{BIC}^{\mathrm{pen}}(\lambda)
=-2\ell(\widehat\Theta_\lambda^{\mathrm{pen}})
+\log(n)\operatorname{df}(S_\lambda)
$$

는 penalized fit에서 support를 고르는 BIC-before다. 현재 E-CGL 주 분석은
각 support를 같은 제약 아래 refit한 후 비교하는 BIC-after다. Study B
two-step 진단에서 BIC-before selected q가 17.10--17.30이었던 두 조건은
BIC-after에서 모두 16.05로 감소했다.

### 11.5 대규모 실행의 shortlist guard

대규모 Study B에서는 pre-refit BIC 상위 40개 고유 support를 먼저 B refit하고
선택 support가 경계에 있으면 전체 path로 확장했다. 최종 1,200개 E 계열
fit에서는 full fallback이 발생하지 않았다. Full-path audit은 이 계산
근사의 검증 기준으로 남긴다.

## 12. $K$와 $\lambda_\eta$의 분리 선택

### 12.1 Main simulation에서 $K$를 고정한 이유

Support-recovery simulation은 penalty target의 효과와 $K$ 오지정 효과를
분리하기 위해 true $K$를 고정한다. $K$ 선택은 별도 진단에서 평가한다.

### 12.2 Practical two-step procedure

$$
\widehat K
=\arg\min_{K\in\mathcal K}\mathcal C_{\mathrm{dense}}(K),
\qquad
\widehat\lambda_\eta
=\arg\min_{\lambda_\eta}
\operatorname{BIC}^{\mathrm{refit}}(\widehat K,\lambda_\eta).
$$

1. 후보 $K$마다 dense vMF mixture를 multistart로 적합한다.
2. Held-out/OOB density, 정보지수와 partition stability를 비교한다.
3. 분석 목적에 맞는 component resolution $\widehat K$를 고정한다.
4. 해당 $K$에서 E-CGL path를 적합하고 BIC-after로 $\lambda_\eta$를 고른다.

### 12.3 All-in-one 선택의 문제

$(K,\lambda)$를 하나의 criterion으로 동시에 고르면 component 수 증가와
regularization이 서로 보상할 수 있다. Study B all-in-one 진단에서 E-CGL과
E-ACGL은 주로 $K=6$--8을 선택했다. Rossi and Barbaro의 CSTR 분석도
component 수와 sparsity를 분리하여 다룬다.

### 12.4 $K$는 하나의 보편적 정답이 아닐 수 있다

Classic3에서 held-out density는 $K=10$, bootstrap partition stability는
$K=3$을 선호했다. $K=10$은 broad topic을 혼합한 것이 아니라 각 topic을
세부 component로 나눴다. Density-optimal resolution과 분석 목적의 broad
topic resolution을 구분하여 보고한다.

## 13. 비교 모형과 명칭

### 13.1 내부 비교 모형

| 모형 | penalty | support target | 역할 |
|:---|:---|:---|:---|
| M-L | $\lambda_\mu\sum_{k,j}\lvert\mu_{kj}\rvert$ | prototype entry union | Rossi-type sparse prototype 기준 |
| M-GL | $\lambda_\mu\sum_j\lVert\mu_{\cdot j}\rVert_2$ | prototype coordinate | $\mu$-space group 진단 |
| M-AGL | $\lambda_\mu\sum_jw_j^{(M)}\lVert\mu_{\cdot j}\rVert_2$ | prototype coordinate | adaptive $\mu$-group 진단 |
| E-CL | $\lambda_\eta\sum_{k,j}\lvert c_{kj}\rvert$ | centered-$\eta$ entry union | entry-wise ablation |
| **E-CGL** | $\lambda_\eta\sum_j\lVert c_{\cdot j}\rVert_2$ | posterior decision coordinate | **주 모형** |
| E-ACGL | $\lambda_\eta\sum_jw_j^{(E)}\lVert c_{\cdot j}\rVert_2$ | posterior decision coordinate | adaptive 보조 확장 |

M-L은 Rossi and Barbaro의 sparse directional-prototype target을 재현한
비교 모형이다. M-GL과 M-AGL은 본 연구의 진단 변형이며 해당 논문의 공식
방법이 아니다.

### 13.2 외부 비교 모형

| 모형 | 정의 | 변수 선택 | 역할 |
|:---|:---|:---:|:---|
| Spherical $k$-means | cosine similarity 기반 hard clustering | 아니오 | 방향자료 clustering-only 기준 |
| Dense vMF, shared $\kappa$ | 모든 component가 같은 concentration | 아니오 | 단순 dense likelihood 기준 |
| Dense vMF, free $\kappa_k$ | component별 concentration 추정 | 아니오 | 유연한 dense likelihood 기준 |
| Sparse $k$-means | feature weight가 있는 clustering objective | 예 | 비모형 기반 feature-selection 기준 |
| dbmovMFs | vMF row/column co-clustering | column block | Rossi-style 보조 기준 |

### 13.3 과거 명칭과 현재 명칭

| 과거 표기 | 현재 표기 | 비고 |
|:---|:---|:---|
| D-L, D-GL, D-AGL | M-L, M-GL, M-AGL | $M$은 $\mu$ target을 뜻함 |
| Eta-group | E-CGL | centered Eta coordinate group lasso |
| E-CAGL | E-ACGL | adaptive centered group lasso |
| Eta ANOVA L1 | E-CL과 관련 | 과거 구현은 현재 matched E-CL과 selector/refit이 다름 |
| E-L | E-CL | centered entry-wise $L_1$임을 명확히 함 |

## 14. 시뮬레이션 설계

### 14.1 기본 S1--S6

공통 설정은 $K=4$, $n=1000$, $d=200$,
$(q_C,q_D,q_N)=(4,16,180)$, rep=50, nstart=10, path 240이다.

| 환경 | 방향 차이 | $\kappa$ | 목적 |
|:---|:---:|:---|:---|
| S1 | 큼, 목표 90도 | $(30,40,50,60)$ | 강한 신호·이분산 |
| S2 | 큼, 목표 90도 | $(45,45,45,45)$ | 강한 신호·등분산 |
| S3 | 보통, 목표 60도 | $(30,40,50,60)$ | 보통 신호·이분산 |
| S4 | 보통, 목표 60도 | $(45,45,45,45)$ | 보통 신호·등분산 |
| S5 | 작음, 목표 30도 | $(43,44,46,47)$ | 약한 신호·약한 이분산 |
| S6 | 작음, 목표 30도 | $(45,45,45,45)$ | 약한 신호·등분산 |

### 14.2 Dense-support negative-control

S1-N--S6-N은 방향과 $\kappa$ 구조를 유지하고

$$
(q_C,q_D,q_N)=(4,80,116)
$$

으로 decision support를 조밀하게 만든다. Sparse decision-support 가정이
약해질 때의 과소선택과 zero-support 빈도를 확인한다.

### 14.3 Shared-background

$$
(q_C,q_D,q_N)=(80,20,100).
$$

Prototype-active coordinate 100개 중 decision coordinate는 20개뿐이다.
M 계열과 E 계열의 target 차이를 가장 직접적으로 보여주는 설정이다.

### 14.4 Oracle Bayes error Study B

$$
K=4,
\quad d=200,
\quad n\in\{300,1000\},
\quad(q_C,q_D,q_N)=(4,16,180),
$$

$$
e_B\in\{0.025,0.05,0.10\},
$$

$$
\kappa\in\{(45,45,45,45),(30,40,50,60)\},
\qquad R=100.
$$

Oracle Bayes error는 true parameter로 분류했을 때의 error

$$
e_B
=P_\Theta\left[
\arg\max_k P_\Theta(Z=k\mid X)\neq Z
\right]
$$

로 정의한다. 같은 평균 각도라도 $\kappa$가 다르면 실제 overlap이 달라질
수 있으므로, common-to-decision signal ratio를 조절하여 목표 $e_B$를
맞춘다.

달성된 error는 다음과 같다.

| target $e_B$ | equal $\kappa$ | heterogeneous $\kappa$ |
|---:|---:|---:|
| 2.5% | 2.33% | 2.73% |
| 5.0% | 5.14% | 5.00% |
| 10.0% | 10.09% | 9.84% |

### 14.5 고차원 확장

$$
n=1000,
\quad d=500,
\quad(q_C,q_D,q_N)=(10,40,450),
\quad e_B=5\%,
\quad\text{path}=600.
$$

모든 unique support를 exact B refit하여 짧은 path truncation을 분리했다.

## 15. 평가 지표

### 15.1 Support decomposition

True decision support를 $S_D$라 하고 추정 support를 $\widehat S$라 하면

$$
TP=|\widehat S\cap S_D|,
\quad
FP=|\widehat S\setminus S_D|,
\quad
FN=|S_D\setminus\widehat S|,
$$

$$
TN=d-TP-FP-FN.
$$

공통·decision·noise 선택 수는

$$
\widehat q_C=|\widehat S\cap S_C|,
\quad
\widehat q_D=|\widehat S\cap S_D|,
\quad
\widehat q_N=|\widehat S\cap S_N|.
$$

### 15.2 Support recovery metrics

$$
\operatorname{TPR}=\frac{TP}{TP+FN},
\qquad
\operatorname{FPR}=\frac{FP}{FP+TN},
$$

$$
\operatorname{Precision}=\frac{TP}{TP+FP},
\qquad
F_1=\frac{2TP}{2TP+FP+FN}.
$$

M 계열도 표에서는 decision support를 정답으로 평가하므로 공통 coordinate를
선택하면 false positive로 계산된다. 이는 M 계열의 prototype target 자체가
잘못되었다는 뜻이 아니라 본 연구의 decision estimand와 다른 데서 생기는
차이다.

### 15.3 Centered natural-parameter error

True/estimated component를 label matching한 뒤

$$
\operatorname{MSE}_{\eta^c}
=\frac1{Kd}\sum_{k=1}^K\sum_{j=1}^d
(\widehat c_{kj}-c_{kj})^2
$$

를 계산한다. 문서 표의 `MSE_eta`는 이 값이다.

### 15.4 Clustering과 prediction

ARI와 NMI는 label permutation에 불변이다. Component별 $\mu$, $\kappa$,
$\eta$ error는 estimated component와 true component를 permutation matching한
뒤 계산한다.

Held-out negative log-likelihood는

$$
\operatorname{NLL}_{\mathrm{test}}
=-\frac1{n_{\mathrm{test}}}
\sum_{i\in\mathrm{test}}
\log\widehat p(x_i)
$$

이며 작은 값이 더 높은 predictive density를 뜻한다. 연속 vMF density의
NLL은 음수일 수 있다.

### 15.5 Conditional과 unconditional F1

모든 반복을 합친 micro F1은

$$
F_{1,\mathrm{all}}
=\frac{2\sum_rTP_r}
{2\sum_rTP_r+\sum_rFP_r+\sum_rFN_r}.
$$

Nonzero support 또는 valid refit 반복 집합 $V$만 사용하면

$$
F_{1,\mathrm{valid}}
=\frac{2\sum_{r\in V}TP_r}
{2\sum_{r\in V}TP_r+\sum_{r\in V}FP_r+\sum_{r\in V}FN_r}.
$$

Zero-support 반복을 제외한 conditional F1만 보고하면 tuning failure가
숨겨질 수 있다. 현재 어려운 조건은 unconditional F1, valid-refit rate와
zero-support rate를 함께 보고한다.

## 16. 핵심 시뮬레이션 근거

### 16.1 Matched penalty/refit ablation

E-CL, E-CGL, E-ACGL에 같은 path 240, BIC-after와 exact B refit을 적용한
rep=20 결과다.

| 환경 | 모형 | selected q | common | decision | noise | F1 | ARI | MSE$_{\eta^c}$ |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| S1, $n=1000$ | E-CL | 16.05 | 0.00 | 16.00 | 0.05 | 0.998 | 0.860 | 0.058 |
|  | **E-CGL** | 16.05 | 0.00 | 16.00 | 0.05 | 0.998 | 0.859 | 0.058 |
|  | E-ACGL | 16.05 | 0.00 | 16.00 | 0.05 | 0.998 | 0.860 | 0.058 |
| Study B hard, $n=300$ | E-CL | 15.95 | 0.00 | 14.95 | 1.00 | 0.935 | 0.695 | 0.516 |
|  | **E-CGL** | 16.00 | 0.00 | 15.15 | 0.85 | 0.947 | 0.699 | 0.595 |
|  | E-ACGL | 15.70 | 0.00 | 15.10 | 0.60 | 0.953 | 0.706 | 0.431 |

S1에서는 세 penalty가 사실상 같았다. Hard cell에서 group 계열의 평균 F1이
높았지만 차이는 MCSE와 비슷했다. Group 구조의 근거는 보편적 우월성보다
coordinate-level selection target과의 일치다.

### 16.2 Shared-background

| 모형 | selected q | common | decision | noise | F1 | ARI | MSE$_{\eta^c}$ |
|:---|---:|---:|---:|---:|---:|---:|---:|
| M-L | 199.02 | 80.00 | 20.00 | 99.02 | 0.183 | 0.591 | 0.993 |
| M-GL | 102.92 | 80.00 | 20.00 | 2.92 | 0.325 | 0.638 | 0.495 |
| M-AGL | 102.40 | 79.96 | 20.00 | 2.44 | 0.327 | 0.638 | 0.492 |
| E-CL | 198.98 | 79.56 | 20.00 | 99.42 | 0.183 | 0.590 | 0.997 |
| **E-CGL** | **20.26** | **0.10** | **20.00** | **0.16** | **0.994** | **0.677** | **0.093** |
| E-ACGL | 20.18 | 0.10 | 20.00 | 0.08 | 0.996 | 0.677 | 0.091 |

M-group은 prototype에 존재하는 공통 coordinate 80개를 유지했다. E-CGL은
decision coordinate 20개를 유지하고 공통·noise coordinate를 거의
제외했다. Centering의 target 차이를 가장 직접적으로 보여주는 결과다.

### 16.3 Study B rep=100: E 계열 요약

등분산·이분산 $\kappa$ 결과의 동일 가중 평균이다.
수치는 exact centered-support B refit을 사용한 canonical 통합 결과
`results/paper_eta_studyb_v2_refitB_guard40_all6_rep100_260712/`
기준이다. 같은 이름의 초기 `oracle_bayes_studyb_difficulty_rep100` summary는
B refit 전 단계의 과거 결과이므로 최종 표의 출처로 사용하지 않는다.

| $e_B$ | $n$ | 모형 | selected q | common | decision | noise | F1 | ARI | MSE$_{\eta^c}$ |
|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| 2.5% | 300 | **E-CGL** | 16.51 | 0.01 | 16.00 | 0.50 | 0.987 | 0.931 | 0.204 |
|  | 300 | E-ACGL | 17.99 | 0.03 | 16.00 | 1.96 | 0.960 | 0.930 | 0.250 |
|  | 1000 | **E-CGL** | 16.05 | 0.00 | 16.00 | 0.05 | 0.999 | 0.934 | 0.053 |
|  | 1000 | E-ACGL | 16.03 | 0.00 | 16.00 | 0.03 | 0.999 | 0.934 | 0.053 |
| 5.0% | 300 | **E-CGL** | 16.24 | 0.00 | 16.00 | 0.24 | 0.993 | 0.856 | 0.210 |
|  | 300 | E-ACGL | 16.80 | 0.01 | 16.00 | 0.79 | 0.981 | 0.856 | 0.231 |
|  | 1000 | **E-CGL** | 16.08 | 0.00 | 16.00 | 0.08 | 0.998 | 0.869 | 0.057 |
|  | 1000 | E-ACGL | 16.08 | 0.00 | 16.00 | 0.08 | 0.998 | 0.869 | 0.057 |
| 10.0% | 300 | **E-CGL** | 17.38 | 0.03 | 15.61 | 1.75 | 0.946 | 0.705 | 0.523 |
|  | 300 | E-ACGL | 17.72 | 0.03 | 15.52 | 2.17 | 0.951 | 0.706 | 0.486 |
|  | 1000 | **E-CGL** | 17.05 | 0.04 | 16.00 | 1.02 | 0.981 | 0.748 | 0.075 |
|  | 1000 | E-ACGL | 16.18 | 0.01 | 16.00 | 0.17 | 0.995 | 0.748 | 0.067 |

$n=1000$에서 E-CGL은 세 난이도 모두 decision 16개를 유지했고 common
선택은 0.00--0.04개였다. $e_B$ 증가에 따라 ARI는 감소했지만 support F1은
0.981 이상이었다. $n=300$, $e_B=10\%$에서는 decision 누락과 noise 선택이
증가했다.

전체 여섯 모형의 rep=100 결과는
`docs/simulations/thesis-simulation_260708.md`의 Study B 표에 있다.

### 16.4 고차원 $d=500$

Rep=50에서 E-CGL은 selected q=40.04, decision q=40.00, noise q=0.04,
F1=1.000이었다. E-ACGL은 selected q=42.60, noise q=2.46, F1=0.976이었다.
초기 dense 결과는 refit보다 짧은 path에 따른 truncation으로 확인되었다.

### 16.5 불리한 조건

| 조건 | M 계열 기준 | E-CGL | 원인 |
|:---|:---|:---|:---|
| S3-N, 조밀 decision | M-AGL F1=0.877 | F1=0.800 | dense support에서 group shrinkage 비용 |
| S4-N | M-group은 common 4개 중심 | F1=0.922, zero 3/50 | 일부 반복의 BIC null 선택 |
| S5 | M-L F1=0.149, dense | F1=0.118, zero 40/50 | 매우 약한 contrast와 과소선택 |
| S6 | M-L F1=0.148, dense | F1=0.002, zero 49/50 | 약한 등분산 신호에서 null support 집중 |
| S5-N/S6-N | M-L F1 약 0.57 | F1=0.000 | 조밀하지만 약한 decision contrast |

M-L의 dense support F1이 E-CGL보다 높아 보이는 약한 조건도 있지만 ARI는
두 계열 모두 매우 낮다. 이는 M-L이 decision coordinate를 선별했다기보다
거의 모든 coordinate를 유지하여 TPR을 확보한 결과다.

## 17. $K$ 선택 진단 결과

### 17.1 Study B all-in-one

| 방법 | equal $\kappa$ | heterogeneous $\kappa$ |
|:---|:---|:---|
| M-GL/M-AGL | 대부분 또는 전부 $K=4$ | 전부 $K=4$ |
| E-CGL all-in-one | 주로 $K=6$--8 | 주로 $K=6$--8 |
| E-ACGL all-in-one | 주로 $K=6$--8 | 주로 $K=7$--8 |

E 계열의 penalty와 component 수를 동시에 선택하면 큰 $K$가 선호되었다.

### 17.2 Dense first-stage rep=20

| 기준 | equal $\kappa$ | heterogeneous $\kappa$ |
|:---|:---|:---|
| BIC | $K=4$: 13/20 | $K=3$: 20/20 |
| RICc | $K=2$: 20/20 | $K=2$: 20/20 |
| EBIC$_{0.5}$, EBIC$_1$ | $K=2$: 20/20 | $K=2$: 20/20 |
| ICL-BIC | $K=4$: 12/20 | $K=3$: 20/20 |
| independent test NLL | $K=4$: 20/20 | $K=4$: 20/20 |

제한된 bootstrap OOB NLL minimum과 1-SE도 두 조건에서 $K=4$를 3/3회
선택했다. Pairwise stability는 heterogeneous 조건에서 $K=2$를 선호하여
단독 기준으로 사용하지 않는다.

### 17.3 연결된 two-step rep=20

Independent test NLL로 $K=4$를 고정하고 E-CGL을 적합한 결과다.

| $\kappa$ | selected q | common | decision | noise | F1 | ARI | MSE$_{\eta^c}$ |
|:---|---:|---:|---:|---:|---:|---:|---:|
| equal | 16.05 | 0.00 | 16.00 | 0.05 | 0.998 | 0.861 | 0.060 |
| heterogeneous | 16.05 | 0.00 | 16.00 | 0.05 | 0.998 | 0.869 | 0.060 |

## 18. Classic3 실자료

### 18.1 자료와 SPLADE 표현

Classic3는 CISI, CRAN, MED 초록 3,890건으로 구성된다. Train 3,111건,
test 779건으로 층화 분할하고 train 분산 상위 SPLADE vocabulary coordinate
2,000개를 사용했다. 문서는 unit $L_2$ norm으로 정규화했다.

SPLADE는 dense LLM embedding이 아니라 pretrained language model 기반
sparse lexical representation이다.

| 표현 | 장점 | 제약 |
|:---|:---|:---|
| TF-IDF | 관측 token에 직접 대응 | 문맥 기반 lexical expansion이 없음 |
| Dense embedding | 의미 유사성을 압축 | latent coordinate라 선택 좌표의 token 해석이 어려움 |
| SPLADE | 문맥 정보를 반영하면서 coordinate가 vocabulary token에 대응 | expansion token이 원문에 그대로 없을 수 있음 |

Label은 train/test 층화와 사후 ARI/NMI 계산에만 사용했고 fitting, support
선택과 초기화에는 사용하지 않았다.

### 18.2 Held-out 결과, $K=3$

| 모형 | selected q | test NLL/document | test ARI | test NMI |
|:---|---:|---:|---:|---:|
| Spherical $k$-means | 2,000 | NA | 0.9856 | 0.9710 |
| Dense vMF, shared $\kappa$ | 2,000 | -4871.6918 | 0.9856 | 0.9710 |
| Dense vMF, free $\kappa_k$ | 2,000 | **-4872.9015** | **0.9927** | **0.9863** |
| M-L | 2,000 | -4871.0937 | 0.9892 | 0.9787 |
| **E-CGL** | **1,347** | -4872.2942 | **0.9927** | **0.9863** |
| E-ACGL | 1,348 | -4872.2981 | 0.9927 | 0.9863 |

E-CGL은 coordinate 32.7%를 제거하고 dense free-$\kappa_k$ vMF와 같은
test ARI/NMI를 유지했다. Dense vMF의 test density는 더 높았으므로 이
결과는 density 최댓값이 아니라 clustering을 유지한 decision-coordinate
축약으로 해석한다.

20회 재선택에서 E-CGL의 selected q는 1343.9(SD 16.8), Nogueira
stability는 0.884, 평균 support Jaccard는 0.927이었다.

### 18.3 Token의 부호 해석

$$
(\widehat\eta_k-\bar{\widehat\eta})^\top x
=\sum_{j=1}^d\widehat c_{kj}x_j,
\qquad x_j\ge0.
$$

| class | $\widehat c_{kj}>0$: 평균보다 score 증가 | $\widehat c_{kj}<0$: 평균보다 score 감소 |
|:---|:---|:---|
| CISI | `library`, `information`, `librarian` | `flow`, `pressure`, `effect` |
| CRAN | `flow`, `mach`, `pressure` | `library`, `information`, `librarian` |
| MED | `tumor`, `inhibitor`, `dose` | `library`, `information`, `flow` |

양·음은 component 평균을 기준으로 한 상대적 linear score contribution이다.
Token 자체의 절대 선호·배척이나 인과효과를 뜻하지 않는다. 20회
재선택에서 표의 30개 class-token 부호가 모두 유지되었고 class별 전체
absolute contrast 순위의 평균 Spearman 상관은 0.950--0.958이었다.

### 18.4 Classic3의 $K$ 해상도

| $K$ | selected q | test NLL | test ARI | purity | completeness |
|---:|---:|---:|---:|---:|---:|
| 3 | 1,347 | -4872.294 | **0.993** | 0.997 | 0.986 |
| 7 | 1,105 | -4905.826 | 0.585 | 0.996 | 0.586 |
| 8 | 1,063 | -4910.602 | 0.493 | 0.996 | 0.536 |
| 10 | 980 | **-4917.546** | 0.398 | 0.992 | 0.475 |

$K=10$은 CISI 3개, CRAN 5개, MED 2개의 세부 component로 분할했다.
주 실자료는 제공된 broad topic을 설명하는 $K=3$ 조건부 분석이며, $K=3$을
유일한 내재 component 수로 주장하지 않는다.

## 19. 실자료 적용 범위

### 19.1 BBC5

| 모형 | selected q | test NLL | test ARI | test NMI |
|:---|---:|---:|---:|---:|
| Dense free $\kappa_k$ | 1,000 | **-2126.228** | **0.896** | **0.874** |
| M-L | 1,000 | -2124.803 | 0.889 | 0.867 |
| E-CGL | 679 | -2124.753 | 0.885 | 0.862 |
| E-ACGL | 691 | -2124.867 | 0.885 | 0.862 |

E-CGL은 32.1%를 제거했지만 dense vMF보다 test ARI가 0.0109 낮고 NLL이
문서당 1.4750 높았다. Decision 정보가 더 많은 coordinate에 분산된
중간·조밀 support 가능성과 일관되는 결과다.

### 19.2 CSTR

| 모형 | selected q | ARI 평균(SD) | NMI 평균 |
|:---|---:|---:|---:|
| Dense shared $\kappa$ | 1,000.0 | 0.8023 (0.0087) | 0.7650 |
| **Rossi M-L** | 888.7 | **0.8083 (0.0079)** | **0.7703** |
| E-CGL | 311.1 | 0.6153 (0.0065) | 0.6449 |
| E-ACGL | 313.3 | 0.6066 (0.0109) | 0.6401 |

Rossi 논문의 dense ARI 0.804와 M-L ARI 0.808을 각각 0.8023과 0.8083으로
근접 재현했다. CSTR에서는 prototype-oriented 또는 dense lexical support가
E-CGL의 sparse decision support보다 적합했다.

실자료에는 true feature support가 없으므로 TPR, FPR, Precision과 F1을
보고하지 않는다. 구조적 원인에 대한 설명은 성능 차이와 일관되는 해석이며
data-generating mechanism을 식별한 결과는 아니다.

## 20. 계산 시간과 Rcpp 검증

### 20.1 단일 반복 시간

$K=4$, $n=1000$, $d=200$, nstart=10, path 240, max_iter=100의 diagnostic
timing이다.

| 모형 | sec | selected q | ARI | F1 | MSE$_{\eta^c}$ |
|:---|---:|---:|---:|---:|---:|
| M-L | 3.750 | 200 | 0.859 | 0.148 | 0.698 |
| M-GL | 8.820 | 20 | 0.895 | 0.889 | 0.074 |
| M-AGL | 8.530 | 20 | 0.895 | 0.889 | 0.074 |
| E-CL | 8.530 | 200 | 0.868 | 0.148 | 0.697 |
| **E-CGL** | **5.620** | **16** | **0.897** | **1.000** | **0.054** |
| E-ACGL | 5.530 | 16 | 0.897 | 1.000 | 0.054 |

단일 반복 결과이므로 이론적 복잡도 순위로 해석하지 않는다. Runtime은
EM 반복 수, path 길이, line search, refit candidate 수와 nstart의 영향을
받는다.

### 20.2 R-only 대 Rcpp-helper 동일성

| 검증 | 결과 |
|:---|---:|
| rep=20 raw rows | 120 vs 120 |
| summary rows | 6 vs 6 |
| character columns | identical |
| raw 최대 수치 차이 | $1.019\times10^{-10}$ |
| summary 최대 수치 차이 | $9.823\times10^{-11}$ |
| 차이 $>10^{-8}$ | 0 |
| 판정 | PASS |

Rcpp는 같은 수식의 low-level helper를 대체하는 구현 가속이며 다른 통계
알고리즘이 아니다. R-only fallback을 유지한다.

### 20.3 반복 benchmark

$K=4$, $n=300$, $d=60$, rep=50, nstart=3, path 40의 warm-cache 결과다.

| mode | repeats | mean sec | SD | median sec | min--max |
|:---|---:|---:|---:|---:|:---|
| R-only | 3 | 59.737 | 0.342 | 59.860 | 59.35--60.00 |
| Rcpp-helper | 3 | 25.387 | 0.031 | 25.380 | 25.36--25.42 |

Median OFF/ON ratio는 2.359이고 이 설정에서 elapsed time은 약 57.6%
감소했다. 이는 diagnostic benchmark이며 일반적인 publication speed claim은
아니다.

## 21. 로컬 R 패키지 검증 상태

패키지 검증은 출판 결과와 분리된 local-only 결과다.

### 21.1 Study B $e_B=5\%$, rep=20

| $n$ | 모형 | q | common | decision | noise | F1 | ARI | MSE$_{\eta^c}$ | exact support |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | M-L | 190.70 | 4.00 | 16.00 | 170.70 | 0.155 | 0.764 | 2.7115 | 0.000 |
| 300 | M-GL | 20.35 | 4.00 | 16.00 | 0.35 | 0.881 | 0.852 | 0.2653 | 0.000 |
| 300 | M-AGL | 20.25 | 4.00 | 16.00 | 0.25 | 0.883 | 0.851 | 0.2603 | 0.000 |
| 300 | E-CL | 198.20 | 3.95 | 16.00 | 178.25 | 0.149 | 0.727 | 3.0147 | 0.000 |
| 300 | **E-CGL** | 16.10 | 0.00 | 16.00 | 0.10 | 0.997 | 0.857 | 0.2099 | 0.900 |
| 300 | E-ACGL | 16.10 | 0.00 | 16.00 | 0.10 | 0.997 | 0.858 | 0.2098 | 0.900 |
| 1000 | M-L | 187.10 | 4.00 | 16.00 | 167.10 | 0.158 | 0.834 | 0.7370 | 0.000 |
| 1000 | M-GL | 20.05 | 4.00 | 16.00 | 0.05 | 0.888 | 0.860 | 0.0738 | 0.000 |
| 1000 | M-AGL | 20.00 | 4.00 | 16.00 | 0.00 | 0.889 | 0.860 | 0.0723 | 0.000 |
| 1000 | E-CL | 198.25 | 3.90 | 16.00 | 178.35 | 0.149 | 0.832 | 0.7515 | 0.000 |
| 1000 | **E-CGL** | 16.05 | 0.00 | 16.00 | 0.05 | 0.998 | 0.861 | 0.0592 | 0.950 |
| 1000 | E-ACGL | 16.15 | 0.00 | 16.00 | 0.15 | 0.995 | 0.860 | 0.0609 | 0.850 |

모든 240 method-repetition이 PASS했고 error row는 없었다. E-ACGL의 자동
path extension은 $n=1000$에서 발동하지 않았다. E-CGL이 주 모형이고
E-ACGL이 adaptive sensitivity라는 기존 결론과 일치했다.

### 21.2 남은 구현 비교상 주의점

현재 패키지의 M 계열은 historical BIC-before/legacy refit 규칙을, E 계열은
BIC-after/exact B refit을 사용한다. 이는 각 support target의 기존 구현을
보존한 것이지만 selector/refit을 완전히 동일하게 맞춘 penalty-only 비교는
아니다. Penalty 효과의 단독 비교는 16.1절의 matched E-CL/E-CGL/E-ACGL
ablation을 사용한다.

## 22. 방법의 특징

| 특징 | 수학적 근거 | 실증 근거 |
|:---|:---|:---|
| Posterior-score target | $\eta_k=\kappa_k\mu_k$ | Same $\mu$/different $\kappa$ 진단, Study B |
| Common/decision 분리 | $\eta=\bar\eta\mathbf1+C$ | Shared-background common 80개 제외 |
| Coordinate-level selection | $\sum_j\lVert c_{\cdot j}\rVert_2$ | Decision q 중심 support |
| Label-invariant support | $\lVert Pc_{\cdot j}\rVert_2=\lVert c_{\cdot j}\rVert_2$ | Label matching 없이 support 평가 |
| Common baseline 유지 | B refit에서 $c_j=0$, $b_j$ 재추정 | Exact constraint audit PASS |
| Token-level 해석 | $c_{kj}x_j$의 signed contribution | Classic3 class-token contrast |
| 계산 가속 | 동일 helper의 Rcpp 구현 | $10^{-8}$ tolerance equality PASS |

## 23. 비용과 한계

### 23.1 통계적 한계

| 조건 | 발생 원인 | 관찰된 결과 |
|:---|:---|:---|
| 매우 약한 contrast | BIC의 likelihood gain이 df penalty보다 작음 | S5/S6 zero support 집중 |
| 조밀 decision support | 많은 active group에 대한 shrinkage와 df 비용 | S3-N에서 M-group보다 낮은 F1 |
| 작은 $n$, 큰 $d$ | centered contrast와 $\kappa_k$ 추정 변동 | Hard Study B에서 noise 선택 증가 |
| Prototype-oriented target | E-CGL과 estimand 불일치 | CSTR에서 M-L 우세 |
| Distributed lexical signal | sparse decision support 가정 약화 | BBC5에서 ARI/NLL 손실 |

### 23.2 모형 선택 한계

- Practical BIC df는 exact effective df가 아니다.
- $K$와 $\lambda$의 all-in-one 선택은 큰 $K$를 선호할 수 있다.
- Dense information criterion은 고차원에서 작은 $K$를 선호할 수 있다.
- Held-out density는 broad topic보다 세부 component를 선호할 수 있다.
- Path가 짧으면 경계 support가 선택될 수 있어 boundary audit이 필요하다.

### 23.3 최적화 한계

- Mixture likelihood는 non-convex하며 local optimum 가능성이 있다.
- Proximal path update는 exact penalized M-step이 아니다.
- Step-halving은 수치 safeguard이며 일반 convergence theorem을 대신하지 않는다.
- Exact B refit의 `exact`는 support 제약에 관한 표현이다.
- Multistart와 warm start는 안정성을 높이지만 전역 해를 보장하지 않는다.

### 23.4 계산 비용

- Path의 각 후보에 EM-type fitting이 필요하다.
- BIC-after는 고유 support별 constrained refit 비용이 추가된다.
- E-ACGL은 initial fit과 weight 계산 및 경계 path sensitivity가 추가된다.
- Sparse 실자료에서 현재 Rcpp E-step은 dense matrix 최적화에 맞춰져 있다.

### 23.5 해석 한계

- $c_{kj}>0$과 $c_{kj}<0$은 component 평균 대비 상대적 score 기여다.
- 선택 token은 원인 변수나 인과효과가 아니다.
- SPLADE token은 lexical expansion일 수 있다.
- 실자료에는 true support가 없어 support recovery를 직접 검증할 수 없다.

## 24. 주장과 근거의 경계

| 문장 | 사용 여부 | 근거 또는 이유 |
|:---|:---:|:---|
| E-CGL은 posterior decision support를 직접 target한다 | 사용 | Score difference와 Proposition 3 |
| Centering은 pairwise linear score contrast를 보존한다 | 사용 | $\eta_k-\eta_\ell=c_k-c_\ell$ |
| Group penalty는 coordinate estimand와 정렬된다 | 사용 | Group norm과 pairwise identity |
| E-CGL은 모든 clustering에서 가장 좋다 | 사용하지 않음 | S3-N, S5/S6, CSTR 반례 |
| Group $L_2$가 entry-wise $L_1$보다 항상 좋다 | 사용하지 않음 | Matched ablation에서 차이가 작음 |
| BIC df가 exact하다 | 사용하지 않음 | Practical approximation |
| 알고리즘이 전역 해에 수렴한다 | 사용하지 않음 | Non-convex mixture, proximal working update |
| Rcpp가 새로운 방법이다 | 사용하지 않음 | 동일 알고리즘의 helper 가속 |
| Classic3의 진짜 $K$는 3이다 | 사용하지 않음 | Density는 $K=10$, stability는 $K=3$ |

## 25. 질문 대응 요약

| 질문 | 답변 |
|:---|:---|
| 왜 $\mu$가 아니라 $\eta$인가? | Posterior linear score coefficient가 $\kappa_k\mu_k=\eta_k$이기 때문이다. |
| $\eta$에서 $\mu,\kappa$를 복원할 수 있는가? | $\eta\neq0$이면 norm과 normalization으로 유일하게 복원된다. |
| 왜 center하는가? | Component 공통 baseline과 pairwise score contrast를 분리하기 위해서다. |
| 공통 coordinate는 모형에서 없어지는가? | Decision contrast만 0이고 common baseline은 B refit에서 남는다. |
| 왜 group $L_2$인가? | 선택 대상이 component entry가 아니라 coordinate 전체이기 때문이다. |
| Group penalty가 음수를 보존하는가? | 선택된 vector의 상대 방향을 함께 축소하지만 음수를 새로 만들거나 부호를 강제하지 않는다. |
| Centered $\mu$는 불가능한가? | 계산 가능하지만 $\kappa$를 누락해 posterior coefficient의 직접 estimand가 아니다. |
| E-CL과 결과가 비슷한데 E-CGL이 필요한가? | 보편적 성능 우위가 아니라 coordinate support 정의와 penalty 단위의 일치가 근거다. |
| E-ACGL을 주 모형으로 두지 않는 이유는? | E-CGL을 일관되게 개선하지 않고 초기 weight에 추가로 의존한다. |
| A와 B refit의 차이는? | A는 비선택 $\eta$ 전체를 0으로, B는 contrast만 0으로 하고 common baseline을 재추정한다. |
| B refit에서도 sparse selection인가? | $S_{\mathrm{dec}}$는 sparse하며 비선택 coordinate의 공통 baseline만 남을 수 있다. |
| BIC는 언제 계산하는가? | E-CGL 주 분석은 각 support를 B refit한 뒤 observed likelihood로 계산한다. |
| 자유도에서 왜 $d$를 세는가? | 비선택 coordinate를 포함한 모든 common baseline $b_j$를 추정하기 때문이다. |
| $K$와 $\lambda$를 왜 분리하는가? | Component 수와 regularization이 서로 보상하는 all-in-one 현상을 줄이기 위해서다. |
| Label switching은 어떻게 처리하는가? | Support/ARI/NMI는 invariant이고 component MSE만 permutation matching한다. |
| Zero-support F1은 어떻게 처리하는가? | Unconditional F1과 zero-support rate를 함께 보고한다. |
| 왜 Classic3가 주 실자료인가? | Clustering을 유지하면서 coordinate 축약과 signed token 해석을 함께 확인했다. |
| CSTR에서 왜 낮은가? | Prototype-oriented 또는 dense lexical support가 더 맞는 조건으로 관찰되었다. |
| E-CGL이 더 빠른가? | 해당 diagnostic에서는 중간 수준이었지만 runtime은 path·iteration·refit 수에 좌우된다. |
| Rcpp가 결과를 바꾸는가? | 검증 오차가 약 $10^{-10}$이고 $10^{-8}$ 초과 차이는 없었다. |

## 26. 연구미팅에서 확인할 사항

1. 논문의 주 estimand를 `posterior decision support`로 명시하는 구성
2. E-CGL을 주 모형, E-ACGL을 adaptive sensitivity로 두는 구성
3. Exact centered-support B refit과 BIC-after를 주 분석으로 두는 구성
4. Practical df를 명시하고 EBIC·df 민감도를 부록에 두는 구성
5. $K$와 $\lambda_\eta$를 분리하는 two-step 절차
6. Classic3를 주 실자료, BBC5와 CSTR을 적용 범위 사례로 두는 구성
7. Formal theory의 범위를 score geometry와 penalty alignment로 한정하는 구성

## 27. 발표 중 빠른 참조 순서

| 질문 주제 | 확인할 절 |
|:---|:---:|
| 연구 목적·estimand | 0--1 |
| vMF와 posterior score | 2--3 |
| Centering과 공통 coordinate | 4 |
| Group penalty와 adaptive weight | 5--6 |
| 이론 명제 | 7 |
| 최적화와 safeguard | 8 |
| A/B refit | 9 |
| Path·BIC·EBIC | 10--11 |
| $K$ 선택 | 12, 17 |
| 비교 모형 | 13 |
| 시뮬레이션 설계·지표 | 14--15 |
| 시뮬레이션 결과·한계 | 16 |
| Classic3·BBC5·CSTR | 18--19 |
| Runtime·Rcpp·패키지 | 20--21 |
| 주장 범위·질문 답변 | 22--25 |

## 28. 근거 문서와 결과

### 핵심 문서

- `docs/meetings/thesis-meeting_260708.md`
- `docs/meetings/thesis-meeting_260714_presentation_script.md`
- `docs/manuscript/methods_draft_260714.md`
- `docs/manuscript/theory_lemmas_draft_260714.md`
- `docs/simulations/thesis-simulation_260708.md`
- `docs/simulations/final_simulation_framework_260714.md`
- `docs/manuscript/thesis-realdata_260714.md`

### 핵심 검증 결과

- `results/submission_qa_260714/submission_quantitative_qa.md`
- `results/paper_eta_studyb_v2_refitB_guard40_all6_rep100_260712/studyb_guard40_all6_rep100_summary.csv`
- `results/paper_eta_exactB_shared_background_rep50_path240_short40_260712/paper_eta_exactB_shared_background_rep50_path240_short40_260712_summary.csv`
- `results/bic_df_audit_260708/bic_df_posthoc_audit_notes.md`
- `results/exact_centered_refit_validation_260711/exact_centered_refit_validation_notes.md`
- `results/rcpp_switch_validation_rep20_260708/rcpp_switch_rep20_validation_notes.md`
- `results/rcpp_vs_r_runtime_benchmark_rep50_260708/runtime_benchmark_notes.md`
- `results/realdata_final_validation_260711/realdata_final_validation_notes.md`
- `local_package/validation/n300_n1000_comparison_260713/report.md`
- `local_package/validation/eacgl_auto_extend_regression_260713/report.md`

### 주요 참고문헌

- Banerjee, A., Dhillon, I. S., Ghosh, J., and Sra, S. (2005). Clustering on the unit hypersphere using von Mises-Fisher distributions. *Journal of Machine Learning Research*, 6, 1345--1382.
- Bondell, H. D. and Reich, B. J. (2009). Simultaneous factor selection and collapsing levels in ANOVA. *Biometrics*, 65, 169--177.
- Guo, J., Levina, E., Michailidis, G., and Zhu, J. (2010). Pairwise variable selection for high-dimensional model-based clustering. *Biometrics*, 66, 793--804.
- Li, Y. et al. (2022). Pursuing sources of heterogeneity in modeling clustered population. *Biometrics*, 78, 716--728.
- Parikh, N. and Boyd, S. (2014). Proximal algorithms. *Foundations and Trends in Optimization*, 1, 127--239.
- Rossi, F. and Barbaro, F. (2022). Mixture of von Mises-Fisher distribution with sparse prototypes. *Neurocomputing*, 501, 41--74.
- Yuan, M. and Lin, Y. (2006). Model selection and estimation in regression with grouped variables. *JRSS Series B*, 68, 49--67.
