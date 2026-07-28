# E-CGL/E-ACGL 및 M-CGL/M-ACGL 알고리즘 상세 설명

> 연구미팅 백업자료
> 작성일: 2026-07-28

## 1. 핵심 구분

두 방법은 같은 vMF 혼합모형에서 출발하지만 서로 다른 좌표 이질성을 선택한다.

| 방법 | 직접 추정하는 모수 | 선택 대상 |
|---|---|---|
| E-CGL | 자연모수 $\eta_k = \kappa_k\mu_k$ | posterior-score 이질성 |
| E-ACGL | 자연모수 $\eta_k = \kappa_k\mu_k$ | adaptive posterior-score 이질성 |
| M-CGL | 평균방향 $\mu_k$ | 방향 이질성 |
| M-ACGL | 평균방향 $\mu_k$ | adaptive 방향 이질성 |

E-CGL에서 좌표 $j$가 제거된다는 것은 해당 좌표가 모든 군집의 posterior 선형 점수에서 소거된다는 뜻이다. M-CGL에서 좌표 $j$가 제거된다는 것은 해당 좌표의 평균방향이 모든 군집에서 같다는 뜻이다.

---

## 2. 공통 vMF 혼합모형

관측치 $x_i\in\mathbb R^d$는 단위구면 위에 있다.

$$
\lVert x_i\rVert_2 = 1
$$

잠재 군집이 $Z_i = k$일 때의 조건부분포는 다음과 같다.

$$
x_i\mid Z_i = k \sim \mathrm{vMF}(\mu_k,\kappa_k)
$$

$$
f(x_i\mid\mu_k,\kappa_k) = C_d(\kappa_k) \exp\{\kappa_k\mu_k^\top x_i\}
$$

모수의 의미는 다음과 같다.

- $\mu_k\in\mathbb S^{d-1}$: 군집 $k$의 평균방향
- $\lVert\mu_k\rVert_2 = 1$: 단위구면 제약
- $\kappa_k\ge0$: 군집 $k$의 집중도
- $\alpha_k = P(Z_i = k)$: 혼합비율
- $C_d(\kappa_k)$: vMF 정규화 상수

$\kappa_k = 0$이면 구면 위에서 균일분포에 가깝고, $\kappa_k$가 커질수록 관측치가 $\mu_k$ 주위에 조밀하게 모인다.

자연모수는 다음과 같다.

$$
\eta_k = \kappa_k\mu_k
$$

역관계는 다음과 같다.

$$
\kappa_k = \lVert\eta_k\rVert_2
$$

$$
\mu_k = \frac{\eta_k}{\lVert\eta_k\rVert_2}
$$

따라서 $\eta_k$는 평균방향과 집중도를 하나의 벡터에 함께 포함한다.

---

## 3. 공통 E-step

현재 모수를 $\Theta = (\alpha,\mu,\kappa)$라고 하자. 관측치 $i$가 군집 $k$에 속할 사후확률은 다음과 같다.

$$
\tau_{ik} = \frac{ \alpha_k C_d(\kappa_k)\exp(\eta_k^\top x_i) }{ \sum_{\ell = 1}^K \alpha_\ell C_d(\kappa_\ell)\exp(\eta_\ell^\top x_i) }
$$

E-step 후 다음 충분통계량을 계산한다.

$$
N_k = \sum_{i = 1}^n\tau_{ik}
$$

$$
r_k = \sum_{i = 1}^n\tau_{ik}x_i
$$

$N_k$는 군집 $k$의 유효 표본 수이고, $r_k$는 군집 $k$ 방향의 가중합이다.

혼합비율은 다음과 같이 갱신한다.

$$
\alpha_k^{\mathrm{new}} = \frac{N_k}{n}
$$

E-CGL과 M-CGL은 이 E-step을 공통으로 사용한다.

---

## 4. 두 방법의 centered contrast

$M$과 $E$를 각각 행이 $\mu_k^\top$, $\eta_k^\top$인 $K\times d$ 행렬로 정의한다.

중심화 행렬은 다음과 같다.

$$
H_K = I_K-\frac{1}{K}\mathbf1_K\mathbf1_K^\top
$$

### 4.1 M-CGL의 centered-$\mu$

좌표 $j$의 평균방향 평균과 contrast는 다음과 같다.

$$
\bar\mu_j = \frac{1}{K}\sum_{k = 1}^K\mu_{kj}
$$

$$
c_{kj}^{(\mu)} = \mu_{kj}-\bar\mu_j
$$

행렬식으로는 다음과 같다.

$$
c_{\cdot j}^{(\mu)} = H_KM_{\cdot j}
$$

M-CGL의 방향 이질성 support는 다음과 같다.

$$
S_\mu = \{ j: \lVert c_{\cdot j}^{(\mu)}\rVert_2>0 \}
$$

좌표 $j\notin S_\mu$이면 다음 관계가 성립한다.

$$
\mu_{1j} = \cdots = \mu_{Kj}
$$

즉 해당 좌표에서 평균방향의 군집 간 차이가 없다.

### 4.2 E-CGL의 centered-$\eta$

좌표 $j$의 자연모수 평균과 contrast는 다음과 같다.

$$
\bar\eta_j = \frac{1}{K}\sum_{k = 1}^K\eta_{kj}
$$

$$
c_{kj}^{(\eta)} = \eta_{kj}-\bar\eta_j
$$

행렬식으로는 다음과 같다.

$$
c_{\cdot j}^{(\eta)} = H_KE_{\cdot j}
$$

E-CGL의 posterior-score 이질성 support는 다음과 같다.

$$
S_\eta = \{ j: \lVert c_{\cdot j}^{(\eta)}\rVert_2>0 \}
$$

좌표 $j\notin S_\eta$이면 다음 관계가 성립한다.

$$
\eta_{1j} = \cdots = \eta_{Kj}
$$

두 군집 $k,\ell$의 posterior log-odds는 다음과 같다.

$$
\log \frac{P(Z = k\mid x)}{P(Z = \ell\mid x)} = \log \frac{\alpha_kC_d(\kappa_k)} {\alpha_\ell C_d(\kappa_\ell)} + (\eta_k-\eta_\ell)^\top x
$$

$\eta_{1j} = \cdots = \eta_{Kj}$이면 $x_j$의 계수 차이가 0이므로 $x_j$가 모든 pairwise posterior 선형 점수에서 소거된다.

---

## 5. group $L_2$ 페널티

E-CGL의 페널티는 다음과 같다.

$$
P_E(E) = \lambda_\eta \sum_{j = 1}^d w_j \lVert c_{\cdot j}^{(\eta)}\rVert_2
$$

M-CGL의 페널티는 다음과 같다.

$$
P_M(M) = \lambda_\mu \sum_{j = 1}^d w_j \lVert c_{\cdot j}^{(\mu)}\rVert_2
$$

하나의 group은 좌표 $j$에 해당하는 $K$개 군집 contrast이다.

$$
c_{\cdot j} = (c_{1j},\ldots,c_{Kj})^\top
$$

Group $L_2$ thresholding은 $c_{\cdot j}$ 전체를 함께 0으로 만든다.

$$
c_{\cdot j} = 0
$$

이는 좌표 $j$의 군집 간 차이가 모두 제거되었음을 의미한다.

중심화 정의에 의해 각 좌표의 contrast는 다음 zero-sum 조건을 만족한다.

$$
\sum_{k = 1}^Kc_{kj} = 0
$$

Zero-sum 조건 때문에 group lasso를 사용하는 것은 아니다. Zero-sum은 centering에서 나온다. Group thresholding은 centered vector 전체에 하나의 스칼라를 곱하므로 zero-sum 구조를 자연스럽게 보존한다.

---

## 6. CGL과 ACGL

기본 CGL에서는 모든 좌표에 같은 가중치를 사용한다.

$$
w_j = 1
$$

Adaptive CGL에서는 dense 초기 추정치의 centered norm으로 가중치를 만든다.

$$
w_j = ( \lVert c_{\cdot j}^{\mathrm{init}}\rVert_2+\epsilon )^{-\gamma}
$$

현재 설정은 다음과 같다.

$$
\gamma = 1
$$

$$
\epsilon = 10^{-6}
$$

가중치는 중앙값으로 정규화한다.

$$
w_j \gets \frac{w_j} {\mathrm{median}(w_1,\ldots,w_d)}
$$

초기 contrast가 큰 좌표는 작은 가중치를 받아 덜 축소되고, 초기 contrast가 작은 좌표는 큰 가중치를 받아 더 강하게 축소된다.

| 방법 | contrast | 가중치 |
|---|---|---|
| E-CGL | centered-$\eta$ | $w_j = 1$ |
| E-ACGL | centered-$\eta$ | adaptive $w_j$ |
| M-CGL | centered-$\mu$ | $w_j = 1$ |
| M-ACGL | centered-$\mu$ | adaptive $w_j$ |

---

## 7. Banerjee 집중도 근사

vMF 집중도 M-step은 다음 방정식을 만족한다.

$$
A_d(\kappa) = \rho
$$

$$
A_d(\kappa) = \frac{I_{d/2}(\kappa)} {I_{d/2-1}(\kappa)}
$$

$I_\nu$는 modified Bessel function이다. 이 Bessel 비율의 역함수는 닫힌 형태가 없으며, 고차원에서는 직접 계산하기 어렵다.

Banerjee et al. (2005)의 근사는 다음과 같다.

$$
\widehat\kappa_B(\rho) = \frac{d\rho-\rho^3}{1-\rho^2}
$$

### 7.1 Dense vMF 초기화

비페널티 dense M-step에서는 다음 값을 계산한다.

$$
\mu_k = \frac{r_k}{\lVert r_k\rVert_2}
$$

$$
\rho_k = \frac{\lVert r_k\rVert_2}{N_k}
$$

$$
\kappa_k \approx \widehat\kappa_B(\rho_k)
$$

현재 구현에서 E 계열과 M 계열은 모두 dense 초기 적합에 Banerjee 근사를 사용한다.

---

## 8. E-CGL/E-ACGL 알고리즘

### 8.1 초기화

1. 각 $x_i$를 단위노름으로 정규화한다.
2. `nstart`개의 초기 군집 분할을 생성한다.
3. 각 초기값에서 dense vMF EM을 실행한다.
4. Dense EM의 $\kappa_k$는 Banerjee 근사로 갱신한다.
5. 관측 로그우도가 가장 큰 dense fit을 선택한다.
6. E-ACGL이면 dense centered-$\eta$에서 adaptive weights를 계산한다.
7. Adaptive weights는 전체 penalty path에서 고정한다.

### 8.2 $\lambda_\eta$ path

Penalty path는 다음 순서를 갖는다.

$$
0 = \lambda_{\eta,1} < \lambda_{\eta,2} < \cdots < \lambda_{\eta,L}
$$

확정 `true_pg` 엔진은 모든 centered contrast가 0인 collapsed solution에서 projected gradient를 계산하여 $\lambda_{\max}$를 정한다.

Collapsed endpoint의 공통 $\kappa$를 구성할 때 Banerjee 근사가 사용된다. 이후 geometric grid를 만들고 이전 $\lambda$의 적합값을 다음 $\lambda$의 초기값으로 사용하는 warm start를 적용한다.

### 8.3 고정 responsibility에서의 목적함수

고정된 $\tau_{ik}$에서 $\eta$에 대한 최소화 문제는 다음과 같다.

$$
\min_E [ -\sum_{k = 1}^K r_k^\top\eta_k -\sum_{k = 1}^K N_k\log C_d(\lVert\eta_k\rVert_2) + \lambda_\eta \sum_{j = 1}^d w_j \lVert H_KE_{\cdot j}\rVert_2 ]
$$

첫 두 항은 smooth part이고 마지막 항은 nonsmooth group penalty이다.

고정된 responsibilities에서는 이 문제는 $\eta$에 대해 convex이다. 그러나 혼합모형 전체는 비볼록이므로 다중 초기값이 필요하다.

### 8.4 Smooth gradient

Smooth part를 다음과 같이 둔다.

$$
g(E) = -\sum_k r_k^\top\eta_k -\sum_kN_k\log C_d(\lVert\eta_k\rVert_2)
$$

$\kappa_k = \lVert\eta_k\rVert_2$일 때 gradient는 다음과 같다.

$$
\nabla_{\eta_k}g = -r_k + N_kA_d(\kappa_k) \frac{\eta_k}{\kappa_k}
$$

이 단계에서는 $A_d(\kappa)$를 평가하기만 한다. $A_d^{-1}$을 계산하지 않으므로 Banerjee 역근사가 필요하지 않다.

### 8.5 Proximal-gradient proposal

현재값 $E^{(t)}$에서 gradient step을 계산한다.

$$
V = E^{(t)} -s\nabla g(E^{(t)})
$$

초기 step size는 다음 값으로 설정한다.

$$
s = \frac{1}{\max_kN_k}
$$

각 좌표에서 공통 baseline과 centered contrast를 분리한다.

$$
\bar v_j = \frac{1}{K}\sum_kv_{kj}
$$

$$
u_{\cdot j} = v_{\cdot j}-\bar v_j\mathbf1_K
$$

Centered contrast에 group soft-thresholding을 적용한다.

$$
u_{\cdot j}^{+} = ( 1-\frac{s\lambda_\eta w_j} {\lVert u_{\cdot j}\rVert_2} )_+ u_{\cdot j}
$$

공통 baseline을 다시 더한다.

$$
v_{\cdot j}^{+} = \bar v_j\mathbf1_K+u_{\cdot j}^{+}
$$

Group이 제거되더라도 공통 baseline은 남을 수 있다.

$$
u_{\cdot j}^{+} = 0
$$

$$
v_{\cdot j}^{+} = \bar v_j\mathbf1_K
$$

따라서 E-CGL은 좌표 전체를 0으로 만드는 방법이 아니라 군집 간 자연모수 차이를 제거하는 방법이다.

### 8.6 Backtracking

Proposal이 smooth objective의 quadratic upper bound를 만족하는지 확인한다.

$$
g(E^+) \le g(E) + \langle\nabla g(E),E^+-E\rangle + \frac{\lVert E^+-E\rVert_F^2}{2s}
$$

조건을 만족하지 않으면 step size를 줄인다.

$$
s\gets s/2
$$

조건을 만족할 때까지 proximal update를 다시 계산한다.

### 8.7 GEM 외부 반복

Proximal M-step 후 $\eta_k$를 $(\mu_k,\kappa_k)$로 변환한다.

$$
\kappa_k^{+} = \lVert\eta_k^{+}\rVert_2
$$

$$
\mu_k^{+} = \frac{\eta_k^{+}} {\lVert\eta_k^{+}\rVert_2}
$$

새 observed penalized objective를 계산한다.

$$
\ell(\Theta^+) -\lambda_\eta P_E(E^+)
$$

고정-responsibility $Q$와 observed penalized objective가 허용 오차보다 감소하면 해당 업데이트를 실패 처리한다.

목적함수의 상대 변화가 tolerance 이하가 될 때까지 E-step과 proximal M-step을 반복한다.

### 8.8 Support 결정

각 $\lambda$에서 다음 support를 저장한다.

$$
\widehat S_\eta(\lambda) = \{ j: \lVert H_K\widehat E_{\cdot j}\rVert_2 > \mathrm{tol} \}
$$

같은 support를 만든 여러 $\lambda$는 하나의 후보로 합친다.

### 8.9 Exact centered-support refit

선택된 support $S$를 고정하고 group penalty를 제거한다.

$$
\eta_{kj} = b_j+c_{kj}
$$

$$
\sum_kc_{kj} = 0
$$

Active 좌표에서는 component contrast를 추정한다.

$$
j\in S
$$

Inactive 좌표에서는 contrast를 0으로 고정한다.

$$
j\notin S \quad\Longrightarrow\quad c_{kj} = 0
$$

따라서 inactive 좌표의 공통 baseline은 유지된다.

$$
\eta_{1j} = \cdots = \eta_{Kj} = b_j
$$

자유 parameter는 $d$개의 common baseline과 active 좌표마다 $K-1$개의 contrast이다. 이 parameterization에서 L-BFGS-B로 likelihood M-step을 풀고 line search로 observed likelihood 비감소를 확인한다.

### 8.10 BIC/EBIC 선택

각 고유 support를 exact refit한 후 정보기준을 계산한다.

$$
\mathrm{df}_E = d+(K-1)m+(K-1)\mathbf1(m>0)
$$

$$
\mathrm{BIC} = -2\ell_{\mathrm{refit}} + \log(n)\mathrm{df}_E
$$

$m = |\widehat S_\eta|$이다. 이 값은 exact effective degrees of freedom가 아니라 practical nominal model dimension이다.

현재 확정 알고리즘은 refit 후 likelihood로 BIC 또는 EBIC를 계산한다.

---

## 9. E 계열에서 Banerjee 근사의 역할

| 단계 | E-CGL `true_pg` |
|---|---|
| Dense 다중 초기화 | 사용 |
| KKT path endpoint의 공통 $\kappa$ | 사용 |
| Penalized proximal-gradient M-step | 사용하지 않음 |
| $\eta$에서 $(\mu,\kappa)$로 변환 | $\kappa = \lVert\eta\rVert_2$ |
| Exact centered-support refit | 사용하지 않음 |
| 최종 $\widehat\kappa$ | $\lVert\widehat\eta\rVert_2$ |

최종 E-CGL `true_pg` 추정량의 $\widehat\kappa$는 Banerjee 근사값이 아니다.

구형 `eta_engine="current"`는 반복마다 Banerjee 근사를 사용하는 dense M-step을 계산한 뒤 eta prox를 적용한다. 논문용 확정 결과는 `eta_engine="true_pg"`를 명시적으로 사용한다. 현재 패키지 기본값은 아직 `"current"`이므로 공개 전 기본 엔진 정책을 확정해야 한다.

---

## 10. M-CGL/M-ACGL 알고리즘

### 10.1 목적함수

M-CGL은 다음 목적함수를 최대화한다.

$$
\max_{\alpha,\mu,\kappa} [ \ell(\alpha,\mu,\kappa) - \lambda_\mu \sum_{j = 1}^d w_j \lVert H_KM_{\cdot j}\rVert_2 ]
$$

각 평균방향은 단위구면 제약을 만족해야 한다.

$$
\lVert\mu_k\rVert_2 = 1
$$

### 10.2 초기화와 path

1. Dense vMF 다중 초기값을 적합한다.
2. 최대 로그우도 dense fit을 선택한다.
3. M-ACGL이면 dense centered-$\mu$에서 adaptive weights를 계산한다.
4. Centered score로 $\lambda_{\max}$ proxy를 계산한다.
5. Geometric $\lambda_\mu$ path를 생성한다.
6. 이전 결과를 다음 $\lambda$의 warm start로 사용한다.

Dense 초기화에서는 Banerjee 근사가 사용된다.

### 10.3 고정 $\kappa$에서의 $\mu$ 문제

고정된 $\kappa$에서 다음 문제를 푼다.

$$
\max_{\lVert\mu_k\rVert_2 = 1} [ \sum_k\kappa_kr_k^\top\mu_k - \lambda_\mu \sum_jw_j \lVert H_KM_{\cdot j}\rVert_2 ]
$$

Group penalty는 component들을 결합하지만 각 행은 동시에 단위구면 위에 있어야 한다. 따라서 E-CGL처럼 Euclidean prox 한 번으로 풀 수 없다.

### 10.4 ADMM variable splitting

분할변수를 도입한다.

$$
Z = H_KM
$$

분할 문제는 다음과 같다.

$$
\max_{M,Z} \sum_k\kappa_kr_k^\top\mu_k - \lambda_\mu \sum_jw_j \lVert Z_{\cdot j}\rVert_2
$$

제약은 다음과 같다.

$$
Z = H_KM
$$

$$
\lVert\mu_k\rVert_2 = 1
$$

ADMM은 $\mu$-update, $Z$-update, dual update를 반복한다.

### 10.5 $\mu$-update

현재 $Z$와 scaled dual variable $U$를 고정하고 augmented objective를 구면 위에서 최대화한다.

먼저 Euclidean gradient $g_k$를 구한 뒤 $\mu_k$의 접공간으로 투영한다.

$$
g_k^{\mathrm{tan}} = g_k-(\mu_k^\top g_k)\mu_k
$$

접선 방향으로 이동한다.

$$
\widetilde\mu_k = \mu_k+s g_k^{\mathrm{tan}}
$$

단위구면으로 되돌리는 retraction을 적용한다.

$$
\mu_k^+ = \frac{\widetilde\mu_k} {\lVert\widetilde\mu_k\rVert_2}
$$

Augmented objective가 충분히 증가하지 않으면 step size를 절반으로 줄인다. 현재 구현은 이 sphere update를 Rcpp로 가속할 수 있다.

### 10.6 $Z$-update

다음 값을 계산한다.

$$
V = H_KM^++U
$$

각 좌표에 group soft-thresholding을 적용한다.

$$
Z_{\cdot j}^+ = ( 1- \frac{\lambda_\mu w_j} {\rho_{\mathrm{ADMM}} \lVert V_{\cdot j}\rVert_2} )_+ V_{\cdot j}
$$

### 10.7 Dual update와 ADMM 수렴

Scaled dual variable을 갱신한다.

$$
U^+ = U+H_KM^+-Z^+
$$

Primal residual은 다음과 같다.

$$
r_{\mathrm{pri}} = \lVert H_KM-Z\rVert_F
$$

Dual residual은 다음과 같다.

$$
r_{\mathrm{dual}} = \rho_{\mathrm{ADMM}} \lVert Z-Z_{\mathrm{old}}\rVert_F
$$

두 residual이 tolerance 이하이면 ADMM을 종료한다.

### 10.8 $\kappa$-update

새 $\mu_k$가 정해지면 $\kappa_k$에 관한 조건부 목적함수는 다음과 같다.

$$
Q_k(\kappa_k) = N_k\log C_d(\kappa_k) + \kappa_kr_k^\top\mu_k
$$

일차조건은 다음과 같다.

$$
A_d(\kappa_k) = \frac{r_k^\top\mu_k}{N_k}
$$

다음 값을 정의한다.

$$
\rho_k = \frac{r_k^\top\mu_k}{N_k}
$$

현재 구현은 다음 순서로 $\kappa_k$를 계산한다.

1. Banerjee 근사 $\kappa_{B,k}$를 계산한다.
2. 초기 상한을 $\max(1,2\kappa_{B,k})$로 설정한다.
3. $A_d(u_k)\ge\rho_k$가 될 때까지 상한을 두 배로 확장한다.
4. `uniroot()`로 $A_d(\kappa_k)-\rho_k = 0$을 푼다.

따라서 Banerjee 값은 최종 $\kappa_k$가 아니라 수치적 근의 bracket을 빠르게 찾는 안내값이다.

### 10.9 $\mu$와 $\kappa$의 내부 교대

M-step 내부에서 다음 과정을 반복한다.

1. 현재 $\kappa$로 ADMM $\mu$-update
2. 새 $\mu$로 numerical-root $\kappa$-update
3. Penalized $Q$ 계산
4. $Q$가 감소하면 $(\mu,\kappa)$ update에 step-halving 적용
5. 상대 변화가 tolerance 이하이면 M-step 종료

### 10.10 외부 EM guard

새 $(\alpha,\mu,\kappa)$의 observed penalized likelihood를 계산한다.

감소하면 현재값과 target 사이의 step size를 줄인다. $\mu$는 보간 후 반드시 다시 행별 정규화한다.

### 10.11 Support 결정

Path support는 ADMM 분할변수 $Z$에서 계산한다.

$$
\widehat S_\mu = \{ j: \lVert Z_{\cdot j}\rVert_2 > \mathrm{tol} \}
$$

ADMM이 수렴하면 다음 관계가 성립한다.

$$
Z\approx H_KM
$$

### 10.12 Exact centered-$\mu$ support refit

고정 support refit에서는 inactive 좌표에 다음 equality를 부여한다.

$$
j\notin S_\mu \quad\Longrightarrow\quad \mu_{1j} = \cdots = \mu_{Kj}
$$

모든 행의 단위구면 제약도 유지한다.

$$
\lVert\mu_k\rVert_2 = 1
$$

고정 $\kappa$에서 inactive 공통 부분과 active component별 방향의 조건부 최대값을 계산한다. 이어서 $A_d(\kappa_k) = \rho_k$의 수치적 근으로 $\kappa_k$를 갱신한다. 두 갱신을 수렴할 때까지 교대한다.

### 10.13 M 계열 BIC/EBIC

M 계열의 practical nominal dimension은 다음과 같이 계산한다.

$$
\mathrm{df}_M = d_\mu(S)+(K-1)+K
$$

$$
d_\mu(S) = \begin{cases} d-1, & m = 0,\\ d+(K-1)m-K, & m>0. \end{cases}
$$

$(K-1)$은 mixing proportions이고, $K$는 component별 $\kappa_k$이다.

M 계열도 각 고유 support를 refit한 후 BIC 또는 EBIC를 계산한다.

---

## 11. M 계열에서 Banerjee 근사의 역할

| 단계 | M-CGL/M-ACGL |
|---|---|
| Dense 초기화 | 근사값을 직접 사용 |
| Penalized $\mu$-update | 사용하지 않음 |
| Penalized $\kappa$-update | 수치근의 초기 규모와 bracket에 사용 |
| Fixed-support refit의 $\kappa$-update | 수치근의 bracket에 사용 |
| 최종 $\widehat\kappa$ | `uniroot()` 해 |
| 근 bracket을 찾지 못한 경우 | `kappa_cap` 반환 |

최종 M-CGL의 $\widehat\kappa$도 Banerjee 근사값 자체가 아니다.

---

## 12. E 계열에서 Banerjee 역근사가 필요하지 않은 이유

M-CGL은 $\mu$를 고정한 뒤 다음 방정식을 풀어야 한다.

$$
A_d(\kappa) = \rho
$$

따라서 $A_d^{-1}$ 계산이 필요하다.

E-CGL은 $\eta$를 직접 최적화하며 gradient에서 다음 값만 평가한다.

$$
A_d(\lVert\eta\rVert_2)
$$

최종 집중도는 다음과 같이 얻는다.

$$
\widehat\kappa_k = \lVert\widehat\eta_k\rVert_2
$$

따라서 Banerjee 역근사를 사용하지 않는 것이 E-CGL parameterization과 일치한다.

현재 코드는 scaled Bessel function을 사용하고, 고차원에서 불안정한 경우 $\log C_d(\kappa)$의 수치미분을 fallback으로 사용한다.

다만 free-$\kappa$ finite mixture likelihood의 component-collapse 문제는 Banerjee 근사 사용 여부와 별개의 문제다. M 계열은 numerical root에 `kappa_cap`을 둔다. E `true_pg`와 exact refit은 현재 $\eta$ norm을 명시적으로 cap하지 않으므로, 최종 논문과 패키지에서는 concentration cap 또는 작은 smooth stabilization의 채택 여부를 확정할 필요가 있다.

---

## 13. 두 알고리즘 비교

| 항목 | E-CGL/E-ACGL | M-CGL/M-ACGL |
|---|---|---|
| 직접 모수 | $\eta_k$ | $\mu_k,\kappa_k$ |
| 선택 대상 | Posterior-score 이질성 | 방향 이질성 |
| 비선택 의미 | $\eta_{1j} = \cdots = \eta_{Kj}$ | $\mu_{1j} = \cdots = \mu_{Kj}$ |
| 모수 공간 | $\eta_k\in\mathbb R^d$ | $\mu_k\in\mathbb S^{d-1}$ |
| 고정 responsibility 문제 | Convex | 구면 제약으로 nonconvex |
| 핵심 최적화 | Proximal gradient | ADMM과 구면 tangent update |
| Group threshold 대상 | $H_KE$ | $Z\approx H_KM$ |
| $\kappa$ 최종 계산 | $\lVert\eta_k\rVert_2$ | $A_d(\kappa_k) = \rho_k$ 수치근 |
| Banerjee 역할 | Dense 초기화와 path endpoint | Dense 초기화와 root bracket |
| Fixed-support refit | 공통 $\eta$ baseline 유지 | 공통 $\mu$와 구면 제약 유지 |
| 계산 복잡도 | 상대적으로 낮음 | ADMM과 반복 수치근으로 높음 |
| 현재 구현 지위 | 논문용 `true_pg` 확정 엔진 | Directional companion 진단 엔진 |

공통 concentration이면 두 support는 일치한다.

$$
\kappa_1 = \cdots = \kappa_K = \kappa
$$

$$
H_KE = \kappa H_KM
$$

$$
S_\eta = S_\mu
$$

집중도가 군집별로 다르면 다음 관계 때문에 두 support가 달라질 수 있다.

$$
\eta_k = \kappa_k\mu_k
$$

따라서 두 방법은 다음 질문에 각각 대응한다.

- E-CGL: 어떤 좌표가 posterior 분류 점수의 군집 간 차이를 만드는가?
- M-CGL: 어떤 좌표에서 군집 평균방향이 달라지는가?
- E-ACGL/M-ACGL: 각 target을 유지하면서 초기 contrast에 따라 좌표별 축소 강도를 조절하는 adaptive 확장

---

## 14. 해석 시 주의점

1. E-CGL과 M-CGL은 서로 다른 estimand를 가지므로 한 종류의 $F_1$만으로 우열을 결정하지 않는다.
2. E-CGL은 $F_{1,\eta}$와 $\mathrm{MSE}_\eta$, M-CGL은 $F_{1,\mu}$와 $\mathrm{MSE}_\mu$를 주 지표로 평가한다.
3. ARI, NMI, test NLL, runtime, failure rate는 두 방법에 공통으로 보고한다.
4. 고정 responsibility에서 E-CGL subproblem은 convex이지만 혼합모형 전체의 전역 최적해를 보장하지는 않는다.
5. M-CGL은 구면 제약으로 고정 responsibility 단계도 nonconvex이며 ADMM은 stationary solution을 찾는 수치 알고리즘이다.
6. Rcpp는 E-step, group proximal map, 행 정규화, M-CGL sphere update 등 low-level 계산을 가속한다. 통계모형과 목적함수는 R-only 구현과 동일하다.

---

## 15. 구현 감사 결과

Banerjee 근사는 M-CGL에만 사용되는 것이 아니다.

1. E 계열과 M 계열의 dense 초기화에 공통으로 사용된다.
2. E-CGL `true_pg`에서는 KKT path endpoint 구성에 추가로 사용된다.
3. E-CGL `true_pg`의 penalized update와 exact refit에서는 사용되지 않는다.
4. M-CGL의 penalized update와 fixed-support refit에서는 수치적 $\kappa$ 근의 bracket 설정에 사용된다.
5. 최종 E-CGL의 $\widehat\kappa_k$는 $\lVert\widehat\eta_k\rVert_2$이다.
6. 최종 M-CGL의 $\widehat\kappa_k$는 `uniroot()`로 구한 $A_d(\kappa_k) = \rho_k$의 해이다.

구현 확인 파일:

- `local_package/etaVmf/R/vmf-core.R`
- `local_package/etaVmf/R/eta-penalty.R`
- `local_package/etaVmf/R/exact-refit.R`
- `local_package/etaVmf/R/fit-eta-vmf.R`
- `local_package/etaVmf/R/control.R`
- `r/simulation/m_cgl_diagnostic_helpers_260722.r`
- `r/simulation/studyb_mcgl_cell_260726.r`

참고문헌:

- Banerjee, A., Dhillon, I. S., Ghosh, J., and Sra, S. (2005). *Clustering on the Unit Hypersphere using von Mises-Fisher Distributions*. Journal of Machine Learning Research, 6, 1345-1382.
