# 연구미팅: M-CGL 도입 검토와 E-CGL의 관계 (2026-07-29)

## 1. 검토 배경

지난 연구미팅에서는 E-CGL과 Rossi 계열 M-L의 패널티 대상과 구조가 달라
직접 비교의 정합성이 낮다는 의견이 있었다.

> 논문의 중심은 E-CGL로 하고, M-CGL은 방향 이질성을 추정하는 이론 및 실증적 companion으로 유지하는 게 어떤지 확인.

| 모형 | 패널티 대상 | 구조 | 추정 대상 |
|---|---|---|---|
| M-L | $\mu_{kj}$ | entry-wise $L_1$, uncentered | prototype support |
| M-CGL | $c_{kj}^{(\mu)}=\mu_{kj}-\bar\mu_j$ | coordinate-wise group $L_2$, centered | directional heterogeneity |
| E-CGL | $c_{kj}^{(\eta)}=\eta_{kj}-\bar\eta_j$ | coordinate-wise group $L_2$, centered | posterior-score heterogeneity |

M-CGL은 E-CGL과 동일한 centered group 구조를 $\mu$ 공간에 적용한 matched
companion이다.

## 2. 모형과 추정 대상

$$
x_i\mid z_i=k\sim\mathrm{vMF}(\mu_k,\kappa_k),
\qquad
\eta_k=\kappa_k\mu_k
$$

$$
\log\frac{\Pr(Z=k\mid x)}{\Pr(Z=\ell\mid x)}=
a_{k\ell}+(\eta_k-\eta_\ell)^{\mathsf T}x
$$

$$
c_{kj}^{(\mu)}=\mu_{kj}-\bar\mu_j,
\qquad
c_{kj}^{(\eta)}=\eta_{kj}-\bar\eta_j
$$

$$
S_\mu=\{j:\lVert\boldsymbol c_j^{(\mu)}\rVert_2>0\},
\qquad
S_\eta=\{j:\lVert\boldsymbol c_j^{(\eta)}\rVert_2>0\}
$$

| Support | 의미 | 모형 |
|---|---|---|
| $S_P=\{j:\exists k,\ \mu_{kj}\neq0\}$ | component prototype에 존재하는 좌표 | M-L |
| $S_\mu$ | 군집 간 평균 방향이 다른 좌표 | M-CGL |
| $S_\eta$ | pairwise posterior-score가 다른 좌표 | E-CGL |

M-L의 직접 선택 단위는 $(k,j)$이며, $S_P$는 선택된 component-coordinate의
좌표별 합집합이다.

## 3. Centered group penalty

E-CGL:

$$
\widehat\Theta_{\lambda_\eta}^{\mathrm{E-CGL}}=
\underset{\pi,\eta}{\arg\max}
\Big[
\ell(\pi,\eta)
-\lambda_\eta\sum_{j=1}^{d}
\lVert\boldsymbol c_j^{(\eta)}\rVert_2
\Big]
$$

M-CGL:

$$
\widehat\Theta_{\lambda_\mu}^{\mathrm{M-CGL}}=
\underset{\pi,\kappa,\mu}{\arg\max}
\Big[
\ell(\pi,\kappa,\mu)
-\lambda_\mu\sum_{j=1}^{d}
\lVert\boldsymbol c_j^{(\mu)}\rVert_2
\Big],
\qquad
\lVert\mu_k\rVert_2=1
$$

경로에서 얻은 각 support에 B 방식의 support-constrained refit을 적용한
후 current nominal-df BIC로 선택한다. 비선택 좌표는 contrast만 0으로
고정하고 공통 baseline은 유지한다.

$$
j\notin S_\eta
\quad\Longrightarrow\quad
\eta_{1j}=\cdots=\eta_{Kj}=\bar\eta_j
$$

## 4. 이론적 관계

공통 집중도에서는 두 support가 일치한다.

$$
\kappa_1=\cdots=\kappa_K=\kappa
\quad\Longrightarrow\quad
c_{kj}^{(\eta)}=\kappa c_{kj}^{(\mu)}
\quad\Longrightarrow\quad
S_\eta=S_\mu
$$

이질적 집중도에서는 $\mu_{kj}=\eta_{kj}/\kappa_k$이므로 두 support가
달라질 수 있다.

$$
\eta_{kj}=\kappa_k\mu_{kj},
\qquad
\mu_{kj}=\frac{\eta_{kj}}{\kappa_k}
$$

$$
\eta_{1j}=\cdots=\eta_{Kj}=b_j\neq0
\quad\Longrightarrow\quad
\boldsymbol c_j^{(\eta)}=0,
\qquad
\mu_{kj}=\frac{b_j}{\kappa_k}
$$

$$
\kappa_k\ \mathrm{heterogeneous}
\quad\Longrightarrow\quad
\boldsymbol c_j^{(\mu)}\neq0,
\qquad
S_\eta\neq S_\mu
$$

E-CGL은 posterior-score contrast를 선택하고, M-CGL은 directional
contrast를 선택한다. E-CGL에서 제외된 좌표의 공통 baseline
$\bar\eta_j=b_j$는 유지된다.

## 5. 구조적 진단

목적은 이론적으로 구분한 $S_\mu$와 $S_\eta$가 구현에서 재현되는지
확인하는 것이다.

$$
K=4,\quad n=400,\quad d=24,\quad
\mathrm{nstart}=5,\quad \mathrm{path}=21
$$

Support selection은 BIC-after-refit을 사용하였다.

| 구조 | 반복 | 참 $(q_\mu,q_\eta)$ | M-CGL | E-CGL |
|---|---:|---:|---:|---:|
| 공통 $\kappa$ | 10 | $(8,8)$ | $q=8$, $F_{1,\mu}=1.000$ | $q=8$, $F_{1,\eta}=1.000$ |
| 이질적 $\kappa$ | 10 | $(8,12)$ | $q=8$, $F_{1,\mu}=1.000$ | $q=12.1$, $F_{1,\eta}=0.996$ |
| 공통 $\eta$ | 5 | $(12,8)$ | $q=12$, $F_{1,\mu}=1.000$ | $q=8$, $F_{1,\eta}=1.000$ |
| 공통 $\mu$ | 3 | $(0,10)$ | $q=3.0$, $\Pr(\widehat q=0)=33.3\%$ | $q=12.67$, $F_{1,\eta}=0.885$ |

공통 $\kappa$, 이질적 $\kappa$, 공통 $\eta$ 구조에서는 두 방법이 각자의
참 support를 회복하였다. 공통 $\mu$ 구조에서는 M-CGL의 참 support가
공집합이지만 평균 3.0개가 선택되어 경계 selector의 불안정성이 확인되었다.

## 6. Study B matched pilot

목적은 기존 Study B 설계에서 M/E 계열의 유한표본 성능을 같은 자료와
초기값으로 비교하는 것이다.

$$
K=4,\quad n=300,\quad d=200,\quad e_B=0.05,
\qquad
(q_C,q_D,q_N)=(4,16,180),
\qquad
\mathrm{rep}=3
$$

Path 120과 BIC-after-refit을 사용하였다. 괄호는
$(\widehat q_C,\widehat q_D,\widehat q_N)$이다.

| $\kappa$ | 모형 | 고유 target | selected $q$ | own-target $F_1$ | $F_{1,\eta}$ | ARI | $\mathrm{MSE}_\mu$ | $\mathrm{MSE}_\kappa$ | $\mathrm{MSE}_\eta$ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 공통 | M-CGL | $S_\mu$ | 16.7 (0.3, 16.0, 0.3) | 0.980 | 0.980 | 0.841 | $4.63\times10^{-4}$ | 8.995 | 0.276 |
| 공통 | M-ACGL | $S_\mu$ | 16.3 (0.0, 16.0, 0.3) | 0.990 | 0.990 | 0.841 | $4.55\times10^{-4}$ | 8.309 | 0.261 |
| 공통 | E-CGL | $S_\eta$ | 16.3 (0.0, 16.0, 0.3) | 0.990 | 0.990 | 0.858 | $4.57\times10^{-4}$ | 6.127 | 0.253 |
| 공통 | E-ACGL | $S_\eta$ | 16.3 (0.0, 16.0, 0.3) | 0.990 | 0.990 | 0.858 | $4.57\times10^{-4}$ | 6.134 | 0.253 |
| 이질적 | M-CGL | $S_\mu$ | 18.3 (3.3, 14.7, 0.3) | 0.933 | 0.851 | 0.822 | $8.24\times10^{-4}$ | 14.418 | 0.545 |
| 이질적 | M-ACGL | $S_\mu$ | 17.3 (1.3, 16.0, 0.0) | 0.928 | 0.960 | 0.852 | $5.63\times10^{-4}$ | 9.174 | 0.383 |
| 이질적 | E-CGL | $S_\eta$ | 16.0 (0.0, 16.0, 0.0) | 1.000 | 1.000 | 0.855 | $5.09\times10^{-4}$ | 10.359 | 0.229 |
| 이질적 | E-ACGL | $S_\eta$ | 16.0 (0.0, 16.0, 0.0) | 1.000 | 1.000 | 0.855 | $5.09\times10^{-4}$ | 10.332 | 0.228 |

Own-target $F_1$은 M 계열에서 $S_\mu$, E 계열에서 $S_\eta$를 기준으로
계산하였다. $F_{1,\eta}$는 모든 모형을 posterior-score target으로 비교한
교차 지표이다. MSE는 centered - $\eta$ 손실을 최소화하는 component
permutation으로 label을 정렬한 후 계산했으며,
$\mathrm{MSE}_\eta$는 centered - $\eta$ contrast의 MSE이다.

- 공통 $\kappa$: $S_\mu=S_\eta$이며 M/E 계열의 support 회복이 유사하였다.
- 이질적 $\kappa$: 참 support는 $(q_\mu,q_\eta)=(20,16)$이다.
- M-CGL의 common 좌표 선택은 directional target에 포함된다.
- E-CGL은 posterior-score target 16개를 선택하였다.

## 7. 실행시간

구조적 진단 설정 $(K,n,d)=(4,400,24)$, path 21에서 측정한 Rcpp 적용 후
평균 초/rep이다. `sourceCpp` 컴파일, 로드 시간은 제외함.

| 구조 | M-CGL | M-ACGL | E-CGL | E-ACGL |
|---|---:|---:|---:|---:|
| 공통 $\kappa$ | 6.2 | 8.1 | 1.9 | 2.0 |
| 이질적 $\kappa$ | 10.0 | 8.4 | 2.6 | 3.0 |
| 공통 $\eta$ | 4.3 | 6.1 | 3.1 | 3.5 |
| 공통 $\mu$ | 20.4 | 24.4 | 17.3 | 16.8 |

M-CGL은 단위구면 제약과 ADMM 내부 반복을 포함하므로 E-CGL보다 계산량이
크다.

---

## 백업 자료: Guarded path algorithms

### Algorithm 1. Guarded path algorithm for E-CGL and E-ACGL

**Input:** $X$, $K$, path size $L$, method indicator, iteration limits,
$\varepsilon_{\mathrm{conv}}$, $\varepsilon_{\mathrm{acc}}$

**Output:** $(\widehat S_\eta,\widehat\lambda_\eta)$,
$\widehat\Theta_\eta^{\mathrm{refit}}$, numerical diagnostics

| 단계 | 절차 |
|---:|---|
|  | **Stage 1: Dense start and path construction** |
| 1 | 여러 초기값에서 dense vMF를 적합하고 최대 log-likelihood 해를 선택 |
| 2 | E-CGL은 $w_j=1$; E-ACGL은 dense fit에서 $w_j$를 계산한 뒤 path 전체에서 고정 |
| 3 | dense-to-sparse KKT-geometric path $\Lambda_\eta=(0,\lambda_{\eta,1},\ldots,\lambda_{\eta,L-1})$를 구성 |
|  | **Stage 2: Guarded penalized path** |
| 4 | 각 $\lambda_\eta\in\Lambda_\eta$에서 직전 accepted fit을 warm start로 사용 |
| 5 | E-step에서 $\tau_{ik}$, $N_k=\sum_i\tau_{ik}$, $r_k=\sum_i\tau_{ik}x_i$를 계산 |
| 6 | M-step에서 $\pi_k^{+}=N_k/n$와 centered-$\eta$ group proximal update를 계산 |
| 7 | majorization 조건이 성립할 때까지 step size를 절반으로 축소 |
| 8 | 보조함수 또는 penalized observed log-likelihood가 허용범위보다 감소하면 해당 update를 거절하고 이전 accepted estimate를 유지 |
| 9 | 상대 criterion 변화가 $\varepsilon_{\mathrm{conv}}$ 미만이 될 때까지 5--8을 반복 |
| 10 | $S_{\eta,\lambda}$, criterion, iteration 수, line-search 진단을 저장 |
| 11 | path 종료 조건 또는 $L$에 도달할 때까지 4--10을 반복 |
|  | **Stage 3: B-method support-constrained refit and selection** |
| 12 | path에서 중복 support를 제거 |
| 13 | 각 $S$에 대해 $j\notin S\Rightarrow c_{kj}^{(\eta)}=0$을 유지하는 support-constrained refit을 수행 |
| 14 | $\mathrm{BIC}^{\mathrm{refit}}(S)=-2\ell(\widehat\Theta_S^{\mathrm{refit}})+\log(n)\mathrm{df}_{\mathrm{nom}}(S)$를 계산 |
| 15 | $\widehat S_\eta=\underset{S}{\arg\min}\ \mathrm{BIC}^{\mathrm{refit}}(S)$를 선택 |
| 16 | $\widehat S_\eta$, $\widehat\lambda_\eta$, $\widehat\Theta_\eta^{\mathrm{refit}}$와 수치 진단을 반환 |

### Algorithm 2. Guarded path algorithm for M-CGL and M-ACGL

**Input:** $X$, $K$, path size $L$, method indicator, iteration limits,
$\varepsilon_{\mathrm{conv}}$, $\varepsilon_{\mathrm{acc}}$

**Output:** $(\widehat S_\mu,\widehat\lambda_\mu)$,
$\widehat\Theta_\mu^{\mathrm{refit}}$, numerical diagnostics

| 단계 | 절차 |
|---:|---|
|  | **Stage 1: Dense start and path construction** |
| 1 | 여러 초기값에서 dense vMF를 적합하고 최대 log-likelihood 해를 선택 |
| 2 | M-CGL은 $w_j=1$; M-ACGL은 dense fit에서 $w_j$를 계산한 뒤 path 전체에서 고정 |
| 3 | centered-$\mu$ norm의 $\lambda_{\max}$ proxy로 geometric path $\Lambda_\mu=(0,\lambda_{\mu,1},\ldots,\lambda_{\mu,L-1})$를 구성 |
|  | **Stage 2: Guarded penalized path** |
| 4 | 각 $\lambda_\mu\in\Lambda_\mu$에서 직전 accepted fit을 warm start로 사용 |
| 5 | E-step에서 $\tau_{ik}$, $N_k=\sum_i\tau_{ik}$, $r_k=\sum_i\tau_{ik}x_i$를 계산 |
| 6 | M-step에서 $\pi_k^{+}=N_k/n$를 갱신 |
| 7 | $Z=C^{(\mu)}$ 분할변수와 dual variable을 두고 ADMM을 수행 |
| 8 | $\mu$-update에서 product of spheres 위의 Rcpp tangent-gradient와 retraction을 수행 |
| 9 | $Z$-update에서 coordinate-wise group soft-thresholding을 적용한 뒤 dual variable을 갱신 |
| 10 | $A_d(\kappa_k)=r_k^{\mathsf T}\mu_k/N_k$의 수치적 근으로 $\kappa_k$를 갱신 |
| 11 | penalized 보조함수 또는 observed criterion이 감소하면 step halving을 적용하고, 실패하면 dense start에서 재시도 |
| 12 | 상대 criterion 변화가 $\varepsilon_{\mathrm{conv}}$ 미만이 될 때까지 5--11을 반복하고 $S_{\mu,\lambda}$와 ADMM 진단을 저장 |
|  | **Stage 3: B-method support-constrained refit and selection** |
| 13 | 각 distinct $S$에 대해 $j\notin S\Rightarrow c_{kj}^{(\mu)}=0$과 $\lVert\mu_k\rVert_2=1$을 유지하는 refit을 수행 |
| 14 | $\mathrm{BIC}^{\mathrm{refit}}(S)=-2\ell(\widehat\Theta_S^{\mathrm{refit}})+\log(n)\mathrm{df}_{\mathrm{nom}}(S)$를 계산 |
| 15 | $\widehat S_\mu=\underset{S}{\arg\min}\ \mathrm{BIC}^{\mathrm{refit}}(S)$를 선택 |
| 16 | $\widehat S_\mu$, $\widehat\lambda_\mu$, $\widehat\Theta_\mu^{\mathrm{refit}}$와 ADMM·구면 제약 진단을 반환 |

$\mathrm{df}_{\mathrm{nom}}(S)$는 엄밀한 유효 자유도가 아니라 BIC 계산에
사용하는 실용적 명목 차원 근사치이다.
