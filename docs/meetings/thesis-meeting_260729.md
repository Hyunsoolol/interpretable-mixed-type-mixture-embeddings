# 연구미팅: M-CGL 도입 검토와 E-CGL의 관계 (2026-07-29)

## 1. 검토 배경

지난 연구미팅에서는 E-CGL과 Rossi 계열 M-L의 패널티 대상과 구조가 달라
직접 비교의 정합성이 낮다는 의견이 있었다.

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
교차 지표이다. MSE는 centered-$\eta$ 손실을 최소화하는 component
permutation으로 label을 정렬한 후 계산했으며,
$\mathrm{MSE}_\eta$는 centered-$\eta$ contrast의 MSE이다.

- 공통 $\kappa$: $S_\mu=S_\eta$이며 M/E 계열의 support 회복이 유사하였다.
- 이질적 $\kappa$: 참 support는 $(q_\mu,q_\eta)=(20,16)$이다.
- M-CGL의 common 좌표 선택은 directional target에 포함된다.
- E-CGL은 posterior-score target 16개를 선택하였다.

## 7. 실행시간

구조적 진단 설정 $(K,n,d)=(4,400,24)$, path 21에서 측정한 Rcpp 적용 후
평균 초/rep이다. `sourceCpp` 컴파일·로드 시간은 제외하였다.

| 구조 | M-CGL | M-ACGL | E-CGL | E-ACGL |
|---|---:|---:|---:|---:|
| 공통 $\kappa$ | 6.2 | 8.1 | 1.9 | 2.0 |
| 이질적 $\kappa$ | 10.0 | 8.4 | 2.6 | 3.0 |
| 공통 $\eta$ | 4.3 | 6.1 | 3.1 | 3.5 |
| 공통 $\mu$ | 20.4 | 24.4 | 17.3 | 16.8 |

M-CGL은 단위구면 제약과 ADMM 내부 반복을 포함하므로 E-CGL보다 계산량이
크다.

## 8. 연구미팅 결론

| 항목 | 결론 |
|---|---|
| M-CGL의 수학적 성립 | 단위구면 제약을 포함한 centered directional-support 최적화로 성립 |
| 공통 $\kappa$ | $S_\mu=S_\eta$이며 두 방법의 target이 일치 |
| 이질적 $\kappa$ | $S_\mu$와 $S_\eta$가 서로 다른 이질성을 나타냄 |
| E-CGL | posterior-score heterogeneity의 주 제안 모형 |
| M-CGL | directional heterogeneity의 matched companion |
| Adaptive 확장 | E-ACGL과 M-ACGL은 보조 분석 |
| 논문 평가축 | target-specific recovery, oracle-support gap, 수렴 안정성 |
| 기존 비교 | Dense vMF와 Rossi sparse prototype을 최소 비교군으로 유지 |
| 남은 검증 | M-CGL 내부 stationarity, 공집합 support 선택, nominal-df 민감도 |

---

## 백업 자료: Guarded path algorithms

### Algorithm 1. E-CGL and E-ACGL

**Input:** $X$, $K$, path size $L$, method, iteration limits,
$\varepsilon_{\mathrm{conv}}$, $\varepsilon_{\mathrm{acc}}$

**Output:** $(\widehat S_\eta,\widehat\lambda_\eta)$,
$\widehat\Theta_\eta^{\mathrm{refit}}$

| 단계 | 절차 |
|---:|---|
| 1 | 여러 초기값에서 dense vMF를 적합하고 최대 log-likelihood 해를 선택 |
| 2 | E-CGL은 $w_j=1$; E-ACGL은 dense fit에서 adaptive weight를 계산한 뒤 고정 |
| 3 | dense-to-sparse path $\Lambda_\eta$를 구성 |
| 4 | 각 $\lambda_\eta$에서 직전 accepted fit을 warm start로 사용 |
| 5 | E-step에서 $\tau_{ik}$, $N_k$, $r_k$를 계산 |
| 6 | $\pi_k$와 centered-$\eta$ group proximal M-step을 갱신 |
| 7 | majorization 또는 observed criterion이 감소하면 step size를 축소 |
| 8 | 수렴 후 $S_{\eta,\lambda}$와 numerical diagnostics를 저장 |
| 9 | 각 distinct support에서 $c_{kj}^{(\eta)}=0$ 제약 refit을 수행 |
| 10 | BIC-after-refit이 최소인 $\widehat S_\eta$를 선택 |

### Algorithm 2. M-CGL and M-ACGL

**Input:** $X$, $K$, path size $L$, method, iteration limits,
$\varepsilon_{\mathrm{conv}}$, $\varepsilon_{\mathrm{acc}}$

**Output:** $(\widehat S_\mu,\widehat\lambda_\mu)$,
$\widehat\Theta_\mu^{\mathrm{refit}}$

| 단계 | 절차 |
|---:|---|
| 1 | 여러 초기값에서 dense vMF를 적합하고 최대 log-likelihood 해를 선택 |
| 2 | M-CGL은 $w_j=1$; M-ACGL은 dense fit에서 adaptive weight를 계산한 뒤 고정 |
| 3 | centered-$\mu$ norm을 기준으로 path $\Lambda_\mu$를 구성 |
| 4 | 각 $\lambda_\mu$에서 직전 accepted fit을 warm start로 사용 |
| 5 | E-step에서 $\tau_{ik}$, $N_k$, $r_k$를 계산 |
| 6 | ADMM에서 product-of-spheres $\mu$-update와 group thresholding을 수행 |
| 7 | $A_d(\kappa_k)=r_k^{\mathsf T}\mu_k/N_k$의 수치적 근으로 $\kappa_k$를 갱신 |
| 8 | criterion이 감소하면 step halving 또는 dense restart를 적용 |
| 9 | 각 distinct support에서 $c_{kj}^{(\mu)}=0$과 $\lVert\mu_k\rVert_2=1$ 제약 refit을 수행 |
| 10 | BIC-after-refit이 최소인 $\widehat S_\mu$를 선택 |
