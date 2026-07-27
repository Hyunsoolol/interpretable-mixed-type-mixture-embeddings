# 연구미팅: M-CGL 도입 검토 (2026-07-29)

## 1. 검토 배경

지난 연구미팅 피드백:

> E-CGL과 Rossi M-L은 패널티 대상과 구조가 달라 직접 비교의 정합성이 낮다.



$$
\mathrm{M-L}
:
(\mu,\ \mathrm{entry-wise}\ L_1,\ \mathrm{uncentered}),
\qquad
\mathrm{E-CGL}
:
(\boldsymbol{c}_j^{(\eta)}=\boldsymbol{\eta}_j-\bar\eta_j\mathbf{1}_K,\   \mathrm{group}\ L_2,\ \mathrm{centered}).
$$

$$
\mathrm{Matched\ comparator}
:
\qquad
\mathrm{M-CGL}=
(\boldsymbol{c}_j^{(\mu)}=\boldsymbol{\mu}_j-\bar\mu_j\mathbf{1}_K,\ \mathrm{group}\ L_2,\ \mathrm{centered}).
$$

## 2. 추정 대상

$$
x_i\mid z_i=k\sim\mathrm{vMF}(\mu_k,\kappa_k),
\quad \eta_k=\kappa_k\mu_k,
\quad
\log\frac{\Pr(Z=k\mid x)}{\Pr(Z=\ell\mid x)}=
a_{k\ell}+(\eta_k-\eta_\ell)^{\mathsf T}x.
$$

$$
\boldsymbol{\mu}_j=(\mu_{1j},\ldots,\mu_{Kj})^{\mathsf T},
\qquad
\bar\mu_j=\frac{1}{K}\sum_{k=1}^{K}\mu_{kj},
\qquad
\boldsymbol{c}_j^{(\mu)}=\boldsymbol{\mu}_j-\bar\mu_j\mathbf{1}_K,
\qquad
c_{kj}^{(\mu)}=\mu_{kj}-\bar\mu_j.
$$

$$
\boldsymbol{\eta}_j=(\eta_{1j},\ldots,\eta_{Kj})^{\mathsf T},
\qquad
\bar\eta_j=\frac{1}{K}\sum_{k=1}^{K}\eta_{kj},
\qquad
\boldsymbol{c}_j^{(\eta)}=\boldsymbol{\eta}_j-\bar\eta_j\mathbf{1}_K,
\qquad
c_{kj}^{(\eta)}=\eta_{kj}-\bar\eta_j.
$$

| 대상 | Support | 모형 |
|---|---|---|
| prototype | $S_P=\{j:\lVert\boldsymbol{\mu}_j\rVert_2>0\}$ | M-L |
| directional heterogeneity | $S_\mu=\{j:\lVert\boldsymbol{c}_j^{(\mu)}\rVert_2>0\}$ | M-CGL |
| posterior-score heterogeneity | $S_\eta=\{j:\lVert\boldsymbol{c}_j^{(\eta)}\rVert_2>0\}$ | E-CGL |

## 3. Centered group penalty

$$
\widehat\Theta_{\lambda_\mu}^{\mathrm{M-CGL}}=
\underset{\pi,\kappa,\mu}{\arg\max}
\Big[
\ell(\pi,\kappa,\mu)
-\lambda_\mu\sum_{j=1}^{d}\lVert\boldsymbol{c}_j^{(\mu)}\rVert_2
\Big],
\qquad
\lVert\mu_k\rVert_2=1.
$$

$$
\widehat\Theta_{\lambda_\eta}^{\mathrm{E-CGL}}=
\underset{\pi,\eta}{\arg\max}
\Big[
\ell(\pi,\eta)
-\lambda_\eta\sum_{j=1}^{d}\lVert\boldsymbol{c}_j^{(\eta)}\rVert_2
\Big].
$$

$\boldsymbol{c}_j^{(\mu)}$는 방향 모수가 아니라 성분 간 좌표 contrast이다. 예를 들어
$\mu_1=(a,b)$, $\mu_2=(a,-b)$, $a^2+b^2=1$이면 단위구면 제약과 첫 번째
좌표 equality가 동시에 성립한다.

## 4. 이론적 관계

$$
\kappa_1=\cdots=\kappa_K=\kappa
\quad\Longrightarrow\quad
c_{kj}^{(\eta)}=\kappa c_{kj}^{(\mu)}
\quad\Longrightarrow\quad
S_\eta=S_\mu.
$$

$$
\boldsymbol{c}_j^{(\eta)}=
\big(\kappa_1\mu_{1j}-\bar\eta_j,\ldots,
\kappa_K\mu_{Kj}-\bar\eta_j\big)^{\mathsf T},
\qquad
\kappa_k\ \mathrm{heterogeneous}
\quad\Longrightarrow\quad
S_\eta\neq S_\mu\ \mathrm{가능}.
$$

---

## 5. 구조적 진단

$$
K=4,\quad n=400,\quad d=24,\quad \mathrm{nstart}=5,\quad
\mathrm{path}=21.
$$

Support selection: BIC-after-refit.

| 구조 | 반복 | 참 $(q_\mu,q_\eta)$ | M-CGL | E-CGL |
|---|---:|---:|---:|---:|
| 공통 $\kappa$ | 10 | $(8,8)$ | $q=8$, $F_{1,\mu}=1.000$ | $q=8$, $F_{1,\eta}=1.000$ |
| 이질적 $\kappa$ | 10 | $(8,12)$ | $q=8$, $F_{1,\mu}=1.000$ | $q=12.1$, $F_{1,\eta}=0.996$ |
| 공통 $\eta$ | 5 | $(12,8)$ | $q=12$, $F_{1,\mu}=1.000$ | $q=8$, $F_{1,\eta}=1.000$ |
| 공통 $\mu$ | 3 | $(0,10)$ | empty 선택 불안정 | $q=12.67$, $F_{1,\eta}=0.885$ |

## 6. Study B matched pilot

$$
K=4,\quad n=300,\quad d=200,\quad e_B=0.05,\quad
(q_C,q_D,q_N)=(4,16,180),\quad \mathrm{rep}=3.
$$

Path 120, BIC-after-refit을 사용하였다. 괄호는
$(\widehat q_C,\widehat q_D,\widehat q_N)$이다.

$F_{1,\mu}$는 추정 support와 참 directional support $S_\mu$의 일치도이고,
$F_{1,\eta}$는 참 posterior-score support $S_\eta$와의 일치도이다.
`-`는 성능 0이 아니라 해당 target 기준을 보고하지 않았음을 뜻한다.

| $\kappa$ | 모형 | selected $q$ | $F_{1,\mu}$ | $F_{1,\eta}$ | ARI | $\mathrm{MSE}_\mu$ | $\mathrm{MSE}_\kappa$ | $\mathrm{MSE}_\eta$ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 공통 | M-CGL | 16.7 (0.3, 16.0, 0.3) | 0.980 | 0.980 | 0.841 | $4.63\times10^{-4}$ | 8.995 | 0.276 |
| 공통 | M-ACGL | 16.3 (0.0, 16.0, 0.3) | 0.990 | 0.990 | 0.841 | $4.55\times10^{-4}$ | 8.309 | 0.261 |
| 공통 | E-CGL | 16.3 (0.0, 16.0, 0.3) | - | 0.990 | 0.858 | $4.57\times10^{-4}$ | 6.127 | 0.253 |
| 공통 | E-ACGL | 16.3 (0.0, 16.0, 0.3) | - | 0.990 | 0.858 | $4.57\times10^{-4}$ | 6.134 | 0.253 |
| 이질적 | M-CGL | 18.3 (3.3, 14.7, 0.3) | 0.933 | 0.851 | 0.822 | $8.24\times10^{-4}$ | 14.418 | 0.545 |
| 이질적 | M-ACGL | 17.3 (1.3, 16.0, 0.0) | 0.928 | 0.960 | 0.852 | $5.63\times10^{-4}$ | 9.174 | 0.383 |
| 이질적 | E-CGL | 16.0 (0.0, 16.0, 0.0) | - | 1.000 | 0.855 | $5.09\times10^{-4}$ | 10.359 | 0.229 |
| 이질적 | E-ACGL | 16.0 (0.0, 16.0, 0.0) | - | 1.000 | 0.855 | $5.09\times10^{-4}$ | 10.332 | 0.228 |

- 공통 $\kappa$: $S_\mu=S_\eta$이며 M/E 계열의 support 회복이 유사하였다.
- 이질적 $\kappa$: 참 support는 $(q_\mu,q_\eta)=(20,16)$이다. M-CGL의
  common 좌표 선택은 directional target에 포함되며, E-CGL은
  posterior-score target 16개를 선택하였다.
- 이질적 $\kappa$의 평균 초/rep은 M-CGL 257.5, M-ACGL 133.6,
  E-CGL 26.7, E-ACGL 24.4였다.
- $n=1000$ 예비 1회에서 공통 $\kappa$는 네 모형 모두 decision 16개를
  선택하였다. 이질적 $\kappa$에서는 M-CGL이 20개, E-CGL이 16개를
  선택하였다.

## 7. 실행시간

Rcpp 적용 후 평균 초/rep이며, `sourceCpp` 컴파일·로드 시간은 제외하였다.

| 구조 | M-CGL | M-ACGL | E-CGL | E-ACGL |
|---|---:|---:|---:|---:|
| 공통 $\kappa$ | 3.6 | 4.7 | 2.2 | 2.1 |
| 이질적 $\kappa$ | 6.5 | 7.2 | 2.7 | 3.6 |
| 공통 $\eta$ | 7.4 | 6.8 | 2.9 | 3.5 |
| 공통 $\mu$ | 20.4 | 24.4 | 17.3 | 16.8 |

모든 모형의 반복 계산에는 Rcpp를 사용하였다.

## 8. E-CGL과 M-CGL 비교

| 구분 | M-CGL | E-CGL |
|---|---|---|
| 패널티 대상 | $c_{kj}^{(\mu)}=\mu_{kj}-\bar\mu_j$ | $c_{kj}^{(\eta)}=\eta_{kj}-\bar\eta_j$ |
| 선택 의미 | 군집 간 방향 차이 | 군집 간 posterior-score 차이 |
| 적합한 상황 | 집중도와 무관한 방향 이질성이 연구 대상인 경우 | 실제 군집 구분에 사용되는 좌표가 연구 대상인 경우 |
| 이질적 $\kappa_k$ | 방향 차이만 반영 | 방향과 집중도 차이를 함께 반영 |
| 계산 구조 | 단위구면 제약을 포함한 결합 최적화 | 자연모수 공간의 proximal 최적화 |
| 논문 내 역할 | directional companion | 주 제안 모형 |

공통 집중도에서는 두 support가 일치한다.

$$
\kappa_1=\cdots=\kappa_K
\quad\Longrightarrow\quad
S_\mu=S_\eta.
$$

| 연구 목적 | 모형 |
|---|---|
| 군집별 평균 방향이 다른 좌표 선택 | M-CGL |
| posterior 분류에 기여하는 좌표 선택 | E-CGL |
| 집중도 차이를 군집 구분 정보에 포함 | E-CGL |
| 약한 신호에 adaptive weight 적용 | M-ACGL 또는 E-ACGL |

$$
\boxed{
\mathrm{E-CGL}:\text{posterior-score support의 주 모형},
\qquad
\mathrm{M-CGL}:\text{방향 이질성의 대응 모형}
}
$$

## 9. Guarded path algorithms


### Algorithm 1. Guarded path algorithm for M-CGL and M-ACGL

**Input:** $X$, $K$, path size $L$, method indicator, iteration limits,
$\varepsilon_{\mathrm{conv}}$, $\varepsilon_{\mathrm{acc}}$

**Output:** $(\widehat S_\mu,\widehat\lambda_\mu)$,
$\widehat\Theta_\mu^{\mathrm{refit}}$, numerical diagnostics

| 단계 | 절차 |
|---:|---|
|  | **Stage 1: Dense start and path construction** |
| 1 | 여러 초기값에서 dense vMF를 적합하고 최대 log-likelihood 해를 선택 |
| 2 | M-CGL은 $w_j=1$; M-ACGL은 dense fit에서 $w_j$를 계산한 뒤 고정 |
| 3 | centered-$\mu$ norm의 $\lambda_{\max}$ proxy로 geometric path $\Lambda_\mu=(0,\lambda_{\mu,1},\ldots,\lambda_{\mu,L-1})$ 구성 |
|  | **Stage 2: Guarded penalized path** |
| 4 | 각 $\lambda_\mu\in\Lambda_\mu$에서 직전 accepted fit을 warm start로 사용 |
| 5 | E-step: $\tau_{ik}$, $N_k=\sum_i\tau_{ik}$, $r_k=\sum_i\tau_{ik}x_i$ 계산 |
| 6 | M-step: $\pi_k^{+}=N_k/n$ 갱신 |
| 7 | $Z=C^{(\mu)}$ 분할변수를 두고 ADMM 수행 |
| 8 | $\mu$-update: product of spheres에서 Rcpp tangent-gradient와 retraction 수행 |
| 9 | $Z$-update: coordinate-wise group soft-thresholding 후 dual variable 갱신 |
| 10 | $\kappa$-update: $A_d(\kappa_k)=r_k^{\mathsf T}\mu_k/N_k$의 수치적 근 계산 |
| 11 | penalized 보조함수와 observed criterion이 감소하면 step halving; 실패 시 dense start로 재시도 |
| 12 | 상대 criterion 변화가 $\varepsilon_{\mathrm{conv}}$ 미만이 될 때까지 5--11 반복하고 $S_{\mu,\lambda}$ 저장 |
|  | **Stage 3: Exact refit and support selection** |
| 13 | 각 distinct $S$에 대해 $j\notin S\Rightarrow c_{kj}^{(\mu)}=0$ 및 $\lVert\mu_k\rVert_2=1$ 제약하에서 refit |
| 14 | $\mathrm{BIC}^{\mathrm{refit}}(S)=-2\ell(\widehat\Theta_S^{\mathrm{refit}})+\log(n)\mathrm{df}(S)$ 계산 |
| 15 | $\widehat S_\mu=\arg\min_S\mathrm{BIC}^{\mathrm{refit}}(S)$ 선택 |
| 16 | $\widehat S_\mu$, $\widehat\lambda_\mu$, $\widehat\Theta_\mu^{\mathrm{refit}}$ 및 ADMM·구면 제약 진단 반환 |

### Algorithm 2. Guarded path algorithm for E-CGL and E-ACGL

**Input:** $X$, $K$, path size $L$, method indicator, iteration limits,
$\varepsilon_{\mathrm{conv}}$, $\varepsilon_{\mathrm{acc}}$

**Output:** $(\widehat S_\eta,\widehat\lambda_\eta)$,
$\widehat\Theta_\eta^{\mathrm{refit}}$, numerical diagnostics

| 단계 | 절차 |
|---:|---|
|  | **Stage 1: Dense start and path construction** |
| 1 | 여러 초기값에서 dense vMF를 적합하고 최대 log-likelihood 해를 선택 |
| 2 | E-CGL은 $w_j=1$; E-ACGL은 dense fit에서 $w_j$를 계산한 뒤 고정 |
| 3 | dense-to-sparse KKT-geometric path $\Lambda_\eta=(0,\lambda_{\eta,1},\ldots,\lambda_{\eta,L-1})$ 구성 |
|  | **Stage 2: Guarded penalized path** |
| 4 | 각 $\lambda_\eta\in\Lambda_\eta$에서 직전 accepted fit을 warm start로 사용 |
| 5 | E-step: $\tau_{ik}$, $N_k=\sum_i\tau_{ik}$, $r_k=\sum_i\tau_{ik}x_i$ 계산 |
| 6 | M-step: $\pi_k^{+}=N_k/n$ 및 centered-$\eta$ group proximal update |
| 7 | majorization 조건이 성립할 때까지 step size를 절반으로 축소 |
| 8 | 보조함수 또는 penalized log-likelihood가 허용범위보다 감소하면 path fit을 중단하고 이전 accepted estimate를 반환 |
| 9 | 상대 criterion 변화가 $\varepsilon_{\mathrm{conv}}$ 미만이 될 때까지 5--8 반복 |
| 10 | $S_{\eta,\lambda}$, criterion 및 수치 진단 저장 |
| 11 | path 종료 조건 또는 $L$에 도달할 때까지 4--10 반복 |
|  | **Stage 3: Exact refit and support selection** |
| 12 | path에서 중복 support 제거 |
| 13 | 각 $S$에 대해 $j\notin S\Rightarrow c_{kj}^{(\eta)}=0$ 제약하에서 exact refit |
| 14 | $\mathrm{BIC}^{\mathrm{refit}}(S)=-2\ell(\widehat\Theta_S^{\mathrm{refit}})+\log(n)\mathrm{df}(S)$ 계산 |
| 15 | $\widehat S_\eta=\arg\min_S\mathrm{BIC}^{\mathrm{refit}}(S)$ 선택 |
| 16 | $\widehat S_\eta$, $\widehat\lambda_\eta$, $\widehat\Theta_\eta^{\mathrm{refit}}$ 및 진단 반환 |
