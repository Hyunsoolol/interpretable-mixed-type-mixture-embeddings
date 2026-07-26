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
(\boldsymbol{c}_j^{(\eta)},\ \mathrm{group}\ L_2,\ \mathrm{centered}).
$$

$$
\mathrm{Matched\ comparator}
:
\qquad
\mathrm{M-CGL}=
(\boldsymbol{c}_j^{(\mu)},\ \mathrm{group}\ L_2,\ \mathrm{centered}).
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

| $\kappa$ | 모형 | selected $q$ | $F_{1,\mu}$ | $F_{1,\eta}$ | ARI | $\mathrm{MSE}_\eta$ |
|---|---|---:|---:|---:|---:|---:|
| 공통 | M-CGL | 16.7 (0.3, 16.0, 0.3) | 0.980 | 0.980 | 0.841 | 0.276 |
| 공통 | M-ACGL | 16.3 (0.0, 16.0, 0.3) | 0.990 | 0.990 | 0.841 | 0.261 |
| 공통 | E-CGL | 16.3 (0.0, 16.0, 0.3) | - | 0.990 | 0.858 | 0.253 |
| 공통 | E-ACGL | 16.3 (0.0, 16.0, 0.3) | - | 0.990 | 0.858 | 0.253 |
| 이질적 | M-CGL | 18.3 (3.3, 14.7, 0.3) | 0.933 | 0.851 | 0.822 | 0.545 |
| 이질적 | M-ACGL | 17.3 (1.3, 16.0, 0.0) | 0.928 | 0.960 | 0.852 | 0.383 |
| 이질적 | E-CGL | 16.0 (0.0, 16.0, 0.0) | - | 1.000 | 0.855 | 0.229 |
| 이질적 | E-ACGL | 16.0 (0.0, 16.0, 0.0) | - | 1.000 | 0.855 | 0.228 |

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

## 8. 수치 검증 및 논문 배치

| 검증 | 결과 |
|---|---:|
| 단위구면 / inactive equality 오차 | $\leq2.22\times10^{-16}$ / $0$ |
| ADMM primal residual | $1.06\times10^{-10}$ |
| M-ACGL weight 및 $w_j=1$ 환원 | PASS |
| truth-blind restart 후 M 계열 BIC 변경 | $3/6$ |
| 내부 M-step 최종 stationarity | 추가 확인 필요 |

$$
\boxed{
\text{E-CGL: primary}
\qquad
\text{M-CGL: directional companion}
\qquad
\text{E-ACGL, M-ACGL: Supplement}
}
$$

본문에는 $S_\mu$와 $S_\eta$의 관계와 matched structural diagnostic을
포함하고, M-CGL의 구면 최적화, 수렴 진단과 adaptive 결과는 Supplement에
배치하는 게 어떨지.
