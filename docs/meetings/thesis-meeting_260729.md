# 연구미팅: M-CGL 도입 검토 (2026-07-29)

## 1. 검토 배경

지난 연구미팅 피드백:

> E-CGL과 Rossi M-L은 패널티 대상과 구조가 달라 직접 비교의 정합성이 낮다.

$$
\mathrm{M\!-\!L}
:
(\mu,\ \mathrm{entry\!-\!wise}\ L_1,\ \mathrm{uncentered}),
\qquad
\mathrm{E\!-\!CGL}
:
(H_KE,\ \mathrm{group}\ L_2,\ \mathrm{centered}).
$$

$$
\mathrm{Matched\ comparator}
:
\qquad
\mathrm{M\!-\!CGL}
=
(H_KM,\ \mathrm{group}\ L_2,\ \mathrm{centered}).
$$

## 2. 추정 대상

$$
x_i\mid z_i=k\sim\mathrm{vMF}(\mu_k,\kappa_k),
\quad \eta_k=\kappa_k\mu_k,
\quad
\log\frac{\Pr(Z=k\mid x)}{\Pr(Z=\ell\mid x)}
=a_{k\ell}+(\eta_k-\eta_\ell)^{\mathsf T}x.
$$

$$
H_K=I_K-\frac{1}{K}{\boldsymbol 1}_K{\boldsymbol 1}_K^{\mathsf T},
\qquad
E=D_\kappa M,
\qquad
D_\kappa=\mathrm{diag}(\kappa_1,\ldots,\kappa_K).
$$

| 대상 | Support | 모형 |
|---|---|---|
| prototype | $S_P=\{j:\lVert M_{\cdot j}\rVert_2>0\}$ | M-L |
| directional heterogeneity | $S_\mu=\{j:\lVert(H_KM)_{\cdot j}\rVert_2>0\}$ | M-CGL |
| posterior-score heterogeneity | $S_\eta=\{j:\lVert(H_KE)_{\cdot j}\rVert_2>0\}$ | E-CGL |

## 3. Centered group penalty

$$
\widehat\Theta_{\lambda_\mu}^{\mathrm{M-CGL}}
=
\underset{\pi,\kappa,M}{\arg\max}
\Big[
\ell(\pi,\kappa,M)
-\lambda_\mu\sum_{j=1}^{d}\lVert(H_KM)_{\cdot j}\rVert_2
\Big],
\qquad
\lVert\mu_k\rVert_2=1.
$$

$$
\widehat\Theta_{\lambda_\eta}^{\mathrm{E-CGL}}
=
\underset{\pi,E}{\arg\max}
\Big[
\ell(\pi,E)
-\lambda_\eta\sum_{j=1}^{d}\lVert(H_KE)_{\cdot j}\rVert_2
\Big].
$$

$H_KM$은 방향 모수가 아니라 성분 간 좌표 contrast이다. 예를 들어
$\mu_1=(a,b)$, $\mu_2=(a,-b)$, $a^2+b^2=1$이면 단위구면 제약과 첫 번째
좌표 equality가 동시에 성립한다.

## 4. 이론적 관계

$$
\begin{aligned}
\kappa&=\bar\kappa{\boldsymbol 1}_K+h,
&
M&={\boldsymbol 1}_K\bar\mu^{\mathsf T}+U,
&
U&=H_KM,\\
H_KE
&=\bar\kappa U
+h\bar\mu^{\mathsf T}
+H_K\mathrm{diag}(h)U.
\end{aligned}
$$

$$
\kappa_1=\cdots=\kappa_K=\kappa
\ \Longrightarrow\
H_KE=\kappa H_KM
\ \Longrightarrow\
S_\eta=S_\mu,
\qquad
\kappa_k\ \mathrm{heterogeneous}
\ \Longrightarrow\
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

## 6. 실행시간

동일 반복에서의 평균 초/rep이며, `sourceCpp` 컴파일·로드 시간은 제외하였다.

| 구조 | M-CGL R $\rightarrow$ Rcpp | M-ACGL R $\rightarrow$ Rcpp | E-CGL Rcpp | E-ACGL Rcpp |
|---|---:|---:|---:|---:|
| 공통 $\kappa$ | $26.3\rightarrow3.6$ | $37.3\rightarrow4.7$ | 2.2 | 2.1 |
| 이질적 $\kappa$ | $51.5\rightarrow6.5$ | $58.3\rightarrow7.2$ | 2.7 | 3.6 |
| 공통 $\eta$ | $60.4\rightarrow7.4$ | $56.1\rightarrow6.8$ | 2.9 | 3.5 |
| 공통 $\mu$ | $141.5\rightarrow20.4$ | $190.9\rightarrow24.4$ | 17.3 | 16.8 |

M 계열은 ADMM 제어를 R에 유지하고, 반복되는 구면 목적함수와
product-of-spheres update를 Rcpp로 계산하였다. 기존 R 구현 대비
$6.9$--$8.2$배 단축되었으나, ADMM 내부 반복이 남아 있어 일반 구조에서는
E 계열보다 약 $1.6$--$2.5$배 오래 걸렸다. 공통 $\mu$는 M 계열의 참
support가 공집합인 경계 구조로, support 선택과 refit이 상대적으로 오래 걸렸다.

## 7. 수치 검증 및 논문 배치

| 검증 | 결과 |
|---|---:|
| 단위구면 / inactive equality 오차 | $\leq2.22\times10^{-16}$ / $0$ |
| ADMM primal residual | $1.06\times10^{-10}$ |
| M-ACGL weight 및 $w_j=1$ 환원 | PASS |
| R 대비 Rcpp support / selected $q$ / BIC 선택 | 동일 |
| Rcpp 수치 차이 | 일반 구조 $\leq10^{-8}$; 공통 $\mu$ 경계 $\leq5\times10^{-8}$ |
| M 계열 Rcpp 단축률 | $6.9$--$8.2$배 |
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
포함하고, M-CGL의 구면 최적화·수렴 진단과 adaptive 결과는 Supplement에
배치한다.
