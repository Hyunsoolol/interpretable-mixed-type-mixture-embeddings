# 연구미팅 자료: Eta-group 방법론과 시뮬레이션 결과 (2026-07-14)

## 1. 핵심 정리

- 선택 대상은 prototype support가 아니라 **posterior decision support**다.
- 자연모수 $\eta_k=\kappa_k\mu_k$의 component 간 centered contrast를 사용한다.
- 주 모형은 coordinate-wise adaptive group penalty인 E-CAGL이다.
- $K$와 sparsity parameter $\lambda_\eta$는 분리해서 선택한다.

전체 수치표는 [시뮬레이션 결과 부록](../simulations/thesis-simulation_260708.md)에 정리했다.

## 2. 제안 penalty

### 2.1 $\mu$가 아니라 $\eta$

vMF mixture에서

$$
\eta_k=\kappa_k\mu_k,
\qquad
s_k(x)=\log\pi_k+\log C_d(\kappa_k)+\eta_k^\top x.
$$

따라서 두 component의 score 차이는

$$
s_k(x)-s_\ell(x)
=a_{k\ell}+(\eta_k-\eta_\ell)^\top x,
$$

$$
a_{k\ell}
=\log\frac{\pi_k}{\pi_\ell}
+\log\frac{C_d(\kappa_k)}{C_d(\kappa_\ell)}.
$$

$\eta_k\neq0$이면

$$
\kappa_k=\lVert\eta_k\rVert_2,
\qquad
\mu_k=\frac{\eta_k}{\lVert\eta_k\rVert_2}.
$$

$\eta_k=0$에서는 $\mu_k$가 식별되지 않으며, mixture label switching은 별도 문제다.

### 2.2 raw $\eta$가 아니라 centered $\eta$

좌표 $j$에 대해

$$
\eta_{\cdot j}=\bar\eta_j\mathbf 1+c_{\cdot j},
\qquad
\bar\eta_j=K^{-1}\sum_{k=1}^K\eta_{kj},
\qquad
\mathbf 1^\top c_{\cdot j}=0.
$$

$$
\eta_{kj}-\eta_{\ell j}=c_{kj}-c_{\ell j}.
$$

따라서 $x_j$가 component 간 선형 score 차이를 만드는 조건은

$$
j\in S_{\mathrm{dec}}
\iff
\lVert c_{\cdot j}\rVert_2>0.
$$

$\bar\eta_j$는 penalized fit에 남으며, $\kappa_k=\lVert\eta_k\rVert_2$를 통해 $C_d(\kappa_k)$에 영향을 줄 수 있다.

### 2.3 entry-wise $L_1$이 아니라 coordinate-wise group $L_2$

$$
P_{\mathrm{CGL}}(\eta)
=\lambda_\eta\sum_{j=1}^d\lVert c_{\cdot j}\rVert_2.
$$

Adaptive 확장은

$$
P_{\mathrm{CAGL}}(\eta)
=\lambda_\eta\sum_{j=1}^d w_j\lVert c_{\cdot j}\rVert_2,
\qquad
w_j=\left(\lVert c_{\cdot j}^{\mathrm{init}}\rVert_2+\epsilon\right)^{-\gamma}.
$$

이번 실험에서는

$$
\gamma=1,
\qquad
\epsilon=10^{-6},
$$

이며 weight에 median normalization을 적용했다. E-CGL은 $w_j=1$인 기본 모형이고, E-CAGL은 선택적 adaptive 확장이다.

### 2.4 비교 모형

| 모형 | penalty |
|:---|:---|
| M-L | $\lambda_\mu\sum_{k,j}\lvert\mu_{kj}\rvert$ |
| M-GL | $\lambda_\mu\sum_j\lVert\mu_{\cdot j}\rVert_2$ |
| M-AGL | $\lambda_\mu\sum_jw_j^{(M)}\lVert\mu_{\cdot j}\rVert_2$ |
| E-CL | $\lambda_\eta\sum_{k,j}\lvert c_{kj}\rvert$ |
| E-CGL | $\lambda_\eta\sum_j\lVert c_{\cdot j}\rVert_2$ |
| E-CAGL | $\lambda_\eta\sum_jw_j^{(E)}\lVert c_{\cdot j}\rVert_2$ |

### 2.5 구조 분해 진단

S1 환경: $K=4$, $n=1000$, $d=200$, common q=4, decision q=16, noise q=180, rep=20.

| 구조 | 모형 | selected q | common q | noise q | F1 | MSE_eta |
|:---|:---|---:|---:|---:|---:|---:|
| $\mu$ group | M-GL | 20.00 | 4.00 | 0.00 | 0.889 | 0.072 |
| raw $\eta$ group | E-GL | 21.15 | 4.00 | 1.15 | 0.862 | 0.089 |
| centered entry-wise | E-CL | 19.05 | 0.05 | 3.00 | 0.915 | 0.098 |
| centered group | E-CGL | 17.50 | 0.00 | 1.50 | 0.958 | 0.079 |
| adaptive centered group | E-CAGL | 16.05 | 0.00 | 0.05 | 0.998 | 0.057 |

여기서 `MSE_eta`는 $\mathrm{MSE}_{\mathrm{centered}\ \eta}$다.

관련 penalty 구조는 다음과 연결된다.

| 문헌 | 핵심 구조 |
|:---|:---|
| Guo et al. (2010) | $\sum_j\sum_{k<\ell}w_{k\ell j}\lvert\mu_{kj}-\mu_{\ell j}\rvert$ |
| Bondell and Reich (2009) | ANOVA level difference와 sum-to-zero constraint |
| Li et al. (2022) | common effect와 cluster-specific deviation 분해 |
| 본 연구 | $\eta_{\cdot j}=\bar\eta_j\mathbf1+c_{\cdot j}$와 coordinate group selection |

## 3. 시뮬레이션 근거

### 3.1 기본 및 negative-control

| 환경 | E-CAGL 결과 | 해석 |
|:---|:---|:---|
| S1-S4: decision q=16 | selected q=16.06-21.22, F1=0.881-0.998, ARI=0.631-0.904 | sparse decision support 복원 |
| S5-S6: 30도 | nonzero refit=1/50, 2/50 | 약한 신호에서 zero-support 선택 |
| S1-N/S2-N: decision q=80 | selected q=81.82-82.40, F1=0.985-0.989 | 큰 평균 차이에서는 dense support도 유지 |
| S3-N | selected q=76.06, F1=0.840 | M-AGL F1=0.877보다 낮음 |
| S4-N | nonzero refit=10/50, F1(all)=0.331 | dense support와 보통 분리에서 tuning failure |
| S5-N/S6-N | nonzero refit=3/50, 3/50 | 약한 신호에서 대부분 zero support |

### 3.2 Shared-background

설정은 common q=80, decision q=20, noise q=100, rep=50이다.

| 모형 | selected q | common q | decision q | noise q | F1 |
|:---|---:|---:|---:|---:|---:|
| M-AGL | 102.40 | 79.96 | 20.00 | 2.44 | 0.327 |
| E-CAGL | 20.48 | 0.30 | 20.00 | 0.18 | 0.988 |

M 계열은 prototype support, E 계열은 posterior decision support를 선택 대상으로 둔다.

### 3.3 Oracle Bayes error 기반 Study B

$K = 4, \quad d = 200, \quad n \in \{300, 1000\}, \quad q_C = 4, \quad q_D = 16, \quad q_N = 180, \quad R = 100.$

```math
e_B \in \{2.5\%, 5\%, 10\%\}, \qquad
\kappa \in \{(45,45,45,45), (30,40,50,60)\}.
```


아래 범위는 equal/heterogeneous $\kappa$ 두 결과의 최솟값과 최댓값이다.

| target $e_B$ | $n$ | selected q | common q | noise q | F1 | ARI | MSE_eta |
|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2.5% | 300 | 17.97-18.23 | 0.02-0.03 | 1.94-2.21 | 0.935-0.942 | 0.930-0.931 | 0.253-0.260 |
| 2.5% | 1000 | 16.03-16.05 | 0.00 | 0.03-0.05 | 0.998-0.999 | 0.936-0.938 | 0.052-0.054 |
| 5.0% | 300 | 16.73-17.95 | 0.03-0.04 | 0.70-1.91 | 0.943-0.978 | 0.866-0.874 | 0.222-0.271 |
| 5.0% | 1000 | 16.06-16.09 | 0.00 | 0.06-0.09 | 0.997-0.998 | 0.880 | 0.056-0.059 |
| 10.0% | 300 | 16.92-19.49 | 0.02-0.09 | 1.35-4.48 | 0.841-0.945 | 0.683-0.703 | 0.355-0.780 |
| 10.0% | 1000 | 16.20-18.94 | 0.01-0.09 | 0.19-2.85 | 0.916-0.994 | 0.722-0.741 | 0.070-0.093 |

Zero-support 반복은 F1=0으로 포함하며, refit이 없는 MSE_eta는 boxplot에서 제외했다.

![Study B F1 boxplot](../simulations/figures/studyb_boxplot_f1_by_eb_n_260714.png)

![Study B selected noise q boxplot](../simulations/figures/studyb_boxplot_noiseq_by_eb_n_260714.png)

![Study B log MSE eta boxplot](../simulations/figures/studyb_boxplot_logmse_eta_by_eb_n_260714.png)

## 4. $K$와 $\lambda_\eta$ 선택

동시 선택 진단은 $K^\ast=4$, $n=1000$, $d=200$, $e_B=5\%$, rep=5에서 수행했다.

| 방법 | equal $\kappa$ | heterogeneous $\kappa$ |
|:---|:---|:---|
| Dense vMF | BIC: $K=4$; EBIC: $K=2$ | BIC: $K=3$; EBIC: $K=2$ |
| M-GL/M-AGL | 대부분 또는 전부 $K=4$ | 전부 $K=4$ |
| E-CAGL all-in-one | 주로 $K=7,8$ | 주로 $K=7,8$ |

현재 선택 절차는

$$
\widehat K
=\arg\min_{K\in\mathcal K}\mathrm{IC}_{\mathrm{dense/group}}(K),
$$

$$
\widehat\lambda_\eta
=\arg\min_{\lambda_\eta}
\mathrm{BIC}(\widehat K,\lambda_\eta)
$$

의 2단계로 분리한다. Rossi and Barbaro (2022)의 dense-$K$ 선택 후 sparsity path 선택 구조와 같은 방향이다.

현재 centered eta BIC는

$$
\mathrm{BIC}(\lambda_\eta)
=-2\ell(\widehat\Theta_{\lambda_\eta})
+\log(n)\left[(K-1)+d+(K-1)m_{\lambda_\eta}\right]
$$

이며 penalized path에서 BIC를 선택한 뒤 support refit을 수행한다.

## 5. 비용과 한계

| 항목 | 확인 결과 |
|:---|:---|
| 계산 시간 | E-CAGL 5.53초; M-L 3.75초; M-GL/M-AGL 8.82/8.53초 |
| 약한 신호 | S5/S6에서 대부분 zero support |
| dense support | S3-N/S4-N에서 과소선택 또는 tuning failure |
| $K$ 동시 선택 | E-CAGL all-in-one에서 $K=7,8$ 선호 |
| refit 정의 | penalty 단계는 common $\eta$ baseline을 유지하지만 현재 refit은 selected coordinate만 유지 |
| BIC df | refit target과 df의 일관성 추가 점검 필요 |

## 6. 현재 결론

$$
\text{posterior decision support} \Rightarrow \eta = \kappa\mu \Rightarrow c_{kj} = \eta_{kj} - \bar{\eta}_{j} \Rightarrow \sum_{j} w_j \|c_{\cdot j}\|_2
$$

* E-CAGL은 sparse posterior decision support 복원에 초점을 둔다.
* 약한 신호, 일부 dense-support 환경, $K$와 $\lambda_\eta$의 동시 선택에서는 성능 저하가 관찰됐다.
* 다음 검증 항목은 refit/df 정합성과 동일한 $\mu$에서 $\kappa$만 다른 concentration-only 환경이다.
