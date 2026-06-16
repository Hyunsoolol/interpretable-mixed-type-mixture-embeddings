# 연구 진행 정리

## 1. 배경

### 1.1 문제 설정과 선행 연구
비지도 상황에서 텍스트 임베딩 $x_i = \phi(d_i) \in \mathbb{R}^d$ 의 군집화. 고차원 과제:
(i) 군집별 분산 추정, (ii) 변수 선택.

- **Directional clustering** (Banerjee et al., 2005)

$$f(z \mid \mu, \kappa) = c_d(\kappa)\exp(\kappa \mu^\top z), \quad z, \mu \in \mathbb{S}^{d-1}$$

- **Sparse vMF mixture** (Rossi & Barbaro, 2022)

$$\mathcal{L}_p(\Theta) = \mathcal{L}(\Theta) - \beta \sum_h \|\mu_h\|_1$$

한계: penalized estimator 최종 사용 ($L_1$ bias), penalty가 $\mu_h$ 에 → 판별항 $\kappa_h \mu_h$ 미반영.

- **Cluster-contrast 변수 선택** (Pan & Shen, 2007; Wang & Zhu, 2008; Xie et al., 2008)

$$\mu_{hj} = \mu_{\ell j} \ \forall h, \ell \iff j \text{ 는 noise variable}$$

- **Two-stage Lasso-MLE** (Meynet 계열)

$$\widehat{S}_\lambda \xrightarrow{\text{screening}} \widehat{\Theta}^{\text{refit}}_{\widehat{S}} \xrightarrow{\text{unpenalized MLE}} \text{final}$$

### 1.2 GMM 한계와 vMF 전환

| 모형 | Sample space | cluster당 퍼짐 모수 | d=768 |
|---|---|---|---|
| GMM full cov. | $\mathbb{R}^{d}$ | $d(d+1)/2$ | $\approx 3 \times 10^5$ |
| GMM diag. cov. | $\mathbb{R}^{d}$ | $d$ | 768 |
| GMM spherical | $\mathbb{R}^{d}$ | 1 | 1 |
| **vMF** | $\mathbb{S}^{d-1}$ | 1 ($\kappa_h$) | 1 |

$$\mu_h \text{ 비슷}, \ \Sigma_h \text{ 다름} \ (\text{GMM}) \quad \longleftrightarrow \quad \mu_h \text{ 비슷}, \ \kappa_h \text{ 다름} \ (\text{vMF})$$

<img width="634" height="498" alt="image" src="https://github.com/user-attachments/assets/1dbe9aa3-0291-4f65-b7ac-4de9c95105ba" />

### 1.3 vMF mixture

$$f(z \mid \mu, \kappa) = c_d(\kappa)\exp(\kappa\mu^\top z), \qquad z, \mu \in \mathbb{S}^{d-1}$$

$$p(z_i \mid \Theta) = \sum_{h=1}^{K} \pi_h c_d(\kappa_h) \exp(\kappa_h \mu_h^\top z_i)$$

<img width="440" height="319" alt="image" src="https://github.com/user-attachments/assets/b7aea979-1e98-4169-afaf-28f9f4645d15" />
<img width="626" height="380" alt="image" src="https://github.com/user-attachments/assets/d2713c3a-7dc7-4392-8074-b6bf49a96eaa" />

### 1.4 채택 입장
vMF mixture를 **parsimonious directional working model** 로 채택. 정확한 분포 가정이 아님.

---

## 2. 연구 질문

- **RQ1.** $\eta_h = \kappa_h \mu_h$ 에 대한 cluster-contrast penalty 가 기존 $\mu$ 기반 penalty (Rossi & Barbaro, 2022) 보다 변수 선택과 군집화 성능에서 우월한가?
- **RQ2.** Two-stage refit 이 penalized estimator 를 그대로 쓰는 것보다 추정 편향과 군집 정확도를 개선하는가?
- **RQ3.** 단일 페널티 (C) 와 이중 페널티 (B) 중 directional sparse clustering 에 더 적합한 형태는?

## 3. 기여

1. **방법론:** vMF 자연모수 $\eta_h = \kappa_h \mu_h$ 에 cluster-contrast group penalty 를 부여하는 첫 정식화
2. **Two-stage:** Meynet 식 Lasso-MLE 원칙을 directional mixture 로 확장, 원래 $\mathbb{S}^{d-1}$ 위 refit
3. **응용:** L2 정규화된 LLM 임베딩의 비지도 군집화에 대한 통계적 파이프라인

---

## 4. 같은 평균 다른 분산 시나리오 — 세 접근

### (A) Xie, Pan & Shen (2008) — GMM 이중 페널티

$$\mathcal{L}_p = \mathcal{L} - \lambda_1 \sum_{h,j} |\mu_{hj}| - \lambda_2 \sum_j \left[\sum_h (\sigma_{h,j}^2 - \bar{\sigma}_j^2)^2\right]^{1/2}$$

- $\mu_A = \mu_B$ 일 때 분산 정보가 두 군집 구분
- Sample space $\mathbb{R}^d$ — L2 정규화 임베딩과 불일치

### (B) vMF 이중 페널티 (보조)

$$\mathcal{L}_p^{\text{(2)}} = \mathcal{L} - \lambda_1 \sum_j \left[\sum_h w_h(\mu_{hj} - \bar{\mu}_j)^2\right]^{1/2} - \lambda_2 \sum_h \left|\kappa_h - \bar{\kappa}\right|$$

- 평균과 분산 정보를 분리된 객체로 명시 → 분해 해석 가능
- 변수 선택은 $\mu$ 페널티, concentration regularize 는 $\kappa$ 페널티
- $\mu_A = \mu_B$, $\kappa_A \neq \kappa_B$ 면 likelihood 가 core/periphery 로 군집 구분. $P^\kappa$ 는 regularize 역할이며, $\lambda_2$ 과도 시 차이 소거 위험.

### (C) vMF 단일 페널티 (제안)

$$\mathcal{L}_p^{\text{(1)}} = \mathcal{L} - \lambda \sum_j \left[\sum_h w_h(\eta_{hj} - \bar{\eta}_j)^2\right]^{1/2}, \qquad \eta_h = \kappa_h \mu_h$$

$\mu_A = \mu_B = \mu$, $\kappa_A \neq \kappa_B$ 일 때:

$$\eta_A - \eta_B = (\kappa_A - \kappa_B)\, \mu \neq 0$$

- 평균과 분산이 자연모수 $\eta_h$ 로 결합, hyperparameter 단일 $\lambda$
- Posterior log-odds $(\eta_h - \eta_\ell)^\top z_i$ 와 직접 정합
- 식별 범위: active 좌표는 $\mu_j \neq 0$ 에 한정 ($\mu_j=0$ 좌표는 log-odds 미기여 → 제외 타당)

### (D) 본 연구의 입장
이론적 정합성과 최적화 효율성을 근거로 **(C) 단일 페널티를 Main Method**로 설정하고, (B)와의 비교로 실증. (B)는 평균/분산 분해 해석용 보조 모형.

| 측면 | (B) 이중 (Sub) | (C) 단일 (Main) |
|---|---|---|
| 페널티 | $\text{group}(\mu) + L_1(\kappa)$ | $\text{group}(\eta)$ |
| Hyperparameter | $\lambda_1, \lambda_2$ | $\lambda$ |
| log-odds 정합 | 간접 | 직접 |
| 변수 선택 단위 | $\mu$(변수) + $\kappa$(regularize) | $\eta$ 단일 |

---

## 5. 모형과 수식

### 5.1 Two-stage refit

$$d_i \xrightarrow{\phi} x_i \in \mathbb{R}^d \xrightarrow{/\|\cdot\|_2} z_i \in \mathbb{S}^{d-1}$$

<img width="713" height="443" alt="image" src="https://github.com/user-attachments/assets/acaedb05-4b72-4f20-ac21-3580d20348a0" />

**Stage 1 (screening)**

- (C) Main: $\widehat{S}_\lambda^{\text{(1)}}$ — $\eta_h$ 에 penalty
- (B) Sub: $\widehat{S}_{\lambda_1}^{\text{(2)}}$ — $\mu$ 로 변수 선택, $\kappa$ 로 concentration regularize

**Stage 2 (refit)**

- $\widehat{\Theta}^{\text{refit}}_{\widehat{S}}$ — unpenalized sparse-vMF on $\mathbb{S}^{d-1}$

### 5.2 Reparametrization 및 Log-odds

$$p(z_i \mid \pi, \eta) = \sum_{h=1}^{K} \pi_h c_d(\|\eta_h\|_2) \exp(\eta_h^\top z_i), \qquad \eta_h = \kappa_h \mu_h$$

$$\log \frac{P(h \mid z_i)}{P(\ell \mid z_i)} = \log \frac{\pi_h}{\pi_\ell} + \log \frac{c_d(\kappa_h)}{c_d(\kappa_\ell)} + (\eta_h - \eta_\ell)^\top z_i$$

- **$\eta_h$ 페널티 유도:** log-odds 의 선형 판별항이 $(\eta_h - \eta_\ell)^\top z_i$ → $\eta_h$ 에 직접 페널티
- **가중치 $w_h$:** 기본값 $\pi_h$ (또는 $N_h/n$), 군집 불균형 반영. penalty-$\pi_h$ 결합을 피해 outer iteration 값 $w_h^{(t)}$ 로 고정, M-step 에서 상수 취급. $1/K$ vs $N_h/n$ sensitivity 비교.

### 5.3 Stage 1 penalty

수학적으로 좌표 $j$ 마다 컴포넌트 contrast 벡터 $(\eta_{1j} - \bar{\eta}_j, \dots, \eta_{Kj} - \bar{\eta}_j)$ 를 group 으로 보는 $\ell_{2,1}$ group penalty (cluster-contrast group LASSO). 변수를 임의로 묶는 표준 group LASSO 와 달리, **각 좌표에서 컴포넌트 간 contrast 를 group 으로 정의**한다.

**(C) 단일 (Main)**

$$P_B^{\text{(1)}}(\eta) = \sum_{j=1}^{d} \left[ \sum_h w_h(\eta_{hj} - \bar{\eta}_j)^2 \right]^{1/2}$$

$$\mathcal{L}^{\text{(1)}}_{\lambda} = \ell_n - n\lambda P_B^{\text{(1)}}(\eta)$$

$$\widehat{S}_\lambda^{\text{(1)}} = \left\lbrace j : \left[ \sum_h w_h (\widehat{\eta}_{hj} - \widehat{\bar{\eta}}_j)^2 \right]^{1/2} > \epsilon \right\rbrace$$

**(B) 이중 (Sub)**

$$P^\mu(\mu) = \sum_{j=1}^d \left[\sum_h w_h (\mu_{hj} - \bar{\mu}_j)^2\right]^{1/2}$$

$$P^\kappa(\kappa) = \sum_h \left|\kappa_h - \bar{\kappa}\right|$$

$$\mathcal{L}^{\text{(2)}}_{\lambda_1, \lambda_2} = \ell_n - n\lambda_1 P^\mu(\mu) - n\lambda_2 P^\kappa(\kappa)$$

변수 선택 ($\mu$ 페널티):

$$\widehat{S}_{\lambda_1}^{\text{(2)}} = \left\lbrace j : \left[\sum_h w_h (\widehat{\mu}_{hj} - \widehat{\bar{\mu}}_j)^2\right]^{1/2} > \epsilon \right\rbrace$$

Concentration regularize ($\kappa$ 페널티): $P^\kappa$ 는 각 군집 집중도를 전역 평균으로 regularize 한다. 평균 방향이 같은 경우 변수 선택 결과가 비더라도, vMF likelihood 가 core/periphery 구조로 두 군집을 구분한다.

### 5.4 Stage 2: Sparse-vMF refit on $\mathbb{S}^{d-1}$

$$\mu_{h,\widehat{S}^c} = 0, \quad \left\|\mu_{h,\widehat{S}}\right\|_2 = 1$$

데이터는 여전히 $\mathbb{S}^{d-1}$ 위에 있고 inactive coordinate 만 mean direction 에서 0 으로 제한하므로, 정규화 상수는 $c_d$ (not $c_{d_\lambda}$). 모든 모델이 공통 $\mathbb{S}^{d-1}$ 위에서 정의되어 likelihood 비교 가능.

$$p(z_i \mid \widetilde{\Theta}_{\widehat{S}}) = \sum_h \widetilde{\pi}_h c_d(\widetilde{\kappa}_h) \exp( \widetilde{\kappa}_h \widetilde{\mu}_{h,\widehat{S}}^\top z_{i,\widehat{S}} )$$

$$\widetilde{\tau}_{ih} = \frac{\widetilde{\pi}_h c_d(\widetilde{\kappa}_h) \exp(\widetilde{\kappa}_h \widetilde{\mu}_{h,\widehat{S}}^\top z_{i,\widehat{S}})}{\sum_\ell \widetilde{\pi}_\ell c_d(\widetilde{\kappa}_\ell) \exp(\widetilde{\kappa}_\ell \widetilde{\mu}_{\ell,\widehat{S}}^\top z_{i,\widehat{S}})}$$

$$r_{h,\widehat{S}} = \sum_i \widetilde{\tau}_{ih} z_{i,\widehat{S}}, \qquad \widetilde{N}_h = \sum_i \widetilde{\tau}_{ih}, \qquad \bar{R}_{h,\widehat{S}} = \frac{\|r_{h,\widehat{S}}\|_2}{\widetilde{N}_h}$$

$$\widehat{\widetilde{\mu}}_{h,\widehat{S}} = \frac{r_{h,\widehat{S}}}{\|r_{h,\widehat{S}}\|_2}, \quad \widehat{\widetilde{\mu}}_{h,\widehat{S}^c} = 0, \quad \widehat{\widetilde{\kappa}}_h \approx \frac{d\bar{R}_{h,\widehat{S}}\  - \bar{R}_{h,\widehat{S}}^3}{1 - \bar{R}_{h,\widehat{S}}^2}$$

Refit 은 (B), (C) 공통.

---

## 6. 선행연구와의 차이

| | Rossi & Barbaro (2022) | 본 연구 |
|---|---|---|
| Penalty 대상 | $\mu_h$ | (C) $\eta_h$ (Main) / (B) $\mu_h, \kappa_h$ |
| 최종 추정량 | penalized | unpenalized refit |
| Bias | $L_1$ 잔존 | refit 완화 |
| Refit | 없음 | $\mathbb{S}^{d-1}$ 위 |

$\mu_{hj} = \mu_{\ell j}$ 라도 $\kappa_h \neq \kappa_\ell$ 이면 $\eta_{hj} \neq \eta_{\ell j}$ ($\mu_j \neq 0$ 에 한함).

---

## 7. Active set 의 해석

$\widehat{S}_\lambda$ 는 전체 vMF density 를 가장 잘 설명하는 좌표 집합이 **아니다**. Component 간 posterior log-odds 를 구분하는 데 기여하는 좌표 집합, 즉 **clustering-oriented coordinate screening** 결과다.

- LLM 임베딩 좌표는 해석 불가능한 latent dimension → "billing", "정서" 같은 의미축으로 해석하지 않는다.
- 해석은 active 좌표가 아니라 $\widehat{\kappa}_h$, 대표 문서, cluster summary 로 수행.
- **한계 (common coordinate):** 좌표 $j$ 에서 $\eta_{1j} = \cdots = \eta_{Kj} = b_j \neq 0$ 이면 contrast 가 0 이라 active set 에서 빠진다. 이 좌표는 군집 구분엔 무관하나 full density 엔 기여할 수 있다. 본 방법은 full-density relevant selection 이 아니라 clustering-oriented screening 이므로 이는 의도된 동작이며, 실제 영향은 실험에서 측정한다.

---

## 8. 작동 가정 (개요)

- True model: $K^*$ 개 vMF component, Sparsity $|S^*| = s \ll d$
- Separation: 군집 간 $\eta$ 차이 충분, Regularity: $\kappa_h$ 유계
- $n \ll d$ 시: common $\kappa$ 또는 $\kappa_h$ cap

## 9. 검증 계획 (개요)

- **Synthetic:** (i) $\mu$ 다름, (ii) $\mu$ 같고 $\kappa$ 다름, (iii) 노이즈 변수 다수 → RQ1–3 검증
- **실제 데이터:** 표준 텍스트 벤치마크 + LLM 임베딩 (SimCSE/E5)
- **Baseline:** Rossi & Barbaro (2022), Witten & Tibshirani (2010), dense vMF + threshold, (B) vs (C)
- **지표:** ARI, NMI, F1(active set), refit 전후 비교

## 10. 범위와 한계

- vMF angular isotropic 가정 (방향별 비등방 분산 미모델링)
- 식별은 $\mu$-support 내로 한정 (4(C) 참조), common nonzero coordinate 제거 (7 참조)
- $\lambda, K$ 선택은 BIC/ICL 등 practical criterion 사용, 이론적 정당화는 후속 과제

---

## 11. 요약

1. GMM $O(Kd^2)$ → vMF scalar $\kappa_h$
2. 같은 평균 다른 분산: **(C) 단일 페널티 $\eta_h$ (Main, log-odds 직접 정합)** / (B) 이중 페널티 (Sub, 분해 해석). 군집 분리는 likelihood 가, penalty 는 regularize.
3. Screening 후 $\mathbb{S}^{d-1}$ 위 unpenalized refit. Active set 은 clustering-oriented coordinate screening.

---

## 참고문헌
- Banerjee, A., Dhillon, I. S., Ghosh, J., & Sra, S. (2005). Clustering on the unit hypersphere using von Mises-Fisher distributions. *JMLR*, 6, 1345–1382.
- Rossi, F., & Barbaro, F. (2022). Mixture of von Mises-Fisher distribution with sparse prototypes. *Neurocomputing*. arXiv:2212.14591.
- Pan, W., & Shen, X. (2007). Penalized model-based clustering with application to variable selection. *JMLR*, 8, 1145–1164.
- Wang, S., & Zhu, J. (2008). Variable selection for model-based high-dimensional clustering and its application to microarray data. *Biometrics*.
- Xie, B., Pan, W., & Shen, X. (2008). Penalized model-based clustering with cluster-specific diagonal covariance matrices and grouped variables. *EJS*, 2, 168–212.
- Witten, D. M., & Tibshirani, R. (2010). A framework for feature selection in clustering. *JASA*, 105(490), 713–726.
