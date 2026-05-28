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

### 1.3 vMF mixture
$$f(z \mid \mu, \kappa) = c_d(\kappa)\exp(\kappa\mu^\top z), \qquad z, \mu \in \mathbb{S}^{d-1}$$
$$p(z_i \mid \Theta) = \sum_{h=1}^{K} \pi_h c_d(\kappa_h) \exp(\kappa_h \mu_h^\top z_i)$$

<img width="634" height="498" alt="image" src="https://github.com/user-attachments/assets/1dbe9aa3-0291-4f65-b7ac-4de9c95105ba" />

### 1.4 채택 입장
vMF mixture를 **parsimonious directional working model** 로 채택. 정확한 분포 가정이 아님.

<img width="440" height="319" alt="image" src="https://github.com/user-attachments/assets/b7aea979-1e98-4169-afaf-28f9f4645d15" />


### 1.5 같은 평균 다른 분산 시나리오 — 세 접근

#### (A) Xie, Pan & Shen (2008) — GMM 이중 페널티
$$\mathcal{L}_p = \mathcal{L} - \lambda_1 \sum_{h,j} |\mu_{hj}| - \lambda_2 \sum_j \left[\sum_h (\sigma_{h,j}^2 - \bar{\sigma}_j^2)^2\right]^{1/2}$$
- $\mu_A = \mu_B$ 일 때 분산 페널티 ($\lambda_2$) 가 두 군집 구분.
- Sample space $\mathbb{R}^d$ — L2 정규화 임베딩과 불일치.

#### (B) vMF 이중 페널티 (제안)
$$\mathcal{L}_p^{\text{(2)}} = \mathcal{L} - \lambda_1 \sum_j \left[\sum_h w_h(\mu_{hj} - \bar{\mu}_j)^2\right]^{1/2} - \lambda_2 \sum_h |\kappa_h - \bar{\kappa}|$$
- 평균 정보와 분산 정보를 **분리된 객체로 명시** → 분해 해석 가능
- Hyperparameter $\lambda_1, \lambda_2$ 동시 튜닝
- $\mu_h$ (단위 벡터) 와 $\kappa_h$ (양의 실수) 의 스케일 분리

#### (C) vMF 단일 페널티 (제안)
$$\mathcal{L}_p^{\text{(1)}} = \mathcal{L} - \lambda \sum_j \left[\sum_h w_h(\eta_{hj} - \bar{\eta}_j)^2\right]^{1/2}, \qquad \eta_h = \kappa_h \mu_h$$
$\mu_A = \mu_B = \mu$, $\kappa_A \neq \kappa_B$ 일 때:
$$\eta_A - \eta_B = (\kappa_A - \kappa_B)\, \mu \neq 0$$
- 평균과 분산이 자연모수 $\eta_h$ 로 **결합**
- Hyperparameter 단일 $\lambda$
- Posterior log-odds $(\eta_h - \eta_\ell)^\top z_i$ 와 직접 정합

#### (D) 본 연구의 입장 — (C) 단일 페널티를 Main Method로 채택
본 연구는 이론적 정합성과 최적화 효율성을 고려하여 **(C) 단일 페널티를 Main Method**로 전개하며, (B)는 active set 분해를 위한 해석용(post-hoc) 보조 모형으로 활용합니다.

| 측면 | (B) 이중 (Sub) | (C) 단일 (Main) |
|---|---|---|
| 페널티 | $\text{group}(\mu) + L_1(\kappa)$ | $\text{group}(\eta)$ |
| Hyperparameter | $\lambda_1, \lambda_2$ | $\lambda$ |
| 평균·분산 | 분리 | $\eta_h$ 결합 |
| log-odds 정합 | 간접 | 직접 |
| 해석 | active set 분해 | post-hoc |

---

## 2. 연구 아이디어: Two-stage refit

$$d_i \xrightarrow{\phi} x_i \in \mathbb{R}^d \xrightarrow{/\|\cdot\|_2} z_i \in \mathbb{S}^{d-1}$$

- **Stage 1 (screening):** 두 페널티 형태
  - (C) 단일: $\widehat{S}_\lambda^{\text{(1)}} \leftarrow$ penalty on $\eta_h$ (Main)
  - (B) 이중: $\widehat{S}_{\lambda_1, \lambda_2}^{\text{(2)}} \leftarrow$ penalty on $\mu_h, \kappa_h$
- **Stage 2 (refit):** $\widehat{\Theta}^{\text{refit}}_{\widehat{S}} \leftarrow$ unpenalized sparse-vMF on $\mathbb{S}^{d-1}$

---

## 3. 모형과 수식

### 3.1 Reparametrization 및 사후 확률의 Log-odds
$$p(z_i \mid \pi, \eta) = \sum_{h=1}^{K} \pi_h c_d(\|\eta_h\|_2) \exp(\eta_h^\top z_i), \qquad \eta_h = \kappa_h \mu_h$$

**사후 확률의 Log-odds:**
$$\log \frac{P(h \mid z_i)}{P(\ell \mid z_i)} = \log \frac{\pi_h}{\pi_\ell} + \log \frac{c_d(\kappa_h)}{c_d(\kappa_\ell)} + (\eta_h - \eta_\ell)^\top z_i$$

- **$\eta_h$ 페널티 유도:** 두 군집을 구분하는 실질적 판별항(decision boundary)은 $\mu_h$가 아닌 자연모수의 차이 $(\eta_h - \eta_\ell)$. 따라서 $\eta_h$에 직접 페널티를 부여하여 분산을 축소(shrinkage)하는 것이 타당함.
- **가중치 $w_h$ 의 타당성 ($w_h = \pi_h$):** $$\bar{\eta}_j = \sum_{h=1}^K w_h \eta_{hj}$$
  군집 크기 불균형을 반영해 전역 평균 $\bar{\eta}_j$를 설정함으로써, (1) 소수 군집의 미세한 노이즈로 인한 변수 선택 오탐지(False Positive)를 방지하고, (2) 로그우도(Log-likelihood) 함수와의 스케일 정합성을 유지함.

### 3.2 Stage 1: 두 형태의 cluster-contrast penalty

#### (C) 단일 페널티 (Main Method)
$$P_B^{\text{(1)}}(\eta) = \sum_{j=1}^{d} \left[ \sum_h w_h(\eta_{hj} - \bar{\eta}_j)^2 \right]^{1/2}, \qquad \bar{\eta}_j = \sum_h w_h \eta_{hj}$$
$$\mathcal{L}^{\text{(1)}}_{\lambda} = \ell_n(\pi, \eta) - n\lambda P_B^{\text{(1)}}(\eta)$$
$$\widehat{S}_\lambda^{\text{(1)}} = \left\lbrace j : \left[ \sum_h w_h (\widehat{\eta}_{hj} - \widehat{\bar{\eta}}_j)^2 \right]^{1/2} > \epsilon \right\rbrace$$

#### (B) 이중 페널티 (Sub Method)
$$P^\mu(\mu) = \sum_{j=1}^d \left[\sum_h w_h (\mu_{hj} - \bar{\mu}_j)^2\right]^{1/2}, \qquad P^\kappa(\kappa) = \sum_h |\kappa_h - \bar{\kappa}|$$
$$\mathcal{L}^{\text{(2)}}_{\lambda_1, \lambda_2} = \ell_n(\pi, \mu, \kappa) - n\lambda_1 P^\mu(\mu) - n\lambda_2 P^\kappa(\kappa)$$
$$\widehat{S}_{\lambda_1, \lambda_2}^{\text{(2)}} = \left\lbrace j : \left[\sum_h w_h (\widehat{\mu}_{hj} - \widehat{\bar{\mu}}_j)^2\right]^{1/2} > \epsilon \right\rbrace$$

### 3.3 Stage 2: Sparse-vMF refit on $\mathbb{S}^{d-1}$
$$\mu_{h,\widehat{S}^c} = 0, \quad \left\|\mu_{h,\widehat{S}}\right\|_2 = 1$$

정규화 상수는 $c_d$ ($c_{d_\lambda}$ 아님).

$$p(z_i \mid \widetilde{\Theta}_{\widehat{S}}) = \sum_h \widetilde{\pi}_h c_d(\widetilde{\kappa}_h) \exp\left( \widetilde{\kappa}_h \widetilde{\mu}_{h,\widehat{S}}^\top z_{i,\widehat{S}} \right)$$

EM update:

$$\widetilde{\tau}_{ih} = \frac{\widetilde{\pi}_h c_d(\widetilde{\kappa}_h) \exp(\widetilde{\kappa}_h \widetilde{\mu}_{h,\widehat{S}}^\top z_{i,\widehat{S}})}{\sum_\ell \widetilde{\pi}_\ell c_d(\widetilde{\kappa}_\ell) \exp(\widetilde{\kappa}_\ell \widetilde{\mu}_{\ell,\widehat{S}}^\top z_{i,\widehat{S}})}$$

$$r_{h,\widehat{S}} = \sum_i \widetilde{\tau}_{ih} z_{i,\widehat{S}}, \qquad N_h = \sum_i \widetilde{\tau}_{ih}$$

$$\widehat{\widetilde{\mu}}_{h,\widehat{S}} = \frac{r_{h,\widehat{S}}}{\|r_{h,\widehat{S}}\|_2}, \quad \widehat{\widetilde{\mu}}_{h,\widehat{S}^c} = 0, \quad \widehat{\widetilde{\kappa}}_h \approx \frac{\bar{R}d - \bar{R}^3}{1 - \bar{R}^2}, \quad \bar{R} = \frac{\|r_{h,\widehat{S}}\|_2}{N_h}$$

Refit 은 (B), (C) 공통.

---

## 4. 선행연구와의 차이

| | Rossi & Barbaro (2022) | 본 연구 |
|---|---|---|
| Penalty 대상 | $\mu_h$ | (C) $\eta_h$ (Main) / (B) $\mu_h, \kappa_h$ |
| 최종 추정량 | penalized | unpenalized refit |
| Bias | $L_1$ 잔존 | refit 완화 |
| Refit | 없음 | $\mathbb{S}^{d-1}$ 위 |

$\mu_{hj} = \mu_{\ell j}$ 라도 $\kappa_h \neq \kappa_\ell$ 이면 $\eta_{hj} \neq \eta_{\ell j}$.

---

## 5. 요약

1. GMM $O(Kd^2)$ → vMF scalar $\kappa_h$.
2. 같은 평균 다른 분산 시나리오의 두 vMF 형태:
   - **(C) 단일 페널티 (Main):** $\eta_h = \kappa_h \mu_h$ 결합 페널티. 사후 확률의 Log-odds 판별항과 직접적으로 정합.
   - **(B) 이중 페널티 (Sub):** $\mu_h, \kappa_h$ 각각 분리 페널티 (해석 보조용).
3. Screening 후 $\mathbb{S}^{d-1}$ 위에서 unpenalized refit.

---

## 6. 미팅 논의 사항

### 6.1 검토 요청
**(1) 문제 설정:** $z_i \in \mathbb{S}^{d-1}$ 위 directional clustering.
**(2) 기본 모형:** vMF mixture. $\mu_h$ 같고 $\kappa_h$ 다른 시나리오 $\eta_h$ 차이로 식별. 한계: angular isotropic.
**(3) 핵심 방법론: (C) 단일 페널티를 Main Method로 채택**
- (C) 단일: $\eta_h$ 에 group penalty (사후확률 Log-odds 기반)
- (B) 이중: $\mu_h$ group penalty + $\kappa_h$ 별도 페널티 (해석 보조용)
- Stage 2 refit 공통

**(4) Novelty**

| 구분 | 기존 | 본 연구 |
|---|---|---|
| Sparse vMF | Rossi & Barbaro (2022) | (C) Main / (B) Sub cluster-contrast |
| Cluster-contrast | Pan-Shen, Xie (GMM) | vMF 확장 |
| Two-stage Lasso-MLE | Meynet | Directional |
| 같은 평균 시나리오 | Xie (GMM) | (B), (C) 정식화 |

### 6.2 검토 요청 사항
- vMF angular isotropic 가정의 적합도 진단
- $\kappa_h$ component-specific vs common
- (B), (C) 식별성 및 weight $w_h$ 선택
- **(C) 단일 페널티를 Main Method로 확정 (최적화 난이도 및 이론적 정합성 고려)** 및 (B)의 활용 방안 (보조 지표)
- 비교 baseline: Rossi & Barbaro (2022), Witten & Tibshirani (2010), dense vMF + threshold

---

## 참고문헌
- Banerjee, A., Dhillon, I. S., Ghosh, J., & Sra, S. (2005). Clustering on the unit hypersphere using von Mises-Fisher distributions. *JMLR*, 6, 1345–1382.
- Rossi, F., & Barbaro, F. (2022). Mixture of von Mises-Fisher distribution with sparse prototypes. *Neurocomputing*. arXiv:2212.14591.
- Pan, W., & Shen, X. (2007). Penalized model-based clustering with application to variable selection. *JMLR*, 8, 1145–1164.
- Wang, S., & Zhu, J. (2008). Variable selection for model-based high-dimensional clustering and its application to microarray data. *Biometrics*.
- Xie, B., Pan, W., & Shen, X. (2008). Penalized model-based clustering with cluster-specific diagonal covariance matrices and grouped variables. *EJS*, 2, 168–212.
- Witten, D. M., & Tibshirani, R. (2010). A framework for feature selection in clustering. *JASA*, 105(490), 713–726.
