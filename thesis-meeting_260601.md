# 연구 진행 정리

## 1. 배경

### 1.1 문제 설정과 선행 연구

비지도 상황에서 텍스트 임베딩 $x_i = \phi(d_i) \in \mathbb{R}^d$ 의 군집화. 고차원 과제:
(i) 군집별 분산 추정
(ii) 변수 선택

본 연구는 다음 네 흐름에 기반한다.

- **Directional clustering** (Banerjee et al., 2005)
  $$f(z \mid \mu, \kappa) = c_d(\kappa)\exp(\kappa \mu^\top z), \quad z, \mu \in \mathbb{S}^{d-1}$$
  $\mathbb{S}^{d-1}$ 위 directional data에 대한 vMF mixture와 EM 추정.

- **Sparse vMF mixture** (Rossi & Barbaro, 2022)
  $$\mathcal{L}_p(\Theta) = \mathcal{L}(\Theta) - \beta \sum_h \|\mu_h\|_1$$
  본 연구의 직접 출발점. 한계: penalized estimator를 최종 모형으로 사용 ($L_1$ bias 잔존), penalty 대상이 $\mu_h$ 이므로 판별항 $\kappa_h \mu_h$ 구조 미반영.

- **Cluster-contrast 변수 선택** (Pan & Shen, 2007; Wang & Zhu, 2008; Xie et al., 2008)
  $$\mu_{hj} = \mu_{\ell j} \ \forall h, \ell \iff j \text{ 는 noise variable}$$
  GMM에서 발전한 통찰. 본 연구는 이를 directional 자연모수 $\eta_h = \kappa_h \mu_h$ 에 적용한다.

- **Two-stage Lasso-MLE** (Meynet 계열)
  $$\widehat{S}_\lambda \xrightarrow{\text{screening}} \widehat{\Theta}^{\text{refit}}_{\widehat{S}} \xrightarrow{\text{unpenalized MLE}} \text{final}$$
  Penalized estimator는 screening 단계로만, 선택된 support에서 unpenalized refit.

---

### 1.2 같은 평균 다른 분산 시나리오 — 세 가지 접근

$\mu_A = \mu_B$ 이지만 퍼짐이 다른 두 군집의 변수선택. 세 접근의 진화:

#### (A) GMM 측 — 분산에 별도 페널티

**Xie, Pan & Shen (2008)** — 평균 + 분산 이중 페널티:

$$\mathcal{L}_p = \mathcal{L} - \lambda_1 \sum_{h,j} |\mu_{hj}| - \lambda_2 \sum_j \left[\sum_h (\sigma_{h,j}^2 - \bar{\sigma}_j^2)^2\right]^{1/2}$$

$\mu_A = \mu_B$ 일 때 분산 페널티 ($\lambda_2$) 가 두 군집 구분. Sample space $\mathbb{R}^d$ — L2 정규화 임베딩과 불일치.

#### (B) vMF 측 — Xie 식 그대로 확장: 이중 페널티 (제안)

본 연구에서 함께 검토하는 형태:

$$\mathcal{L}_p^{\text{(2)}} = \mathcal{L} - \lambda_1 \sum_j \left[\sum_h w_h(\mu_{hj} - \bar{\mu}_j)^2\right]^{1/2} - \lambda_2 \sum_h |\kappa_h - \bar{\kappa}|$$

$\mu_A = \mu_B$ 일 때 $\kappa$ 페널티가 두 군집 구분. Sample space는 $\mathbb{S}^{d-1}$ 로 정합.

**특징:**
- 평균 정보와 분산 정보를 **분리된 객체로 명시** → 분해 해석 가능
- Hyperparameter $\lambda_1, \lambda_2$ 동시 튜닝
- $\mu_h$ (단위 벡터) 와 $\kappa_h$ (양의 실수) 의 스케일 분리

#### (C) vMF 측 — 자연모수 $\eta_h = \kappa_h \mu_h$: 단일 페널티 (제안)

$$\mathcal{L}_p^{\text{(1)}} = \mathcal{L} - \lambda \sum_j \left[\sum_h w_h(\eta_{hj} - \bar{\eta}_j)^2\right]^{1/2}$$

$\mu_A = \mu_B = \mu$, $\kappa_A \neq \kappa_B$ 일 때:

$$\eta_A - \eta_B = (\kappa_A - \kappa_B)\, \mu \neq 0$$

$\mu_j \neq 0$ 인 좌표에서 자연모수 차이 자동 보존.

**특징:**
- 평균과 분산이 자연모수 $\eta_h$ 로 **결합**
- Hyperparameter 단일 $\lambda$
- Posterior log-odds $(\eta_h - \eta_\ell)^\top z_i$ 와 직접 정합

#### (D) 본 연구의 입장 — (B) 와 (C) 를 함께 시도

| 구분 | (B) 이중 페널티 | (C) 단일 페널티 |
|---|---|---|
| 페널티 형태 | $\text{group}(\mu) + L_1(\kappa)$ | $\text{group}(\eta = \kappa\mu)$ |
| Hyperparameter | $\lambda_1, \lambda_2$ | $\lambda$ |
| 평균·분산 정보 | 분리 객체 | $\eta_h$ 로 결합 |
| Posterior log-odds 정합 | 간접 | 직접 |
| Group LASSO 구조 | 분리 ($\mu$ vs $\kappa$) | 통일 ($\eta$) |
| 스케일 | 분리 (단위벡터 vs 스칼라) | 자연 통합 |
| 해석가능성 | 평균/분산 active set 분해 | $\widehat{\mu}_h, \widehat{\kappa}_h$ post-hoc |
| 같은 평균 해결 | ✓ ($\kappa$ 페널티) | ✓ (단일 페널티 자동) |

본 연구는 (B) 와 (C) 두 형태를 모두 정식화하여 비교한다. 같은 평균 시나리오, 같은 분산 시나리오, 혼합 시나리오에서 각각의 작동을 검증하고, 어느 페널티 구조가 directional sparse clustering 에 더 적합한지 실증한다.

---

### 1.3 GMM의 한계와 vMF로의 전환

L2 정규화된 임베딩 $z_i \in \mathbb{S}^{d-1}$ 의 군집화에서 GMM 적용을 검토한다.

- **GMM full / diagonal:** 모수 수 $O(d^2)$, $O(d)$ — 고차원 추정 불안정.
- **GMM spherical** ($\Sigma_h = \sigma_h^2 I$): cluster당 1개로 줄지만 sample space가 $\mathbb{R}^d$ → 단위 norm 제약 $\|z_i\| = 1$ 과 불일치.

| **모형**         | Sample space       | **cluster당 퍼짐 모수**    | **d=768 예시**            |
| -------------- | ------------------ | --------------------- | ----------------------- |
| GMM full cov.  | $\mathbb{R}^{d}$   | $d(d+1)/2$            | $\approx 3 \times 10^5$ |
| GMM diag. cov. | $\mathbb{R}^{d}$   | $d$                   | 768                     |
| GMM spherical  | $\mathbb{R}^{d}$   | 1                     | 1                       |
| vMF            | $\mathbb{S}^{d-1}$ | 1 (scalar $\kappa_h$) | 1                       |

vMF mixture 는 **코사인 기하학과의 정합성**과 **고차원 모수 절약**을 동시에 만족한다.

$$\mu_h \text{ 비슷}, \ \Sigma_h \text{ 다름} \ (\text{GMM}) \quad \longleftrightarrow \quad \mu_h \text{ 비슷}, \ \kappa_h \text{ 다름} \ (\text{vMF})$$

<img width="634" height="498" alt="image" src="https://github.com/user-attachments/assets/1dbe9aa3-0291-4f65-b7ac-4de9c95105ba" />

vMF는 angular spread isotropic 가정의 한계가 있으나, L2 정규화된 임베딩에서는 구면 기하학과의 정합성 이득이 더 크다. $n \ll d$ 인 경우 common $\kappa$ 또는 $\kappa_h$ cap 을 함께 검토.

### 1.4 vMF mixture

$$f(z \mid \mu, \kappa) = c_d(\kappa)\exp(\kappa\mu^\top z), \qquad z, \mu \in \mathbb{S}^{d-1},\ \kappa \geq 0$$

$$p(z_i \mid \Theta) = \sum_{h=1}^{K} \pi_h c_d(\kappa_h) \exp(\kappa_h \mu_h^\top z_i)$$

Banerjee et al. (2005) 은 vMF mixture가 spherical k-means의 확률모형적 일반화임을 보였다.

### 1.5 vMF mixture 채택의 입장

본 연구는 $z_i$ 가 vMF mixture를 **정확히** 따른다고 가정하지 않는다. **Parsimonious directional working model** 로 채택하는 세 근거:

(i) **기하학적 부합** — Cosine similarity 기반 임베딩에서 의미 정보는 방향에 담김.
(ii) **모수 절약** — GMM $O(Kd^2)$ → scalar $\kappa_h$.
(iii) **선행 연구 표준성** — Banerjee et al. (2005), Reisinger et al. (2010), Davidson et al. (2018), Rossi & Barbaro (2022).

모형의 경험적 적합도는 실험 단계에서 검증한다.

<img width="626" height="380" alt="image" src="https://github.com/user-attachments/assets/d2713c3a-7dc7-4392-8074-b6bf49a96eaa" />


## 2. 연구 아이디어: Two-stage refit

$$d_i \xrightarrow{\phi} x_i \in \mathbb{R}^d \xrightarrow{/\|\cdot\|_2} z_i \in \mathbb{S}^{d-1}$$

- Stage 1 (screening): 두 페널티 형태를 모두 검토
  - **단일 페널티 (C):** $\widehat{S}_\lambda^{\text{(1)}} \leftarrow L_1$-penalized vMF mixture on $\eta_h = \kappa_h \mu_h$
  - **이중 페널티 (B):** $\widehat{S}_{\lambda_1, \lambda_2}^{\text{(2)}} \leftarrow$ separately penalized $\mu_h, \kappa_h$
- Stage 2 (refit): $\widehat{\Theta}^{\text{refit}}_{\widehat{S}} \leftarrow$ unpenalized sparse-vMF on $\mathbb{S}^{d-1}$

## 3. 선행연구와의 차이

| **구분** | **Rossi & Barbaro (2022)** | **본 연구** |
|---|---|---|
| Penalty 대상 | $\mu_h$ 만 | (B) $\mu_h, \kappa_h$ 각각 / (C) $\eta_h = \kappa_h \mu_h$ |
| Sparsity 의미 | cluster별 prototype | cluster 간 판별 좌표 |
| 최종 추정량 | penalized | unpenalized refit |
| Bias | $L_1$ shrinkage 잔존 | refit으로 완화 |
| Refit 단계 | 없음 | 선택된 support에서 원래 $\mathbb{S}^{d-1}$ 위 |

Penalty 대상을 자연모수 (또는 $\mu, \kappa$ 각각) 로 두는 근거는 posterior log-ratio가 $\eta_h$ 에 선형으로 의존하기 때문이다.

$$\log \frac{P(h \mid z_i)}{P(\ell \mid z_i)} = \log \frac{\pi_h}{\pi_\ell} + \log \frac{c_d(\kappa_h)}{c_d(\kappa_\ell)} + (\eta_h - \eta_\ell)^\top z_i$$

$\mu_{hj} = \mu_{\ell j}$ 라도 $\kappa_h \neq \kappa_\ell$ 이면 $\eta_{hj} \neq \eta_{\ell j}$ 이므로 판별 정보가 보존된다.

## 4. 모형과 수식

### 4.1 Reparametrization

$\eta_h = \kappa_h \mu_h$ 로 두면

$$p(z_i \mid \pi, \eta) = \sum_{h=1}^{K} \pi_h c_d(\|\eta_h\|_2) \exp(\eta_h^\top z_i)$$

### 4.2 Stage 1: 두 형태의 cluster-contrast penalty

#### (C) 단일 페널티 — 자연모수 $\eta_h$

$$\bar{\eta}_j = \sum_h w_h \eta_{hj}, \qquad P_B^{\text{(1)}}(\eta) = \sum_{j=1}^{d} \left[ \sum_h w_h(\eta_{hj} - \bar{\eta}_j)^2 \right]^{1/2}$$

$$\mathcal{L}^{\text{(1)}}_{\lambda}(\pi, \eta) = \ell_n(\pi, \eta) - n\lambda P_B^{\text{(1)}}(\eta)$$

$$\widehat{S}_\lambda^{\text{(1)}} = \left\lbrace j : \left[ \sum_h w_h (\widehat{\eta}_{hj} - \widehat{\bar{\eta}}_j)^2 \right]^{1/2} > \epsilon \right\rbrace$$

#### (B) 이중 페널티 — $\mu_h, \kappa_h$ 각각

$$P^\mu(\mu) = \sum_{j=1}^d \left[\sum_h w_h (\mu_{hj} - \bar{\mu}_j)^2\right]^{1/2}, \qquad P^\kappa(\kappa) = \sum_h |\kappa_h - \bar{\kappa}|$$

$$\mathcal{L}^{\text{(2)}}_{\lambda_1, \lambda_2}(\pi, \mu, \kappa) = \ell_n(\pi, \mu, \kappa) - n\lambda_1 P^\mu(\mu) - n\lambda_2 P^\kappa(\kappa)$$

$$\widehat{S}_{\lambda_1, \lambda_2}^{\text{(2)}} = \left\lbrace j : \left[\sum_h w_h (\widehat{\mu}_{hj} - \widehat{\bar{\mu}}_j)^2\right]^{1/2} > \epsilon \right\rbrace$$

(B) 의 경우 $\kappa$ 페널티는 좌표가 아닌 군집 단위 → active set 은 $\mu$ 페널티로 정의, $\kappa$ 정보는 cluster 간 spread 차이 식별에 사용.

두 형태 모두 "cluster 공통으로 큰 좌표는 active set 제외" 라는 동일한 원리를 따르되, 구현 단위가 다르다.

### 4.3 Stage 2: Sparse-vMF refit on $\mathbb{S}^{d-1}$

제약: $\mu_{h,\widehat{S}^c} = 0,\ \|\mu_{h,\widehat{S}}\|_2 = 1$. Density를 원래 $\mathbb{S}^{d-1}$ 위에서 정의하므로 정규화 상수는 $c_d$ (not $c_{d_\lambda}$).

$$p(z_i \mid \widetilde{\Theta}_{\widehat{S}}) = \sum_h \widetilde{\pi}_h c_d(\widetilde{\kappa}_h) \exp\left( \widetilde{\kappa}_h \widetilde{\mu}_{h,\widehat{S}}^\top z_{i,\widehat{S}} \right)$$

EM update:

$$\widetilde{\tau}_{ih} = \frac{\widetilde{\pi}_h c_d(\widetilde{\kappa}_h) \exp(\widetilde{\kappa}_h \widetilde{\mu}_{h,\widehat{S}}^\top z_{i,\widehat{S}})}{\sum_\ell \widetilde{\pi}_\ell c_d(\widetilde{\kappa}_\ell) \exp(\widetilde{\kappa}_\ell \widetilde{\mu}_{\ell,\widehat{S}}^\top z_{i,\widehat{S}})}$$

$$r_{h,\widehat{S}} = \sum_i \widetilde{\tau}_{ih} z_{i,\widehat{S}}, \qquad N_h = \sum_i \widetilde{\tau}_{ih}$$

$$\widehat{\widetilde{\mu}}_{h,\widehat{S}} = \frac{r_{h,\widehat{S}}}{\|r_{h,\widehat{S}}\|_2}, \qquad \widehat{\widetilde{\mu}}_{h,\widehat{S}^c} = 0, \qquad A_d(\widehat{\widetilde{\kappa}}_h) = \frac{\|r_{h,\widehat{S}}\|_2}{N_h}$$

Banerjee 근사:

$$\widehat{\widetilde{\kappa}}_h \approx \frac{\bar{R}d - \bar{R}^3}{1 - \bar{R}^2}, \qquad \bar{R} = \frac{\|r_{h,\widehat{S}}\|_2}{N_h}$$

Refit 단계는 두 형태 (B), (C) 모두 공통.

## 5. 요약

1. GMM 공분산 모수 폭발을 vMF의 scalar $\kappa_h$ 로 우회.
2. **같은 평균 다른 분산 시나리오의 두 vMF 접근:**
   - (B) 이중 페널티: $\mu_h, \kappa_h$ 각각에 페널티
   - (C) 단일 페널티: 자연모수 $\eta_h = \kappa_h \mu_h$ 에 페널티
   - 두 형태를 모두 정식화하고 비교
3. Penalty 대상은 모두 log-density 선형 판별항과 연결.
4. Penalized estimator는 screening 단계로만, 원래 $\mathbb{S}^{d-1}$ 위에서 unpenalized refit.
5. 해석은 active 좌표가 아니라 $\widehat{\kappa}_h$ 와 대표 문서, cluster summary로.

## 6. 미팅 논의 사항

### 6.1 현 단계 프레임워크 및 기본 모형에 대한 검토 요청

#### (1) 문제 설정 — L2 정규화된 텍스트 임베딩의 directional clustering

- LLM 임베딩 $x_i = \phi(d_i) \in \mathbb{R}^d$ 를 L2 정규화하여 $z_i \in \mathbb{S}^{d-1}$ 위 directional data로.
- Cosine similarity 기반 임베딩의 기하학적 구조와 부합.
- 비지도 군집화 가정.

#### (2) 기본 확률 모형 — vMF mixture

- GMM full covariance의 $O(Kd^2)$ 모수 폭발 회피.
- Cluster별 angular concentration을 scalar $\kappa_h$ 로.
- $\mu_h$ 같고 $\kappa_h$ 다른 시나리오 자연모수 $\eta_h$ 차이로 식별 가능.
- 한계: angular spread isotropic 가정.

#### (3) 핵심 방법론 — Two-stage sparse vMF mixture (두 페널티 형태 검토)

본 연구는 같은 평균 다른 분산 시나리오를 다루는 두 페널티 형태를 모두 정식화한다.

- **(B) 이중 페널티:** $\mu_h$ 에 group penalty, $\kappa_h$ 에 별도 페널티
  - 평균 정보와 분산 정보를 분리된 객체로 명시
  - 평균/분산 active set 분해 해석 가능

- **(C) 단일 페널티:** 자연모수 $\eta_h = \kappa_h \mu_h$ 에 group penalty
  - 평균과 분산이 자연모수로 결합
  - Posterior log-odds $(\eta_h - \eta_\ell)^\top z_i$ 와 직접 정합

- Stage 2 refit (선택된 support 에서 원래 $\mathbb{S}^{d-1}$ 위 unpenalized) 은 두 형태 공통.
- Meynet식 Lasso-MLE 원칙을 directional mixture에 적용.

#### (4) Novelty 위치 설정

| 구분 | 기존 연구 | 본 연구 |
|---|---|---|
| Sparse vMF mixture | Rossi & Barbaro (2022): $L_1$ on $\mu_h$ | (B) 이중 페널티 / (C) 단일 페널티 cluster-contrast |
| Cluster-contrast 변수선택 | Pan & Shen (2007), Xie et al. (2008): Euclidean GMM | Directional setting (vMF) 으로 확장 |
| Two-stage Lasso-MLE | Meynet 계열: regression / GMM | Sparse directional mixture에 적용 |
| 같은 평균 시나리오 | Xie et al. (2008) GMM | vMF 의 두 페널티 형태 정식화 및 비교 |

### 6.2 검토 요청 사항

- vMF의 angular isotropic 가정이 LLM 임베딩 자료에 적합한지 진단 방법
- $\kappa_h$ component-specific vs common 결정 기준 ($n \ll d$ 포함)
- 두 페널티 형태 (B), (C) 의 식별성 및 weight $w_h$ 선택
- 두 형태 중 어느 쪽을 main 으로 설정할지에 대한 의견
- 비교 baseline: Rossi & Barbaro (2022), Witten & Tibshirani (2010), dense vMF + threshold 후 refit

## 참고문헌

- Banerjee, A., Dhillon, I. S., Ghosh, J., & Sra, S. (2005). Clustering on the unit hypersphere using von Mises-Fisher distributions. *Journal of Machine Learning Research*, 6, 1345–1382.
- Rossi, F., & Barbaro, F. (2022). Mixture of von Mises-Fisher distribution with sparse prototypes. *Neurocomputing*. arXiv:2212.14591.
- Pan, W., & Shen, X. (2007). Penalized model-based clustering with application to variable selection. *Journal of Machine Learning Research*, 8, 1145–1164.
- Wang, S., & Zhu, J. (2008). Variable selection for model-based high-dimensional clustering and its application to microarray data. *Biometrics*.
- Xie, B., Pan, W., & Shen, X. (2008). Penalized model-based clustering with cluster-specific diagonal covariance matrices and grouped variables. *Electronic Journal of Statistics*, 2, 168–212.
- Zhou, H., Pan, W., & Shen, X. (2009). Penalized model-based clustering with unconstrained covariance matrices. *Electronic Journal of Statistics*, 3, 1473–1496.
- Witten, D. M., & Tibshirani, R. (2010). A framework for feature selection in clustering. *Journal of the American Statistical Association*, 105(490), 713–726.
