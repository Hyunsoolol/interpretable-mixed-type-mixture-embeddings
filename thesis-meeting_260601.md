# 연구 진행 정리

## 1. 배경

### 1.1 문제 설정과 선행 연구

비지도 상황에서 텍스트 임베딩 $x_i = \phi(d_i) \in \mathbb{R}^d$ 의 군집화. 고차원 과제
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

### 1.2 GMM의 한계와 vMF로의 전환

L2 정규화된 임베딩 $z_i \in \mathbb{S}^{d-1}$ 의 군집화에서 GMM 적용을 검토한다.

- **GMM full / diagonal:** 모수 수가 각각 $O(d^2)$, $O(d)$ 로 고차원에서 추정 불안정.
- **GMM spherical** ($\Sigma_h = \sigma_h^2 I$): 모수는 cluster당 1개로 줄지만, sample space가 여전히 $\mathbb{R}^d$ 이므로 단위 norm 제약 $\|z_i\| = 1$ 과 불일치.

vMF mixture 는 sample space 자체가 $\mathbb{S}^{d-1}$ 이고 $\kappa_h$ 가 angular spread 를 직접 표현한다. 본 연구는 **코사인 기하학과의 정합성**과 **고차원 모수 절약**을 동시에 만족한다는 점에서 vMF를 채택한다.

| **모형**         | Sample space       | **cluster당 퍼짐 모수**    | **d=768 예시**            |
| -------------- | ------------------ | --------------------- | ----------------------- |
| GMM full cov.  | $\mathbb{R}^{d}$   | $d(d+1)/2$            | $\approx 3 \times 10^5$ |
| GMM diag. cov. | $\mathbb{R}^{d}$   | $d$                   | 768                     |
| GMM spherical  | $\mathbb{R}^{d}$   | 1                     | 1                       |
| vMF            | $\mathbb{S}^{d-1}$ | 1 (scalar $\kappa_h$) | 1                       |

vMF setting에서 평균이 비슷하지만 퍼짐이 다른 군집은 다음과 같이 표현된다.

$$\mu_h \text{ 비슷}, \ \Sigma_h \text{ 다름} \ (\text{GMM}) \quad \longleftrightarrow \quad \mu_h \text{ 비슷}, \ \kappa_h \text{ 다름} \ (\text{vMF})$$

<img width="634" height="498" alt="image" src="https://github.com/user-attachments/assets/1dbe9aa3-0291-4f65-b7ac-4de9c95105ba" />

단, vMF는 평균 방향 주변의 angular spread를 isotropic으로 가정하므로 좌표별 또는 방향별로 비등방인 분산 구조는 모델링하지 못한다. L2 정규화된 임베딩에서는 이 한계보다 구면 기하학과의 정합성에서 얻는 이득이 크다고 판단한다. 또한 component-specific $\kappa_h$ 가 너무 자유로우면 소수 관측치에 과적합할 수 있으므로, $n \ll d$ 인 경우에는 common $\kappa$ 또는 $\kappa_h$ 에 cap을 두는 방안을 함께 검토한다.

" $\mu$가 같아도 $\kappa$가 다르면 $\eta = \kappa\mu$는 다르다. 그래서 두 군집을 구분할 수 있고, 그래서 페널티도 $\eta$에 걸어야 한다. "

### 1.3 vMF mixture

$$f(z \mid \mu, \kappa) = c_d(\kappa)\exp(\kappa\mu^\top z), \qquad z, \mu \in \mathbb{S}^{d-1},\ \kappa \geq 0$$

$$p(z_i \mid \Theta) = \sum_{h=1}^{K} \pi_h c_d(\kappa_h) \exp(\kappa_h \mu_h^\top z_i)$$

Banerjee et al. (2005) 은 vMF mixture가 spherical k-means의 확률모형적 일반화임을 보였다.

### 1.4 vMF mixture 채택의 입장

본 연구는 L2 정규화된 임베딩 $z_i \in \mathbb{S}^{d-1}$ 이 vMF mixture 
분포를 정확히 따른다고 가정하지 않는다. 대신, 다음 세 가지 근거에 기반하여 
vMF mixture를 **parsimonious directional working model** 로 채택한다.

(i) **기하학적 부합:** Cosine similarity 기반 임베딩에서 의미 정보는 
    방향에 담기므로, $\mathbb{S}^{d-1}$ 위 directional model이 자연스럽다.

(ii) **모수 절약:** GMM full covariance의 $O(Kd^2)$ 모수를 cluster당 
     scalar $\kappa_h$ 로 우회한다.

(iii) **선행 연구의 표준성:** Banerjee et al. (2005) 이후 vMF mixture는 
      고차원 텍스트 directional data 군집화의 표준 도구로 자리잡아왔다 
      (Reisinger et al., 2010; Davidson et al., 2018; Rossi & Barbaro, 2022).

모형의 경험적 적합도는 실험 단계에서 검증한다.

<img width="626" height="380" alt="image" src="https://github.com/user-attachments/assets/d2713c3a-7dc7-4392-8074-b6bf49a96eaa" />


## 2. 연구 아이디어: Two-stage refit

$$d_i \xrightarrow{\phi} x_i \in \mathbb{R}^d \xrightarrow{/\|\cdot\|_2} z_i \in \mathbb{S}^{d-1}$$

- Stage 1 (screening): $\widehat{S}_\lambda \leftarrow L_1$-penalized vMF mixture on $\eta_h = \kappa_h \mu_h$
- Stage 2 (refit): $\widehat{\Theta}^{\text{refit}}_{\widehat{S}} \leftarrow$ unpenalized sparse-vMF on $\mathbb{S}^{d-1}$

## 3. 선행연구와의 차이

| **구분** | **Rossi & Barbaro (2022)** | **본 연구** |
|---|---|---|
| Penalty 대상 | $\mu_h$ | $\eta_h = \kappa_h \mu_h$ (cluster contrast) |
| Sparsity 의미 | cluster별 prototype | cluster 간 판별 좌표 |
| 최종 추정량 | penalized | unpenalized refit |
| Bias | $L_1$ shrinkage 잔존 | refit으로 완화 |
| Refit 단계 | 없음 | 선택된 support에서 원래 $\mathbb{S}^{d-1}$ 위 |
| Model selection | path + IC | refit-likelihood 기반 BIC/ICL |

Penalty 대상을 $\mu_h$ 가 아닌 $\eta_h$ 로 두는 근거는 posterior log-ratio가 $\eta_h$ 에 선형으로 의존하기 때문이다.

$$\log \frac{P(h \mid z_i)}{P(\ell \mid z_i)} = \log \frac{\pi_h}{\pi_\ell} + \log \frac{c_d(\kappa_h)}{c_d(\kappa_\ell)} + (\eta_h - \eta_\ell)^\top z_i$$

$\mu_{hj} = \mu_{\ell j}$ 라도 $\kappa_h \neq \kappa_\ell$ 이면 $\eta_{hj} \neq \eta_{\ell j}$ 이므로 판별 정보가 보존된다.

## 4. 모형과 수식

### 4.1 Reparametrization

$\eta_h = \kappa_h \mu_h$ 로 두면

$$p(z_i \mid \pi, \eta) = \sum_{h=1}^{K} \pi_h c_d(\|\eta_h\|_2) \exp(\eta_h^\top z_i)$$

### 4.2 Stage 1: Cluster-contrast penalty

$\mu_h$ 가 같아도 $\kappa_h$ 가 다르면 자연모수 $\eta_h = \kappa_h \mu_h$ 는 다르다. 두 군집의 판별 정보는 $\mu_h$ 차이가 아니라 $\eta_h$ 차이에 담겨 있으므로, 페널티 대상도 $\eta_h$ 가 되어야 한다.
$$\bar{\eta}_j = \sum_h w_h \eta_{hj}, \qquad P_B(\eta) = \sum_{j=1}^{d} \left[ \sum_h w_h(\eta_{hj} - \bar{\eta}_j)^2 \right]^{1/2}$$

$$\mathcal{L}^{B}_{\lambda_n}(\pi, \eta) = \ell_n(\pi, \eta) - n\lambda_n P_B(\eta)$$

$$\widehat{S}_\lambda = \left\lbrace j : \left[ \sum_{h=1}^{K} w_h (\widehat{\eta}_{hj} - \widehat{\bar{\eta}}_j)^2 \right]^{1/2} > \epsilon \right\rbrace$$

모든 cluster에 공통으로 큰 좌표는 active set에서 제외된다. 따라서 support는 분포 설명이 아니라 cluster 판별을 기준으로 정의된다.

### 4.3 Stage 2: Sparse-vMF refit on $\mathbb{S}^{d-1}$

제약은 $\mu_{h,\widehat{S}^c} = 0,\ \|\mu_{h,\widehat{S}}\|_2 = 1$ 이다. density를 selected sphere가 아니라 원래 $\mathbb{S}^{d-1}$ 위에서 정의하므로 정규화 상수는 $c_{d_\lambda}$ 가 아니라 $c_d$ 를 사용한다.

$$p(z_i \mid \widetilde{\Theta}_{\widehat{S}}) = \sum_h \widetilde{\pi}_h c_d(\widetilde{\kappa}_h) \exp\left( \widetilde{\kappa}_h \widetilde{\mu}_{h,\widehat{S}}^\top z_{i,\widehat{S}} \right)$$

EM update:

$$\widetilde{\tau}_{ih} = \frac{\widetilde{\pi}_h c_d(\widetilde{\kappa}_h) \exp(\widetilde{\kappa}_h \widetilde{\mu}_{h,\widehat{S}}^\top z_{i,\widehat{S}})}{\sum_\ell \widetilde{\pi}_\ell c_d(\widetilde{\kappa}_\ell) \exp(\widetilde{\kappa}_\ell \widetilde{\mu}_{\ell,\widehat{S}}^\top z_{i,\widehat{S}})}$$

$$r_{h,\widehat{S}} = \sum_i \widetilde{\tau}_{ih} z_{i,\widehat{S}}, \qquad N_h = \sum_i \widetilde{\tau}_{ih}$$

$$\widehat{\widetilde{\mu}}_{h,\widehat{S}} = \frac{r_{h,\widehat{S}}}{\|r_{h,\widehat{S}}\|_2}, \qquad \widehat{\widetilde{\mu}}_{h,\widehat{S}^c} = 0, \qquad A_d(\widehat{\widetilde{\kappa}}_h) = \frac{\|r_{h,\widehat{S}}\|_2}{N_h}$$

Banerjee 근사:

$$\widehat{\widetilde{\kappa}}_h \approx \frac{\bar{R}d - \bar{R}^3}{1 - \bar{R}^2}, \qquad \bar{R} = \frac{\|r_{h,\widehat{S}}\|_2}{N_h}$$

## 5. Model selection

모든 $(K,\lambda)$ 후보가 공통 sample space $\mathbb{S}^{d-1}$ 위에 있으므로 refit log-likelihood로 BIC 비교가 가능하다.

$$\nu(K,\lambda) = K d_\lambda + K - 1 \quad (\kappa_h \text{ component-specific}), \qquad \nu_{\text{common}}(K,\lambda) = K d_\lambda$$

$$\text{BIC}(K,\lambda) = -2\ell_n(\widehat{\widetilde{\Theta}}^{\text{refit}}_{K,\lambda}) + \nu(K,\lambda)\log n$$

현 단계에서는 practical criterion으로 제시하고, 이론적 정당화는 후속 과제로 둔다.

## 6. 요약

1. GMM 공분산 모수 폭발을 vMF의 scalar $\kappa_h$ 로 우회한다.
2. Penalty 대상을 log-density의 선형 판별항인 $\eta_h = \kappa_h \mu_h$ 로 둔다.
3. Penalized estimator를 최종 모형이 아니라 screening 단계로만 사용한다.
4. Refit을 원래 $\mathbb{S}^{d-1}$ 위에서 정식화하여 서로 다른 $\lambda$ 모델이 같은 sample space에서 비교되도록 한다.
5. 해석은 active 좌표가 아니라 $\kappa_h$ 와 대표 문서, cluster summary로 수행한다.

## 7. 미팅 논의 사항

### 7.1 현 단계 프레임워크 및 기본 모형에 대한 검토 요청

본 미팅에서는 다음 프레임워크와 기본 모형의 타당성에 대해 교수님의 검토와 의견을 구하고자 한다.

#### (1) 문제 설정 — L2 정규화된 텍스트 임베딩의 directional clustering

- LLM 임베딩 $x_i = \phi(d_i) \in \mathbb{R}^d$ 를 L2 정규화하여 $z_i \in \mathbb{S}^{d-1}$ 위의 directional data로 다룬다.
- Cosine similarity 기반 임베딩의 기하학적 구조와 부합한다.
- 비지도 군집화 상황을 가정한다.

#### (2) 기본 확률 모형 — vMF mixture

- GMM full covariance의 $O(Kd^2)$ 모수 폭발을 회피한다.
- Cluster별 angular concentration을 scalar $\kappa_h$ 로 표현한다.
- 평균이 비슷해도 $\kappa_h$ 가 다르면 분리 가능한 구조를 제공한다.
- 단, vMF는 평균 방향 주변의 angular spread를 isotropic으로 가정한다는 한계를 인지한다.

#### (3) 핵심 방법론 — Two-stage sparse vMF mixture

- **Stage 1 (screening):** 자연모수 $\eta_h = \kappa_h \mu_h$ 에 cluster-contrast group penalty $P_B(\eta)$ 를 부여한 penalized vMF mixture로 active set $\widehat{S}_\lambda$ 를 추출.
- **Stage 2 (refit):** 선택된 support에서 원래 $\mathbb{S}^{d-1}$ 위의 unpenalized sparse-vMF submodel로 refit.
- Meynet식 Lasso-MLE 원칙을 directional mixture에 적용한 정식화.

#### (4) Novelty 위치 설정

선행 연구 대비 본 연구의 기여 위치를 다음과 같이 설정하고자 한다.

| 구분 | 기존 연구 | 본 연구 |
|---|---|---|
| Sparse vMF mixture | Rossi & Barbaro (2022): $L_1$ on $\mu_h$ | $\eta_h$ 의 cluster-contrast group penalty |
| Cluster-contrast 변수선택 | Pan & Shen (2007) 등: Euclidean GMM | Directional setting (vMF) 으로 확장 |
| Two-stage Lasso-MLE | Meynet 계열: regression / GMM | Sparse directional mixture에 적용 |

### 7.2 후속 검토 항목

위 프레임워크가 확정되면 다음 단계에서 구체화할 항목들이다.

- vMF의 angular isotropic 가정이 LLM 임베딩 자료에 어느 정도 적합한지에 대한 진단 방법
- $\kappa_h$ 를 component-specific으로 둘지 common으로 둘지의 결정 기준 ($n \ll d$ 인 경우 포함)
- $P_B(\eta)$ 의 식별성 및 weight $w_h$ 선택 ($\pi_h$ vs $1/K$ vs adaptive)
- BIC/ICL의 정당화 범위와 이론적 보장 가능 영역
- Threshold $\epsilon$ 의 자료기반 결정 방법 (예: stability selection, bootstrap)
- 실험 setup: synthetic 자료 설계, SBERT/OpenAI/SimCSE 등 실제 임베딩 비교군 구성
- 비교 baseline: Rossi & Barbaro (2022) 직접 비교, Witten & Tibshirani (2010) sparse k-means, dense vMF + threshold 후 refit- BIC/ICL의 정당화 범위와 이론적 보장 가능 영역
- Threshold $\epsilon$ 의 자료기반 결정 방법 (예: stability selection, bootstrap)
- 실험 setup: synthetic 자료 설계, SBERT/OpenAI/SimCSE 등 실제 임베딩 비교군 구성
- 비교 baseline: Rossi & Barbaro (2022) 직접 비교, Witten & Tibshirani (2010) sparse k-means, dense vMF + threshold 후 refit

## 참고문헌

- Banerjee, A., Dhillon, I. S., Ghosh, J., & Sra, S. (2005). Clustering on the unit hypersphere using von Mises-Fisher distributions. *Journal of Machine Learning Research*, 6, 1345–1382.
- Rossi, F., & Barbaro, F. (2022). Mixture of von Mises-Fisher distribution with sparse prototypes. *Neurocomputing*. arXiv:2212.14591.
- Pan, W., & Shen, X. (2007). Penalized model-based clustering with application to variable selection. *Journal of Machine Learning Research*, 8, 1145–1164.
- Wang, S., & Zhu, J. (2008). Variable selection for model-based high-dimensional clustering and its application to microarray data. *Biometrics*.
- Xie, B., Pan, W., & Shen, X. (2008). Penalized model-based clustering with cluster-specific diagonal covariance matrices and grouped variables. *Electronic Journal of Statistics*, 2, 168–212.
- Witten, D. M., & Tibshirani, R. (2010). A framework for feature selection in clustering. *Journal of the American Statistical Association*, 105(490), 713–726.
