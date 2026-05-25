# 연구 진행 정리

## 1. 배경

### 1.1 GMM의 한계와 vMF로의 전환

미팅 피드백은 평균이 비슷해도 퍼짐이 다르면 별개의 군집으로 분리하고 싶다는 것이었다. GMM에서 이를 다루려면 cluster별 공분산 $\Sigma_h$ 가 필요한데, 고차원에서 모수가 폭발한다.

|**모형**|**cluster당 퍼짐 모수**|**d=768 예시**|
|---|---|---|
|GMM full cov.|$d(d+1)/2$|$\approx 3 \times 10^5$|
|GMM diag. cov.|$d$|768|
|vMF|1 (scalar $\kappa_h$)|1|
|vMF (common $\kappa$)|0 (전역 1)|0|

vMF로 옮기면 두 setting이 다음과 같이 대응된다.

$$\mu_h \text{ 비슷}, \ \Sigma_h \text{ 다름} \quad \longleftrightarrow \quad \mu_h \text{ 비슷}, \ \kappa_h \text{ 다름}$$

단, vMF는 평균 방향 주변의 angular spread를 isotropic으로 가정하므로 좌표별 또는 방향별로 비등방인 분산 구조는 모델링하지 못한다. L2 정규화된 임베딩에서는 이 손실보다 모수 폭발을 해소하는 이득이 크다고 판단한다. 또한 component-specific $\kappa_h$ 가 너무 자유로우면 소수 관측치에 과적합할 수 있으므로, $n \ll d$ 인 경우에는 common $\kappa$ 또는 $\kappa_h$ 에 cap을 두는 방안을 함께 검토한다.

### 1.2 vMF mixture

$$f(z \mid \mu, \kappa) = c_d(\kappa)\exp(\kappa\mu^\top z), \qquad z, \mu \in \mathbb{S}^{d-1},\ \kappa \geq 0$$

$$p(z_i \mid \Theta) = \sum_{h=1}^{K} \pi_h c_d(\kappa_h) \exp(\kappa_h \mu_h^\top z_i)$$

Banerjee et al. (2005) 은 vMF mixture가 spherical k-means의 확률모형적 일반화임을 보였다.

### 1.3 선행연구: Rossi & Barbaro (2022)

$L_1$ penalty로 $\mu_h$ 자체를 sparse하게 추정한다.

$$\mathcal{L}_p(\Theta) = \mathcal{L}(\Theta) - \beta \sum_{h=1}^{K} |\mu_h|_1$$

한계는 두 가지다. penalized estimator를 그대로 최종 모형으로 사용하므로 $L_1$ shrinkage bias가 남고, penalty가 $\mu_h$ 에만 걸려 실제 판별항 $\kappa_h \mu_h$ 의 구조를 반영하지 못한다.

## 2. 연구 아이디어: Two-stage refit

$$d_i \xrightarrow{\phi} x_i \in \mathbb{R}^d \xrightarrow{/|\cdot|_2} z_i \in \mathbb{S}^{d-1}$$

- Stage 1 (screening): $\widehat{S}_\lambda \leftarrow L_1$-penalized vMF mixture on $\eta_h = \kappa_h \mu_h$
    
- Stage 2 (refit): $\widehat{\Theta}^{\text{refit}}_{\widehat{S}} \leftarrow$ unpenalized sparse-vMF on $\mathbb{S}^{d-1}$
    

## 3. 선행연구와의 차이

|**구분**|**Rossi & Barbaro (2022)**|**본 연구**|
|---|---|---|
|Penalty 대상|$\mu_h$|$\eta_h = \kappa_h \mu_h$ (cluster contrast)|
|Sparsity 의미|cluster별 prototype|cluster 간 판별 좌표|
|최종 추정량|penalized|unpenalized refit|
|Bias|$L_1$ shrinkage 잔존|refit으로 완화|
|Refit 단계|없음|선택된 support에서 원래 $\mathbb{S}^{d-1}$ 위|
|Model selection|path + IC|refit-likelihood 기반 BIC/ICL|

Penalty 대상을 $\mu_h$ 가 아닌 $\eta_h$ 로 두는 근거는 posterior log-ratio가 $\eta_h$ 에 선형으로 의존하기 때문이다.

$$\log \frac{P(h \mid z_i)}{P(\ell \mid z_i)} = \log \frac{\pi_h}{\pi_\ell} + \log \frac{c_d(\kappa_h)}{c_d(\kappa_\ell)} + (\eta_h - \eta_\ell)^\top z_i$$

$\mu_{hj} = \mu_{\ell j}$ 라도 $\kappa_h \neq \kappa_\ell$ 이면 $\eta_{hj} \neq \eta_{\ell j}$ 이므로 판별 정보가 보존된다.

## 4. 모형과 수식

### 4.1 Reparametrization

$\eta_h = \kappa_h \mu_h$ 로 두면

$$p(z_i \mid \pi, \eta) = \sum_{h=1}^{K} \pi_h c_d(|\eta_h|_2) \exp(\eta_h^\top z_i)$$

### 4.2 Stage 1: Cluster-contrast penalty

$$\bar{\eta}_j = \sum_h w_h \eta_{hj}, \qquad P_B(\eta) = \sum_{j=1}^{d} \left[ \sum_h w_h(\eta_{hj} - \bar{\eta}_j)^2 \right]^{1/2}$$

$$\mathcal{L}^{B}_{\lambda_n}(\pi, \eta) = \ell_n(\pi, \eta) - n\lambda_n P_B(\eta)$$

$$\widehat{S}_\lambda = \left\{ j : \left[ \sum_{h=1}^{K} w_h (\widehat{\eta}_{hj} - \widehat{\bar{\eta}}_j)^2 \right]^{1/2} > \epsilon \right\}$$

모든 cluster에 공통으로 큰 좌표는 active set에서 제외된다. 따라서 support는 분포 설명이 아니라 cluster 판별을 기준으로 정의된다.

### 4.3 Stage 2: Sparse-vMF refit on $\mathbb{S}^{d-1}$

제약은 $\mu_{h,\widehat{S}^c} = 0,\ |\mu_{h,\widehat{S}}|_2 = 1$ 이다. density를 selected sphere가 아니라 원래 $\mathbb{S}^{d-1}$ 위에서 정의하므로 정규화 상수는 $c_{d_\lambda}$ 가 아니라 $c_d$ 를 사용한다.

$$p(z_i \mid \widetilde{\Theta}_{\widehat{S}}) = \sum_h \widetilde{\pi}_h c_d(\widetilde{\kappa}_h) \exp\left( \widetilde{\kappa}_h \widetilde{\mu}_{h,\widehat{S}}^\top z_{i,\widehat{S}} \right)$$

EM update:

$$\widetilde{\tau}_{ih} = \frac{\widetilde{\pi}_h c_d(\widetilde{\kappa}_h) \exp(\widetilde{\kappa}_h \widetilde{\mu}_{h,\widehat{S}}^\top z_{i,\widehat{S}})}{\sum_\ell \widetilde{\pi}_\ell c_d(\widetilde{\kappa}_\ell) \exp(\widetilde{\kappa}_\ell \widetilde{\mu}_{\ell,\widehat{S}}^\top z_{i,\widehat{S}})}$$

$$r_{h,\widehat{S}} = \sum_i \widetilde{\tau}_{ih} z_{i,\widehat{S}}, \qquad N_h = \sum_i \widetilde{\tau}_{ih}$$

$$\widehat{\widetilde{\mu}}_{h,\widehat{S}} = \frac{r_{h,\widehat{S}}}{|r_{h,\widehat{S}}|_2}, \qquad \widehat{\widetilde{\mu}}_{h,\widehat{S}^c} = 0, \qquad A_d(\widehat{\widetilde{\kappa}}_h) = \frac{|r_{h,\widehat{S}}|_2}{N_h}$$

Banerjee 근사:

$$\widehat{\widetilde{\kappa}}_h \approx \frac{\bar{R}d - \bar{R}^3}{1 - \bar{R}^2}, \qquad \bar{R} = \frac{|r_{h,\widehat{S}}|_2}{N_h}$$

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

- vMF의 angular isotropic 가정이 본 자료에 적합한지
    
- $\kappa_h$ 를 component-specific으로 둘지 common으로 둘지, $n \ll d$ 인 경우의 판단 기준
    
- $P_B(\eta)$ 의 식별성과 weight $w_h$ 선택
    
- BIC/ICL의 정당화 범위
    
- Threshold $\epsilon$ 의 자료기반 결정 방법
    
- 실험 setup, synthetic 자료와 SBERT/OpenAI embedding 등 실제 임베딩에서의 비교군 구성
