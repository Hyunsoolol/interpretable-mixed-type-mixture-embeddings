# SZL-Refit-Cov: Mean / Variance / Correlation Effect Selection

## 1. Motivation

**Mean-only SZL-Refit**은 다음 target만 선택합니다.

$$S_\mu = \{j : \mu_{1j}, \dots, \mu_{Kj} \text{ 가 서로 다름} \}$$

하지만 실제 군집 차이는 평균뿐 아니라 분산, 상관구조에서도 발생할 수 있음

**예시:**

$$\mu_A = \mu_B = (0,0,0)$$

$$\Sigma_A = \begin{pmatrix} 1 & 0.1 & 0 \\ 0.1 & 1 & 0.2 \\ 0 & 0.2 & 1 \end{pmatrix}, \qquad \Sigma_B = \begin{pmatrix} 1 & 0.1 & 0 \\ 0.1 & 1 & 0.5 \\ 0 & 0.5 & 1 \end{pmatrix}$$

$$S_\mu = \varnothing, \qquad S_\sigma = \varnothing, \qquad E_\rho = \{(2,3)\}$$

즉, 군집 신호가 mean이 아니라 **correlation structure**에 존재할 수 있음

---

## 2. Gaussian Mixture with Cluster-specific Covariance

$$Z_i \in \{1, \dots, K\}, \qquad P(Z_i=k) = \pi_k$$

$$X_i \mid Z_i=k \sim N_p(\mu_k, \Sigma_k)$$

**Covariance decomposition:**

$$\Sigma_k = D_k R_k D_k$$

$$D_k = \operatorname{diag}(\sigma_{k1}, \dots, \sigma_{kp}), \qquad R_k = (\rho_{k,j\ell})_{j,\ell=1}^{p}$$

---

## 3. Mean Effect

$$\mu_{k,j} = \mu_{0,j} + \delta_{k,j}$$

$$\sum_{k=1}^{K} \delta_{k,j} = 0$$

**Mean-active variables:**

$$S_\mu^0 = \{j : \delta_{\cdot,j}^0 \neq 0\}$$

---

## 4. Variance Effect

$$v_{k,j} = \log\sigma_{k,j}^2$$

$$v_{k,j} = v_{0,j} + \eta_{k,j}$$

$$\sum_{k=1}^{K} \eta_{k,j} = 0$$

**Variance-active variables:**

$$S_\sigma^0 = \{j : \eta_{\cdot,j}^0 \neq 0\}$$

---

## 5. Correlation Effect

$$\rho_{k,j\ell} = \operatorname{Corr}(X_j, X_\ell \mid Z=k)$$

**Fisher transform:**

$$z_{k,j\ell} = \operatorname{atanh}(\rho_{k,j\ell}) = \frac{1}{2} \log \left( \frac{1+\rho_{k,j\ell}}{1-\rho_{k,j\ell}} \right)$$

$$z_{k,j\ell} = z_{0,j\ell} + \gamma_{k,j\ell}$$

$$\sum_{k=1}^{K} \gamma_{k,j\ell} = 0$$

**Correlation-active pairs:**

$$E_\rho^0 = \{(j,\ell) : \gamma_{\cdot,j\ell}^0 \neq 0\}$$

**Correlation-active variables:**

$$S_\rho^0 = \{j : \exists \ell, (j,\ell) \in E_\rho^0\}$$

---

## 6. Total Cluster-driving Structure

$$S_{\mathrm{total}}^0 = S_\mu^0 \cup S_\sigma^0 \cup S_\rho^0$$

$$\text{Cluster-driving source} = (S_\mu^0, S_\sigma^0, E_\rho^0)$$

**해석:**

- **$S_\mu^0$**: mean 차이 변수
    
- **$S_\sigma^0$**: variance 차이 변수
    
- **$E_\rho^0$**: correlation 차이 pair
    

---

## 7. EM Weights

**E-step:**

$$\hat r_{ik} = P(Z_i=k \mid X_i; \hat\Theta)$$

$$\hat r_{ik} = \frac{\hat\pi_k \phi_p(X_i; \hat\mu_k, \hat\Sigma_k)}{\sum_{h=1}^{K} \hat\pi_h \phi_p(X_i; \hat\mu_h, \hat\Sigma_h)}$$

**Effective sample size:**

$$\hat N_k = \sum_{i=1}^{n} \hat r_{ik}$$

---

## 8. Weighted Moment Estimates

- **Mean:** $\bar x_{k,j} = \frac{\sum_{i=1}^{n} \hat r_{ik}X_{ij}}{\hat N_k}$
    
- **Variance:** $\hat\sigma_{k,j}^2 = \frac{\sum_{i=1}^{n} \hat r_{ik}(X_{ij}-\bar x_{k,j})^2}{\hat N_k}$
    
- **Covariance:** $\hat\Sigma_{k,j\ell} = \frac{\sum_{i=1}^{n} \hat r_{ik}(X_{ij}-\bar x_{k,j})(X_{i\ell}-\bar x_{k,\ell})}{\hat N_k}$
    
- **Correlation:** $\hat\rho_{k,j\ell} = \frac{\hat\Sigma_{k,j\ell}}{\sqrt{\hat\sigma_{k,j}^2 \hat\sigma_{k,\ell}^2}}$
    
- **Fisher-z correlation:** $\hat z_{k,j\ell} = \operatorname{atanh}(\hat\rho_{k,j\ell})$
    

---

## 9. Effect Estimates

- **Mean effect:** $\hat\delta_{k,j} = \bar x_{k,j} - \bar x_{0,j}, \qquad \bar x_{0,j} = \frac{1}{K} \sum_{h=1}^{K} \bar x_{h,j}$
    
- **Variance effect:** $\hat v_{k,j} = \log \hat\sigma_{k,j}^2 \implies \hat\eta_{k,j} = \hat v_{k,j} - \frac{1}{K} \sum_{h=1}^{K} \hat v_{h,j}$
    
- **Correlation effect:** $\hat\gamma_{k,j\ell} = \hat z_{k,j\ell} - \frac{1}{K} \sum_{h=1}^{K} \hat z_{h,j\ell}$
    

---

## 10. Screening Sets

- **Mean screening:** $\hat S_\mu = \{j : \|\hat\delta_{\cdot,j}\|_2 > \tau_\mu\}$
    
- **Variance screening:** $\hat S_\sigma = \{j : \|\hat\eta_{\cdot,j}\|_2 > \tau_\sigma\}$
    
- **Correlation screening:** $\hat E_\rho = \{(j,\ell) : \|\hat\gamma_{\cdot,j\ell}\|_2 > \tau_\rho\}$
    
- **Correlation-active variables:** $\hat S_\rho = \{j : \exists \ell, (j,\ell) \in \hat E_\rho\}$
    
- **Total selected variables:** $\hat S_{\mathrm{total}} = \hat S_\mu \cup \hat S_\sigma \cup \hat S_\rho$
    

---

## 11. Lasso-type Screening Form

### Mean effect

$$\min_{\mu_{0,j},\delta_{1,j},\dots,\delta_{K,j}} \sum_{k=1}^{K} w_{k,j}^{\mu} (\bar x_{k,j}-\mu_{0,j}-\delta_{k,j})^2 + \lambda_\mu \sum_{k=1}^{K} |\delta_{k,j}|$$

$$\text{s.t.} \qquad \sum_{k=1}^{K} \delta_{k,j}=0$$

### Variance effect

$$\min_{v_{0,j},\eta_{1,j},\dots,\eta_{K,j}} \sum_{k=1}^{K} w_{k,j}^{\sigma} (\hat v_{k,j}-v_{0,j}-\eta_{k,j})^2 + \lambda_\sigma \sum_{k=1}^{K} |\eta_{k,j}|$$

$$\text{s.t.} \qquad \sum_{k=1}^{K} \eta_{k,j}=0$$

### Correlation effect

$$\min_{z_{0,j\ell},\gamma_{1,j\ell},\dots,\gamma_{K,j\ell}} \sum_{k=1}^{K} w_{k,j\ell}^{\rho} (\hat z_{k,j\ell}-z_{0,j\ell}-\gamma_{k,j\ell})^2 + \lambda_\rho \sum_{k=1}^{K} |\gamma_{k,j\ell}|$$

$$\text{s.t.} \qquad \sum_{k=1}^{K} \gamma_{k,j\ell}=0$$

**Approximate weights:**

$$w_{k,j}^{\mu} \asymp \frac{\hat N_k}{\hat\sigma_{k,j}^2}, \qquad w_{k,j}^{\sigma} \asymp \hat N_k, \qquad w_{k,j\ell}^{\rho} \asymp \hat N_k-3$$

---

## 12. Positive Definiteness Issue

Pairwise correlation selection alone으로 $\hat\Sigma_k \succ 0$를 보장하지 못합니다.

- **Selected correlation graph:** $\hat G_\rho = (\hat S_\rho, \hat E_\rho)$
    
- **Connected components:** $\hat B_1, \dots, \hat B_M$
    
- **Block covariance refit:** $\Sigma_{k,\hat B_m,\hat B_m}$ is cluster-specific.
    
- **Outside selected blocks:** $\Sigma_{k,\hat B_m^c,\hat B_m^c}$ is common or diagonal nuisance.
    

---

## 13. Unpenalized Refit

**Fix selected structure:** $(\hat S_\mu, \hat S_\sigma, \hat E_\rho)$

**Refit by MLE:**

$$\hat\Theta^{\mathrm{refit}} = \arg\max_{\Theta \in \mathcal{M}(\hat S_\mu, \hat S_\sigma, \hat E_\rho)} \ell_n(\Theta)$$

**Structural constraints:**

- $j \notin \hat S_\mu \implies \mu_{1,j} = \dots = \mu_{K,j}$
    
- $j \notin \hat S_\sigma \implies \sigma_{1,j}^2 = \dots = \sigma_{K,j}^2$
    
- $(j,\ell) \notin \hat E_\rho \implies \rho_{1,j\ell} = \dots = \rho_{K,j\ell}$
    

**Selected effects $(\delta, \eta, \gamma)$ are refit without penalty.**

---

## 14. Recovery Ratios

- **Mean recovery:** $R_j^\mu = \frac{\|\hat\delta_{\cdot,j}\|_2}{\|\delta_{\cdot,j}^0\|_2}$
    
- **Variance recovery:** $R_j^\sigma = \frac{\|\hat\eta_{\cdot,j}\|_2}{\|\eta_{\cdot,j}^0\|_2}$
    
- **Correlation recovery:** $R_{j\ell}^\rho = \frac{\|\hat\gamma_{\cdot,j\ell}\|_2}{\|\gamma_{\cdot,j\ell}^0\|_2}$
    

---

## 15. Core Message

> **Mean-only SZL-Refit:** $S_\mu^0$ 만 고려
> 
> **SZL-Refit-Cov:** $(S_\mu^0, S_\sigma^0, E_\rho^0)$ 통합 고려

**Cluster-driving signal:**

$$\boxed{\text{Mean effect} + \text{Variance effect} + \text{Correlation effect}}$$

**Post-selection refit:**

$$\boxed{\text{sparse screening} \quad \rightarrow \quad \text{unpenalized recovery of selected effects}}$$
