# [연구 미팅 보고서] 고차원 혼합 평균 효과 클러스터링을 위한 Debiased Sum-to-Zero Lasso

**미팅 일자:** 2026년 5월 8일

---

## 핵심 요약

본 보고서는 고차원 비지도 학습 환경에서 군집 간 평균 차이를 유발하는 변수(mean-heterogeneity-driving variables)를 식별하기 위한 **Debiased Sum-to-Zero Lasso Mixture Clustering** 방법론을 제안한다. 직전 미팅에서 검토된 group penalty 기반 설계(HP-L, HP-AL)에 대한 지도교수 의견과 기존 시뮬레이션 결과를 종합 재검토한 결과, 본 연구의 메인 contribution을 다음 구조로 재정립하였다.

$$\text{Sum-to-zero constrained lasso screening} \;\longrightarrow\; \text{variable-level aggregation} \;\longrightarrow\; \text{unpenalized GMM refit} \;\longrightarrow\; \text{refit-likelihood EBIC tuning}$$

이번 재설계의 핵심 변경 및 근거는 다음과 같다.

- **메인 방법의 전환:** 기존 보고서의 메인 제안 모형이었던 HP-AL(Adaptive Group Lasso)을 메인 contribution에서 제외한다. Group penalty 계열은 (1) 지도교수 의견과의 정합성, (2) Pan-Shen, Xie-Pan-Shen, Guo et al. 등 선행연구와의 중복 위험, (3) sum-to-zero 제약 하에서 element-wise selection이 자연스럽게 variable-level selection을 유도한다는 구조적 사실을 종합 고려하여 메인에서 제외한다.

- **Naive Lasso 결과의 재해석을 통한 메인 메시지 전환:** 기존 보고서에서 Naive Lasso는 주 분석 구간($a \in \{1.6, 1.4, 1.2\}$)에서 TPR이 1.000, FPR이 0.001~0.019 수준으로 **변수 선택 자체는 거의 완벽**했음에도 ARI는 oracle-feature baseline 대비 0.15~0.20 낮은 값에 머물렀다. 이는 실패의 원인이 screening의 약함이 아니라 **선택 변수의 mean contrast가 lasso shrinkage로 과소추정된 데** 있음을 시사한다. 따라서 본 연구의 메인 메시지는 다음으로 정립된다.

  > "Lasso는 최종 추정기가 아니라 mean-heterogeneity 변수의 screening 도구로 사용하고, 선택 변수 위에서 unpenalized refit을 수행함으로써 shrinkage-induced clustering error를 줄인다."

- **Adaptive lasso의 위치 조정:** Adaptive lasso는 메인이 아닌 **secondary extension**으로 둔다. Pilot estimator 안정화 문제가 메인 contribution을 침범하지 않도록 분리하며, plain version으로 oracle gap이 충분히 메워지지 않는 저신호 구간에 대한 robustness variant 역할을 부여한다.

- **이론 챕터의 재구성:** 이론은 (i) sure screening, (ii) selection size control, (iii) oracle refit equivalence의 세 정리로 닫는다. 점근 정규성과 post-selection inference는 selection consistency 조건 하의 corollary 또는 future work로 미룬다.

- **변수 단위 aggregation 기준의 정교화:** 기존 $\|\hat\delta_{\cdot k}\|_2 > \tau$ 기준을 유지하되, 본문 정의는 **standardized max-pairwise contrast** 기준으로 통일하여 "변수 $k$에서 군집 평균이 갈라지는가"라는 연구 질문과 직접 정합되도록 한다.

---

# Part I. 연구 방법론

## 1. 연구배경 및 문제의식

혼합모형 기반 회귀에서는 단순히 중요한 설명변수를 찾는 것만으로 충분하지 않고, 그중에서도 실제로 군집 간 차이를 만들어내는 변수, 즉 source of heterogeneity를 구분하는 것이 더 해석가능하고 더 간명한 모형을 만든다. Li et al.의 혼합회귀 연구는 predictor effect를 공통효과와 군집특이효과로 분해하고, relevant predictor와 heterogeneity-driving predictor를 동시에 식별하는 regularized finite mixture effects regression을 제안하였다.

그러나 비지도학습, 특히 고차원 클러스터링에서는 이와 같은 이질성의 원천 추적이 상대적으로 덜 정식화되어 있다. Pan and Shen(2007)의 $\ell_1$-penalized model-based clustering 이후 Xie, Pan and Shen(2008), Zhou, Pan and Shen(2009), Guo, Levina, Michailidis and Zhu(2010)의 pairwise variable selection, Witten and Tibshirani(2010)의 sparse $K$-means, Celeux et al.(2018)의 SelvarMix, Liu et al.(2023)의 SC-FS 등 다양한 sparse clustering 방법론이 제안되었으나, 다음 두 가지 측면이 통합적으로 다루어진 사례는 드물다.

**첫째, effects-style mean parameterization의 부재.** 기존 sparse clustering 방법론은 대체로 군집 평균 $\mu_j$를 직접 다루거나, 사전 centering을 통해 처리한다. 군집 평균을 공통 평균과 군집특이 편차로 분해하여 "어떤 좌표가 mean heterogeneity를 실제로 유발하는가"를 직접 추적하는 effects-style parameterization $\mu_j = \mu_0 + \delta_j$ with $\sum_j \delta_{jk}=0$ 은 비지도 mixture 문헌에서 명시적으로 강조된 바 없다.

**둘째, screening과 estimation 분리의 부재.** Pan-Shen 계열의 penalized GMM은 penalty estimator 자체를 최종 estimator로 사용한다. 따라서 선택된 변수에서도 mean contrast가 shrinkage된다. 회귀 문헌에서는 Belloni and Chernozhukov(2013)의 post-lasso, Meinshausen(2007)의 relaxed lasso 등 "screening과 unpenalized refit의 분리" 원리가 잘 정립되어 있으나, 비지도 mixture clustering에 명시적으로 이 원리를 적용하고 shrinkage debiasing 효과를 직접 추적한 사례는 찾기 어렵다.

본 연구는 이 두 결손 지점을 정면으로 다룬다. 즉, 반응변수 $Y_i$ 가 없는 상황에서 군집 평균을 latent mean structure로 보고, 이를 공통 평균 파라미터와 군집특이 편차로 분해하여 "어떤 변수들이 군집 간 평균 차이를 만들어내는가"를 직접 추적하는 effects-style 클러스터링 방법론을 개발하되, lasso는 screening 도구로만 사용하고 unpenalized refit으로 shrinkage bias를 제거하는 **두 단계 debiased pipeline**을 구성한다.

다만 현재 1차 범위는 "모든 형태의 군집 형성 변수"가 아니라, 공통 공분산 구조 하에서 mean shift를 통해 군집 분리를 유발하는 변수를 식별하는 데 한정된다. 분산 차이나 상관구조 차이만으로 군집이 갈리는 경우는 현재 baseline model의 범위 밖에 있다.

---

## 2. 연구목표

본 연구의 목표는 고차원 데이터에서 mean-heterogeneity-driving variable selection과 군집 구조 추정을 동시에 수행하되, 추정 단계에서 shrinkage bias가 cluster assignment를 왜곡하지 않도록 설계된 완결된 단일 파이프라인 방법론을 구축하는 것이다.

**첫째,** 비지도 Gaussian mixture에 effects-style mean parameterization $\mu_j = \mu_0 + \delta_j$ 와 sum-to-zero 식별성 제약 $\sum_j \delta_{jk}=0$ 을 명시적으로 도입하고, 이를 통해 mean-heterogeneity-driving variable set $S_0 = \{k : \delta_{\cdot k}^0 \neq 0\}$ 을 자연스럽게 정의한다.

**둘째,** sum-to-zero constrained lasso screening과 unpenalized post-selection refit을 결합한 **debiased two-stage pipeline**을 구축하고, 이 pipeline이 Naive Lasso의 shrinkage-induced clustering error를 직접적으로 제거함을 시뮬레이션과 이론 양측면에서 보인다.

**셋째,** screening의 sure screening property, 선택집합 size control, oracle refit equivalence를 결합한 **two-stage oracle property**를 본 연구의 메인 정리로 제시한다.

---

## 3. 핵심 연구질문

- **Q1.** 비지도 Gaussian mixture에서 sum-to-zero constrained $\ell_1$ penalty는 sure screening property를 가지며, 그 선택집합의 크기는 참 support 크기에 비례하여 제어되는가?

- **Q2.** Naive Lasso가 변수는 잘 찾으면서도 ARI가 oracle-feature baseline에 미달하는 현상은 shrinkage bias로 설명되는가? 그리고 unpenalized refit은 이 gap을 어디까지 메우는가?

- **Q3.** Refit estimator는 어떤 조건에서 oracle GMM MLE와 점근적으로 동등한가? 그리고 이 동등성은 어떤 의미에서 mean contrast의 효과 크기 회복(recovery ratio $R_k \to 1$)으로 시각화되는가?

---

## 4. 제안모형

### 4.1 기본 모형 (기존 보고서와 동일하게 유지)

관측치 $X_i = (X_{i1}, \dots, X_{ip})^\top \in \mathbb{R}^p$, 잠재 군집 $Z_i \in \{1, \dots, K\}$ 에 대하여 다음 baseline model을 사용한다.

$$P(Z_i=j)=\pi_j,\qquad j=1,\dots,K$$

$$X_i\mid Z_i=j \sim N_p(\mu_j,\Sigma)$$

$$\mu_j=\mu_0+\delta_j,\qquad \sum_{j=1}^K \delta_{jk}=0,\qquad k=1,\dots,p$$

여기서 $\mu_0 \in \mathbb{R}^p$ 는 sum-to-zero coding 하의 grand mean parameter이고, $\delta_j \in \mathbb{R}^p$ 는 군집 $j$ 의 mean deviation vector이다.

선택한 제약 $\sum_{j=1}^K \delta_{jk}=0$ 하에서 $\mu_0$ 는 일반적으로 marginal population mean과 동일하지 않다. 실제로 $E(X_i)=\sum_{j=1}^K \pi_j\mu_j=\mu_0+\sum_{j=1}^K \pi_j\delta_j$ 이므로, $\mu_0$ 는 $\pi_j$ 가 모두 같거나 $\sum_j \pi_j\delta_j=0$ 인 특수한 경우에만 marginal mean과 일치한다. 본 연구에서 $\mu_0$ 는 effects-style parameterization에서의 기준점 역할을 하는 grand mean parameter로 해석한다. 특히 sum-to-zero 제약 하에서

$$\mu_0 = \frac{1}{K} \sum_{j=1}^K \mu_j$$

가 성립하므로, $\mu_0$는 군집 평균들의 균등평균(unweighted mean of component means)이다.

### 4.2 이질적 변수의 정의 (기존 보고서와 동일하게 유지)

변수 $k$ 에 대하여 $\delta_{\cdot k}=(\delta_{1k},\dots,\delta_{Kk})^\top$ 라 두면, mean heterogeneity를 유발하는 변수 집합은 다음과 같이 정의된다.

$$S_0=\{k:\exists j\neq \ell,\ \mu_{jk}\neq \mu_{\ell k}\}=\{k:\|\delta_{\cdot k}\|_2 \neq 0\}$$

본 모형이 직접 식별하는 것은 variance heterogeneity나 covariance heterogeneity를 포함한 일반적 의미의 cluster-forming variable 전체가 아니라, 공통 공분산 구조 아래에서 mean shift를 통해 군집 분리를 유발하는 변수이다.

### 4.3 공분산 구조

본 연구의 1차 시뮬레이션에서는 $\Sigma=\mathrm{diag}(\sigma_1^2, \dots, \sigma_p^2)$ 또는 $\Sigma=I_p$로 두는 것이 타당하다. 이 가정 아래에서는 군집이 주어졌을 때 각 좌표가 조건부 독립이므로, mean heterogeneity selection 문제를 가장 선명하게 분리하여 볼 수 있다. 특히 공통 대각 공분산 가정 하에서는 $\delta_{\cdot k}=0 \iff \sigma_k^{-1}\delta_{\cdot k}=0$ 이 성립하므로, 현 단계에서 scaling은 support 정의 자체보다는 군집 간 분리도(separation)와 변수 중요도(importance)에 더 직접적으로 관련된다.

### 4.4 Sum-to-zero 제약의 핵심적 의미

기존 보고서가 식별성 확보를 위해 sum-to-zero 제약을 도입했다면, 본 보고서에서는 이 제약이 **element-wise lasso로도 variable-level selection이 가능하다는 핵심 구조**를 제공한다는 점을 추가로 강조한다. 즉, 제약 하에서

$$\delta_{jk}=0\ \forall j \iff \|\delta_{\cdot k}\|_2 = 0$$

이 성립하므로, "모든 좌표가 0"과 "그룹 norm이 0"이 동치이다. 따라서 element-wise penalty $\sum_j |\delta_{jk}|$ 만으로도 group penalty $\|\delta_{\cdot k}\|_2$ 와 동일한 selection target을 달성할 수 있다. 이 관찰이 본 연구가 group penalty를 명시적으로 사용하지 않는 구조적 근거이다.

---

## 5. 추정방법: Debiased Sum-to-Zero Lasso Pipeline

### 5.1 정규화된 목적함수

모수 $\Theta=(\pi_1,\dots,\pi_K,\mu_0,\delta_1,\dots,\delta_K,\Sigma)$ 에 대해 본 연구에서 고려하는 목적함수는 다음 두 가지로 단순화된다.

**Naive Lasso (Pan-Shen type, sum-to-zero, ablation):**

$$\mathcal{L}_n^{\mathrm{Naive}}(\Theta) = \frac{1}{n}\sum_{i=1}^n \log\left[ \sum_{j=1}^K \pi_j\phi_p(X_i;\mu_0+\delta_j,\Sigma) \right] - \lambda_n \sum_{k=1}^p \sum_{j=1}^K |\delta_{jk}|$$

> subject to $\sum_{j=1}^K \delta_{jk}=0, \quad k=1,\dots,p$

이 모형은 본 연구에서 **shrinkage bias의 직접적 비교 baseline**으로만 사용한다(refit 없음).

**SZL-Screen (제안 메인의 Stage 1):**

$$\mathcal{L}_n^{\mathrm{SZL}}(\Theta) = \frac{1}{n}\sum_{i=1}^n \log\left[ \sum_{j=1}^K \pi_j\phi_p(X_i;\mu_0+\delta_j,\Sigma) \right] - \lambda_n \sum_{k=1}^p \sum_{j=1}^K |\delta_{jk}|$$

> subject to $\sum_{j=1}^K \delta_{jk}=0, \quad k=1,\dots,p$

수식 형태는 Naive Lasso와 동일하나, **이 단계에서 얻은 추정치는 final estimator가 아니라 screening estimator**임을 명시한다. 즉

$$\hat{\Theta}_\lambda^{\mathrm{SZL}} \text{ is a screening estimator, not the final estimator.}$$

**ASZL-Screen (Secondary extension의 Stage 1):**

$$\mathcal{L}_n^{\mathrm{ASZL}}(\Theta) = \frac{1}{n}\sum_{i=1}^n \log\left[ \sum_{j=1}^K \pi_j\phi_p(X_i;\mu_0+\delta_j,\Sigma) \right] - \lambda_n \sum_{k=1}^p \sum_{j=1}^K a_{jk}|\delta_{jk}|$$

> subject to $\sum_{j=1}^K \delta_{jk}=0$, $\;a_{jk} = (|\tilde\delta_{jk}|+\varepsilon_n)^{-\gamma}$

Adaptive weight는 component-wise로 부여한다(Zou 2006의 정의). Pilot $\tilde\delta_{jk}$ 는 plain SZL screening 결과를 사용하여 pilot 안정화 문제를 추가 모형으로 끌어들이지 않는다.

### 5.2 식별성 제약 및 계산 알고리즘

$\mu_j = \mu_0 + \delta_j$ 만으로는 분해가 유일하지 않으므로, $\sum_{j=1}^K \delta_{jk}=0$ 이라는 sum-to-zero 제약이 필수적이다. 실제 EM 구현에서는 $\mathbf{1}_K$ 의 직교여공간 basis $Q \in \mathbb{R}^{K \times (K-1)}$ 를 사용하여

$$\delta_{\cdot k} = Q\alpha_k$$

로 재파라미터화하면 제약이 사라진다. $Q$ 가 column-orthonormal이면

$$\sum_{j=1}^K |\delta_{jk}| = \sum_{j=1}^K \left| \sum_{m=1}^{K-1} Q_{jm}\alpha_{km} \right|$$

이 되어 element-wise lasso의 KKT는 $\alpha_k$ 좌표 단위 soft-thresholding의 변형으로 닫힌 형태에 가깝게 풀린다. 이는 group penalty + Q-basis 결합이 강제했던 무거운 block-wise update를 회피하면서 식별성을 자연스럽게 처리하는 구조이다.

### 5.3 Stage 1.5: Variable-level Aggregation

Lasso는 개별 $\delta_{jk}$ 를 0으로 만들지만, 본 연구의 선택 대상은 변수 $k$ 이다. 따라서 lasso 결과를 변수 단위로 aggregate한다. 본 연구는 standardized max-pairwise contrast 기준을 사용한다.

$$\hat S_\lambda = \left\{ k : \max_{j<\ell} \frac{|\hat\mu_{jk,\lambda}^{\mathrm{SZL}} - \hat\mu_{\ell k,\lambda}^{\mathrm{SZL}}|}{\hat\sigma_{k,\lambda}} > \tau_{\mathrm{num}} \right\}$$

이 기준은 "변수 $k$에서 군집 평균이 실제로 갈라지는가"라는 연구 질문과 직접 정합한다. 여기서 $\tau_{\mathrm{num}}$은 통계적 tuning parameter가 아니라 수치적 파편화 제거를 위한 고정 tolerance이며, 기존 보고서의 $10^{-4}$ 수준을 그대로 유지한다.

### 5.4 Stage 2: Unpenalized Post-Selection Refit (논문의 핵심 단계)

각 $\lambda$가 만든 $\hat S_\lambda$ 를 고정하고, penalty 없이 mixture likelihood를 다시 최대화한다.

$$\hat\Theta_\lambda^{\mathrm{refit}} = \arg\max_{\Theta:\,\delta_{\cdot k}=0,\ k\notin \hat S_\lambda} \frac{1}{n}\sum_{i=1}^n \log\left[ \sum_{j=1}^K \pi_j\phi_p(X_i;\mu_0+\delta_j,\Sigma) \right]$$

즉, $k \in \hat S_\lambda$ 인 변수는 component-specific mean을 자유롭게 추정하고, $k \notin \hat S_\lambda$ 인 변수는 $\mu_{1k}=\cdots=\mu_{Kk}$ 를 강제한다. **이 단계에서 lasso shrinkage가 제거되며, 이것이 본 논문의 핵심 단계이다.**

### 5.5 Stage 3: Refit-likelihood EBIC Tuning

EBIC는 lasso fit 자체가 아니라 refit estimator 기준으로 계산한다.

$$\mathrm{EBIC}_\alpha(\lambda) = -2\ell_n(\hat\Theta_\lambda^{\mathrm{refit}}) + \log n \cdot \mathrm{df}(\hat S_\lambda) + 2\alpha|\hat S_\lambda|\log p$$

공통 대각 공분산 가정 하에서 자유도는

$$\mathrm{df}(\hat S_\lambda) = (K-1) + p + p + (K-1)|\hat S_\lambda|$$

이며, 각 항은 (mixing proportions, common means, diagonal variances, selected variables의 component-specific mean contrasts)에 해당한다. $\Sigma=I_p$로 고정한 경우 variance 자유도 $p$는 빠진다.

최종 선택은

$$\hat\lambda = \arg\min_\lambda \mathrm{EBIC}_\alpha(\lambda),\quad \hat S = \hat S_{\hat\lambda},\quad \hat\Theta = \hat\Theta_{\hat\lambda}^{\mathrm{refit}}$$

이다. $\alpha \in [0, 1]$ 이며, 고차원에서는 $\alpha=0.5$ 또는 $\alpha=1$ 을 기본값으로 두고 sensitivity analysis를 수행한다(Chen and Chen, 2008).

### 5.6 구현상 튜닝과 해석 주의점

- 본 연구에서 제안하는 방법론의 명칭은 다음과 같이 정리한다.
  - **SZL-Refit (메인 제안):** Plain Sum-to-Zero Lasso screening + Unpenalized Refit + Refit-EBIC
  - **ASZL-Refit (Secondary extension):** Adaptive version, component-wise weight $a_{jk}$ 사용

- HP-AL은 더 이상 메인 제안 모형이 아니다. 다만 직전 보고서의 단일 단계(near-oracle) 결과를 비교 reference로 활용하기 위해 group penalty 계열 결과 표는 참고용으로 보존하되, 본 보고서의 메인 비교군에는 포함하지 않는다.

- Sparse K-means의 "사용 차원"은 실제 clustering 단계에서 사용한 변수 수가 아니라, 가중치 threshold를 기준으로 후처리한 유효 선택 변수 수로 해석한다.

- 선택 변수 집합은 다음 수식으로 정의한다.

$$\hat S_\lambda = \left\{ k : \max_{j<\ell} \frac{|\hat\mu_{jk,\lambda} - \hat\mu_{\ell k,\lambda}|}{\hat\sigma_{k,\lambda}} > \tau_{\mathrm{num}} \right\},\quad \tau_{\mathrm{num}} = 10^{-4}$$

여기서 $\tau_{\mathrm{num}}$ 은 수치적 파편화를 제거하기 위한 고정 tolerance이며, $\lambda$ 의 함수가 아닌 상수이다. TPR과 FPR은 모두 $\hat S_\lambda$ 를 기준으로 계산된다.

---

## 6. 이론 구조

본 연구의 이론은 세 정리로 구성된다. 점근 정규성과 post-selection inference는 selection consistency 조건 하의 corollary 또는 future work로 미룬다.

### 6.1 Assumption (요약)

- **(A1) Identifiability and separation.** $\min_{j\neq\ell}\|\mu_j^0 - \mu_\ell^0\|_{\Sigma^{-1}} \geq c_n$ 이고, posterior responsibility가 충분히 안정적으로 추정되도록 $c_n$이 충분히 크다.
- **(A2) Sparsity and dimension.** $s_0 = |S_0| \ll n$, $\log p = o(n)$. 보다 구체적으로 $s_0 \log p = o(n)$.
- **(A3) Beta-min condition.** $B_{\min} = \min_{k \in S_0} \sum_j \pi_j (\mu_{jk}^0 - \bar\mu_k^0)^2 / \sigma_k^2$ 이 sure screening rate를 충분히 만족하는 수준.
- **(A4) Local EM identifiability.** 좋은 initialization 근방에서 EM이 local contraction을 만족하거나, population objective의 true parameter 근방에서 local identifiability가 성립한다.

### 6.2 Theorem 1. Sure screening

조건 (A1)–(A4) 하에서 적절한 $\lambda_n$ sequence에 대해

$$P\left( S_0 \subseteq \hat S_\lambda \right) \to 1.$$

즉, sum-to-zero constrained lasso screening은 참 mean-heterogeneity-driving variables를 빠뜨리지 않는다. Refit이 빠진 변수를 복구할 수 없으므로, Stage 1 목표를 exact selection이 아니라 sure screening으로 두는 것이 본 설계의 안전성을 뒷받침한다.

### 6.3 Theorem 2. Selection size control

같은 조건 하에서 EBIC-tuned $\hat\lambda$ 에 대해

$$|\hat S_{\hat\lambda}| = O_p(s_0).$$

이 결과는 refit 단계의 variance inflation을 제어하기 위해 필요하다.

### 6.4 Theorem 3. Oracle refit equivalence

Oracle estimator를

$$\hat\Theta^{\mathrm{oracle}} = \arg\max_{\Theta:\,\delta_{\cdot k}=0,\ k\notin S_0} \ell_n(\Theta)$$

로 두고, 제안 refit estimator를

$$\hat\Theta^{\mathrm{refit}} = \arg\max_{\Theta:\,\delta_{\cdot k}=0,\ k\notin \hat S} \ell_n(\Theta)$$

로 두자. Theorems 1–2 하에서

$$d(\hat\Theta^{\mathrm{refit}}, \Theta^0) = O_p\left( \sqrt{\frac{(K-1)|\hat S| + \text{nuisance df}}{n}} \right)$$

이며, 더 강하게 $P(\hat S = S_0) \to 1$ 이 성립하면 label permutation을 고려해

$$d(\hat\Theta^{\mathrm{refit}}, \hat\Theta^{\mathrm{oracle}}) = o_p(n^{-1/2}).$$

이 결과를 Theorems 1–3을 결합한 **two-stage oracle property**로 본 정리에 제시한다.

### 6.5 Corollary (post-selection inference, 제한된 형태)

$P(\hat S = S_0) \to 1$ 이 성립하는 강한 조건 하에서 selected mean contrast parameter에 대해 oracle MLE의 점근 정규성을 계승한다. 다만 finite-sample valid post-selection inference는 본 논문의 메인 contribution이 아니며, Berk et al.(2013), Lee et al.(2016) 등 별도 framework로의 확장은 future work로 둔다.

---

# Part II. 시뮬레이션 결과

## 1. 비교 방법론 및 벤치마크 (재구성)

이번 보고서의 비교군은 메인 메시지("screening은 충분하나 shrinkage가 문제이며, refit이 이를 푼다")를 직접 입증하도록 재구성된다. 직전 보고서의 group lasso 계열(HP-L, HP-AL)은 제외한다.

**1) 전통적 비지도 학습**
- K-means
- PCA + K-means
- GMM (Unpenalized)

**2) 기존 sparse / model-based clustering**
- Sparse K-means (Witten and Tibshirani, 2010)
- Naive Lasso (Pan-Shen type, EW-$\ell_1$ + sum-to-zero, **no refit**)
- SelvarMix (Celeux et al., 2018) [신규 추가, 가능 시]
- SC-FS (Liu et al., 2023) [신규 추가, 가능 시]

**3) Critical sanity check 비교군**
- Naive Lasso → Plain SZL screening 후 **refit 추가** (Stage 1+2의 직접 효과 측정)

**4) 제안 모형**
- **SZL-Refit (메인):** Plain Sum-to-Zero Lasso screening + Unpenalized Refit + Refit-EBIC
- **ASZL-Refit (Secondary):** Adaptive Sum-to-Zero Lasso screening + Unpenalized Refit + Refit-EBIC

**5) 오라클 벤치마크**
- Oracle-feature baseline (True Vars)
- True-parameter oracle

> **표 컬럼 해석 주의사항.** $p_{\mathrm{fit}}$는 실제 fitting에 들어간 차원이고, $\hat S_\tau$ 또는 $\hat S$ 는 standardized max-contrast 기준 후 선택된 변수 수이다. Penalty 기반 모형의 single-stage fitting은 전체 $p$ 차원에서 수행되므로 $p_{\mathrm{fit}}=p$ 이며, 실질적 변수 선택 결과는 별도 열 $\hat S$ 를 기준으로 해석한다. TPR과 FPR도 동일 기준이다.

> **Oracle-feature baseline의 해석.** 직전 보고서에서 일부 setting에서 HP-AL이 Oracle-feature baseline을 소폭 상회한 결과가 관찰되었다. 이는 Oracle-feature GMM도 finite-sample/local-optimum 영향을 받는 불안정한 기준값임을 의미한다. 따라서 본 보고서에서는 성능의 주된 기준을 true-parameter oracle 대비 gap으로 두고, Oracle-feature baseline은 보조 reference로만 해석한다.

---

## 2. Critical Sanity Check (메인 결정 실험)

본 연구의 메인 설계 확정을 위해 가장 먼저 수행되어야 할 실험은 다음이다.

### 2.1 실험 설계

- **세팅:** $n=300$, $p \in \{100, 300\}$, $a \in \{1.6, 1.4, 1.2\}$, 반복수 $R \geq 50$
- **비교 방법:** (1) Naive Lasso (refit 없음, 기존 보고서 baseline), (2) **Plain SZL-Refit** (메인 제안), (3) **ASZL-Refit** (Secondary), (4) Oracle-feature baseline, (5) True-parameter oracle

### 2.2 가설과 결정 기준

본 연구의 메인 설계는 다음 패턴이 관찰될 때 확정된다.

| 패턴 | 결과 해석 | 메인 설계 결정 |
|---|---|---|
| **A** | Naive Lasso → SZL-Refit ARI gap의 대부분이 회복됨; SZL-Refit ≈ ASZL-Refit | **Plain SZL-Refit이 메인.** Adaptive는 secondary로만 둠. |
| **B** | SZL-Refit이 부분 회복; ASZL-Refit이 명확히 우수 | ASZL-Refit이 메인 또는 co-main으로 승격. |
| **C** | 두 refit 모두 oracle gap을 메우지 못함 | Refit만의 문제가 아님. 설계 전면 재고. |

직전 보고서의 Naive Lasso 결과 — 주 분석 구간에서 TPR ≈ 1.000, FPR ≈ 0.001~0.019, $\hat S \approx q$ (정답 변수 수에 거의 일치) — 를 고려하면 패턴 A가 가장 가능성이 높다. 즉, screening은 이미 거의 완벽하므로, 진짜 병목은 shrinkage일 가능성이 크다. 다만 이는 가설이며, 본 sanity check 결과가 메인 설계를 최종 확정한다.

### 2.3 핵심 보고 지표

ARI, TPR, FPR, $\hat S$ 외에 다음 지표를 반드시 보고한다.

1. **Mean center MSE**
$$\mathrm{MSE}_\mu = \min_{\pi \in \mathcal{P}_K} \frac{1}{Kp} \sum_{j=1}^K \|\hat\mu_{\pi(j)} - \mu_j^0\|_2^2$$

2. **Mean heterogeneity effect MSE**
$$\mathrm{MSE}_{\Delta, S} = \frac{1}{K|S_0|} \sum_{k \in S_0} \|\hat\delta_{\cdot k} - \delta_{\cdot k}^0\|_2^2$$

3. **Recovery ratio (논문 핵심 지표)**
$$R_k = \frac{\|\hat\delta_{\cdot k}\|_2}{\|\delta_{\cdot k}^0\|_2},\quad k \in S_0$$

4. **Classification entropy**
$$\mathrm{Entropy} = -\frac{1}{n}\sum_{i=1}^n \sum_{j=1}^K \hat r_{ij} \log \hat r_{ij}$$

### 2.4 Hero Figure

본 논문의 핵심 figure는 ARI bar plot이 아니라 $R_k$ 의 분포 비교이다.

- **Panel A (Naive Lasso, no refit):** $R_k$ 분포가 약 0.5~0.8에 분포 (shrinkage 시각화)
- **Panel B (SZL-Refit):** $R_k$ 분포가 1.0 근처에 집중 (debiasing 시각화)

이 한 장이 본 논문의 메인 메시지("lasso는 변수는 잘 찾지만 효과 크기를 줄이고, refit이 이를 복원한다")를 가장 직접적으로 시각화한다.

---

## 3. 직전 보고서 결과의 재해석

직전 보고서의 시뮬레이션 결과는 본 재설계의 강력한 근거가 된다. 주 분석 구간에서 Naive Lasso의 행동을 다시 정리하면 다음과 같다.

### 3.1 Naive Lasso의 selection 거동 — 거의 완벽

| 환경 | TPR | FPR | $\hat S_\tau$ (정답 수) |
|---|---|---|---|
| $p=100, a=1.6$ ($q=5$) | 1.000 | 0.001 | 5.100 (5) |
| $p=100, a=1.4$ ($q=5$) | 1.000 | 0.015 | 6.450 (5) |
| $p=100, a=1.2$ ($q=5$) | 1.000 | 0.006 | 5.580 (5) |
| $p=300, a=1.6$ ($q=5$) | 1.000 | 0.000 | 5.000 (5) |
| $p=300, a=1.2$ ($q=5$) | 1.000 | 0.001 | 5.400 (5) |

주 분석 구간에서 Naive Lasso의 TPR은 1.000, FPR은 0.001~0.015 수준이다. 즉 screening 자체는 사실상 oracle 수준이다.

### 3.2 그러나 ARI는 oracle 대비 큰 gap을 보임

| 환경 | Naive Lasso ARI | Oracle-feature ARI | Gap |
|---|---|---|---|
| $p=100, a=1.6$ | 0.807 | 0.856 | 0.049 |
| $p=100, a=1.4$ | 0.630 | 0.775 | 0.145 |
| $p=100, a=1.2$ | 0.466 | 0.666 | 0.200 |
| $p=300, a=1.4$ | 0.620 | 0.773 | 0.153 |
| $p=300, a=1.2$ | 0.452 | 0.625 | 0.173 |

특히 $p=100, a=1.2$ 와 $p=300, a=1.2$ 에서는 변수를 거의 완벽하게 찾았음에도 ARI gap이 0.17~0.20에 달한다. 변수 선택 자체가 거의 완벽하다는 사실과 ARI gap이 크다는 사실이 양립할 수 있는 가장 자연스러운 설명은 **선택된 변수의 mean contrast 효과 크기가 lasso shrinkage로 과소추정되어 cluster assignment가 약화되었다**는 것이다.

이 해석은 본 보고서의 메인 메시지를 직접 뒷받침한다.

> "Naive Lasso는 screening은 충분히 잘 한다. 진짜 병목은 shrinkage이고, refit이 이를 풀어준다."

### 3.3 실패 구간(한계 신호 $a \leq 0.8$)의 거동은 본 재설계 범위 밖

직전 보고서의 한계 신호 구간($a \in \{0.8, 0.6\}$)에서 Naive Lasso는 TPR이 0.000~0.600으로 내려간다. 이 구간은 beta-min 조건의 경계 근방이며, refit으로 해결 가능한 문제가 아니다(refit은 빠진 변수를 복구하지 않는다). 따라서 본 재설계의 주 분석 구간은 직전 보고서와 동일하게 $a \in \{1.6, 1.4, 1.2\}$ 로 둔다.

---

## 4. 시뮬레이션 시나리오 (재구성)

### 4.1 주 분석 시나리오 (메인)

기존 보고서의 시나리오를 그대로 유지한다.

- $n = 300$
- $p \in \{20, 100, 300\}$
- $q \in \{3, 5, 5\}$ (각 $p$에 대응)
- $a \in \{1.6, 1.4, 1.2\}$
- 반복수 $R \geq 100$ (직전 보고서의 $R=10$ pilot에서 확장)

### 4.2 보조 시나리오 (Supplement)

- 한계 신호: $a \in \{1.0, 0.8\}$ (refit의 한계 검증용; refit이 못 푸는 영역의 명시)
- Unequal mixing: $\pi_1, \dots, \pi_K$ 비대칭
- Correlated predictors: $\mathrm{corr}(X_k, X_\ell) \neq 0$
- Variance heterogeneity ($\Sigma_j \neq \Sigma$): 본 모형 범위 밖이나 robustness 검증용으로만 보고

### 4.3 보고할 표와 그림

각 환경별로 다음 표를 보고한다.

| 비교군 그룹 | 방법 |
|---|---|
| 전통 baseline | K-means, PCA + K-means, Unpenalized GMM |
| 기존 sparse | Sparse K-means, Naive Lasso (no refit) |
| 기존 sparse 추가 | SelvarMix, SC-FS (가능 시) |
| **메인 제안** | **SZL-Refit** |
| **Secondary** | **ASZL-Refit** |
| Oracle | Oracle-feature GMM, True-parameter oracle |

각 셀에는 ARI, TPR, FPR, $\hat S$, $\mathrm{MSE}_\mu$, $\mathrm{MSE}_{\Delta, S}$ 의 평균과 표준오차를 보고한다.

핵심 figure는 다음 두 종류이다.

- **Hero Figure 1:** $R_k$ histogram (Naive Lasso vs SZL-Refit), 가장 도전적인 환경($p=300, a=1.2$)에서.
- **Hero Figure 2:** ARI vs $\lambda$ (또는 EBIC vs $\lambda$) 곡선, Naive Lasso path와 SZL-Refit path를 동일 axis에서 비교하여 refit이 shrinkage bias를 제거하는 모습을 visualize.

---

## 5. 실데이터 분석

직전 보고서에는 없었으나, 본 재설계에서는 실데이터 분석을 1~2개 추가한다. 이는 탑티어 통계 저널 진입을 위한 거의 필수 요소이다.

### 5.1 데이터 후보

가장 자연스러운 응용은 **gene expression** 또는 **single-cell RNA-seq** 이다. 이유는 다음과 같다.

- $p \gg n$ 또는 $p$ 가 큰 고차원 setting이 자연스럽다.
- Cluster-specific mean effect $\delta_{\cdot k}$ 를 gene-specific cell-type-defining effect로 해석할 수 있다.
- Pan-Shen 계열 선행연구도 유사한 응용을 사용한다.

### 5.2 보고할 내용

- 선택 gene 수와 그 안정성 (subsampling 또는 bootstrap stability)
- 선택 gene의 생물학적 해석 (known marker gene 비율 등)
- Refit 전후 cluster assignment entropy 변화
- Refit 전후 mean contrast의 변화 (실데이터 버전 $R_k$ 분석)
- 알려진 subtype label과의 ARI 비교

---

## 6. 논문 framing과 피해야 할 주장

### 6.1 메인 framing

본 논문의 메인 메시지는 다음 한 문장으로 정립된다.

> 고차원 Gaussian mixture clustering에서 lasso는 최종 추정기가 아니라 mean-heterogeneity 변수의 screening 도구로 사용하고, 선택 변수 위에서 unpenalized refit을 수행함으로써 shrinkage-induced clustering error를 줄인다.

### 6.2 핵심 contribution 세 가지

1. **Effects-style mean parameterization** $\mu_j = \mu_0 + \delta_j$ with $\sum_j \delta_{jk}=0$ 의 비지도 mixture clustering으로의 명시적 도입.
2. **Shrinkage debiasing for mixture clustering** — Belloni-Chernozhukov post-lasso framework를 EM 기반 비지도 mixture로 확장.
3. **Refit-likelihood EBIC tuning** — penalized estimator의 likelihood가 아니라 refit estimator의 likelihood로 $\lambda$ 를 선택.

### 6.3 피해야 할 주장

- ❌ "Lasso가 group lasso보다 낫다." 이는 방어하기 어렵다. 대신 "Group penalty는 variable-level target과 자연스러우나, 본 연구는 group-penalty-free refit 전략으로 screening과 estimation을 분리한다"로 표현.
- ❌ "Adaptive weighting이 핵심이다." 본 연구의 메인은 plain version이며, adaptive는 secondary extension이다.
- ❌ "Refit 후 신뢰구간과 검정이 자동으로 valid하다." Post-selection inference는 별도 문제이며, selection consistency 조건 하의 corollary 수준으로만 제시한다.
- ❌ "Oracle-feature baseline이 항상 upper bound이다." 직전 보고서에서 일부 setting에서 HP-AL이 oracle-feature를 상회한 사실이 보여주듯, oracle-feature는 finite-sample/local-optimum 영향을 받는 불안정한 기준이다.

---

## 7. 결론 및 향후 계획

### 7.1 본 재설계의 요약

직전 보고서의 group lasso 메인 설계를 다음과 같이 재정립한다.

1. **메인 contribution을 group penalty 기반(HP-L, HP-AL)에서 group-penalty-free debiased refit pipeline(SZL-Refit)으로 전환.**
2. **Adaptive variant(ASZL-Refit)는 secondary extension으로 배치하여 pilot 안정화 문제를 메인 contribution에서 분리.**
3. **이론은 sure screening + size control + oracle refit equivalence 세 정리로 닫고, 점근 정규성과 PSI는 corollary/future work로 미룸.**
4. **시뮬레이션의 hero metric을 ARI bar plot에서 $R_k$ 분포로 전환하여 메인 메시지(shrinkage debiasing)를 직접 시각화.**
5. **실데이터 응용 1~2개 추가(gene expression 또는 single-cell RNA-seq)로 탑티어 진입 가능성 확보.**

### 7.2 즉시 실행할 우선순위 작업

1. **Critical sanity check 시뮬레이션 우선 실행** — Naive Lasso, SZL-Refit, ASZL-Refit, Oracle-feature 4개 비교군으로 $p \in \{100, 300\}$, $a \in \{1.6, 1.4, 1.2\}$ 구간에서 $R \geq 50$ 반복. 이 결과가 메인 설계 패턴(A/B/C)을 확정한다.
2. **선행연구 정밀 차별화 메모 작성** — SelvarMix(Celeux et al., 2018)와 SC-FS(Liu et al., 2023)에 대한 1~2페이지 차별화 메모를 introduction의 building block으로 준비.
3. **SelvarMix와 SC-FS 시뮬레이션 비교군 추가 구현** — R 패키지로 공개되어 있어 비교 실험 추가 가능.
4. **이론 챕터 outline 초안** — Theorems 1~3의 정확한 statement와 가정 세트, 주요 lemma 구성.

### 7.3 후속 단계

- $R \geq 200$ 으로 시뮬레이션 확대.
- Correlated predictors, unequal mixing 시나리오.
- 실데이터 응용 분석.
- 이론 정리의 자세한 증명 작성.
- 본격 논문 manuscript 작성.

---

## 부록 A. 직전 보고서 대비 변경 사항 요약

| 항목 | 직전 보고서 | 본 재설계 |
|---|---|---|
| 메인 방법 | HP-AL (Adaptive Group Lasso) | **SZL-Refit (Plain Sum-to-Zero Lasso + Refit + EBIC)** |
| Penalty 형태 | Group penalty $\sum_k w_k \|\delta_{\cdot k}\|_2$ | Element-wise $\sum_{k,j} \|\delta_{jk}\|$ |
| Adaptive 위치 | 메인 contribution의 핵심 기제 | Secondary extension |
| 추정 단계 | Single-stage (no refit) | Two-stage (screening + unpenalized refit) |
| Tuning | Heuristic BIC | Refit-likelihood EBIC |
| 핵심 메시지 | "Adaptive group structure가 oracle gap을 줄인다" | "Lasso는 screening 도구일 뿐, refit이 shrinkage debiasing으로 oracle gap을 줄인다" |
| Hero metric | ARI 비교 | $R_k$ 분포 (recovery ratio) |
| 이론 야심 | Group penalty 하 single-stage near-oracle | Two-stage oracle property (sure screening + oracle refit) |
| Inference | 미정 | Selection consistency 조건 하 corollary; PSI는 future work |
| 실데이터 | 없음 | 1~2개 (gene expression / single-cell RNA-seq) |
| Group 계열의 운명 | 메인 제안 | 본 보고서 메인 비교군에서 제외; 직전 보고서 결과는 reference로 보존 |

본 재설계는 지도교수 의견(group penalty 회피), 직전 시뮬레이션 데이터(Naive Lasso의 screening 충분성), 선행연구 차별화(SelvarMix, SC-FS와의 명시적 차별화), 이론 방어 가능성(post-lasso framework로의 정합성)의 네 측면을 모두 만족시키는 방향으로 정리되었다.
