# 연구 미팅 보고서

**Debiased Sum-to-Zero Lasso Mixture Clustering for High-Dimensional Mean-Heterogeneity Selection**

**미팅 일자:** 2026년 5월 13일

---

## 1. 본 미팅의 개요

본 보고서는 직전 미팅에서 검토된 group lasso 기반 설계(HP-L, HP-AL)를 발전시키는 과정에서 발견된 다음 사항들을 정리하고, 새 메인 설계를 제시한다.

첫째, 직전 시뮬레이션에서 Naive Lasso는 주 분석 구간 전반에서 TPR이 높고 FPR이 낮았으며, 특히 중간·약신호 구간 $a=1.4,1.2$에서 oracle-feature baseline 대비 의미 있는 ARI gap을 보였다. 이 패턴은 변수선택의 실패라기보다, lasso shrinkage로 인한 선택 변수의 mean contrast 과소추정 가능성을 시사한다.

둘째, 이 해석을 따르면 본 연구의 핵심 병목은 screening이 아니라 **post-screening estimation의 shrinkage bias**이다. 따라서 lasso를 final estimator가 아닌 screening estimator로 사용하고, 선택 변수 위에서 unpenalized GMM refit을 수행하는 **두 단계 debiased pipeline**이 자연스러운 해법이 된다.

셋째, 이 설계는 group penalty 구조를 사용하지 않으므로 직전 미팅의 연구 방향(group lasso 미사용)과 정합한다. Sum-to-zero 제약 하에서 $\delta_{\cdot k}=0$와 $\delta_{1k}=\cdots=\delta_{Kk}=0$는 동치이므로, element-wise lasso 결과를 변수 단위 mean contrast로 aggregate하여 $S_0$를 추정할 수 있다. 다만 element-wise lasso의 penalty geometry는 component-level sparsity를 유도하므로, group-lasso와 동일한 penalty 구조로 해석하지 않는다.

본 보고서는 이러한 관찰을 바탕으로 새 메인 방법론 **Debiased Sum-to-Zero Lasso Mixture Clustering (SZL-Refit)** 을 제안하고, 그 모형, 추정 절차, 이론 구조, 시뮬레이션 설계, 실데이터 분석 계획을 정리한다.

---

## 2. 연구 배경 및 문제의식

고차원 비지도 클러스터링에서는 단순히 군집 label을 복원하는 것뿐 아니라, 어떤 변수가 군집 간 평균 차이를 실제로 만들어내는지를 식별하는 것이 중요하다. 본 연구는 이를 **mean-heterogeneity-driving variable selection** 문제로 정의한다.

기존 model-based clustering variable selection 문헌에서는 $\ell_1$-penalized likelihood를 이용해 고차원 Gaussian mixture에서 변수선택을 수행하는 방법이 제안되어 왔다. Pan and Shen (2007)은 공통 대각 공분산 Gaussian mixture에서 $\ell_1$ penalty를 이용해 sparse solution과 variable selection을 얻는 penalized likelihood approach를 제안하였다. Xie, Pan and Shen (2008)은 cluster-specific diagonal covariance와 grouped variables를 고려하는 확장을 제시하였다. Guo, Levina, Michailidis and Zhu (2010)는 cluster pair 단위 변수 식별을 위한 pairwise fusion penalty를 제안하였다. Celeux, Maugis-Rabusseau and Sedki (2018)는 lasso-like ranking과 SRUW 변수 역할 분류를 결합한 SelvarMix 절차를 제시하였다. Liu, Lu, Zhu and Zhao (2023)는 spectral clustering으로 초기 label을 얻은 뒤 $R^2$ 기반 feature selection을 수행하는 SC-FS 절차를 제안하였다.

이 선행연구들은 공통적으로 penalized estimator 자체를 final estimator로 사용하거나(Pan-Shen 계열), 변수 ranking 또는 marginal screening을 통한 두 단계 절차를 따르되 두 번째 단계를 unpenalized GMM MLE로 두지는 않는다(SelvarMix, SC-FS). 또한 mixture 평균을 공통 평균과 군집특이 편차로 분해하여 sum-to-zero 식별성 제약 하에 mean heterogeneity의 effect size를 직접 추적하는 effects-style parameterization은 비지도 mixture 문헌에 명시적으로 도입된 사례를 찾기 어렵다.

본 연구는 두 가지 보완을 통해 위 흐름과 차별화된다. 첫째, mixture 평균에 대한 effects-style parameterization $\mu_j = \mu_0 + \delta_j$ with $\sum_j \delta_{jk} = 0$ 을 도입하여 mean-heterogeneity-driving variable의 정의와 효과 크기 추적을 명시화한다. 둘째, 회귀 문헌의 post-lasso/relaxed lasso 원리(Belloni and Chernozhukov, 2013; Meinshausen, 2007)를 EM 기반 비지도 mixture로 확장하여, lasso를 screening 도구로만 사용하고 unpenalized refit으로 shrinkage bias를 제거하는 debiased two-stage pipeline을 구축한다.

---

## 3. 연구 목표와 가설

본 연구의 목표는 다음과 같다.

**(G1)** Gaussian mixture mean structure를 effects-style로 분해하고, sum-to-zero 식별성 제약 하에 mean-heterogeneity-driving variable set
$$S_0 = \big\lbrace k : \exists j \neq \ell,\ \mu_{jk}^0 \neq \mu_{\ell k}^0 \big\rbrace = \big\lbrace k : \delta_{\cdot k}^0 \neq 0 \big\rbrace$$
을 명시적으로 정의한다.

**(G2)** sum-to-zero constrained lasso를 mean-heterogeneity-driving variable의 screening estimator로 사용하고, 선택 변수 집합 $\hat S$ 위에서 unpenalized GMM MLE refit을 수행하는 **debiased two-stage pipeline (SZL-Refit)** 을 구축한다.

**(G3)** SZL-Refit이 (i) sure screening, (ii) selection size control, (iii) oracle refit equivalence 의 three-step 형태로 정합적인 통계적 보장을 가지며, 이들을 결합한 **two-stage oracle property**를 본 연구의 메인 정리로 제시한다.

**(G4)** SZL-Refit의 shrinkage debiasing 효과를 recovery ratio
$$R_k = \frac{\|\hat\delta_{\cdot k}\|_2}{\|\delta_{\cdot k}^0\|_2}, \qquad k \in S_0$$
와 mean center MSE, heterogeneity effect MSE, classification entropy, ARI 의 다중 지표로 시뮬레이션과 실데이터 양쪽에서 검증한다.

본 연구의 핵심 가설은 다음 세 가지이다.

**(H1) Sure screening 가설.** Sum-to-zero constrained lasso는 sure screening property
$$P(S_0 \subseteq \hat S_\lambda) \to 1$$
을 가진다. 직전 시뮬레이션에서 Naive Lasso의 주 분석 구간 TPR이 1.000으로 관찰된 것이 이 가설을 뒷받침한다.

**(H2) Shrinkage bias 가설.** Naive Lasso의 ARI gap의 주 원인은 selection 실패가 아니라 선택 변수의 mean contrast가 lasso shrinkage로 과소추정된 것이다. 즉, $R_k^{\text{Naive}} < 1$ 이다.

**(H3) Refit recovery 가설.** 선택 변수 위에서 unpenalized refit을 수행하면 mean contrast가 복원되어 $R_k^{\text{SZL-Refit}} \approx 1$ 이며, 그 결과 ARI는 oracle-feature baseline 수준에 근접한다.

---

## 4. 모형

### 4.1 Gaussian mixture mean-shift model

관측치 $X_i = (X_{i1}, \dots, X_{ip})^\top \in \mathbb{R}^p$ 와 잠재군집 $Z_i \in \lbrace 1, \dots, K \rbrace$ 에 대해

$$P(Z_i = j) = \pi_j, \quad j = 1, \dots, K,$$

$$X_i \mid Z_i = j \sim N_p(\mu_j, \Sigma),$$

$$\mu_j = \mu_0 + \delta_j, \qquad \sum_{j=1}^K \delta_{jk} = 0, \quad k = 1, \dots, p.$$

여기서 $\mu_0 \in \mathbb{R}^p$는 sum-to-zero coding 하의 grand mean parameter이고, $\delta_j \in \mathbb{R}^p$는 군집 $j$의 mean deviation vector이다.

선택한 sum-to-zero 제약 하에서

$$\mu_0 = \frac{1}{K} \sum_{j=1}^K \mu_j$$

이므로, $\mu_0$는 component means의 unweighted grand mean이다. 이는 marginal mean $E(X_i) = \mu_0 + \sum_j \pi_j \delta_j$ 와 일반적으로 일치하지 않으며, $\pi_j$ 가 모두 같거나 $\sum_j \pi_j \delta_j = 0$ 인 특수한 경우에만 일치한다.

### 4.2 Mean-heterogeneity-driving variable

변수 $k$가 군집 평균 차이를 유발한다는 것은 다음을 의미한다.

$$\exists j \neq \ell \text{ such that } \mu_{jk}^0 \neq \mu_{\ell k}^0.$$

따라서

$$S_0 = \big\lbrace k : \exists j \neq \ell,\ \mu_{jk}^0 \neq \mu_{\ell k}^0 \big\rbrace = \big\lbrace k : \|\delta_{\cdot k}^0\|_2 > 0 \big\rbrace.$$

Sum-to-zero 제약 하에서는

$$\delta_{\cdot k} = 0 \iff \delta_{1k} = \cdots = \delta_{Kk} = 0$$

이 성립한다. 따라서 element-wise lasso 결과를 변수 단위로 aggregate하면 $S_0$ 를 추정할 수 있다. Element-wise lasso와 group lasso는 penalty geometry가 다르므로 본 연구에서는 element-wise lasso의 component-level sparsity 결과를 variable-level mean contrast로 aggregate하여 $S_0$ 를 추정한다.

### 4.3 공분산 구조

본 연구의 1차 시뮬레이션에서는 공분산을

$$\Sigma = \mathrm{diag}(\sigma_1^2, \dots, \sigma_p^2) \quad \text{또는} \quad \Sigma = I_p$$

로 둔다. 이 가정 하에서는 군집이 주어졌을 때 각 좌표가 조건부 독립이므로 mean heterogeneity selection 문제를 가장 명료하게 다룰 수 있다. Cluster-specific 또는 비대각 공분산으로의 확장은 본 연구의 후속 작업으로 둔다.

---

## 5. 제안 방법: Debiased Sum-to-Zero Lasso Mixture Clustering (SZL-Refit)

제안 방법은 다음 네 단계로 구성된다.

- **Stage 1.** Sum-to-zero constrained lasso screening (penalized EM)
- **Stage 1.5.** Variable-level aggregation (standardized max-pairwise contrast)
- **Stage 2.** Unpenalized GMM refit on the selected support
- **Stage 3.** Refit-likelihood EBIC tuning of the regularization parameter

### 5.1 Stage 1: Sum-to-zero lasso screening

$\ell_n(\Theta) = n^{-1} \sum_{i=1}^n \log\big[\sum_{j=1}^K \pi_j \phi_p(X_i; \mu_0 + \delta_j, \Sigma)\big]$ 를 average observed log-likelihood로 두고, 각 $\lambda$에 대해 다음 penalized objective를 최대화한다.

$$\hat\Theta_\lambda^{\text{SZL}} = \arg\max_\Theta \left[ \ell_n(\Theta) - \lambda \sum_{k=1}^p \sum_{j=1}^K \frac{|\delta_{jk}|}{\hat\sigma_k} \right]$$

subject to

$$\sum_{j=1}^K \delta_{jk} = 0, \quad k = 1, \dots, p.$$

여기서 $\hat\sigma_k$로 나누는 항은 변수별 scale normalization이며, $\Sigma = I_p$ 로 고정한 경우 생략 가능하다.

본 단계에서 얻은 $\hat\Theta_\lambda^{\text{SZL}}$ 는 **screening estimator**로 정의된다. 즉 선택집합 $\hat S_\lambda$ 의 도출에만 사용되며, 본 단계의 모수 추정치는 final estimator로 보고하지 않는다.

### 5.2 계산상 sum-to-zero 제약 처리

$\mu_j = \mu_0 + \delta_j$ 만으로는 분해가 식별되지 않으므로 sum-to-zero 제약은 필수적이다. 본 연구에서는 element-wise $\ell_1$ penalty와 sum-to-zero 제약을 결합하기 위해 **direct constrained update** 방식을 메인 알고리즘으로 사용한다.

EM의 M-step에서 변수 $k$별로 다음 constrained weighted lasso subproblem을 푼다.

$$\min_{\delta_{1k}, \dots, \delta_{Kk}} \sum_{j=1}^K w_{jk} (\delta_{jk} - z_{jk})^2 + \lambda_k \sum_{j=1}^K |\delta_{jk}|$$

subject to $\sum_j \delta_{jk} = 0$. 여기서 $w_{jk}$, $z_{jk}$는 E-step에서 계산되는 weight와 weighted target이고, $\lambda_k = \lambda / \hat\sigma_k$이다. 본 problem은 $K$차원의 작은 convex problem으로, Lagrangian-augmented coordinate descent 또는 표준 convex solver로 안정적으로 풀린다.

직전 보고서의 group penalty 설계에서 사용된 $Q$-basis 재파라미터화 $\delta_{\cdot k} = Q\alpha_k$ ($Q \in \mathbb{R}^{K \times (K-1)}$, $Q^\top \mathbf{1}_K = 0$, $Q^\top Q = I_{K-1}$) 는 group penalty $\|\delta_{\cdot k}\|_2 = \|\alpha_k\|_2$ 가 재파라미터화 후에도 보존되기 때문에 자연스러웠다. 그러나 element-wise $\ell_1$ penalty의 경우

$$\sum_{j=1}^K |\delta_{jk}| = \sum_{j=1}^K \left| \sum_{m=1}^{K-1} Q_{jm} \alpha_{km} \right|$$

이 되어 일반적으로 $\alpha_k$ 좌표별 separable하지 않다. 따라서 본 연구의 element-wise penalty 설계와 정합되는 알고리즘은 direct constrained update이며, $Q$-basis는 (필요 시) 수치 안정화 목적의 보조 도구로만 사용한다.

### 5.3 Stage 1.5: Variable-level aggregation

Lasso는 개별 $\delta_{jk}$ 에 작용하나, 본 연구의 선택 대상은 변수 $k$ 이다. 따라서 lasso 결과를 변수 단위로 aggregate한다. 본 연구에서는 **standardized max-pairwise contrast** 기준을 사용한다.

$$\hat S_\lambda = \Big\lbrace k : \max_{j < \ell} \frac{|\hat\mu_{jk, \lambda}^{\text{SZL}} - \hat\mu_{\ell k, \lambda}^{\text{SZL}}|}{\hat\sigma_{k, \lambda}} > \tau_{\text{num}} \Big\rbrace.$$

여기서 $\tau_{\text{num}} = 10^{-4}$는 statistical tuning parameter가 아니라 수치적 파편화 제거를 위한 numerical tolerance이며, 직전 보고서의 동일 threshold를 그대로 유지한다.

이 기준은 "변수 $k$에서 군집 평균이 실제로 갈라지는가"라는 본 연구의 정의 ($S_0$의 정의)와 일대일로 대응한다. 대안으로 group norm 기준

$$\hat S_\lambda = \Big\lbrace k : \|\hat\delta_{\cdot k, \lambda}^{\text{SZL}}\|_2 > \tau_{\text{num}} \Big\rbrace$$

을 supplement에서 보고한다.

### 5.4 Stage 2: Unpenalized post-selection refit

각 $\lambda$가 만든 $\hat S_\lambda$ 를 고정하고, penalty 없이 GMM likelihood를 다시 최대화한다.

$$\hat\Theta_\lambda^{\text{refit}} = \arg\max_{\Theta:\ \delta_{\cdot k} = 0,\ k \notin \hat S_\lambda} \ell_n(\Theta).$$

즉, $k \in \hat S_\lambda$ 인 변수는 component-specific mean을 자유롭게 추정하고, $k \notin \hat S_\lambda$ 인 변수는 $\mu_{1k} = \cdots = \mu_{Kk}$ 를 강제한다. 본 단계에서 lasso shrinkage가 제거되며, 본 단계가 SZL-Refit pipeline의 핵심이다.

### 5.5 Stage 3: Refit-likelihood EBIC tuning

EBIC는 lasso fit이 아니라 refit estimator 기준으로 계산한다.

$$\text{EBIC}_\alpha(\lambda) = -2 n \cdot \ell_n(\hat\Theta_\lambda^{\text{refit}}) + \log n \cdot \text{df}(\hat S_\lambda) + 2\alpha |\hat S_\lambda| \log p, \quad \alpha \in [0, 1].$$

$\lambda$ 별 모델 비교에서 mixing proportions $(K-1)$, common means $p$, diagonal variances $p$ 항은 모든 후보에 공통인 nuisance dimension이다. 따라서 모델 비교에서 작용하는 effective complexity는

$$\text{df}_{\text{eff}}(\hat S_\lambda) = (K-1) |\hat S_\lambda|$$

이고, EBIC의 model-comparison-relevant 부분은

$$-2 n \cdot \ell_n(\hat\Theta_\lambda^{\text{refit}}) + \log n \cdot (K-1) |\hat S_\lambda| + 2\alpha |\hat S_\lambda| \log p$$

로 환원된다. 본 연구에서는 이 effective form을 사용한다.

최종 선택은 다음과 같다.

$$\hat\lambda = \arg\min_\lambda \text{EBIC}_\alpha(\lambda), \qquad \hat S = \hat S_{\hat\lambda}, \qquad \hat\Theta = \hat\Theta_{\hat\lambda}^{\text{refit}}.$$

$\alpha$의 기본값은 $0.5$로 두며, $\alpha \in \lbrace 0, 0.5, 1 \rbrace$ 의 sensitivity analysis를 함께 보고한다 (Chen and Chen, 2008).

---

## 6. 보조 확장: Adaptive SZL-Refit (ASZL-Refit)

본 연구는 SZL-Refit을 메인 방법으로 두되, adaptive lasso 기반 변형 **ASZL-Refit** 을 보조 확장으로 함께 제시한다.

### 6.1 Adaptive Stage 1

$$\hat\Theta_\lambda^{\text{ASZL}} = \arg\max_\Theta \left[ \ell_n(\Theta) - \lambda \sum_{k=1}^p \sum_{j=1}^K a_{jk} \frac{|\delta_{jk}|}{\hat\sigma_k} \right]$$

subject to $\sum_{j=1}^K \delta_{jk} = 0$, where

$$a_{jk} = (|\tilde\delta_{jk}| + \varepsilon_n)^{-\gamma}.$$

Pilot $\tilde\delta_{jk}$ 는 plain SZL screening 결과를 사용하여 별도의 추가 모형이 도입되지 않도록 한다. 결과적으로 ASZL-Refit은 (SZL screening) → (ASZL screening) → (unpenalized refit) 의 3-stage 구조를 가진다.

### 6.2 가중치 형태에 대한 주의

본 연구에서 adaptive weight는 **component-wise** 형태 $a_{jk}$ 로 한정한다. Variable-level 형태

$$v_k \sum_{j=1}^K |\delta_{jk}|$$

는 엄밀히 group lasso는 아니나, 변수 $k$ 전체에 동일 가중치를 부여하므로 group-like behavior로 해석될 여지가 있다. 본 연구는 group penalty 구조를 사용하지 않는 방향을 따르므로 variable-level adaptive weight는 채택하지 않는다.

### 6.3 보조 확장의 역할

ASZL-Refit은 plain SZL-Refit이 oracle gap을 충분히 메우지 못하는 저신호 setting에서의 robustness variant 역할을 가진다. 그 효과는 후술하는 시뮬레이션 (8장)에서 직접 비교하여 보고한다.

---

## 7. 이론

본 연구의 이론은 다음 세 정리로 구성된다.

### 7.1 가정

**(A1) Identifiability and separation.**
$$\min_{j \neq \ell} \|\mu_j^0 - \mu_\ell^0\|_{\Sigma^{-1}} \geq c_n.$$
모수 거리는 label permutation에 대해 정의된다.

**(A2) Sparsity and dimensionality.**
$$s_0 = |S_0| \ll n, \qquad s_0 \log p = o(n).$$

**(A3) Beta-min condition.**
변수 $k$의 standardized mean heterogeneity strength를 sum-to-zero parameterization과 정합되도록 다음과 같이 정의한다.

$$B_k^0 = \frac{1}{\sigma_k^2} \cdot \frac{1}{K} \sum_{j=1}^K (\mu_{jk}^0 - \mu_{0k})^2 = \frac{\|\delta_{\cdot k}^0\|_2^2}{K \sigma_k^2},$$

$$B_{\min} = \min_{k \in S_0} B_k^0.$$

Sure screening rate를 위해 $n B_{\min} \gg \log p$ 수준의 lower bound를 가정한다.

**(A4) Local EM condition.**
혼합모형의 non-convexity를 고려하여, 좋은 initialization 근방에서 EM이 local contraction을 만족하거나, population likelihood가 true parameter 근방에서 locally identifiable하다고 가정한다.

### 7.2 Theorem 1. Sure screening

가정 (A1)–(A4) 와 적절한 $\lambda_n$ sequence 하에서

$$P(S_0 \subseteq \hat S_\lambda) \to 1.$$

본 결과는 Stage 1 screening이 참 mean-heterogeneity-driving variables를 빠뜨리지 않음을 의미한다. Refit이 빠진 변수를 복구할 수 없으므로, 본 정리는 SZL-Refit pipeline의 가장 중요한 보장이다.

### 7.3 Theorem 2. Selection size control

EBIC-tuned $\hat\lambda$에 대해

$$|\hat S_{\hat\lambda}| = O_p(s_0).$$

본 결과는 Stage 2 refit의 variance inflation을 제어하기 위해 필요하다.

### 7.4 Theorem 3. Oracle refit equivalence

Oracle estimator를

$$\hat\Theta^{\text{oracle}} = \arg\max_{\Theta:\ \delta_{\cdot k} = 0,\ k \notin S_0} \ell_n(\Theta),$$

제안 refit estimator를

$$\hat\Theta^{\text{refit}} = \arg\max_{\Theta:\ \delta_{\cdot k} = 0,\ k \notin \hat S} \ell_n(\Theta)$$

로 두자. Theorems 1–2 하에서, label permutation에 대해 정의된 모수 거리 $d(\cdot, \cdot)$ 에 대해

$$d(\hat\Theta^{\text{refit}}, \Theta^0) = O_p\left(\sqrt{\frac{(K-1) s_0}{n}}\right)$$

이며, common means와 variances 등 nuisance parameter는 표준 $\sqrt{p/n}$ rate로 별도로 수렴한다. 더 강하게 $P(\hat S = S_0) \to 1$ 이 성립하면

$$d(\hat\Theta^{\text{refit}}, \hat\Theta^{\text{oracle}}) = o_p(n^{-1/2})$$

이며, 이는 oracle MLE와 점근적으로 동등함을 의미한다.

본 정리들의 결합을 본 연구에서는 **two-stage oracle property**로 제시한다.

### 7.5 추론 측면의 범위

선택 이후 finite-sample valid post-selection inference는 본 연구의 메인 contribution에 포함하지 않는다. Selection consistency 조건 하 oracle MLE의 점근 정규성은 corollary 형태로 인용하며, finite-sample 신뢰구간 및 selective inference framework (Berk et al., 2013; Lee et al., 2016) 로의 확장은 후속 연구로 둔다.

---

## 8. 시뮬레이션 설계

본 보고서의 시뮬레이션은 다음 두 단계로 구성된다.

(1) **핵심 검증 실험.** 본 연구의 메인 가설 (H1)–(H3)을 직접 검증하는 최소 비교 실험으로, Naive Lasso, SZL-Refit, ASZL-Refit, Oracle 의 ARI와 recovery ratio $R_k$ 를 비교한다.

(2) **전체 벤치마크 실험.** 핵심 검증 결과 위에 전통 baseline과 기존 sparse clustering 비교군을 추가하여 종합 성능을 보고한다.

### 8.1 핵심 검증 실험 (메인 가설 검증)

**Setting.** $n = 300$, $p \in \lbrace 100, 300 \rbrace$, $a \in \lbrace 1.6, 1.4, 1.2 \rbrace$, 반복수 $R \geq 100$.

**비교 방법.**

| 구분 | 방법 | 본 실험에서의 역할 |
|---|---|---|
| Reference (no refit) | Naive Lasso | (H2) shrinkage bias 확인 |
| Main | SZL-Refit | (H1)+(H3) sure screening + refit recovery 확인 |
| Auxiliary | ASZL-Refit | adaptive weighting의 추가효과 확인 |
| Oracle benchmark | Oracle-feature GMM | oracle gap의 상한 reference |
| Oracle benchmark | True-parameter oracle | 이상적 reference |

**가설별 결정 기준.**

| 관찰 패턴 | 가설 평가 |
|---|---|
| SZL-Refit이 Naive Lasso 대비 ARI gap을 대부분 회복하고 ASZL-Refit과 유사 | (H1)–(H3) 모두 지지. Plain SZL-Refit을 메인으로 확정. |
| SZL-Refit이 부분 회복, ASZL-Refit이 명확히 우수 | (H3) 부분 지지. ASZL-Refit을 co-main으로 승격. |
| 두 refit 모두 ARI gap을 회복하지 못함 | (H2) 기각. 설계 재검토 필요. |

직전 시뮬레이션에서 Naive Lasso의 주 분석 구간 TPR이 1.000, FPR이 0.001–0.019, $|\hat S| \approx q$ (정답 변수 수에 거의 일치) 로 관찰된 점은 (H1)을 강하게 뒷받침한다. 동일 구간에서 ARI gap이 0.15–0.20 으로 관찰된 점은 (H2)와 정합적이다. 본 실험에서 (H3) 의 양적 크기가 확정된다.

### 8.2 전체 벤치마크 실험

**비교군.**

| 그룹 | 방법 |
|---|---|
| Traditional baselines | $K$-means, PCA + $K$-means, Unpenalized GMM |
| Existing sparse clustering | Sparse $K$-means, SelvarMix, SC-FS |
| Existing penalized GMM | Naive Lasso (no refit) |
| Proposed (main) | SZL-Refit |
| Proposed (auxiliary) | ASZL-Refit |
| Oracle | Oracle-feature GMM, True-parameter oracle |

직전 보고서의 group lasso 계열 결과 (HP-L, HP-AL) 는 본 보고서의 메인 비교군에 포함하지 않는다.

**주 분석 시나리오.** 직전 보고서의 시나리오를 그대로 유지한다.

- $n = 300$, $p \in \lbrace 20, 100, 300 \rbrace$, $q \in \lbrace 3, 5, 5 \rbrace$, $a \in \lbrace 1.6, 1.4, 1.2 \rbrace$.

**보조 시나리오 (supplement).**

- 한계 신호: $a \in \lbrace 1.0, 0.8 \rbrace$ — refit이 회복할 수 없는 영역의 명시.
- Unequal mixing: $\pi_j$ 비대칭.
- Correlated predictors: $\mathrm{corr}(X_k, X_\ell) \neq 0$.
- Cluster-specific covariance ($\Sigma_j \neq \Sigma$) 는 본 모형 범위 밖이며 robustness 검증용으로만 보고.

### 8.3 보고 지표

ARI, TPR, FPR, $|\hat S|$ 외에 다음 네 지표를 함께 보고한다.

**(M1) Mean center MSE.**
$$\text{MSE}_\mu = \min_{\rho \in \mathcal{P}_K} \frac{1}{Kp} \sum_{j=1}^K \|\hat\mu_{\rho(j)} - \mu_j^0\|_2^2.$$

**(M2) Mean heterogeneity effect MSE.**
$$\text{MSE}_{\Delta, S} = \frac{1}{K |S_0|} \sum_{k \in S_0} \|\hat\delta_{\cdot k} - \delta_{\cdot k}^0\|_2^2.$$

**(M3) Recovery ratio (본 연구의 핵심 지표).**
$$R_k = \frac{\|\hat\delta_{\cdot k}\|_2}{\|\delta_{\cdot k}^0\|_2}, \qquad k \in S_0.$$

**(M4) Classification entropy.**
$$\text{Entropy} = -\frac{1}{n} \sum_{i=1}^n \sum_{j=1}^K \hat r_{ij} \log \hat r_{ij}.$$

본 지표들은 직전 보고서의 추가 분석 계획에서 이미 제안된 것을 그대로 사용한다.

### 8.4 핵심 visualization

본 연구의 핵심 figure는 **recovery ratio $R_k$ 의 분포 비교**이다. 두 panel 구성으로 다음을 시각화한다.

- Panel A. Naive Lasso (no refit): $R_k < 1$ 방향의 분포 — shrinkage bias 의 직접 가시화.
- Panel B. SZL-Refit: $R_k \approx 1$ 방향으로 회복 — refit의 debiasing 효과 가시화.

본 figure는 본 연구의 핵심 메시지 ("lasso는 변수를 찾으나 effect size를 줄이고, refit이 이를 복원한다")를 직접 전달하는 역할을 한다.

---
## 8.5 예비 시뮬레이션 결과: $p=100, a=1.2$ (Pilot)

본 섹션은 전체 시뮬레이션에 앞서, 연구의 핵심 가설을 검증하기 위해 수행된 $p=100, a=1.2$ 환경에서의 pilot sanity check 결과를 정리한다.

### 8.5.1 실험 목적
본 예비 실험의 목적은 본 연구의 핵심 가설인 다음 질문에 답하는 것이다.

> **"Screening failure가 아니라 shrinkage bias가 ARI gap의 주 원인인가?"**

구체적으로 다음 세 가지 가설을 확인한다.
*   **(H1) Sure screening:** Sum-to-zero lasso는 true active variables를 빠뜨리지 않고 찾아낼 수 있는가?
*   **(H2) Shrinkage bias:** Naive Lasso는 변수를 찾더라도 lasso shrinkage로 인해 mean contrast를 과소추정하는가? ($R_k < 1$)
*   **(H3) Refit recovery:** 선택 변수 위에서 unpenalized refit을 수행하면 $R_k$와 ARI가 oracle 수준으로 회복되는가?

### 8.5.2 시뮬레이션 세팅
*   **Data Structure:** $n=300, p=100, K=3, q=5$
*   **Signal Strength:** $a=1.2$ (중저신호 구간), 반복수 $R=3$
*   **Active Set:** $S_0 = \{1, 2, 3, 4, 5\}$
*   **Pattern:** 각 active variable에 대해 군집 평균 패턴을 $(-a, 0, a)$로 설정.
*   **Covariance:** $\Sigma = I_p$ (공분산 추정 영향을 제거하고 mean contrast 효과에 집중)
*   **Tuning:** EBIC (Refit 방법의 경우 refit estimator의 likelihood를 기준으로 계산)

### 8.5.3 비교 모형
이번 pilot에서는 방법론의 효과를 다각도로 분석하기 위해 아래 모델들을 비교하였다.

| 구분 | 방법 | 설명 |
| :--- | :--- | :--- |
| **Traditional** | K-means | 전체 변수 기반 기본 clustering |
| **Traditional** | PCA + K-means | PCA 차원축소 후 K-means |
| **Model-based** | Unpenalized GMM | 전체 변수에서 비벌점 GMM |
| **Sparse Clustering** | Sparse K-means | `sparcl` 패키지 기반, EBIC-tuned |
| **Spectral/Screening** | SCFS | `SC-FS` 패키지 기반 |
| **Model-based VS** | SelvarMix (SS) | `SelvarMix`의 strict selected clustering set |
| **Model-based VS** | SelvarMix (nonW) | SelvarMix에서 noise ($W$)로 분류되지 않은 변수 전체 |
| **Ablation** | Naive Lasso (at refit $\lambda$) | SZL-Refit과 동일 support에서 refit을 제거한 결과 |
| **Penalized GMM** | Naive Lasso (self-tuned) | Lasso likelihood 기반 EBIC로 선택한 standalone lasso |
| **Proposed** | **SZL-Refit** | **Plain sum-to-zero lasso screening + unpenalized refit** |
| **Auxiliary** | ASZL-Refit | Adaptive SZL screening + unpenalized refit |
| **Oracle** | Oracle-feature GMM | 정답 변수 집합 $S_0$를 알고 수행한 GMM refit |
| **Oracle** | True-parameter oracle | True parameter를 이용한 이상적 classification 결과 |

---

### 8.5.4 핵심 검증 결과 (Main Hypothesis Testing)

**Table 1. 핵심 방법론 비교 결과**

| Method | ARI | TPR | FPR | $\hat S$ | $\text{MSE}_{\mu}$ | $\text{MSE}_{\Delta,S}$ | $R_{\text{mean}}$ | Entropy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Naive Lasso at refit $\lambda$ | 0.467 | 1.000 | 0.000 | 5.000 | 0.0170 | 0.2672 | 0.478 | 0.200 |
| Naive Lasso self-tuned | 0.506 | 1.000 | 0.014 | 6.333 | 0.0081 | 0.0888 | 0.700 | 0.289 |
| **SZL-Refit** | **0.663** | **1.000** | **0.000** | **5.000** | **0.0040** | **0.0082** | **0.967** | **0.287** |
| ASZL-Refit | 0.665 | 1.000 | 0.000 | 5.000 | 0.0039 | 0.0076 | 0.968 | 0.288 |
| Oracle-feature GMM | 0.633 | 1.000 | 0.000 | 5.000 | 0.0041 | 0.0076 | 1.007 | 0.276 |
| True-parameter oracle | 0.668 | 1.000 | 0.000 | 5.000 | 0.0000 | 0.0000 | 1.000 | 0.288 |

*   **Shrinkage 관찰:** Naive Lasso는 $\text{TPR}=1, \hat S=5$로 변수를 정확히 찾았으나, $R_{\text{mean}}=0.478$로 효과 크기를 절반 이하로 줄여 ARI가 0.467에 머물렀다.
*   **Refit 효과:** 동일 support에서 refit을 수행한 **SZL-Refit은 $R_{\text{mean}}=0.967$, ARI=0.663**으로 Oracle 수준(0.668)까지 성능을 회복했다. 이는 \(R=3\) pilot 수준에서 가설 (H2)와 (H3)을 지지하는 예비 근거로 해석된다. 반복수를 늘린 \(R=50\sim100\) 실험에서 동일 패턴이 유지되는지 확인할 예정이다.

<img width="988" height="690" alt="image" src="https://github.com/user-attachments/assets/cfb64f24-a313-4c2f-a7df-cbeb04d6a121" />
Figure 1. Mean-heterogeneity effect의 회복 정도. 
Naive Lasso는 true support를 찾았음에도 \(R_{\text{mean}}<1\)로 mean contrast를 과소추정하는 반면, SZL-Refit은 \(R_{\text{mean}}\approx 1\) 수준으로 effect size를 복원한다. 이는 Naive Lasso의 ARI 손실이 support recovery 실패가 아니라 shrinkage bias에서 기인한다는 해석을 뒷받침한다.

<img width="988" height="690" alt="image" src="https://github.com/user-attachments/assets/59512033-6b5d-4b13-9c7b-4f93d2e75b17" />
Figure 2. ARI 기준 군집 성능 비교. 
Post-selection refit을 수행하면 lasso screening estimator의 ARI가 크게 개선되며 oracle reference와의 gap이 대부분 줄어든다. 또한 SZL-Refit과 ASZL-Refit의 성능이 거의 유사하므로, 현재 pilot setting에서는 adaptive weighting보다 refit에 의한 debiasing 효과가 주요 개선 요인으로 해석된다.

<img width="988" height="690" alt="image" src="https://github.com/user-attachments/assets/3b5d979f-af74-4412-8de0-e87e3f83541e" />
Figure 3. Mean-shift effect 추정오차 비교. 
Unpenalized refit 이후 \(\mathrm{MSE}_{\Delta,S}\)가 크게 감소한다. 이는 refit 단계가 단순히 군집 할당 성능만 개선하는 것이 아니라, 선택된 변수의 cluster-specific mean contrast 자체를 oracle 수준에 가깝게 복원함을 보여준다.

---

### 8.5.5 전체 Benchmark Pilot 결과

**Table 2. 전체 Benchmark 성능 비교**

| Group | Method | ARI | TPR | FPR | $\hat S$ | $R_{\text{mean}}$ | $\text{MSE}_{\Delta,S}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Traditional | K-means | 0.502 | NA | NA | NA | 0.954 | 0.0362 |
| Traditional | PCA + K-means | 0.489 | NA | NA | NA | 0.962 | 0.0383 |
| Model-based | Unpenalized GMM | 0.501 | NA | NA | NA | 0.945 | 0.0319 |
| Sparse Clustering | Sparse K-means | 0.622 | 1.000 | 0.000 | 5.000 | 1.028 | 0.0112 |
| Spectral/Screen | SCFS | 0.561 | 1.000 | 0.007 | 5.667 | 0.994 | 0.0186 |
| MB Variable Sel | SelvarMix (SS) | 0.396 | 0.800 | 0.000 | 4.000 | 0.828 | 0.2981 |
| MB Variable Sel | SelvarMix (nonW) | 0.396 | 1.000 | 0.042 | 9.000 | 0.988 | 0.1234 |
| MB Variable Sel | SelvarMix (nonW + Refit) | 0.396 | 1.000 | 0.042 | 9.000 | 0.981 | 0.0095 |
| Proposed | **SZL-Refit** | **0.663** | **1.000** | **0.000** | **5.000** | **0.967** | **0.0082** |
| Oracle | True-parameter oracle | 0.668 | 1.000 | 0.000 | 5.000 | 1.000 | 0.0000 |

<img width="1478" height="610" alt="image" src="https://github.com/user-attachments/assets/1f8cd0fc-a18e-4edb-b1ee-768d8243ff0c" />
Figure 4. Pilot study에서의 전체 benchmark ARI 비교
SZL-Refit은 oracle reference에 가까운 ARI를 보이며, Sparse K-means 및 SCFS와 같은 외부 benchmark와도 경쟁적인 성능을 보인다. 단, 현재 결과는 \(R=3\) pilot이므로 방법 간 순위 자체보다는 성능 경향을 확인하기 위한 예비 결과로 해석한다.

### 8.5.6 결과 해석

1.  **Lasso의 병목은 변수선택이 아닌 Shrinkage Bias:** Naive Lasso는 변수($S_0$)는 정확히 찾아냈으나, 효과 크기(mean contrast)를 과도하게 축소하여 ARI 손실을 발생시켰다. (H2 지지)
2.  **SZL-Refit의 Oracle 도달:** Unpenalized refit을 통해 mean contrast를 복원($R_{\text{mean}} \approx 1$)함으로써 Oracle ARI와 통계적으로 동등한 수준에 도달했다. (H3 지지)
3.  **ASZL-Refit의 효율성:** 현재 $p=100, a=1.2$ 환경에서는 Adaptive Lasso의 이득이 미미했다. 따라서 **Plain SZL-Refit을 메인 방법론으로 유지**하고 ASZL-Refit은 확장형으로 둔다.
4.  **SelvarMix의 역할 재정의:** SelvarMix는 변수를 역할(S, R, U, W)별로 나누기 때문에, 단순 선택 변수 집합($SS$)과 비교하기보다 `nonW` 집합과 `Refit`을 결합하여 별도의 role-based benchmark로 보고하는 것이 공정하다.

---

### 8.5.7 Pilot 결과의 한계 및 향후 작업
*   **Monte Carlo 변동성:** $R=3$의 결과이므로 통계적 유의성 확보를 위해 $R \geq 100$ 확장이 필요하다.
*   **고차원 약신호 환경:** $p=300, a=1.2$ 등 더 어려운 조건에서의 강건성을 추가 검증해야 한다.
*   **공분산 구조:** $\Sigma = I_p$ 가정을 해제하고 대각 공분산 추정 시의 성능을 확인해야 한다.

### 8.5.8 현재 pilot 결과에 따른 의사결정

| 쟁점 | 현재 결과 | 결정 |
| :--- | :--- | :--- |
| **Naive Lasso의 문제** | Support는 정확히 찾지만 $R_{\text{mean}}=0.478$ | **Shrinkage bias가 핵심 병목**임을 명시 |
| **Refit 효과** | ARI가 Oracle 수준으로 회복됨 | **Refit 단계 유지 및 강조** |
| **Plain vs Adaptive** | 성능 차이가 매우 작음 | **SZL-Refit(Main), ASZL(Secondary)** |
| **외부 Benchmark** | Sparse K-means 등이 의미 있는 성능 보임 | 최종 시뮬레이션에 포함 유지 |
| **실험 규모** | $R=3$ Pilot | **$R=100$으로 확장 수행** |

---

### 8.5.9 다음 시뮬레이션 계획
1.  **Step 1:** 현재 세팅 반복수 확장 ($p=100, a=1.2, R=50 \sim 100$)
2.  **Step 2:** 초고차원 약신호 세팅 ($p=300, a=1.2$) - 본 연구의 핵심 타겟 구간
3.  **Step 3:** 전체 Grid 확장 및 Robustness 테스트 (Unequal mixing, Correlated predictors)

---

## 참고문헌 (주요 인용)

- Belloni, A. and Chernozhukov, V. (2013). Least squares after model selection in high-dimensional sparse models. *Bernoulli*, 19(2), 521–547.
- Berk, R., Brown, L., Buja, A., Zhang, K. and Zhao, L. (2013). Valid post-selection inference. *Annals of Statistics*, 41(2), 802–837.
- Celeux, G., Maugis-Rabusseau, C. and Sedki, M. (2018). Variable selection in model-based clustering and discriminant analysis with a regularization approach. *Advances in Data Analysis and Classification*, 13(1), 259–278.
- Chen, J. and Chen, Z. (2008). Extended Bayesian information criteria for model selection with large model spaces. *Biometrika*, 95(3), 759–771.
- Guo, J., Levina, E., Michailidis, G. and Zhu, J. (2010). Pairwise variable selection for high-dimensional model-based clustering. *Biometrics*, 66(3), 793–804.
- Lee, J. D., Sun, D. L., Sun, Y. and Taylor, J. E. (2016). Exact post-selection inference, with application to the lasso. *Annals of Statistics*, 44(3), 907–927.
- Liu, T., Lu, Y., Zhu, B. and Zhao, H. (2023). Clustering high-dimensional data via feature selection. *Biometrics*, 79(2), 940–950.
- Meinshausen, N. (2007). Relaxed Lasso. *Computational Statistics and Data Analysis*, 52(1), 374–393.
- Pan, W. and Shen, X. (2007). Penalized model-based clustering with application to variable selection. *Journal of Machine Learning Research*, 8, 1145–1164.
- Witten, D. M. and Tibshirani, R. (2010). A framework for feature selection in clustering. *Journal of the American Statistical Association*, 105(490), 713–726.
- Xie, B., Pan, W. and Shen, X. (2008). Penalized model-based clustering with cluster-specific diagonal covariance matrices and grouped variables. *Electronic Journal of Statistics*, 2, 168–212.
- Zou, H. (2006). The adaptive lasso and its oracle properties. *Journal of the American Statistical Association*, 101(476), 1418–1429.
