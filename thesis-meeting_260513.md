# 연구 미팅 보고서

**Debiased Sum-to-Zero Lasso Mixture Clustering for High-Dimensional Mean-Heterogeneity Selection**

**미팅 일자:** 2026년 5월 13일

**미팅 목적:** group-lasso 계열을 현재 메인 설계에서 제외하고, sum-to-zero constrained lasso를 screening 도구로 사용하는 two-stage debiased mixture clustering framework를 검토한다.

---

## 0. 미팅 핵심 의사결정 사항

이번 미팅에서 확인하고 싶은 핵심은 다음 세 가지이다.

**1) 메인 방법을 Plain SZL-Refit으로 둘 것인가?**

$$\text{Sum-to-zero lasso screening} \longrightarrow \text{unpenalized GMM refit} \longrightarrow \text{refit-likelihood EBIC tuning}$$

**2) Adaptive SZL-Refit은 secondary extension으로 둘 것인가?**

$$\lambda \sum_{k=1}^p \sum_{j=1}^K a_{jk} |\delta_{jk}|$$

형태의 component-wise adaptive lasso는 가능하나, pilot estimator 안정성 문제가 따라온다. 따라서 메인이 아니라 보조 확장으로 검토한다.

**3) Critical sanity check를 먼저 수행한 뒤 최종 메인 방법을 확정할 것인가?**

기존 결과는 Naive Lasso가 변수선택은 잘하지만 ARI가 낮은 패턴을 보인다. 즉 $p=100, a=1.2$, $p=300, a=1.2$ 등 고차원 약신호 구간에서도 TPR이 거의 1이고 FPR이 매우 낮으나 ARI는 oracle-feature baseline보다 낮았다. 이는 screening 실패보다 lasso shrinkage로 인한 mean contrast 과소추정 가능성을 시사한다. 따라서 먼저 "Naive Lasso + Refit"이 oracle gap을 얼마나 회복하는지 확인한 뒤 메인을 확정한다.

---

## 1. 연구 배경 및 문제의식

고차원 비지도 클러스터링에서는 단순히 군집 label을 복원하는 것뿐 아니라, 어떤 변수가 군집 간 평균 차이를 실제로 만들어내는지를 식별하는 것이 중요하다. 본 연구에서는 이를 mean-heterogeneity-driving variable selection 문제로 정의한다.

기존 model-based clustering variable selection 문헌에서는 $\ell_1$-penalized likelihood를 이용해 고차원 Gaussian mixture에서 변수선택을 수행하는 방법이 제안되어 왔다. Pan and Shen은 공통 대각 공분산 Gaussian mixture에서 $\ell_1$ penalty를 이용해 sparse solution과 variable selection을 얻는 penalized likelihood approach를 제안하였다. 이후 Xie, Pan and Shen은 cluster-specific diagonal covariance와 grouped variables를 고려하는 penalized model-based clustering 방법을 제안하였다. SelvarMix는 model-based clustering과 discriminant analysis에서 regularization 기반 변수선택을 수행하는 R package로, lasso-like ranking과 변수 역할 정의 절차를 사용한다. 또한 SC-FS는 spectral clustering으로 초기 label을 얻은 뒤 label 설명력 $R^2$가 큰 feature를 선택하고 재클러스터링하는 feature selection clustering 절차이다.

본 연구의 차별점은 기존 penalized mixture estimator 자체를 최종 estimator로 쓰지 않는다는 점이다. Lasso는 변수 후보를 찾는 screening estimator로만 사용하고, 선택된 변수 위에서 unpenalized GMM refit을 수행하여 shrinkage bias를 줄인다. 이 아이디어는 회귀 문헌의 post-lasso principle과 연결된다. Belloni and Chernozhukov는 first-step penalized estimator가 선택한 model 위에서 least squares refit을 수행하는 post-lasso estimator를 분석했고, post-lasso가 lasso보다 bias 측면에서 유리할 수 있음을 보였다.

본 연구는 이 post-selection refit 원리를 비지도 Gaussian mixture clustering의 mean-heterogeneity selection 문제로 옮긴다.

---

## 2. 연구 목표

본 연구의 목표는 다음과 같다.

**첫째,** Gaussian mixture mean structure를 effects-style로 분해한다.

$$\mu_j = \mu_0 + \delta_j, \qquad j = 1, \dots, K.$$

식별성을 위해 각 변수 $k$에 대해

$$\sum_{j=1}^K \delta_{jk} = 0$$

를 둔다.

**둘째,** mean-heterogeneity-driving variable set을 다음과 같이 정의한다.

$$S_0 = {k : \exists j \neq \ell,\ \mu_{jk}^0 \neq \mu_{\ell k}^0}.$$

sum-to-zero parameterization에서는

$$S_0 = {k : \delta_{\cdot k}^0 \neq 0}$$

이다.

**셋째,** sum-to-zero constrained lasso를 final estimator가 아니라 screening estimator로 사용한다.

**넷째,** 선택된 변수 집합 $\hat S$ 위에서 unpenalized GMM MLE를 refit한다.

**다섯째,** lasso shrinkage로 인한 mean contrast attenuation이 refit으로 복원되는지 $R_k$, $\mathrm{MSE}_{\Delta, S}$, entropy, ARI를 통해 검증한다.

---

## 3. 핵심 연구 질문

### Q1. Sum-to-zero constrained lasso는 mean-heterogeneity variables의 screening rule로 충분한가?

목표는 exact support recovery보다 먼저 sure screening이다.

$$P(S_0 \subseteq \hat S_\lambda) \to 1.$$

기존 시뮬레이션에서 Naive Lasso는 주 분석 구간에서 TPR이 높고 FPR이 낮게 나타났으므로, 현재 가장 자연스러운 가설은 "screening은 이미 충분하고, 병목은 shrinkage"라는 것이다.

### Q2. Naive Lasso의 낮은 ARI는 shrinkage bias 때문인가?

이를 확인하기 위해 recovery ratio를 핵심 지표로 둔다.

$$R_k = \frac{|\hat\delta_{\cdot k}|_2}{|\delta_{\cdot k}^0|_2}, \qquad k \in S_0.$$

기대되는 패턴은 다음이다.

$$R_k^{\text{Naive}} < 1, \qquad R_k^{\text{SZL-Refit}} \approx 1.$$

즉, Naive Lasso가 변수는 찾지만 mean contrast를 줄이고, refit이 이를 복원하는지 확인한다.

### Q3. Refit estimator는 oracle-feature GMM과의 gap을 줄이는가?

Oracle estimator를

$$\hat\Theta^{\text{oracle}} = \arg\max_{\Theta:\ \delta_{\cdot k} = 0,\ k \notin S_0} \ell_n(\Theta)$$

로 두고, 제안 estimator를

$$\hat\Theta^{\text{refit}} = \arg\max_{\Theta:\ \delta_{\cdot k} = 0,\ k \notin \hat S} \ell_n(\Theta)$$

로 둔다.

여기서 $\ell_n(\Theta) = \frac{1}{n}\sum_{i=1}^n \log\big[\sum_{j=1}^K \pi_j \phi_p(X_i; \mu_0+\delta_j, \Sigma)\big]$는 average observed log-likelihood이다(이하 본 문서 전체에서 동일 표기를 사용).

실증적으로는 다음 패턴을 확인한다.

$$\text{ARI}(\hat\Theta^{\text{refit}}) > \text{ARI}(\hat\Theta^{\text{Naive}}),$$ $$\text{ARI}(\hat\Theta^{\text{refit}}) \approx \text{ARI}(\hat\Theta^{\text{oracle}}).$$

---

## 4. 제안 모형

### 4.1 Gaussian mixture mean-shift model

관측치 $X_i = (X_{i1}, \dots, X_{ip})^\top \in \mathbb{R}^p$ 와 잠재군집 $Z_i \in {1, \dots, K}$ 에 대해

$$P(Z_i = j) = \pi_j, \quad j = 1, \dots, K,$$

$$X_i \mid Z_i = j \sim N_p(\mu_j, \Sigma),$$

$$\mu_j = \mu_0 + \delta_j.$$

식별성을 위해

$$\sum_{j=1}^K \delta_{jk} = 0, \quad k = 1, \dots, p$$

를 둔다. 이 제약하에서

$$\mu_0 = \frac{1}{K} \sum_{j=1}^K \mu_j$$

이다. 따라서 $\mu_0$는 marginal population mean이 아니라 component means의 unweighted grand mean이다. 실제로

$$E(X_i) = \sum_{j=1}^K \pi_j \mu_j = \mu_0 + \sum_{j=1}^K \pi_j \delta_j.$$

따라서 $\pi_j$가 모두 같거나 $\sum_j \pi_j \delta_j = 0$인 특수한 경우를 제외하면 $\mu_0 \neq E(X_i)$이다.

### 4.2 Mean-heterogeneity variable

변수 $k$가 군집 평균 차이를 유발한다는 것은 다음과 같다.

$$\exists j \neq \ell \text{ such that } \mu_{jk}^0 \neq \mu_{\ell k}^0.$$

따라서

$$S_0 = {k : \exists j \neq \ell,\ \mu_{jk}^0 \neq \mu_{\ell k}^0}.$$

sum-to-zero parameterization에서는

$$S_0 = {k : |\delta_{\cdot k}^0|_2 > 0}.$$

중요한 점은 다음이다.

$$\delta_{\cdot k} = 0 \iff \delta_{1k} = \cdots = \delta_{Kk} = 0.$$

따라서 element-wise lasso 결과를 변수 단위로 aggregate하면 $S_0$를 추정할 수 있다. 다만 element-wise lasso와 group lasso는 penalty geometry가 다르므로, "element-wise lasso가 group lasso와 동일하다"고 표현하지 않는다. 정확한 표현은 다음이다.

> Element-wise lasso는 component-level sparsity를 유도하지만, 본 연구에서는 그 결과를 variable-level mean contrast로 aggregate하여 $S_0$를 추정한다.

---

## 5. 제안 추정 절차

제안 방법의 이름은 다음으로 둔다.

> **Debiased Sum-to-Zero Lasso Mixture Clustering**

약칭은 **SZL-Refit**으로 둔다. 전체 절차는 다음 네 단계이다.

- **Stage 1:** SZ-Lasso screening
- **Stage 1.5:** Variable-level aggregation
- **Stage 2:** Unpenalized GMM refit
- **Stage 3:** Refit-likelihood EBIC tuning

### 5.1 Stage 1: Sum-to-zero lasso screening

각 $\lambda$에 대해 다음 penalized objective를 최대화한다.

$$\hat\Theta_\lambda^{\text{SZL}} = \arg\max_\Theta \left{ \ell_n(\Theta) - \lambda \sum_{k=1}^p \sum_{j=1}^K \frac{|\delta_{jk}|}{\hat\sigma_k} \right}$$

subject to

$$\sum_{j=1}^K \delta_{jk} = 0, \quad k = 1, \dots, p.$$

여기서 $\hat\sigma_k$로 나누는 항은 변수별 scale normalization이다. $\Sigma = I_p$로 고정하는 초기 실험에서는 생략 가능하다.

이 단계에서 얻은 estimator는 final estimator가 아니다.

> $\hat\Theta_\lambda^{\text{SZL}}$ is a screening estimator, not the final estimator.

기존 Naive Lasso와 수식 형태는 동일하지만 해석이 다르다. 기존 Naive Lasso는 no-refit baseline이고, SZL-Refit은 이 결과를 screening으로만 사용한다.

### 5.2 계산상 sum-to-zero 제약 처리

$\mu_j = \mu_0 + \delta_j$만으로는 분해가 식별되지 않으므로 sum-to-zero 제약은 필수적이다. 본 연구에서는 element-wise $\ell_1$ penalty와 sum-to-zero 제약을 결합하기 위해 **direct constrained update** 방식을 메인 알고리즘으로 사용한다.

**Direct constrained update.** EM의 M-step에서 변수 $k$별로 다음 constrained weighted lasso subproblem을 푼다.

$$\min_{\delta_{1k}, \dots, \delta_{Kk}} \sum_{j=1}^K w_{jk} (\delta_{jk} - z_{jk})^2 + \lambda_k \sum_{j=1}^K |\delta_{jk}|$$

subject to $\sum_j \delta_{jk} = 0$.

여기서 $w_{jk}$, $z_{jk}$는 E-step에서 계산되는 weighted target과 weight이고, $\lambda_k = \lambda / \hat\sigma_k$이다. 이 problem은 $K$차원의 작은 convex problem이므로 표준 convex solver(또는 Lagrangian-augmented coordinate descent)로 안정적으로 풀린다.

**Q-basis 재파라미터화에 대한 주석.** 직전 보고서의 group penalty 설계에서는 $Q \in \mathbb{R}^{K \times (K-1)}$ ($Q^\top \mathbf{1}_K = 0$, $Q^\top Q = I_{K-1}$)를 통한 재파라미터화 $\delta_{\cdot k} = Q\alpha_k$가 자연스러웠다. 그 이유는 group penalty $|\delta_{\cdot k}|_2 = |\alpha_k|_2$가 재파라미터화 후에도 그대로 보존되기 때문이다. 그러나 element-wise $\ell_1$ penalty의 경우

$$\sum_{j=1}^K |\delta_{jk}| = \sum_{j=1}^K \left| \sum_{m=1}^{K-1} Q_{jm} \alpha_{km} \right|$$

이 되어, 일반적으로 $\alpha_k$ 좌표별 separable lasso가 아니다. 따라서 본 연구에서는 element-wise penalty와 정합되는 direct constrained update를 메인 알고리즘으로 채택한다. Q-basis는 (필요 시) numerical 안정화 목적의 보조 도구로만 사용한다.

### 5.3 Stage 1.5: Variable-level aggregation

Lasso는 개별 $\delta_{jk}$에 작용하지만, 연구 대상은 변수 $k$이다. 따라서 lasso 결과를 변수 단위로 aggregate한다. 권장 기준은 standardized max-pairwise contrast이다.

$$\hat S_\lambda = \left{ k : \max_{j < \ell} \frac{|\hat\mu_{jk, \lambda}^{\text{SZL}} - \hat\mu_{\ell k, \lambda}^{\text{SZL}}|}{\hat\sigma_{k, \lambda}} > \tau_{\text{num}} \right}.$$

여기서 $\tau_{\text{num}} = 10^{-4}$는 statistical tuning parameter가 아니라 **numerical tolerance**이다. 기존 보고서에서도 $\tau = 10^{-4}$는 수치적 파편화 제거용 threshold로 사용되었다.

대안적으로

$$\hat S_\lambda = \left{ k : |\hat\delta_{\cdot k, \lambda}^{\text{SZL}}|_2 > \tau_{\text{num}} \right}$$

도 가능하지만, 본문에서는 max-pairwise contrast 기준을 기본으로 둔다. 이 기준이 "변수 $k$에서 군집 평균이 갈라지는가"라는 연구 질문과 더 직접적으로 연결되기 때문이다.

### 5.4 Stage 2: Unpenalized post-selection refit

각 $\lambda$가 만든 $\hat S_\lambda$를 고정하고, penalty 없이 GMM likelihood를 다시 최대화한다.

$$\hat\Theta_\lambda^{\text{refit}} = \arg\max_{\Theta:\ \delta_{\cdot k} = 0,\ k \notin \hat S_\lambda} \ell_n(\Theta).$$

즉, $k \in \hat S_\lambda$인 변수는 $\mu_{1k}, \dots, \mu_{Kk}$를 자유롭게 추정하고, $k \notin \hat S_\lambda$인 변수는 $\mu_{1k} = \cdots = \mu_{Kk}$를 강제한다. **이 단계에서 lasso shrinkage가 제거된다.**

> Lasso는 변수 후보를 찾고, refit은 선택 변수의 군집 중심을 shrinkage 없이 재추정한다.

### 5.5 Stage 3: Refit-likelihood EBIC tuning

EBIC는 penalized lasso estimator가 아니라 refit estimator 기준으로 계산한다.

$$\text{EBIC}_\alpha(\lambda) = -2 n \cdot \ell_n(\hat\Theta_\lambda^{\text{refit}}) + \log n \cdot \text{df}(\hat S_\lambda) + 2\alpha |\hat S_\lambda| \log p,$$

여기서 $\alpha \in [0, 1]$이다. 첫 항의 $n$ scaling은 $\ell_n$이 average log-likelihood로 정의되었기 때문이다(만약 total log-likelihood로 정의한다면 $-2L_n$이 된다). EBIC는 큰 model space에서 BIC의 과선택 경향을 보완하기 위해 제안된 기준으로, high-dimensional model selection에서 자주 사용된다(Chen and Chen, 2008).

**자유도의 effective form.** $\lambda$별 모델 비교에서 mixing proportions $(K-1)$, common means $p$, diagonal variances $p$ 항은 모든 후보 모델에 공통으로 들어가는 nuisance dimension이다. 따라서 모델 비교에서 실제로 작용하는 effective complexity는

$$\text{df}_{\text{eff}}(\hat S_\lambda) = (K-1) |\hat S_\lambda|$$

이고, EBIC의 model-comparison-relevant 부분은

$$\text{EBIC}_\alpha(\lambda) \propto -2 n \cdot \ell_n(\hat\Theta_\lambda^{\text{refit}}) + \log n \cdot (K-1) |\hat S_\lambda| + 2\alpha |\hat S_\lambda| \log p$$

로 환원된다. 본 연구에서는 이 effective form을 tuning에 사용한다.

최종 선택은

$$\hat\lambda = \arg\min_\lambda \text{EBIC}_\alpha(\lambda), \qquad \hat S = \hat S_{\hat\lambda}, \qquad \hat\Theta = \hat\Theta_{\hat\lambda}^{\text{refit}}.$$

$\alpha$는 기본값을 $0.5$로 두고, sensitivity analysis로 $\alpha \in {0, 0.5, 1}$을 비교한다.

---

## 6. Secondary extension: Adaptive SZL-Refit

Adaptive lasso는 메인 방법이 아니라 secondary extension으로 둔다.

### 6.1 Adaptive Stage 1

Adaptive version의 Stage 1은 다음이다.

$$\hat\Theta_\lambda^{\text{ASZL}} = \arg\max_\Theta \left{ \ell_n(\Theta) - \lambda \sum_{k=1}^p \sum_{j=1}^K a_{jk} \frac{|\delta_{jk}|}{\hat\sigma_k} \right}$$

subject to $\sum_{j=1}^K \delta_{jk} = 0$, where

$$a_{jk} = (|\tilde\delta_{jk}| + \varepsilon_n)^{-\gamma}.$$

### 6.2 Pilot estimator 선택과 결과적 stage 수

Pilot $\tilde\delta_{jk}$는 plain SZL screening 결과 또는 SZL-Refit 결과를 사용한다.

이 선택은 ASZL-Refit pipeline의 실제 단계 수에 영향을 준다.

|Pilot 종류|전체 pipeline|비고|
|---|---|---|
|Plain SZL screening|3-stage: SZL screening → ASZL screening → unpenalized refit|Pilot이 가장 가벼움|
|Plain SZL-Refit|4-stage: SZL screening → unpenalized refit (pilot용) → ASZL screening → unpenalized refit (final)|Pilot이 더 안정적이나 계산비용 증가|

본 연구의 secondary extension에서는 **3-stage version**(plain SZL screening pilot)을 기본으로 두고, 4-stage version은 robustness 검증용으로만 보고한다.

### 6.3 Variable-level adaptive weight 제외 이유

다음 형태의 variable-level adaptive weight

$$v_k \sum_{j=1}^K |\delta_{jk}|$$

는 메인 방법으로 사용하지 않는다. 이 구조는 엄밀히 group lasso는 아니지만, 변수 $k$ 전체에 동일한 weight를 부여하므로 group-like behavior로 해석될 여지가 있다. 이번 연구미팅의 방향이 group penalty를 메인에서 제외하는 것이므로, adaptive extension을 쓰더라도 component-wise weight $a_{jk}$에 한정한다.

### 6.4 Adaptive variant의 역할

Adaptive version의 역할은 다음으로 제한한다.

> Plain SZL-Refit이 oracle gap을 충분히 메우지 못하는 저신호 setting에서의 robustness variant.

따라서 critical sanity check 결과가 중요하다.

---

## 7. 이론 계획

본 연구의 이론은 세 정리로 구성한다. 점근 정규성과 finite-sample valid post-selection inference는 본 논문의 핵심 정리로 두지 않고, selection consistency 조건하의 corollary 또는 future work로 둔다.

### 7.1 Assumptions

**A1. Identifiability and separation.**

$$\min_{j \neq \ell} |\mu_j^0 - \mu_\ell^0|_{\Sigma^{-1}} \geq c_n.$$

혼합모형의 label switching을 고려하여 모수 거리는 label permutation에 대해 정의한다.

**A2. Sparsity and dimensionality.**

$$s_0 = |S_0| \ll n, \qquad s_0 \log p = o(n).$$

**A3. Beta-min condition.**

Sum-to-zero parameterization과 정합되도록, 변수 $k$의 standardized mean heterogeneity strength를 다음으로 정의한다.

$$B_k^0 = \frac{1}{\sigma_k^2} \cdot \frac{1}{K} \sum_{j=1}^K (\mu_{jk}^0 - \mu_{0k})^2 = \frac{1}{\sigma_k^2} \cdot \frac{1}{K} \sum_{j=1}^K \delta_{jk}^{0,2} = \frac{|\delta_{\cdot k}^0|_2^2}{K \sigma_k^2}.$$

여기서 $\mu_{0k} = K^{-1}\sum_j \mu_{jk}^0$는 sum-to-zero 제약 하의 grand mean이다 ($\pi_j$-weighted marginal mean이 아님). 이 정의는 sum-to-zero parameterization과 자연스럽게 정합되며, $\delta_{\cdot k}^0$의 효과 크기를 직접 반영한다.

Beta-min 조건은

$$B_{\min} = \min_{k \in S_0} B_k^0$$

가 sure screening rate를 만족할 만큼 충분히 크다는 것이다. 구체적으로는

$$n B_{\min} \gg \log p$$

수준의 lower bound가 필요하다.

**A4. Local EM condition.**

혼합모형의 non-convexity 때문에 global optimum 이론을 바로 주장하지 않는다. 대신 좋은 initialization 근방에서 EM이 local contraction을 만족하거나, population likelihood가 true parameter 근방에서 locally identifiable하다고 가정한다.

### 7.2 Theorem 1. Sure screening

적절한 $\lambda_n$ sequence와 위 조건하에서

$$P(S_0 \subseteq \hat S_\lambda) \to 1.$$

즉, Stage 1 screening은 참 mean-heterogeneity variables를 빠뜨리지 않는다. Refit은 빠진 변수를 복구할 수 없으므로, 이는 refit 설계에서 가장 중요한 조건이다.

### 7.3 Theorem 2. Selection size control

EBIC-tuned $\hat\lambda$에 대해

$$|\hat S_{\hat\lambda}| = O_p(s_0).$$

즉, 선택집합이 true support를 포함하되 지나치게 커지지 않는다. 이 결과는 Stage 2 refit의 variance inflation을 제어하기 위해 필요하다.

### 7.4 Theorem 3. Oracle refit equivalence

Oracle estimator를

$$\hat\Theta^{\text{oracle}} = \arg\max_{\Theta:\ \delta_{\cdot k} = 0,\ k \notin S_0} \ell_n(\Theta)$$

로 두고, 제안 refit estimator를

$$\hat\Theta^{\text{refit}} = \arg\max_{\Theta:\ \delta_{\cdot k} = 0,\ k \notin \hat S} \ell_n(\Theta)$$

로 둔다.

**(a) Rate result.** Theorems 1–2 하에서, 적절한 metric $d(\cdot, \cdot)$ (label permutation에 대해 정의된 모수 거리)에 대해

$$d(\hat\Theta^{\text{refit}}, \Theta^0) = O_p\left(\sqrt{\frac{(K-1) s_0}{n}}\right).$$

여기서 $(K-1) s_0$는 selected mean contrast의 effective dimension이며, common means와 variances 같은 nuisance parameter는 표준 $\sqrt{p/n}$ rate로 수렴한다(별도 항).

**(b) Oracle equivalence.** 더 강하게 $P(\hat S = S_0) \to 1$이 성립하면, 동일한 local optimum basin에서

$$d(\hat\Theta^{\text{refit}}, \hat\Theta^{\text{oracle}}) = o_p(n^{-1/2})$$

이며, 이는 oracle MLE와 점근적으로 동등하다.

이를 본 연구의 **two-stage oracle property**로 제시한다.

### 7.5 Inference에 관한 주의

Finite-sample post-selection confidence interval과 selective inference는 본 연구의 메인 목표가 아니다. 선택 이후 추론은 별도의 framework(Berk et al., 2013; Lee et al., 2016)가 필요하므로 discussion 또는 future work로 둔다. 메인 정리에서는 selection consistency 조건 하 oracle MLE의 점근 정규성을 corollary 형태로만 인용한다.

---

## 8. 시뮬레이션 설계

이번 미팅 자료에서는 group-lasso benchmark를 메인 비교군에서 제외한다. 다만 직전 보고서의 HP-L, HP-AL 결과는 연구 방향 전환의 배경 자료로만 보존한다. 기존 HP-AL은 adaptive group lasso 기반 single-stage 방법으로 정리되어 있었고, 이번 설계에서는 이를 메인 contribution에서 제외한다.

### 8.1 Critical sanity check (메인 결정 실험)

메인 방법을 확정하기 위해 가장 먼저 수행해야 할 실험은 다음이다.

- $n = 300$
- $p \in {100, 300}$
- $a \in {1.6, 1.4, 1.2}$
- 반복수 $R \geq 50$ (안정 확인 후 $R = 100$ 이상으로 확장)

비교 방법은 다음으로 제한한다.

|구분|방법|목적|
|---|---|---|
|Baseline|K-means|전통적 비지도 기준|
|Baseline|PCA + K-means|차원축소 후 clustering|
|Baseline|Unpenalized GMM|고차원 비벌점 GMM 기준|
|Existing sparse|Sparse K-means|sparse clustering 기준|
|Existing penalized|Naive Lasso (no refit)|기존 single-stage lasso의 shrinkage 확인|
|**Main**|**SZL-Refit**|**refit 효과 검증**|
|**Secondary**|**ASZL-Refit**|adaptive weight의 추가효과 확인|
|Oracle|Oracle-feature GMM|true variables를 아는 경우|
|Oracle|True-parameter oracle|이상적 기준|

SelvarMix와 SC-FS는 가능하면 후속 full simulation에 추가한다. 두 방법 모두 고차원 clustering variable selection과 직접 관련된 benchmark이므로, reviewer 대응을 위해 최종 논문에는 포함하는 것이 좋다.

### 8.2 핵심 비교

가장 중요한 비교는 다음이다.

$$\text{Naive Lasso} \quad \text{vs} \quad \text{SZL-Refit} \quad \text{vs} \quad \text{ASZL-Refit}.$$

의사결정 기준은 다음과 같다.

|결과 패턴|해석|메인 방법 결정|
|---|---|---|
|SZL-Refit이 oracle gap을 대부분 회복하고 ASZL-Refit과 유사|병목은 screening이 아니라 shrinkage|Plain SZL-Refit 메인|
|SZL-Refit은 일부 회복, ASZL-Refit이 명확히 우수|adaptive screening이 필요|ASZL-Refit을 co-main 또는 main으로 승격|
|Refit해도 ARI가 거의 회복되지 않음|shrinkage만의 문제가 아님|설계 재검토|

현재 기존 결과상 가장 가능성이 높은 것은 첫 번째 패턴이다. 그러나 이는 아직 가설이므로 critical sanity check 결과로 확정한다.

### 8.3 성능 지표

ARI, TPR, FPR, $|\hat S|$ 외에 다음 지표를 반드시 보고한다.

**1) Mean center MSE**

$$\text{MSE}_\mu = \min_{\rho \in \mathcal{P}_K} \frac{1}{Kp} \sum_{j=1}^K |\hat\mu_{\rho(j)} - \mu_j^0|_2^2.$$

**2) Mean heterogeneity effect MSE**

$$\text{MSE}_{\Delta, S} = \frac{1}{K |S_0|} \sum_{k \in S_0} |\hat\delta_{\cdot k} - \delta_{\cdot k}^0|_2^2.$$

**3) Recovery ratio**

$$R_k = \frac{|\hat\delta_{\cdot k}|_2}{|\delta_{\cdot k}^0|_2}, \qquad k \in S_0.$$

**4) Classification entropy**

$$\text{Entropy} = -\frac{1}{n} \sum_{i=1}^n \sum_{j=1}^K \hat r_{ij} \log \hat r_{ij}.$$

기존 보고서에서도 $\text{MSE}_\mu$, $\text{MSE}_{\Delta, S}$, $R_k$, entropy가 ARI 차이 원인을 규명하기 위한 추가 지표로 제안되어 있었다.

### 8.4 Hero figure

본 논문의 핵심 figure는 단순 ARI bar plot보다 $R_k$ 분포 비교가 되어야 한다.

- **Panel A. Naive Lasso (no refit):** $R_k < 1$ 방향으로 분포가 치우치는지 확인한다.
- **Panel B. SZL-Refit:** $R_k \approx 1$ 방향으로 회복되는지 확인한다.

이 figure가 본 연구의 핵심 메시지를 가장 직접적으로 보여준다.

> Lasso는 변수를 찾지만 effect size를 줄이고, refit은 이를 복원한다.

---

## 9. 실데이터 분석 계획

실데이터 분석은 critical sanity check 이후 수행한다. 추천 데이터 유형은 gene expression 또는 single-cell RNA-seq이다. 이유는 다음이다.

- 고차원 setting이 자연스럽다 ($p \gg n$ 또는 $p$가 큰 경우).
- 선택 변수 $k$를 gene 또는 feature로 해석할 수 있다.
- $\delta_{\cdot k}$는 cluster-specific gene effect로 해석 가능하다.

실데이터에서는 단순 ARI만 보고하지 않고 다음을 함께 본다.

- 선택 feature 수
- Cluster stability (subsampling 또는 bootstrap stability)
- Classification entropy
- Refit 전후 mean contrast 변화 (실데이터 버전 $R_k$ 분석)
- 선택 feature의 해석 가능성 (known marker gene 등)

---

## 10. 교수님께 확인할 질문

이번 미팅에서는 다음 네 가지를 확인받는 것이 좋다.

**질문 1. 메인 방법.** Plain SZL-Refit을 메인으로 두고, ASZL-Refit을 secondary extension으로 두는 방향이 적절한가?

**질문 2. Critical sanity check.** Naive Lasso + Refit이 oracle gap을 얼마나 줄이는지 먼저 확인한 뒤 메인 방법을 최종 확정하는 전략이 적절한가?

**질문 3. 이론 범위.** 이론을 sure screening, size control, oracle refit equivalence 세 정리로 제한하는 것이 적절한가? 점근 정규성과 post-selection inference는 corollary/future work로 미루는 결정이 적절한가?

**질문 4. Benchmark 범위.** 이번 단계에서는 group-lasso benchmark를 제외하고, 추후 full simulation 또는 supplement에서만 참고로 두는 것이 적절한가?

---

## 11. 최종 요약

본 연구의 최종 방향은 다음이다.

> **Debiased Sum-to-Zero Lasso Mixture Clustering**
> 
> SZ-Lasso screening → variable-level aggregation → unpenalized GMM refit → refit-likelihood EBIC tuning

핵심 메시지는 다음이다.

- 고차원 Gaussian mixture clustering에서 lasso는 최종 추정기가 아니라 screening 도구로 사용한다.
- 선택 변수 위에서 unpenalized refit을 수행하여 lasso shrinkage로 약해진 mean contrast를 복원한다.
- 따라서 ARI gap의 원인을 selection failure와 shrinkage bias로 분리해 분석할 수 있다.

현재까지의 기존 결과는 Naive Lasso가 변수선택은 잘하지만 ARI에서 oracle-feature baseline과 gap을 보이는 패턴을 보여준다. 따라서 가장 먼저 검증해야 할 것은 adaptive weighting이 아니라 plain SZ-Lasso + Refit이 이 gap을 회복하는지이다. 이 critical sanity check 결과에 따라 Plain SZL-Refit을 메인으로 확정할지, ASZL-Refit을 co-main으로 올릴지 결정한다.
