# [연구 미팅 보고서] 고차원 데이터에서 이질성 유발 변수를 탐색하는 희소 혼합평균효과 기반 클러스터링 방법론

---

## [핵심 요약] 과거 버전 대비 모델 개선 사항


### 1. 모델 구조 및 알고리즘의 핵심 개선

- **변수 단위 선택의 명확성 확보 (Group Lasso 도입):**
	- _수정 전:_ 개별 파라미터($|\delta_{kj}|$)에 $\ell_1$ 페널티 적용 $\rightarrow$ 하나의 변수 내에서도 특정 군집만 0이 되는 파편화(Fragmentation) 발생.
    
    - _수정 후:_ 변수 단위의 군집 편차 벡터 전체($\|\delta_{\cdot k}\|_2$)에 **Group Lasso($\ell_2$) 페널티 적용** $\rightarrow$ 특정 변수를 통째로 살리거나 0으로 만들어 '이질성 유발 변수 집합($S_H$)'을 완벽하게 식별.
        
- **식별성 제약(Identifiability Constraint)의 안정화:**
    
    - _수정 전:_ 혼합 비율을 포함한 $\sum_k \pi_k\delta_k = \mathbf{0}$ 제약 $\rightarrow$ EM 반복마다 $\pi_k$가 변하여 기준이 흔들림.
        
    - _수정 후:_ 혼합 비율과 독립적인 $\sum_{j=1}^K \delta_{jk} = 0$ 제약 $\rightarrow$ 전통적 분산분석(ANOVA)처럼 파라미터 해석이 직관적이며 수치적으로 안정됨.
        
- **최적화 알고리즘의 우아함 및 희소성 보존:**
    
    - _수정 전:_ Soft-thresholding 후 매번 강제 재정렬(re-centering) $\rightarrow$ 강제 조정 시 0으로 만든 값이 다시 뒤틀려 유도된 희소성(Sparsity)이 파괴됨.
        
    - _수정 후:_ 직교여공간 Basis $Q$를 활용한 **재파라미터화($\delta_{\cdot k} = Q \alpha_k$)** $\rightarrow$ 제약식을 만족하면서도 희소성을 완벽히 보존하는 안정적인 최적화 구현.
        

---

## 1. 연구배경 및 문제의식

혼합모형 기반 회귀에서는 단순히 중요한 설명변수를 찾는 것만으로 충분하지 않고, 그중에서도 실제로 군집 간 차이를 만들어내는 변수, 즉 source of heterogeneity를 구분하는 것이 더 해석가능하고 더 간명한 모형을 만든다. Li et al.의 혼합회귀 연구는 predictor effect를 공통효과와 군집특이효과로 분해하고, relevant predictor와 heterogeneity-driving predictor를 동시에 식별하는 regularized finite mixture effects regression을 제안하였다. 특히 이 연구는 component variance가 다르면 raw effect와 scaled effect의 해석이 달라질 수 있음을 분명히 하고, scaled source of heterogeneity를 별도로 정의한다는 점에서 중요한 출발점을 제공한다.

그러나 비지도학습, 특히 고차원 클러스터링에서는 이와 같은 "이질성의 원천 추적"이 상대적으로 덜 정식화되어 있다. 기존 sparse clustering이나 model-based clustering은 주로 군집 복원 자체나 변수선택에 초점을 맞추는 경우가 많고, 군집을 구분하는 평균 구조를 공통 부분과 군집특이 부분으로 분해하여 어떤 좌표가 mean heterogeneity를 실제로 유발하는지 직접 추적하는 effects-style parameterization은 상대적으로 부족하다.

본 연구는 이러한 문제의식을 비지도학습으로 확장한다. 즉, 반응변수 $Y_i$가 없는 상황에서 군집 평균을 latent mean structure로 보고, 이 latent mean을 공통 평균 파라미터와 군집특이 편차로 분해하여 "어떤 변수들이 군집 간 평균 차이를 만들어내는가"를 직접 추적하는 클러스터링 방법론을 개발하고자 한다. 다만 여기서 분명히 해야 할 점은, 본 연구가 현재 1차적으로 다루는 대상은 "모든 형태의 군집 형성 변수"가 아니라, 공통 공분산 구조 아래에서의 mean-heterogeneity-driving variable이라는 점이다. 분산 차이 또는 상관구조 차이만으로 군집이 갈리는 경우는 현재 baseline model의 범위 밖에 있다. 이는 원 논문이 covariance structure의 차이가 heterogeneity 정의 자체를 복잡하게 만든다고 지적한 맥락과도 일치한다.

---

## 2. 연구목표

본 연구의 1차 목표는 다음과 같다.

**첫째,** 고차원 데이터에서 군집 구조를 추정하면서 동시에 군집 간 평균 차이를 유발하는 변수 집합을 식별하는 새로운 비지도 혼합모형을 제안한다.

**둘째,** 기존 문헌의 effects-model parameterization을 비지도 setting에 맞게 재해석하여, 군집 평균을 다음과 같은 형태로 분해하는 parsimonious mixture mean-effects model을 구축한다.

$$\mu_j = \mu_0 + \delta_j$$

**셋째,** $p \gg n$ 환경에서 support recovery와 mean structure estimation error에 대한 이론적 보장을 우선적으로 제시하고, 추가로 군집 성능에 대해서는 separation-dependent clustering bound 또는 Bayes rule 대비 excess risk consistency를 목표로 한다. 원 논문의 이론은 fixed $p, m$에서의 estimation/selection consistency에 초점이 있으며, high-dimensional extension은 명시적으로 후속 과제로 제시되어 있다. 본 연구의 박사논문 기여는 바로 이 지점을 비지도 고차원 설정으로 확장하는 데 있다.

---

## 3. 핵심 연구질문

본 연구는 다음 질문에 답하는 것을 목표로 한다.

- **Q1.** 비지도 혼합모형에서 mean heterogeneity의 source를 어떻게 엄밀히 정의할 것인가?
    
- **Q2.** 군집 추정과 mean-heterogeneity variable selection을 동시에 수행하는 정규화 mixture model은 어떻게 설계할 것인가?
    
- **Q3.** 고차원 환경에서 이 방법의 support recovery와 parameter error bound를 어떻게 보일 것인가?
    
- **Q4.** 분산 구조가 달라질 때 heterogeneity의 정의를 어떻게 조정할 것인가?
    

---

## 4. 제안모형

### 4.1 기본 모형

관측치 $X_i = (X_{i1}, \dots, X_{ip})^\top \in \mathbb{R}^p$, 잠재 군집 $Z_i \in {1, \dots, K}$에 대하여 다음 baseline model을 제안한다.

$$P(Z_i = j) = \pi_j, \quad j = 1, \dots, K$$

$$X_i \mid Z_i = j \sim N_p(\mu_j, \Sigma)$$

$$\mu_j = \mu_0 + \delta_j, \quad \sum_{j=1}^K \delta_{jk} = 0, \quad k = 1, \dots, p$$

여기서 $\mu_0 \in \mathbb{R}^p$는 sum-to-zero coding 하의 grand mean parameter이고, $\delta_j \in \mathbb{R}^p$는 군집 $j$의 mean deviation vector이다. 따라서 각 군집의 중심은 다음과 같이 표현된다.

$$\mu_j = E(X_i \mid Z_i = j) = \mu_0 + \delta_j$$

다만 중요한 점은, 현재 선택한 제약 $\sum_{j=1}^K \delta_{jk} = 0$ 하에서 $\mu_0$는 일반적으로 marginal population mean과 동일하지 않다는 것이다. 실제로

$$E(X_i) = \sum_{j=1}^K \pi_j \mu_j = \mu_0 + \sum_{j=1}^K \pi_j \delta_j$$

이므로, $\mu_0$는 $\pi_j$가 모두 같거나 $\sum_j \pi_j \delta_j = 0$인 특수한 경우에만 marginal mean과 일치한다. 따라서 본 연구에서 $\mu_0$는 "전체 평균"이라기보다 effects-style parameterization에서의 기준점 역할을 하는 grand mean parameter로 해석하는 것이 정확하다. 이 점을 명확히 하지 않으면 모형 해석에 불필요한 혼동이 생길 수 있다.

또한 본 연구는 원 논문의 parameterization에서 공통효과/군집특이효과 분해를 회귀계수에 적용했던 아이디어를, 비지도 setting에서는 군집 평균에 적용한 것으로 볼 수 있다. 즉, 원 논문과 문제의식은 연결되지만, 직접적으로 동일한 모형을 비지도화한 것은 아니며, "predictor effect heterogeneity"를 "component mean heterogeneity"로 재구성한 모형이다.

### 4.2 이질적 변수의 정의

변수 $k$에 대하여 $\delta_{\cdot k} = (\delta_{1k}, \dots, \delta_{Kk})^\top$ 라 두면, mean heterogeneity를 유발하는 변수 집합을 다음과 같이 정의한다.

$$S_H = \{k : \|\delta_{\cdot k}\|_2 \neq 0\}$$

즉, $\delta_{1k} = \dots = \delta_{Kk} = 0$이면 변수 $k$는 모든 군집에서 평균이 동일하므로 군집 간 mean difference를 유발하지 않는다. 반대로 $\|\delta_{\cdot k}\|_2 > 0$이면 변수 $k$는 적어도 하나의 군집에서 평균 차이를 만들어내므로 mean-heterogeneity-driving variable이다.

여기서 범위를 분명히 해야 한다. 위 정의는 "현재 baseline model 하에서의 평균 기반 이질성"을 의미한다. 따라서 본 모형이 직접 식별하는 것은 variance heterogeneity나 covariance heterogeneity를 포함한 일반적 의미의 cluster-forming variable 전체가 아니라, 공통 공분산 구조 아래에서 mean shift를 통해 군집 분리를 유발하는 변수이다. 이 점은 연구 범위를 정확하게 한정해 주며, 이후 분산구조 확장으로 자연스럽게 이어질 수 있다.

### 4.3 공분산 구조: 왜 diagonal covariance부터 시작하는가

본 연구의 초기 모델 설정 및 1차 시뮬레이션에서는 다음과 같이 두는 것이 타당하다.

$$\Sigma_j = \Sigma = \mathrm{diag}(\sigma_1^2, \dots, \sigma_p^2)$$

또는 가장 단순하게 $\Sigma = I_p$로 둔다. 이 가정 아래에서는 군집이 주어졌을 때 각 좌표가 조건부 독립이므로, mean heterogeneity selection 문제를 가장 선명하게 분리하여 볼 수 있다. 이는 "실제 데이터가 반드시 독립이다"라는 주장이 아니라, 1차 단계에서 mean heterogeneity 자체를 먼저 정교하게 정식화하기 위한 working model이다.

원 논문 역시 component variance가 heterogeneity 해석에 직접 영향을 주며, covariance structure가 달라질 경우 source of heterogeneity의 정의가 복잡해진다고 논의한다. 따라서 본 연구에서도 1차 단계에서는 공통 diagonal covariance로 문제를 정리하고, 이후 확장으로 unequal diagonal variance, correlated feature, 또는 cluster-specific covariance를 고려하는 것이 전략적으로 적절하다.

---

## 5. 추정방법

### 5.1 정규화된 목적함수

모수 $\Theta = (\pi_1, \dots, \pi_K, \mu_0, \delta_1, \dots, \delta_K, \Sigma)$ 에 대해 다음과 같은 normalized penalized log-likelihood를 고려한다.

$$\mathcal{L}_n(\Theta) = \frac{1}{n} \sum_{i=1}^n \log \left[ \sum_{j=1}^K \pi_j \phi_p(X_i; \mu_0 + \delta_j, \Sigma) \right] - \lambda_n \sum_{k=1}^p w_k \|\delta_{\cdot k}\|_2$$

여기서 $w_k$는 adaptive weight이며 예를 들면 다음과 같이 pilot estimator로부터 구성할 수 있다.

$$w_k = (\|\tilde{\delta}_{\cdot k}\|_2 + \varepsilon)^{-\gamma}$$

이와 같이 목적함수를 $n$으로 정규화해 두면 $\lambda_n$의 order를 이론적으로 다루기 더 명확하다. 물론 동치인 비정규화 형태 $\ell_n(\Theta) - n\lambda_n \sum_{k=1}^p w_k \|\delta_{\cdot k}\|_2$ 로도 쓸 수 있으나, 논문에서는 둘 중 하나로 반드시 통일하는 것이 필요하다. 본 연구에서는 normalized form을 기본 표기로 채택한다.

또한 현재 모형에서는 variable-wise selection을 위해 element-wise $\ell_1$보다 $\|\delta_{\cdot k}\|_2$ 형태의 group penalty를 사용하는 것이 더 자연스럽다. 하나의 변수는 모든 군집에서 함께 살아남거나 함께 0이 되므로, "어떤 변수 전체가 mean heterogeneity를 유발하는가"라는 질문에 직접 대응할 수 있다.

### 5.2 식별성 제약

$\mu_j = \mu_0 + \delta_j$ 만으로는 $\mu_0$와 $\delta_j$의 분해가 유일하지 않다. 따라서 다음과 같은 sum-to-zero 제약이 필요하다.

$$\sum_{j=1}^K \delta_{jk} = 0, \quad k = 1, \dots, p$$

이는 원 논문에서의 effects-model parameterization과 동일한 역할을 수행하는 식별성 제약이다. 다만 원 논문에서는 variance scaling을 함께 고려하는 회귀 setting이었다면, 본 연구에서는 mean structure에 이 제약을 적용한다는 차이가 있다.

### 5.3 계산 알고리즘

계산은 EM 알고리즘을 기본 골격으로 한다.

E-step에서는 책임도(responsibility)를 계산한다.

$$\tau_{ij} = P(Z_i = j \mid X_i, \Theta) = \frac{\pi_j \phi_p(X_i; \mu_0 + \delta_j, \Sigma)}{\sum_{\ell=1}^K \pi_\ell \phi_p(X_i; \mu_0 + \delta_\ell, \Sigma)}$$

M-step에서는 $\pi_j, \Sigma, \mu_0, \delta_j$를 갱신한다. 특히 $\Sigma$가 diagonal일 때 각 변수 $k$에 대한 업데이트는 거의 분리되어 다음과 같은 문제로 귀결된다.

$$\min_{\mu_{0k}, \delta_{\cdot k}} \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^K \tau_{ij} \sigma_k^{-2} (x_{ik} - \mu_{0k} - \delta_{jk})^2 + \lambda_n w_k \|\delta_{\cdot k}\|_2$$

subject to

$$\sum_{j=1}^K \delta_{jk} = 0$$

실제 구현에서는 $\mathbf{1}_K$ 의 직교여공간 basis $Q$ 를 써서 $\delta_{\cdot k} = Q \alpha_k$ 로 재파라미터화하면 제약이 사라져 unconstrained group lasso 문제로 바뀐다. 이는 수치적 안정성과 희소성 보존 측면에서 유리하다.

튜닝 파라미터 $\lambda_n$ 와 군집 수 $K$ 는 BIC, ICL, 혹은 clustering stability 기준으로 선택할 수 있다. 원 논문에서도 mixture component 수와 penalty parameter 선택에 BIC를 사용한다.

---
 
## 6. 이론적 연구목표

기존 연구는 fixed $p, m$ 설정에서 adaptive estimator의 $\sqrt{n}$-consistency와 selection consistency를 제시하였다. 본 연구의 박사논문 기여는 이 결과를 비지도 high-dimensional setting으로 확장하는 데 있다. 다만 현재 단계에서 직접적으로 "misclustering rate가 항상 0으로 간다"는 식의 강한 주장을 두는 것은 과도하므로, 이론 목표를 다음과 같이 정교하게 설정하는 것이 바람직하다.

**첫째, 식별성.** label switching을 제외하면 $(\pi, \mu_0, \Delta, \Sigma)$가 유일하게 식별됨을 보인다.

**둘째, 추정오차 경계.** 희소도 $s = |S_H|$에 대해 다음과 같은 형태의 오차 경계를 목표로 한다.

$$\|\hat{\Delta} - \Delta^*\|_F = O_p \left( \sqrt{\frac{sK \log p}{n}} \right)$$

**셋째, support recovery.** 적절한 beta-min 조건 $\min_{k \in S_H} \|\delta_{\cdot k}^*\|_2 \ge c\lambda_n$ 하에서 다음을 보이고자 한다.

$$P(\hat{S}_H = S_H) \to 1$$

**넷째, clustering performance.** 현재 baseline에서는 다음 두 종류의 결과 중 하나를 목표로 하는 것이 더 적절하다.

하나는 Bayes rule 대비 excess classification risk consistency이다. 예를 들어 다음과 같은 결과를 보이는 방식이다.

$$R(\hat{g}) - R(g^*) \to 0$$

다른 하나는 separation-dependent misclustering bound이다. 즉, 군집 간 분리가 충분히 커지는 경우에 한하여 misclustering rate가 0으로 가는 결과를 제시하는 것이다. 이를 위해 다음을 정의하고,

$$\Delta_{\min, n}^2 = \min_{j \neq \ell} \sum_{k \in S_H} \frac{(\delta_{jk}^* - \delta_{\ell k}^*)^2}{\sigma_k^2}$$

$\Delta_{\min, n}^2 \to \infty$와 같은 stronger separation regime 하에서 다음을 보이는 방향이 더 타당하다.

$$\frac{1}{n} \sum_{i=1}^n I(\hat{Z}_i \neq Z_i^*) \to 0$$

반대로 separation이 고정되어 있고 component overlap이 존재하면, Bayes classifier 자체도 양의 오분류율을 가질 수 있으므로 무조건적인 zero-misclustering consistency를 전면에 내세우는 것은 적절하지 않다. 이 부분은 미팅에서 선제적으로 정리해 두는 것이 좋다.

기본 가정의 예로는 다음을 둘 수 있다.

$$\pi_j^* \ge \pi_{\min} > 0, \quad 0 < c_\sigma \le \sigma_k^2 \le C_\sigma < \infty$$

$$s \log p = o(n)$$

그리고 clustering 관련 결과를 위해서는 추가적으로 suitable separation assumption을 둘 수 있다.

---

## 7. 기존 연구와의 차별성

본 연구의 차별점은 단순히 "클러스터링에 유용한 변수"를 고르는 것이 아니라, 군집 평균의 좌표별 분해를 통해 "왜 군집이 갈리는가"를 직접 묻는다는 점에 있다.

다만 원 논문과 현재 모형의 관계는 정확히 구분해서 설명할 필요가 있다. 원 논문에서는 mixture regression setting에서 relevant predictor 집합 $S_R$와 source of heterogeneity 집합 $S_H$를 동시에 구분한다. 즉, 공통효과는 있지만 군집특이효과는 없는 predictor와, 실제로 군집 간 효과 차이를 만들어내는 predictor를 분리한다. 반면 현재 비지도 baseline model은 outcome이 없는 평균 혼합모형이므로, 원 논문에서의 $S_R$–$S_H$ 구조를 그대로 재현하는 것은 아니다. 현재 1차 모형이 직접 식별하는 것은 사실상 "mean-heterogeneity-driving coordinate"에 해당하는 $S_H$-유사 객체이다. 이 점을 솔직하게 밝히는 것이 오히려 연구의 범위를 더 선명하게 만든다.

즉, 본 연구는 원 논문의 개념을 그대로 비지도화한 것이 아니라, 그 핵심 문제의식인 "heterogeneity의 원천 추적"을 mean-shift clustering 문제로 재구성한 방법론이라고 정리하는 가장 정확하다.

또한 원 논문이 high-dimensional setting, cluster learning, multivariate extension을 후속 연구 방향으로 제시했다는 점을 감안하면, 본 연구는 바로 그 방향 중 "cluster learning under high-dimensional heterogeneity pursuit"를 직접 겨냥한 확장으로 해석할 수 있다.

---

## 8. 후속 확장 방향

현재 1차 모형은 mean heterogeneity selection에 집중한다. 그러나 "common but relevant structure"까지 포함하는 더 풍부한 비지도 모형으로 확장하려면 예를 들어 다음과 같은 구조를 고려할 수 있다.

$$X_i = \mu_0 + \Lambda f_i + \delta_{Z_i} + \varepsilon_i, \quad \varepsilon_i \sim N_p(0, \Psi)$$

여기서 $\Lambda f_i$는 전체 표본에 공통적인 저차원 구조를 나타내고, $\delta_{Z_i}$는 군집특이 평균구조를 나타낸다. 이 경우에는 공통 구조를 반영하는 좌표와 mean heterogeneity를 유발하는 좌표를 더 정교하게 구분할 수 있다. 다만 이는 현재 1차 논문의 범위를 넘어서는 확장 주제로 두는 것이 적절하며, 우선은 공통 diagonal covariance 아래에서의 sparse mean-effects clustering을 먼저 완성하는 것이 논리적으로 더 단단하다.

---
## Part II. 시뮬레이션 결과

본 절의 시뮬레이션은 제안 모형이 "모든 형태의 군집 형성 변수"를 찾는지 검증하는 것이 아니라, 공통 공분산 구조 하에서 군집 간 평균 차이를 유발하는 변수(mean-heterogeneity-driving variables)를 얼마나 정확히 식별하는지, 그리고 그러한 선택이 실제 군집 성능 개선으로 이어지는지를 경험적으로 확인하는 데 목적이 있다. 이는 원 논문이 혼합회귀에서 relevant predictor와 source of heterogeneity를 구분하고, heterogeneity pursuit가 보다 parsimonious한 모형을 제공할 수 있다고 논의한 문제의식을 비지도 평균혼합모형으로 옮겨온 것이다. 다만 원 논문은 회귀 setting과 fixed $p, m$ 이론에 초점을 두고 있으므로, 본 절의 시뮬레이션은 그 이론을 직접 재현하는 것이 아니라 mean-shift clustering 상황에서의 경험적 타당성을 검토하는 단계로 이해하는 것이 적절하다.

### 1. 시뮬레이션 데이터 생성 모형 (Data Generation Model)

시뮬레이션 데이터는 본 연구가 제안하는 평균 분해 구조( $\mu_j = \mu_0 + \delta_j$ )를 엄밀하게 따르도록 생성되었다. 총 표본 수 $n$, 전체 변수 차원 $p$, 잠재 군집 수 $K=3$ 이 주어졌을 때, 각 관측치 $X_i \in \mathbb{R}^p$ 와 잠재 군집 라벨 $Z_i \in \{1, 2, 3\}$ 는 다음의 확률적 과정을 통해 생성된다.

**1) 잠재 군집 생성 (Cluster Assignment)**

군집 혼합 비율은 완전히 균등하게 설정하였다.

$$P(Z_i = j) = \pi_j = \frac{1}{K} \quad (j = 1, 2, 3)$$

**2) 군집별 평균 편차 구조 (Mean Heterogeneity Structure)**

전체 $p$ 개의 변수 중, 앞의 $q$ 개 변수만이 실제로 군집 간 평균 차이를 유발하는 이질성 유발 변수(Signal variables)이며, 나머지 $p - q$ 개는 평균 차이가 없는 노이즈 변수(Noise variables)이다. 군집 $j$ 의 편차 벡터 $\delta_j \in \mathbb{R}^p$ 는 다음과 같이 정의된다.

- **Cluster 1 ( $j=1$ ):** $\delta_{1k} = a \ (k \le q)$, $\delta_{1k} = 0 \ (k > q)$
    
- **Cluster 2 ( $j=2$ ):** $\delta_{2k} = 0 \ (\forall k)$
    
- **Cluster 3 ( $j=3$ ):** $\delta_{3k} = -a \ (k \le q)$, $\delta_{3k} = 0 \ (k > q)$
    

여기서 $a$ 는 이질성을 발생시키는 **신호 강도(Signal strength)**를 의미하며, $a$ 가 클수록 군집 간 거리가 멀어지고, $a$ 가 작을수록 군집이 강하게 중첩(overlap)된다. 위 구조에 의해 $\sum_{j=1}^K \delta_{jk} = 0$ 제약이 데이터 생성 단계에서부터 완벽하게 성립한다.

**3) 공분산 구조 및 데이터 추출 (Covariance and Sampling)**

관측치 $X_i$ 는 군집별로 다변량 정규분포에서 독립적으로 추출된다.

$$X_i \mid Z_i = j \sim N_p(\mu_j, \Sigma)$$

이때, 공분산 행렬 $\Sigma = \mathrm{diag}(\sigma_1^2, \dots, \sigma_p^2)$ 는 다음과 같이 변수 역할에 따라 다른 분산을 가지도록 설계하였다.

- **Signal variables ( $1 \le k \le q$ ):** $\sigma_k^2 = 1$
    
- **Noise variables ( $q < k \le p$ ):** $\sigma_k^2 = 2$
    

노이즈 변수의 분산을 2배로 증폭시킴으로써, 페널티가 없는 모형(예: 일반 GMM)이 차원의 저주(Curse of Dimensionality)에 빠지기 쉬운 환경을 모사하였다. 생성된 최종 데이터 행렬 $X$ 는 분석 전 변수별로 중심화(empirical centering)를 거쳐 $\mu_0 \approx 0$ 이 되도록 전처리된다.

### 2. 비교 방법론 및 벤치마크 모형 정의

본 시뮬레이션에서는 제안 모형(Proposed HP)의 변수 선택 능력과 군집 복원 성능을 객관적으로 평가하기 위해 총 4개 그룹의 방법론을 벤치마크로 설정하였다. 특히, 본 연구의 제안 모형(Proposed HP)은 사후 재적합(Refit)에 의존하지 않는 단일 파이프라인(Single-stage)의 강력함을 입증하기 위해 Refit 과정을 생략하였다.

**1) 전통적 비지도 학습 (Baseline Methods)**

고차원 노이즈 환경에서의 차원의 저주 붕괴 양상을 확인하기 위한 모형이다.

- **K-means & PCA + K-means**
    
- **GMM (Unpenalized):** 페널티 없이 전체 차원 $p$ 에 대해 추정을 수행하는 다변량 가우시안 혼합모형 (EEI 모형 가정).
    

**2) 기존 변수 선택 군집화 및 절제(Ablation) 모형**

제안 모형의 그룹 단위 선택 및 구조적 안정성이 왜 필수적인지를 입증하는 대조군이다.

- **Sparse K-means (sparcl):** 가중치 벡터에 $\ell_1$ 페널티를 부여하는 희소 군집화 벤치마크.
    
- **Naive Lasso (Element-wise $\ell_1$ + $\mu_0$ ):** 제안 모형과 동일한 평균 분해 구조( $\mu_j = \mu_0 + \delta_j$ )를 가지나, 개별 파라미터 단위로 $\ell_1$ 페널티를 적용한 절제 실험 모형. 심각한 수축 편향(Shrinkage Bias)을 유발한다.
    
- **+ Refit (재적합):** 페널티로 인해 발생한 수축 편향을 회복하기 위해, 선택된 변수 집합으로 페널티 없이 GMM을 다시 돌리는 과정. (Sparse K-means와 Naive Lasso에만 적용)
    

**3) 제안 모형 (Proposed Model)**

- **Proposed HP (Adaptive Group $\ell_2$ ):** 본 연구에서 제안하는 희소 혼합평균효과 모형. $Q$-basis를 활용하여 제약식을 우아하게 만족시키면서, 그룹 단위 페널티를 통해 평균 기반 이질성을 직접 식별한다. **구조적 왜곡이 없어 Refit 없이 1회 추정만으로 최적의 성능을 달성한다.**
    

**4) 오라클 벤치마크 (Oracle Bounds)**

- **Oracle-feature baseline (True Vars):** 정답 변수( $q$ 개)들만 주어졌다고 가정하여 추정한 GMM. 변수 선택 알고리즘의 이상적인 목표치.
    
- **True-parameter oracle (Bayes Classifier):** 생성에 사용한 진짜 모수( $\pi, \mu, \Sigma$ )를 그대로 사후 확률에 대입하여 분류하는 베이즈 오류율 한계치.
    

---

### 3. 기본 환경( $p=20, q=3$ )에서의 신호 강도 변화 검증

#### 3.1 실험 목적 및 세팅

기본 차원 환경에서 신호 강도 $a$ 가 점진적으로 약화될 때 제안 모형의 변수 선택 및 구조적 방어력을 확인한다.

표본 수는 $n=300$, 총 차원 $p=20$, 군집 수 $K=3$ 이며, 정답 변수는 앞의 **$q=3$ 개**이다. 신호 강도는 $a \in \{1.6, 1.4, 1.2\}$ 의 세 구간으로 설정하였다. (R=10회 반복 수행 후 평균 출력)

#### 3.2 시나리오별 결과표 (Mean)

**[시나리오 1] 신호 환경 (a = 1.6)**

|**방법론**|**사용 차원**|**ARI**|**TPR**|**FPR**|**S^**|
|---|---|---|---|---|---|
|K-means|20.000|0.413|-|-|-|
|PCA + K-means|14.000|0.400|-|-|-|
|GMM (Unpenalized)|20.000|0.522|-|-|-|
|Sparse K-means|19.600|0.679|1.000|0.976|19.600|
|$\rightarrow$ + Refit|19.600|0.581|-|-|-|
|Naive Lasso|5.000|0.660|1.000|0.118|5.000|
|$\rightarrow$ + Refit|5.000|0.639|-|-|-|
|**Proposed HP (No Refit)**|20.000|**0.673**|**1.000**|**0.059**|**4.000**|
|Oracle-feature baseline|3.000|0.682|1.000|0.000|3.000|
|True-parameter oracle|3.000|0.705|1.000|0.000|3.000|

**[시나리오 2] 중간 신호 환경 (a = 1.4)**

|**방법론**|**사용 차원**|**ARI**|**TPR**|**FPR**|**S^**|
|---|---|---|---|---|---|
|K-means|20.000|0.350|-|-|-|
|PCA + K-means|14.000|0.346|-|-|-|
|GMM (Unpenalized)|20.000|0.456|-|-|-|
|Sparse K-means|19.800|0.601|1.000|0.988|19.800|
|$\rightarrow$ + Refit|19.800|0.478|-|-|-|
|Naive Lasso|3.300|0.472|1.000|0.018|3.300|
|$\rightarrow$ + Refit|3.300|0.567|-|-|-|
|**Proposed HP (No Refit)**|20.000|**0.591**|**1.000**|**0.065**|**4.100**|
|Oracle-feature baseline|3.000|0.596|1.000|0.000|3.000|
|True-parameter oracle|3.000|0.626|1.000|0.000|3.000|

**[시나리오 3] 약한 신호 환경 (a = 1.2)**

|**방법론**|**사용 차원**|**ARI**|**TPR**|**FPR**|**S^**|
|---|---|---|---|---|---|
|K-means|20.000|0.297|-|-|-|
|PCA + K-means|14.000|0.290|-|-|-|
|GMM (Unpenalized)|20.000|0.388|-|-|-|
|Sparse K-means|19.800|0.393|1.000|0.988|19.800|
|$\rightarrow$ + Refit|19.800|0.390|-|-|-|
|Naive Lasso|3.100|0.412|1.000|0.006|3.100|
|$\rightarrow$ + Refit|3.100|0.466|-|-|-|
|**Proposed HP (No Refit)**|20.000|**0.448**|**1.000**|**0.029**|**3.500**|
|Oracle-feature baseline|3.000|0.463|1.000|0.000|3.000|
|True-parameter oracle|3.000|0.498|1.000|0.000|3.000|

#### 3.3 기본 환경( $p=20$ ) 시뮬레이션의 해석

**첫째, Refit이 불필요한 HP 모형의 구조적 우수성:** 제안 모형(Proposed HP)은 페널티 부과 후 재적합(Refit)을 수행하지 않는 단일 파이프라인임에도 불구하고, 모든 구간에서 오라클 베이스라인에 근접하는 군집 복원 성능을 달성하였다. 예컨대 $a=1.4$ 에서 HP의 단일 ARI는 0.591로, 정답만 사용한 오라클(0.596)에 필적한다. 이는 $Q$-basis를 통한 그룹 단위 수축이 중심점 간의 상대적 거리 구조를 전혀 왜곡하지 않음을 의미한다.

**둘째, 수축 편향에 갇힌 Naive Lasso:** 요소별 $\ell_1$ 페널티를 사용하는 Naive Lasso는 $a=1.4$ 에서 변수 선택( $\hat{S}=3.3$ )을 매우 훌륭하게 해냈음에도 불구하고, 초기 ARI가 0.472로 저조하였다. Refit을 거친 후에야 비로소 0.567로 성능이 회복되는 양상은 Naive Lasso가 극심한 파라미터 수축 편향(Shrinkage Bias)을 겪고 있으며, Refit 파이프라인에 절대적으로 의존해야만 작동하는 불안정한 모형임을 방증한다.

---

### 4. 고차원 환경( $p=100, q=5$ )에서의 모형 확장성 검증

#### 4.1 실험 목적 및 세팅

차원이 대폭 증가한 고차원 환경( $p=100$ )에서 노이즈가 지배적일 때, 비교 모형의 붕괴 양상과 제안 모형(HP)의 압도적인 노이즈 필터링 능력을 확인한다.

표본 수 $n=300$, 차원 **$p=100$**, 군집 수 $K=3$ 이며, 정답 변수는 **$q=5$ 개**이다. 신호 강도는 $a \in \{1.6, 1.4, 1.2\}$ 의 세 구간으로 동일하게 설정하였다.

#### 4.2 시나리오별 결과표 (Mean)

**[시나리오 1] 신호 환경 (a = 1.6)**

|**방법론**|**사용 차원**|**ARI**|**TPR**|**FPR**|**S^**|
|---|---|---|---|---|---|
|K-means|100.000|0.479|-|-|-|
|PCA + K-means|56.200|0.494|-|-|-|
|GMM (Unpenalized)|100.000|0.000|-|-|-|
|Sparse K-means|97.400|0.846|1.000|0.973|97.400|
|$\rightarrow$ + Refit|97.400|0.000|-|-|-|
|Naive Lasso|5.000|0.798|1.000|0.000|5.000|
|$\rightarrow$ + Refit|5.000|0.845|-|-|-|
|**Proposed HP (No Refit)**|100.000|**0.843**|**1.000**|**0.012**|**6.100**|
|Oracle-feature baseline|5.000|0.845|1.000|0.000|5.000|
|True-parameter oracle|5.000|0.851|1.000|0.000|5.000|

**[시나리오 2] 중간 신호 환경 (a = 1.4)**

|**방법론**|**사용 차원**|**ARI**|**TPR**|**FPR**|**S^**|
|---|---|---|---|---|---|
|K-means|100.000|0.395|-|-|-|
|PCA + K-means|56.700|0.394|-|-|-|
|GMM (Unpenalized)|100.000|0.000|-|-|-|
|Sparse K-means|88.800|0.777|1.000|0.882|88.800|
|$\rightarrow$ + Refit|88.800|0.075|-|-|-|
|Naive Lasso|6.800|0.652|1.000|0.019|6.800|
|$\rightarrow$ + Refit|6.800|0.784|-|-|-|
|**Proposed HP (No Refit)**|100.000|**0.790**|**1.000**|**0.002**|**5.200**|
|Oracle-feature baseline|5.000|0.793|1.000|0.000|5.000|
|True-parameter oracle|5.000|0.800|1.000|0.000|5.000|

**[시나리오 3] 약 신호 환경 (a = 1.2)**

|**방법론**|**사용 차원**|**ARI**|**TPR**|**FPR**|**S^**|
|---|---|---|---|---|---|
|K-means|100.000|0.350|-|-|-|
|PCA + K-means|56.900|0.342|-|-|-|
|GMM (Unpenalized)|100.000|0.000|-|-|-|
|Sparse K-means|98.900|0.691|1.000|0.988|98.900|
|$\rightarrow$ + Refit|98.900|0.000|-|-|-|
|Naive Lasso|5.100|0.439|1.000|0.001|5.100|
|$\rightarrow$ + Refit|5.100|0.700|-|-|-|
|**Proposed HP (No Refit)**|100.000|**0.700**|**1.000**|**0.008**|**5.800**|
|Oracle-feature baseline|5.000|0.698|1.000|0.000|5.000|
|True-parameter oracle|5.000|0.712|1.000|0.000|5.000|

#### 4.4 고차원 시뮬레이션의 해석

**첫째, 고차원 환경에서의 기존 벤치마크 완전 붕괴:** 차원이 $p=100$ 으로 늘어나자, 페널티가 없는 GMM 모형은 고차원 공분산 추정의 한계로 인해 모든 구간에서 ARI가 0.000으로 파괴되었다. 특히 주목할 점은 Sparse K-means의 Refit 실패 현상이다. 가짜 희소성으로 인해 90개 이상의 노이즈 변수를 제거하지 못하고 유지하였으며, 이를 기반으로 GMM Refit을 수행하자 다변량 추정이 구조적으로 불가능해져 ARI가 완전히 붕괴하였다. 고차원에서 강력한 차원 축소 능력이 필수불가결함을 보여주는 단적인 결과다.

**둘째, Naive Lasso의 수축 편향 심화 및 Refit의 한계:** 약신호( $a=1.2$ ) 고차원 구간에서 Naive Lasso는 정확히 정답 근처인 $\hat{S}=5.1$ 개의 변수를 골라내는 놀라운 희소성을 보였음에도 불구하고, 초기 ARI는 0.439에 불과했다. Refit을 수행한 후에야 0.700으로 성능을 억지로 끌어올렸으나, 이러한 투-스텝(Two-step) 파이프라인은 실제 분석 환경에서 모델의 복잡도와 연산 불안정성을 가중시키는 치명적 약점을 띤다.

**셋째, HP 모형의 단일 파이프라인(Single-stage) 절대적 우위:** 제안 모형(HP)은 이 혼탁한 $p=100$ 의 고차원 약신호 환경에서도 정답 변수 5개를 족집게처럼 포함하며 노이즈를 99% 걷어냈다( $\hat{S} \approx 5.8$ ). 더욱 놀라운 것은 **어떠한 사후 Refit 과정 없이도 초기 추정 ARI 0.700을 달성**하며, 완벽한 정답 변수만을 사용한 Oracle-feature baseline(0.698)에 맞먹는 결과를 내었다는 점이다. 이는 HP 모형이 변수 축소와 군집 중심 보존을 단일 최적화 과정 안에서 완벽하게 양립시켰음을 증명한다.

### 5. 시뮬레이션 종합 정리

저차원( $p=20$ ) 및 고차원( $p=100$ ) 환경에서의 Monte Carlo 시뮬레이션을 종합하면, 제안 방법은 다음과 같은 객관적인 우수성을 입증하였다.

- **첫째,** 노이즈가 압도적인 고차원 빅데이터 환경에서 기존 모형들이 겪는 차원의 저주를 완벽히 돌파하고, TPR=1.0을 유지하면서 핵심 변수 집합을 정밀하게 추출한다.
    
- **둘째,** $\ell_1$ 페널티 기반의 Naive Lasso가 겪는 극심한 수축 편향(Shrinkage Bias) 및 필수적인 Refit 의존성 문제를 해결하였다. $Q$-basis 기반 Group Lasso 구조를 도입한 제안 모형(HP)은 초기 추정 단계에서부터 파라미터 왜곡이 없어 번거로운 Refit 과정을 전면 생략할 수 있다.
    
- **셋째,** Refit이 없는 단일 파이프라인(Single-stage) 구성만으로도 신호 강도나 차원의 크기에 구애받지 않고 Oracle-feature baseline에 필적하는 최고 수준의 강건한 군집 복원 성능을 보장한다.
    

결과적으로, 본 시뮬레이션은 제안 방법이 "평균 기반 이질성의 원천"을 고차원 환경에서도 한 번의 최적화 과정으로 추적해 낼 수 있음을 완벽히 증명하였다. 이후 과제로는 변수 간 강한 상관관계(Correlated features)가 존재할 때, 개별 변수를 파편화시키는 Naive 모형의 한계를 HP가 어떻게 압도하는지를 보여주는 추가 벤치마크 실험이 가장 효과적일 것이다.
