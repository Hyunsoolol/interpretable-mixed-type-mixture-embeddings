## 핵심 선행 연구 지형 분석 및 제안 모형의 차별성

본 문서는 고차원 Gaussian mixture clustering에서 mean-heterogeneity-driving variable selection을 다루는 본 연구와 핵심 선행연구의 관계를 정리합니다. 특히 Caroline Meynet의 박사학위논문(Variable selection in model-based clustering for high-dimensional data, Lasso-MLE procedure)을 반영하여, 본 연구의 Novelty claim과 논문 포지셔닝을 전면 재정리하였습니다.

본 연구의 핵심은 단순히 새로운 페널티를 제안하거나 "Lasso 후 MLE refit"을 수행하는 데 있지 않습니다. Meynet and Maugis-Rabusseau의 Lasso-MLE 절차가 이미 $\ell_1$-regularization으로 모델 후보군을 만들고 선택된 모델에서 MLE를 적합하는 구조를 제안했기 때문입니다.

따라서 본 연구는 새로운 Lasso-MLE 방법론이 아닌, 다음 세 요소를 결합한 **Debiased mean-effect refit framework**로 포지셔닝합니다.

1. **Sum-to-zero mean-effect parameterization**
    
2. **Lasso shrinkage bias의 $R_k$ 기반 진단과 분해**
    
3. **선택 변수 위에서 unpenalized GMM refit을 통한 debiased mean-effect recovery**
    

권장하는 논문의 최종 제목 후보는 다음과 같습니다.

- **Debiased Mean-Effect Refit for High-Dimensional Gaussian Mixture Clustering**
    
- **Post-Selection Mean-Effect Recovery under Sum-to-Zero Gaussian Mixtures**
    

---

### 1. 가장 직접적으로 겹치는 방법론

제안 모형과 유사하게 고차원 군집화에서 $\ell_1$ 페널티를 활용하거나 변수 선택 후 재적합을 수행하는 핵심 선행 연구들입니다. 이 논문들과의 명확한 선 긋기가 논문 심사 방어의 핵심입니다.

**Meynet and Maugis-Rabusseau - Lasso-MLE Procedure (가장 가까운 선행연구)**

- **내용:** $\ell_1$-regularization을 이용해 Data-driven model collection을 만들고, 각 선택 모델에서는 Lasso estimator가 아니라 MLE를 사용해 Mixture parameter를 추정합니다.
    
- **겹치는 지점:** Lasso로 후보 Support/Model collection 생성, Lasso estimator 대신 선택된 모델에서 MLE/Refit 사용, Lasso shrinkage가 추정을 악화시킨다는 문제의식.
    
- **결정적 차별점:** 본 연구는 Lasso-MLE 원리 자체를 새 기여로 주장하지 않습니다. 대신 $\mu_j=\mu_0+\delta_j$, $\sum_j \delta_{jk}=0$ 재파라미터화를 통해 $S_0={k:\delta_{\cdot k}^0\neq 0}$를 직접 타깃으로 정의합니다. 또한 Lasso shrinkage를 단순한 Estimation bias로 치부하지 않고 $R_k$로 정의하여 ARI gap의 원인을 진단하며, Refit-likelihood EBIC를 사용하여 최종 모델을 선택합니다.
    

**Pan and Shen (2007) - $L_1$-penalized model-based clustering**

- **차별점:** Pan and Shen은 Penalized estimator 자체를 최종(Final) 추정량으로 사용합니다. 반면 본 연구는 Lasso를 Screening 도구로만 활용하고, 선택된 변수 위에서 페널티 없이 GMM을 Refit하여 축소된 Cluster mean contrast를 복원합니다.
    

**Xie, Pan, and Shen (2008) - Penalized model-based clustering with grouped variables**

- **차별점:** 기존 연구는 Raw cluster mean $\mu_{\cdot k}$ 자체를 페널티 타깃으로 삼습니다. 본 연구는 $\delta_{\cdot k}$를 타깃으로 삼으며, Element-wise lasso 결과를 Sum-to-zero 제약 하에서 집계하여 $S_0$를 추정합니다.
    

**Wang and Zhu (2008) - Variable selection for model-based high-dimensional clustering**

- **차별점:** Wang and Zhu의 목적은 전반적인 Informative-variable selection입니다. 본 연구는 Sum-to-zero mean-effect target $S_0$와 Lasso shrinkage recovery에 초점을 맞춥니다.
    

---

### 2. 타깃 및 구조적 접근 방식이 유사한 방법론

- **Guo et al. (2010):** 특정 군집 쌍을 분리하는 변수를 찾는 Pairwise variable selection입니다. 타깃이 Pairwise separability($\mu_{jk} - \mu_{\ell k}$)인 반면, 본 연구는 Variable-level mean-heterogeneity support $S_0$ 자체를 추정합니다.
    
- **Li et al. (2022):** 지도학습(Supervised) 환경인 유한 혼합 회귀에서 변수와 이질성 요인을 식별합니다. 본 연구는 반응 변수(Outcome)가 없는 순수 비지도(Unsupervised) Gaussian mixture clustering 환경입니다.
    
- **Li et al. (2023, ZINBMM) & Celeux et al. (SelvarMix):** 각각 카운트 데이터의 Global/Cluster mean 차이 패널티, 변수 역할(Role) 분류를 다룹니다. 본 연구 특유의 Gaussian mean-effects 분해와 Refit을 통한 Shrinkage recovery 프레임워크와는 다릅니다.
    

---

### 3. 모형 및 방법론 상세 비교표

#### 표 1. 모형 및 방법론 비교표

|**논문 / 방법**|**핵심 문제**|**기본 모형 및 주요 가정**|**식별성 / 제약**|**벌점 구조 및 튜닝**|
|---|---|---|---|---|
|**현재 방법론 (SZL-Refit)**|비지도 고차원 Gaussian clustering에서 mean-heterogeneity 변수 식별 및 shrinkage debiasing|$\mu_j=\mu_0+\delta_j$, 공통 diagonal covariance 또는 $\Sigma=I_p$|$\sum_{j=1}^K\delta_{jk}=0$|Stage 1: element-wise $\ell_1$ penalty<br><br>  <br><br>Stage 2: unpenalized refit (EBIC)|
|**Meynet & Maugis-Rabusseau**|변수 선택을 model selection 문제로 재정의|finite Gaussian mixture, spherical common covariance|relevant 변수는 cluster means가 동일하지 않은 변수|$\ell_1$-regularization으로 후보군 생성 후 MLE refit (Slope heuristic)|
|**Pan and Shen (2007)**|고차원 model-based clustering에서 변수 선택|Raw mean 중심, high-dimension low-sample|explicit effects coding 없음, empirical centering|$L_1$-penalized likelihood on cluster means (mBIC)|
|**Wang and Zhu (2008)**|informative variable selection|공통 diagonal covariance, centered data|명시적 ANOVA형 sum-to-zero coding 아님|adaptive/hierarchical grouped regularization|
|**Xie, Pan, and Shen (2008)**|grouped parameters를 통한 penalized clustering|Raw cluster means 중심|explicit effects coding 없음|grouped mean penalty, grouped variable selection|
|**Guo et al. (2010)**|특정 cluster pair 분리 변수 식별|pairwise difference 중심|zero가 특별한 기준이 아님|pairwise fusion penalty|
|**Li et al. (2022)**|혼합 회귀에서 predictors와 heterogeneity sources 식별|response ($Y$) 존재|$\sum_{j=1}^m\beta_{jk}=0$|common/relevant/heterogeneity penalty|

#### 표 2. 유사성 및 결정적 차이점

|**비교 논문**|**현재 방법과 가장 겹치는 지점**|**결정적 차이점 (방어 논리)**|
|---|---|---|
|**Meynet & Maugis-Rabusseau**|$\ell_1$-regularization으로 후보 support 생성 후 MLE refit|"Lasso + Refit" 절차 자체가 아닌, Sum-to-zero mean-effect 파라미터화, $S_0$ 타깃, $R_k$ 기반 shrinkage-bias 진단 논리에 독창성이 있음.|
|**Pan and Shen (2007)**|Gaussian mixture, $L_1$ penalty, EM, variable selection|Penalized estimator를 최종 산출물로 쓰지 않고, Screening 도구로 한정하여 편향을 제거함.|
|**Xie, Pan, and Shen (2008)**|Grouped variable selection의 적용|Raw mean이 아닌 $\delta_{\cdot k}$ mean-effect support를 타깃으로 하며, Stage 1에서 Group penalty를 강제하지 않음.|

#### 표 3. Meynet Lasso-MLE와 본 연구(SZL-Refit)의 세부 비교

|**항목**|**Meynet-Maugis Lasso-MLE**|**본 연구 (SZL-Refit / Debiased mean-effect refit)**|
|---|---|---|
|**가장 큰 공통점**|Lasso로 model collection 생성 후 MLE 적합|Lasso로 support path 생성 후 선택 support에서 unpenalized GMM refit|
|**Variable Target**|Cluster means가 동일하면 irrelevant, 다르면 relevant|$S_0=\{k:\delta_{\cdot k}^0\neq 0\}$|
|**Mean Parameterization**|Raw means ($\mu_{kj}$), Empirical centering 논의 중심|$\mu_j=\mu_0+\delta_j$, $\sum_j\delta_{jk}=0$|
|**Lasso Shrinkage 해석**|Lasso estimator의 mean underestimation 및 density estimation 악화|Shrinkage를 $R_k$, $\text{MSE}_{\Delta,S}$, ARI gap으로 직접 분해 및 진단|
|**Refit 목적**|Density estimation과 clustering 개선|Mean-effect recovery 및 shrinkage-induced ARI loss 완화|
|**Centering**|Empirical centering의 장단점 및 대안 논의|Sum-to-zero effect coding으로 Empirical centering 없이 $\delta$ 타깃화|
|**Model Selection**|Non-asymptotic criterion, slope heuristic|Refit-likelihood EBIC, sensitivity $\alpha \in \{0, 0.5, 1\}$|
|**논문상 포지션**|Lasso-MLE procedure|Lasso-MLE principle의 mean-effect target 특수화 및 진단 프레임워크 확장|

---

### 4. 정리
|**측면**|**SZL-Refit**|**Meynet (2012) Lasso-MLE**|**중복 여부**|
|---|---|---|---|
|**Stage 1: Lasso screening**|sum-to-zero $\ell_1$|Pan-Shen $\ell_1$|본질적 동일|
|**Stage 2: unpenalized refit on selected support**|O|O (Theorem 6.2.2)|동일|
|**Model selection criterion**|refit-EBIC|slope heuristics|도구 다름, 목적 동일|
|**식별성 처리**|sum-to-zero parameterization (명시)|empirical centering (전처리)|등가|
|**이론 보장**|sure screening + size control + oracle equivalence (목표)|Hellinger oracle inequality (증명 완료)|우리가 후행|
|**응용**|gene expression / single-cell (계획)|functional data (wavelets)|다름|
|**Variable role classification**|$S_0$ vs $S_0^c$ 단일|$(K, J_r, J_a)$ triple|Meynet이 더 정교|
