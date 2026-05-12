## 핵심 선행 연구 지형 분석 및 제안 모형의 차별성

본 연구 관련 기존 문헌을 분석하고 제안 모형(SZL-Refit)과의 관계를 재정리하였습니다. 본 연구의 핵심은 단순히 새로운 penalty를 제안하거나 "Lasso 후 MLE refit"을 수행하는 데 있지 않습니다. 진정한 기여는 **Sum-to-zero mean-effect parameterization**, **Lasso shrinkage bias의 진단과 분해($R_k$)**, 그리고 **선택 변수 위에서 unpenalized GMM refit을 통한 Debiased mean-effect recovery**의 결합에 있습니다.

따라서 본 연구는 새로운 Lasso-MLE 방법론이 아닌, **Debiased mean-effect refit framework**로 포지셔닝합니다.

### 1. 가장 직접적으로 겹치는 방법론

제안 모형과 유사하게 고차원 군집화에서 $\ell_1$ 페널티를 활용하거나 변수 선택 후 재적합을 수행하는 핵심 선행 연구들입니다. 이 논문들과의 명확한 선 긋기가 논문 심사 방어의 핵심입니다.

- **Meynet and Maugis-Rabusseau - Lasso-MLE Procedure**
    
    - **내용:** $\ell_1$-regularization을 사용하여 Data-driven model subcollection을 구성하고, 선택된 각 모델에서 EM 알고리즘을 이용해 Maximum Likelihood Estimator(MLE)를 적합하는 절차를 제안했습니다.
        
    - **차별점:** "Lasso로 후보 모델을 만들고 MLE로 refit한다"는 큰 틀은 본 연구와 가장 직접적으로 겹칩니다. 그러나 Lasso-MLE는 일반적인 변수 선택과 Model selection 관점인 반면, 본 연구는 $\mu_j = \mu_0 + \delta_j$, $\sum_{j=1}^K \delta_{jk}=0$ 기반의 Mean-heterogeneity target $S_0 = {k: \delta_{\cdot k}^0 \neq 0}$를 명시적으로 정의합니다. 나아가 Lasso shrinkage가 군집 간 평균 차이와 ARI에 미치는 영향을 $R_k$로 분해 및 진단하고, Refit-likelihood EBIC 튜닝을 통해 편향을 제거(Debiasing)하는 데 초점을 둡니다.
        
- **Pan and Shen (2007) - $L_1$-penalized model-based clustering**
    
    - **내용:** 고차원 Gaussian model-based clustering에서 $L_1$-penalized likelihood를 EM 알고리즘으로 최적화하며, Thresholding을 통해 변수 선택을 수행합니다.
        
    - **차별점:** Pan & Shen 계열은 Penalized estimator 자체를 Final estimator로 사용합니다. 반면 본 연구는 Lasso estimator를 Screening 도구로만 활용하고, 선택된 변수 위에서 페널티 없이 GMM을 Refit하여 Lasso shrinkage로 인해 축소된 Cluster mean contrast를 복원합니다.
        
- **Xie, Pan, and Shen (2008) - Penalized model-based clustering with grouped variables**
    
    - **내용:** 동일 변수의 여러 군집 관련 파라미터들을 묶어 Shrinkage하는 Grouped variable selection을 제안했습니다.
        
    - **차별점:** Xie et al.은 Raw cluster mean $\mu_{\cdot k}$ 자체를 페널티의 대상으로 삼습니다. 반면 본 연구는 Effects-style parameterization 하에서 Cluster-specific mean deviation $\delta_{\cdot k}$를 타깃으로 삼습니다. 또한 본 연구의 메인 방법인 SZL-Refit은 Group penalty를 쓰지 않고, Element-wise lasso 결과를 Sum-to-zero 제약 하에서 Variable-level mean contrast로 집계하여 $S_0$를 추정합니다.
        

### 2. 타깃 및 구조적 접근 방식이 유사한 방법론

- **Guo et al. (2010):** 어떤 변수로 특정 군집 쌍이 분리되는지 찾는 Pairwise variable selection을 수행합니다. 변수 선택에 '구조적인 타깃'을 둔다는 철학은 공유하나, 대상이 Pairwise separability($\mu_{jk} - \mu_{\ell k}$)입니다. 반면 본 연구는 변수 단위의 Mean-heterogeneity support $S_0$ 자체를 추정하는 것을 최종 목표로 합니다.
    
- **Li et al. (2022):** 지도학습(Supervised) 환경인 유한 혼합 회귀(Finite mixture regression)에서 예측 변수와 이질성 유발 변수(Heterogeneity sources)를 동시에 식별합니다. 본 연구는 반응 변수(Outcome)가 없는 순수 비지도(Unsupervised) Gaussian mixture clustering 환경에서의 대응 모형(Outcome-free analogue)으로 볼 수 있습니다.
    
- **Li et al. (2023, ZINBMM):** Single-cell RNA-seq 카운트 데이터에서 전역 평균(Global mean)과 군집별 평균(Cluster-specific mean)의 차이에 페널티를 부여합니다. 개념적으로 인접하나, 본 연구의 Gaussian mean-effects 분해, Sum-to-zero coding, Refit-likelihood EBIC, Lasso shrinkage recovery ratio($R_k$) 프레임워크와는 다릅니다.
    
- **Celeux et al. (SelvarMix):** Lasso-like ranking을 통해 변수를 정렬하고, 역할(S, R, U, W)을 분류합니다. Target이 단순 Selected/Non-selected가 아닌 Role classification에 있으므로, $S_0$를 직접 타깃으로 하는 본 연구와 차이가 있습니다.
    

---

### 3. 모형 및 방법론 상세 비교표

#### 표 1. 모형 및 방법론 비교표

|**논문 / 방법**|**핵심 문제**|**기본 모형**|**주요 가정**|**식별성 / 제약**|**벌점 구조**|**추정 / 튜닝**|
|---|---|---|---|---|---|---|
|현재 방법론<br><br>  <br><br>(SZL-Refit)|비지도 고차원 Gaussian clustering에서 mean-heterogeneity-driving variables 식별 및 lasso shrinkage debiasing|$P(Z_i=j)=\pi_j$<br><br>  <br><br>$X_i \mid Z_i=j \sim N_p(\mu_j, \Sigma)$<br><br>  <br><br>$\mu_j=\mu_0+\delta_j$|공통 diagonal covariance 또는 $\Sigma=I_p$, mean-shift 중심|$\sum_{j=1}^K \delta_{jk}=0$|Stage 1: element-wise $\ell_1$ penalty<br><br>  <br><br>$\lambda \sum_{k,j} \vert\delta_{jk}\vert$|Lasso screening 후 선택 변수에서 unpenalized GMM Refit, Refit-likelihood EBIC|
|**Meynet and Maugis-Rabusseau (Lasso-MLE)**|Gaussian mixture clustering에서 variable selection을 model selection 문제로 재정의|Gaussian mixture|high-dimensional clustering, 일반 model subcollection 관점|-|$\ell_1$-regularization으로 data-driven model collection 생성|선택 모델에서 EM으로 MLE refit, model selection criterion|
|**Pan and Shen (2007)**|high-dimensional model-based clustering에서 variable selection|Gaussian mixture|high-dimension low-sample setting, raw mean 중심|explicit effects coding 없음|$L_1$-penalized likelihood on cluster means|EM, thresholding, modified BIC|
|**Wang and Zhu (2008)**|model-based high-dimensional clustering에서 informative variable selection|Gaussian mixture|공통 diagonal covariance, centered data|explicit ANOVA형 sum-to-zero coding은 아님|adaptive/hierarchical grouped regularization|EM 기반 penalized estimation|
|**Xie, Pan, and Shen (2008)**|grouped parameters를 통한 penalized model-based clustering|Gaussian mixture|common 또는 cluster-specific diagonal covariance|raw cluster means 중심|grouped mean penalty, grouped variable selection|EM + grouped updates, BIC|
|**Guo et al. (2010)**|어떤 변수로 어떤 cluster pair가 분리되는지 찾는 pairwise variable selection|Gaussian mixture|공통 diagonal covariance, pairwise difference 중심|zero가 특별한 기준이 아님, pairwise fusion 중심|pairwise fusion penalty<br><br>  <br><br>$\sum_{j, k<\ell} \vert\mu_{kj}-\mu_{\ell j}\vert$|-|
|**Li et al. (2022)**|supervised finite mixture regression에서 relevant predictors와 heterogeneity sources 동시 식별|finite mixture regression|response ($Y$) 존재|$\sum_{j=1}^m \beta_{jk} = 0$|common/relevant/heterogeneity penalty|EM, BIC-type tuning|
|**Li et al. (2023, ZINBMM)**|scRNA-seq count data clustering과 gene selection|ZINB mixture model|count data, dropout, batch effects|global mean baseline, sum-to-zero coding 없음|cluster-specific mean과 global mean 차이 penalty|penalized likelihood|
|**SelvarMix**|model-based clustering / discriminant analysis에서 variable role selection|Gaussian mixture / discriminant analysis|high-dimensional setting, role classification 중심|-|lasso-like ranking + role selection|ranking 후 ($S, R, U, W$) variable roles 정의|

---

#### 표 2. 유사성 및 결정적 차이점

| **비교 논문**                                   | **현재 방법과 가장 겹치는 지점**                                                                      | **결정적 차이점**                                                                                                                                                                                         |
| ------------------------------------------- | ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Meynet and Maugis-Rabusseau (Lasso-MLE)** | $\ell_1$-regularization으로 candidate model/support 생성 후 MLE refit 과정이 High-level에서 매우 가깝다. | 본 연구의 독창성은 "Lasso + Refit" 절차 자체가 아니라, Sum-to-zero mean-effect parameterization, $S_0=\{k:\delta_{\cdot k}^0 \neq 0\}$ 타깃 설정, $R_k$ 기반 Shrinkage-bias decomposition, 그리고 Refit-likelihood EBIC에 있다. |
| **Pan and Shen (2007)**                     | Gaussian mixture, $L_1$-penalized likelihood, EM, variable selection을 사용한다.               | Pan & Shen은 Penalized estimator 자체를 Final estimator로 사용한다. 본 연구는 Lasso를 Screening으로만 사용하고 Refit으로 Shrinkage를 제거한다.                                                                                  |
| **Xie, Pan, and Shen (2008)**               | Grouped variable selection과 model-based clustering의 결합.                                   | Xie 등은 Raw cluster mean/grouped parameters를 페널티 대상으로 삼는다. 본 연구는 $\delta_{\cdot k}$ mean-effect support를 타깃으로 하며, 메인 프레임워크(SZL)에서 Group penalty를 강제하지 않는다.                                           |
| **Wang and Zhu (2008)**                     | High-dimensional Gaussian mixture, adaptive/grouped regularization 방식.                    | Wang & Zhu는 Informative-variable selection이 목적이다. 본 연구는 Sum-to-zero mean-effect target $S_0$와 Shrinkage recovery에 초점을 맞춘다.                                                                          |
| **Guo et al. (2010)**                       | Generic variable selection을 넘어 더 구조적인 타깃을 둔다는 점.                                          | Guo 등은 Pairwise separability가 목표이다. 본 연구는 Variable-level mean-heterogeneity support recovery가 목표이다.                                                                                                 |
| **Li et al. (2022)**                        | Common/cluster-specific decomposition, heterogeneity pursuit, EM/BIC framing.             | Li 등은 Supervised finite mixture regression이다. 본 연구는 Outcome-free unsupervised mean-mixture clustering이다.                                                                                            |
| **Li et al. (2023, ZINBMM)**                | Cluster-specific mean과 global mean의 차이에 페널티를 부여한다.                                        | Count/ZINB/scRNA-seq 특화 세팅이다. Gaussian sum-to-zero mean-effect refit 프레임워크는 없다.                                                                                                                     |
| **SelvarMix**                               | Lasso-like ranking과 high-dimensional model-based variable selection.                      | SelvarMix는 변수의 역할(Role) 선택($S, R, U, W$)이 타깃이다. 본 연구는 $S_0=\{k:\delta_{\cdot k}^0 \neq 0\}$ 자체를 직접 타깃으로 한다.                                                                                         |

- 제안하는 SZL-Refit 프레임워크의 핵심 차별성은 "새로운 패널티의 발명"이나 "Refit 기법의 최초 도입"이 아님
- $\mu_j = \mu_0 + \delta_j$ 및 $\sum_{j=1}^K \delta_{jk}=0$ 분해 구조를 기반으로 Baseline-adjusted deviation 벡터($\delta_{\cdot k}$)를 타깃으로 함
- Lasso가 유발하는 군집 분리력 저하(Shrinkage bias)를 $R_k$ 관점에서 진단하여 이를 무벌점(Unpenalized) GMM 재적합으로 완벽히 복원(Debiasing)하는 통합적 시각에 있음
