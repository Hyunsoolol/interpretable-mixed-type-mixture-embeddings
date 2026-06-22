# Thesis Methods Note 260624
**업데이트:** 2026-06-23

**관련 문서:** `docs/implementation/thesis-implementation_260624.md`, `docs/meetings/thesis-meeting_260624.md`, `docs/methods/thesis-algorithm_260624.md`

이 문서는 논문 Methods section의 뼈대로 사용할 수 있도록 모형, penalty, 추정량, 추정 절차, tuning, refit, 한계를 이론 중심으로 정리한 노트다.

---

## 1. Problem Setup

관측값은 단위구 위의 방향자료다. 각 관측치 $x_i \in S^{d-1}$는 $\|x_i\|_2=1$을 만족한다. 목표는 $K$개의 잠재 군집을 추정하는 동시에, 군집 구분에 실제로 기여하는 coordinate support를 sparse하고 해석 가능하게 회복하는 것이다.

### 기호 정의

| 기호 | 의미 |
| :--- | :--- |
| $n$ | number of observations |
| $d$ | ambient dimension |
| $K$ | number of mixture components |
| $x_i$ | $i$번째 방향자료, $x_i \in S^{d-1}$ |
| $z_i$ | latent component label |
| $\alpha_k$ | mixing proportion |
| $\mu_k$ | component mean direction, $\|\mu_k\|_2=1$ |
| $\kappa_k$ | concentration parameter |
| $\eta_k$ | natural decision parameter, $\eta_k=\kappa_k\mu_k$ |
| $S$ | selected coordinate support |

이 연구의 주장은 ARI를 크게 올리는 것이 아니라, vMF mixture 안에서 posterior decision parameter의 component contrast를 sparse하게 만들어 clustering 성능을 유지하면서 해석 가능한 support를 제공하는 것이다.

---

## 2. vMF Mixture Model

기본 모형은 $K$ component von Mises-Fisher mixture다.

$$
f(x_i;\Theta)=\sum_{k=1}^K \alpha_k C_d(\kappa_k)\exp(\kappa_k\mu_k^\top x_i)
$$

여기서 $\alpha_k>0$, $\sum_{k=1}^K \alpha_k=1$, $\lVert\mu_k\rVert_2=1$, $\kappa_k \ge 0$이다. $C_d(\kappa)$는 $d$차원 vMF density의 normalizing constant다.

자연모수 형태로 쓰면 다음과 같다.

$$
\eta_k=\kappa_k\mu_k
$$

$$
f(x_i\mid \eta_k)=C_d(\lVert\eta_k\rVert_2)\exp(\eta_k^\top x_i)
$$

Posterior classification score는 $\log \alpha_k+\log C_d(\lVert\eta_k\rVert_2)+\eta_k^\top x_i$를 포함한다. 따라서 coordinate $j$가 군집 구분에 기여하는지는 $\mu_{kj}$만이 아니라 $\kappa_k\mu_{kj}$, 즉 $\eta_{kj}$의 component 간 차이로 보는 것이 자연스럽다.

---

## 3. Complete-data Log-likelihood and E-step

잠재 label indicator를 $z_{ik}=1(z_i=k)$라고 두면 complete-data log-likelihood는 다음과 같다.

$$
\ell_c(\Theta) = \sum_{i=1}^n\sum_{k=1}^K z_{ik} \left\lbrace \log \alpha_k+\log C_d(\kappa_k)+\kappa_k\mu_k^\top x_i \right\rbrace
$$

현재 parameter $\Theta^{\mathrm{old}}$에서 posterior responsibility는 다음과 같다.

$$
\tau_{ik} = \frac{ \alpha_k C_d(\kappa_k)\exp(\kappa_k\mu_k^\top x_i) }{ \sum_{\ell=1}^K \alpha_\ell C_d(\kappa_\ell)\exp(\kappa_\ell\mu_\ell^\top x_i) }
$$

자연모수로 쓰면 다음과 같다.

$$
\tau_{ik} = \frac{ \alpha_k C_d(\|\eta_k\|_2)\exp(\eta_k^\top x_i) }{ \sum_{\ell=1}^K \alpha_\ell C_d(\|\eta_\ell\|_2)\exp(\eta_\ell^\top x_i) }
$$

구현에서는 log-sum-exp 형태로 계산해 underflow를 줄인다. E-step 이후 필요한 sufficient statistics는 다음이다.

$$
N_k=\sum_{i=1}^n \tau_{ik}, \qquad r_k=\sum_{i=1}^n \tau_{ik}x_i
$$

---

## 4. Unpenalized vMF M-step

Penalty가 없을 때 mixing proportion은 다음처럼 업데이트된다.

$$
\hat{\alpha}_k=\frac{N_k}{n}
$$

Mean direction은 resultant vector 방향으로 추정한다.

$$
\hat{\mu}_k=\frac{r_k}{\|r_k\|_2}
$$

Concentration parameter는 mean resultant length에 의해 결정된다.

$$
\rho_k=\frac{\hat{\mu}_k^\top r_k}{N_k}
$$

정확한 MLE는 $A_d(\kappa_k)=\rho_k$를 푸는 문제다. 현재 구현에서는 널리 쓰이는 근사식을 사용한다.

$$
\hat{\kappa}_k \approx \frac{d\rho_k-\rho_k^3}{1-\rho_k^2}
$$

이 단계는 approximation이다. 특히 $\rho_k$가 1에 가까우면 $\kappa_k$가 커질 수 있으므로 수치 bound가 필요하다. 추정된 $\hat{\mu}_k$, $\hat{\kappa}_k$로부터 자연모수는 $\hat{\eta}_k=\hat{\kappa}_k\hat{\mu}_k$로 복원된다.

> Reference note. 이 concentration update는 Banerjee et al. (2005)의 vMF mixture EM 문헌에서 쓰이는 mean resultant length 기반 approximation과 같은 계열이다. 정확한 M-step은 Bessel function ratio $A_d(\kappa)$의 inverse를 풀어야 하므로, 본 연구에서는 위 식을 closed-form MLE가 아니라 계산 효율을 위한 approximation으로 해석한다.

---

## 5. Baseline Sparse vMF Penalties

### 5.1 Rossi Sparse vMF
Rossi sparse vMF는 component direction에 $\ell_1$ penalty를 둔다.

$$
P_{\mathrm{Rossi}}(\mu) = \beta \sum_{k=1}^K \|\mu_k\|_1
$$

Penalized objective는 log-likelihood에서 이 penalty를 뺀 형태다. Support는 component별 nonzero $\mu_{kj}$의 union으로 정의한다.

현재 비교에서는 fixed grid가 아니라 $\beta$ path를 사용한다. Dense fit에서 시작해 coordinate가 shrink될 수 있는 threshold를 따라 $\beta$ 후보를 만들고, 각 후보에서 fit을 계산한 뒤 BIC가 가장 작은 후보를 선택한다. 이 방식은 sparse prototype baseline으로 적절하지만, penalty target이 $\mu$라서 concentration 차이가 coordinate decision에 어떻게 반영되는지는 직접적으로 분리하지 못한다.

> Reference note. Rossi and Barbaro (2022)는 본 연구의 sparse vMF mixture baseline이다. 해당 방법은 direction/prototype parameter에 $\ell_1$ sparsity를 부여하는 반면, Eta-group은 posterior decision score에 직접 들어가는 natural parameter $\eta_k=\kappa_k\mu_k$의 component contrast를 penalize한다.

### 5.2 Separate Mu/Kappa Penalty
Separate penalty baseline은 $\mu_k$와 $\kappa_k$에 별도 penalty를 둔다.

$$
P_{\mathrm{Separate}}(\mu,\kappa) = \lambda_\mu \sum_{k=1}^K \|\mu_k\|_1 + \lambda_\kappa \sum_{k=1}^K \kappa_k
$$

이 방식은 direction sparsity와 concentration shrinkage를 따로 조정할 수 있다. 그러나 $\lambda_\kappa$는 component-level scalar에 작용하므로, 어떤 coordinate가 concentration-driven separation에 기여하는지 직접 선택하지 못한다. 따라서 coordinate-level interpretability는 여전히 $\mu$ support에 의존한다.

---

## 6. Proposed Eta-group Penalty

### 6.1 Motivation
Posterior decision에는 $\mu_k$가 아니라 $\eta_k=\kappa_k\mu_k$가 선형 score로 들어간다. 따라서 coordinate-level variable selection은 $\eta$의 component contrast에 대해 정의하는 것이 자연스럽다.

Eta-group의 목적은 clustering score에 직접 들어가는 coordinate contrast를 sparse하게 만드는 것이다. 이 방법은 ARI를 항상 높이는 절차가 아니라, vMF mixture 안에서 interpretable support recovery를 개선하기 위한 penalty다.

### 6.2 $K=2$ Eta Contrast Penalty
$K=2$에서는 component 간 decision contrast가 하나다.

$$
\delta=\eta_2-\eta_1
$$

Coordinate $j$의 군집 구분 효과는 $\delta_j$로 표현된다. 이에 대한 penalty는 다음과 같다.

$$
P_{\mathrm{Eta},K=2}(\eta) = \lambda_\eta \sum_{j=1}^d |\delta_j|
$$

선택된 coordinate는 $\hat{\delta}_j \ne 0$인 좌표다. 즉 두 component의 posterior decision score를 구분하는 eta contrast가 남아 있는 coordinate를 active로 본다.

### 6.3 $K>2$ Centered Eta Group Penalty
$K>2$에서는 하나의 pairwise contrast만으로 component 간 차이를 표현하기 어렵다. 각 coordinate $j$에서 component별 eta를 centered form으로 바꾼다.

$$
\bar{\eta}_j = \frac{1}{K}\sum_{k=1}^K \eta_{kj}
$$

$$
c_{kj}=\eta_{kj}-\bar{\eta}_j
$$

Coordinate $j$에서 component 간 eta contrast 크기는 $\|c_{\cdot j}\|_2$로 요약한다. Eta-group penalty는 coordinate별 centered eta vector에 group lasso penalty를 둔다.

$$
P_{\mathrm{Eta-group}}(\eta) = \lambda_\eta\sum_{j=1}^d \left(\sum_{k=1}^K c_{kj}^2\right)^{1/2}
$$

이 penalty는 coordinate 단위로 component contrast 전체를 함께 선택하거나 제거한다. ANOVA-type coordinate-wise $\ell_1$ shrinkage는 component별 centered effect를 더 개별적으로 줄일 수 있지만, pilot comparison에서는 dense support로 가기 쉬웠다. 현재 연구의 목표는 coordinate-level support recovery이므로 centered eta group lasso 형태의 Eta-group을 main penalty로 둔다.

> Reference note. 이 penalty는 Yuan and Lin (2006)의 group lasso 아이디어를 centered eta contrast에 적용한 것이다. 원래 group lasso는 미리 정의된 변수 group을 함께 선택하거나 제거하기 위한 penalty이며, 여기서는 coordinate $j$별 centered eta vector $c_{\cdot j}$를 하나의 group으로 본다.

### 6.4 Adaptive Eta-group Penalty as a Diagnostic Extension

Adaptive Eta-group penalty는 현재 official method가 아니라 penalty weighting sensitivity를 확인하기 위한 diagnostic extension이다. 기본 Eta-group penalty는 모든 coordinate에 같은 penalty weight를 둔다. Adaptive version은 초기 dense 또는 low-penalty fit에서 얻은 centered eta norm을 이용해 coordinate별 penalty weight를 다르게 둔다.

초기 centered eta contrast norm을

$$
a_j = \|c_{\cdot j}^{\mathrm{init}}\|_2
$$

라고 두고, adaptive weight를 다음처럼 정의한다.

$$
w_j = (a_j+\epsilon)^{-\gamma}
$$

구현에서는 numerical stability를 위해 $\epsilon>0$을 두고, weight scale이 전체 penalty level과 혼동되지 않도록 median normalization을 사용했다.

$$
w_j \leftarrow \frac{w_j}{\mathrm{median}(w_1,\ldots,w_d)}
$$

Adaptive Eta-group penalty는 다음과 같다.

$$
P_{\mathrm{adaptive}}(\eta) =
\lambda_\eta \sum_{j=1}^d w_j \|c_{\cdot j}\|_2
$$

이에 대응하는 proximal shrinkage step은 coordinate별 threshold가 달라진다.

$$
c_{\cdot j}^{\mathrm{new}}
=
\left(1-\frac{\lambda_\eta w_j}{\|c_{\cdot j}^{0}\|_2}\right)_+c_{\cdot j}^{0}
$$

진단 결과, adaptive penalty는 strong/weak $d=100$ setting에서는 selected $q$와 FPR을 낮추고 Precision/F1을 개선했다. 그러나 $d=200$에서는 adaptive penalty alone이 dense support 문제를 해결하지 못했고, long path와 결합해야 개선됐다. 더 중요하게는 $d=400$ high-dimensional stress에서 long path와 결합해도 selected $q=308.00$, FPR=0.760으로 악화되었다. 따라서 adaptive penalty는 현재 official algorithm으로 올리지 않고, appendix-level diagnostic candidate로만 둔다.

---

## 7. Penalized Objective

각 방법은 다음 형태의 penalized objective를 최대화한다.

$$
\mathcal{Q}_{\mathrm{pen}}(\Theta) = \ell(\Theta)-P(\Theta)
$$

| 방법 | penalty target | penalty |
| :--- | :--- | :--- |
| Rossi sparse vMF | $\mu$ | $\beta\sum_k\lVert\mu_k\rVert_1$ |
| Separate | $\mu, \kappa$ | $\lambda_\mu\sum_k\lVert\mu_k\rVert_1+\lambda_\kappa\sum_k\kappa_k$ |
| Eta-group | centered $\eta$ contrast | $\lambda_\eta\sum_j\lVert c_{\cdot j}\rVert_2$ |
| Adaptive Eta-group | centered $\eta$ contrast | $\lambda_\eta\sum_j w_j\lVert c_{\cdot j}\rVert_2$ |

Adaptive Eta-group row는 diagnostic extension이며 current official objective는 Eta-group이다. Eta-group의 BIC 계산에는 implementation-level degrees of freedom approximation을 사용한다. 이는 model selection을 위한 계산 규칙이며, 엄밀한 effective degrees of freedom 이론으로 주장하지 않는다.

---

## 8. Proximal EM-type Estimation

Eta-group의 penalized M-step은 vMF normalizing constant $C_d(\|\eta_k\|_2)$와 centered eta group penalty가 동시에 얽혀 있어 closed-form penalized update로 직접 풀기 어렵다. 현재 구현은 proximal EM-type update다.

기본 반복은 다음과 같다.

1. 현재 parameter에서 standard mixture E-step을 수행한다.
2. Unpenalized vMF M-step으로 candidate $\eta_k^{0}$를 만든다.
3. Candidate eta를 coordinate별 centered eta $c_{\cdot j}^{0}$로 바꾼다.
4. Centered eta에 group soft-thresholding을 적용한다.
5. Shrink된 eta에서 $\kappa_k=\|\eta_k\|_2$, $\mu_k=\eta_k/\|\eta_k\|_2$를 복원한다.
6. Penalized objective가 악화되는 경우 이전 iterate와 candidate 사이에서 step-halving line-search를 적용한다.

Group soft-thresholding은 다음과 같다.

$$
c_{\cdot j}^{\mathrm{new}} = \left(1-\frac{\lambda_\eta}{\|c_{\cdot j}^{0}\|_2}\right)_+ c_{\cdot j}^{0}
$$

Adaptive diagnostic을 켠 경우에는 $\lambda_\eta$ 대신 coordinate별 $\lambda_\eta w_j$가 threshold로 들어간다. 이 변경은 shrinkage step만 바꾸며, official algorithm으로 확정된 것은 아니다.

Centered eta를 shrink한 뒤에는 coordinate mean $\bar{\eta}_ j$를 더해 $\eta_{kj}^{\mathrm{new}}$를 구성한다.

$$
\eta_{kj}^{\mathrm{new}}=\bar{\eta}_j+c_{kj}^{\mathrm{new}}
$$

> Reference note. 이 update는 group lasso penalty의 proximal operator, 즉 block soft-thresholding이다. Yuan and Lin (2006)의 group lasso penalty와 Parikh and Boyd (2014)의 proximal algorithms 문헌에서 표준적으로 쓰이는 shrinkage 형태와 같은 계열이다. 여기서 이 식은 전체 mixture objective가 convex라는 주장이 아니라, proximal EM-type update 내부에서 centered eta candidate를 shrink하는 step으로 사용된다.

Line-search safeguard는 implementation-level 안정화 장치다. Objective trace smoke test에서 `n_decrease = 0`을 확인했지만, 이것이 전역 수렴이나 이론적 단조성 증명을 의미하지는 않는다.

---

## 9. Tuning Path and Model Selection

현재 공식 tuning은 path+BIC다. 각 penalty parameter path에서 후보 fit을 만들고, 다음 BIC가 가장 작은 후보를 선택한다.

$$
\mathrm{BIC} = \log(n)\,\mathrm{df} - 2\ell(\hat{\Theta})
$$

EBIC, RIC-like criteria, positive-support BIC, adaptive refinement, stability selection, adaptive penalty weighting은 diagnostic 또는 sensitivity로만 둔다. High-dimensional setting에서는 기본 path가 중간 support 후보를 충분히 만들지 못할 수 있다. 이 경우 penalty 기준만 바꾸어도 선택이 거의 달라지지 않는다.

Long path 240 diagnostic은 $d=200$과 $d=400$에서 selected $q$, FPR, Precision, F1을 개선했다. 그러나 true union $q$ 근처 support를 안정적으로 회복하지는 못했으므로 official tuning change로 확정하지 않는다.

Adaptive penalty diagnostic은 $d=100$ strong/weak setting에서는 유망했지만, $d=400$ stress에서 dense support로 악화되었다. 따라서 current official tuning/selection rule은 Eta-group path+BIC + refit으로 유지한다.

---

## 10. Refit After Selection

Refit은 variable selection 후 shrinkage bias를 줄이기 위한 단계다. Penalized fit에서 선택된 support $S$를 고정하고, 그 support 안에서 penalty 없이 vMF mixture를 다시 추정한다.

| 항목 | refit에서의 처리 |
| :--- | :--- |
| support | penalized fit의 selected coordinate를 고정 |
| penalty | 사용하지 않음 |
| parameter | $\alpha_k, \mu_k, \kappa_k$ 재추정 |
| posterior label | refit posterior로 다시 계산 |

Refit은 support를 바꾸는 단계가 아니다. Zero support가 선택된 경우 support-constrained refit은 정의하기 어렵기 때문에 해당 row를 `zero_active_support`로 기록하고, 전체 repetition을 ERROR로 처리하지 않는다.

> Reference note. 이 post-selection refit은 high-dimensional model-based clustering에서 variable selection 후 선택된 변수 subset 또는 selected model을 기준으로 clustering model을 다시 평가/추정하는 흐름과 관련이 있다. Celeux, Maugis-Rabusseau, and Sedki (2017)는 lasso-like regularization으로 변수를 먼저 ranking한 뒤, model-selection criterion으로 clustering에 관련된 변수 역할을 결정하는 절차를 제안한다. 본 연구의 refit은 그 아이디어를 그대로 복제한 것은 아니고, Eta-group penalty가 선택한 support $S$를 고정한 뒤 penalty 없이 $\alpha_k,\mu_k,\kappa_k$를 재추정하여 shrinkage bias를 줄이는 vMF mixture용 post-selection step이다. 따라서 refit은 support를 다시 선택하는 단계가 아니라 selected support fixed unpenalized refit으로 해석한다.

---

## 11. Degrees of Freedom Approximation

Eta-group의 BIC에는 다음 implementation-level df approximation을 사용한다.

$$
\mathrm{df}_{\eta}=(K-1)+d+(K-1)m
$$

여기서 $m$은 selected coordinate count다. $(K-1)$은 mixing proportion 자유도, $d$는 coordinate별 공통 eta baseline, $(K-1)m$은 선택된 coordinate에서 centered component contrast의 자유도를 근사한다.

이 $\mathrm{df}$는 BIC 계산을 위한 approximation이다. 엄밀한 effective $\mathrm{df}$ 유도는 아직 별도 과제이며, 논문에서는 sensitivity와 limitation을 함께 제시하는 편이 안전하다.

> Reference note. Path+BIC는 penalized mixture model에서 tuning parameter를 고르는 실용적 기준으로 사용한다. 다만 위 $\mathrm{df}_{\eta}$는 엄밀한 effective degrees of freedom이 아니라 implementation-level approximation이다. 따라서 EBIC/RIC-like criteria와 path diagnostic은 official rule이 아니라 sensitivity로 보고하는 것이 안전하다.

---

## 12. Diagnostics and Current Limitations

현재 구현과 결과 기준의 주요 diagnostic 및 한계는 다음과 같다.

| 항목 | 현재 상태 |
| :--- | :--- |
| objective trace | line-search 이후 smoke tests에서 `n_decrease = 0` 확인 |
| line-search | step-halving safeguard로 objective decrease를 줄임 |
| weak setting | support plateau와 path/BIC instability 가능 |
| high-dimensional setting | 기본 path가 dense support로 가기 쉬움 |
| long path | FPR/Precision/F1 개선, true support size 회복은 제한적 |
| adaptive penalty | strong/weak $d=100$ 개선, $d=400$ stress 실패 |
| kappa approximation | 고차원 또는 weak signal에서 outlier 가능 |
| diagnostic methods | positive-support/adaptive/stability는 appendix 후보 |

다음 보강 후보는 path construction 개선, MM 또는 coordinate update 개선, high-dimensional screening으로 생각함.

---

## References Mentioned

- Banerjee, A., Dhillon, I. S., Ghosh, J., and Sra, S. (2005). *Clustering on the Unit Hypersphere using von Mises-Fisher Distributions*. Journal of Machine Learning Research, 6(46), 1345-1382.
- Celeux, G., Maugis-Rabusseau, C., and Sedki, M. (2017). *Variable selection in model-based clustering and discriminant analysis with a regularization approach*. arXiv:1705.00946.
- Rossi, F. and Barbaro, F. (2022). *Mixture of von Mises-Fisher distribution with sparse prototypes*. Neurocomputing, 501, 41-74. DOI: 10.1016/j.neucom.2022.05.118.
- Yuan, M. and Lin, Y. (2006). *Model Selection and Estimation in Regression with Grouped Variables*. Journal of the Royal Statistical Society: Series B, 68(1), 49-67. DOI: 10.1111/j.1467-9868.2005.00532.x.
- Parikh, N. and Boyd, S. (2014). *Proximal Algorithms*. Foundations and Trends in Optimization, 1(3), 123-231.
