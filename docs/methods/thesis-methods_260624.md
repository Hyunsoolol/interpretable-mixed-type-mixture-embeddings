# Thesis Methods Note 260624

업데이트: 2026-06-17
관련 문서: `docs/implementation/thesis-implementation_260624.md`, `docs/meetings/thesis-meeting_260624.md`

이 문서는 논문 Methods section의 뼈대로 사용할 수 있도록 모형, penalty, 추정량, 추정 절차, tuning, refit, 한계를 이론 중심으로 정리한 노트다.

## 1. Problem Setup

관측값은 단위구 위의 방향자료다. 각 관측치 $x_i \in S^{d-1}$는 $\|x_i\|_2=1$을 만족한다. 목표는 $K$개의 잠재 군집을 추정하는 동시에, 군집 구분에 실제로 기여하는 coordinate support를 sparse하고 해석 가능하게 회복하는 것이다.

| 기호 | 의미 |
|:---|:---|
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

## 2. vMF Mixture Model

기본 모형은 $K$ component von Mises-Fisher mixture다.

$$
f(x_i;\Theta)=\sum_{k=1}^K \alpha_k C_d(\kappa_k)\exp(\kappa_k\mu_k^\top x_i)
$$

여기서 $\alpha_k>0$, $\sum_{k=1}^K \alpha_k=1$, $\|\mu_k\|_2=1$, $\kappa_k \ge 0$이다. $C_d(\kappa)$는 $d$차원 vMF density의 normalizing constant다.

자연모수 형태로 쓰면 다음과 같다.

$$
\eta_k=\kappa_k\mu_k
$$

$$
f(x_i\mid \eta_k)=C_d(\|\eta_k\|_2)\exp(\eta_k^\top x_i)
$$

Posterior classification score는 $\log \alpha_k+\log C_d(\|\eta_k\|_2)+\eta_k^\top x_i$를 포함한다. 따라서 coordinate $j$가 군집 구분에 기여하는지는 $\mu_{kj}$만이 아니라 $\kappa_k\mu_{kj}$, 즉 $\eta_{kj}$의 component 간 차이로 보는 것이 자연스럽다.

## 3. Complete-data Log-likelihood and E-step

잠재 label indicator를 $z_{ik}=1(z_i=k)$라고 두면 complete-data log-likelihood는 다음과 같다.

$$\ell_c(\Theta) = \sum_{i=1}^n\sum_{k=1}^K z_{ik} \left\{ \log \alpha_k+\log C_d(\kappa_k)+\kappa_k\mu_k^\top x_i \right\}.$$

현재 parameter $\Theta^{old}$에서 posterior responsibility는 다음과 같다.


$$
\tau_{ik} = \frac{
\alpha_k C_d(\kappa_k)\exp(\kappa_k\mu_k^\top x_i)
}{
\sum_{\ell=1}^K
\alpha_\ell C_d(\kappa_\ell)\exp(\kappa_\ell\mu_\ell^\top x_i)
}.
$$

자연모수로 쓰면 다음과 같다.

$$\tau_{ik}
=
\frac{
\alpha_k C_d(\|\eta_k\|_2)\exp(\eta_k^\top x_i)
}{
\sum_{\ell=1}^K
\alpha_\ell C_d(\|\eta_\ell\|_2)\exp(\eta_\ell^\top x_i)
}.$$

구현에서는 log-sum-exp 형태로 계산해 underflow를 줄인다. E-step 이후 필요한 sufficient statistics는 다음이다.

$$N_k=\sum_{i=1}^n \tau_{ik},
\qquad
r_k=\sum_{i=1}^n \tau_{ik}x_i.$$

## 4. Unpenalized vMF M-step

Penalty가 없을 때 mixing proportion은 다음처럼 업데이트된다.

$$
\hat{\alpha}_k=\frac{N_k}{n}.
$$

Mean direction은 resultant vector 방향으로 추정한다.

$$
\hat{\mu}_k=\frac{r_k}{\|r_k\|_2}.
$$

Concentration parameter는 mean resultant length에 의해 결정된다.

$$
\rho_k=\frac{\hat{\mu}_k^\top r_k}{N_k}.
$$

정확한 MLE는 $A_d(\kappa_k)=\rho_k$를 푸는 문제다. 현재 구현에서는 널리 쓰이는 근사식을 사용한다.

$$
\hat{\kappa}_k
\approx
\frac{d\rho_k-\rho_k^3}{1-\rho_k^2}.
$$

이 단계는 approximation이다. 특히 $\rho_k$가 1에 가까우면 $\kappa_k$가 커질 수 있으므로 수치 bound가 필요하다. 추정된 $\hat{\mu}_k$, $\hat{\kappa}_k$로부터 자연모수는 $\hat{\eta}_k=\hat{\kappa}_k\hat{\mu}_k$로 복원된다.

## 5. Baseline Sparse vMF Penalties

### 5.1 Rossi Sparse vMF

Rossi sparse vMF는 component direction에 $\ell_1$ penalty를 둔다.

$$
P_{\mathrm{Rossi}}(\mu)
=
\beta \sum_{k=1}^K \|\mu_k\|_1.
$$

Penalized objective는 log-likelihood에서 이 penalty를 뺀 형태다. Support는 component별 nonzero $\mu_{kj}$의 union으로 정의한다.

현재 비교에서는 fixed grid가 아니라 $\beta$ path를 사용한다. Dense fit에서 시작해 coordinate가 shrink될 수 있는 threshold를 따라 $\beta$ 후보를 만들고, 각 후보에서 fit을 계산한 뒤 BIC가 가장 작은 후보를 선택한다. 이 방식은 sparse prototype baseline으로 적절하지만, penalty target이 $\mu$라서 concentration 차이가 coordinate decision에 어떻게 반영되는지는 직접적으로 분리하지 못한다.

### 5.2 Separate Mu/Kappa Penalty

Separate penalty baseline은 $\mu_k$와 $\kappa_k$에 별도 penalty를 둔다.

$$
P_{\mathrm{Separate}}(\mu,\kappa)
=
\lambda_\mu \sum_{k=1}^K \|\mu_k\|_1
+
\lambda_\kappa \sum_{k=1}^K \kappa_k.
$$

이 방식은 direction sparsity와 concentration shrinkage를 따로 조정할 수 있다. 그러나 $\lambda_\kappa$는 component-level scalar에 작용하므로, 어떤 coordinate가 concentration-driven separation에 기여하는지 직접 선택하지 못한다. 따라서 coordinate-level interpretability는 여전히 $\mu$ support에 의존한다.

## 6. Proposed Eta-group Penalty

### 6.1 Motivation

Posterior decision에는 $\mu_k$가 아니라 $\eta_k=\kappa_k\mu_k$가 선형 score로 들어간다. 따라서 coordinate-level variable selection은 $\eta$의 component contrast에 대해 정의하는 것이 자연스럽다.

Eta-group의 목적은 clustering score에 직접 들어가는 coordinate contrast를 sparse하게 만드는 것이다. 이 방법은 ARI를 항상 높이는 절차가 아니라, vMF mixture 안에서 interpretable support recovery를 개선하기 위한 penalty다.

### 6.2 K=2 Eta Contrast Penalty

$K=2$에서는 component 간 decision contrast가 하나다.

$$
\delta=\eta_2-\eta_1.
$$

Coordinate $j$의 군집 구분 효과는 $\delta_j$로 표현된다. 이에 대한 penalty는 다음과 같다.

$$
P_{\mathrm{Eta},K=2}(\eta)
=
\lambda_\eta \sum_{j=1}^d |\delta_j|.
$$

선택된 coordinate는 $\hat{\delta}_j \ne 0$인 좌표다. 즉 두 component의 posterior decision score를 구분하는 eta contrast가 남아 있는 coordinate를 active로 본다.

### 6.3 K>2 Centered Eta Group Penalty

$K>2$에서는 하나의 pairwise contrast만으로 component 간 차이를 표현하기 어렵다. 각 coordinate $j$에서 component별 eta를 centered form으로 바꾼다.

$$
\bar{\eta}_j = \frac{1}{K}\sum_{k=1}^K \eta_{kj}
$$

$$
c_{kj}=\eta_{kj}-\bar{\eta}_j
$$

Coordinate $j$에서 component 간 eta contrast 크기는 $\|c_{\cdot j}\|_2$로 요약한다. Eta-group penalty는 coordinate별 centered eta vector에 group lasso penalty를 둔다.

$$
P_{\mathrm{Eta-group}}(\eta)
=
\lambda_\eta\sum_{j=1}^d
\left(\sum_{k=1}^K c_{kj}^2\right)^{1/2}.
$$

이 penalty는 coordinate 단위로 component contrast 전체를 함께 선택하거나 제거한다. ANOVA-type coordinate-wise $\ell_1$ shrinkage는 component별 centered effect를 더 개별적으로 줄일 수 있지만, pilot comparison에서는 dense support로 가기 쉬웠다. 현재 연구의 목표는 coordinate-level support recovery이므로 centered eta group lasso 형태의 Eta-group을 main penalty로 둔다.

## 7. Penalized Objective

각 방법은 다음 형태의 penalized objective를 최대화한다.

$$
\mathcal{Q}_{\mathrm{pen}}(\Theta)
=
\ell(\Theta)-P(\Theta).
$$

| 방법 | penalty target | penalty |
|:---|:---|:---|
| Rossi sparse vMF | $\mu$ | $\beta\sum_k\|\mu_k\|_1$ |
| Separate | $\mu,\kappa$ | $\lambda_\mu\sum_k\|\mu_k\|_1+\lambda_\kappa\sum_k\kappa_k$ |
| Eta-group | centered $\eta$ contrast | $\lambda_\eta\sum_j\|c_{\cdot j}\|_2$ |

Eta-group의 BIC 계산에는 implementation-level degrees of freedom approximation을 사용한다. 이는 model selection을 위한 계산 규칙이며, 엄밀한 effective degrees of freedom 이론으로 주장하지 않는다.

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
c_{\cdot j}^{new}
=
\left(1-\frac{\lambda_\eta}{\|c_{\cdot j}^{0}\|_2}\right)_+
c_{\cdot j}^{0}.
$$

Centered eta를 shrink한 뒤에는 coordinate mean $\bar{\eta}_j$를 더해 $\eta_{kj}^{new}$를 구성한다.

$$
\eta_{kj}^{new}=\bar{\eta}_j+c_{kj}^{new}.
$$

Line-search safeguard는 implementation-level 안정화 장치다. Objective trace smoke test에서 `n_decrease = 0`을 확인했지만, 이것이 전역 수렴이나 이론적 단조성 증명을 의미하지는 않는다.

## 9. Tuning Path and Model Selection

현재 공식 tuning은 path+BIC다. 각 penalty parameter path에서 후보 fit을 만들고, 다음 BIC가 가장 작은 후보를 선택한다.

$$
\mathrm{BIC}
=
\log(n)\,df
-
2\ell(\hat{\Theta}).
$$

EBIC, RIC-like criteria, positive-support BIC, adaptive refinement, stability selection은 diagnostic 또는 sensitivity로만 둔다. High-dimensional setting에서는 기본 path가 중간 support 후보를 충분히 만들지 못할 수 있다. 이 경우 penalty 기준만 바꾸어도 선택이 거의 달라지지 않는다.

Long path 240 diagnostic은 d=200과 d=400에서 selected q, FPR, Precision, F1을 개선했다. 그러나 true union q 근처 support를 안정적으로 회복하지는 못했으므로 official tuning change로 확정하지 않는다.

## 10. Refit After Selection

Refit은 variable selection 후 shrinkage bias를 줄이기 위한 단계다. Penalized fit에서 선택된 support $S$를 고정하고, 그 support 안에서 penalty 없이 vMF mixture를 다시 추정한다.

| 항목 | refit에서의 처리 |
|:---|:---|
| support | penalized fit의 selected coordinate를 고정 |
| penalty | 사용하지 않음 |
| parameter | $\alpha_k$, $\mu_k$, $\kappa_k$ 재추정 |
| posterior label | refit posterior로 다시 계산 |

Refit은 support를 바꾸는 단계가 아니다. Zero support가 선택된 경우 support-constrained refit은 정의하기 어렵기 때문에 해당 row를 `zero_active_support`로 기록하고, 전체 repetition을 ERROR로 처리하지 않는다.

## 11. Degrees of Freedom Approximation

Eta-group의 BIC에는 다음 implementation-level df approximation을 사용한다.

$$
df_{\eta}=(K-1)+d+(K-1)m.
$$

여기서 $m$은 selected coordinate count다. $(K-1)$은 mixing proportion 자유도, $d$는 coordinate별 공통 eta baseline, $(K-1)m$은 선택된 coordinate에서 centered component contrast의 자유도를 근사한다.

이 df는 BIC 계산을 위한 approximation이다. 엄밀한 effective df 유도는 아직 별도 과제이며, 논문에서는 sensitivity와 limitation을 함께 제시하는 편이 안전하다.

## 12. Diagnostics and Current Limitations

현재 구현과 결과 기준의 주요 diagnostic 및 한계는 다음과 같다.

| 항목 | 현재 상태 |
|:---|:---|
| objective trace | line-search 이후 smoke tests에서 `n_decrease = 0` 확인 |
| line-search | step-halving safeguard로 objective decrease를 줄임 |
| weak setting | support plateau와 path/BIC instability 가능 |
| high-dimensional setting | 기본 path가 dense support로 가기 쉬움 |
| long path | FPR/Precision/F1 개선, true support size 회복은 제한적 |
| kappa approximation | 고차원 또는 weak signal에서 outlier 가능 |
| diagnostic methods | positive-support/adaptive/stability는 appendix 후보 |

다음 보강 후보는 path construction 개선, MM 또는 coordinate update 개선, high-dimensional screening이다.

## 13. What Can Be Claimed in the Paper

현재 결과와 구현으로 비교적 안전하게 주장할 수 있는 내용은 다음이다.

1. Eta-group provides sparse interpretable support inside a vMF mixture framework.
2. In the main strong common+specific setting, Eta-group preserves clustering performance while substantially reducing false positive coordinate selection.
3. Compared with dense sparse-vMF baselines, Eta-group better targets coordinate-level posterior decision contrasts.
4. Refit after selection can reduce shrinkage bias without changing the selected support.

조심해야 할 내용은 다음이다.

1. Penalized update에 대해 closed-form EM theory를 주장하지 않는다.
2. 전역 최적해 도달을 주장하지 않는다.
3. 모든 setting에서 ARI가 더 좋다고 주장하지 않는다.
4. High-dimensional support recovery가 해결됐다고 주장하지 않는다.
5. Real data 결과를 discovery claim으로 과장하지 않는다.
