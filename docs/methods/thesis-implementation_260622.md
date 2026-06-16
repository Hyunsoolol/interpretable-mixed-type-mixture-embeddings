# 모형, 추정 방법, 코드 구현 정리

업데이트: 2026-06-11
이 문서는 260622 연구미팅에서 구현 관련 질문에 답하기 위한 정리다. 연구미팅용 요약은 `thesis-meeting_260622.md`, 전체 시뮬레이션 결과는 `thesis-simulation_260611.md`에 분리했다.

## 1. 현재 공식 비교 기준

260622 연구미팅 자료에서 사용하는 공식 기준은 다음과 같다.

| 항목 | 기준 |
|:---|:---|
| 비교 방법 | Rossi, Rossi + refit, Separate, Separate + refit, Eta-group, Eta-group + refit |
| tuning 후보 | path 기반 후보 생성 |
| tuning 선택 | BIC 최소 지점 |
| EBIC | 고차원 setting의 보조 지표 |
| refit | 선택된 coordinate support를 고정하고 penalty 없이 vMF mixture 재추정 |
| K | 시뮬레이션별로 K=2 또는 K=4 고정 |
| 주요 결과 파일 | `thesis-simulation_260611.md` |

주의할 점은 예전 prototype 코드에는 고정 grid 방식 실험이 남아 있지만, 현재 연구미팅 자료의 공식 비교는 path tuning + BIC 기준이라는 것이다.

## 2. vMF Mixture 모형

관측값 $x_i$는 단위구 $S^{d-1}$ 위의 방향자료다. $K$개 component를 갖는 vMF mixture는 다음과 같다.

$$f(x_i) = \sum_{k=1}^{K} \alpha_k C_d(\kappa_k)\exp(\kappa_k\mu_k^T x_i).$$

| 기호 | 의미 |
|:---|:---|
| $\alpha_k$ | component mixing proportion |
| $\mu_k$ | component direction, $\|\mu_k\|_2=1$ |
| $\kappa_k$ | concentration parameter |
| $C_d(\kappa_k)$ | vMF normalizing constant |

자연모수 관점에서는 다음처럼 쓴다.

$$\eta_k = \kappa_k\mu_k.$$

그러면 density는 다음과 같다.

$$f(x_i\mid\eta_k) = C_d(\|\eta_k\|_2)\exp(\eta_k^T x_i).$$

이 표현이 중요한 이유는 posterior classification에 $\eta_k^T x_i$가 직접 들어가기 때문이다.

## 3. E-step

현재 parameter $\Theta$가 주어졌을 때 posterior responsibility는 다음과 같다.

$$\tau_{ik} = \frac{\alpha_k C_d(\kappa_k)\exp(\kappa_k\mu_k^T x_i)}{\sum_{\ell=1}^{K}\alpha_\ell C_d(\kappa_\ell)\exp(\kappa_\ell\mu_\ell^T x_i)}.$$

자연모수로 쓰면 다음과 같다.

$$\tau_{ik} = \frac{\alpha_k C_d(\|\eta_k\|_2)\exp(\eta_k^T x_i)}{\sum_{\ell=1}^{K}\alpha_\ell C_d(\|\eta_\ell\|_2)\exp(\eta_\ell^T x_i)}.$$

코드에서는 log scale로 계산한 뒤 row-wise normalization을 하여 underflow를 줄인다. 관련 함수는 `e_step_vmf()`다.

## 4. 공통 M-step

E-step 후 sufficient statistics는 다음이다.

$$N_k = \sum_i \tau_{ik}, \qquad r_k = \sum_i \tau_{ik}x_i.$$

Mixing proportion은 다음처럼 업데이트한다.

$$\alpha_k = N_k/n.$$

패널티가 없으면 평균 방향은 $r_k$ 방향이다.

$$\mu_k = r_k/\|r_k\|_2.$$

집중도는 다음 값을 만든 뒤 $A_d(\kappa_k)=\rho_k$를 푸는 문제다.

$$\rho_k = \mu_k^T r_k/N_k.$$

정확히는 $A_d^{-1}(\rho_k)$를 계산해야 하지만, 현재 구현에서는 다음 근사식을 사용한다.

$$\kappa_k \approx \frac{d\rho_k-\rho_k^3}{1-\rho_k^2}.$$

이 근사식은 vMF mixture EM에서 널리 쓰이는 Banerjee et al. (2005)의 concentration update approximation에 기반한다. 방향자료의 vMF MLE와 $A_d(\kappa)$ 식은 Mardia and Jupp (2000), inverse approximation 논의는 Sra (2012)를 함께 참고할 수 있다.

구현 위치는 `estimate_kappa()`다. 코드에서는 $\rho_k$가 1에 너무 가까워질 때 수치적으로 발산하지 않도록 작은 bound를 둔다.

## 5. 초기값과 multi-start

초기값은 다음 흐름으로 만든다.

1. 데이터에서 $K$개 관측치를 무작위로 선택해 초기 $\mu_k$로 둔다.
2. 각 관측치를 cosine similarity가 가장 큰 component에 hard assignment한다.
3. 초기 assignment로 $\alpha_k$, $r_k$, $\kappa_k$를 계산한다.
4. 여러 random start를 수행하고, penalized objective 또는 log-likelihood가 가장 좋은 해를 선택한다.

관련 함수는 `init_vmf_mixture()`, `fit_svMF_multistart()`다.

수렴 기준은 상대 objective 변화량이다.

$$\frac{|Q^{(t)}-Q^{(t-1)}|}{\max(1,|Q^{(t-1)}|)} < tol.$$

기본 tolerance는 대체로 `1e-7` 또는 driver에서 지정한 값을 사용한다. M-step 중 모든 coordinate가 0으로 shrink되어 direction을 만들 수 없으면 해당 fit은 실패로 처리한다.

## 6. Rossi & Barbaro (2022) Sparse vMF

Rossi & Barbaro (2022)는 component direction $\mu_k$에 L1 penalty를 둔다.

$$\ell_{\mathrm{pen}}(\Theta) = \ell(\Theta) - \beta\sum_{k=1}^{K}\|\mu_k\|_1.$$

E-step은 기본 vMF mixture와 같다. 차이는 M-step에서 $\mu_k$ update가 soft-thresholding 형태가 된다는 점이다. 현재 $\kappa_k$가 주어졌을 때 coordinate별 score는 $\kappa_k |r_{kj}|$이고, update는 다음과 같다.

$$z_{kj} = \mathrm{sgn}(r_{kj})(\kappa_k |r_{kj}|-\beta)_+.$$

그 다음 정규화한다.

$$\mu_k = z_k/\|z_k\|_2.$$

이후 $\rho_k = \mu_k^T r_k/N_k$를 계산하고 `estimate_kappa()`로 $\kappa_k$를 업데이트한다. 실제 구현에서는 $\mu_k$와 $\kappa_k$가 서로 영향을 주므로 inner loop에서 반복한다.

관련 함수는 `update_mu_kappa_one()`, `fit_svMF_em()`, `fit_svMF_path()`다.

### 6.1. Rossi beta path

Rossi 방법은 단순 fixed grid가 아니라 path-following 방식으로 $\beta$ 후보를 만든다.

1. $\beta=0$에서 dense vMF mixture를 적합한다.
2. 현재 fit에서 각 coordinate의 threshold margin을 계산한다.
3. 아직 active인 coordinate 중 다음으로 0이 될 수 있는 가장 작은 threshold로 $\beta$를 증가시킨다.
4. 이전 fit을 warm start로 사용해 새 $\beta$에서 EM을 수행한다.
5. 이 과정을 `max_path_steps`까지 반복한다.
6. path 위 후보 중 BIC가 가장 작은 fit을 선택한다.

개념적으로 다음 threshold를 따라 움직인다.

$$\beta_{\mathrm{next}} \approx \min_{k,j:\ \kappa_k|r_{kj}|>\beta}\kappa_k|r_{kj}|.$$

구현에서는 너무 작은 증가로 path가 정체되지 않도록 `min_rel_beta` 또는 유사한 relative increment를 둔다.

## 7. 분리 패널티 EM

분리 패널티는 교수님 제안에 따라 $\mu_k$와 $\kappa_k$에 penalty를 따로 둔 baseline이다.

$$\ell_{\mathrm{pen}}(\Theta) = \ell(\Theta) - \lambda_\mu\sum_k\|\mu_k\|_1 - \lambda_\kappa\sum_k\kappa_k.$$

E-step은 기본 vMF mixture와 같다. M-step에서 $\mu_k$ update는 Rossi와 같은 형태다.

$$z_{kj} = \mathrm{sgn}(r_{kj})(\kappa_k |r_{kj}|-\lambda_\mu)_+, \qquad \mu_k = z_k/\|z_k\|_2.$$

차이는 concentration penalty가 $\kappa_k$ update에 들어간다는 점이다.

$$\rho_k = \frac{\mu_k^T r_k-\lambda_\kappa}{N_k}.$$

그 다음 `estimate_kappa()` 근사식을 사용한다.

$$\kappa_k \approx \frac{d\rho_k-\rho_k^3}{1-\rho_k^2}.$$

관련 함수는 `update_mu_kappa_separate_one()`, `fit_separate_penalty_em()`이다.

### 7.1. 분리 패널티 path/grid tuning

분리 패널티에는 tuning parameter가 두 개 있다. 현재 공식 비교에서는 $\lambda_\kappa$는 데이터 기반 grid로 두고, 각 $\lambda_\kappa$에서 $\lambda_\mu$ path를 만든다.

절차는 다음이다.

1. 먼저 dense vMF fit을 만든다.
2. dense fit의 sufficient statistic을 이용해 feasible $\lambda_\kappa$ scale을 계산한다.
3. feasible scale에 사전에 정한 fraction을 곱해 $\lambda_\kappa$ 후보를 만든다.
4. 각 $\lambda_\kappa$에 대해 $\lambda_\mu=0$에서 시작한다.
5. 현재 score $\kappa_k|r_{kj}|$ threshold를 따라 $\lambda_\mu$ path를 증가시킨다.
6. 각 후보 fit에 대해 BIC와 EBIC를 계산한다.
7. 전체 후보 중 BIC가 가장 작은 fit을 분리 패널티 결과로 선택한다.

K=4 driver에서는 주로 `fit_separate_path_for_kappa()`, `fit_separate_specific_pair()`에서 이 흐름을 사용한다.

해석상 중요한 한계는 $\lambda_\kappa$가 component-level scalar penalty라는 점이다. 즉 concentration 크기는 줄일 수 있지만, 어떤 coordinate가 concentration-driven separation에 기여하는지는 직접 선택하지 못한다.

## 8. Eta-group proximal EM-type update

Eta-group의 출발점은 $\eta_k=\kappa_k\mu_k$가 posterior decision에 직접 들어간다는 점이다. 두 component의 경우 posterior log odds는 다음 형태다.

$$\log\frac{\tau_{i2}}{\tau_{i1}} = \mathrm{const} + (\eta_2-\eta_1)^T x_i.$$

따라서 coordinate $j$의 군집 구분 효과는 $\eta_{2j}-\eta_{1j}$로 직접 표현된다.

### 8.1. K=2 eta contrast penalty

K=2에서는 contrast를 직접 둔다.

$$\delta_j = \eta_{2j}-\eta_{1j}.$$

패널티는 다음과 같다.

$$\ell_{\mathrm{pen}}(\eta) = \ell(\eta) - \lambda_\eta\sum_j|\delta_j|.$$

Exact penalized M-step은 closed form으로 바로 풀기 어렵다. 이유는 likelihood에 $C_d(\|\eta_k\|_2)$가 들어가고, 동시에 $\eta_2-\eta_1$ 또는 centered eta contrast에 penalty가 걸리기 때문이다. 현재 구현은 unpenalized eta M-step 후 soft-thresholding을 적용하는 proximal EM-type update이며, monotone EM 보장을 아직 갖고 있지는 않다.

절차는 다음이다.

1. 현재 parameter로 E-step을 수행한다.
2. unpenalized vMF M-step으로 $\eta_1$, $\eta_2$를 얻는다.
3. 평균 eta와 contrast를 만든다.
4. contrast에 coordinate-wise soft-thresholding을 적용한다.
5. shrink된 contrast와 평균 eta로 $\eta_1$, $\eta_2$를 복원한다.
6. $\eta_k$에서 $\kappa_k=\|\eta_k\|_2$, $\mu_k=\eta_k/\|\eta_k\|_2$를 복원한다.

수식으로 쓰면 다음과 같다.

$$\bar{\eta} = (\eta_1+\eta_2)/2, \qquad \delta = \eta_2-\eta_1.$$

$$\delta_{\lambda,j} = \mathrm{sgn}(\delta_j)(|\delta_j|-\lambda_\eta)_+.$$

$$\eta_{1,\lambda} = \bar{\eta}-\delta_\lambda/2, \qquad \eta_{2,\lambda} = \bar{\eta}+\delta_\lambda/2.$$

관련 함수는 `fit_eta_penalty_em()`, `prox_eta_contrast_k2()`, `fit_eta_lambda_path()`다.

### 8.2. K>2 centered eta penalty

K=4에서는 $\eta_2-\eta_1$ 하나만으로 전체 component 차이를 표현할 수 없다. 그래서 coordinate별 centered eta를 사용한다.

$$\bar{\eta}_j = K^{-1}\sum_{k=1}^{K}\eta_{kj}, \qquad c_{kj} = \eta_{kj}-\bar{\eta}_j.$$

Coordinate $j$의 component 간 eta 차이는 다음 group norm으로 측정한다.

$$\|c_{\cdot j}\|_2 = \sqrt{\sum_{k=1}^{K}c_{kj}^2}.$$

Penalty는 group-lasso 형태다.

$$\ell_{\mathrm{pen}}(\eta) = \ell(\eta) - \lambda_\eta\sum_{j=1}^{d}\|c_{\cdot j}\|_2.$$

Proximal update는 coordinate별 group soft-thresholding이다.

$$c_{\cdot j,\lambda} = (1-\lambda_\eta/\|c_{\cdot j}\|_2)_+c_{\cdot j}.$$

다시 eta로 되돌릴 때는 다음을 사용한다.

$$\eta_{kj,\lambda} = \bar{\eta}_j + c_{kj,\lambda}.$$

관련 함수는 `prox_eta_centered()`, `fit_eta_centered_em()`, `fit_eta_centered_path_pair()`다. 함수명에는 `em`이 들어가지만, 논문 표현에서는 exact EM이 아니라 proximal EM-type update로 부르는 것이 안전하다.

### 8.3. Eta-group path

Eta-group도 fixed grid가 아니라 threshold path를 사용한다.

K=2에서는 현재 unpenalized eta M-step에서 얻은 $|\eta_{2j}-\eta_{1j}|$ 값들이 support 변화 threshold가 된다. K=4에서는 $\|c_{\cdot j}\|_2$ 값들이 support 변화 threshold가 된다.

절차는 다음이다.

1. $\lambda_\eta=0$에서 dense 또는 weakly penalized fit을 만든다.
2. 현재 fit의 eta contrast threshold를 계산한다.
3. 현재 $\lambda_\eta$보다 큰 가장 작은 threshold를 다음 후보로 둔다.
4. 이전 fit을 warm start로 사용해 새 $\lambda_\eta$에서 적합한다.
5. path 위 후보 중 BIC가 가장 작은 fit을 선택한다.

### 8.4. Weak setting path refinement 진단

2026-06-16에 weak concentration setting에서 Eta path 후보 전체를 저장하고, target-free adaptive refinement를 여러 방식으로 점검했다. 목적은 공식 tuning rule을 바꾸는 것이 아니라, Eta BIC가 null support 또는 dense support로 튀는 원인이 단순 path grid 부족인지 확인하는 것이었다.

| path construction | scope | near22 후보율 | BIC null 선택률 | positive-support dense 선택률 | 구현상 의미 |
|:---|:---|---:|---:|---:|:---|
| no refinement | weak100 | 0.23 | 0.73 | 0.72 | 기본 path가 q=22 근처 후보를 충분히 만들지 못함 |
| oracle target-refine | weak100 | 0.89 | 0.14 | 0.09 | true q 주변 범위를 쓰므로 논문 공식 알고리즘 불가 |
| adaptive v1 support-jump | weak100 | 0.73 | 0.24 | 0.24 | target-free refinement로 개선되지만 oracle에는 부족 |
| adaptive v2 priority midpoint | smoke10 | 0.50 | 0.50 | 0.50 | priority interval midpoint만으로는 개선 부족 |
| adaptive v2.1 duplicate endpoint | smoke10 | 0.50 | 0.40 | 0.40 | evaluated 254, saved 6, duplicate endpoint 248 |
| adaptive v3 multi-point | smoke10 | 0.50 | 0.60 | 0.40 | evaluated 990, saved 6, duplicate endpoint 984 |

중요한 해석은 다음이다.

1. Oracle target-refine은 q=17-27이라는 true support 주변 정보를 사용하므로 공식 알고리즘에 넣을 수 없다.
2. Adaptive v2/v2.1/v3는 target-free지만, smoke 결과에서 unique support 다양성을 충분히 늘리지 못했다.
3. 특히 v3는 990개 후보를 평가했는데도 saved unique support가 6개뿐이었고, near22 후보율도 개선하지 못했다.
4. 따라서 weak setting의 Eta BIC 실패는 단순히 lambda grid가 성긴 문제라기보다, proximal EM-type update가 만드는 support plateau와 BIC/positive-support tuning instability 문제로 보는 것이 더 타당하다.

논문 본문 알고리즘에는 oracle target-refine 또는 adaptive v2/v3를 넣지 않는다. 이 결과는 appendix 또는 supplementary diagnostic으로 두고, 본문에서는 "weak concentration setting에서 현재 proximal path+BIC가 불안정할 수 있다"는 한계로 정리하는 것이 안전하다.

다음 방법론 보강 후보는 path grid를 더 촘촘히 하는 방향보다 다음 쪽이 더 적절하다.

| 후보 | 이유 |
|:---|:---|
| stability selection | BIC 한 번의 선택이 null/dense로 튀는 문제를 완화할 수 있음 |
| alternative IC / EBIC / RICc | df approximation과 weak signal에서의 BIC 불안정성을 sensitivity로 점검 |
| MM safeguard 또는 coordinate/proximal update 개선 | support plateau가 update 구조에서 오는지 확인하고 objective monotonicity를 강화 |
| repeated random starts + path aggregation | local solution과 support plateau에 대한 민감도를 줄이는 보조 전략 |

## 9. Refit

Refit은 variable selection 후 선택된 coordinate support를 고정하고 penalty 없이 vMF mixture를 다시 추정하는 단계다.

| 항목 | refit에서 하는 일 |
|:---|:---|
| support | penalized fit에서 선택된 coordinate 그대로 고정 |
| penalty | 사용하지 않음 |
| $\alpha_k$ | 다시 추정 |
| $\mu_k$ | 선택된 coordinate 안에서 다시 추정 |
| $\kappa_k$ | 다시 추정 |
| cluster label | refit posterior에서 다시 계산 |

Rossi와 Separate의 support는 $\mu_k$의 nonzero coordinate union으로 정의한다. Eta-group의 support는 eta contrast 또는 centered eta group norm이 0이 아닌 coordinate로 정의한다.

Refit의 목적은 selection을 바꾸는 것이 아니라 shrinkage bias를 줄이는 것이다. 특히 Eta-group은 $\eta_k=\kappa_k\mu_k$ 크기를 직접 줄이므로, refit 후 $\kappa$ ratio와 eta contrast norm이 true value에 가까워지는지 확인한다.

관련 함수는 `fit_support_refit()` 또는 K=2 실험의 `fit_support_constrained_vmf()`다.

## 10. BIC, EBIC, 자유도

공식 tuning 선택 기준은 BIC다.

$$\mathrm{BIC} = \log(n)df - 2\ell(\hat{\Theta}).$$

EBIC는 고차원 setting에서 보조 지표로 계산했다.

$$\mathrm{EBIC} = \{\log(n)+2\gamma\log(d)\}df - 2\ell(\hat{\Theta}), \qquad \gamma=0.5.$$

현재 구현에서 사용하는 자유도 근사는 다음과 같다.

| 방법 | 자유도 근사 | 해석 |
|:---|:---|:---|
| Rossi | `(2K - 1) + sum_k max(1, nnz_k - 1)` | mixing proportion, component별 kappa, 그리고 unit-norm 제약을 반영한 active mu 자유도 |
| Separate | `(K - 1) + active_kappa + sum_k max(1, nnz_k - 1)` | mixing proportion, 살아남은 kappa, active mu 자유도 |
| Eta-group | `(K - 1) + d + (K - 1) * m` | mixing proportion, coordinate별 공통 eta baseline, active centered eta contrast 자유도 |

Eta-group에서 `m`은 active centered eta coordinate 수다. `d`는 모든 component에 공통으로 남는 eta baseline을 나타내고, `(K - 1) * m`은 선택된 coordinate에서 component 간 centered contrast가 갖는 자유도를 근사한다.

이 df는 현재 구현에서 BIC/EBIC를 계산하기 위한 implementation-level approximation이다. 아직 엄밀한 effective degrees of freedom 이론으로 증명한 것은 아니므로, 논문에서는 sensitivity 기준으로 BIC, EBIC, RICc를 함께 확인하는 것이 안전하다. 특히 고차원에서는 BIC가 dense한 model을 선택할 수 있어 EBIC/RICc 또는 stability selection 검토가 필요하다.

관련 함수는 `model_ic()`, `separate_model_ic()`, `eta_centered_ic()`, `support_ic()`다.

## 11. 코드 실행 흐름

전체 simulation driver는 대체로 다음 순서로 동작한다.

1. true parameter와 support를 만든다.
2. vMF mixture에서 data를 생성한다.
3. 같은 data에 대해 6가지 방법을 모두 적합한다.
4. 각 방법에서 path 후보를 만들고 BIC 최소 fit을 선택한다.
5. refit 방법은 선택된 support를 고정하고 penalty 없이 다시 EM을 수행한다.
6. posterior responsibility로 cluster label을 정한다.
7. ARI, selected q, TPR, FPR, Precision, F1을 계산한다.
8. loglik, df, BIC, EBIC를 계산한다.
9. true parameter와 label matching 후 MSE_mu, MSE_kappa, MSE_eta를 계산한다.
10. 반복별 raw csv와 평균 summary csv를 저장한다.

## 12. 주요 코드 파일

| 역할 | 파일 | 주요 함수 또는 내용 |
|:---|:---|:---|
| 공통 vMF 및 Rossi 재현 | `r/rossi_barbaro_2022_reproduction.r` | `e_step_vmf()`, `estimate_kappa()`, `fit_svMF_em()`, `fit_svMF_path()` |
| K=2 path tuning 비교 | `r/eta_path_tuning_compare_run.r` | Rossi beta path, separate path/grid, eta path |
| K=4 path tuning 비교 | `r/k4_path_tuning_compare_run.r` | paperlike sparse-active, same-mean stress, realistic concentration-dominant |
| K=4 controlled setting | `r/k4_controlled_concdom_run.r` | stress setting과 같은 support, 평균 방향만 완화 |
| K=4 specific-effect setting | `r/k4_specific_effect_run.r` | 공통 변수 + component-specific 변수 |
| K=4 공통 함수 | `r/rb2022_k4_pilot_compare_run.r` | centered eta penalty, support refit, metric 계산 |
| 초기 prototype | `r/eta_penalty_vmf_run.r`, `r/separate_penalty_vmf_run.r` | path tuning 정리 전 grid/prototype 기록 |

## 13. 시뮬레이션 setting 생성 방식

| 번호 | setting | 변수 구조 | 모수 구조 |
|:---|:---|:---|:---|
| 1 | K=2 기본 메커니즘 | 두 component가 같은 active coordinate 10개 공유 | concentration 차이 큼, 작음, 평균+집중도 차이 |
| 2.1 | Rossi 2022 재현 | 논문 기준 sparse prototype 구조 | overlap 0.05, sparsity 0.10 |
| 2.2 | K=4 sparse-active 비교 | component별 support를 랜덤 생성 | Rossi가 유리한 sparse direction 구조 |
| 3 | K=4 stress | 모든 component가 같은 active coordinate 10개 공유 | $\mu_1=\cdots=\mu_4$, $\kappa=(20,35,60,100)$ |
| 4 | K=4 controlled | 3번과 같은 support | pairwise cosine 0.95, $\kappa=(25,40,65,100)$ |
| 5 | K=4 specific-effect | 공통 6개 + component별 specific 4개 | 공통 weight 1.0, specific weight 0.5, $\kappa=(30,45,65,90)$ |
| 5.8 | 약한 집중도 차이 | 5번과 같은 support | $\kappa=(40,50,60,70)$ |
| 6.1 | 고차원 kappa 조정 | 5번과 같은 support를 d 증가에 적용 | d에 따라 kappa도 증가 |
| 6.2 | 고차원 kappa 고정 | 5번과 같은 support를 d 증가에 적용 | $\kappa=(30,45,65,90)$ 고정 |

5번 specific-effect setting에서 raw mean vector는 다음과 같다.

```text
common variables: v_kj = 1.0 for all components
component-specific variables: v_kj = 0.5 only for component k
noise variables: v_kj = 0
mu_k = v_k / ||v_k||
```

따라서 component별 nonzero coordinate는 10개이고, union active coordinate는 공통 6개와 specific 16개를 합쳐 22개다.

## 14. 지표 계산

Variable selection은 union coordinate 기준으로 계산한다.

| 지표 | 계산 의미 |
|:---|:---|
| selected q | 선택된 coordinate 개수 |
| TPR | true active coordinate 중 선택된 비율 |
| FPR | true noise coordinate 중 선택된 비율 |
| Precision | 선택 coordinate 중 true active 비율 |
| F1 | TPR과 Precision의 조화평균 |

모수 MSE는 label switching을 정리한 뒤 계산한다. K=2 concentration-driven setting에서는 $\kappa$가 작은 component와 큰 component 순서로 정렬한다. K=4 setting에서는 true $\mu_k$와 추정 $\mu_k$ 사이의 cosine similarity가 최대가 되는 permutation을 사용한다.

K=4에서 centered eta MSE는 centered true eta와 centered estimated eta를 비교한다.

$$\eta_{kj}^c = \eta_{kj} - K^{-1}\sum_{\ell=1}^{K}\eta_{\ell j}.$$

## 15. 구현 검증 포인트

현재 문서 기준으로 확인한 구현상 중요한 포인트는 다음이다.

| 항목 | 확인 내용 |
|:---|:---|
| Rossi 재현 | 원 논문 setting에서 ARI와 sparsity pattern이 논문 Figure 범위와 유사 |
| tuning 기준 | 연구미팅 공식 결과는 path 후보 + BIC 선택 |
| 분리 패널티 | $\lambda_\kappa$ grid와 $\lambda_\mu$ path를 결합한 2D 후보 사용 |
| Eta-group | K=2는 eta contrast, K=4는 centered eta group norm 사용 |
| refit | support는 바꾸지 않고 penalty 없이 재추정 |
| 고차원 | BIC가 느슨해질 수 있어 EBIC sensitivity가 필요 |

주의할 점은 다음이다.

* 현재 Eta-group M-step은 exact penalized M-step이 아니라 proximal EM-type update다. Line-search safeguard 이후 objective trace smoke test는 통과했지만, 이것은 exact EM 이론을 의미하지 않는다.
* Weak setting path diagnostic에서는 Eta BIC가 null/dense support로 불안정하게 선택되는 문제가 남아 있다. 이는 단순 grid density 문제가 아니라 support plateau/tuning instability로 정리하는 것이 현재 증거에 더 맞다.
* 고차원 일부 반복에서는 $\kappa$ 추정 outlier가 생겨 MSE_kappa 평균이 매우 커질 수 있다.
* ARI가 비슷하더라도 selected q와 FPR이 크면 variable selection 성능은 좋지 않은 것으로 해석해야 한다.
* `thesis-simulation_260611.md`의 고차원 fixed-kappa setting은 signal이 약해지는 stress setting이므로, 일반적인 high-dimensional robustness 결과와 구분해야 한다.

## 참고문헌

* Banerjee, A., Dhillon, I. S., Ghosh, J., and Sra, S. (2005). Clustering on the unit hypersphere using von Mises-Fisher distributions.
* Mardia, K. V., and Jupp, P. E. (2000). Directional Statistics.
* Rossi, F., and Barbaro, F. (2022). Mixture of von Mises-Fisher distribution with sparse prototypes.
* Sra, S. (2012). A short note on parameter approximation for von Mises-Fisher distributions.
