# 모형, 추정 방법, 코드 구현 정리

이 문서는 연구미팅에서 구현 관련 질문에 답하기 위한 정리다. 핵심은 세 가지다.

1. vMF mixture에서 EM이 어떤 값을 업데이트하는지
2. Rossi, 분리 패널티, 에타 패널티에서 penalty와 tuning parameter를 어떻게 처리했는지
3. 현재 R 코드가 실제로 어떤 순서로 fitting, model selection, refit을 수행하는지

현재 260622 연구미팅 자료의 공식 기준은 다음과 같다.

| 항목 | 공식 기준 |
|:---|:---|
| tuning 후보 생성 | path 기반 후보 생성 |
| tuning 선택 | BIC 최소 지점 선택 |
| EBIC | 보조 적합 지표로만 계산 |
| 최종 추정 | 선택된 support를 고정하고 penalty 없이 refit |
| 4번 시뮬레이션 | K=4 controlled concentration-dominant setting |

---

## 1. 기본 vMF Mixture 모형

관측값 $x_i$는 단위구 $S^{d-1}$ 위의 방향자료다. $K$개 component를 갖는 vMF mixture는 다음과 같다.

$$f(x_i) = \sum_{k=1}^{K} \alpha_k C_d(\kappa_k) \exp(\kappa_k \mu_k^\top x_i).$$

여기서

| 기호 | 의미 |
|:---|:---|
| $\alpha_k$ | component mixing proportion |
| $\mu_k$ | component direction, $\|\mu_k\|_2=1$ |
| $\kappa_k$ | concentration, 클수록 $\mu_k$ 주변으로 자료가 강하게 모임 |
| $C_d(\kappa_k)$ | vMF normalizing constant |

posterior responsibility는

$$\tau_{ik} = \frac{ \alpha_k C_d(\kappa_k)\exp(\kappa_k\mu_k^\top x_i) }{ \sum_{\ell=1}^{K} \alpha_\ell C_d(\kappa_\ell)\exp(\kappa_\ell\mu_\ell^\top x_i) }.$$

코드에서는 이 계산이 `rossi_barbaro_2022_reproduction.r`의 `e_step_vmf()`에 들어 있다.

---

## 2. 공통 EM 구조

E-step 이후 M-step에서 쓰는 sufficient statistics는 다음이다.

$$N_k = \sum_i \tau_{ik}, \qquad r_k = \sum_i \tau_{ik}x_i.$$

mixing proportion은

$$\alpha_k = \frac{N_k}{n}$$

로 업데이트한다.

패널티가 없으면 평균 방향은 $r_k$ 방향으로 간다.

$$\mu_k = \frac{r_k}{\|r_k\|_2}.$$

집중도는

$$\rho_k = \frac{\mu_k^\top r_k}{N_k}$$

를 만든 뒤 $A_d(\kappa_k)=\rho_k$를 푸는 방식이다. 현재 코드는 정확한 numerical inversion 대신 다음 근사식을 쓴다.

$$\kappa_k \approx \frac{d\rho_k-\rho_k^3}{1-\rho_k^2}.$$

코드 위치는 `estimate_kappa()`다. 이 근사식은 vMF mixture EM에서 널리 쓰이는 Banerjee et al. (2005)의 concentration update approximation에 기반한다. 방향자료의 vMF MLE 기본식은 Mardia and Jupp (2000)에서 표준적으로 다루며, inverse approximation 관련 논의는 Sra (2012)도 참고할 수 있다.

---

## 3. 초기값과 수렴 기준

현재 구현의 초기값 생성은 다음 순서다.

1. 데이터 중 $K$개 관측치를 무작위로 골라 초기 $\mu_k$로 둔다.
2. cosine similarity가 가장 큰 component에 각 관측치를 hard assignment한다.
3. 그 assignment로 초기 $\alpha_k$, $r_k$, $\kappa_k$를 만든다.

코드 위치는 `init_vmf_mixture()`다.

random start는 `fit_svMF_multistart()`에서 처리한다. 여러 초기값으로 EM을 돌린 뒤 penalized log-likelihood가 가장 큰 해를 선택한다.

수렴 기준은 상대 objective 변화량이다.

$$\frac{|Q^{(t)}-Q^{(t-1)}|}{\max(1, |Q^{(t-1)}|)} < tol.$$

기본적으로 `tol = 1e-7`을 사용했다. component가 비거나, M-step에서 모든 좌표가 0이 되어 방향벡터를 만들 수 없으면 해당 fit은 실패로 처리한다.

---

## 4. Rossi & Barbaro (2022) Sparse vMF

Rossi & Barbaro (2022)는 component direction $\mu_k$에 L1 penalty를 둔다.

$$\ell_{pen}(\Theta) = \ell(\Theta) - \beta \sum_{k=1}^{K}\|\mu_k\|_1.$$

E-step은 기본 vMF mixture와 같다. 차이는 M-step에서 $\mu_k$를 업데이트할 때 coordinate-wise shrinkage가 들어간다는 점이다.

각 component $k$에 대해 현재 $\kappa_k$가 주어졌을 때,

$$z_{kj} = sgn(r_{kj}) \left(\kappa_k |r_{kj}|-\beta\right)_+$$

를 만들고,

$$\mu_k = \frac{z_k}{\|z_k\|_2}$$

로 정규화한다. 여기서 $sgn(\cdot)$은 부호 함수다. 그 다음

$$\rho_k = \frac{\mu_k^\top r_k}{N_k}$$

를 계산하고 `estimate_kappa()`로 $\kappa_k$를 업데이트한다. 코드에서는 `update_mu_kappa_one()` 안에서 $\mu_k$와 $\kappa_k$를 inner loop로 반복한다.

### Rossi beta path

Rossi 방법은 $\beta$를 fixed grid로 두지 않고 path-following 방식으로 만든다. 코드 위치는 `fit_svMF_path()`다.

절차는 다음과 같다.

1. $\beta=0$에서 dense vMF mixture를 multi-start EM으로 적합한다.
2. 현재 해에서

$$m_{kj} = \kappa_k |r_{kj}|-\beta$$

를 계산한다.

3. 양수인 $m_{kj}$ 중 가장 작은 값을 찾아 다음 beta로 이동한다.

$$\beta_{next} = \beta + \min_{k,j: m_{kj}>0} m_{kj}.$$

4. 너무 작은 증가를 피하기 위해, 이미 $\beta>0$이면

$$\beta_{next} \ge \beta(1+\epsilon_\beta)$$

가 되도록 한다. 여기서 $\epsilon_\beta$는 코드의 `min_rel_beta`에 해당한다.

5. 이전 해를 초기값으로 warm start해서 새 $\beta$에서 EM을 다시 수행한다.
6. 이 과정을 `max_path_steps`까지 반복한다.
7. path 위의 후보 중 BIC가 가장 작은 해를 Rossi 결과로 선택한다.

현재 연구미팅 자료에서 Rossi는 BIC 기준으로 선택했다. 논문 재현 코드에서는 AIC, BIC, RIC, RICc, EBIC도 같이 계산할 수 있다.

---

## 5. Refit이 하는 일

Refit은 선택된 변수 집합을 고정한 뒤 penalty 없이 vMF mixture를 다시 추정하는 단계다.

예를 들어 Rossi refit에서는 먼저 Rossi가 선택한 coordinate support를 만든다.

$$\hat{S} = \{j:\ \exists k,\ |\hat{\mu}_{kj}|>0\}.$$

그 다음 $\hat{S}$ 밖의 좌표는 계속 0으로 고정하고, $\hat{S}$ 안에서만 unpenalized EM을 수행한다.

중요한 점은 다음이다.

| 항목 | 의미 |
|:---|:---|
| support | 바꾸지 않는다 |
| $\alpha_k$ | 다시 추정한다 |
| $\mu_k$ | 선택된 좌표 안에서 다시 추정한다 |
| $\kappa_k$ | 다시 추정한다 |
| penalty | refit 단계에서는 사용하지 않는다 |

코드 위치는 `fit_support_refit()` 또는 K=2 단독 스크립트의 `fit_support_constrained_vmf()`다.

Refit을 넣은 이유는 L1 penalty로 생기는 shrinkage bias를 줄이기 위해서다. 특히 에타 패널티에서는 penalty가 $\eta_k=\kappa_k\mu_k$의 크기 자체를 줄이므로, refit 후 $\kappa$ ratio와 eta contrast norm이 true value에 더 가까워지는지 확인한다.

---

## 6. 분리 패널티 EM

분리 패널티는 교수님 제안에 따라 $\mu_k$와 $\kappa_k$에 penalty를 따로 둔 baseline이다.

$$\ell_{pen}(\Theta) = \ell(\Theta) - \lambda_\mu \sum_k \|\mu_k\|_1 - \lambda_\kappa \sum_k \kappa_k.$$

E-step은 기본 vMF mixture와 같다. M-step에서는 $\mu_k$ update는 Rossi와 거의 같다.

$$z_{kj} = sgn(r_{kj}) \left(\kappa_k |r_{kj}|-\lambda_\mu\right)_+, \qquad \mu_k = \frac{z_k}{\|z_k\|_2}.$$

차이는 $\kappa_k$ update다. concentration penalty가 있으므로

$$\rho_k = \frac{\mu_k^\top r_k-\lambda_\kappa}{N_k}$$

를 사용하고,

$$\kappa_k \approx \frac{d\rho_k-\rho_k^3}{1-\rho_k^2}$$

로 업데이트한다.

코드 위치는 `update_mu_kappa_separate_one()`과 `fit_separate_penalty_em()`이다.

### 분리 패널티 path/grid tuning

현재 260622 연구미팅 자료의 공식 비교에서는 고정 grid만 사용하지 않고, Rossi의 path tuning에 맞춰 분리 패널티도 path/grid 방식으로 정리했다. 분리 패널티에는 tuning parameter가 두 개 있으므로, $\lambda_\kappa$는 데이터 기반 grid로 두고 각 $\lambda_\kappa$에서 $\lambda_\mu$ path를 생성한다.

$\lambda_\kappa$ 후보는 dense fit에서 계산한 feasible scale에 비율을 곱해 만든다.

| setting | $\lambda_\kappa$ 후보 |
|:---|:---|
| K=2 | feasible scale $\times$ {0, 0.05, 0.1, 0.2, 0.35, 0.5} |
| K=4 | feasible scale $\times$ {0, 0.05, 0.1, 0.2} |

각 $\lambda_\kappa$가 고정되면 $\lambda_\mu$는 현재 M-step score에서 support가 바뀔 수 있는 threshold를 따라 증가시킨다.

구현 흐름은 다음이다.

1. 먼저 $\beta=0$ dense vMF를 적합한다.
2. dense fit에서 $\lambda_\kappa$ 후보 grid를 만든다.
3. 각 $\lambda_\kappa$에 대해 dense fit에서 시작한다.
4. 같은 $\lambda_\kappa$ 안에서는 현재 M-step의 $\kappa_k |r_{kj}|$ threshold를 이용해 $\lambda_\mu$ path를 만든다.
5. 각 path 후보에 대해 BIC를 계산한다.
6. 전체 후보 중 BIC가 가장 작은 fit을 선택한다.
7. 선택된 support로 refit을 수행한다.

코드 위치는 K=2의 `fit_separate_path_grid()`, `fit_separate_path_for_kappa()`와 K=4의 `fit_separate_path_grid_pair()`다.

---

## 7. 에타 패널티 EM

에타 패널티의 핵심은 vMF mixture를 natural parameter로 보는 것이다.

$$\eta_k = \kappa_k \mu_k.$$

그러면 density는

$$f(x_i\mid \eta_k) = C_d(\|\eta_k\|_2) \exp(\eta_k^\top x_i)$$

로 쓸 수 있다. posterior decision에 직접 들어가는 항은 $\mu_k$가 아니라 $\eta_k^\top x_i$다. 따라서 coordinate $j$가 군집 구분에 기여하는지는 $\mu_{kj}$보다 $\eta_{kj}$의 component 간 차이를 보는 것이 자연스럽다.

### K=2 eta contrast penalty

K=2에서는 직접적으로

$$\delta_j = \eta_{2j}-\eta_{1j}$$

를 정의하고,

$$\ell_{pen}(\eta) = \ell(\eta) - \lambda_\eta\sum_j |\delta_j|$$

를 사용했다.

exact penalized M-step은 closed form으로 바로 풀기 어렵다. 이유는 likelihood에 $C_d(\|\eta_k\|_2)$가 들어가고, 동시에 $\eta_2-\eta_1$에 L1 penalty가 걸리기 때문이다. 그래서 현재 코드는 proximal EM prototype을 사용한다.

절차는 다음이다.

1. 현재 $\theta$로 E-step을 수행한다.
2. unpenalized vMF M-step을 수행해 $\eta_1,\eta_2$를 얻는다.
3. 평균 eta와 contrast를 만든다.

$$\bar{\eta} = \frac{\eta_1+\eta_2}{2}, \qquad \delta = \eta_2-\eta_1.$$

4. contrast에 coordinate-wise soft-thresholding을 적용한다.

$$\delta_{\lambda,j} = sgn(\delta_j) \left(|\delta_j|-\lambda_\eta\right)_+.$$

5. 다시 eta로 변환한다.

$$\eta_{1,\lambda} = \bar{\eta} - \frac{1}{2}\delta_\lambda, \qquad \eta_{2,\lambda} = \bar{\eta} + \frac{1}{2}\delta_\lambda.$$

6. $\eta_k$에서 $\kappa_k$, $\mu_k$를 복원한다.

$$\kappa_k = \|\eta_k\|_2, \qquad \mu_k = \frac{\eta_k}{\|\eta_k\|_2}.$$

코드 위치는 `eta_penalty_vmf_run.r`의 `fit_eta_penalty_em()`과 `prox_eta_contrast_k2()`다.

K=2 공식 비교에서는 고정 grid가 아니라 $\lambda_\eta$ path를 사용했다. 각 EM 반복에서 unpenalized eta M-step을 계산하고, $|\eta_{2j}-\eta_{1j}|$ 값 중 현재 $\lambda_\eta$보다 큰 가장 작은 threshold를 다음 후보로 둔다. 각 후보는 이전 fit을 warm start로 사용해 적합하고, path 위에서 BIC가 가장 작은 $\lambda_\eta$를 선택한다.

코드 위치는 `fit_eta_lambda_path()`다.

### K>2 centered eta penalty

K=4에서는 $\eta_2-\eta_1$ 하나만으로 전체 component 차이를 표현할 수 없다. 그래서 coordinate별 centered eta를 사용했다.

$$\bar{\eta}_j = \frac{1}{K}\sum_{k=1}^{K}\eta_{kj}, \qquad c_{kj} = \eta_{kj}-\bar{\eta}_j.$$

coordinate $j$의 component 간 eta 차이는

$$\|c_{\cdot j}\|_2 = \sqrt{\sum_{k=1}^{K} c_{kj}^2}$$

로 측정했다. penalty는 group-lasso 형태다.

$$\ell_{pen}(\eta) = \ell(\eta) - \lambda_\eta \sum_{j=1}^{d}\|c_{\cdot j}\|_2.$$

proximal update는 coordinate별로 다음 shrinkage를 적용한다.

$$c_{\cdot j,\lambda} = \left(1-\frac{\lambda_\eta}{\|c_{\cdot j}\|_2}\right)_+ c_{\cdot j}.$$

그 다음

$$\eta_{kj,\lambda} = \bar{\eta}_j + c_{kj,\lambda}$$

로 되돌린다.

코드 위치는 `rb2022_k4_pilot_compare_run.r`의 `prox_eta_centered()`와 `fit_eta_centered_em()`이다.

K=4 공식 비교에서도 고정 grid가 아니라 centered eta norm의 $\lambda_\eta$ path를 사용했다. 각 후보는 현재 M-step의 $\|c_{\cdot j}\|_2$ threshold에서 생성하고, 이전 fit을 warm start로 사용한다. 최종 fit은 path 위에서 BIC가 가장 작은 지점으로 선택한다.

코드 위치는 `fit_eta_centered_path_pair()`다.

---

## 8. BIC, EBIC, 자유도 계산

공식 simulation에서 tuning parameter 선택은 BIC를 사용했다.

$$BIC = \log(n)df - 2\ell(\hat{\Theta}).$$

EBIC도 보조 적합 지표로 함께 계산했지만, 현재 연구미팅 자료의 공식 tuning 기준으로 사용하지 않았다.

$$EBIC = \{\log(n)+2\gamma\log(d)\}df - 2\ell(\hat{\Theta}), \qquad \gamma=0.5.$$

Rossi sparse vMF의 자유도는 코드에서 다음처럼 근사했다.

$$df = (2K-1) + \sum_{k=1}^{K}\max(1, nnz_k-1).$$

여기서 $(2K-1)$은 mixing proportion과 component별 $\kappa_k$를 포함한 항이고, $nnz_k-1$은 $\|\mu_k\|_2=1$ 제약 때문에 component direction에서 자유도 하나를 뺀 것이다.

K=4 centered eta penalty에서는 선택된 coordinate 수를 $m$이라 두고,

$$df = (K-1)+d+(K-1)m$$

로 계산했다. 이는 구현상 eta의 coordinate-level contrast support를 반영하기 위한 근사 자유도다.

코드 위치는 `model_ic()`, `separate_model_ic()`, `eta_centered_ic()`, `support_ic()`다.

---

## 9. 6가지 방법의 코드상 정의

현재 연구미팅 자료에서 비교한 6가지 방법은 다음처럼 구현했다.

| 방법 | penalty 기준 | support 정의 | tuning 선택 | refit |
|:---|:---|:---|:---|:---|
| Rossi | $\mu_k$ L1 | $\mu_k$ nonzero coordinate union | beta path에서 BIC 최소 | 없음 |
| Rossi + refit | Rossi와 동일 | Rossi support | Rossi가 선택한 beta 사용 | support 고정 후 unpenalized EM |
| 분리 패널티 | $\mu_k$ L1, $\kappa_k$ L1 | $\mu_k$ nonzero coordinate union | $\lambda_\kappa$ grid와 $\lambda_\mu$ path 후보 중 BIC 최소 | 없음 |
| 분리 패널티 + refit | 분리 패널티와 동일 | 분리 패널티 support | 선택된 lambda 사용 | support 고정 후 unpenalized EM |
| 에타 패널티 | $\eta$ contrast 또는 centered eta | eta contrast support | $\lambda_\eta$ path에서 BIC 최소 | 없음 |
| 에타 패널티 + refit | 에타 패널티와 동일 | 에타 support | 선택된 lambda 사용 | support 고정 후 unpenalized EM |

K=2에서는 $\eta_2-\eta_1$ 직접 contrast를 사용했다. K=4에서는 centered eta의 coordinate별 group norm을 사용했다.

---

## 10. 코드 실행 흐름

전체 simulation driver의 공통 흐름은 다음이다.

1. data를 생성한다.
2. 같은 data에 대해 6가지 방법을 모두 적합한다.
3. 각 방법에서 tuning parameter를 BIC로 선택한다.
4. refit 방법은 선택된 support를 고정하고 penalty 없이 다시 EM을 수행한다.
5. cluster label은 posterior responsibility에서 가장 큰 component로 정한다.
6. ARI, selected $q$, TPR, FPR, Precision, F1, loglik, BIC, EBIC, MSE 지표를 계산한다.
7. 반복별 결과를 raw csv로 저장하고, 평균 요약을 summary csv로 저장한다. K=2 driver에서는 표준오차도 함께 계산한다.

현재 연구미팅 자료에 직접 연결되는 simulation driver는 다음과 같다.

| 역할 | 파일 | 주요 함수 또는 내용 |
|:---|:---|:---|
| 공통 vMF, Rossi 재현 | `rossi_barbaro_2022_reproduction.r` | `e_step_vmf()`, `estimate_kappa()`, `fit_svMF_em()`, `fit_svMF_path()` |
| 1번 K=2 기본 메커니즘 | `eta_path_tuning_compare_run.r` | Rossi beta path, separate path/grid, eta path를 BIC로 선택 |
| 2, 3번 K=4 비교 | `k4_path_tuning_compare_run.r` | paperlike sparse-active setting, same-mean kappa stress setting |
| 4번 K=4 controlled setting | `k4_controlled_concdom_run.r` | common active support 유지, 평균 방향만 완화한 concentration-dominant setting |
| K=4 공통 함수 | `rb2022_k4_pilot_compare_run.r` | `fit_eta_centered_em()`, `prox_eta_centered()`, `fit_support_refit()` |
| 초기 grid 실험 기록 | `eta_penalty_vmf_run.r`, `separate_penalty_vmf_run.r` | path tuning으로 정리하기 전 단독 penalty prototype |

### 현재 1-4번 simulation setting 요약

| 번호 | setting | 변수 구조 | 모수 구조 | 공식 tuning |
|:---|:---|:---|:---|:---|
| 1 | K=2 기본 메커니즘 | $d=100$, true $q=10$ | 집중도 차이 큼, 집중도 차이 작음, 평균+집중도 차이 | path 후보 + BIC |
| 2 | Rossi 2022 재현 및 K=4 sparse-active 비교 | $d=100$, paperlike sparse prototype 또는 component별 nonzero 10개 | Rossi 2022 artificial simulation 기준 | beta path 또는 6-method path/grid + BIC |
| 3 | K=4 stress setting | 모든 component가 같은 active coordinate 10개 공유 | $\mu_1=\cdots=\mu_4$, $\kappa=(20,35,60,100)$ | path 후보 + BIC |
| 4 | K=4 controlled concentration-dominant setting | 3번과 동일하게 모든 component가 같은 active coordinate 10개 공유 | 평균 방향 pairwise cosine 0.95, $\kappa=(25,40,65,100)$ | path 후보 + BIC |

4번은 기존 heterogeneous-support setting을 대체한 버전이다. 기존 4번은 common active 7개와 component-specific active 3개씩을 섞어 union active가 19개였지만, 현재 버전은 변수 구조를 3번과 동일하게 고정한다. 따라서 3번의 pure concentration stress setting에서 평균 방향 차이만 약간 추가한 controlled setting으로 해석한다.

---

## 참고문헌

* Banerjee, A., Dhillon, I. S., Ghosh, J., and Sra, S. (2005). Clustering on the unit hypersphere using von Mises-Fisher distributions.
* Mardia, K. V., and Jupp, P. E. (2000). Directional Statistics.
* Rossi, F., and Barbaro, F. (2022). Mixture of von Mises-Fisher distribution with sparse prototypes.
* Sra, S. (2012). A short note on parameter approximation for von Mises-Fisher distributions.
