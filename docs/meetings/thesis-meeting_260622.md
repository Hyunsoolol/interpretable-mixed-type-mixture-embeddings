# Thesis Meeting 260622

업데이트: 2026-06-16

## 1. 핵심 결론

Eta penalty는 ARI를 크게 올리는 방법이라기보다, vMF mixture 안에서 posterior decision parameter인 `eta = kappa * mu`의 component contrast를 sparse하게 만들어, ARI를 유지하면서 해석 가능한 변수 선택을 제공하는 방법이다.

따라서 본문 주장은 “clustering accuracy 개선”보다 “model-based sparse interpretation”으로 정리함.

## 2. 모형 아이디어

관측값은 단위구면 위의 방향자료 $x_i \in \mathbb{S}^{d-1}$이고, $K$-component vMF mixture를 사용한다.

**Mixture model**

$$p(x_i;\Theta)=\sum_{k=1}^K \alpha_k C_d(\kappa_k)\exp(\kappa_k \mu_k^\top x_i), \qquad \|\mu_k\|_2=1,\quad \kappa_k>0$$

**Natural parameter**

$$\eta_k=\kappa_k\mu_k$$

Posterior responsibility는 다음과 같다.

$$\tau_{ik}=\frac{\alpha_k C_d(\kappa_k)\exp(\eta_k^\top x_i)}{\sum_{\ell=1}^K \alpha_\ell C_d(\kappa_\ell)\exp(\eta_\ell^\top x_i)}$$

K=2에서 posterior decision boundary는

$$\log\frac{\tau_{i2}}{\tau_{i1}}=\mathrm{const}+(\eta_2-\eta_1)^\top x_i$$

로 정리된다. 따라서 변수 선택은 $\mu_k$ 자체보다 posterior decision에 직접 들어가는 $\eta_k$ contrast를 기준으로 하는 것이 자연스럽다.

**Observed log-likelihood**

$$\ell(\Theta)=\sum_{i=1}^n \log\left[\sum_{k=1}^K \alpha_k C_d(\kappa_k)\exp(\eta_k^\top x_i)\right]$$

K>2에서는 coordinate별 centered eta contrast를 사용한다.

$$\bar{\eta}_j=\frac{1}{K}\sum_{\ell=1}^K \eta_{\ell j}, \qquad c_{kj}=\eta_{kj}-\bar{\eta}_j$$

**Penalty and objective**

K>2에서는 두 가지 penalty 선택지가 있다.

$$P_{\mathrm{group}}(\Theta)=\lambda_\eta\sum_{j=1}^d \|c_{\cdot j}\|_2$$

$$P_{\mathrm{ANOVA}\text{-}L1}(\Theta)=\lambda_\eta\sum_{j=1}^d\sum_{k=1}^K |c_{kj}|$$

Group lasso는 coordinate $j$ 전체가 component contrast를 갖는지 선택한다. ANOVA-type L1은 coordinate 안의 component별 deviation을 개별적으로 shrink한다.

현재 연구 목적은 component별 세부 deviation보다 “posterior decision에 관여하는 coordinate-level eta contrast 선택”이므로 group lasso가 더 자연스럽다.

$$\mathcal{L}_p(\Theta)=\ell(\Theta)-P_{\mathrm{group}}(\Theta)$$

같은 K=4 common+specific setting에서 rep=20 pilot으로 비교하면 ANOVA-type L1은 BIC에서 거의 모든 좌표를 선택했다.

| Scenario | Penalty | ARI | Selected q | FPR | Precision | F1 |
|:---|:---|---:|---:|---:|---:|---:|
| strong | Group lasso + refit | 0.684 | 25.45 | 0.046 | 0.867 | 0.925 |
| strong | ANOVA L1 + refit | 0.652 | 99.90 | 0.999 | 0.220 | 0.361 |
| weak | Group lasso + refit | 0.565 | 23.90 | 0.024 | 0.924 | 0.960 |
| weak | ANOVA L1 + refit | 0.515 | 99.50 | 0.994 | 0.221 | 0.362 |

모수 추정 결과도 group lasso + refit 쪽이 더 안정적이었다.

| Scenario | Penalty | MSE_mu | MSE_kappa | MSE_centered_eta | kappa_hat_mean |
|:---|:---|---:|---:|---:|---:|
| strong | Group lasso + refit | 0.000101 | 1.992 | 0.191 | 58.007 |
| strong | ANOVA L1 + refit | 0.000329 | 3.604 | 0.581 | 58.677 |
| weak | Group lasso + refit | 0.000073 | 1.614 | 0.172 | 55.298 |
| weak | ANOVA L1 + refit | 0.000288 | 2.989 | 0.659 | 56.043 |

따라서 현재 구현과 논문 주장은 centered eta group lasso를 주 penalty로 두고, ANOVA-type L1은 대안 또는 sensitivity 후보로만 남기는 것이 적절하다.

현재 구현은 exact penalized EM이 아니라 proximal EM-type update다.

## 3. 핵심 simulation 결과

### 3.1 K=2 toy setting

설정:

```text
K = 2, n = 1000, d = 100, rep = 20
shared active variables = 10
true q = 10
kappa = (20, 200)
mu_1 = mu_2
```

| Method | ARI | Selected q | TPR | FPR | Precision | F1 |
|:---|---:|---:|---:|---:|---:|---:|
| Rossi path BIC | 1.000 | 23.300 | 1.000 | 0.148 | 0.443 | 0.610 |
| Rossi path BIC + refit | 1.000 | 23.300 | 1.000 | 0.148 | 0.443 | 0.610 |
| Separate path/grid BIC | 1.000 | 23.300 | 1.000 | 0.148 | 0.443 | 0.610 |
| Separate path/grid BIC + refit | 1.000 | 23.300 | 1.000 | 0.148 | 0.443 | 0.610 |
| Eta path BIC | 1.000 | 13.200 | 1.000 | 0.036 | 0.792 | 0.875 |
| Eta path BIC + refit | 1.000 | 13.200 | 1.000 | 0.036 | 0.792 | 0.875 |

모수 추정 결과는 다음과 같다. 3.1과 3.2의 MSE 지표는 모두 raw scale로 통일했다.

| Method | MSE_mu | MSE_kappa | MSE_Delta_eta | kappa ratio | eta contrast norm |
|:---|---:|---:|---:|---:|---:|
| Rossi path BIC | 0.000176 | 1.276 | 0.245 | 10.062 | 181.179 |
| Rossi path BIC + refit | 0.000061 | 1.410 | 0.378 | 9.951 | 180.821 |
| Separate path/grid BIC | 0.000176 | 1.276 | 0.245 | 10.062 | 181.179 |
| Separate path/grid BIC + refit | 0.000061 | 1.410 | 0.378 | 9.951 | 180.821 |
| Eta path BIC | 0.000180 | 7.415 | 0.292 | 8.559 | 175.542 |
| Eta path BIC + refit | 0.000041 | 1.185 | 0.216 | 9.960 | 180.630 |

K=2 toy setting에서는 모든 방법의 ARI가 1.000이지만, Eta penalty가 selected q와 FPR을 가장 낮춘다. Refit 후 kappa ratio와 eta contrast norm도 true value에 가장 가깝다.

### 3.2 K=4 strong common+specific setting

설정:

```text
K = 4, n = 1000, d = 100, rep = 100
common variables = 6
component-specific variables = 16
true union q = 22
kappa = (30, 45, 65, 90)
```

| Method | ARI | Selected q | TPR | FPR | Precision | F1 |
|:---|---:|---:|---:|---:|---:|---:|
| Rossi path BIC | 0.680 | 98.520 | 1.000 | 0.981 | 0.223 | 0.365 |
| Rossi path BIC + refit | 0.653 | 98.520 | 1.000 | 0.981 | 0.223 | 0.365 |
| Separate path/grid BIC | 0.684 | 86.460 | 1.000 | 0.826 | 0.258 | 0.409 |
| Separate path/grid BIC + refit | 0.657 | 86.460 | 1.000 | 0.826 | 0.258 | 0.409 |
| Eta centered path BIC | 0.625 | 24.750 | 0.994 | 0.037 | 0.890 | 0.937 |
| Eta centered path BIC + refit | 0.686 | 24.750 | 0.994 | 0.037 | 0.890 | 0.937 |

모수 추정 결과는 다음과 같다. MSE 지표는 raw scale이다.

| Method | MSE_mu | MSE_kappa | MSE_centered_eta | kappa_hat_mean |
|:---|---:|---:|---:|---:|
| Rossi path BIC | 0.00015 | 2.989 | 0.314 | 58.661 |
| Rossi path BIC + refit | 0.00033 | 3.427 | 0.594 | 58.735 |
| Separate path/grid BIC | 0.00008 | 8.762 | 0.179 | 56.089 |
| Separate path/grid BIC + refit | 0.00030 | 3.064 | 0.552 | 58.599 |
| Eta centered path BIC | 0.00029 | 14.485 | 0.424 | 58.468 |
| Eta centered path BIC + refit | 0.00010 | 1.901 | 0.185 | 58.040 |

해석:

Eta penalty + refit은 ARI를 유지하면서 true union q=22에 가까운 support를 선택한다. Refit 후 MSE_kappa도 1.901로 가장 낮다. 이 결과가 현재 본문에 가장 적합하다.

### 3.3 Concentration-dominant setting

| Method | ARI | Selected q | TPR | FPR | Precision | F1 |
|:---|---:|---:|---:|---:|---:|---:|
| Rossi | 0.513 | 98.500 | 1.000 | 0.983 | 0.102 | 0.184 |
| Separate penalty | 0.525 | 95.600 | 1.000 | 0.951 | 0.105 | 0.190 |
| Eta penalty + refit | 0.523 | 28.800 | 1.000 | 0.209 | 0.443 | 0.586 |

해석:

Rossi와 separate penalty는 clustering은 어느 정도 되지만 거의 모든 변수를 선택한다. Eta penalty는 ARI를 비슷하게 유지하면서 noise selection을 줄인다.

## 4. Weak concentration setting

### 4.1 K=4 weak common+specific setting

설정:

```text
K = 4, n = 1000, d = 100, rep = 100
common variables = 6
component-specific variables = 16
true union q = 22
kappa = (40, 50, 60, 70)
official path+BIC
target/adaptive/stability refinement = off
```

| Method | ARI | Selected q | TPR | FPR | Precision | F1 |
|:---|---:|---:|---:|---:|---:|---:|
| Rossi path BIC | 0.542 | 99.95 | 1.000 | 0.999 | 0.220 | 0.361 |
| Rossi path BIC + refit | 0.527 | 99.95 | 1.000 | 0.999 | 0.220 | 0.361 |
| Separate 2D path/grid BIC | 0.543 | 99.67 | 1.000 | 0.996 | 0.221 | 0.362 |
| Separate 2D path/grid BIC + refit | 0.526 | 99.67 | 1.000 | 0.996 | 0.221 | 0.362 |
| Eta centered path BIC | 0.568 | 24.09 | 1.000 | 0.027 | 0.918 | 0.956 |
| Eta centered path BIC + refit | 0.575 | 24.09 | 1.000 | 0.027 | 0.918 | 0.956 |

모수 추정 결과는 다음과 같다. MSE 지표는 raw scale이다.

| Method | MSE_mu | MSE_kappa | MSE_centered_eta | kappa_hat_mean |
|:---|---:|---:|---:|---:|
| Rossi path BIC | 0.000219 | 3.437 | 0.519 | 56.249 |
| Rossi path BIC + refit | 0.000285 | 3.623 | 0.668 | 56.247 |
| Separate 2D path/grid BIC | 0.000188 | 8.596 | 0.415 | 54.429 |
| Separate 2D path/grid BIC + refit | 0.000286 | 3.536 | 0.667 | 56.209 |
| Eta centered path BIC | 0.000172 | 7.409 | 0.355 | 54.506 |
| Eta centered path BIC + refit | 0.000075 | 1.824 | 0.183 | 55.497 |

해석:

Eta penalty + refit은 weak setting에서도 ARI를 유지하면서 true union q=22에 가까운 support를 선택한다. Rossi와 separate penalty는 거의 모든 변수를 선택하지만, Eta는 selected q=24.09와 FPR=0.027로 noise selection을 크게 줄인다. Refit 후 MSE_mu, MSE_kappa, MSE_centered_eta도 각각 0.000075, 1.824, 0.183으로 가장 낮다.

### 4.2 Diagnostic note

이번 official rerun에서는 Eta BIC가 null 또는 dense support로 튀지 않았다.

| Diagnostic | 결과 |
|:---|:---|
| raw ERROR rows | 0 |
| eta path candidates | 5508 rows, 100 reps |
| rep with q=17-27 candidate | 99/100 |
| Eta BIC selected q=0 | 0/100 |
| Eta BIC selected q=17-27 | 97/100 |
| Eta BIC selected q>=75 | 0/100 |
| Eta + refit zero-support reps | 0/100 |
| objective n_decrease max | 0 |
| min objective diff | -8.37e-09 |
| max line-search halving | 19 |
| line-search rejected candidates | 0 |

Positive-support BIC는 `selected_q > 0` 후보 중에서만 BIC를 고르는 sensitivity check다. 이번 rerun에서는 standard Eta BIC가 이미 nonzero near-true support를 골라서 positive-support 결과가 동일하므로 본문 표에서는 제외했다.

다만 이전 diagnostic run에서는 path construction에 따라 null/dense instability가 나타났다. 따라서 weak setting은 strong setting만큼 안정적인 main evidence라기보다, 보조 robustness evidence로 두는 것이 안전하다.

## 5. Stability selection 진단

Stability selection도 바로 해결책이 되지는 않았다.

| Diagnostic | 결과 | 해석 |
|:---|:---|:---|
| threshold 0.6 smoke5 | 5회 중 4회 zero support | 공식 tuning 후보로 보기 어려움 |
| threshold sweep 0.2-0.6 | 모든 threshold에서 4/5 zero support | threshold만 낮춰도 해결 안 됨 |
| subsample diagnostic | zero-support reps에서는 20개 subsample 모두 q=0 선택 | fit failure가 아니라 subsample BIC가 반복적으로 null을 선택 |
| IC slope sensitivity | gamma를 낮추면 zero는 줄지만 dense/FPR 증가 | 단순 df 상수항 수정은 해결책 아님 |

다음 보강은 stability threshold 조정보다 alternative IC, selection rule, 또는 Eta update 자체의 개선 쪽이 우선이다.

## 6. 교수님께 확인할 질문

1. Weak setting을 본문 보조 결과로 둘지 appendix/limitation으로 낮출지?
2. 다음 보강은 alternative IC와 update 개선 중 어디를 우선할지?
