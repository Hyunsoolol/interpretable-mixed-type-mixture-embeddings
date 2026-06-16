# Thesis Meeting 260624

업데이트: 2026-06-16

## 1. 핵심 결론

Eta-group은 ARI를 크게 올리는 방법이라기보다, vMF mixture 안에서 posterior decision parameter인 `eta = kappa * mu`의 component contrast를 sparse하게 만들어, ARI를 유지하면서 해석 가능한 변수 선택을 제공하는 방법이다.

따라서 본문 주장은 “clustering accuracy 개선”보다 “model-based sparse interpretation”으로 정리함.

| 표기 | 의미 |
|:---|:---|
| Rossi | Rossi & Barbaro sparse vMF |
| Separate | separate mu/kappa penalty |
| Eta-group | centered eta group lasso |
| Eta-ANOVA | centered eta ANOVA-L1 |
| + refit | selected support 고정 후 unpenalized refit |

## 2. 모형 아이디어

**Model**

$$p(x_i;\Theta)=\sum_{k=1}^K \alpha_k C_d(\kappa_k)\exp(\kappa_k \mu_k^\top x_i), \qquad \|\mu_k\|_2=1,\quad \kappa_k>0$$

$$\eta_k=\kappa_k\mu_k$$

**Posterior decision**

$$\log\frac{\tau_{i2}}{\tau_{i1}}=\mathrm{const}+(\eta_2-\eta_1)^\top x_i$$

변수 선택 대상은 $\mu_k$ 자체보다 posterior decision에 직접 들어가는 $\eta_k$ contrast로 둔다.

**Observed log-likelihood**

$$\ell(\Theta)=\sum_{i=1}^n \log\left[\sum_{k=1}^K \alpha_k C_d(\kappa_k)\exp(\eta_k^\top x_i)\right]$$

K>2에서는 coordinate별 centered eta contrast를 사용한다.

$$\bar{\eta}_j=\frac{1}{K}\sum_{\ell=1}^K \eta_{\ell j}, \qquad c_{kj}=\eta_{kj}-\bar{\eta}_j$$

**Penalty**

$$P_{\mathrm{group}}(\Theta)=\lambda_\eta\sum_{j=1}^d \|c_{\cdot j}\|_2$$

$$P_{\mathrm{ANOVA}\text{-}L1}(\Theta)=\lambda_\eta\sum_{j=1}^d\sum_{k=1}^K |c_{kj}|$$

$$\mathcal{L}_p(\Theta)=\ell(\Theta)-P_{\mathrm{group}}(\Theta)$$

현재 주 penalty는 Eta-group이다. 이유는 coordinate-level eta contrast 선택이 목표이고, Eta-ANOVA는 pilot에서 dense support로 갔기 때문이다.

| Scenario | Penalty | ARI | Selected q | FPR | Precision | F1 |
|:---|:---|---:|---:|---:|---:|---:|
| strong | Eta-group + refit | 0.684 | 25.45 | 0.046 | 0.867 | 0.925 |
| strong | Eta-ANOVA + refit | 0.652 | 99.90 | 0.999 | 0.220 | 0.361 |
| weak | Eta-group + refit | 0.565 | 23.90 | 0.024 | 0.924 | 0.960 |
| weak | Eta-ANOVA + refit | 0.515 | 99.50 | 0.994 | 0.221 | 0.362 |

| Scenario | Penalty | MSE_mu | MSE_kappa | MSE_centered_eta | kappa_hat_mean |
|:---|:---|---:|---:|---:|---:|
| strong | Eta-group + refit | 0.000101 | 1.992 | 0.191 | 58.007 |
| strong | Eta-ANOVA + refit | 0.000329 | 3.604 | 0.581 | 58.677 |
| weak | Eta-group + refit | 0.000073 | 1.614 | 0.172 | 55.298 |
| weak | Eta-ANOVA + refit | 0.000288 | 2.989 | 0.659 | 56.043 |

Eta-ANOVA는 sensitivity 후보로만 남긴다. 현재 구현은 exact penalized EM이 아니라 proximal EM-type update다.

## 3. 핵심 simulation 결과

### 3.1 K=2 toy setting

설정:

| 항목 | 값 |
|:---|:---|
| 데이터 크기 | K = 2, n = 1000, d = 100, rep = 20 |
| 활성 변수 구조 | 두 component가 같은 10개 active 좌표 사용 = true q 10 |
| 평균방향 | mu_1 = mu_2 |
| concentration | kappa = (20, 200) |
| 목적 | 평균방향은 같고 kappa 차이만 있는 toy setting에서 eta contrast 선택 확인 |

| Method | ARI | Selected q | TPR | FPR | Precision | F1 |
|:---|---:|---:|---:|---:|---:|---:|
| Rossi BIC | 1.000 | 23.300 | 1.000 | 0.148 | 0.443 | 0.610 |
| Rossi BIC + refit | 1.000 | 23.300 | 1.000 | 0.148 | 0.443 | 0.610 |
| Separate BIC | 1.000 | 23.300 | 1.000 | 0.148 | 0.443 | 0.610 |
| Separate BIC + refit | 1.000 | 23.300 | 1.000 | 0.148 | 0.443 | 0.610 |
| Eta-group BIC | 1.000 | 13.200 | 1.000 | 0.036 | 0.792 | 0.875 |
| Eta-group BIC + refit | 1.000 | 13.200 | 1.000 | 0.036 | 0.792 | 0.875 |

모수 추정 결과는 다음과 같다. 3.1과 3.2의 MSE 지표는 모두 raw scale로 통일했다.

| Method | MSE_mu | MSE_kappa | MSE_Delta_eta | kappa ratio | eta contrast norm |
|:---|---:|---:|---:|---:|---:|
| Rossi BIC | 0.000176 | 1.276 | 0.245 | 10.062 | 181.179 |
| Rossi BIC + refit | 0.000061 | 1.410 | 0.378 | 9.951 | 180.821 |
| Separate BIC | 0.000176 | 1.276 | 0.245 | 10.062 | 181.179 |
| Separate BIC + refit | 0.000061 | 1.410 | 0.378 | 9.951 | 180.821 |
| Eta-group BIC | 0.000180 | 7.415 | 0.292 | 8.559 | 175.542 |
| Eta-group BIC + refit | 0.000041 | 1.185 | 0.216 | 9.960 | 180.630 |

K=2 toy setting에서는 모든 방법의 ARI가 1.000이지만, Eta-group이 selected q와 FPR을 가장 낮춘다. Refit 후 kappa ratio와 eta contrast norm도 true value에 가장 가깝다.

### 3.2 Main setting: K=4 strong common+specific

설정:

| 항목 | 값 |
|:---|:---|
| 데이터 크기 | K = 4, n = 1000, d = 100, rep = 100 |
| 활성 변수 구조 | common 6개 + component-specific 4개씩 = true union q 22 |
| raw mu loading | common = 1.0, own-specific = w = 0.5, 나머지 = 0 |
| normalized mu 값 | common = 0.378, own-specific = 0.189, 나머지 = 0 |
| 평균방향 유사도 | mean pairwise cos(mu_k, mu_l) = 0.857 |
| concentration | kappa = (30, 45, 65, 90) |
| 목적 | common + component-specific 구조에서 ARI 유지와 sparse support recovery 확인 |

| Method | ARI | Selected q | TPR | FPR | Precision | F1 |
|:---|---:|---:|---:|---:|---:|---:|
| Rossi BIC | 0.680 | 98.520 | 1.000 | 0.981 | 0.223 | 0.365 |
| Rossi BIC + refit | 0.653 | 98.520 | 1.000 | 0.981 | 0.223 | 0.365 |
| Separate BIC | 0.684 | 86.460 | 1.000 | 0.826 | 0.258 | 0.409 |
| Separate BIC + refit | 0.657 | 86.460 | 1.000 | 0.826 | 0.258 | 0.409 |
| Eta-group BIC | 0.625 | 24.750 | 0.994 | 0.037 | 0.890 | 0.937 |
| Eta-group BIC + refit | 0.686 | 24.750 | 0.994 | 0.037 | 0.890 | 0.937 |

모수 추정 결과는 다음과 같다. MSE 지표는 raw scale이다.

| Method | MSE_mu | MSE_kappa | MSE_centered_eta | kappa_hat_mean |
|:---|---:|---:|---:|---:|
| Rossi BIC | 0.00015 | 2.989 | 0.314 | 58.661 |
| Rossi BIC + refit | 0.00033 | 3.427 | 0.594 | 58.735 |
| Separate BIC | 0.00008 | 8.762 | 0.179 | 56.089 |
| Separate BIC + refit | 0.00030 | 3.064 | 0.552 | 58.599 |
| Eta-group BIC | 0.00029 | 14.485 | 0.424 | 58.468 |
| Eta-group BIC + refit | 0.00010 | 1.901 | 0.185 | 58.040 |

해석:

Eta-group + refit은 ARI를 유지하면서 true union q=22에 가까운 support를 선택한다. Refit 후 MSE_kappa도 1.901로 가장 낮다. 이 결과가 현재 본문에 가장 적합하다.

보조 확인: concentration-dominant setting

| 항목 | 값 |
|:---|:---|
| 데이터 크기 | K = 4, n = 1000, d = 100 |
| 활성 변수 구조 | 모든 component가 같은 10개 좌표 사용 = true q 10 |
| mu construction | 첫 10개 좌표만 active, 평균방향 간 pairwise cosine을 약 0.95로 설정 |
| w 사용 여부 | component-specific weight가 없는 설계라 w는 사용하지 않음 |
| concentration | kappa = (25, 40, 65, 100) |
| 목적 | 평균방향이 매우 유사하고 concentration 차이가 클 때 noise selection 감소 확인 |

| Method | ARI | Selected q | TPR | FPR | Precision | F1 |
|:---|---:|---:|---:|---:|---:|---:|
| Rossi | 0.513 | 98.500 | 1.000 | 0.983 | 0.102 | 0.184 |
| Separate | 0.525 | 95.600 | 1.000 | 0.951 | 0.105 | 0.190 |
| Eta-group + refit | 0.523 | 28.800 | 1.000 | 0.209 | 0.443 | 0.586 |

이 보조 setting에서도 Eta-group은 ARI를 비슷하게 유지하면서 noise selection을 줄인다.

### 3.3 Signal strength sensitivity: w

설정:

| 항목 | 값 |
|:---|:---|
| 데이터 크기 | K = 4, n = 1000, d = 100, rep = 100 |
| 활성 변수 구조 | common 6개 + component-specific 4개씩 = true union q 22 |
| 변화시킨 값 | component-specific raw loading w = 0.25, 0.35, 0.50 |
| 고정한 값 | common raw loading = 1.0, 나머지 = 0 |
| normalization | 각 component의 raw mu vector를 row-normalize |
| concentration | kappa = (30, 45, 65, 90) |
| 목적 | component-specific signal strength가 변할 때 variable selection 안정성 확인 |

정규화 후 실제 mu 좌표값은 다음과 같다.

| w | row norm | common mu_j | own-specific mu_j | other/noise mu_j | mean pairwise cos(mu_k, mu_l) |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 2.500 | 0.400 | 0.100 | 0 | 0.960 |
| 0.35 | 2.548 | 0.393 | 0.137 | 0 | 0.924 |
| 0.50 | 2.646 | 0.378 | 0.189 | 0 | 0.857 |

성능 결과는 다음과 같다.

| w | Method | ARI | Selected q | TPR | FPR | Precision | F1 |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 0.25 | Rossi BIC | 0.394 | 98.07 | 1.000 | 0.975 | 0.225 | 0.367 |
| 0.25 | Rossi BIC + refit | 0.368 | 98.07 | 1.000 | 0.975 | 0.225 | 0.367 |
| 0.25 | Separate BIC | 0.401 | 93.50 | 0.999 | 0.917 | 0.236 | 0.382 |
| 0.25 | Separate BIC + refit | 0.367 | 93.50 | 0.999 | 0.917 | 0.236 | 0.382 |
| 0.25 | Eta-group BIC | 0.358 | 32.13 | 0.813 | 0.183 | 0.621 | 0.682 |
| 0.25 | Eta-group BIC + refit | 0.399 | 32.13 | 0.813 | 0.183 | 0.621 | 0.682 |
| 0.35 | Rossi BIC | 0.514 | 98.82 | 1.000 | 0.985 | 0.223 | 0.364 |
| 0.35 | Rossi BIC + refit | 0.481 | 98.82 | 1.000 | 0.985 | 0.223 | 0.364 |
| 0.35 | Separate BIC | 0.528 | 90.70 | 1.000 | 0.881 | 0.247 | 0.394 |
| 0.35 | Separate BIC + refit | 0.488 | 90.70 | 1.000 | 0.881 | 0.247 | 0.394 |
| 0.35 | Eta-group BIC | 0.458 | 29.81 | 0.919 | 0.123 | 0.723 | 0.794 |
| 0.35 | Eta-group BIC + refit | 0.505 | 29.81 | 0.919 | 0.123 | 0.723 | 0.794 |
| 0.50 | Rossi BIC | 0.680 | 98.52 | 1.000 | 0.981 | 0.223 | 0.365 |
| 0.50 | Rossi BIC + refit | 0.653 | 98.52 | 1.000 | 0.981 | 0.223 | 0.365 |
| 0.50 | Separate BIC | 0.684 | 86.46 | 1.000 | 0.826 | 0.258 | 0.409 |
| 0.50 | Separate BIC + refit | 0.657 | 86.46 | 1.000 | 0.826 | 0.258 | 0.409 |
| 0.50 | Eta-group BIC | 0.625 | 24.75 | 0.994 | 0.037 | 0.890 | 0.937 |
| 0.50 | Eta-group BIC + refit | 0.686 | 24.75 | 0.994 | 0.037 | 0.890 | 0.937 |

모수 추정 결과는 다음과 같다. MSE 지표는 raw scale이다.

| w | Method | MSE_mu | MSE_kappa | MSE_centered_eta | kappa_hat_mean |
|---:|:---|---:|---:|---:|---:|
| 0.25 | Rossi BIC | 0.000466 | 53.570 | 1.288 | 61.007 |
| 0.25 | Rossi BIC + refit | 0.000878 | 78.474 | 2.351 | 61.228 |
| 0.25 | Separate BIC | 0.000289 | 60.743 | 0.935 | 58.975 |
| 0.25 | Separate BIC + refit | 0.000731 | 66.807 | 1.967 | 60.640 |
| 0.25 | Eta-group BIC | 0.000543 | 4.999e9 | 3.352e7 | 5062.159 |
| 0.25 | Eta-group BIC + refit | 0.000682 | 4.999e9 | 3.352e7 | 5062.191 |
| 0.35 | Rossi BIC | 0.000285 | 16.756 | 0.656 | 59.694 |
| 0.35 | Rossi BIC + refit | 0.000556 | 20.778 | 1.167 | 59.713 |
| 0.35 | Separate BIC | 0.000149 | 12.341 | 0.318 | 57.198 |
| 0.35 | Separate BIC + refit | 0.000460 | 12.747 | 0.927 | 59.190 |
| 0.35 | Eta-group BIC | 0.000405 | 99.000 | 1.283 | 61.973 |
| 0.35 | Eta-group BIC + refit | 0.000359 | 43.882 | 0.969 | 60.408 |
| 0.50 | Rossi BIC | 0.000153 | 2.989 | 0.314 | 58.661 |
| 0.50 | Rossi BIC + refit | 0.000326 | 3.427 | 0.594 | 58.735 |
| 0.50 | Separate BIC | 0.000084 | 8.762 | 0.179 | 56.089 |
| 0.50 | Separate BIC + refit | 0.000302 | 3.064 | 0.552 | 58.599 |
| 0.50 | Eta-group BIC | 0.000291 | 14.485 | 0.424 | 58.468 |
| 0.50 | Eta-group BIC + refit | 0.000102 | 1.901 | 0.185 | 58.040 |

해석:

w가 커질수록 component-specific signal이 강해지고, 평균 방향 간 cosine은 낮아져 군집 구분이 쉬워진다. Rossi와 Separate는 모든 w에서 거의 모든 변수를 선택한다. Eta-group은 w=0.25에서도 selected q와 FPR을 크게 줄이지만 specific 변수 일부를 놓친다. w=0.50에서는 true union q=22에 가장 가깝고 F1도 가장 높다. 단, w=0.25의 Eta-group 모수 MSE 평균은 일부 kappa outlier 때문에 보조적으로만 해석한다.

### 3.4 Weak concentration setting

설정:

| 항목 | 값 |
|:---|:---|
| 데이터 크기 | K = 4, n = 1000, d = 100, rep = 100 |
| 활성 변수 구조 | common 6개 + component-specific 4개씩 = true union q 22 |
| raw mu loading | common = 1.0, own-specific = w = 0.5, 나머지 = 0 |
| normalized mu 값 | common = 0.378, own-specific = 0.189, 나머지 = 0 |
| 평균방향 유사도 | mean pairwise cos(mu_k, mu_l) = 0.857 |
| concentration | kappa = (40, 50, 60, 70) |
| tuning | official path+BIC, target/adaptive/stability refinement off |
| 목적 | concentration 차이가 약할 때 sparse support recovery가 유지되는지 확인 |

| Method | ARI | Selected q | TPR | FPR | Precision | F1 |
|:---|---:|---:|---:|---:|---:|---:|
| Rossi BIC | 0.542 | 99.95 | 1.000 | 0.999 | 0.220 | 0.361 |
| Rossi BIC + refit | 0.527 | 99.95 | 1.000 | 0.999 | 0.220 | 0.361 |
| Separate BIC | 0.543 | 99.67 | 1.000 | 0.996 | 0.221 | 0.362 |
| Separate BIC + refit | 0.526 | 99.67 | 1.000 | 0.996 | 0.221 | 0.362 |
| Eta-group BIC | 0.568 | 24.09 | 1.000 | 0.027 | 0.918 | 0.956 |
| Eta-group BIC + refit | 0.575 | 24.09 | 1.000 | 0.027 | 0.918 | 0.956 |

모수 추정 결과는 다음과 같다. MSE 지표는 raw scale이다.

| Method | MSE_mu | MSE_kappa | MSE_centered_eta | kappa_hat_mean |
|:---|---:|---:|---:|---:|
| Rossi BIC | 0.000219 | 3.437 | 0.519 | 56.249 |
| Rossi BIC + refit | 0.000285 | 3.623 | 0.668 | 56.247 |
| Separate BIC | 0.000188 | 8.596 | 0.415 | 54.429 |
| Separate BIC + refit | 0.000286 | 3.536 | 0.667 | 56.209 |
| Eta-group BIC | 0.000172 | 7.409 | 0.355 | 54.506 |
| Eta-group BIC + refit | 0.000075 | 1.824 | 0.183 | 55.497 |

해석:

Eta-group + refit은 weak setting에서도 ARI를 유지하면서 true union q=22에 가까운 support를 선택한다. Rossi와 Separate는 거의 모든 변수를 선택하지만, Eta-group은 selected q=24.09와 FPR=0.027로 noise selection을 크게 줄인다. 다만 weak setting은 concentration 차이가 작아지는 경우의 robustness evidence로 두고, main success claim은 strong setting 중심으로 두는 편이 안전하다.

### 3.5 Moderate high-dimensional robustness: d=200

설정:

| 항목 | 값 |
|:---|:---|
| 데이터 크기 | K = 4, n = 1000, d = 200, rep = 50 |
| 활성 변수 구조 | common 6개 + component-specific 4개씩 = true union q 22 |
| raw mu loading | common = 1.0, own-specific = w = 0.5, 나머지 = 0 |
| normalized mu 값 | common = 0.378, own-specific = 0.189, 나머지 = 0 |
| 평균방향 유사도 | mean pairwise cos(mu_k, mu_l) = 0.857 |
| concentration | kappa = (30, 45, 65, 90) |
| 목적 | d가 커질 때 Eta-group의 sparse support recovery가 유지되는지 확인 |

| Method | ARI | Selected q | TPR | FPR | Precision | F1 |
|:---|---:|---:|---:|---:|---:|---:|
| Rossi BIC | 0.393 | 200.00 | 1.000 | 1.000 | 0.110 | 0.198 |
| Rossi BIC + refit | 0.389 | 200.00 | 1.000 | 1.000 | 0.110 | 0.198 |
| Separate BIC | 0.398 | 200.00 | 1.000 | 1.000 | 0.110 | 0.198 |
| Separate BIC + refit | 0.388 | 200.00 | 1.000 | 1.000 | 0.110 | 0.198 |
| Eta-group BIC | 0.459 | 120.06 | 0.989 | 0.552 | 0.208 | 0.331 |
| Eta-group BIC + refit | 0.430 | 120.06 | 0.989 | 0.552 | 0.208 | 0.331 |

모수 추정 결과는 다음과 같다. MSE 지표는 raw scale이다.

| Method | MSE_mu | MSE_kappa | MSE_centered_eta | kappa_hat_mean |
|:---|---:|---:|---:|---:|
| Rossi BIC | 0.000964 | 123.189 | 2.407 | 63.286 |
| Rossi BIC + refit | 0.001040 | 123.889 | 2.572 | 63.223 |
| Separate BIC | 0.000898 | 73.406 | 1.823 | 58.812 |
| Separate BIC + refit | 0.001033 | 104.516 | 2.420 | 62.509 |
| Eta-group BIC | 0.000434 | 62.041 | 0.905 | 60.350 |
| Eta-group BIC + refit | 0.000691 | 93.797 | 1.818 | 62.299 |

해석:

d=200에서는 Rossi와 Separate가 모든 변수를 선택한다. Eta-group은 ARI와 모수 MSE에서는 상대적으로 낫지만, 기본 path에서는 selected q=120.06, FPR=0.552로 true union q=22 근처의 sparse recovery는 유지하지 못한다. 따라서 기본 d=200 결과는 본문 핵심 성공 사례라기보다, 고차원에서 Eta-group이 dense baseline보다는 낫지만 selection sparsity가 약해지는 robustness evidence로 보는 것이 안전하다.

### 3.6 High-dimensional stress: d=400

설정:

| 항목 | 값 |
|:---|:---|
| 데이터 크기 | K = 4, n = 1000, d = 400, rep = 20 |
| 활성 변수 구조 | common 6개 + component-specific 4개씩 = true union q 22 |
| raw mu loading | common = 1.0, own-specific = w = 0.5, 나머지 = 0 |
| normalized mu 값 | common = 0.378, own-specific = 0.189, 나머지 = 0 |
| 평균방향 유사도 | mean pairwise cos(mu_k, mu_l) = 0.857 |
| concentration | kappa = (30, 45, 65, 90) |
| 목적 | 더 강한 고차원 stress에서 Eta-group의 한계를 확인 |

| Method | ARI | Selected q | TPR | FPR | Precision | F1 |
|:---|---:|---:|---:|---:|---:|---:|
| Rossi BIC | 0.142 | 400.00 | 1.000 | 1.000 | 0.055 | 0.104 |
| Rossi BIC + refit | 0.141 | 400.00 | 1.000 | 1.000 | 0.055 | 0.104 |
| Separate BIC | 0.146 | 399.95 | 1.000 | 1.000 | 0.055 | 0.104 |
| Separate BIC + refit | 0.144 | 399.95 | 1.000 | 1.000 | 0.055 | 0.104 |
| Eta-group BIC | 0.206 | 262.95 | 0.920 | 0.642 | 0.080 | 0.146 |
| Eta-group BIC + refit | 0.182 | 262.95 | 0.920 | 0.642 | 0.080 | 0.146 |

모수 추정 결과는 다음과 같다. MSE 지표는 raw scale이다.

| Method | MSE_mu | MSE_kappa | MSE_centered_eta | kappa_hat_mean |
|:---|---:|---:|---:|---:|
| Rossi BIC | 0.001252 | 411.108 | 4.692 | 71.872 |
| Rossi BIC + refit | 0.001300 | 412.507 | 4.854 | 71.886 |
| Separate BIC | 0.001246 | 306.790 | 4.196 | 67.977 |
| Separate BIC + refit | 0.001330 | 397.506 | 4.860 | 71.287 |
| Eta-group BIC | 0.000701 | 216.991 | 1.942 | 63.839 |
| Eta-group BIC + refit | 0.001107 | 343.629 | 4.016 | 70.811 |

해석:

d=400에서는 모든 방법의 ARI가 낮고, Rossi와 Separate는 거의 완전히 dense하게 작동한다. Eta-group은 ARI와 FPR에서 상대적으로 낫지만 selected q=262.95로 여전히 매우 크고, true union q=22 근처의 sparse support recovery라고 보기 어렵다. 따라서 d=400은 본문 성공 사례가 아니라 high-dimensional stress limitation으로 두는 것이 적절하다. 이 결과는 고차원 setting에서 BIC/path tuning 또는 update 보강이 필요하다는 근거로 쓰는 편이 안전하다.

d=100 strong setting의 Eta-group + refit은 ARI=0.686, selected q=24.75, FPR=0.037, F1=0.937이었다. 같은 구조에서 d=200, d=400으로 가면 Eta-group은 dense baseline보다는 낫지만 selected q와 FPR이 크게 증가한다. 즉 고차원에서는 clustering보다 sparse support recovery가 먼저 불안정해진다.

### 3.7 High-dimensional tuning/path diagnostic

고차원에서 BIC가 dense support를 고르는지 확인하기 위해, 새 simulation 없이 저장된 Eta path candidates에서 stronger tuning criteria를 재계산했다. 이는 official tuning 변경이 아니라 diagnostic-only sensitivity다.

| setting | criterion | Selected q | Zero rate | Dense rate | ARI | FPR | Precision | F1 |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| d=200 | BIC_current | 120.06 | 0.00 | 0.20 | 0.459 | 0.552 | 0.208 | 0.331 |
| d=200 | EBIC_gamma_0.5 | 119.98 | 0.00 | 0.20 | 0.459 | 0.552 | 0.209 | 0.331 |
| d=200 | EBIC_gamma_1.0 | 119.98 | 0.00 | 0.20 | 0.459 | 0.552 | 0.209 | 0.331 |
| d=200 | RIC_like | 119.98 | 0.00 | 0.20 | 0.459 | 0.552 | 0.209 | 0.331 |
| d=400 | BIC_current | 262.95 | 0.00 | 0.30 | 0.206 | 0.642 | 0.080 | 0.146 |
| d=400 | EBIC_gamma_0.5 | 262.95 | 0.00 | 0.30 | 0.206 | 0.642 | 0.080 | 0.146 |
| d=400 | EBIC_gamma_1.0 | 262.95 | 0.00 | 0.30 | 0.206 | 0.642 | 0.080 | 0.146 |
| d=400 | RIC_like | 262.90 | 0.00 | 0.30 | 0.206 | 0.642 | 0.080 | 0.146 |

해석:

EBIC/RIC-like/log(d)-slope 재선택은 BIC 결과를 거의 바꾸지 못했다. 기본 d=200 path에서 q=17-27 후보가 존재한 replication은 2%뿐이고, d=400에서는 0%였다. 따라서 고차원 실패는 단순히 BIC penalty가 약해서 생긴 문제가 아니라, 현재 Eta path가 true-support-size 후보를 거의 만들지 못하는 path/update 문제로 보는 것이 더 타당하다.

d=200에서 Eta path를 더 길게 잡은 diagnostic도 확인했다. 이는 official tuning 변경이 아니라, path density/range가 성능에 미치는 영향을 확인하기 위한 추가 점검이다.

| setting | path | reps | near22 후보율 | Selected q | Dense rate | ARI | FPR | Precision | F1 |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| d=200 | 기본 path | 50 | 0.02 | 120.06 | 0.20 | 0.459 | 0.552 | 0.208 | 0.331 |
| d=200 | long path 240 | 50 | 0.34 | 62.14 | 0.04 | 0.447 | 0.235 | 0.507 | 0.584 |

long path는 ARI를 크게 올리지는 않지만 selected q와 FPR을 줄이고 Precision/F1을 개선한다. 다만 selected q=62.14는 여전히 true union q=22보다 크고, near22 후보율도 34%에 그친다. 즉 고차원에서는 path density/range가 중요한 next tuning candidate이지만, 이 결과만으로 official tuning을 바꾸기는 이르다. 다음 보강 후보는 path construction, MM/coordinate update, 또는 고차원용 screening 전략이다.
