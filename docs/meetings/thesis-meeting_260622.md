# Thesis Meeting 260622

업데이트: 2026-06-16

## 1. 핵심 결론

Eta penalty는 ARI를 크게 올리는 방법이라기보다, vMF mixture 안에서 posterior decision parameter인 `eta = kappa * mu`의 component contrast를 sparse하게 만들어, ARI를 유지하면서 해석 가능한 변수 선택을 제공하는 방법이다.

따라서 본문 주장은 “clustering accuracy 개선”보다 “model-based sparse interpretation”으로 정리함.

## 2. 모형 아이디어

관측값은 단위구면 위의 방향자료 $x_i \in \mathbb{S}^{d-1}$이고, $K$-component vMF mixture를 사용한다.

**Mixture model**

$$
p(x_i;\Theta)
=
\sum_{k=1}^K
\alpha_k C_d(\kappa_k)
\exp(\kappa_k \mu_k^\top x_i),
\qquad
\|\mu_k\|_2 = 1,\quad \kappa_k > 0
$$

**Natural parameter**

$$
\eta_k = \kappa_k \mu_k
$$

Posterior responsibility는 다음과 같다.

$$
\tau_{ik}
=
\frac{
\alpha_k C_d(\kappa_k)\exp(\eta_k^\top x_i)
}{
\sum_{\ell=1}^K
\alpha_\ell C_d(\kappa_\ell)\exp(\eta_\ell^\top x_i)
}
$$

K=2에서 posterior decision boundary는

$$
\log\frac{\tau_{i2}}{\tau_{i1}}
=
\mathrm{const}
+(\eta_2-\eta_1)^\top x_i
$$

로 정리된다. 따라서 변수 선택은 $\mu_k$ 자체보다 posterior decision에 직접 들어가는 $\eta_k$ contrast를 기준으로 하는 것이 자연스럽다.

**Observed log-likelihood**

$$
\ell(\Theta)
=
\sum_{i=1}^n
\log\left[
\sum_{k=1}^K
\alpha_k C_d(\kappa_k)\exp(\eta_k^\top x_i)
\right]
$$

K>2에서는 coordinate별 centered eta contrast를 사용한다.

$$
\bar{\eta}_j
=
\frac{1}{K}\sum_{\ell=1}^K \eta_{\ell j},
\qquad
c_{kj}
=
\eta_{kj}-\bar{\eta}_j
$$

**Penalty and objective**

$$
P_\eta(\Theta)
=
\lambda_\eta
\sum_{j=1}^d
\|c_{\cdot j}\|_2
$$

$$
\mathcal{L}_p(\Theta)
=
\ell(\Theta)-P_\eta(\Theta)
$$

현재 구현은 exact penalized EM이 아니라 proximal EM-type update다.

## 3. 핵심 simulation 결과

### 3.1 K=2 기본 메커니즘

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

모수 추정 결과는 다음과 같다. MSE 지표는 기존 simulation 문서와 같이 100배 스케일로 표시했다.

| Method | MSE_mu | MSE_kappa | MSE_Delta_eta | kappa ratio | eta contrast norm |
|:---|---:|---:|---:|---:|---:|
| Rossi path BIC | 0.0176 | 127.629 | 24.534 | 10.062 | 181.179 |
| Rossi path BIC + refit | 0.0061 | 140.965 | 37.797 | 9.951 | 180.821 |
| Separate path/grid BIC | 0.0176 | 127.629 | 24.534 | 10.062 | 181.179 |
| Separate path/grid BIC + refit | 0.0061 | 140.965 | 37.797 | 9.951 | 180.821 |
| Eta path BIC | 0.0180 | 741.495 | 29.167 | 8.559 | 175.542 |
| Eta path BIC + refit | 0.0041 | 118.451 | 21.585 | 9.960 | 180.630 |

K=2 기본 환경에서는 모든 방법의 ARI가 1.000이지만, Eta penalty가 selected q와 FPR을 가장 낮춘다. Refit 후 kappa ratio와 eta contrast norm도 true value에 가장 가깝다.

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

모수 추정 결과는 다음과 같다.

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

## 4. Weak setting 진단

Weak concentration setting에서는 처음 summary만 보면 Eta penalty가 잘 작동하는 것처럼 보였다. 하지만 line-search safeguard, zero-support 처리, path candidates 저장 후 다시 확인하니 Eta BIC가 null support 또는 dense support로 불안정하게 튀는 문제가 확인됐다.

| path construction | scope | near22 후보율 | BIC null률 | positive dense률 | 판단 |
|:---|:---|---:|---:|---:|:---|
| no refinement | weak100 | 0.23 | 0.73 | 0.72 | 기본 path가 중간 support를 충분히 만들지 못함 |
| oracle target-refine | weak100 | 0.89 | 0.14 | 0.09 | true q 주변 정보를 쓰므로 공식 알고리즘 불가 |
| adaptive v1 | weak100 | 0.73 | 0.24 | 0.24 | 개선은 있으나 oracle 수준은 아님 |
| adaptive v2 | smoke10 | 0.50 | 0.50 | 0.50 | midpoint refinement만으로 부족 |
| adaptive v2.1 | smoke10 | 0.50 | 0.40 | 0.40 | duplicate endpoint를 써도 support 다양성 부족 |
| adaptive v3 | smoke10 | 0.50 | 0.60 | 0.40 | 990개 평가에도 saved unique support 6개뿐 |

결론:

Weak setting의 문제는 단순히 lambda grid가 성긴 문제가 아니다. Proximal path가 같은 support plateau에 머물고, BIC가 null/dense 쪽으로 불안정하게 선택하는 문제로 보는 것이 더 타당하다. 따라서 weak setting은 본문 성공 사례가 아니라 appendix diagnostic 또는 limitation으로 낮추는 것이 안전하다.

## 5. Stability selection 진단

Stability selection도 바로 해결책이 되지는 않았다.

| Diagnostic | 결과 | 해석 |
|:---|:---|:---|
| threshold 0.6 smoke5 | 5회 중 4회 zero support | 공식 tuning 후보로 보기 어려움 |
| threshold sweep 0.2-0.6 | 모든 threshold에서 4/5 zero support | threshold만 낮춰도 해결 안 됨 |
| subsample diagnostic | zero-support reps에서는 20개 subsample 모두 q=0 선택 | fit failure가 아니라 subsample BIC가 반복적으로 null을 선택 |
| IC slope sensitivity | gamma를 낮추면 zero는 줄지만 dense/FPR 증가 | 단순 df 상수항 수정은 해결책 아님 |

다음 보강은 stability threshold 조정보다 alternative IC, selection rule, 또는 Eta update 자체의 개선 쪽이 우선이다.

