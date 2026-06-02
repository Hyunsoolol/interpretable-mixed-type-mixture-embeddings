# 연구 진행 정리: Rossi & Barbaro (2022) sparse vMF 재현 실험

## 1. 목적

제안 모형의 시뮬레이션을 바로 수행하기 전에, 먼저 Rossi & Barbaro (2022)의 **sparse von Mises-Fisher mixture** 모형을 R로 재현하였다.

본 단계의 목적은 다음과 같다.

1. vMF mixture 기반 sparse prototype 모형의 추정 구조를 정확히 이해한다.
2. 논문 artificial simulation의 핵심 패턴이 재현되는지 확인한다.
3. 이후 제안 모형 `eta_h = kappa_h mu_h` 기반 penalty 및 two-stage refit의 baseline으로 사용한다.

참고 논문:

- Rossi, F., & Barbaro, F. (2022). *Mixture of von Mises-Fisher distribution with sparse prototypes*. Neurocomputing.
- arXiv: <https://arxiv.org/abs/2212.14591>

---

## 2. Rossi & Barbaro (2022) 모형

### 2.1 vMF mixture

관측치 `x_i`는 단위구 `S^{d-1}` 위의 방향 데이터라고 가정한다.

```math
f(x \mid \mu, \kappa)
= c_d(\kappa)\exp(\kappa \mu^\top x),
\qquad x,\mu \in S^{d-1}.
```

혼합모형은 다음과 같다.

```math
p(x_i \mid \Theta)
= \sum_{k=1}^{K}
\alpha_k c_d(\kappa_k)
\exp(\kappa_k \mu_k^\top x_i).
```

여기서

- `alpha_k`: mixing proportion
- `mu_k`: component direction, `||mu_k||_2 = 1`
- `kappa_k`: concentration parameter

이다.

### 2.2 Sparse prototype penalty

논문의 핵심은 각 component direction `mu_k`에 직접 `L1` penalty를 주는 것이다.

```math
L_p(\Theta \mid X)
= L(\Theta \mid X)
- \beta \sum_{k=1}^{K} \|\mu_k\|_1.
```

즉 penalty 대상은 자연모수 `eta_k = kappa_k mu_k`가 아니라 **방향 평균 `mu_k` 자체**이다.

이 구조의 의미:

- `mu_k`의 많은 좌표를 0으로 만들어 sparse prototype을 얻는다.
- 해석 가능한 방향 prototype을 만드는 데 유리하다.
- 하지만 posterior log-odds의 판별항은 `kappa_k mu_k`에 의해 결정되므로, `kappa` 차이가 penalty에 직접 반영되지는 않는다.
- 최종 추정량은 penalized estimator이므로 L1 shrinkage bias가 남는다.

이는 본 연구에서 제안하는 `eta_h = kappa_h mu_h` 기반 cluster-contrast penalty 및 two-stage refit과 구분되는 핵심 지점이다.

---

## 3. 추정 알고리즘

### 3.1 E-step

responsibility는 다음과 같다.

```math
\tau_{ik}
=
\frac{
\alpha_k c_d(\kappa_k)\exp(\kappa_k \mu_k^\top x_i)
}{
\sum_{\ell=1}^{K}
\alpha_\ell c_d(\kappa_\ell)\exp(\kappa_\ell \mu_\ell^\top x_i)
}.
```

### 3.2 M-step

```math
N_k = \sum_i \tau_{ik},
\qquad
r_k = \sum_i \tau_{ik}x_i.
```

mixing proportion은

```math
\alpha_k = N_k/n
```

으로 업데이트한다.

`mu_k` 업데이트는 soft-thresholding 형태를 갖는다.

```math
\mu_{kj}
\propto
\operatorname{sign}(r_{kj})
\max(\kappa_k |r_{kj}| - \beta, 0),
```

이후 `||mu_k||_2 = 1`이 되도록 정규화한다.

`kappa_k`는 평균 resultant length를 이용해 근사 업데이트한다.

```math
\rho_k
=
\frac{\mu_k^\top r_k}{N_k},
\qquad
\kappa_k
\approx
\frac{d\rho_k - \rho_k^3}{1-\rho_k^2}.
```

`mu_k`와 `kappa_k`가 서로 의존하므로, M-step 내부에서 fixed-point iteration을 수행한다.

### 3.3 Beta path-following

논문은 단순 grid search가 아니라 `beta` path-following을 사용한다.

시작점은 `beta = 0`인 dense vMF mixture이고, 이후 현재 solution에서 sparse pattern이 바뀌는 다음 지점으로 `beta`를 증가시킨다.

현재 component/coordinate에 대해

```math
\kappa_k |r_{kj}| - \beta
```

가 양수인 값 중 가장 작은 margin을 이용하여 다음 `beta` 후보를 정한다.

---

## 4. 구현

R로 구현한 재현 스크립트:

- `rossi_barbaro_2022_reproduction.r`
- `rb2022_paperlike_n20_run.r`

구현 내용:

1. vMF random sampler
2. vMF mixture EM
3. `mu_k`에 대한 L1 sparse update
4. component-specific `kappa_k`
5. `beta` path-following
6. AIC, BIC, RIC, RICc, EBIC 계산
7. spherical k-means baseline
8. 논문 artificial simulation grid 재현
9. 결과 요약 및 시각화

---

## 5. 시뮬레이션 세팅

### 5.1 논문 세팅

논문의 artificial simulation은 다음 구조를 사용한다.

| 항목 | 값 |
|---|---|
| dimension | `d = 100` |
| true components | `K* = 4` |
| sample size | `N = 200, 1000` |
| overlap | `2.5%`, `5%` |
| true nonzero fraction | `5%`, `10%`, `15%` |
| candidate K | `1, ..., 6` |
| repetitions | `100 datasets` |
| random starts | `10` |
| concentration | component-specific `kappa_k` |

### 5.2 이번 재현 세팅

계산 시간을 고려해 같은 grid를 유지하되 반복 수와 초기화 수를 줄였다.

| 항목 | 값 |
|---|---|
| dimension | `d = 100` |
| true components | `K* = 4` |
| sample size | `N = 200, 1000` |
| overlap | `2.5%`, `5%` |
| true nonzero fraction | `5%`, `10%`, `15%` |
| candidate K | `1, ..., 6` |
| repetitions | `20 datasets` |
| random starts | `5` |
| max path steps | `400` |
| workers | `4` |

따라서 이번 결과는 논문 full replication이 아니라 **medium-budget reproduction**으로 해석한다.

---

## 6. 결과

### 6.1 BIC 기준 compact summary

| N | overlap | true nonzero | dense K | sparse K | dense ARI | sparse ARI | sparse nonzero |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | 0.025 | 0.05 | 1.82 | 3.64 | 0.259 | 0.758 | 0.152 |
| 200 | 0.025 | 0.10 | 1.82 | 3.64 | 0.255 | 0.772 | 0.186 |
| 200 | 0.025 | 0.15 | 1.78 | 3.96 | 0.250 | 0.855 | 0.205 |
| 200 | 0.050 | 0.05 | 1.64 | 3.28 | 0.169 | 0.579 | 0.144 |
| 200 | 0.050 | 0.10 | 1.65 | 3.37 | 0.184 | 0.629 | 0.185 |
| 200 | 0.050 | 0.15 | 1.61 | 3.51 | 0.170 | 0.655 | 0.199 |
| 1000 | 0.025 | 0.05 | 4.06 | 4.00 | 0.929 | 0.941 | 0.125 |
| 1000 | 0.025 | 0.10 | 4.04 | 4.00 | 0.949 | 0.956 | 0.143 |
| 1000 | 0.025 | 0.15 | 4.02 | 4.04 | 0.942 | 0.946 | 0.178 |
| 1000 | 0.050 | 0.05 | 4.05 | 4.01 | 0.872 | 0.892 | 0.101 |
| 1000 | 0.050 | 0.10 | 4.08 | 4.02 | 0.887 | 0.905 | 0.141 |
| 1000 | 0.050 | 0.15 | 4.06 | 4.10 | 0.899 | 0.903 | 0.181 |

결과 파일:

- `results/rb2022_paperlike_n20_260602/rb2022_paperlike_n20_summary.csv`
- `results/rb2022_paperlike_n20_260602/rb2022_paperlike_n20_BIC_compact.csv`

참고: full raw path 및 cell-level 중간 CSV는 용량이 커서 git tracking에서는 제외하고 로컬 산출물로 보관한다.

### 6.2 ARI by K, N=1000

![ARI by K, N=1000](results/rb2022_paperlike_n20_260602/plots/fig_ari_by_k_N1000.png)

해석:

- 모든 조건에서 `K=4` 근처에서 ARI가 최고다.
- `K=5,6`에서는 ARI가 다소 낮아진다.
- dense, AIC, BIC의 ARI curve가 매우 유사하다.
- 논문 Figure 13의 핵심 패턴과 일치한다.

### 6.3 Selected K counts, N=1000

![Selected K counts, N=1000](results/rb2022_paperlike_n20_260602/plots/fig_selected_k_counts_N1000.png)

해석:

- Sparse-BIC는 대부분 조건에서 `K=4`를 거의 항상 선택한다.
- Dense-BIC도 `K=4`에 집중하지만, 일부 조건에서 `K=5`를 선택한다.
- Sparse-AIC는 더 큰 `K`를 선택하는 경향이 있다.
- 논문에서 언급한 것처럼 BIC가 true `K` 회복에 더 안정적이다.

### 6.4 Selected K counts, N=200

![Selected K counts, N=200](results/rb2022_paperlike_n20_260602/plots/fig_selected_k_counts_N200.png)

해석:

- Dense-BIC는 `K=1` 또는 `K=2`로 underfit하는 경향이 강하다.
- Sparse-BIC는 `K=3` 또는 `K=4`로 올라가며, dense보다 true `K=4`에 가깝다.
- Sparse-AIC는 `K=5,6`까지 선택하는 overfit 경향이 있다.
- 작은 sample에서는 model selection이 불안정하지만, sparsification이 effective parameter 수를 줄여 BIC의 K 선택을 개선한다.

### 6.5 Nonzero fraction by K, N=1000

![Nonzero fraction by K, N=1000](results/rb2022_paperlike_n20_260602/plots/fig_sparsity_by_k_N1000.png)

해석:

- Dense model은 거의 모든 좌표가 nonzero이다.
- Sparse-BIC는 true `K=4`에서 nonzero fraction이 낮게 유지된다.
- true sparsity가 커질수록 estimated nonzero fraction도 증가한다.
- 다만 estimated nonzero fraction은 true value보다 큰 편이다.

### 6.6 Precision/Recall, K=4, N=1000

![Precision/Recall, K=4, N=1000](results/rb2022_paperlike_n20_260602/plots/fig_precision_recall_K4_N1000.png)

`K=4`, `N=1000`에서 BIC 기준 nonzero entry recovery:

| overlap | true nonzero | precision | recall |
|---:|---:|---:|---:|
| 0.025 | 0.05 | 0.444 | 0.960 |
| 0.025 | 0.10 | 0.767 | 0.900 |
| 0.025 | 0.15 | 0.891 | 0.863 |
| 0.050 | 0.05 | 0.602 | 0.925 |
| 0.050 | 0.10 | 0.781 | 0.893 |
| 0.050 | 0.15 | 0.891 | 0.869 |

해석:

- Recall은 전반적으로 높다.
- Precision은 true sparsity가 낮을 때 낮고, true sparsity가 커질수록 높아진다.
- 즉 true nonzero 좌표를 많이 포함하지만, false positive도 남는다.
- 논문 Figure 16의 주요 메시지와 일치한다.

---

## 7. 논문 결과와 비교

### 7.1 잘 재현된 부분

1. **N=1000에서 K 회복**
   - 논문: BIC는 dense/sparse 모두 `K=4`를 안정적으로 회복.
   - 이번 결과: sparse-BIC는 거의 정확히 `K=4`, dense-BIC는 평균 `K=4.02~4.08`.

2. **Sparse model의 ARI 개선**
   - 논문: sparse vMF는 dense vMF보다 ARI가 같거나 약간 높다.
   - 이번 결과: `N=1000`에서 sparse ARI가 dense ARI보다 대체로 높다.

3. **N=200에서 sparse penalty의 K 선택 개선**
   - 논문: small sample에서는 dense BIC가 underfit할 수 있고, sparsity가 effective dimension을 줄여 K 선택을 개선한다.
   - 이번 결과: dense-BIC는 `K≈1.6~1.8`, sparse-BIC는 `K≈3.3~4.0`.

4. **Precision/Recall trade-off**
   - 논문: recall은 높고 precision은 낮거나 중간 수준.
   - 이번 결과: `N=1000, K=4`에서 BIC recall `0.86~0.96`, precision `0.44~0.89`.

### 7.2 차이가 남는 부분

1. **반복 수 차이**
   - 논문은 100 datasets.
   - 이번 재현은 20 datasets.
   - 따라서 exact numerical replication은 아니다.

2. **초기화 수 차이**
   - 논문은 10 random starts.
   - 이번 재현은 5 random starts.
   - mixture EM의 local optimum 영향이 남을 수 있다.

3. **N=200의 불안정성**
   - small sample에서는 K 선택과 ARI 변동이 크다.
   - 반복 수를 100으로 올리면 논문 figure와 더 가까운 smooth pattern이 예상된다.

---

## 8. 본 연구와의 연결

Rossi & Barbaro (2022)는 sparse vMF mixture의 중요한 baseline이다.

그러나 본 연구의 문제의식에서는 다음 한계가 있다.

1. **Penalty 대상**
   - Rossi & Barbaro: `mu_k`
   - 본 연구 제안: `eta_k = kappa_k mu_k`

2. **판별항 정합성**
   - posterior log-odds의 선형 판별항은 `(eta_h - eta_l)^T x_i`.
   - 따라서 `eta`에 대한 penalty가 clustering-oriented screening에 더 직접 정합된다.

3. **최종 추정량**
   - Rossi & Barbaro: penalized estimator를 최종 사용.
   - 본 연구 제안: screening 후 unpenalized refit.

4. **같은 평균, 다른 concentration 시나리오**
   - `mu_A = mu_B`, `kappa_A != kappa_B`이면 `eta_A != eta_B`.
   - `mu` penalty는 이 차이를 직접 반영하지 못할 수 있다.
   - `eta` penalty는 이를 자연스럽게 반영한다.

---

## 9. 현재 결론

이번 medium-budget reproduction 결과, Rossi & Barbaro (2022)의 핵심 artificial simulation 패턴은 재현되었다.

요약하면:

1. `N=1000`에서는 BIC가 true `K=4`를 안정적으로 회복한다.
2. sparse vMF는 dense vMF보다 ARI가 같거나 약간 높다.
3. sparse penalty는 effective nonzero dimension을 줄이며, 특히 `N=200`에서 K 선택을 개선한다.
4. variable recovery에서는 recall이 높고 precision은 상대적으로 낮다.
5. 이 baseline은 본 연구의 `eta` penalty 및 two-stage refit 필요성을 설명하는 출발점으로 적절하다.

---

## 10. 다음 작업

1. **논문 full replication 선택**
   - 필요하면 `n_rep = 100`, `nstart = 10`으로 overnight run.
   - 현재 결과는 research meeting용 medium-budget reproduction으로 충분.

2. **제안 모형 시뮬레이션 설계**
   - Main: `eta_h = kappa_h mu_h` cluster-contrast penalty.
   - Sub: `mu` penalty + `kappa` penalty.
   - Baseline: Rossi & Barbaro sparse vMF.

3. **핵심 비교 시나리오**
   - `mu` 다름, `kappa` 같음.
   - `mu` 같음, `kappa` 다름.
   - noise coordinate 다수.
   - two-stage refit 유무.

4. **보고 지표**
   - ARI, NMI.
   - active set F1.
   - selected nonzero fraction.
   - refit 전후 ARI 및 parameter recovery.
