## 1. 기본 메커니즘 시뮬레이션

제안 방법의 작동 원리를 확인하기 위해 $K=2$의 단순한 환경에서 6가지 방법을 비교했다. 이전 결과에서는 일부 방법의 tuning 방식이 서로 달랐으므로, 아래 환경에서는 가능한 한 동일한 원칙으로 path 기반 tuning을 적용했다.

* **Tuning:** path 기반 후보를 생성하고, 각 path 위에서 BIC가 최소인 지점을 선택.
* **Rossi:** $\beta$ path를 따라 적합하고 BIC가 최소인 지점을 선택.
* **분리 패널티:** $\lambda_\kappa$는 데이터 기반 2D grid로 두고, 각 $\lambda_\kappa$에서 $\lambda_\mu$ path를 따라 적합한 뒤 전체 후보 중 BIC 최소 지점을 선택.
* **에타 패널티:** $\lambda_\eta$ path를 따라 적합하고 BIC가 최소인 지점을 선택.
* **Refit:** 각 penalized fit에서 선택된 coordinate support를 고정한 뒤 penalty 없이 vMF mixture를 다시 추정.
* **Simulation:** 반복 수 20회, random start 5회.

### 1.1. 집중도 주도 환경

* **Setting:** $K=2$, $n=1000$.
* **Variables:** 전체 변수 개수는 $d=100$개, 유효 변수 개수는 $q=10$개.
* **True Params:** $\mu_1 = \mu_2$, $\kappa_1=20, \kappa_2=200$. (True $\kappa$ ratio = 10, True $\|\eta_2-\eta_1\| = 180$)

**a. 군집화 및 변수 선택 성능**

| Method | ARI | Selected $q$ | TPR | FPR | Precision | F1 |
|:---|---:|---:|---:|---:|---:|---:|
| Rossi path BIC | 1.000 | 23.300 | 1.000 | 0.148 | 0.443 | 0.610 |
| Rossi path BIC + refit | 1.000 | 23.300 | 1.000 | 0.148 | 0.443 | 0.610 |
| Separate 2D path/grid BIC | 1.000 | 23.300 | 1.000 | 0.148 | 0.443 | 0.610 |
| Separate 2D path/grid BIC + refit | 1.000 | 23.300 | 1.000 | 0.148 | 0.443 | 0.610 |
| **Eta path BIC** | 1.000 | **13.200** | 1.000 | **0.036** | **0.792** | **0.875** |
| **Eta path BIC + refit** | 1.000 | **13.200** | 1.000 | **0.036** | **0.792** | **0.875** |

**b. 모수 추정 성능** (MSE 지표 $\times 100$)

| Method | MSE_mu | MSE_kappa | MSE_Delta_eta | kappa ratio | $\|\eta_2-\eta_1\|$ |
|:---|---:|---:|---:|---:|---:|
| Rossi path BIC | 0.0176 | 127.629 | 24.534 | 10.062 | 181.179 |
| Rossi path BIC + refit | 0.0061 | 140.965 | 37.797 | 9.951 | 180.821 |
| Separate 2D path/grid BIC | 0.0176 | 127.629 | 24.534 | 10.062 | 181.179 |
| Separate 2D path/grid BIC + refit | 0.0061 | 140.965 | 37.797 | 9.951 | 180.821 |
| Eta path BIC | 0.0180 | 741.495 | 29.167 | 8.559 | 175.542 |
| **Eta path BIC + refit** | **0.0041** | **118.451** | **21.585** | **9.960** | **180.630** |

---

### 1.2. 집중도 차이가 약한 환경

* **Setting:** $K=2$, $n=1000$.
* **Variables:** 전체 변수 개수는 $d=100$개, 유효 변수 개수는 $q=10$개.
* **True Params:** $\mu_1 = \mu_2$, $\kappa_1=20, \kappa_2=40$. (True $\kappa$ ratio = 2, True $\|\eta_2-\eta_1\| = 20$)

**a. 군집화 및 변수 선택 성능**

| Method | ARI | Selected $q$ | TPR | FPR | Precision | F1 |
|:---|---:|---:|---:|---:|---:|---:|
| Rossi path BIC | 0.345 | 57.650 | 1.000 | 0.529 | 0.238 | 0.355 |
| Rossi path BIC + refit | 0.322 | 57.650 | 1.000 | 0.529 | 0.238 | 0.355 |
| Separate 2D path/grid BIC | 0.332 | 57.000 | 1.000 | 0.522 | 0.239 | 0.355 |
| Separate 2D path/grid BIC + refit | 0.315 | 57.000 | 1.000 | 0.522 | 0.239 | 0.355 |
| **Eta path BIC** | 0.333 | **21.700** | 1.000 | **0.130** | **0.740** | **0.812** |
| **Eta path BIC + refit** | 0.343 | **21.700** | 1.000 | **0.130** | **0.740** | **0.812** |

**b. 모수 추정 성능** (MSE 지표 $\times 100$)

| Method | MSE_mu | MSE_kappa | MSE_Delta_eta | kappa ratio | $\|\eta_2-\eta_1\|$ |
|:---|---:|---:|---:|---:|---:|
| Rossi path BIC | 0.0140 | 211.689 | 30.690 | 1.982 | 20.776 |
| Rossi path BIC + refit | 0.0364 | 198.151 | 78.808 | 1.938 | 21.622 |
| Separate 2D path/grid BIC | 0.0162 | 462.257 | 29.889 | 2.027 | 20.906 |
| Separate 2D path/grid BIC + refit | 0.0405 | 244.391 | 79.549 | 1.943 | 21.657 |
| Eta path BIC | 0.0215 | 800.937 | 45.011 | 1.678 | 16.214 |
| **Eta path BIC + refit** | **0.0154** | **185.900** | **34.919** | **1.966** | **20.636** |

---

### 1.3. 평균과 집중도 차이가 모두 존재하는 환경

* **Setting:** $K=2$, $n=1000$.
* **Variables:** 전체 변수 개수는 $d=100$개, 유효 변수 개수는 $q=10$개.
* **True Params:** $\mu_{cos} = 0.95$, $\kappa_1=20, \kappa_2=100$. (True $\kappa$ ratio = 5, True $\|\eta_2-\eta_1\| = 81.240$)

**a. 군집화 및 변수 선택 성능**

| Method | ARI | Selected $q$ | TPR | FPR | Precision | F1 |
|:---|---:|---:|---:|---:|---:|---:|
| Rossi path BIC | 0.993 | 13.650 | 1.000 | 0.041 | 0.759 | 0.856 |
| Rossi path BIC + refit | 0.993 | 13.650 | 1.000 | 0.041 | 0.759 | 0.856 |
| Separate 2D path/grid BIC | 0.993 | 13.550 | 1.000 | 0.039 | 0.764 | 0.859 |
| Separate 2D path/grid BIC + refit | 0.993 | 13.550 | 1.000 | 0.039 | 0.764 | 0.859 |
| **Eta path BIC** | 0.993 | **13.300** | 1.000 | **0.037** | **0.783** | **0.871** |
| **Eta path BIC + refit** | 0.993 | **13.300** | 1.000 | **0.037** | **0.783** | **0.871** |

**b. 모수 추정 성능** (MSE 지표 $\times 100$)

| Method | MSE_mu | MSE_kappa | MSE_Delta_eta | kappa ratio | $\|\eta_2-\eta_1\|$ |
|:---|---:|---:|---:|---:|---:|
| Rossi path BIC | 0.0125 | 52.145 | 15.515 | 5.042 | 81.611 |
| Rossi path BIC + refit | 0.0037 | 51.758 | 14.554 | 5.018 | 81.370 |
| Separate 2D path/grid BIC | 0.0126 | 52.180 | 15.578 | 5.042 | 81.614 |
| Separate 2D path/grid BIC + refit | 0.0037 | 51.867 | 14.423 | 5.018 | 81.368 |
| Eta path BIC | 0.0144 | 589.450 | 27.340 | 4.347 | 77.159 |
| **Eta path BIC + refit** | 0.0046 | **51.698** | 16.627 | 5.011 | 81.368 |

---

### 1.4. 해석

* 공정한 path tuning 기준에서도 $\eta$ penalty는 concentration-driven 환경에서 selected $q$를 true $q=10$에 가장 가깝게 유지하고, FPR과 F1을 가장 크게 개선한다.
* 집중도 차이가 약한 경우에는 ARI가 전체적으로 낮다. 이는 평균 방향이 같고 concentration 차이도 작아 군집화 자체가 어려운 상황이기 때문이다. 그럼에도 $\eta$ penalty는 변수 선택에서 Rossi와 분리 패널티보다 훨씬 안정적이다.
* 평균 방향 차이와 집중도 차이가 모두 존재하는 경우에는 세 방법 모두 잘 작동하지만, $\eta$ penalty가 가장 작은 selected $q$와 가장 높은 F1을 보인다.
* 분리 패널티의 2D path/grid-BIC는 대부분 $\lambda_\kappa=0$ 또는 매우 작은 값을 선택했다. 따라서 실제 선택 결과는 $\mu$ penalty 중심으로 작동하며, concentration contrast를 직접 겨냥하는 데에는 한계가 있다.
* Penalized fit에서 $\eta$ penalty는 $\kappa$를 shrink하므로 refit 전 MSE_kappa가 커질 수 있다. 선택된 support로 penalty 없이 refit하면 kappa ratio와 $\|\eta_2-\eta_1\|$가 true 값에 가까워진다.

---

## 2. 2022년 논문 시뮬레이션 세팅 재현

### 2.1. 논문 기준 corrected reproduction

* **목적:** Rossi & Barbaro (2022)의 artificial simulation setting에서 Rossi sparse vMF가 논문 Figure와 유사하게 재현되는지 확인.
* **Setting:** $K=4$ 고정, 군집 비율 균일, $n=1000$, $d=100$, overlap $=0.05$.
* **Concentration:** 논문 기준 $d=100$, overlap $=0.05$에서 base $\kappa=15.09$를 사용하고, component별 $\kappa_k$는 $N(\kappa, 0.025\kappa)$에서 생성한 뒤 평균 방향 간 분리에 맞게 조정했다.
* **Sparsity 정의:** 논문은 directional mean의 일부 좌표를 0으로 만드는 방식으로 sparsity를 설정한다. 따라서 논문 sparsity $=0.10$은 zero coordinate가 10%라는 뜻이고, 코드에서는 `nonzero_fraction = 0.90`으로 실행해야 한다.
* **Variables:** component별 zero coordinate는 10개, nonzero coordinate는 90개. Entry-level 기준 true nonzero는 $4 \times 90 = 360$개, true zero는 $4 \times 10 = 40$개.
* **Simulation:** 반복 수 100회, random start 10회, Rossi beta path 최대 700 steps.
* **Tuning:** Rossi $\beta$ path에서 BIC 기준으로 penalty parameter 선택. Corrected run의 평균 $\beta=72.673$.

논문 결과는 exact table이 아니라 Figure 13, Figure 15, Figure 16으로 제시되어 있다. 아래 논문 값은 $K=4$, $n=1000$, overlap $=0.05$, sparsity $=0.10$, BIC panel에서 읽은 근사 범위다.

| Metric | 논문 Figure 근사값 | Corrected reproduction |
|:---|---:|---:|
| ARI | 약 0.80-0.90 | 0.871 |
| achieved sparsity | 약 0.30-0.40 | 0.347 |
| zero precision | 약 0.25-0.35 | 0.265 |
| zero recall | 약 0.85-1.00 | 0.905 |
| nonzero fraction | - | 0.653 |

해석상 중요한 점은 corrected reproduction이 논문 Figure의 정성적 패턴과 같은 범위에 있다는 것이다. BIC는 zero recall을 높게 유지하지만 zero precision은 낮은 편이고, achieved sparsity도 true sparsity 0.10보다 크게 선택된다. 이는 논문에서 설명한 “BIC가 recall은 좋지만 precision 손실을 감수하며 더 sparse한 표현을 선택하는 경향”과 일치한다.

기존에 사용한 `nonzero_fraction = 0.10` 세팅은 component별 nonzero coordinate가 10개인 경우다. 이 세팅은 우리 연구의 sparse active-variable 상황을 보기에는 유용하지만, 논문 Figure와 직접 비교할 때는 논문 sparsity 정의와 반대이므로 corrected reproduction과 구분해서 해석해야 한다.

### 2.2. 추가 비교: nonzero 10% 기준 6가지 방법

아래 2.2-2.4의 6가지 방법 비교는 우리 연구에서 먼저 살펴본 sparse active-variable 세팅이다. 전체 변수는 $d=100$개이고, component별 nonzero coordinate는 10개이며, union 기준 유효 변수 개수는 반복 평균 34.1개다. 반복 수는 20회, random start는 5회다.

* **Tuning:** 1절과 같은 path tuning 기준으로 맞췄다. Rossi는 $\beta$ path에서 BIC를 최소화하는 지점을 선택하고, 분리 패널티는 $\lambda_\kappa$ grid와 $\lambda_\mu$ path의 2D 후보 중 BIC 최소 지점을 선택한다. 에타 패널티는 centered $\eta$ norm의 $\lambda_\eta$ path에서 BIC 최소 지점을 선택한다.

아래 표는 coordinate union 기준이다. 즉 하나의 coordinate가 어느 component에서든 선택되면 selected coordinate로 계산했다.

| Method | ARI | True union $q$ | Selected $q$ | TPR | FPR | Precision | F1 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Rossi path BIC | 0.903 | 34.100 | 33.550 | 0.904 | 0.041 | 0.921 | **0.910** |
| Rossi path BIC + refit | 0.902 | 34.100 | 33.550 | 0.904 | 0.041 | 0.921 | **0.910** |
| Separate 2D path/grid BIC | 0.903 | 34.100 | 34.150 | 0.906 | 0.049 | 0.909 | 0.904 |
| Separate 2D path/grid BIC + refit | 0.902 | 34.100 | 34.150 | 0.906 | 0.049 | 0.909 | 0.904 |
| Eta centered path BIC | 0.899 | 34.100 | **32.150** | 0.878 | **0.034** | **0.933** | 0.903 |
| Eta centered path BIC + refit | 0.902 | 34.100 | **32.150** | 0.878 | **0.034** | **0.933** | 0.903 |

### 2.3. 모형 적합 지표

BIC는 공식 tuning 선택 기준이고, EBIC는 보조 적합 지표로 계산했다. 두 값 모두 낮을수록 좋다.

| Method | loglik | df | BIC | EBIC |
|:---|---:|---:|---:|---:|
| **Rossi path BIC** | 90291.304 | 41.950 | **-180292.829** | **-180099.642** |
| Rossi path BIC + refit | 90380.602 | 137.200 | -179813.459 | -179181.630 |
| Separate 2D path/grid BIC | 90292.884 | 42.550 | -180291.843 | -180095.893 |
| Separate 2D path/grid BIC + refit | 90383.009 | 139.600 | -179801.695 | -179158.813 |
| Eta centered path BIC | 90300.723 | 199.450 | -179223.695 | -178305.194 |
| Eta centered path BIC + refit | 90372.555 | 131.600 | -179836.050 | -179230.010 |

### 2.4. 모수 추정 성능

MSE 지표는 기존 표와 같이 $\times 100$으로 표시했다.

| Method | MSE_mu | MSE_kappa | MSE_centered_eta | kappa_hat_mean |
|:---|---:|---:|---:|---:|
| Rossi path BIC | **0.013** | **58.688** | 10.881 | 33.680 |
| Rossi path BIC + refit | 0.017 | 78.007 | 14.104 | 34.062 |
| Separate 2D path/grid BIC | **0.013** | 58.905 | **10.744** | 33.686 |
| Separate 2D path/grid BIC + refit | 0.017 | 78.758 | 14.569 | 34.070 |
| Eta centered path BIC | 0.035 | 989.428 | 31.849 | 30.603 |
| Eta centered path BIC + refit | 0.017 | 73.426 | 14.993 | 34.030 |

### 2.5. 핵심 해석

* 실제 논문 Figure와 비교할 때는 sparsity를 zero coordinate 비율로 해석해야 한다. 이 기준에서는 corrected reproduction 결과가 논문 Figure 13, 15, 16의 패턴과 유사하다.
* 논문 기준 corrected reproduction에서는 BIC가 true sparsity 0.10보다 더 sparse한 표현을 선택한다. 이때 zero recall은 높고 zero precision은 낮아지는 경향이 나타나며, 이는 논문 설명과 일치한다.
* 2.2-2.4의 6-method 비교는 component별 nonzero coordinate가 10개인 별도 sparse active-variable 세팅이다. 이 세팅에서는 $\mu_k$ sparsity가 true structure이므로 Rossi와 분리 패널티가 BIC 선택 결과에서 좋은 성능을 보이는 것이 자연스럽다.
* 에타 패널티는 selected $q$를 더 작게 만들고 FPR을 가장 낮추지만, true structure가 $\mu_k$ sparsity인 상황에서는 TPR이 다소 낮아진다.
* 따라서 논문 재현은 Rossi 방법의 정상 작동을 확인하는 기준점으로 두고, 제안 방법의 필요성은 이후 concentration-driven 한계 세팅에서 보여주는 것이 적절하다.

## 3. K=4 stress setting: 평균 방향 동일, 집중도 차이

2022 논문과 같은 기본 계산 환경을 유지하되, 군집 차이가 평균 방향이 아니라 집중도에서만 생기도록 stress setting을 구성했다.

### 3.1. Simulation setting

* **Setting:** $K=4$, $n=1000$, 군집 비율 균일.
* **Variables:** 전체 변수 개수는 $d=100$개, 유효 변수 개수는 $q=10$개.
* **Simulation:** 반복 수 20회, random start 5회.
* **Tuning:** Rossi는 $\beta$ path, 분리 패널티는 $\lambda_\kappa$ grid와 $\lambda_\mu$ path, 에타 패널티는 centered $\eta$ norm의 $\lambda_\eta$ path에서 BIC 기준으로 선택.
* **Mean direction:** $\mu_1=\mu_2=\mu_3=\mu_4$.
* **Sparsity:** 공통 평균 방향은 유효 변수 10개에서만 nonzero.
* **Concentration:** $\kappa=(20,35,60,100)$.

이 세팅에서는 평균 방향이 완전히 같으므로, 군집 차이는 posterior decision에 들어가는 $\eta_k=\kappa_k\mu_k$의 크기 차이에서만 발생한다. 2022 논문에서 사용하는 overlap 조정식은 평균 방향 간 분리를 전제로 하므로 여기서는 사용하지 않고, $\kappa$를 직접 지정했다.

### 3.2. 군집화 및 변수 선택 성능

| Method | ARI | Selected $q$ | TPR | FPR | Precision | F1 |
|:---|---:|---:|---:|---:|---:|---:|
| Rossi path BIC | 0.432 | 85.650 | 1.000 | 0.841 | 0.128 | 0.223 |
| Rossi path BIC + refit | 0.420 | 85.650 | 1.000 | 0.841 | 0.128 | 0.223 |
| Separate 2D path/grid BIC | 0.429 | 92.550 | 1.000 | 0.917 | 0.109 | 0.196 |
| Separate 2D path/grid BIC + refit | 0.412 | 92.550 | 1.000 | 0.917 | 0.109 | 0.196 |
| Eta centered path BIC | 0.380 | **27.950** | 1.000 | **0.199** | **0.553** | **0.661** |
| **Eta centered path BIC + refit** | **0.440** | **27.950** | 1.000 | **0.199** | **0.553** | **0.661** |

### 3.3. 모형 적합 지표

BIC는 공식 tuning 선택 기준이고, EBIC는 보조 적합 지표로 계산했다. 두 값 모두 낮을수록 좋다.

| Method | loglik | df | BIC | EBIC |
|:---|---:|---:|---:|---:|
| Rossi path BIC | 96895.361 | 191.800 | -192465.815 | -191582.543 |
| Rossi path BIC + refit | 96983.712 | 345.600 | -191580.104 | -189988.557 |
| Separate 2D path/grid BIC | 96910.965 | 207.200 | -192390.644 | -191436.452 |
| Separate 2D path/grid BIC + refit | 96996.497 | 373.200 | -191415.019 | -189696.369 |
| Eta centered path BIC | 96758.566 | 186.850 | -192226.417 | -191365.941 |
| **Eta centered path BIC + refit** | 96846.197 | 114.800 | **-192899.383** | **-192370.710** |

### 3.4. 모수 추정 성능

이 세팅에서는 $\mu_k$가 모두 같으므로, label matching에 따른 모수 MSE 해석에는 주의가 필요하다. 따라서 아래 MSE는 raw scale로 표시하고 보조 지표로만 본다.

| Method | MSE_mu | MSE_kappa | MSE_centered_eta | kappa_hat_mean |
|:---|---:|---:|---:|---:|
| Rossi path BIC | 0.001 | 1858.207 | 19.501 | 58.147 |
| Rossi path BIC + refit | 0.001 | 1796.361 | 20.324 | 59.484 |
| Separate 2D path/grid BIC | 0.001 | 1861.088 | 19.653 | 58.379 |
| Separate 2D path/grid BIC + refit | 0.001 | 1757.400 | 20.433 | 58.947 |
| Eta centered path BIC | **0.000** | 1763.004 | **17.652** | 61.474 |
| Eta centered path BIC + refit | 0.001 | 2091.288 | 21.816 | 59.807 |

### 3.5. 해석

* 평균 방향이 같고 집중도만 다른 경우에는 Rossi와 분리 패널티가 대부분의 변수를 선택한다. path tuning 후에도 Rossi의 FPR은 0.841, 분리 패널티의 FPR은 0.917로 높다.
* 에타 패널티는 selected $q=27.950$, FPR 0.199, Precision 0.553, F1 0.661로 변수 선택 성능을 크게 개선한다.
* 모형 적합 지표 표에서는 에타 패널티 + refit이 가장 좋다.
* ARI는 전체적으로 낮다. 이는 $K=4$에서 평균 방향이 완전히 같고 집중도만 다른 상황 자체가 군집화하기 어려운 stress setting임을 의미한다.
* 모수 MSE는 기존 코드가 $\mu$ cosine 기준으로 label matching을 수행하므로 보조 지표로만 본다. 이 결과에서는 ARI, 변수 선택 지표, BIC와 보조 EBIC를 우선적으로 해석한다.

## 4. K=4 realistic concentration-dominant setting

Stress setting은 평균 방향을 완전히 같게 둔 강한 한계 상황이다. 실제 자료에서는 군집 평균 방향이 완전히 같기보다는 상당 부분을 공유하면서 일부 좌표에서 차이가 있고, 동시에 집중도 차이도 존재할 가능성이 높다. 이를 반영하여 다음과 같은 실제 유사 환경을 구성했다.

### 4.1. Simulation setting

* **Setting:** $K=4$, $n=1000$, 군집 비율 균일.
* **Simulation:** 반복 수 20회, random start 5회.
* **Tuning:** Rossi는 $\beta$ path, 분리 패널티는 $\lambda_\kappa$ grid와 $\lambda_\mu$ path, 에타 패널티는 centered $\eta$ norm의 $\lambda_\eta$ path에서 BIC 기준으로 선택.
* **Mean direction:** 평균 방향은 완전히 같지 않지만 매우 유사하게 설정. Pairwise cosine은 약 0.90 수준.
* **Variables:** 전체 변수 개수는 $d=100$개, union 기준 유효 변수 개수는 19개.
  * Component별 active coordinate는 10개.
  * 모든 component가 공유하는 공통 active coordinate는 7개.
  * Component별 specific active coordinate는 3개씩.
  * Entry-level active 개수는 $4\times10=40$개.
* **Concentration:** $\kappa=(25,40,65,100)$.

이 세팅은 Rossi가 완전히 불리한 stress setting이 아니라, 평균 방향 차이도 일부 존재하여 Rossi가 작동할 여지를 남긴 상태에서 concentration 차이의 영향을 함께 보는 중간 난이도 설정이다.

### 4.2. 군집화 및 변수 선택 성능

| Method | ARI | True union $q$ | Selected $q$ | TPR | FPR | Precision | F1 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Rossi path BIC | 0.640 | 19.000 | 48.850 | 1.000 | 0.369 | 0.415 | 0.580 |
| Rossi path BIC + refit | 0.626 | 19.000 | 48.850 | 1.000 | 0.369 | 0.415 | 0.580 |
| Separate 2D path/grid BIC | **0.641** | 19.000 | 67.700 | 1.000 | 0.601 | 0.303 | 0.458 |
| Separate 2D path/grid BIC + refit | 0.615 | 19.000 | 67.700 | 1.000 | 0.601 | 0.303 | 0.458 |
| Eta centered path BIC | 0.567 | 19.000 | **37.200** | 0.968 | **0.232** | **0.565** | **0.692** |
| Eta centered path BIC + refit | 0.609 | 19.000 | **37.200** | 0.968 | **0.232** | **0.565** | **0.692** |

### 4.3. 모형 적합 지표

BIC는 공식 tuning 선택 기준이고, EBIC는 보조 적합 지표로 계산했다. 두 값 모두 낮을수록 좋다.

| Method | loglik | df | BIC | EBIC |
|:---|---:|---:|---:|---:|
| **Rossi path BIC** | 97548.322 | 80.200 | **-194542.643** | **-194173.308** |
| Rossi path BIC + refit | 97646.273 | 198.400 | -193922.048 | -193008.382 |
| Separate 2D path/grid BIC | 97583.758 | 118.600 | -194348.256 | -193802.083 |
| Separate 2D path/grid BIC + refit | 97681.027 | 273.800 | -193470.711 | -192209.816 |
| Eta centered path BIC | 97522.296 | 214.600 | -193562.188 | -192573.919 |
| Eta centered path BIC + refit | 97605.510 | 151.800 | -194162.422 | -193463.357 |

### 4.4. 모수 추정 성능

MSE 지표는 기존 표와 같이 $\times 100$으로 표시했다.

| Method | MSE_mu | MSE_kappa | MSE_centered_eta | kappa_hat_mean |
|:---|---:|---:|---:|---:|
| Rossi path BIC | 0.019 | **238.004** | **18.605** | 57.822 |
| Rossi path BIC + refit | 0.026 | 324.928 | 41.337 | 58.239 |
| Separate 2D path/grid BIC | **0.015** | 265.754 | 18.641 | 57.665 |
| Separate 2D path/grid BIC + refit | 0.035 | 340.097 | 53.145 | 58.440 |
| Eta centered path BIC | 0.053 | 9954.189 | 125.900 | 62.681 |
| Eta centered path BIC + refit | 0.046 | 6773.728 | 126.210 | 60.987 |

### 4.5. 해석

* 실제 유사 환경에서는 Rossi가 BIC 기반 모형 적합에서 가장 좋고, ARI는 Rossi와 분리 패널티가 비슷하다. 이는 평균 방향 차이가 일부 존재하므로 $\mu_k$ sparsity가 여전히 유효하기 때문이다.
* 그러나 Rossi는 selected $q=48.850$으로 true union $q=19$보다 많은 변수를 선택하고, FPR이 0.369로 남아 있다.
* 에타 패널티 + refit은 ARI가 0.609로 Rossi + refit 0.626보다 낮지만, selected $q=37.200$, FPR 0.232, F1 0.692로 변수 선택 성능이 더 좋다.
* 따라서 이 세팅의 핵심 메시지는 에타 방법이 clustering 자체를 압도한다는 것이 아니라, 군집 성능을 크게 잃지 않으면서 더 sparse하고 해석 가능한 변수 선택을 제공한다는 점이다.
* 현실적인 중간 세팅에서는 Rossi가 모형 적합 기준에서 강하고, 에타 패널티는 해석 가능한 변수 선택 기준에서 강하다. 따라서 제안 방법의 장점은 concentration-driven effect를 반영한 sparse interpretation으로 정리하는 것이 적절하다.

## 5. 요약 결론

* **기본 메커니즘:** $K=2$ 환경에서는 평균 방향이 같고 집중도 차이가 군집을 만드는 경우, $\eta$-penalty가 Rossi 및 분리 패널티보다 FPR을 크게 낮추고 F1을 개선한다.
* **Refit 역할:** $\eta$-penalty 단독은 $\kappa$ 수축 편향이 생길 수 있으나, 선택된 support를 고정한 refit을 수행하면 $\kappa$ ratio와 $\eta$ contrast가 true value에 가깝게 복원된다.
* **논문 재현:** Rossi & Barbaro (2022)의 sparse vMF는 논문 Figure와 유사하게 재현된다. 특히 논문 기준 sparsity는 zero coordinate 비율로 해석해야 한다.
* **K=4 stress setting:** 평균 방향이 같고 집중도만 다른 어려운 상황에서는 path tuning 후에도 Rossi와 분리 패널티가 불필요한 변수를 많이 선택한다. $\eta$-penalty는 FPR을 낮추고 해석 가능한 변수 선택을 제공한다.
* **현실적 중간 setting:** 평균 차이도 일부 존재하는 경우 Rossi가 BIC 기반 모형 적합에서 강하지만, $\eta$-penalty + refit은 ARI 손실을 감수하는 대신 더 sparse한 변수 선택을 제공한다.

---

## 부록. $\eta$-Penalty 모형의 수리적 타당성

**① 베이즈 결정 경계 직접 수축**
$$\log \frac{\tau_{i2}}{\tau_{i1}} = \text{Const} + (\eta_2 - \eta_1)^\top x_i$$
* 사후 확률을 결정하는 실질적 선형 판별 계수는 $\mu$가 아닌 $\eta$의 대조임.
* $-\lambda_\eta \|\eta_2 - \eta_1\|_1$ 패널티는 노이즈 차원의 판별 계수를 직접 0으로 강제하여 정확한 변수 선택을 수행함.

**② 집중도 주도 환경 식별**
* **조건:** $\mu_1 = \mu_2$, $\kappa_1 \ll \kappa_2$ 
* **$\mu$-penalty (기존)**: $\|\mu_2 - \mu_1\| = 0 \rightarrow$ 군집 식별 불가, False Positive 증가.
* **$\eta$-penalty (제안)**: $\|\eta_2 - \eta_1\| = \kappa_2 - \kappa_1 \neq 0 \rightarrow$ 평균 방향이 동일하더라도 집중도 차이로 발생하는 좌표별(Coordinate-level) 분리 효과를 완벽히 포착.

**③ 내재적 정규화**
$$\|\eta_k\|_2 = \|\kappa_k \mu_k\|_2 = \kappa_k$$
* 자연 모수의 $L_2$ 노름이 곧 집중도($\kappa$)이므로, $\eta$ 벡터에 대한 $L_1$ 패널티는 필연적으로 $\kappa$ 스케일의 수축을 유도함.
* 고차원 모형의 고질적 한계인 $\kappa_k \rightarrow \infty$ (소수 관측치 과적합 발산) 현상을 인위적인 제약(`shared kappa`) 없이 수리적으로 원천 차단.

**④ 수축 편향 제거**

$$\hat{S}_\eta = \{j : |\hat{\eta}_{2j} - \hat{\eta}_{1j}| > 0 \}$$
$$\text{Refit constraint: } \mu_{kj} = 0 \text{ for } j \notin \hat{S}_\eta$$
* Phase 1에서 도출된 Support($\hat{S}_\eta$)를 고정한 채, 패널티 없이 재학습(Unpenalized EM)하는 Relaxed LASSO 구조 도입.
* $L_1$ 패널티로 인해 축소된 $\kappa$ 추정치와 $\eta$ Contrast를 True Value 스케일로 복원하여 FPR 통제와 모수 추정 정확도를 동시에 달성.
