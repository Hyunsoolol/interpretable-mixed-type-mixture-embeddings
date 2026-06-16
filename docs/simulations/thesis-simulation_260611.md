# Thesis Simulation Archive 260611

업데이트: 2026-06-11

이 문서는 260622 연구미팅 자료에서 사용한 시뮬레이션 결과를 전체 형태로 보관한 archive다. 연구미팅용 핵심 요약은 `thesis-meeting_260622.md`, 추정 방법과 코드 구현 설명은 `thesis-implementation_260622.md`에 정리했다.

## 0. 읽는 방법

### 0.1. 공식 비교 기준

| 항목 | 기준 |
|:---|:---|
| 비교 방법 | Rossi, Rossi + refit, 분리 패널티, 분리 패널티 + refit, 에타 패널티, 에타 패널티 + refit |
| tuning 후보 | path 기반 후보 생성 |
| 공식 tuning 선택 | BIC 최소 지점 |
| EBIC | 고차원 setting의 보조 지표 |
| refit | 선택된 coordinate support를 고정하고 penalty 없이 vMF mixture 재추정 |
| 기본 해석 순서 | ARI보다 variable selection 지표를 함께 해석 |

### 0.2. 지표 정의

| 지표 | 의미 | 좋게 보는 방향 |
|:---|:---|:---|
| ARI | 추정 군집과 true label의 일치도 | 클수록 좋음 |
| Selected q | 선택된 coordinate 수 | true union q에 가까울수록 좋음 |
| TPR | true active coordinate 중 선택된 비율 | 클수록 좋음 |
| FPR | noise coordinate 중 잘못 선택된 비율 | 작을수록 좋음 |
| Precision | 선택된 coordinate 중 true active 비율 | 클수록 좋음 |
| F1 | TPR과 Precision의 조화평균 | 클수록 좋음 |
| BIC, EBIC | 모형 적합 정보기준 | 작을수록 좋음 |
| MSE_mu | 평균 방향 추정 오차 | 작을수록 좋음 |
| MSE_kappa | 집중도 추정 오차 | 작을수록 좋음 |
| MSE_centered_eta | centered eta 기준 추정 오차 | 작을수록 좋음 |

ARI는 군집 라벨이 맞았는지만 본다. 따라서 Rossi의 ARI가 높아도 Selected q와 FPR이 매우 크면 sparse variable selection에는 실패한 것으로 해석한다. 이번 연구의 핵심은 군집화 성능을 유지하면서 불필요한 노이즈 변수를 줄이는 것이다.

### 0.3. 시뮬레이션 구조 요약

| 번호 | setting | 목적 | 전체 변수 d | true union q | 반복수 |
|:---|:---|:---|---:|---:|---:|
| 1 | K=2 기본 메커니즘 | concentration-dominant 상황에서 eta penalty의 기본 작동 확인 | 100 | 10 | 20 |
| 2.1 | Rossi 2022 corrected reproduction | Rossi 원 논문 setting 재현 | 100 | 반복별 상이 | 100 |
| 2.2 | K=4 sparse-active 비교 | Rossi가 유리한 sparse direction setting에서 6개 방법 비교 | 100 | 평균 34.1 | 20 |
| 3 | K=4 stress setting | 평균 방향 동일, 집중도 차이만 존재하는 한계 setting | 100 | 10 | 20 |
| 4 | K=4 controlled concentration-dominant | 평균 방향 차이를 약간 추가한 controlled setting | 100 | 10 | 20 |
| 5 | 공통 변수 + 군집별 특정 변수 | 현실적인 sparse structure에서 변수 선택 확인 | 100 | 22 | 100 |
| 5.7 | 군집별 특정 변수 weight 변화 | specific signal 강도에 따른 robustness 확인 | 100 | 22 | 100 |
| 5.8 | 약한 집중도 차이 | concentration contrast가 약한 상황 확인 | 100 | 22 | 100 |
| 6.1 | 고차원, kappa 차원별 조정 | d 증가 시 eta penalty의 robustness 확인 | 100, 200, 500 | 22 | 20 |
| 6.2 | 고차원, kappa 고정 | 차원 증가로 signal이 약해지는 stress setting 확인 | 100, 200, 400 | 22 | 20 |

### 0.4. 전체 결론 요약

* Rossi 2022 방법은 원 논문 setting에서는 정상적으로 재현된다.
* Rossi가 유리한 sparse direction setting에서는 Rossi와 분리 패널티가 좋은 성능을 보인다.
* 평균 방향 차이가 작고 집중도 차이가 중요한 setting에서는 Rossi와 분리 패널티가 true active variable을 포함하면서도 노이즈 변수를 과도하게 선택한다.
* 에타 패널티 + refit은 selected q를 true union q에 가깝게 유지하고 FPR을 크게 낮춘다.
* 고차원에서는 BIC가 느슨할 수 있으므로 EBIC 또는 더 강한 tuning 기준을 추가 검토해야 한다.

## 1. 기본 메커니즘 시뮬레이션

제안 방법의 작동 원리를 확인하기 위해 $K=2$의 단순한 환경에서 6가지 방법을 비교했다. 이전 결과에서는 일부 방법의 tuning 방식이 서로 달랐으므로, 아래 환경에서는 가능한 한 동일한 원칙으로 path 기반 tuning을 적용했다.

**공통 시뮬레이션 환경**

```text
K = 2
n = 1000
반복수 = 20
random start = 5
d = 100
공통 변수 = 10개
군집별 특정 변수 = 없음
w = 해당 없음
max long path = Rossi 220 / Separate 220 / Eta 120
```

* **Tuning:** path 기반 후보를 생성하고, 각 path 위에서 BIC가 최소인 지점을 선택.
* **Rossi:** $\beta$ path를 따라 적합하고 BIC가 최소인 지점을 선택.
* **분리 패널티:** $\lambda_\kappa$는 데이터 기반 2D grid로 두고, 각 $\lambda_\kappa$에서 $\lambda_\mu$ path를 따라 적합한 뒤 전체 후보 중 BIC 최소 지점을 선택.
* **에타 패널티:** $\lambda_\eta$ path를 따라 적합하고 BIC가 최소인 지점을 선택.
* **Refit:** 각 penalized fit에서 선택된 coordinate support를 고정한 뒤 penalty 없이 vMF mixture를 다시 추정.

### 1.1. 집중도 주도 환경

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

**시뮬레이션 환경**

```text
K = 4
n = 1000
반복수 = 100
random start = 10
d = 100
공통 변수 = 명시적으로 분리하지 않음
군집별 특정 변수 = 명시적으로 분리하지 않음
w = 해당 없음
max long path = Rossi beta path 700
```

* **Concentration:** 논문 기준 $d=100$, overlap $=0.05$에서 base $\kappa=15.09$를 사용하고, component별 $\kappa_k$는 $N(\kappa, 0.025\kappa)$에서 생성한 뒤 평균 방향 간 분리에 맞게 조정했다.
* **Sparsity 정의:** 논문은 directional mean의 일부 좌표를 0으로 만드는 방식으로 sparsity를 설정한다. 따라서 논문 sparsity $=0.10$은 zero coordinate가 10%라는 뜻이고, 코드에서는 `nonzero_fraction = 0.90`으로 실행해야 한다.
* **Variables:** component별 zero coordinate는 10개, nonzero coordinate는 90개. Entry-level 기준 true nonzero는 $4 \times 90 = 360$개, true zero는 $4 \times 10 = 40$개.
* **Overlap:** 0.05.
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

아래 2.2-2.4의 6가지 방법 비교는 우리 연구에서 먼저 살펴본 sparse active-variable 세팅이다.

**시뮬레이션 환경**

```text
K = 4
n = 1000
반복수 = 20
random start = 5
d = 100
공통 변수 = 랜덤 support 구조라 고정하지 않음
군집별 특정 변수 = 랜덤 support 구조라 고정하지 않음
w = 해당 없음
max long path = Rossi 220 / Separate 300 / Eta 120
```

component별 nonzero coordinate는 10개이며, union 기준 유효 변수 개수는 반복 평균 34.1개다.

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

**시뮬레이션 환경**

```text
K = 4
n = 1000
반복수 = 20
random start = 5
d = 100
공통 변수 = 10개
군집별 특정 변수 = 없음
w = 해당 없음
max long path = Rossi 220 / Separate 300 / Eta 120
```

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

## 4. K=4 controlled concentration-dominant setting

Stress setting은 평균 방향을 완전히 같게 둔 강한 한계 상황이다. 여기서는 변수 구조는 3번 stress setting과 동일하게 고정하고, 평균 방향만 완전히 같지 않도록 완화했다. 따라서 active variable 구조의 차이는 제거하고, 평균 방향 차이가 약간 존재하는 상황에서 concentration 차이를 반영하는 방법이 어떻게 작동하는지 확인한다.

### 4.1. Simulation setting

**시뮬레이션 환경**

```text
K = 4
n = 1000
반복수 = 20
random start = 5
d = 100
공통 변수 = 10개
군집별 특정 변수 = 없음
w = 해당 없음
max long path = Rossi 220 / Separate 300 / Eta 120
```

* **Tuning:** Rossi는 $\beta$ path, 분리 패널티는 $\lambda_\kappa$ grid와 $\lambda_\mu$ path, 에타 패널티는 centered $\eta$ norm의 $\lambda_\eta$ path에서 BIC 기준으로 선택.
* **Mean direction:** 평균 방향은 완전히 같지 않지만 매우 유사하게 설정. Pairwise cosine은 0.95.
* **Variables:** 변수 구조는 3번 stress setting과 동일하게 둔다.
  * 모든 component가 같은 10개 active coordinate를 공유한다.
  * Entry-level active 개수는 $4\times10=40$개.
* **Concentration:** $\kappa=(25,40,65,100)$.

이 세팅은 평균 방향이 완전히 같은 3번보다 현실적이지만, 변수 구조는 동일하게 통제한 설정이다. 따라서 기존 4번처럼 component-specific active coordinate가 섞인 상황보다 해석이 단순하고, 모수 차이에 따른 방법별 작동을 더 직접적으로 비교할 수 있다.

### 4.2. 군집화 및 변수 선택 성능

| Method | ARI | True union $q$ | Selected $q$ | TPR | FPR | Precision | F1 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Rossi path BIC | 0.513 | 10.000 | 98.500 | 1.000 | 0.983 | 0.102 | 0.184 |
| Rossi path BIC + refit | 0.485 | 10.000 | 98.500 | 1.000 | 0.983 | 0.102 | 0.184 |
| Separate 2D path/grid BIC | **0.525** | 10.000 | 95.600 | 1.000 | 0.951 | 0.105 | 0.190 |
| Separate 2D path/grid BIC + refit | 0.486 | 10.000 | 95.600 | 1.000 | 0.951 | 0.105 | 0.190 |
| Eta centered path BIC | 0.489 | 10.000 | **28.800** | 1.000 | **0.209** | **0.443** | **0.586** |
| **Eta centered path BIC + refit** | 0.523 | 10.000 | **28.800** | 1.000 | **0.209** | **0.443** | **0.586** |

### 4.3. 모형 적합 지표

BIC는 공식 tuning 선택 기준이고, EBIC는 보조 적합 지표로 계산했다. 두 값 모두 낮을수록 좋다.

| Method | loglik | df | BIC | EBIC |
|:---|---:|---:|---:|---:|
| Rossi path BIC | 97790.511 | 267.300 | -193734.579 | -192503.617 |
| Rossi path BIC + refit | 97843.888 | 397.000 | -192945.398 | -191117.145 |
| Separate 2D path/grid BIC | 97759.380 | 221.850 | -193986.274 | -192964.617 |
| Separate 2D path/grid BIC + refit | 97837.013 | 385.400 | -193011.778 | -191236.945 |
| Eta centered path BIC | 97637.404 | 189.400 | -193966.480 | -193094.261 |
| **Eta centered path BIC + refit** | 97697.307 | 118.200 | **-194578.117** | **-194033.786** |

### 4.4. 모수 추정 성능

MSE 지표는 기존 표와 같이 $\times 100$으로 표시했다.

| Method | MSE_mu | MSE_kappa | MSE_centered_eta | kappa_hat_mean |
|:---|---:|---:|---:|---:|
| Rossi path BIC | 0.069 | **5075.147** | **113.572** | 59.792 |
| Rossi path BIC + refit | 0.117 | 8218.911 | 214.306 | 61.092 |
| Separate 2D path/grid BIC | 0.051 | 15923.643 | 224.683 | 58.360 |
| Separate 2D path/grid BIC + refit | 0.109 | 10013.210 | 216.211 | 60.722 |
| Eta centered path BIC | **0.032** | 17712.209 | 152.040 | 65.087 |
| Eta centered path BIC + refit | 0.039 | 13627.592 | 173.639 | 62.749 |

### 4.5. 해석

* 변수 구조를 3번과 동일하게 고정하고 평균 방향만 약간 다르게 두면, Rossi와 분리 패널티는 거의 모든 변수를 선택한다. Rossi + refit의 selected $q$는 98.500이고, 분리 패널티 + refit의 selected $q$는 95.600이다.
* 에타 패널티 + refit은 selected $q=28.800$, FPR 0.209, Precision 0.443, F1 0.586으로 변수 선택 성능을 크게 개선한다.
* Refit 기준 ARI도 에타 패널티가 0.523으로 Rossi + refit 0.485, 분리 패널티 + refit 0.486보다 높다.
* 모형 적합 지표 표에서도 에타 패널티 + refit이 가장 좋다. 이는 평균 방향 차이가 약간 존재하더라도 concentration-driven support를 직접 겨냥하는 방식이 유효할 수 있음을 보여준다.
* 이 controlled setting은 기존 heterogeneous-support setting보다 논문 연결에 더 적합하다. 변수 구조를 고정했기 때문에 3번 stress setting에서 4번 controlled setting으로 자연스럽게 난이도를 완화하는 흐름을 만들 수 있다.

## 5. K=4 공통 변수 + 군집별 특정 변수 setting

앞의 3번과 4번은 모든 component가 같은 active coordinate를 공유하는 setting이다. 여기서는 현실적인 sparse structure를 더 직접적으로 보기 위해, 공통 변수와 군집별 특정 변수를 함께 포함하는 setting을 구성했다. 핵심 질문은 특정 군집에만 영향을 주는 변수를 유지하면서, 순수 노이즈 변수를 제거할 수 있는지이다.

### 5.1. Simulation setting

**시뮬레이션 환경**

```text
K = 4
n = 1000
반복수 = 100
random start = 10
d = 100
공통 변수 = 6개
군집별 특정 변수 = component마다 4개, 총 16개
w = 0.50
max long path = Rossi 100 / Separate 140 / Eta 80
```

* **Tuning:** Rossi는 $\beta$ path, 분리 패널티는 $\lambda_\kappa$ grid와 $\lambda_\mu$ path, 에타 패널티는 centered $\eta$ norm의 $\lambda_\eta$ path에서 BIC 기준으로 선택.
* **Variables:** union 기준 유효 변수 개수는 22개.
  * 공통 변수는 1-6번 coordinate로 두고, 모든 component에서 nonzero로 설정했다.
  * 군집별 특정 변수는 component마다 4개씩 부여했다.
  * Component 1의 특정 변수는 7-10번, component 2는 11-14번, component 3은 15-18번, component 4는 19-22번이다.
  * 노이즈 변수는 23-100번 coordinate이다.
  * Entry-level active 개수는 공통 변수 $4\times6=24$개와 특정 변수 $4\times4=16$개를 합쳐 40개이다.
* **Mean direction:** raw vector에서 공통 변수는 1.0, 자기 component의 특정 변수는 $w=0.50$, 나머지는 0으로 둔 뒤 normalize했다.
* **Concentration:** $\kappa=(30,45,65,90)$.

Raw mean vector는 다음과 같다.

```text
common variables: v_kj = 1.0 for j = 1,...,6 and all k
component-specific variables: v_kj = 0.5 only for component k
noise variables: v_kj = 0
mu_k = v_k / ||v_k||
```

따라서 각 component는 공통 변수 6개와 자기 component의 특정 변수 4개를 사용한다. Component별 nonzero coordinate는 10개이고, union active coordinate는 22개이다. 평균 방향의 pairwise cosine은 0.857이다.

### 5.2. 군집화 및 변수 선택 성능

| Method | ARI | True union $q$ | Selected $q$ | TPR | FPR | Precision | F1 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Rossi path BIC | 0.680 | 22.000 | 98.520 | 1.000 | 0.981 | 0.223 | 0.365 |
| Rossi path BIC + refit | 0.653 | 22.000 | 98.520 | 1.000 | 0.981 | 0.223 | 0.365 |
| Separate 2D path/grid BIC | 0.684 | 22.000 | 86.460 | 1.000 | 0.826 | 0.258 | 0.409 |
| Separate 2D path/grid BIC + refit | 0.657 | 22.000 | 86.460 | 1.000 | 0.826 | 0.258 | 0.409 |
| Eta centered path BIC | 0.625 | 22.000 | **24.750** | 0.994 | **0.037** | **0.890** | **0.937** |
| **Eta centered path BIC + refit** | **0.686** | 22.000 | **24.750** | 0.994 | **0.037** | **0.890** | **0.937** |

### 5.3. 변수 유형별 선택률

| Method | Common selection | Specific selection | Noise selection |
|:---|---:|---:|---:|
| Rossi path BIC | 1.000 | 1.000 | 0.981 |
| Rossi path BIC + refit | 1.000 | 1.000 | 0.981 |
| Separate 2D path/grid BIC | 1.000 | 1.000 | 0.826 |
| Separate 2D path/grid BIC + refit | 1.000 | 1.000 | 0.826 |
| Eta centered path BIC | 1.000 | 0.992 | **0.037** |
| **Eta centered path BIC + refit** | **1.000** | **0.992** | **0.037** |

### 5.4. 모형 적합 지표

BIC는 공식 tuning 선택 기준이고, EBIC는 보조 적합 지표로 계산했다. 두 값 모두 낮을수록 좋다.

| Method | loglik | df | BIC | EBIC |
|:---|---:|---:|---:|---:|
| Rossi path BIC | 97540.991 | 277.850 | -193162.661 | -191883.115 |
| Rossi path BIC + refit | 97574.528 | 397.080 | -192406.124 | -190577.503 |
| Separate 2D path/grid BIC | 97465.244 | 172.040 | -193742.077 | -192949.804 |
| Separate 2D path/grid BIC + refit | 97562.901 | 348.840 | -192716.101 | -191109.633 |
| Eta centered path BIC | 97362.221 | 177.250 | -193500.043 | -192683.777 |
| **Eta centered path BIC + refit** | 97423.650 | 102.000 | **-194142.708** | **-193672.981** |

### 5.5. 모수 추정 성능

MSE 지표는 기존 표와 같이 $\times 100$으로 표시했다.

| Method | MSE_mu | MSE_kappa | MSE_centered_eta | kappa_hat_mean |
|:---|---:|---:|---:|---:|
| Rossi path BIC | 0.015 | 298.864 | 31.435 | 58.661 |
| Rossi path BIC + refit | 0.033 | 342.749 | 59.409 | 58.735 |
| Separate 2D path/grid BIC | **0.008** | 876.187 | **17.939** | 56.089 |
| Separate 2D path/grid BIC + refit | 0.030 | 306.433 | 55.247 | 58.599 |
| Eta centered path BIC | 0.029 | 1448.494 | 42.419 | 58.468 |
| **Eta centered path BIC + refit** | 0.010 | **190.092** | 18.541 | 58.040 |

### 5.6. 해석

* 공통 변수와 군집별 특정 변수가 함께 있는 setting에서도 Rossi와 분리 패널티는 true active variable을 모두 선택하지만, 노이즈 변수도 대부분 함께 선택한다.
* Rossi path BIC의 selected $q$는 98.520이고, 분리 패널티의 selected $q$는 86.460이다. True union $q=22$에 비해 지나치게 많은 변수를 선택한다.
* 에타 패널티 + refit은 selected $q=24.750$으로 true union $q=22$에 가장 가깝고, FPR은 0.037로 가장 낮다.
* 변수 유형별로 보면 에타 패널티 + refit은 공통 변수 선택률 1.000, 군집별 특정 변수 선택률 0.992를 유지하면서 노이즈 선택률을 0.037로 낮춘다.
* Refit 기준 ARI는 에타 패널티 + refit이 0.686으로 가장 높고, BIC도 가장 좋다.
* 이 setting은 제안 방법이 단순히 공통 support를 찾는 것이 아니라, 군집별 특정 변수까지 유지하면서 노이즈 변수를 제거할 수 있음을 보여준다.

### 5.7. 군집별 특정 변수 weight 변화

군집별 특정 변수의 신호 세기 $w$를 변화시켜 robustness를 확인했다.

**시뮬레이션 환경**

```text
K = 4
n = 1000
반복수 = 100
random start = 10
d = 100
공통 변수 = 6개
군집별 특정 변수 = component마다 4개, 총 16개
w = 0.25, 0.35, 0.50
max long path = Rossi 100 / Separate 140 / Eta 80
```

나머지 변수 구조와 tuning 기준은 5.1과 동일하다.

| $w$ | mean cosine | Method | ARI | Selected $q$ | TPR | FPR | Precision | F1 | Specific selection | Noise selection |
|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.960 | Rossi path BIC | 0.394 | 98.070 | 1.000 | 0.975 | 0.225 | 0.367 | 1.000 | 0.975 |
| 0.25 | 0.960 | Rossi path BIC + refit | 0.368 | 98.070 | 1.000 | 0.975 | 0.225 | 0.367 | 1.000 | 0.975 |
| 0.25 | 0.960 | Separate 2D path/grid BIC | 0.401 | 93.500 | 0.999 | 0.917 | 0.236 | 0.382 | 0.998 | 0.917 |
| 0.25 | 0.960 | Separate 2D path/grid BIC + refit | 0.367 | 93.500 | 0.999 | 0.917 | 0.236 | 0.382 | 0.998 | 0.917 |
| 0.25 | 0.960 | Eta centered path BIC | 0.358 | 32.130 | 0.813 | 0.183 | 0.621 | 0.682 | 0.743 | 0.183 |
| 0.25 | 0.960 | Eta centered path BIC + refit | **0.399** | **32.130** | 0.813 | **0.183** | **0.621** | **0.682** | 0.743 | **0.183** |
| 0.35 | 0.924 | Rossi path BIC | 0.514 | 98.820 | 1.000 | 0.985 | 0.223 | 0.364 | 1.000 | 0.985 |
| 0.35 | 0.924 | Rossi path BIC + refit | 0.481 | 98.820 | 1.000 | 0.985 | 0.223 | 0.364 | 1.000 | 0.985 |
| 0.35 | 0.924 | Separate 2D path/grid BIC | 0.528 | 90.700 | 1.000 | 0.881 | 0.247 | 0.394 | 1.000 | 0.881 |
| 0.35 | 0.924 | Separate 2D path/grid BIC + refit | 0.488 | 90.700 | 1.000 | 0.881 | 0.247 | 0.394 | 1.000 | 0.881 |
| 0.35 | 0.924 | Eta centered path BIC | 0.458 | 29.810 | 0.919 | 0.123 | 0.723 | 0.794 | 0.888 | 0.123 |
| 0.35 | 0.924 | Eta centered path BIC + refit | **0.505** | **29.810** | 0.919 | **0.123** | **0.723** | **0.794** | 0.888 | **0.123** |
| 0.50 | 0.857 | Rossi path BIC | 0.680 | 98.520 | 1.000 | 0.981 | 0.223 | 0.365 | 1.000 | 0.981 |
| 0.50 | 0.857 | Rossi path BIC + refit | 0.653 | 98.520 | 1.000 | 0.981 | 0.223 | 0.365 | 1.000 | 0.981 |
| 0.50 | 0.857 | Separate 2D path/grid BIC | 0.684 | 86.460 | 1.000 | 0.826 | 0.258 | 0.409 | 1.000 | 0.826 |
| 0.50 | 0.857 | Separate 2D path/grid BIC + refit | 0.657 | 86.460 | 1.000 | 0.826 | 0.258 | 0.409 | 1.000 | 0.826 |
| 0.50 | 0.857 | Eta centered path BIC | 0.625 | 24.750 | 0.994 | 0.037 | 0.890 | 0.937 | 0.992 | 0.037 |
| 0.50 | 0.857 | Eta centered path BIC + refit | **0.686** | **24.750** | 0.994 | **0.037** | **0.890** | **0.937** | 0.992 | **0.037** |

Specific signal이 강해질수록 에타 패널티의 specific selection은 높아지고 noise selection은 낮아진다. $w=0.25$는 평균 방향이 매우 유사한 어려운 상황으로, 에타 패널티도 일부 군집별 특정 변수를 놓친다. 그러나 이 경우에도 Rossi와 분리 패널티에 비해 노이즈 변수 선택률은 크게 낮다. $w=0.25$에서는 에타 refit 중 일부 반복에서 $\kappa$ 추정이 크게 튀는 outlier가 있어, 모수 MSE 평균은 보조적으로 해석한다.

### 5.8. 집중도 차이가 약한 setting

5.1의 specific-effect 구조를 유지하되, 집중도 차이를 약하게 설정했다.

**시뮬레이션 환경**

```text
K = 4
n = 1000
반복수 = 100
random start = 10
d = 100
공통 변수 = 6개
군집별 특정 변수 = component마다 4개, 총 16개
w = 0.50
max long path = Rossi 100 / Separate 140 / Eta 80
```

집중도는 $\kappa=(40,50,60,70)$로 두었고, true $\kappa$ ratio는 1.75이다.

**a. 군집화 및 변수 선택 성능**

| Method | ARI | True union $q$ | Selected $q$ | TPR | FPR | Precision | F1 | Specific selection | Noise selection |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Rossi path BIC | 0.563 | 22.000 | 97.580 | 1.000 | 0.969 | 0.226 | 0.368 | 1.000 | 0.969 |
| Rossi path BIC + refit | 0.527 | 22.000 | 97.580 | 1.000 | 0.969 | 0.226 | 0.368 | 1.000 | 0.969 |
| Separate 2D path/grid BIC | 0.567 | 22.000 | 78.150 | 1.000 | 0.720 | 0.294 | 0.450 | 1.000 | 0.720 |
| Separate 2D path/grid BIC + refit | 0.538 | 22.000 | 78.150 | 1.000 | 0.720 | 0.294 | 0.450 | 1.000 | 0.720 |
| Eta centered path BIC | 0.564 | 22.000 | **23.980** | 1.000 | **0.025** | **0.921** | **0.958** | 1.000 | **0.025** |
| **Eta centered path BIC + refit** | **0.575** | 22.000 | **23.980** | 1.000 | **0.025** | **0.921** | **0.958** | 1.000 | **0.025** |

**b. 모형 적합 지표**

BIC는 공식 tuning 선택 기준이고, EBIC는 보조 적합 지표로 계산했다. 두 값 모두 낮을수록 좋다.

| Method | loglik | df | BIC | EBIC |
|:---|---:|---:|---:|---:|
| Rossi path BIC | 96843.815 | 267.210 | -191841.809 | -190611.261 |
| Rossi path BIC + refit | 96880.841 | 393.320 | -191044.724 | -189233.418 |
| Separate 2D path/grid BIC | 96753.400 | 149.400 | -192474.781 | -191786.768 |
| Separate 2D path/grid BIC + refit | 96858.138 | 315.600 | -191536.188 | -190082.796 |
| Eta centered path BIC | 96686.669 | 174.940 | -192164.895 | -191359.266 |
| **Eta centered path BIC + refit** | 96729.758 | **98.920** | **-192776.201** | **-192320.658** |

**c. 모수 추정 성능**

MSE 지표는 기존 표와 같이 $\times 100$으로 표시했다.

| Method | MSE_mu | MSE_kappa | MSE_centered_eta | kappa_hat_mean |
|:---|---:|---:|---:|---:|
| Rossi path BIC | 0.013 | 283.180 | 31.986 | 56.098 |
| Rossi path BIC + refit | 0.028 | 356.347 | 66.134 | 56.233 |
| Separate 2D path/grid BIC | **0.007** | 1163.311 | **17.589** | 53.194 |
| Separate 2D path/grid BIC + refit | 0.024 | 298.511 | 56.878 | 56.035 |
| Eta centered path BIC | 0.017 | 734.190 | 35.487 | 54.535 |
| **Eta centered path BIC + refit** | 0.007 | **180.401** | 18.125 | 55.490 |

집중도 차이가 약해져도 에타 패널티 + refit은 selected $q=23.980$으로 true union $q=22$에 가장 가깝고, noise selection을 0.025로 낮춘다. 모형 적합 지표에서도 에타 패널티 + refit의 BIC와 EBIC가 가장 좋다.

## 6. 고차원 시뮬레이션

공통 변수와 군집별 특정 변수가 함께 있는 5번 구조를 유지하고, 전체 차원 $d$를 증가시켜 고차원 상황에서의 변수 선택 성능을 확인했다.

### 6.1. 차원에 따라 $\kappa$를 조정한 setting

차원이 증가해도 concentration 수준이 지나치게 약해지지 않도록 $\kappa$를 차원에 따라 함께 증가시킨 setting이다.

**시뮬레이션 환경**

```text
K = 4
n = 600
반복수 = 20
random start = 10
d = 100, 200, 500
공통 변수 = 6개
군집별 특정 변수 = component마다 4개, 총 16개
w = 0.50
max long path = d=100,200: Rossi 220 / Separate 240 / Eta 240; d=500: Rossi 540 / Separate 540 / Eta 540
```

추가 설정은 다음과 같다.

```text
d = 100: kappa = (30, 45, 65, 90)
d = 200: kappa = (60, 90, 130, 180)
d = 500: kappa = (150, 225, 325, 450)
true union q = 22
tuning = long path + BIC
```

**a. 군집화 및 변수 선택 성능**

| d | Method | reps | ARI | Selected $q$ | TPR | FPR | Precision | F1 | Noise selection |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | Rossi path BIC | 20 | 0.654 | 98.950 | 1.000 | 0.987 | 0.222 | 0.364 | 0.987 |
| 100 | Rossi path BIC + refit | 20 | 0.613 | 98.950 | 1.000 | 0.987 | 0.222 | 0.364 | 0.987 |
| 100 | Separate 2D path/grid BIC | 20 | 0.657 | 97.800 | 1.000 | 0.972 | 0.225 | 0.367 | 0.972 |
| 100 | Separate 2D path/grid BIC + refit | 20 | 0.617 | 97.800 | 1.000 | 0.972 | 0.225 | 0.367 | 0.972 |
| 100 | Eta centered path BIC | 20 | 0.584 | 25.700 | 0.945 | 0.063 | 0.842 | 0.884 | 0.063 |
| 100 | Eta centered path BIC + refit | 20 | 0.658 | 25.700 | 0.945 | 0.063 | 0.842 | 0.884 | 0.063 |
| 200 | Rossi path BIC | 20 | 0.838 | 199.900 | 1.000 | 0.999 | 0.110 | 0.198 | 0.999 |
| 200 | Rossi path BIC + refit | 20 | 0.833 | 199.900 | 1.000 | 0.999 | 0.110 | 0.198 | 0.999 |
| 200 | Separate 2D path/grid BIC | 20 | 0.840 | 199.900 | 1.000 | 0.999 | 0.110 | 0.198 | 0.999 |
| 200 | Separate 2D path/grid BIC + refit | 20 | 0.833 | 199.900 | 1.000 | 0.999 | 0.110 | 0.198 | 0.999 |
| 200 | Eta centered path BIC | 20 | 0.848 | 33.200 | 1.000 | 0.063 | 0.865 | 0.911 | 0.063 |
| 200 | Eta centered path BIC + refit | 20 | 0.872 | 33.200 | 1.000 | 0.063 | 0.865 | 0.911 | 0.063 |
| 500 | Rossi path BIC | 20 | 0.879 | 499.150 | 1.000 | 0.998 | 0.044 | 0.084 | 0.998 |
| 500 | Rossi path BIC + refit | 20 | 0.879 | 499.150 | 1.000 | 0.998 | 0.044 | 0.084 | 0.998 |
| 500 | Separate 2D path/grid BIC | 20 | 0.849 | 499.100 | 1.000 | 0.998 | 0.044 | 0.084 | 0.998 |
| 500 | Separate 2D path/grid BIC + refit | 20 | 0.848 | 499.100 | 1.000 | 0.998 | 0.044 | 0.084 | 0.998 |
| 500 | Eta centered path BIC | 20 | 0.849 | 269.550 | 1.000 | 0.518 | 0.327 | 0.414 | 0.518 |
| 500 | Eta centered path BIC + refit | 20 | 0.850 | 269.550 | 1.000 | 0.518 | 0.327 | 0.414 | 0.518 |

**b. 모형 적합 지표**

BIC는 공식 tuning 선택 기준이고, EBIC는 보조 적합 지표로 계산했다. 두 값 모두 낮을수록 좋다.

| d | Method | loglik | df | BIC | EBIC |
|---:|:---|---:|---:|---:|---:|
| 100 | Rossi path BIC | 58606.959 | 294.250 | -115331.622 | -113976.551 |
| 100 | Rossi path BIC + refit | 58633.683 | 398.800 | -114716.270 | -112879.729 |
| 100 | Separate 2D path/grid BIC | 58579.585 | 255.800 | -115522.836 | -114344.833 |
| 100 | Separate 2D path/grid BIC + refit | 58631.806 | 394.200 | -114741.943 | -112926.584 |
| 100 | Eta centered path BIC | 58417.556 | 180.100 | -115683.025 | -114853.634 |
| 100 | Eta centered path BIC + refit | 58472.715 | 105.800 | -116268.636 | -115781.409 |
| 200 | Rossi path BIC | 160244.502 | 722.900 | -315864.664 | -312034.510 |
| 200 | Rossi path BIC + refit | 160252.190 | 802.600 | -315370.203 | -311117.774 |
| 200 | Separate 2D path/grid BIC | 160218.804 | 684.650 | -316057.951 | -312430.458 |
| 200 | Separate 2D path/grid BIC + refit | 160252.968 | 802.600 | -315371.761 | -311119.332 |
| 200 | Eta centered path BIC | 159891.736 | 302.600 | -317847.762 | -316244.491 |
| 200 | Eta centered path BIC + refit | 159915.445 | 135.800 | -318962.188 | -318242.676 |
| 500 | Rossi path BIC | 541005.006 | 1546.300 | -1072118.439 | -1062508.790 |
| 500 | Rossi path BIC + refit | 541086.996 | 1999.600 | -1069382.692 | -1056955.961 |
| 500 | Separate 2D path/grid BIC | 541102.936 | 1545.850 | -1072317.178 | -1062710.326 |
| 500 | Separate 2D path/grid BIC + refit | 541193.705 | 1999.400 | -1069597.389 | -1057171.901 |
| 500 | Eta centered path BIC | 540795.469 | 1311.650 | -1073200.405 | -1065049.014 |
| 500 | Eta centered path BIC + refit | 540761.756 | 1081.200 | -1074607.152 | -1067887.917 |

**c. 모수 추정 성능**

MSE 지표는 기존 표와 같이 $\times 100$으로 표시했다. $d=200$과 $d=500$에서는 일부 반복에서 $\kappa$ 추정이 크게 튀어 MSE_kappa 평균이 매우 커지므로 보조 지표로 해석한다.

| d | Method | reps | MSE_mu | MSE_kappa | MSE_centered_eta | kappa_hat_mean |
|---:|:---|---:|---:|---:|---:|---:|
| 100 | Rossi path BIC | 20 | 0.033 | 745.104 | 68.213 | 59.457 |
| 100 | Rossi path BIC + refit | 20 | 0.061 | 1184.728 | 120.360 | 59.835 |
| 100 | Separate 2D path/grid BIC | 20 | 0.024 | 1050.635 | 46.612 | 56.525 |
| 100 | Separate 2D path/grid BIC + refit | 20 | 0.056 | 772.304 | 106.414 | 59.400 |
| 100 | Eta centered path BIC | 20 | 0.051 | 3459.646 | 85.941 | 59.706 |
| 100 | Eta centered path BIC + refit | 20 | 0.034 | 1034.261 | 58.854 | 58.803 |
| 200 | Rossi path BIC | 20 | 0.027 | 1249850006190.260 | 4687412283.415 | 12617.062 |
| 200 | Rossi path BIC + refit | 20 | 0.032 | 1249850006190.010 | 4687414160.191 | 12617.019 |
| 200 | Separate 2D path/grid BIC | 20 | 0.026 | 1249850007300.090 | 4687414792.978 | 12613.389 |
| 200 | Separate 2D path/grid BIC + refit | 20 | 0.032 | 1249850006210.540 | 4687417290.502 | 12616.894 |
| 200 | Eta centered path BIC | 20 | 0.022 | 1249850007569.000 | 4687374387.273 | 12614.434 |
| 200 | Eta centered path BIC + refit | 20 | 0.013 | 1249850005273.270 | 4687374348.186 | 12614.785 |
| 500 | Rossi path BIC | 20 | 0.034 | 9997000249285.570 | 14998912607.026 | 100272.838 |
| 500 | Rossi path BIC + refit | 20 | 0.037 | 9997000249223.540 | 14998919681.703 | 100273.287 |
| 500 | Separate 2D path/grid BIC | 20 | 0.042 | 12496250310739.600 | 18748731607.535 | 125266.060 |
| 500 | Separate 2D path/grid BIC + refit | 20 | 0.045 | 12496250310033.900 | 18748746188.797 | 125268.088 |
| 500 | Eta centered path BIC | 20 | 0.042 | 12496062842582.900 | 18748597790.034 | 125264.996 |
| 500 | Eta centered path BIC + refit | 20 | 0.041 | 12496062840506.500 | 18748597742.965 | 125265.669 |

**d. 해석**

차원 증가에 맞추어 $\kappa$도 증가시키면 $d=200$까지는 에타 패널티 + refit이 ARI와 변수 선택 성능 모두에서 가장 안정적이다. $d=200$에서 에타 패널티 + refit은 selected $q=33.200$, FPR 0.063, F1 0.911을 보인다. 반면 Rossi와 분리 패널티는 거의 전체 변수를 선택한다.

$d=500$에서는 에타 패널티도 selected $q=269.550$으로 노이즈 변수를 많이 선택한다. 따라서 단순히 path를 길게 두는 것만으로는 매우 높은 차원에서 충분하지 않고, EBIC 또는 더 강한 high-dimensional tuning 기준을 추가로 검토할 필요가 있다.

### 6.2. $\kappa$를 고정한 setting

이번에는 모든 차원에서 $\kappa=(30,45,65,90)$를 동일하게 고정했다. 이 setting은 차원이 증가하면서 concentration signal이 상대적으로 약해지는 stress setting으로 해석한다.

**시뮬레이션 환경**

```text
K = 4
n = 600
반복수 = 20
random start = 10
d = 100, 200, 400
공통 변수 = 6개
군집별 특정 변수 = component마다 4개, 총 16개
w = 0.50
max long path = Rossi 540 / Separate 540 / Eta 540
```

추가 설정은 다음과 같다.

```text
kappa = (30, 45, 65, 90)
true union q = 22
tuning = long path + BIC
```

$d=200$은 20회 중 1회, $d=400$은 20회 중 13회에서 active coordinate를 선택하지 못해 실패했다. 따라서 $d=200$은 성공한 19회, $d=400$은 성공한 7회 기준으로 요약했다.

**a. 군집화 및 변수 선택 성능**

| d | Method | reps | ARI | Selected $q$ | TPR | FPR | Precision | F1 | Noise selection |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | Rossi path BIC | 20 | 0.678 | 44.850 | 0.989 | 0.296 | 0.547 | 0.688 | 0.296 |
| 100 | Rossi path BIC + refit | 20 | 0.659 | 44.850 | 0.989 | 0.296 | 0.547 | 0.688 | 0.296 |
| 100 | Separate 2D path/grid BIC | 20 | 0.680 | 39.400 | 0.989 | 0.226 | 0.587 | 0.726 | 0.226 |
| 100 | Separate 2D path/grid BIC + refit | 20 | 0.662 | 39.400 | 0.989 | 0.226 | 0.587 | 0.726 | 0.226 |
| 100 | Eta centered path BIC | 20 | 0.584 | 25.700 | 0.945 | 0.063 | 0.842 | 0.884 | 0.063 |
| 100 | Eta centered path BIC + refit | 20 | 0.658 | 25.700 | 0.945 | 0.063 | 0.842 | 0.884 | 0.063 |
| 200 | Rossi path BIC | 19 | 0.408 | 182.474 | 1.000 | 0.902 | 0.121 | 0.216 | 0.902 |
| 200 | Rossi path BIC + refit | 19 | 0.357 | 182.474 | 1.000 | 0.902 | 0.121 | 0.216 | 0.902 |
| 200 | Separate 2D path/grid BIC | 19 | 0.423 | 180.632 | 1.000 | 0.891 | 0.122 | 0.218 | 0.891 |
| 200 | Separate 2D path/grid BIC + refit | 19 | 0.363 | 180.632 | 1.000 | 0.891 | 0.122 | 0.218 | 0.891 |
| 200 | Eta centered path BIC | 19 | 0.391 | 34.105 | 0.773 | 0.096 | 0.559 | 0.627 | 0.096 |
| 200 | Eta centered path BIC + refit | 19 | 0.422 | 34.105 | 0.773 | 0.096 | 0.559 | 0.627 | 0.096 |
| 400 | Rossi path BIC | 7 | 0.094 | 396.714 | 1.000 | 0.991 | 0.055 | 0.105 | 0.991 |
| 400 | Rossi path BIC + refit | 7 | 0.092 | 396.714 | 1.000 | 0.991 | 0.055 | 0.105 | 0.991 |
| 400 | Separate 2D path/grid BIC | 7 | 0.144 | 395.143 | 1.000 | 0.987 | 0.056 | 0.105 | 0.987 |
| 400 | Separate 2D path/grid BIC + refit | 7 | 0.133 | 395.143 | 1.000 | 0.987 | 0.056 | 0.105 | 0.987 |
| 400 | Eta centered path BIC | 7 | 0.193 | 13.143 | 0.494 | 0.006 | 0.847 | 0.619 | 0.006 |
| 400 | Eta centered path BIC + refit | 7 | 0.234 | 13.143 | 0.494 | 0.006 | 0.847 | 0.619 | 0.006 |

**b. 모형 적합 지표**

BIC는 공식 tuning 선택 기준이고, EBIC는 보조 적합 지표로 계산했다. 두 값 모두 낮을수록 좋다.

| d | Method | loglik | df | BIC | EBIC |
|---:|:---|---:|---:|---:|---:|
| 100 | Rossi path BIC | 58446.650 | 77.150 | -116399.777 | -116044.489 |
| 100 | Rossi path BIC + refit | 58537.354 | 182.400 | -115907.909 | -115067.926 |
| 100 | Separate 2D path/grid BIC | 58435.770 | 65.600 | -116451.902 | -116149.803 |
| 100 | Separate 2D path/grid BIC + refit | 58526.717 | 160.600 | -116026.087 | -115286.496 |
| 100 | Eta centered path BIC | 58417.556 | 180.100 | -115683.025 | -114853.634 |
| 100 | Eta centered path BIC + refit | 58472.715 | 105.800 | -116268.636 | -115781.409 |
| 200 | Rossi path BIC | 150963.583 | 377.579 | -299511.821 | -297511.288 |
| 200 | Rossi path BIC + refit | 151131.706 | 732.895 | -297575.137 | -293692.028 |
| 200 | Separate 2D path/grid BIC | 150945.339 | 362.632 | -299570.949 | -297649.612 |
| 200 | Separate 2D path/grid BIC + refit | 151125.802 | 725.526 | -297610.463 | -293766.395 |
| 200 | Eta centered path BIC | 150682.357 | 305.316 | -299411.631 | -297793.971 |
| 200 | Eta centered path BIC + refit | 150736.538 | 139.421 | -300581.209 | -299842.512 |
| 400 | Rossi path BIC | 380356.028 | 1150.429 | -753352.845 | -746460.093 |
| 400 | Rossi path BIC + refit | 380460.158 | 1589.857 | -750750.113 | -741224.540 |
| 400 | Separate 2D path/grid BIC | 380259.655 | 1073.429 | -753652.663 | -747221.254 |
| 400 | Separate 2D path/grid BIC + refit | 380449.591 | 1583.571 | -750769.188 | -741281.276 |
| 400 | Eta centered path BIC | 379466.412 | 442.429 | -756102.640 | -753451.845 |
| 400 | Eta centered path BIC + refit | 379323.433 | 55.571 | -758291.379 | -757958.425 |

**c. 모수 추정 성능**

MSE 지표는 기존 표와 같이 $\times 100$으로 표시했다.

| d | Method | reps | MSE_mu | MSE_kappa | MSE_centered_eta | kappa_hat_mean |
|---:|:---|---:|---:|---:|---:|---:|
| 100 | Rossi path BIC | 20 | 0.024 | 326.225 | 32.467 | 58.047 |
| 100 | Rossi path BIC + refit | 20 | 0.034 | 580.206 | 66.349 | 58.815 |
| 100 | Separate 2D path/grid BIC | 20 | 0.021 | 444.695 | 26.366 | 57.617 |
| 100 | Separate 2D path/grid BIC + refit | 20 | 0.026 | 413.219 | 50.872 | 58.545 |
| 100 | Eta centered path BIC | 20 | 0.051 | 3459.646 | 85.941 | 59.706 |
| 100 | Eta centered path BIC + refit | 20 | 0.034 | 1034.261 | 58.854 | 58.803 |
| 200 | Rossi path BIC | 19 | 0.058 | 17314.487 | 187.750 | 64.555 |
| 200 | Rossi path BIC + refit | 19 | 0.120 | 20974.115 | 364.693 | 65.999 |
| 200 | Separate 2D path/grid BIC | 19 | 0.054 | 6774.716 | 126.638 | 60.529 |
| 200 | Separate 2D path/grid BIC + refit | 19 | 0.121 | 9677.156 | 298.681 | 63.932 |
| 200 | Eta centered path BIC | 19 | 0.062 | 21711.201 | 167.048 | 62.003 |
| 200 | Eta centered path BIC + refit | 19 | 0.075 | 18984.266 | 220.282 | 63.049 |
| 400 | Rossi path BIC | 7 | 0.121 | 65790.169 | 528.171 | 77.245 |
| 400 | Rossi path BIC + refit | 7 | 0.151 | 70271.549 | 665.392 | 78.374 |
| 400 | Separate 2D path/grid BIC | 7 | 0.129 | 35268.919 | 402.803 | 66.263 |
| 400 | Separate 2D path/grid BIC + refit | 7 | 0.162 | 59171.933 | 635.593 | 75.979 |
| 400 | Eta centered path BIC | 7 | 0.060 | 24118.178 | 107.938 | 52.047 |
| 400 | Eta centered path BIC + refit | 7 | 0.058 | 21401.868 | 172.748 | 60.762 |

**d. 해석**

$\kappa$를 고정하고 $d$만 증가시키면 concentration signal이 상대적으로 약해지므로, 단순한 high-dimensional robustness라기보다 stress setting에 가깝다. $d=100$에서는 에타 패널티가 true union $q=22$에 가장 가깝게 변수를 선택한다. $d=200$에서는 에타 패널티가 여전히 Rossi와 분리 패널티보다 훨씬 적은 변수를 선택하지만, TPR이 0.773으로 떨어져 일부 true active variable을 놓친다. $d=400$에서는 에타 패널티가 noise를 거의 선택하지 않지만 true active variable도 많이 놓치며, 전체 20회 중 13회에서 active coordinate를 선택하지 못했다.

따라서 $\kappa$ 고정 setting은 제안 방법의 noise control 장점과 동시에, 차원이 커질 때 concentration signal이 약해지는 상황에서 발생할 수 있는 과소선택 한계를 보여준다.

## 7. 시뮬레이션 변수 구조 요약

| 번호 | setting | 전체 변수 $d$ | 공통 active 변수 | 군집별 특정 변수 | 노이즈 변수 | union active $q$ |
|:---|:---|---:|:---|:---|:---|---:|
| 1 | K=2 기본 메커니즘 | 100 | 두 component가 같은 10개 active coordinate 공유 | 없음 | 90개 | 10 |
| 2.1 | Rossi 2022 corrected reproduction | 100 | component별 nonzero가 90개로 매우 많아 대부분의 coordinate가 여러 component에서 active | 명시적으로 설계하지 않음 | component별 zero coordinate 10개 | 반복별 상이 |
| 2.2 | K=4 sparse-active 비교 | 100 | component별 support를 랜덤 생성하므로 반복마다 겹침 정도가 달라짐 | 명시적으로 설계하지 않았지만, 랜덤 support 때문에 일부 coordinate는 특정 component에서만 active | 평균 약 65.9개 | 평균 34.1 |
| 3 | K=4 stress setting | 100 | 모든 component가 같은 10개 active coordinate 공유 | 없음 | 90개 | 10 |
| 4 | K=4 controlled concentration-dominant setting | 100 | 모든 component가 같은 10개 active coordinate 공유 | 없음 | 90개 | 10 |
| 5 | K=4 공통 변수 + 군집별 특정 변수 setting | 100 | 6개 | component마다 4개씩, 총 16개 | 78개 | 22 |
| 5.7 | 군집별 특정 변수 weight 변화 | 100 | 6개 | component마다 4개씩, 총 16개. $w=0.25,0.35,0.50$ 변화 | 78개 | 22 |
| 5.8 | 집중도 차이가 약한 setting | 100 | 6개 | component마다 4개씩, 총 16개. $w=0.50$ 고정 | 78개 | 22 |
| 6.1 | 고차원, $\kappa$ 차원별 조정 | 100, 200, 500 | 6개 | component마다 4개씩, 총 16개. $w=0.50$ 고정 | $d-22$개 | 22 |
| 6.2 | 고차원, $\kappa$ 고정 | 100, 200, 400 | 6개 | component마다 4개씩, 총 16개. $w=0.50$ 고정 | $d-22$개 | 22 |

1, 3, 4번은 같은 active coordinate를 공유하는 공통 support setting이다. 2번은 Rossi 논문식 sparse prototype 구조 또는 랜덤 sparse-active 구조이므로 공통 변수와 군집별 특정 변수를 명시적으로 통제하지 않는다. 5번 계열은 공통 변수와 군집별 특정 변수를 명시적으로 나누어, 제안 방법이 특정 군집 변수와 노이즈 변수를 구분할 수 있는지 확인하는 setting이다. 5.7은 군집별 특정 변수의 신호 세기 $w$를 바꾼 robustness 확인이고, 5.8은 같은 변수 구조에서 집중도 차이만 약하게 만든 추가 확인이다. 6번 계열은 5번 변수 구조를 유지한 채 전체 차원 $d$를 증가시킨 고차원 setting이다.

## 8. 요약 결론

* **기본 메커니즘:** $K=2$ 환경에서는 평균 방향이 같고 집중도 차이가 군집을 만드는 경우, $\eta$-penalty가 Rossi 및 분리 패널티보다 FPR을 크게 낮추고 F1을 개선한다.
* **Refit 역할:** $\eta$-penalty 단독은 $\kappa$ 수축 편향이 생길 수 있으나, 선택된 support를 고정한 refit을 수행하면 $\kappa$ ratio와 $\eta$ contrast가 true value에 가깝게 복원된다.
* **논문 재현:** Rossi & Barbaro (2022)의 sparse vMF는 논문 Figure와 유사하게 재현된다. 특히 논문 기준 sparsity는 zero coordinate 비율로 해석해야 한다.
* **K=4 stress setting:** 평균 방향이 같고 집중도만 다른 어려운 상황에서는 path tuning 후에도 Rossi와 분리 패널티가 불필요한 변수를 많이 선택한다. $\eta$-penalty는 FPR을 낮추고 해석 가능한 변수 선택을 제공한다.
* **Controlled concentration-dominant setting:** 변수 구조를 K=4 stress setting과 동일하게 고정하고 평균 방향만 약간 다르게 두면, Rossi와 분리 패널티는 거의 전체 변수를 선택하지만 $\eta$-penalty + refit은 FPR을 낮추고 refit 기준 ARI와 BIC도 가장 좋다.
* **공통 변수 + 군집별 특정 변수 setting:** $\eta$-penalty + refit은 공통 변수와 군집별 특정 변수를 거의 모두 유지하면서 노이즈 선택률을 0.034로 낮춘다. 이는 제안 방법이 특정 군집 변수까지 반영하는 sparse structure에서 유리할 수 있음을 보여준다.
* **약한 집중도 차이 setting:** $\kappa=(40,50,60,70)$로 concentration contrast를 줄여도 $\eta$-penalty + refit은 selected $q=23.980$, FPR 0.025, F1 0.958로 가장 안정적인 변수 선택을 보인다.
* **고차원 setting:** $\kappa$를 차원에 따라 조정하면 $d=200$까지는 $\eta$-penalty + refit이 selected $q=33.200$, FPR 0.063, F1 0.911로 안정적이다. 그러나 $d=500$에서는 BIC 기준이 여전히 느슨해져 selected $q=269.550$으로 증가한다.
* **고차원 fixed-$\kappa$ stress setting:** $\kappa$를 고정하고 $d$만 증가시키면 concentration signal이 상대적으로 약해져 $d=400$에서 적합 실패와 과소선택이 발생한다. 이 결과는 고차원에서는 tuning 기준과 concentration scaling을 함께 설계해야 함을 보여준다.

## 부록. $\eta$-Penalty 모형의 수리적 타당성

**① 베이즈 결정 경계 직접 수축**
$$\log \frac{\tau_{i2}}{\tau_{i1}} = \mathrm{const} + (\eta_2 - \eta_1)^T x_i$$
* 사후 확률을 결정하는 실질적 선형 판별 계수는 $\mu$가 아닌 $\eta$의 대조다.
* $-\lambda_\eta \|\eta_2 - \eta_1\|_1$ 패널티는 노이즈 차원의 판별 계수를 직접 줄이는 방향으로 작동한다.

**② 집중도 주도 환경 식별**
* **조건:** $\mu_1 = \mu_2$, $\kappa_1 \ll \kappa_2$
* **$\mu$-penalty (기존)**: $\|\mu_2 - \mu_1\| = 0$이므로 평균 방향 차이만으로는 집중도 차이를 설명하기 어렵다.
* **$\eta$-penalty (제안)**: $\eta_2 - \eta_1 = (\kappa_2-\kappa_1)\mu_1$이므로 평균 방향이 동일하더라도 집중도 차이로 생기는 coordinate-level 효과를 반영할 수 있다.

**③ 내재적 정규화**
$$\|\eta_k\|_2 = \|\kappa_k \mu_k\|_2 = \kappa_k$$
* 자연모수의 $L_2$ 노름이 곧 집중도 $\kappa$이므로, $\eta$ 벡터에 대한 penalty는 $\kappa$ scale에도 영향을 준다.
* 이 성질은 고차원에서 $\kappa_k$가 과도하게 커지는 현상을 완화할 가능성이 있다. 다만 이는 추가 이론 검토가 필요한 부분이다.

**④ 수축 편향 제거**

$$\hat{S}_\eta = \{j : |\hat{\eta}_{2j} - \hat{\eta}_{1j}| > 0 \}$$
$$\mu_{kj}=0 \quad \mathrm{for}\quad j\notin\hat{S}_\eta$$
* Phase 1에서 도출된 support $\hat{S}_\eta$를 고정한 채, penalty 없이 unpenalized EM을 수행한다.
* 이 단계는 $L_1$ penalty로 인해 축소된 $\kappa$ 추정치와 eta contrast를 보정하기 위한 post-selection refit으로 해석한다.
