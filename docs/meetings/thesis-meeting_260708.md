# Thesis Meeting 260708

## 1. 이번 미팅 목적

이번 문서는 2026-06-24 연구미팅에서 받은 피드백에 대한 답변을 정리하고, 다음 실험 방향을 확인받기 위한 자료다. 핵심은 다음 네 가지다.

- 피드백 1: $\eta$, $\mu$, $\kappa$의 의존성과 유일성
- 피드백 2: Eta-group penalty가 불리한 상황
- 추가 정리: 논문용 S1-S6 시뮬레이션, dense-support negative-control, 외부 baseline 결과
- 추가 정리: Eta penalty ablation 진단

전체 시뮬레이션 표는 [thesis-simulation_260708.md](../simulations/thesis-simulation_260708.md)에 정리했고, 이전 negative-control 세부 진단은 [negative_control_summary_260708.md](../../results/negative_control_summary_260708/negative_control_summary_260708.md)에 따로 남겼다.

## 2. 의존성과 유일성

vMF mixture에서 posterior decision score에 직접 들어가는 자연모수는

$$
\eta_k=\kappa_k\mu_k
$$

이다. 여기서 $\mu_k$는 평균 방향, $\kappa_k$는 그 방향의 집중도 또는 decision strength를 나타낸다. posterior decision score에는 다음 항이 들어간다.

$$
\log \alpha_k+\log C_d(\|\eta_k\|_2)+\eta_k^\top x_i.
$$

$\eta_k\ne0$, $\kappa_k>0$, $\|\mu_k\|_2=1$이면 단위 구면 제약 때문에

$$
\kappa_k=\|\eta_k\|_2,\qquad
\mu_k=\eta_k/\|\eta_k\|_2
$$

로 복원된다. 따라서 component-level parameterization에서는 $\eta_k$가 주어졌을 때 $\mu_k$와 $\kappa_k$가 유일하게 정해진다.

예를 들어

$$
\eta=(3,4,0)
$$

이면

$$
\kappa=\|\eta\|_2=5,\qquad
\mu=(3/5,4/5,0)=(0.6,0.8,0)
$$

이다. 이때 $\|\mu\|_2=1$이므로 구면 제약을 만족하는 분해가 하나로 정해진다.

단, $\eta=0$ 또는 $\kappa=0$이면 방향 $\mu$는 식별되지 않는다. 또한 mixture model의 label switching은 별도 문제다. 따라서 여기서 말하는 유일성은 mixture 전체의 전역 식별성 증명이 아니라, component-level parameterization에 대한 설명이다.

### 2.1 왜 Eta-group인가?

vMF mixture의 posterior decision score에는

$$
\eta_k^\top x_i=(\kappa_k\mu_k)^\top x_i
$$

가 직접 들어간다. 따라서 posterior decision boundary에 작동하는 parameter는 $\mu$ 단독이나 $\kappa$ 단독이 아니라 자연모수 $\eta_k=\kappa_k\mu_k$다.

Clustering에서 중요한 것은 각 component의 개별 $\eta_k$가 아니라, component 사이에서 어떤 coordinate가 posterior decision score 차이를 만드는가이다. 그래서 centered eta contrast

$$
c_{kj}=\eta_{kj}-\bar{\eta}_j,\qquad
\bar{\eta}_j=K^{-1}\sum_{\ell=1}^K\eta_{\ell j}
$$

를 본다. Eta-group penalty는

$$
\lambda_\eta\sum_{j=1}^d \|c_{\cdot j}\|_2
$$

로 두어, coordinate $j$가 component 간 posterior decision boundary를 만드는지 직접 선택한다.

요약하면 Eta-group은 posterior decision score에 들어가는 centered eta contrast를 coordinate 단위로 선택하는 구조다. 현재 주장은 일괄적 우월성이 아니라 posterior decision support recovery에 둔다.

### 2.2 Eta penalty ablation 진단

이 절은 S1-S6 성능 결과가 아니라, 제안 모형의 구조를 분해한 진단이다. 질문은 eta 자연모수만으로 충분한지, group penalty만으로 충분한지, 아니면 `centered eta contrast + coordinate-wise group penalty` 조합이 필요한지이다.

| 비교 목적 | method | penalty / model | reps | selected q | ARI | TPR | FPR | Precision | F1 | MSE_eta | 해석 |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| Proposed reference | Eta-group + refit | $\lambda_\eta\sum_j\lVert c_{\cdot j}\rVert_2$ | 20 | 25.45 | 0.684 | 0.995 | 0.046 | 0.867 | 0.925 | 0.191 | true q=22 근처 support를 선택 |
| Same eta, no group | Eta entry-wise L1 + refit | $\lambda_\eta\sum_{k,j}\lvert c_{kj}\rvert$ | 20 | 99.90 | 0.652 | 1.000 | 0.999 | 0.220 | 0.361 | 0.581 | 같은 eta라도 entry-wise L1은 거의 dense support |
| Rossi $\mu$-group | Rossi $\mu$-group + refit | $\lambda_\mu\sum_j\lVert\mu_{\cdot j}\rVert_2$ | 20 | 29.10 | 0.685 | 1.000 | 0.091 | 0.813 | 0.883 | 0.192 | $\mu$ group penalty는 dense support를 줄임 |
| Rossi $\mu$ baseline | Rossi $\mu$ + refit | $\lambda_\mu\sum_{k,j}\lvert\mu_{kj}\rvert$ | 20 | 98.80 | 0.653 | 1.000 | 0.985 | 0.223 | 0.364 | 0.581 | $\mu$ entry-wise penalty는 거의 dense support |

`MSE_eta`는 `MSE_centered_eta`를 줄여 쓴 표기다. `Eta entry-wise L1`과 `Rossi $\mu$-group`은 정식 제안 모형이 아니라 진단용 변형이다.

- Eta 자연모수만 사용해도 entry-wise L1이면 support가 거의 dense해졌다. 따라서 eta parameterization만으로는 충분하지 않다.
- $\mu$-space에 group penalty를 둔 Rossi $\mu$-group은 기존 Rossi $\mu$보다 개선되지만, Eta-group보다 FPR이 크고 F1이 낮았다. 따라서 group penalty만으로도 충분하지 않다.
- 현재 ablation 기준에서는 posterior decision score에 직접 들어가는 centered eta contrast를 coordinate 단위로 묶어 선택하는 Eta-group 구조가 가장 직접적이다.

## 3. 논문용 시뮬레이션 업데이트

이번에 논문용 시뮬레이션을 S1-S6 기본 시뮬레이션과 S1-N~S6-N dense-support negative-control로 정리했다. 목적은 clustering accuracy 자체가 아니라, posterior decision support recovery가 언제 잘 되고 언제 약해지는지 확인하는 것이다. 전체 표는 [thesis-simulation_260708.md](../simulations/thesis-simulation_260708.md)에 둔다.

비교 모형은 penalty를 거는 공간과 group/adaptive 여부로 구분한다. 여기서 $c_{kj}=\eta_{kj}-\bar\eta_j$는 centered eta contrast다.

| Model | penalty |
|:---|:---|
| D-L | $\lambda_\mu\sum_{k,j}\lvert\mu_{kj}\rvert$ |
| D-GL | $\lambda_\mu\sum_j\lVert\mu_{\cdot j}\rVert_2$ |
| D-AGL | $\lambda_\mu\sum_j w_j^{(D)}\lVert\mu_{\cdot j}\rVert_2$ |
| E-L | $\lambda_\eta\sum_{k,j}\lvert c_{kj}\rvert$ |
| E-GL | $\lambda_\eta\sum_j\lVert c_{\cdot j}\rVert_2$ |
| E-AGL | $\lambda_\eta\sum_j w_j^{(E)}\lVert c_{\cdot j}\rVert_2$ |

Adaptive weight는 $w_j^{(D)}\propto(\lVert\mu_{\cdot j}^{init}\rVert_2+\epsilon)^{-\gamma}$, $w_j^{(E)}\propto(\lVert c_{\cdot j}^{init}\rVert_2+\epsilon)^{-\gamma}$로 두고, 이번 시뮬레이션에서는 $\gamma=1$, $\epsilon=10^{-6}$을 사용했다.

### 3.1 기본 시뮬레이션 S1-S6

기본 시뮬레이션은 true decision support가 16개인 sparse decision-support setting이다. common q=4는 모든 component에 공통으로 들어가므로 decision support에는 포함하지 않는다.

| Scenario | 설정 | E-AGL 핵심 결과 | 외부 baseline 결과 | 해석 |
|:---|:---|:---|:---|:---|
| S1 | 평균 차이 큼(90도), 집중도 이분산 | ARI=0.865, selected q=16.06, F1=0.998, MSE_eta=0.057 | Dense vMF ARI=0.836 | E-AGL이 true q=16에 가장 가깝게 복원 |
| S2 | 평균 차이 큼(90도), 집중도 등분산 | ARI=0.904, selected q=16.12, F1=0.996, MSE_eta=0.057 | Dense vMF ARI=0.880 | 집중도 차이가 없어도 support recovery 안정적 |
| S3 | 평균 차이 보통(60도), 집중도 이분산 | ARI=0.631, selected q=21.22, F1=0.881, MSE_eta=0.250 | Dense vMF ARI=0.539 | 가장 중요한 중간 난도 setting. E-AGL 장점이 남음 |
| S4 | 평균 차이 보통(60도), 집중도 등분산 | ARI=0.651, selected q=16.32, F1=0.990, MSE_eta=0.079 | Dense vMF ARI=0.561 | 평균 차이만 있어도 decision support 복원 가능 |
| S5 | 평균 차이 작음(30도), 약한 집중도 이분산 | ARI=0.015, selected q=0.02, F1=0.118, MSE_eta=1.040 | Dense vMF ARI=0.029 | weak-signal stress-test. zero-support 쪽으로 수축 |
| S6 | 평균 차이 작음(30도), 집중도 등분산 | ARI=0.012, selected q=0.56, F1=0.105, MSE_eta=2.354 | Dense vMF ARI=0.011 | 가장 어려운 setting. 모든 방법의 성능이 낮음 |

핵심은 S1-S4에서는 E-AGL이 clustering 성능을 유지하면서 selected q를 true q=16 근처로 맞춘다는 점이다. D-L과 E-L은 ARI는 어느 정도 유지하지만 selected q가 거의 전체 차원에 가까워 support recovery에는 부적합하다. S5-S6은 주요 결과가 아니라 weak-signal limitation으로 두는 것이 안전하다.

### 3.2 Dense-support negative-control S1-N~S6-N

Negative-control은 평균 방향 차이와 집중도 차이의 축은 유지하되, decision q를 16에서 80으로 늘린 dense decision-support setting이다. 이 설정은 Eta-group이 sparse support setting에서만 유리한지 확인하기 위한 진단이다.

| Scenario | 설정 | E-AGL 핵심 결과 | 외부 baseline 결과 | 해석 |
|:---|:---|:---|:---|:---|
| S1-N | 평균 차이 큼(90도), 집중도 이분산 | ARI=0.857, selected q=82.40, F1=0.985 | Dense vMF ARI=0.835 | dense support에서도 안정적 |
| S2-N | 평균 차이 큼(90도), 집중도 등분산 | ARI=0.897, selected q=81.82, F1=0.989 | Dense vMF ARI=0.886 | dense support에서도 true q=80 근처 선택 |
| S3-N | 평균 차이 보통(60도), 집중도 이분산 | ARI=0.565, selected q=76.06, F1=0.840 | Dense vMF ARI=0.545 | D-AGL보다 support F1이 낮아지는 limitation |
| S4-N | 평균 차이 보통(60도), 집중도 등분산 | ARI=0.629, selected q=16.70, F1=0.979 | Dense vMF ARI=0.562 | E-AGL이 decision q=80 중 일부만 선택. BIC 튜닝 실패 후보 |
| S5-N | 평균 차이 작음(30도), 약한 집중도 이분산 | ARI=0.001, selected q=0.06, F1=0.025 | Dense vMF ARI=0.024 | weak signal에서 전체적으로 성능 저하 |
| S6-N | 평균 차이 작음(30도), 집중도 등분산 | ARI=0.005, selected q=2.04, F1=0.172 | Dense vMF ARI=0.012 | dense/sparse 여부와 무관하게 signal이 너무 약함 |

S1-N과 S2-N은 dense support에서도 E-AGL의 성능이 바로 붕괴되지는 않음을 보여준다. 반면 S3-N과 S4-N은 평균 방향 차이가 보통 수준일 때 Eta-group 계열이 support를 과소선택하거나 BIC 튜닝 실패를 보일 수 있음을 보여준다. S5-N/S6-N은 signal 자체가 너무 약한 stress-test다.

### 3.3 외부 baseline 해석

외부 baseline은 내부 ablation 모형과 목적이 다르다. Spherical k-means와 Dense vMF free kappa는 support recovery 모형이 아니므로 ARI/NMI/purity 중심으로만 비교한다. Sparse k-means는 feature support를 선택하지만, posterior decision support와 같은 목표가 아니므로 보조 지표로만 해석한다.

- Dense vMF free kappa는 S1-S4와 S1-N~S4-N에서 강한 clustering-only baseline이다.
- 그러나 Dense vMF는 sparse support를 제공하지 않으므로, posterior decision support recovery claim과 직접 경쟁하지 않는다.
- Sparse k-means는 support를 제공하지만 S1-S6 및 S1-N~S6-N에서 selected q가 과도하거나 ARI가 낮은 경우가 많다.
- Rossi 2022 시뮬레이션은 `mu_k` prototype sparsity setting이다. `d=100`에서 directional mean sparsity가 5%, 10%, 15%이면 모든 군집에서 nonzero인 common-like coordinate의 기대값이 약 81개, 66개, 52개라서 공통/공유 변수가 많은 구조다.
- 따라서 Rossi-style setting은 특정 군집 변수와 noise q를 고정한 posterior decision-support setting이 아니라, prototype sparsity 비교용 setting으로 분리해서 해석한다.
- dbmovMFs는 현재 로컬 R 환경에 패키지가 없어 이번 결과에서는 실행하지 못했다.

## 4. 제안 모형의 비용과 불리한 조건

제안 모형 E-AGL의 비용은 계산 시간 하나보다, 튜닝과 해석 구조가 더 복잡하다는 점에 있다. S1 1회 실행 시간 benchmark 기준(`K=4`, `n=1000`, `d=200`, `nstart=10`, path=240, Rcpp helper ON)에서 E-AGL은 D-L보다 느렸지만 D-GL/D-AGL보다는 빠르게 나왔다.

| 비교 | 1회 시간 | 해석 |
|:---|---:|:---|
| D-L | 3.75 sec | 가장 빠르지만 selected q=200으로 support recovery에는 부적합 |
| D-GL / D-AGL | 8.82 / 8.53 sec | group/adaptive group penalty 때문에 더 무거움 |
| E-GL / E-AGL | 5.62 / 5.53 sec | D-L보다 약 1.5배 느리지만, true q=16에 가깝게 복원 |

따라서 E-AGL의 주요 비용은 다음처럼 정리한다.

| 비용 또는 불리한 조건 | 내용 | 논문에서의 처리 |
|:---|:---|:---|
| 계산 비용 | D-L보다 약 1.47배 느림 | 실행 시간 diagnostic으로만 보고 |
| 튜닝 비용 | eta path, adaptive weight, BIC 선택에 민감 | path와 튜닝 규칙 명확화 필요 |
| weak signal | S5/S6처럼 signal이 약하면 zero-support 또는 과소선택 가능 | limitation/stress-test |
| dense decision support | S3-N/S4-N처럼 decision q가 크면 필요한 좌표까지 줄일 수 있음 | negative-control diagnostic |
| prototype sparsity target | Rossi-style setting은 posterior decision support가 아니라 prototype sparsity가 목표 | 별도 comparability experiment로 분리 |

즉, E-AGL은 모든 상황에서 가장 빠르거나 ARI가 가장 높은 방법이 아니라, sparse posterior decision support recovery가 목표일 때 계산/튜닝 비용을 감수할 가치가 있는 방법으로 설명하는 것이 안전하다.
