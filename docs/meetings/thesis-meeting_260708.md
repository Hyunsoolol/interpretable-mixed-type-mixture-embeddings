# Thesis Meeting 260708

## 1. 이번 미팅 목적

이번 문서는 2026-06-24 연구미팅에서 받은 피드백에 대한 답변을 정리하고, 다음 실험 방향을 확인받기 위한 자료다. 핵심은 다음 세 가지다.

- 피드백 1: $\eta$, $\mu$, $\kappa$의 의존성과 유일성
- 피드백 2: Eta-group penalty가 불리한 상황
- 추가 정리: 논문용 S1-S6 simulation, dense-support negative-control, 외부 baseline 결과
- 추가 정리: Eta penalty ablation 진단과 optimization safeguard 설명

전체 simulation 표는 [thesis-simulation_260708.md](../simulations/thesis-simulation_260708.md)에 정리했고, 이전 negative-control 세부 진단은 [negative_control_summary_260708.md](../../results/negative_control_summary_260708/negative_control_summary_260708.md)에 따로 남겼다.

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
\lambda\sum_{j=1}^d \|c_{\cdot j}\|_2
$$

로 두어, coordinate $j$가 component 간 posterior decision boundary를 만드는지 직접 선택한다.

요약하면 Eta-group은 posterior decision score에 들어가는 centered eta contrast를 coordinate 단위로 선택하는 구조다. 현재 claim은 일괄적 우월성이 아니라 posterior decision support recovery에 둔다.

### 2.2 Eta penalty ablation 진단

앞 절의 이론적 직관은 posterior decision score에 직접 들어가는 centered eta contrast를 coordinate 단위로 선택하는 것이 자연스럽다는 것이다. 이 직관을 확인하기 위해 Eta-group, 같은 eta에 entry-wise L1을 둔 진단 변형, 기존 Rossi $\mu$ baseline, 그리고 Rossi $\mu$-group 진단 변형을 비교했다.

핵심은 eta 자연모수만으로 충분한지가 아니라, centered eta contrast를 어떤 단위로 줄일 것인가이다. Entry-wise L1은 $c_{kj}$를 개별 성분별로 줄이므로 coordinate $j$ 전체가 posterior decision boundary에 필요한지 판단하지 못한다. 반면 Eta-group은 $c_{\cdot j}$를 하나의 축으로 보고 선택하므로, 해당 coordinate가 component 간 posterior decision score 차이를 만드는지를 직접 평가한다.

| 비교 목적 | method | penalty / model | reps | selected q | ARI | TPR | FPR | Precision | F1 | MSE_eta | 해석 |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| Proposed reference | Eta-group + refit | $\lambda\sum_j\lVert c_{\cdot j}\rVert_2$ | 20 | 25.45 | 0.684 | 0.995 | 0.046 | 0.867 | 0.925 | 0.191 | true q=22 근처 support를 선택 |
| Same eta, no group | Eta entry-wise L1 + refit | $\lambda\sum_{k,j}\lvert c_{kj}\rvert$ | 20 | 99.90 | 0.652 | 1.000 | 0.999 | 0.220 | 0.361 | 0.581 | 같은 eta라도 entry-wise L1은 거의 dense support |
| Rossi $\mu$-group | Rossi $\mu$-group + refit | $\lambda_\mu\sum_j\lVert\mu_{\cdot j}\rVert_2$ | 20 | 29.10 | 0.685 | 1.000 | 0.091 | 0.813 | 0.883 | 0.192 | $\mu$ group penalty는 dense support를 줄임 |
| Rossi $\mu$ baseline | Rossi $\mu$ + refit | $\lambda_\mu\sum_{k,j}\lvert\mu_{kj}\rvert$ | 20 | 98.80 | 0.653 | 1.000 | 0.985 | 0.223 | 0.364 | 0.581 | $\mu$ entry-wise penalty는 거의 dense support |

`MSE_eta`는 `MSE_centered_eta`를 줄여 쓴 표기다. `Eta entry-wise L1`은 centered eta contrast에 entry-wise L1 penalty를 둔 diagnostic variant다.

- `Eta entry-wise L1`은 같은 eta 자연모수라도 entry-wise L1이면 거의 dense support로 가므로, eta 자연모수만으로는 충분하지 않다.
- `Rossi $\mu$ + refit`은 기존 Rossi baseline에 해당하는 $\mu$ entry-wise penalty다. 같은 strong setting에서 거의 dense support를 선택했고, Eta-group보다 FPR이 크고 F1이 낮았다.
- `Rossi $\mu$-group + refit`은 $\mu$-space에 group penalty를 둔 진단 변형이다. 기존 Rossi $\mu$보다는 support recovery가 개선되지만, Eta-group보다 FPR이 크고 F1이 낮았다.
- Eta-group은 음수 contrast를 새로 만드는 방법이 아니다. 선택된 coordinate 안에서 원래 추정된 centered eta contrast의 상대적 양수/음수 방향을 보존하면서 축 단위로 shrink한다.
- centered eta contrast에서 $c_{kj}>0$은 component $k$의 posterior decision score를 상대적으로 높이는 방향, $c_{kj}<0$은 상대적으로 낮추는 방향이다.
- 현재 diagnostic 기준에서는 eta 자연모수만으로도, group penalty만으로도 충분하지 않으며 `centered eta contrast + coordinate-wise group penalty` 조합이 support recovery에서 가장 안정적으로 보인다.

## 3. 논문용 simulation 업데이트

이번에 논문용 simulation을 S1-S6 기본 시뮬레이션과 S1-N~S6-N dense-support negative-control로 정리했다. 목적은 clustering accuracy 자체가 아니라, posterior decision support recovery가 언제 잘 되고 언제 약해지는지 확인하는 것이다. 전체 표는 [thesis-simulation_260708.md](../simulations/thesis-simulation_260708.md)에 둔다.

### 3.1 기본 시뮬레이션 S1-S6

기본 시뮬레이션은 true decision support가 16개인 sparse decision-support setting이다. common q=4는 모든 component에 공통으로 들어가므로 decision support에는 포함하지 않는다.

| Scenario | 설정 | E-AGL 핵심 결과 | 외부 baseline 결과 | 해석 |
|:---|:---|:---|:---|:---|
| S1 | 평균 차이 큼(90도), 집중도 이분산 | ARI=0.865, selected q=16.06, F1=0.998, MSE_eta=0.057 | Dense vMF ARI=0.836 | E-AGL이 true q=16을 가장 정확히 복원 |
| S2 | 평균 차이 큼(90도), 집중도 등분산 | ARI=0.904, selected q=16.12, F1=0.996, MSE_eta=0.057 | Dense vMF ARI=0.880 | 집중도 차이가 없어도 support recovery 안정적 |
| S3 | 평균 차이 보통(60도), 집중도 이분산 | ARI=0.631, selected q=21.22, F1=0.881, MSE_eta=0.250 | Dense vMF ARI=0.539 | 가장 중요한 중간 난도 setting. E-AGL 장점이 남음 |
| S4 | 평균 차이 보통(60도), 집중도 등분산 | ARI=0.651, selected q=16.32, F1=0.990, MSE_eta=0.079 | Dense vMF ARI=0.561 | 평균 차이만 있어도 decision support 복원 가능 |
| S5 | 평균 차이 작음(30도), 약한 집중도 이분산 | ARI=0.015, selected q=0.02, F1=0.118, MSE_eta=1.040 | Dense vMF ARI=0.029 | weak-signal stress-test. zero-support 쪽으로 수축 |
| S6 | 평균 차이 작음(30도), 집중도 등분산 | ARI=0.012, selected q=0.56, F1=0.105, MSE_eta=2.354 | Dense vMF ARI=0.011 | 가장 어려운 setting. 모든 방법이 거의 실패 |

핵심은 S1-S4에서는 E-AGL이 clustering 성능을 유지하면서 selected q를 true q=16 근처로 맞춘다는 점이다. D-L과 E-L은 ARI는 어느 정도 유지하지만 selected q가 거의 전체 차원에 가까워 support recovery에는 부적합하다. S5-S6은 main result가 아니라 weak-signal limitation으로 두는 것이 안전하다.

### 3.2 Dense-support negative-control S1-N~S6-N

Negative-control은 평균 방향 차이와 집중도 차이의 축은 유지하되, decision q를 16에서 80으로 늘린 dense decision-support setting이다. 이 설정은 Eta-group이 sparse support setting에서만 유리한지 확인하기 위한 진단이다.

| Scenario | 설정 | E-AGL 핵심 결과 | 외부 baseline 결과 | 해석 |
|:---|:---|:---|:---|:---|
| S1-N | 평균 차이 큼(90도), 집중도 이분산 | ARI=0.857, selected q=82.40, F1=0.985 | Dense vMF ARI=0.835 | dense support에서도 안정적 |
| S2-N | 평균 차이 큼(90도), 집중도 등분산 | ARI=0.897, selected q=81.82, F1=0.989 | Dense vMF ARI=0.886 | dense support에서도 true q=80 근처 선택 |
| S3-N | 평균 차이 보통(60도), 집중도 이분산 | ARI=0.565, selected q=76.06, F1=0.840 | Dense vMF ARI=0.545 | D-AGL보다 support F1이 낮아지는 limitation |
| S4-N | 평균 차이 보통(60도), 집중도 등분산 | ARI=0.629, selected q=16.70, F1=0.979 | Dense vMF ARI=0.562 | E-AGL이 decision q=80 중 일부만 선택. tuning failure 후보 |
| S5-N | 평균 차이 작음(30도), 약한 집중도 이분산 | ARI=0.001, selected q=0.06, F1=0.025 | Dense vMF ARI=0.024 | weak signal에서 전체적으로 실패 |
| S6-N | 평균 차이 작음(30도), 집중도 등분산 | ARI=0.005, selected q=2.04, F1=0.172 | Dense vMF ARI=0.012 | dense/sparse 여부와 무관하게 signal이 너무 약함 |

S1-N과 S2-N은 dense support에서도 E-AGL이 바로 무너지지 않음을 보여준다. 반면 S3-N과 S4-N은 평균 방향 차이가 보통 수준일 때 Eta-group 계열이 support를 과소선택하거나 tuning failure를 보일 수 있음을 보여준다. S5-N/S6-N은 signal 자체가 너무 약한 stress-test다.

### 3.3 외부 baseline 해석

외부 baseline은 내부 ablation 모형과 목적이 다르다. Spherical k-means와 Dense vMF free kappa는 support recovery 모형이 아니므로 ARI/NMI/purity 중심으로만 비교한다. Sparse k-means는 feature support를 선택하지만, posterior decision support와 같은 목표가 아니므로 보조 지표로만 해석한다.

- Dense vMF free kappa는 S1-S4와 S1-N~S4-N에서 강한 clustering-only baseline이다.
- 그러나 Dense vMF는 sparse support를 제공하지 않으므로, posterior decision support recovery claim과 직접 경쟁하지 않는다.
- Sparse k-means는 support를 제공하지만 S1-S6 및 S1-N~S6-N에서 selected q가 과도하거나 ARI가 낮은 경우가 많다.
- dbmovMFs는 현재 로컬 R 환경에 패키지가 없어 이번 결과에서는 실행하지 못했다.

### 3.4 Simulation claim 정리

논문에서 안전한 simulation claim은 다음과 같다.

- E-AGL은 sparse posterior decision-support setting(S1-S4)에서 true decision support를 가장 안정적으로 복원한다.
- 제안법의 강점은 universal ARI improvement가 아니라 clustering을 유지하면서 posterior decision support를 복원하는 데 있다.
- true decision support가 조밀하거나 signal이 약하면 E-AGL도 과소선택 또는 zero-support tuning failure를 보일 수 있다.
- 따라서 S1-S4는 main simulation, S5-S6과 S1-N~S6-N은 stress-test/limitation 또는 appendix diagnostic으로 두는 것이 안전하다.

## 4. proximal EM-type update와 단조증가

본 방법의 추정은 닫힌형 M-step이 아니라 proximal EM-type update다. vMF normalizing constant와 centered eta group penalty 때문에 penalized M-step을 한 번에 닫힌 형태로 풀기 어렵다.

따라서 각 iteration에서는 현재 위치에서 proximal shrinkage candidate를 만들고 objective를 확인한다. 보폭이 너무 크면 objective가 감소할 수 있으므로, 구현에서는 objective decrease가 생길 때 step size를 줄이는 step-halving line search를 사용한다.

이 부분은 자동 단조증가 정리나 전역 수렴 보장이 아니라 optimization safeguard로 설명하는 것이 안전하다.

## 5. 현재 결론

- $\eta_k=\kappa_k\mu_k$는 posterior decision score에 직접 들어가는 자연모수다.
- $\eta_k\ne0$이고 $\kappa_k>0$이면 $\mu_k$와 $\kappa_k$는 component-level에서 유일하게 복원된다.
- Eta-group의 이론적 동기는 posterior decision score에 직접 들어가는 centered eta contrast를 coordinate 단위로 선택한다는 점이다.
- ablation diagnostic에서는 centered eta contrast와 coordinate-wise group penalty의 조합이 support recovery 안정성에 중요해 보인다.
- 조밀 support 또는 약한 신호에서는 Eta-group이 과소선택하거나 BIC tuning failure를 보일 수 있다.
- 다음 단계는 S1-S4를 main simulation으로 두고, S5-S6 및 S1-N~S6-N을 limitation/appendix diagnostic으로 배치해도 되는지 교수님께 확인받는 것이다.
