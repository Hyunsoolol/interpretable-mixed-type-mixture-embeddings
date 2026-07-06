# Thesis Meeting 260708

## 1. 이번 미팅 목적

이번 문서는 2026-06-24 연구미팅 피드백에 대한 답변과 이후 실험 방향을 정리한 자료다. 이번 미팅에서 확인할 내용은 다음 네 가지다.

- 피드백 1: $\eta$, $\mu$, $\kappa$의 의존성과 유일성
- 피드백 2: Eta-group penalty가 불리한 상황
- 추가 정리: 논문용 S1-S6 시뮬레이션, dense-support negative-control, 외부 baseline 결과
- 추가 정리: 공통 변수와 decision support 구분, Eta penalty ablation 진단

전체 시뮬레이션 표는 [thesis-simulation_260708.md](../simulations/thesis-simulation_260708.md)에 정리했고, 이전 negative-control 세부 진단은 [negative_control_summary_260708.md](../../results/negative_control_summary_260708/negative_control_summary_260708.md)에 별도 문서로 남겼다.

## 2. Eta-group 방법론 근거와 변수 선택 기준

### 2.1 $\eta$, $\mu$, $\kappa$ 관계와 유일성

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

### 2.2 왜 Eta-group인가?

vMF mixture의 posterior score는 자연모수 $\eta_k=\kappa_k\mu_k$를 통해 결정된다.

$$
s_k(x)=\log\pi_k+\log C_d(\kappa_k)+\eta_k^\top x.
$$

두 component의 decision score 차이는

$$
s_k(x)-s_\ell(x) =
\mathrm{const}_{k\ell}
+
(\eta_k-\eta_\ell)^\top x.
$$

따라서 변수 선택 대상은 $\mu$나 $\kappa$ 단독이 아니라, component 간 $\eta$ 차이를 만드는 coordinate다. 이를 centered contrast로 쓰면

$$
c_{kj}=\eta_{kj}-\bar\eta_j,\qquad
\bar\eta_j=K^{-1}\sum_{\ell=1}^K\eta_{\ell j}.
$$

제안 penalty는

$$
\lambda_\eta\sum_{j=1}^d\|c_{\cdot j}\|_2.
$$

즉, coordinate $j$가 posterior decision boundary에 필요한지를 직접 선택한다.

### 2.3 공통 변수가 선택되지 않는 이유

제안 모형의 support target은 $\mu$에 존재하는 좌표가 아니라 posterior decision score 차이를 만드는 좌표다. 두 component $k,\ell$에 대해

$$
s_k(x)-s_\ell(x) =
\log\frac{\pi_k}{\pi_\ell}
+
\log\frac{C_d(\kappa_k)}{C_d(\kappa_\ell)}
+
(\eta_k-\eta_\ell)^\top x.
$$

따라서 좌표 $j$가 공통 변수이면

$$
\eta_{1j}=\cdots=\eta_{Kj}
\quad\Rightarrow\quad
\eta_{kj}-\eta_{\ell j}=0.
$$

즉, 해당 좌표는 각 component score에는 들어가지만 posterior decision boundary에는 기여하지 않는다. Centered eta contrast로 쓰면

$$
c_{kj}=\eta_{kj}-\bar\eta_j,\qquad
\eta_{\cdot j}=a_j\mathbf{1}
\Rightarrow
c_{\cdot j}=0.
$$

따라서 Eta-group penalty

$$
\lambda_\eta\sum_{j=1}^d\|c_{\cdot j}\|_2
$$

에서는 공통 좌표가 선택되지 않는다. 반면 Rossi 계열의 $\mu$-support는 각 군집 중심을 설명하는 좌표를 보기 때문에 공통 좌표도 선택될 수 있다.

예외적으로 $\mu$가 공통이어도 $\kappa_k$가 다르면 $\eta_{kj}=\kappa_k\mu_{kj}$가 달라져 $c_{\cdot j}\ne0$이 될 수 있다. 이 경우 해당 좌표는 decision support에 포함될 수 있다.

### 2.4 Eta penalty ablation 진단

이 절은 S1-S6 성능 결과가 아니라, 제안 모형의 구조를 분해한 ablation 진단이다. eta 자연모수, group penalty, centered contrast의 역할을 분리해 확인한다.

진단 시뮬레이션 환경:

| 항목 | 설정 |
|:---|:---|
| 목적 | penalty target, centering, group/adaptive 효과 분해 |
| 반복 수 | rep=20 |
| 차원/표본/군집 수 | $d=100$, $n=1000$, $K=4$ |
| 변수 구성 | common q=6, component-specific q=4 per component, true union q=22 |
| specific weight | 0.5 |
| 집중도 | $\kappa=(30,45,65,90)$ |
| 선택 기준 | BIC |
| 재적합 | 선택된 support에서 refit |

| 비교 목적 | method | penalty / model | reps | selected q | specific q | noise q | ARI | TPR | FPR | Precision | F1 | MSE_eta | 해석 |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| $\mu$ entry-wise | M-L + refit | $\lambda_\mu\sum_{k,j}\lvert\mu_{kj}\rvert$ | 20 | 25.90 | 16.00 | 3.90 | 0.687 | 1.000 | 0.050 | 0.855 | 0.920 | 0.162 | baseline |
| $\mu$ group | M-GL + refit | $\lambda_\mu\sum_j\lVert\mu_{\cdot j}\rVert_2$ | 20 | 23.95 | 16.00 | 1.95 | 0.687 | 1.000 | 0.025 | 0.921 | 0.958 | 0.146 | group penalty 효과 |
| adaptive $\mu$ group | M-AGL + refit | $\lambda_\mu\sum_j w_j^{(M)}\lVert\mu_{\cdot j}\rVert_2$ | 20 | 22.55 | 16.00 | 0.55 | 0.689 | 1.000 | 0.007 | 0.977 | 0.988 | 0.131 | true q=22 근처 |
| raw $\eta$ entry-wise | E-L + refit | $\lambda_\eta\sum_{k,j}\lvert\eta_{kj}\rvert$ | 20 | 30.85 | 16.00 | 8.85 | 0.680 | 1.000 | 0.113 | 0.722 | 0.836 | 0.226 | raw eta L1은 noise 선택 증가 |
| raw $\eta$ group | E-GL + refit | $\lambda_\eta\sum_j\lVert\eta_{\cdot j}\rVert_2$ | 20 | 23.20 | 16.00 | 1.20 | 0.687 | 1.000 | 0.015 | 0.950 | 0.974 | 0.141 | eta group만의 효과 |
| centered $\eta$ entry-wise | E-CL + refit | $\lambda_\eta\sum_{k,j}\lvert c_{kj}\rvert$ | 20 | 24.40 | 15.80 | 2.60 | 0.688 | 0.991 | 0.033 | 0.898 | 0.941 | 0.177 | centering + entry-wise |
| centered $\eta$ group | E-CGL + refit | $\lambda_\eta\sum_j\lVert c_{\cdot j}\rVert_2$ | 20 | 24.00 | 15.90 | 2.10 | 0.689 | 0.995 | 0.027 | 0.918 | 0.954 | 0.166 | centered eta group |
| adaptive centered $\eta$ group | E-CAGL + refit | $\lambda_\eta\sum_j w_j^{(E)}\lVert c_{\cdot j}\rVert_2$ | 20 | 22.05 | 15.80 | 0.25 | 0.687 | 0.991 | 0.003 | 0.989 | 0.990 | 0.139 | adaptive 제안형 |

`MSE_eta`는 `MSE_centered_eta`를 줄여 쓴 표기다. 이 표는 rep=20 diagnostic 결과이며, S1-S6 본 결과가 아니라 구조 분해 결과로 해석한다. 여기서 $c_{kj}=\eta_{kj}-\bar\eta_j$이고, true decision q는 22다. `specific q`와 `noise q`는 각각 specific 좌표 16개, noise 좌표 78개 중 선택된 평균 개수다.

- M-GL과 E-GL은 entry-wise L1보다 FPR을 낮췄다. 즉 group penalty 자체의 효과가 있다.
- E-L은 raw eta를 쓰지만 entry-wise penalty라 noise 선택이 늘었다. eta parameterization만으로는 충분하지 않고 group 구조가 필요하다.
- E-CGL/E-CAGL은 centered eta contrast를 사용하므로 posterior decision support를 직접 겨냥한다. 특히 E-CAGL은 selected q가 true q=22에 가장 가까웠고 FPR이 가장 낮았다.

표에서 제외한 진단 후보:

| 제외 후보 | 형식 | 제외 사유 |
|:---|:---|:---|
| M-CGL | $\lambda_\mu\sum_j\lVert\mu_{\cdot j}-\bar\mu_j\mathbf{1}\rVert_2$ | $\mu$는 단위 구면 제약을 갖는 방향 모수이고 posterior score에는 $\eta_k=\kappa_k\mu_k$가 들어간다. 따라서 $\mu$만 중심화하면 $\kappa$ 차이를 반영하지 못하고, posterior decision support 목표와 직접 맞지 않는다. 보조 diagnostic에서는 selected q=38.75, FPR=0.215, MSE_eta=1.359로 noise 선택도 컸다. |

## 3. 시뮬레이션 결과 요약

시뮬레이션은 S1-S6 기본 시뮬레이션과 S1-N~S6-N dense-support negative-control로 정리했다. 목적은 clustering accuracy 자체보다 posterior decision support recovery가 유지되는 조건과 약해지는 조건을 확인하는 데 있다. 전체 표는 [thesis-simulation_260708.md](../simulations/thesis-simulation_260708.md)에 둔다.

비교 모형은 penalty를 거는 공간과 group/adaptive 여부로 구분한다. 여기서 $c_{kj}=\eta_{kj}-\bar\eta_j$는 centered eta contrast다.

| Model | penalty |
|:---|:---|
| M-L | $\lambda_\mu\sum_{k,j}\lvert\mu_{kj}\rvert$ |
| M-GL | $\lambda_\mu\sum_j\lVert\mu_{\cdot j}\rVert_2$ |
| M-AGL | $\lambda_\mu\sum_j w_j^{(M)}\lVert\mu_{\cdot j}\rVert_2$ |
| E-CL | $\lambda_\eta\sum_{k,j}\lvert c_{kj}\rvert$ |
| E-CGL | $\lambda_\eta\sum_j\lVert c_{\cdot j}\rVert_2$ |
| E-CAGL | $\lambda_\eta\sum_j w_j^{(E)}\lVert c_{\cdot j}\rVert_2$ |

Adaptive weight는 $w_j^{(M)}\propto(\lVert\mu_{\cdot j}^{init}\rVert_2+\epsilon)^{-\gamma}$, $w_j^{(E)}\propto(\lVert c_{\cdot j}^{init}\rVert_2+\epsilon)^{-\gamma}$로 두고, 이번 시뮬레이션에서는 $\gamma=1$, $\epsilon=10^{-6}$을 사용했다.

### 3.1 기본 시뮬레이션 S1-S6

기본 시뮬레이션은 true decision support가 16개인 sparse decision-support setting이다. common q=4는 모든 component에 공통으로 들어가므로 decision support에는 포함하지 않는다.

| Scenario | 설정 | E-CAGL 핵심 결과 | 외부 baseline 결과 | 해석 |
|:---|:---|:---|:---|:---|
| S1 | 평균 차이 큼(90도), 집중도 이분산 | ARI=0.865, selected q=16.06, F1=0.998, MSE_eta=0.057 | Dense vMF ARI=0.836 | E-CAGL selected q가 true q=16에 가까움 |
| S2 | 평균 차이 큼(90도), 집중도 등분산 | ARI=0.904, selected q=16.12, F1=0.996, MSE_eta=0.057 | Dense vMF ARI=0.880 | 집중도 차이가 없어도 support recovery 안정적 |
| S3 | 평균 차이 보통(60도), 집중도 이분산 | ARI=0.631, selected q=21.22, F1=0.881, MSE_eta=0.250 | Dense vMF ARI=0.539 | 중간 난도 setting. E-CAGL 장점이 유지됨 |
| S4 | 평균 차이 보통(60도), 집중도 등분산 | ARI=0.651, selected q=16.32, F1=0.990, MSE_eta=0.079 | Dense vMF ARI=0.561 | 평균 차이만 있어도 decision support 복원 가능 |
| S5 | 평균 차이 작음(30도), 약한 집중도 이분산 | ARI=0.015, selected q=0.02, F1=0.118, MSE_eta=1.040 | Dense vMF ARI=0.029 | weak-signal stress-test. zero-support 쪽으로 수축 |
| S6 | 평균 차이 작음(30도), 집중도 등분산 | ARI=0.012, selected q=0.56, F1=0.105, MSE_eta=2.354 | Dense vMF ARI=0.011 | weak-signal setting. 모든 방법의 성능이 낮음 |

S1-S4에서는 E-CAGL이 clustering 성능을 유지하면서 selected q를 true q=16 근처로 맞췄다. M-L과 E-CL은 ARI는 유지했지만 selected q가 거의 전체 차원에 가까워 support recovery 관점에서는 한계가 있었다. S5-S6은 주요 결과가 아니라 weak-signal limitation으로 분리해 해석한다.

### 3.2 Dense-support negative-control S1-N~S6-N

Negative-control은 평균 방향 차이와 집중도 차이의 축은 유지하되, decision q를 16에서 80으로 늘린 dense decision-support setting이다. sparse support 가정 밖에서 Eta-group의 동작을 확인하기 위한 설정이다.

| Scenario | 설정 | E-CAGL 핵심 결과 | 외부 baseline 결과 | 해석 |
|:---|:---|:---|:---|:---|
| S1-N | 평균 차이 큼(90도), 집중도 이분산 | ARI=0.857, selected q=82.40, F1=0.985 | Dense vMF ARI=0.835 | dense support에서도 안정적 |
| S2-N | 평균 차이 큼(90도), 집중도 등분산 | ARI=0.897, selected q=81.82, F1=0.989 | Dense vMF ARI=0.886 | dense support에서도 true q=80 근처 선택 |
| S3-N | 평균 차이 보통(60도), 집중도 이분산 | ARI=0.565, selected q=76.06, F1=0.840 | Dense vMF ARI=0.545 | M-AGL보다 support F1이 낮아지는 limitation |
| S4-N | 평균 차이 보통(60도), 집중도 등분산 | ARI=0.629, selected q=16.70, F1=0.979 | Dense vMF ARI=0.562 | E-CAGL이 decision q=80 중 일부만 선택. BIC 튜닝 실패 가능성 |
| S5-N | 평균 차이 작음(30도), 약한 집중도 이분산 | ARI=0.001, selected q=0.06, F1=0.025 | Dense vMF ARI=0.024 | weak signal에서 전체적으로 성능 저하 |
| S6-N | 평균 차이 작음(30도), 집중도 등분산 | ARI=0.005, selected q=2.04, F1=0.172 | Dense vMF ARI=0.012 | dense/sparse 여부와 무관하게 signal 자체가 약함 |

S1-N과 S2-N에서는 dense support에서도 E-CAGL의 성능 저하가 크지 않았다. 반면 S3-N과 S4-N에서는 평균 방향 차이가 보통 수준일 때 Eta-group 계열의 과소선택 또는 BIC 튜닝 실패 가능성이 확인됐다. S5-N/S6-N은 signal 자체가 약한 stress-test로 해석한다.

### 3.3 외부 baseline 해석

외부 baseline은 제안 모형과 목표가 다르므로 ARI/NMI/purity 중심으로만 비교한다.

- Spherical k-means와 Dense vMF free kappa는 clustering-only baseline이며 sparse support를 제공하지 않는다.
- Sparse k-means는 feature support를 선택하지만 posterior decision support와는 목표가 다르다.
- Rossi 2022 setting은 $\mu_k$ prototype sparsity 비교용이다. `d=100`에서 sparsity 5%, 10%, 15%이면 common-like coordinate 기대값이 약 81개, 66개, 52개로, 공통/공유 변수가 많은 구조다. dbmovMFs는 현재 로컬 R 환경에 없어 제외했다.

## 4. 제안 모형의 비용과 불리한 조건

제안 모형 E-CAGL의 비용은 계산 시간뿐 아니라 튜닝과 해석 구조가 더 복잡하다는 점에 있다. S1 1회 실행 시간 benchmark 기준(`K=4`, `n=1000`, `d=200`, `nstart=10`, path=240, Rcpp helper ON)에서 E-CAGL은 M-L보다 느렸지만 M-GL/M-AGL보다는 빠르게 나왔다.

| 비교 | 1회 시간 | 해석 |
|:---|---:|:---|
| M-L | 3.75 sec | 계산 시간은 가장 짧지만 selected q=200으로 support recovery에는 한계 |
| M-GL / M-AGL | 8.82 / 8.53 sec | group/adaptive group penalty로 계산량 증가 |
| E-CGL / E-CAGL | 5.62 / 5.53 sec | M-L보다 약 1.5배 느리지만, true q=16에 가깝게 복원 |

E-CAGL의 주요 비용과 제약은 다음과 같다.

| 비용 또는 불리한 조건 | 내용 |
|:---|:---|
| 계산 비용 | M-L보다 약 1.47배 느림 |
| 튜닝 비용 | eta path, adaptive weight, BIC 선택에 민감 |
| weak signal | S5/S6처럼 signal이 약하면 zero-support 또는 과소선택 가능 |
| dense decision support | S3-N/S4-N처럼 decision q가 크면 필요한 좌표까지 줄일 수 있음 |
| prototype sparsity target | Rossi-style setting은 posterior decision support가 아니라 prototype sparsity가 목표 |

정리하면 E-CAGL은 모든 상황에서 가장 빠르거나 ARI가 가장 높은 방법은 아니다. 계산/튜닝 비용을 수반하지만, sparse posterior decision support recovery를 목표로 할 때 사용하는 방법으로 정리한다.
