# Thesis Meeting 260708

## 1. 이번 미팅 목적

이번 문서는 2026-06-24 연구미팅 피드백에 대한 답변과 이후 실험 결과를 정리한 자료다. 주요 내용은 다음 네 가지다.

- 피드백 1: $\eta$, $\mu$, $\kappa$의 의존성과 유일성
- 피드백 2: Eta-group penalty가 불리한 상황
- 추가 정리: 논문용 S1-S6 시뮬레이션, dense-support negative-control, 외부 baseline 결과
- 추가 정리: 공통 변수와 decision support 구분, Eta penalty ablation 진단

전체 시뮬레이션 표는 [thesis-simulation_260708.md](../simulations/thesis-simulation_260708.md)에 정리했고, 이전 negative-control 세부 진단은 [negative_control_summary_260708.md](../../results/negative_control_summary_260708/negative_control_summary_260708.md)에 별도 문서로 남겼다.

## 2. Eta-group penalty 구성과 변수 선택 기준

제안 모형의 penalty는 다음 네 요소로 구성된다.

$$
\eta_k=\kappa_k\mu_k,\qquad
c_{kj}=\eta_{kj}-\bar\eta_j,\qquad
\bar\eta_j=K^{-1}\sum_{\ell=1}^K\eta_{\ell j}.
$$

최종적으로 사용하는 adaptive centered Eta-group penalty는

$$
\boxed{
\lambda_\eta
\sum_{j=1}^d
w_j\|c_{\cdot j}\|_2
}
$$

이다. 여기서 선택되는 support는

$$
\widehat S_\eta=\{j:\|c_{\cdot j}\|_2>0\}
$$

로 정의한다.

### 2.1 $\mu$가 아니라 $\eta$를 기준으로 둔다

vMF mixture의 posterior score는

$$
s_k(x) =
\log \pi_k+\log C_d(\kappa_k)+\eta_k^\top x,
\qquad
\eta_k=\kappa_k\mu_k.
$$

두 component $k,\ell$의 decision 차이는

$$
s_k(x)-s_\ell(x)=
\mathrm{const}_{k\ell}
+
(\eta_k-\eta_\ell)^\top x
$$

이다. 따라서 posterior decision에 직접 들어가는 모수는 $\mu_k$ 단독이 아니라 $\eta_k=\kappa_k\mu_k$이다.

간단한 예로 $\mu_{1j}=\mu_{2j}=0.1$이어도 $\kappa_1=20$, $\kappa_2=80$이면

$$
\eta_{1j}=2,\qquad \eta_{2j}=8.
$$

$\mu$ 기준으로는 같은 좌표처럼 보이지만, posterior score에서는 component 간 차이를 만든다.

또한 $\eta_k\ne0$, $\kappa_k>0$, $\|\mu_k\|_2=1$이면

$$
\kappa_k=\|\eta_k\|_2,\qquad
\mu_k=\eta_k/\|\eta_k\|_2
$$

로 복원된다. 다만 $\eta_k=0$ 또는 $\kappa_k=0$이면 방향 $\mu_k$는 식별되지 않고, mixture label switching은 별도 문제다.

### 2.2 raw $\eta$가 아니라 centered contrast를 사용한다

좌표 $j$에 대해

$$
\eta_{\cdot j}
=
\bar\eta_j\mathbf 1+c_{\cdot j},
\qquad
\mathbf 1^\top c_{\cdot j}=0.
$$

공통 성분 $\bar\eta_j\mathbf 1$은 모든 component score에 동일하게 더해지므로 decision 차이에서는 사라진다.

$$
(\bar\eta_j\mathbf 1)_k-(\bar\eta_j\mathbf 1)_\ell=0,
\qquad
\eta_{kj}-\eta_{\ell j}=c_{kj}-c_{\ell j}.
$$

따라서 coordinate $j$가 posterior decision boundary에 기여하는지는 $\eta_{\cdot j}$의 절대 크기가 아니라 $c_{\cdot j}$의 존재 여부로 판단한다.

예를 들어

$$
\eta_{\cdot j}=(5,5,5,5)
\quad\Rightarrow\quad
c_{\cdot j}=(0,0,0,0)
$$

이므로 이 좌표는 공통 효과다. 반면

$$
\eta_{\cdot j}=(8,4,4,4)
\quad\Rightarrow\quad
c_{\cdot j}=(3,-1,-1,-1)
$$

이므로 component 간 posterior score 차이를 만든다.

### 2.3 entry-wise가 아니라 coordinate-wise group $L_2$ penalty를 사용한다

제안 penalty는 coordinate $j$의 centered contrast vector 전체를 하나의 단위로 선택한다.

$$
\lambda_\eta\sum_{j=1}^d\|c_{\cdot j}\|_2.
$$

이는

$$
\|c_{\cdot j}\|_2=0
$$

이면 coordinate $j$ 전체를 decision support에서 제외하고,

$$
\|c_{\cdot j}\|_2>0
$$

이면 해당 coordinate를 posterior decision에 사용하는 구조다.

대비되는 entry-wise penalty는

$$
\lambda_\eta\sum_{k,j}|c_{kj}|
$$

이다. 이 경우 같은 coordinate 안에서 일부 component contrast만 남거나 사라질 수 있다. 예를 들어

$$
c_{\cdot j}=(3,-1,-1,-1)
$$

은 하나의 decision coordinate가 만드는 contrast pattern이다. Group $L_2$ penalty는 이 좌표 전체를 선택 또는 제외한다.

### 2.4 adaptive weight를 사용한다

Adaptive Eta-group penalty는

$$
\lambda_\eta
\sum_{j=1}^d
w_j\|c_{\cdot j}\|_2
$$

로 둔다. 초기 추정값 $c_{\cdot j}^{init}$에 대해

$$
w_j
=
\left(\|c_{\cdot j}^{init}\|_2+\epsilon\right)^{-\gamma}
$$

를 사용한다. 이번 시뮬레이션에서는

$$
\gamma=1,\qquad \epsilon=10^{-6}
$$

로 두고, weight는 median-normalization을 적용했다.

초기 contrast가 큰 좌표는 작은 weight를 받아 상대적으로 약하게 축소되고, 초기 contrast가 작은 좌표는 큰 weight를 받아 더 강하게 축소된다. 예를 들어

$$
\|c_{\cdot j}^{init}\|_2=10
\Rightarrow
w_j\approx0.1,
\qquad
\|c_{\cdot j'}^{init}\|_2=0.5
\Rightarrow
w_{j'}\approx2.
$$

### 2.5 관련 penalty 문헌과 본 연구의 위치

Guo et al. (2010)은 model-based clustering에서 cluster center의 pairwise difference에 fusion penalty를 두어 cluster pair별 변수 선택을 다루었다.

$$
\lambda
\sum_{j=1}^p
\sum_{k<\ell}
w_{k\ell j}
|\mu_{kj}-\mu_{\ell j}|.
$$

Bondell and Reich (2009)는 ANOVA에서 level effect의 pairwise difference를 shrink하여 level collapsing과 factor selection을 동시에 다루었다.

$$
\sum_j
\sum_{k<m}
w_j^{(km)}
|\beta_{jk}-\beta_{jm}|,
\qquad
\sum_k\beta_{jk}=0.
$$

두 문헌은 절대 크기보다 집단 간 차이를 기준으로 변수 선택을 구성한다는 점에서 본 연구의 centered contrast penalty와 연결된다. 본 연구에서는 이 아이디어를 vMF mixture의 posterior decision score에 맞추어, $\eta=\kappa\mu$의 centered contrast $c_{kj}$에 coordinate-wise group penalty를 둔다.

### 2.6 Eta penalty ablation 진단

이 절은 S1-S6 성능 결과가 아니라, 제안 모형의 구조를 분해한 ablation 진단이다. eta 자연모수, group penalty, centered contrast의 역할을 분리해 확인한다.

진단 시뮬레이션 환경:

| 항목 | 설정 |
|:---|:---|
| 목적 | penalty target, centering, group/adaptive 효과 분해 |
| 반복 수 | rep=20 |
| 기준 환경 | S1 data-generating setting |
| 차원/표본/군집 수 | $d=200$, $n=1000$, $K=4$ |
| 변수 구성 | common q=4, decision/specific q=16, noise q=180, true decision q=16 |
| 평균 방향 차이 | target angle 90도 |
| 집중도 | $\kappa=(30,40,50,60)$ |
| 선택 기준 | BIC |
| 재적합 | 선택된 support에서 refit |

| 비교 목적 | method | penalty / model | selected q | common q | specific q | noise q | ARI | TPR | FPR | Precision | F1 | MSE_eta | 해석 |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| $\mu$ entry-wise | M-L + refit | $\lambda_\mu\sum_{k,j}\lvert\mu_{kj}\rvert$ | 40.65 | 4.00 | 16.00 | 20.65 | 0.853 | 1.000 | 0.134 | 0.399 | 0.568 | 0.232 | common/noise 좌표 선택 많음 |
| $\mu$ group | M-GL + refit | $\lambda_\mu\sum_j\lVert\mu_{\cdot j}\rVert_2$ | 20.00 | 4.00 | 16.00 | 0.00 | 0.859 | 1.000 | 0.022 | 0.800 | 0.889 | 0.072 | noise는 줄지만 common 좌표 유지 |
| adaptive $\mu$ group | M-AGL + refit | $\lambda_\mu\sum_j w_j^{(M)}\lVert\mu_{\cdot j}\rVert_2$ | 20.00 | 4.00 | 16.00 | 0.00 | 0.859 | 1.000 | 0.022 | 0.800 | 0.889 | 0.072 | $\mu$-support 기준 |
| raw $\eta$ entry-wise | E-L + refit | $\lambda_\eta\sum_{k,j}\lvert\eta_{kj}\rvert$ | 43.65 | 4.00 | 16.00 | 23.65 | 0.853 | 1.000 | 0.150 | 0.378 | 0.545 | 0.254 | raw eta L1은 noise 선택 증가 |
| raw $\eta$ group | E-GL + refit | $\lambda_\eta\sum_j\lVert\eta_{\cdot j}\rVert_2$ | 21.15 | 4.00 | 16.00 | 1.15 | 0.858 | 1.000 | 0.028 | 0.759 | 0.862 | 0.089 | common 좌표 유지 |
| centered $\eta$ entry-wise | E-CL + refit | $\lambda_\eta\sum_{k,j}\lvert c_{kj}\rvert$ | 19.05 | 0.05 | 16.00 | 3.00 | 0.860 | 1.000 | 0.017 | 0.846 | 0.915 | 0.098 | centering은 common 선택을 크게 줄임 |
| centered $\eta$ group | E-CGL + refit | $\lambda_\eta\sum_j\lVert c_{\cdot j}\rVert_2$ | 17.50 | 0.00 | 16.00 | 1.50 | 0.861 | 1.000 | 0.008 | 0.922 | 0.958 | 0.079 | centered eta group |
| adaptive centered $\eta$ group | E-CAGL + refit | $\lambda_\eta\sum_j w_j^{(E)}\lVert c_{\cdot j}\rVert_2$ | 16.05 | 0.00 | 16.00 | 0.05 | 0.860 | 1.000 | 0.000 | 0.997 | 0.998 | 0.057 | true decision q=16 근처 |

`MSE_eta`는 `MSE_centered_eta`를 줄여 쓴 표기다. 이 표는 S1 환경에서 실행한 rep=20 diagnostic 결과이며, S1-S6 본 결과가 아니라 구조 분해 결과로 해석한다. 여기서 $c_{kj}=\eta_{kj}-\bar\eta_j$이고, true decision q는 16이다. `common q`, `specific q`, `noise q`는 각각 common 좌표 4개, decision/specific 좌표 16개, noise 좌표 180개 중 선택된 평균 개수다.

- M-GL/M-AGL은 noise를 제거하지만 $\mu$-support 기준이므로 S1의 common q=4를 그대로 선택한다.
- raw eta 계열 E-L/E-GL도 common 좌표를 유지한다. eta 자연모수만으로는 posterior decision support가 충분히 분리되지 않았다.
- E-CL은 common q를 거의 제거하지만 entry-wise penalty라 noise q가 남는다.
- E-CGL/E-CAGL은 centered eta contrast를 group 단위로 선택한다. E-CAGL은 selected q=16.05, common q=0.00, noise q=0.05로 S1의 decision support와 거의 일치했다.

표에서 제외한 진단 후보:

| 제외 후보 | 형식 | 제외 사유 |
|:---|:---|:---|
| M-CGL | $\lambda_\mu\sum_j\lVert\mu_{\cdot j}-\bar\mu_j\mathbf{1}\rVert_2$ | $\mu$는 단위 구면 제약을 갖는 방향 모수이고 posterior score에는 $\eta_k=\kappa_k\mu_k$가 들어간다. 따라서 $\mu$만 중심화하면 $\kappa$ 차이를 반영하지 못해 posterior decision support 목표와 일치하지 않는다. S1 보조 diagnostic에서도 selected q=21.00, common q=4.00, noise q=1.00으로 common 좌표를 유지했다. |

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
| S1 | 평균 차이 큼(90도), 집중도 이분산 | ARI=0.865, selected q=16.06, F1=0.998, MSE_eta=0.057 | Dense vMF ARI=0.836 | E-CAGL selected q가 true q=16 근처 |
| S2 | 평균 차이 큼(90도), 집중도 등분산 | ARI=0.904, selected q=16.12, F1=0.996, MSE_eta=0.057 | Dense vMF ARI=0.880 | 집중도 차이가 없는 조건에서도 support recovery 유지 |
| S3 | 평균 차이 보통(60도), 집중도 이분산 | ARI=0.631, selected q=21.22, F1=0.881, MSE_eta=0.250 | Dense vMF ARI=0.539 | 중간 난도 setting. support recovery 일부 저하 |
| S4 | 평균 차이 보통(60도), 집중도 등분산 | ARI=0.651, selected q=16.32, F1=0.990, MSE_eta=0.079 | Dense vMF ARI=0.561 | selected q가 true q=16 근처 |
| S5 | 평균 차이 작음(30도), 약한 집중도 이분산 | ARI=0.015, selected q=0.02, F1=0.118, MSE_eta=1.040 | Dense vMF ARI=0.029 | weak-signal stress-test. zero-support 쪽으로 수축 |
| S6 | 평균 차이 작음(30도), 집중도 등분산 | ARI=0.012, selected q=0.56, F1=0.105, MSE_eta=2.354 | Dense vMF ARI=0.011 | weak-signal setting. 모든 방법의 성능이 낮음 |

S1-S4에서는 E-CAGL이 clustering 성능을 유지하면서 selected q를 true q=16 근처로 맞췄다. M-L과 E-CL은 ARI는 유지했지만 selected q가 거의 전체 차원에 가까웠다. S5-S6은 weak-signal limitation으로 분리해 해석한다.

### 3.2 Dense-support negative-control S1-N~S6-N

Negative-control은 평균 방향 차이와 집중도 차이의 축은 유지하되, decision q를 16에서 80으로 늘린 dense decision-support setting이다. sparse support 가정 밖에서 Eta-group의 동작을 확인하기 위한 설정이다.

| Scenario | 설정 | E-CAGL 핵심 결과 | 외부 baseline 결과 | 해석 |
|:---|:---|:---|:---|:---|
| S1-N | 평균 차이 큼(90도), 집중도 이분산 | ARI=0.857, selected q=82.40, F1=0.985 | Dense vMF ARI=0.835 | dense support에서도 true q=80 근처 선택 |
| S2-N | 평균 차이 큼(90도), 집중도 등분산 | ARI=0.897, selected q=81.82, F1=0.989 | Dense vMF ARI=0.886 | dense support에서도 true q=80 근처 선택 |
| S3-N | 평균 차이 보통(60도), 집중도 이분산 | ARI=0.565, selected q=76.06, F1=0.840 | Dense vMF ARI=0.545 | M-AGL보다 support F1이 낮아지는 limitation |
| S4-N | 평균 차이 보통(60도), 집중도 등분산 | ARI=0.629, selected q=16.70, F1=0.979 | Dense vMF ARI=0.562 | E-CAGL이 decision q=80 중 일부만 선택. BIC 튜닝 한계 |
| S5-N | 평균 차이 작음(30도), 약한 집중도 이분산 | ARI=0.001, selected q=0.06, F1=0.025 | Dense vMF ARI=0.024 | weak signal에서 전체적으로 성능 저하 |
| S6-N | 평균 차이 작음(30도), 집중도 등분산 | ARI=0.005, selected q=2.04, F1=0.172 | Dense vMF ARI=0.012 | dense/sparse 여부와 무관하게 signal 자체가 약함 |

S1-N과 S2-N에서는 E-CAGL의 selected q가 true q=80 근처였다. 반면 S3-N과 S4-N에서는 평균 방향 차이가 보통 수준일 때 Eta-group 계열의 과소선택 또는 BIC 튜닝 한계가 관찰됐다. S5-N/S6-N은 signal 자체가 약한 stress-test로 해석한다.

### 3.3 외부 baseline 해석

외부 baseline은 제안 모형과 목표가 다르므로 ARI/NMI/purity 중심으로만 비교한다.

- Spherical k-means와 Dense vMF free kappa는 clustering-only baseline이며 sparse support를 제공하지 않는다.
- Sparse k-means는 feature support를 선택하지만 posterior decision support와는 목표가 다르다.
- Rossi 2022 setting은 $\mu_k$ prototype sparsity 비교용이다. `d=100`에서 sparsity 5%, 10%, 15%이면 common-like coordinate 기대값이 약 81개, 66개, 52개로, 공통/공유 변수가 많은 구조다. dbmovMFs는 현재 로컬 R 환경에 없어 제외했다.

## 4. 제안 모형의 비용과 불리한 조건

제안 모형 E-CAGL의 비용은 계산 시간뿐 아니라 튜닝과 해석 구조가 더 복잡하다는 점에 있다. S1 1회 실행 시간 benchmark 기준(`K=4`, `n=1000`, `d=200`, `nstart=10`, path=240, Rcpp helper ON)에서 E-CAGL은 M-L보다 느렸지만 M-GL/M-AGL보다는 빠르게 나왔다.

| 비교 | 1회 시간 | 해석 |
|:---|---:|:---|
| M-L | 3.75 sec | 계산 시간은 짧지만 selected q=200으로 support recovery에는 한계 |
| M-GL / M-AGL | 8.82 / 8.53 sec | group/adaptive group penalty로 계산량 증가 |
| E-CGL / E-CAGL | 5.62 / 5.53 sec | M-L보다 약 1.5배 느리고 selected q는 true q=16 근처 |

E-CAGL의 주요 비용과 제약은 다음과 같다.

| 비용 또는 불리한 조건 | 내용 |
|:---|:---|
| 계산 비용 | M-L보다 약 1.47배 느림 |
| 튜닝 비용 | eta path, adaptive weight, BIC 선택에 민감 |
| weak signal | S5/S6처럼 signal이 약한 조건에서 zero-support 또는 과소선택 관찰 |
| dense decision support | S3-N/S4-N에서 과소선택 관찰 |
| prototype sparsity target | Rossi-style setting은 posterior decision support가 아니라 prototype sparsity가 목표 |

E-CAGL은 모든 조건에서 계산 시간 또는 ARI 기준의 우위 모형은 아니다. 본 결과에서는 계산/튜닝 비용을 수반하며, sparse posterior decision support recovery 지표가 높게 나타났다.
