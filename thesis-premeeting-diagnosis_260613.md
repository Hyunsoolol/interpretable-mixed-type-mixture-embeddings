# 연구미팅 전 객관 진단 및 보완 계획

업데이트: 2026-06-13

검토한 파일과 결과는 다음이다.

| 구분 | 확인 대상 |
|:---|:---|
| 연구미팅 요약 | `thesis-meeting_260622.md` |
| 구현/추정 정리 | `thesis-implementation_260622.md` |
| 전체 simulation | `thesis-simulation_260611.md` |
| PBMC real data | `results/pbmc3k_lymphoid3_eta_nstart30_path260/` |
| PBMC 통계 baseline | `results/screen_pbmc3k_lymphoid3_hvg1000_seed20260613_nstart30_path220/` |
| BBC pairwise text | `results/text_main_benchmark_validate_260613/validate_bbc_pair_sport_entertainment_d1000_k2_seed20260614/` |
| BBC K=3 text | `results/text_main_bbc_tri_sport_entertainment_tech_k3_260613/validate_bbc_tri_sport_entertainment_tech_d1000_k3_seed20260614/` |

## 1. 현재 연구의 핵심 기여

현재 연구의 핵심 기여는 비교적 명확하다.

Rossi & Barbaro (2022)의 sparse vMF mixture는 component direction `mu_k`에 penalty를 둔다. 하지만 vMF mixture의 posterior classification에는 `mu_k` 자체가 아니라 natural parameter

$$
\eta_k = \kappa_k \mu_k
$$

가 직접 들어간다. 두 component의 posterior log odds는 다음처럼 쓸 수 있다.

$$
\log\frac{\tau_{i2}}{\tau_{i1}}
= \text{constant} + (\eta_2-\eta_1)^T x_i .
$$

따라서 variable selection의 목표가 "component를 구분하는 coordinate"라면, `mu_k`의 sparsity보다 `eta_k` 또는 component 간 `eta` contrast의 sparsity를 보는 것이 더 직접적이다.

K개 component에서는 coordinate별 centered eta contrast를 사용한다.

$$
\bar{\eta}_j = K^{-1}\sum_{k=1}^K \eta_{kj},
\qquad
c_{kj} = \eta_{kj}-\bar{\eta}_j .
$$

제안 penalty는 다음 형태다.

$$
Q_{\eta}(\Theta)
= \ell(\Theta) - \lambda_{\eta}\sum_{j=1}^{d}\|c_{\cdot j}\|_2 .
$$

현재 논문 스토리는 다음 흐름으로 성립한다.

| 단계 | 메시지 |
|:---|:---|
| 기존 방법 | Rossi 방법은 sparse direction setting에서는 잘 작동한다 |
| 한계 상황 | 평균 방향 차이가 작고 concentration 차이가 중요한 경우 Rossi는 노이즈 변수를 많이 선택한다 |
| 제안 방법 | eta contrast penalty는 posterior decision에 직접 관련된 coordinate를 선택한다 |
| simulation | ARI는 유지하면서 selected q, FPR, Precision, F1을 크게 개선한다 |
| real data | PBMC에서는 vMF 계열 안에서 좋은 결과, BBC pairwise에서는 sparse text interpretation 가능 |

현재 결과만으로도 "논문 가능성"은 있다. 다만 "모든 real data에서 기존 방법보다 우월하다"는 스토리는 아직 성립하지 않는다. 더 안전한 주장은 다음이다.

> Eta contrast penalty는 concentration-dominant 또는 contrast-driven sparse structure에서 Rossi의 `mu`-penalty보다 해석 가능한 variable selection을 제공한다.

## 2. 현재 결과의 강점과 약점

### 2.1. Simulation 강점

가장 강한 결과는 공통 변수와 component-specific 변수가 함께 있는 K=4 setting이다.

```text
K = 4
n = 1000
d = 100
rep = 100
common variables = 6
component-specific variables = 4 per component, total 16
true union q = 22
w = 0.50
kappa = (30, 45, 65, 90)
```

| Method | ARI | True q | Selected q | TPR | FPR | Precision | F1 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Rossi | 0.680 | 22.000 | 98.520 | 1.000 | 0.981 | 0.223 | 0.365 |
| Separate penalty | 0.684 | 22.000 | 86.460 | 1.000 | 0.826 | 0.258 | 0.409 |
| Eta penalty + refit | 0.686 | 22.000 | 24.750 | 0.994 | 0.037 | 0.890 | 0.937 |

이 결과는 연구의 가장 강한 근거다. Rossi와 separate penalty는 true active coordinate를 찾지만 noise coordinate도 거의 모두 선택한다. Eta penalty + refit은 ARI를 유지하면서 selected q를 true union q에 가깝게 만든다.

약한 concentration 차이에서도 같은 메시지가 유지된다.

| Method | ARI | True q | Selected q | TPR | FPR | Precision | F1 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Rossi | 0.563 | 22.000 | 97.580 | 1.000 | 0.969 | 0.226 | 0.368 |
| Separate penalty | 0.567 | 22.000 | 78.150 | 1.000 | 0.720 | 0.294 | 0.450 |
| Eta penalty + refit | 0.575 | 22.000 | 23.980 | 1.000 | 0.025 | 0.921 | 0.958 |

### 2.2. Simulation 약점

현재 simulation에서 약한 부분은 세 가지다.

| 약점 | 의미 | 보완 방향 |
|:---|:---|:---|
| 고차원 d=500에서 selected q 증가 | BIC가 너무 느슨할 수 있음 | EBIC 또는 stability selection sensitivity를 추가 |
| 일부 MSE_kappa outlier | kappa approximation/EM 수치 안정성 이슈 가능 | median/IQR도 함께 보고, outlier 원인 점검 |
| exact M-step이 아님 | eta penalty는 proximal EM prototype | 논문에서는 "proximal update"로 명확히 표현 |

고차원 결과는 본문 주장의 중심으로 두기보다, "향후 tuning 개선이 필요한 영역"으로 배치하는 것이 안전하다.

## 3. Real Data 결과 진단

### 3.1. PBMC 3K lymphoid3

PBMC에서는 eta 방법이 vMF 계열 안에서 가장 좋다.

| Method | Criterion | ARI | NMI | Purity | Selected q | Kappa ratio |
|:---|:---|---:|---:|---:|---:|---:|
| Eta penalty | RICc | 0.984 | 0.971 | 0.995 | 438 | 1.057 |
| Eta penalty | EBIC | 0.981 | 0.966 | 0.994 | 589 | 1.071 |
| Eta penalty + refit | EBIC | 0.975 | 0.960 | 0.992 | 589 | 1.116 |
| Dense vMF free-kappa | NA | 0.972 | 0.956 | 0.991 | 1000 | 1.110 |
| Spherical k-means | NA | 0.970 | 0.953 | 0.990 | 1000 | NA |
| Rossi BIC | BIC | 0.965 | 0.946 | 0.989 | 1000 | 1.000 |

다만 통계 baseline까지 포함하면 sparse k-means가 더 높게 나온다.

| Baseline | ARI | NMI | Purity | Selected q |
|:---|---:|---:|---:|---:|
| Sparse k-means | 0.990 | 0.979 | 0.996 | 62 |
| Spectral clustering | 0.979 | 0.961 | 0.993 | NA |
| Spherical k-means | 0.970 | 0.953 | 0.990 | 1000 |
| mclust PCA20 | 0.644 | 0.670 | 0.824 | NA |

따라서 PBMC에서 가능한 주장은 다음이다.

> Eta penalty는 vMF mixture 계열에서는 가장 좋은 clustering 성능과 sparse marker selection을 제공한다. 다만 sparse k-means가 ARI만 보면 더 높으므로, "전체 baseline 중 최고"라고 주장하면 안 된다.

PBMC는 본문 real data로 쓸 수 있지만, label 정의와 marker 해석을 더 확인해야 한다.

### 3.2. BBC News pairwise: sport vs entertainment

pairwise text에서는 eta + refit BIC가 좋은 보조 결과다.

| Method | Criterion | ARI | NMI | Purity | Selected q | Kappa ratio |
|:---|:---|---:|---:|---:|---:|---:|
| Eta penalty + refit | BIC | 0.978 | 0.955 | 0.994 | 692 | 1.016 |
| Spherical k-means | NA | 0.973 | 0.941 | 0.993 | 1000 | NA |
| Eta penalty + refit | EBIC | 0.973 | 0.948 | 0.993 | 565 | 1.024 |
| Rossi BIC | BIC | 0.969 | 0.934 | 0.992 | 1000 | 1.000 |
| Dense vMF free-kappa | NA | 0.969 | 0.934 | 0.992 | 1000 | 1.004 |

이 결과는 "텍스트에서도 clustering 성능을 유지하면서 단어 수를 줄인다"는 보조 사례로 쓸 수 있다.

### 3.3. BBC News K=3: sport, entertainment, tech

K=3에서는 eta의 장점이 약하다.

| Method | Criterion | ARI | NMI | Purity | Selected q | Kappa ratio |
|:---|:---|---:|---:|---:|---:|---:|
| Spherical k-means | NA | 0.916 | 0.868 | 0.970 | 1000 | NA |
| Eta penalty + refit | BIC | 0.909 | 0.860 | 0.968 | 1000 | 1.008 |
| Rossi BIC | BIC | 0.902 | 0.853 | 0.966 | 1000 | 1.000 |
| Eta penalty + refit | EBIC | 0.605 | 0.647 | 0.791 | 465 | 2.341 |
| Eta penalty + refit | RICc | 0.510 | 0.563 | 0.759 | 307 | 2.775 |

BIC에서는 eta penalty가 사실상 0으로 선택되어 dense vMF와 비슷해진다. EBIC/RICc로 희소하게 만들면 selected q는 줄지만 clustering 성능이 크게 떨어진다.

따라서 BBC K=3은 본문 핵심 결과가 아니라 부록 또는 한계 분석에 적합하다.

## 4. 연구미팅에서 지적받을 가능성이 높은 부분

| 예상 질문 | 현재 위험도 | 답변 방향 |
|:---|:---:|:---|
| eta penalty의 이론적 정당화가 충분한가? | 높음 | posterior decision boundary와 natural parameter 관점으로 설명 |
| exact EM인가? | 높음 | exact M-step이 아니라 proximal penalized update라고 명확히 말해야 함 |
| tuning 기준이 공정한가? | 높음 | 공식 비교는 path 후보 + BIC, 고차원은 EBIC sensitivity 필요 |
| 왜 refit을 쓰는가? | 중간 | support 선택 후 shrinkage bias를 줄이는 post-selection refit |
| real data에서 항상 좋은가? | 높음 | 항상 좋은 것은 아니고, contrast-driven structure에서 강점이 있다고 정리 |
| sparse k-means보다 좋은가? | 높음 | PBMC에서 ARI는 sparse k-means가 높음. eta는 model-based vMF 해석성이 장점 |
| 텍스트 K=3에서 왜 약한가? | 중간 | 평균 방향만으로 잘 구분되는 multiclass에서는 eta contrast sparsity가 추가 이득을 주지 않음 |
| 고차원에서 d=500 결과가 약한가? | 중간 | BIC가 느슨하며 high-dimensional tuning 보완 필요 |

## 5. 연구미팅 전 보완 우선순위

### 5.1. 하루 안에 가능한 것

| 우선순위 | 작업 | 이유 |
|---:|:---|:---|
| 1 | 연구미팅 자료에 real data 결과를 "주장 범위"와 함께 추가 | 교수님이 논문 가능성을 판단하기 쉬움 |
| 2 | simulation 결과를 본문/부록으로 분리 | 좋은 결과와 stress 결과가 섞여 메시지가 흐려지는 것을 방지 |
| 3 | eta penalty가 exact EM이 아니라 proximal update임을 명확히 표현 | 구현 질문에 대한 취약점 선제 대응 |
| 4 | PBMC sparse k-means 우위와 BBC K=3 약점을 숨기지 않고 정리 | 과장된 주장으로 보이는 위험 감소 |

### 5.2. 2-3일 안에 가능한 것

| 작업 | 기대 결과 | 실패 시 해석 |
|:---|:---|:---|
| PBMC marker gene top list 정리 | eta가 해석 가능한 marker를 선택함을 보임 | marker overlap이 약하면 clustering 중심으로만 사용 |
| BBC pairwise top words 정리 | selected words가 class-specific임을 보임 | text는 부록으로만 이동 |
| EBIC/stability sensitivity 소규모 확인 | 고차원 selected q 문제 완화 가능성 확인 | tuning 개선 필요로 정직하게 보고 |
| MSE_kappa outlier 점검 | 모수 추정 결과 신뢰성 향상 | MSE는 보조 지표로 낮춤 |

### 5.3. 오래 걸리지만 논문에 중요한 것

| 작업 | 중요성 |
|:---|:---|
| Penalized objective의 monotonicity 또는 proximal EM 성질 정리 | 방법론 논문으로 설득력 증가 |
| Eta penalty의 support recovery 조건 또는 직관적 proposition | 통계학술지 투고 가능성 증가 |
| Real data benchmark 1개 추가 검증 | 실제 적용 가능성 강화 |
| Code reproducibility 정리 | CSDA/ADAC 투고 시 유리 |

### 5.4. 지금은 하지 않아도 되는 것

| 작업 | 이유 |
|:---|:---|
| 모든 real data를 새로 탐색 | 연구미팅 전에는 시간 대비 효율 낮음 |
| LLM embedding text 분석 | word-level variable selection 해석과 목적이 다름 |
| top-tier journal용 이론 완성 | 현재 연구미팅 전 목표와 범위가 다름 |
| TCGA/Zeisel full benchmark 대규모 실행 | 좋은 결과 보장이 낮고 시간이 큼 |

## 6. 연구미팅 자료 수정 방향

### 본문에 남길 결과

1. Rossi 2022 재현 결과
2. K=4 concentration-dominant stress setting
3. 공통 변수 + component-specific 변수 setting
4. 약한 concentration 차이 robustness
5. PBMC 3K lymphoid3 real data
6. BBC News sport vs entertainment pairwise

### 부록으로 뺄 결과

1. 고차원 d=500에서 selected q가 커지는 결과
2. fixed-kappa high-dimensional stress setting
3. BBC News K=3 결과
4. 여러 text 후보 탐색 실패 결과
5. 초기 prototype grid tuning 결과

### 삭제하거나 축소할 결과

1. 반복되는 K=2 작은 simulation 표
2. 같은 메시지를 주는 여러 concentration-dominant 변형
3. real data에서 성능이 좋지 않은 후보의 상세 표

### 교수님께 질문할 포인트

1. 논문의 핵심 기여를 eta contrast penalty로 잡아도 충분한지
2. exact EM이 아닌 proximal EM update를 현재 단계에서 어떻게 표현하는 것이 좋은지
3. 본문 real data를 PBMC 중심으로 두고 BBC pairwise를 보조로 두는 구성이 적절한지
4. 고차원 setting에서 BIC 대신 EBIC 또는 stability selection을 공식 기준으로 검토해야 하는지
5. sparse k-means가 PBMC에서 높은 ARI를 보일 때, eta vMF의 장점을 어떻게 서술하는 것이 좋은지

## 7. 가장 먼저 해야 할 작업 3개

### 1순위: 연구미팅 자료의 주장 범위 수정

현재 문서의 simulation 메시지는 강하지만, real data와 한계가 빠져 있어 논문 가능성 판단이 어려울 수 있다. "좋은 결과"와 "아직 약한 부분"을 함께 정리해야 한다.

좋은 결과가 나오면:

* 교수님이 논문 스토리를 빠르게 이해할 수 있다.
* 추가 실험보다 논문 구조 논의로 넘어갈 수 있다.

실패하면:

* 연구가 약하다는 뜻이 아니라, 아직 claim이 너무 넓다는 뜻이다.
* claim을 "contrast-driven sparse variable selection"으로 좁히면 된다.

### 2순위: Real data 결과표 정리

PBMC와 BBC pairwise는 쓸 수 있다. 다만 sparse k-means와 BBC K=3 결과를 같이 보고해야 한다.

좋은 결과가 나오면:

* PBMC는 본문 real data, BBC는 보조 real data로 정리 가능하다.

실패하면:

* real data는 "illustration"으로 낮추고 simulation 중심 논문으로 구성한다.

### 3순위: Implementation 표현 정리

eta penalty update는 exact closed-form EM이라고 말하면 안 된다. proximal update라고 명확히 표현해야 한다.

좋은 결과가 나오면:

* 구현 질문에 방어 가능하다.

실패하면:

* 교수님께 이론 보강 방향을 직접 질문해야 한다.

## 8. 연구미팅 전 보완 체크리스트

| 완료 | 작업 |
|:---:|:---|
|  | `thesis-meeting_260622.md`에 real data 요약 섹션 추가 |
|  | PBMC 결과표에 sparse k-means baseline을 함께 제시 |
|  | BBC pairwise와 BBC K=3을 구분해서 정리 |
|  | simulation 본문에서 가장 강한 setting 2-3개만 남기기 |
|  | 고차원/실패성 결과는 부록 또는 한계로 이동 |
|  | eta penalty update를 proximal EM으로 표현 |
|  | refit의 역할을 "support 선택 후 post-selection unpenalized estimation"으로 정리 |
|  | 교수님께 물어볼 질문 5개를 마지막에 정리 |
|  | 깃 업데이트 전 변경 내용 diff 확인 |

## 9. 현재 판단

현재 연구는 "망한 방향"이 아니다. 오히려 simulation에서는 논문 스토리가 꽤 선명하다. 다만 지금 부족해 보이는 이유는 결과가 약해서라기보다, 결과의 종류가 많아져서 핵심 메시지가 흐려졌기 때문이다.

가장 안전한 논문 방향은 다음이다.

> Rossi의 `mu`-sparsity는 sparse prototype 해석에는 유용하지만, posterior classification 관점에서는 `eta = kappa mu`가 직접적인 decision parameter다. 평균 방향 차이가 작고 concentration 또는 eta contrast가 중요한 상황에서는 centered eta contrast penalty가 noise variable selection을 줄이고 해석 가능한 sparse contrast를 제공한다.

이 방향이면 simulation 결과가 충분히 받쳐준다. Real data는 PBMC를 중심으로 쓰고, BBC pairwise는 보조 사례로 두는 것이 좋다. BBC K=3과 고차원 d=500 문제는 숨기지 말고 한계 또는 tuning 보완 필요성으로 정리하는 편이 더 학술적으로 안전하다.
