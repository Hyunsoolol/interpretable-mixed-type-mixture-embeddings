# Thesis Meeting 260708

## 1. 이번 미팅 목적

2026-06-24 연구미팅에서 받은 두 가지 피드백에 대해 현재 답변과 후속 확인 결과를 정리한다.

1. $\mu_k$, $\kappa_k$, $\eta_k=\kappa_k\mu_k$의 의존성과 유일성
2. Rossi / Separate penalty 대비 Eta-group penalty가 불리한 상황

이번 문서는 full simulation 보고서가 아니라, 피드백에 대한 핵심 답변과 다음 실험 방향을 확인받기 위한 미팅 자료다.

## 2. Feedback 1: $\eta$, $\mu$, $\kappa$ 관계

### 2.1 핵심 답변

vMF mixture에서 posterior decision score에 직접 들어가는 것은 $\mu_k$ 단독이 아니라 자연모수

$$
\eta_k=\kappa_k\mu_k
$$

이다. 여기서 $\mu_k$는 방향, $\kappa_k$는 그 방향의 concentration 또는 decision strength를 나타낸다. 따라서 두 모수는 해석과 최적화에서 $\eta_k$를 통해 결합되어 작동한다.

Posterior score에는 다음 항이 들어간다.

$$
\log \alpha_k+\log C_d(\|\eta_k\|_2)+\eta_k^\top x_i.
$$

$\eta_k\ne0$이고 $\kappa_k>0$이면 단위 구면 제약 $\|\mu_k\|_2=1$ 때문에

$$
\kappa_k=\|\eta_k\|_2,\qquad
\mu_k=\eta_k/\|\eta_k\|_2
$$

로 복원된다. 즉 component-level parameterization에서는 $\eta_k$가 $\mu_k$와 $\kappa_k$를 유일하게 정한다. 다만 이것은 mixture 전체의 전역 식별성 증명이 아니라, label switching과 near-empty component 문제는 별도로 남는다.

### 2.2 예시 1: 같은 방향, 다른 집중도

두 component의 평균 방향이 같다고 하자.

$$
\mu_1=(1,0,0),\quad \kappa_1=2,\qquad
\mu_2=(1,0,0),\quad \kappa_2=10.
$$

그러면 자연모수는

$$
\eta_1=(2,0,0),\qquad \eta_2=(10,0,0)
$$

가 된다. $\mu$만 보면 두 component는 같은 방향을 갖지만, $\eta$로 보면 posterior decision strength가 다르다. concentration 차이가 $\eta$의 크기로 직접 반영되기 때문이다.

### 2.3 예시 2: $\eta$에서 $\mu,\kappa$ 복원

예를 들어

$$
\eta=(3,4,0)
$$

이면

$$
\kappa=\|\eta\|_2=5,\qquad
\mu=(3/5,4/5,0)
$$

이다. 이때 $\|\mu\|_2=1$이므로 구면 제약을 만족하는 분해가 하나로 정해진다.

### 2.4 예외

$\eta=(0,0,0)$이면 $\kappa=0$이고, 이때 $\mu$ 방향은 식별되지 않는다. 또한 mixture model에서는 label switching 문제가 별도로 남아 있으므로, 본 연구에서는 유일성을 component-level parameterization 설명으로 제한한다.

## 3. Feedback 1 추가: proximal EM-type update와 단조증가

본 연구의 추정은 closed-form M-step이 아니라 proximal EM-type update다. vMF normalizing constant와 centered eta group lasso penalty 때문에 penalized M-step을 한 번에 닫힌형태로 풀기 어렵다.

따라서 각 iteration에서는 현재 위치에서 proximal shrinkage candidate를 만들고 objective를 확인한다. 보폭이 너무 크면 objective가 감소할 수 있으므로, 구현에서는 objective decrease가 생길 때 step size를 줄이는 step-halving line search를 사용한다.

정리하면 이 부분은 "자동 단조증가 정리"가 아니라 optimization safeguard로 설명하는 것이 안전하다.

### 3.1 추정 대상에 따른 failure mode

Eta-group은 posterior decision score에 직접 들어가는 $\eta_k=\kappa_k\mu_k$의 component contrast를 penalize한다. 따라서 $\eta$ contrast가 약하거나 tuning이 너무 강하면, 실제 decision에 필요한 좌표까지 shrink될 수 있다. 이 경우 복원된 $\mu$와 $\kappa$도 함께 영향을 받는다. 즉 Eta-group의 장점은 decision parameter를 직접 선택한다는 점이지만, 그만큼 $\eta$를 과소선택하면 clustering과 모수 해석이 동시에 나빠질 수 있다.

반대로 Rossi / Separate는 $\mu$와 $\kappa$를 나누어 다룬다고 해서 자동으로 더 안정적이라는 뜻은 아니다. 다만 weak signal에서는 BIC가 거의 full support를 선택하면서 clustering signal을 보존하는 쪽으로 움직일 수 있다. 이 경우 ARI는 Eta-group보다 덜 나빠질 수 있지만, selected q와 FPR이 매우 커져 sparse support 해석성은 약해진다.

따라서 두 방법의 차이는 "어느 쪽이 일괄적으로 안정적인가"가 아니라 failure mode의 차이로 설명하는 것이 안전하다.

| 방법 | 주된 위험 | 관찰된 패턴 | 해석 |
|:---|:---|:---|:---|
| Eta-group | $\eta$ contrast 과소선택 | Weak-signal tuning diagnostic에서 평균 selected q=2.68, 50회 중 43회 q=0 | sparse하지만 decision signal까지 잃을 수 있음 |
| Rossi / Separate | 과대선택 또는 dense support | Weak-signal tuning diagnostic에서 selected q가 약 98-100 | clustering signal은 남기지만 support 해석성이 약함 |

## 4. Feedback 2: Eta-group이 불리한 상황

Eta-group의 핵심 장점은 universal ARI improvement가 아니라 posterior decision parameter 기반의 sparse support 해석성이다. 따라서 signal이 약하거나, true support가 dense하거나, path/tuning이 부족하거나, real-data representation이 맞지 않으면 Eta-group도 불리해질 수 있다.

| Setting | 불리한 지표 | Eta-group | Rossi / Separate | 해석 |
|:---|:---|---:|---:|:---|
| K=4 strong, refit 전 | ARI | 0.625 | Rossi 0.680 / Separate 0.684 | penalty shrinkage가 refit 전 clustering을 낮출 수 있음 |
| $w=0.35$ | ARI | 0.505 | Separate 0.528 | signal 약화 시 ARI 손실 가능 |
| $w=0.25$ | $\kappa$ MSE | 약 $4.999\times10^9$ | Separate 60.743 | concentration instability 가능 |
| d=200 basic | selected q / FPR | 120.06 / 0.552 | true q=22 | sparse recovery 불충분 |
| d=400 basic | selected q / FPR | 262.95 / 0.642 | true q=22 | high-dimensional limitation |
| d=400 path+adaptive | selected q / FPR | 308.00 / 0.760 | long path Eta q=68.75 / FPR=0.146 | adaptive penalty는 official로 부적절 |
| SPLADE BBC5 | ARI / selected q | 0.817 / 500 | Rossi 0.857 / 500 | harder real-data benchmark에서는 Eta가 불리 |
| SPLADE 20NG4 | ARI | 0.461 | Rossi 0.715 | sparse tuning 시 clustering 붕괴 |

## 5. Negative-control simulation 업데이트

### 5.1 Dense-support negative control

Dense-support negative control은 현재까지 가장 명확한 Eta-group limitation이다. True decision support가 dense한 setting에서 Eta-group + refit은 selected q=52.82로 support를 줄였지만 ARI=0.368, F1=0.726으로 Rossi/Separate보다 낮았다. Separate + refit은 ARI=0.378, F1=0.890이고 MSE_centered_eta도 더 낮았다.

해석: 많은 coordinate가 약하게 separation에 기여하는 경우에는 Eta-group이 과도하게 shrink하여 TPR/F1, ARI, MSE_centered_eta에서 손해를 볼 수 있다.

### 5.2 Weak-signal tuning diagnostic

Weak-signal tuning diagnostic은 weak signal을 완화한 rep50 diagnostic이다. Rossi/Separate의 ARI는 약 0.13-0.14 수준이지만 support는 거의 full support로 선택된다. Eta-group BIC는 평균 selected q=2.68이고, 50회 중 43회에서 selected q=0을 선택해 BIC-selected refit valid replicate가 7회뿐이었다. Positive-support diagnostic을 쓰면 selected q=15.52, F1=0.531이지만 ARI=0.137로 낮다.

해석: 이 setting은 Rossi/Separate의 깨끗한 우위라기보다 Eta BIC zero-support failure와 weak-signal clustering difficulty를 보여주는 diagnostic에 가깝다.

### 5.3 Support-target diagnostics

이 계열은 “어떤 support를 recovery target으로 둘 것인가”를 확인하기 위한 diagnostic이다.

- Direction-sparse metric diagnostic: equal concentration과 direction-sparse 구조에서 Rossi/Separate는 ARI=0.999였지만 selected q=100, FPR=1.000으로 거의 모든 coordinate를 선택했다. Eta-group은 ARI=0.998을 유지하면서 selected q=40.60, F1=0.658로 union-support 기준에서는 더 sparse했다. 다만 Rossi/Separate는 entry-level prototype support에서 entry_TPR=1.000을 보였다.
- Entry-sparse / union-dense diagnostic: $d=100$, true union q=80, 각 component가 서로 다른 20개 좌표를 사용한다. 모든 방법의 ARI가 0.999였고, union support에서는 Eta-group이 selected q=91.00, F1=0.936으로 더 좋았다. 반면 prototype entry support에서는 Separate BIC의 entry_F1=0.438로 Rossi보다 높아, metric 분리가 필요함을 보였다.
- Fragmented-support diagnostic: 공유 좌표가 없는 구조에서도 현재 generator만으로는 Rossi/Separate가 명확히 유리한 결과를 만들지 못했다. 저차원에서는 Eta-group이 selected q=42.80으로 더 sparse했고, 고차원에서는 Rossi/Separate도 selected q=400, Eta-group도 selected q=368.33으로 모두 dense해졌다.

따라서 현재 generator만으로는 "Rossi/Separate가 명확하게 유리한 block-diagonal setting"이 아직 만들어지지 않았다. 이 질문에 답하려면 dedicated block-diagonal 또는 binary-style generator와 prototype-support metric이 필요하다.

자세한 표는 [negative_control_summary_260708.md](../../results/negative_control_summary_260708/negative_control_summary_260708.md)에 정리했다.

### 5.4 Support metric 정리

Rossi/Separate와 Eta-group은 sparse하게 만들고자 하는 대상이 다르므로, support recovery를 하나의 지표로만 비교하면 해석이 섞일 수 있다.

| Metric | 정의 | 주된 의미 | 한계 |
|:---|:---|:---|:---|
| Coordinate union support | $S_{\mathrm{union}}=\{j:\exists k,\ active_{kj}\}$ | coordinate-level variable selection. 모든 방법에 공통으로 계산 가능 | Rossi/Separate의 component별 sparsity 구조를 하나로 합친다 |
| Prototype entry support | $S_{\mathrm{entry}}=\{(k,j):\mu_{kj}\ne0\}$ | Rossi/Separate처럼 direction/prototype sparsity를 목표로 하는 방법에 자연스러움 | Eta-group은 coordinate-level centered eta group penalty라서 같은 방식의 직접 비교가 어렵다 |
| Posterior decision support | $S_{\eta}=\{j:\|c_{\cdot j}\|_ 2>0\}$, $c_{kj}=\eta_{kj}-K^{-1}\sum_\ell \eta_{\ell j}$ | posterior decision boundary에 들어가는 coordinate. Eta-group의 main claim에 가장 적합 | Rossi/Separate의 prototype sparsity와는 목표가 다르다 |

Entry-sparse / union-dense diagnostic에서는 모든 방법의 ARI가 0.999로 거의 같았다. Coordinate union support 기준에서는 Eta-group이 selected q=91.00, F1=0.936으로 Rossi/Separate보다 좋았고, prototype entry support 기준에서는 Separate BIC가 entry_F1=0.438로 Rossi BIC의 0.399보다 높았다. 따라서 이 결과의 핵심은 특정 방법의 전체 우위가 아니라, 논문에서 어떤 support target을 main claim으로 둘 것인지의 문제다.

### 5.5 Negative-control 핵심 정리

| Diagnostic | 현재 결론 | 미팅에서 확인할 점 |
|:---|:---|:---|
| Dense-support negative control | dense true support에서 Eta-group이 support를 과도하게 줄이며 Separate + refit보다 ARI/F1/MSE_centered_eta가 낮다. | 가장 명확한 Eta-group limitation으로 둘 수 있는가? |
| Weak-signal tuning diagnostic | weak signal에서 Eta BIC가 zero support를 자주 선택한다. 50회 중 43회가 selected q=0이고 BIC-selected refit valid replicate는 7회다. | tuning failure diagnostic으로 appendix에 둘 것인가? |
| Support-target diagnostics | 현재 generator만으로는 Rossi/Separate가 명확히 유리한 setting이 만들어지지 않았다. 다만 entry support와 union support가 다른 결론을 줄 수 있음을 보였다. | prototype support metric과 block-diagonal generator를 추가할 것인가? |


## 6. 현재 결론

- $\eta_k=\kappa_k\mu_k$는 $\kappa_k>0$에서 $\mu_k$와 $\kappa_k$를 복원할 수 있는 natural decision parameter다.
- 이 유일성은 mixture 전체 식별성 증명이 아니라 component-level parameterization 설명이다.
- Eta-group은 strong sparse support setting에서는 설득력 있지만, Rossi / Separate보다 일괄적으로 좋은 방법은 아니다.
- Dense true support setting에서는 Eta-group이 과도하게 shrink하여 ARI/F1과 MSE_centered_eta에서 손해를 볼 수 있다.
- 다음 단계는 prototype support recovery와 posterior decision support recovery를 분리하고, 필요하면 block-diagonal generator를 별도로 설계하는 것이다.
