# Thesis Meeting 260708

## 1. 이번 미팅 목적

2026-06-24 연구미팅에서 받은 두 가지 피드백에 대해 현재 답변과 다음 확인 계획을 정리한다.

1. $\mu_k$, $\kappa_k$, $\eta_k=\kappa_k\mu_k$의 의존성과 유일성
2. Rossi / Separate penalty 대비 Eta-group penalty가 불리한 상황

이번 문서는 새 full simulation 결과 보고가 아니라, 피드백에 대한 이론적 답변과 negative-control simulation 설계를 확인받기 위한 초안이다.

## 2. Feedback 1: $\mu$, $\kappa$, $\eta$ 의존성과 유일성

### 2.1 핵심 답변

vMF mixture에서 posterior decision score에 직접 들어가는 것은 $\mu_k$ 단독이 아니라 자연모수

$$
\eta_k=\kappa_k\mu_k
$$

이다. 여기서 $\mu_k$는 방향, $\kappa_k$는 그 방향의 concentration 또는 decision strength를 나타낸다. 따라서 두 모수는 최적화와 해석에서 분리되어 작동하기보다 $\eta_k$를 통해 결합되어 작동한다.

Posterior score는 다음 항을 포함한다.

$$
\log \alpha_k+\log C_d(\|\eta_k\|_2)+\eta_k^\top x_i.
$$

$\eta_k\ne 0$이면 $\eta_k$에서 원래 vMF parameter를 다음처럼 복원할 수 있다.

$$
\kappa_k=\|\eta_k\|_2,\qquad
\mu_k=\eta_k/\|\eta_k\|_2.
$$

따라서 단일 component 수준에서는 $\kappa_k>0$ 조건 아래 $\eta_k$ 표현이 $(\mu_k,\kappa_k)$와 one-to-one이다. 다만 이것은 전역 식별성 증명이 아니라, parameterization-level uniqueness로 정리하는 것이 안전하다.

### 2.2 직관적 예시 1: 같은 방향, 다른 집중도

두 component의 평균 방향이 같다고 하자.

$$
\mu_1=(1,0,0),\qquad \kappa_1=2
$$

$$
\mu_2=(1,0,0),\qquad \kappa_2=10.
$$

그러면 자연모수는

$$
\eta_1=(2,0,0),\qquad \eta_2=(10,0,0)
$$

가 된다. $\mu$만 보면 두 component는 같은 방향을 갖기 때문에 구분이 약해 보인다. 그러나 $\eta$로 보면 두 component의 posterior decision strength가 다르다. 즉 concentration 차이는 $\eta$의 크기로 직접 반영된다.

이 예시는 Eta-group penalty가 $\mu$가 아니라 $\eta$ contrast를 보는 이유를 보여준다. posterior decision에는 direction 자체보다 direction과 concentration이 결합된 $\eta$가 들어간다.

### 2.3 직관적 예시 2: 같은 $\eta$에서 $\mu,\kappa$ 복원

예를 들어

$$
\eta=(3,4,0)
$$

이면

$$
\kappa=\|\eta\|_2=5,\qquad
\mu=(3/5,4/5,0)
$$

이다. 즉 $\eta\ne 0$이면 방향과 크기가 유일하게 정해진다.

### 2.4 직관적 예시 3: 유일성이 깨지는 예외

반대로

$$
\eta=(0,0,0)
$$

이면

$$
\kappa=0
$$

이고, 이때 $\mu$ 방향은 어떤 방향이어도 같은 균일분포에 가까운 component를 만든다. 따라서 $\eta=0$ 또는 $\kappa=0$에서는 $\mu$ 방향이 식별되지 않는다.

추가로 mixture model에서는 label switching이 남아 있고, near-empty component나 component collapse에서는 수치적으로 안정적인 유일성 주장을 하기 어렵다.

### 2.5 논문에 반영할 표현

안전한 표현:

- Under $\kappa_k>0$, the natural parameter $\eta_k$ uniquely determines $\kappa_k$ and $\mu_k$ up to the usual mixture label switching.
- Eta-group penalizes component contrasts in the posterior decision parameter, not the direction vector alone.
- The uniqueness claim is at the parameterization level, not a full mixture identifiability theorem.

피해야 할 표현:

- 전역 식별성이 증명되었다.
- $\eta$ parameterization이 모든 식별성 문제를 해결한다.
- proximal EM-type update가 정확한 EM 알고리즘이다.

### 2.6 예상 질문에 대한 짧은 답변

예상 질문은 다음과 같다.

> $\eta_k$를 추정한 뒤 $\eta_k=\kappa_k\mu_k$로 쪼갤 때, 예를 들어 $4=2\times2=1\times4$처럼 여러 해가 생기지 않는가?

답변은 단위 구면 제약 때문에 해가 하나로 고정된다는 것이다. 일반적인 벡터 분해라면 여러 스케일 조합이 가능하지만, vMF에서는 반드시

$$
\|\mu_k\|_2=1
$$

이어야 한다. 따라서 $\eta_k\ne0$이면

$$
\kappa_k=\|\eta_k\|_2,\qquad
\mu_k=\eta_k/\|\eta_k\|_2
$$

로만 분해된다.

예를 들어 $\eta=(3,4)$이면 $\|\eta\|_2=5$이므로 $\kappa=5$, $\mu=(3/5,4/5)=(0.6,0.8)$이다. 이때 $\|\mu\|_2=1$이므로 구면 제약을 만족하는 유일한 분해가 된다.

직관적으로 $\eta$가 "동북쪽으로 5만큼 간 위치"라면, $\mu$는 길이 1로 고정된 나침반 방향이고 $\kappa$는 그 방향으로 간 거리다. 나침반 화살표의 길이가 1로 고정되어 있으므로 목적지 $\eta$가 주어지면 방향과 거리가 모호하지 않다.

## 3. Proximal EM-type update와 목적함수 단조증가

### 3.1 왜 자동 단조증가를 주장하지 않는가?

일반 EM은 M-step에서 Q-function을 정확히 최대화하면 observed likelihood가 단조 증가한다. 그러나 본 연구의 update는 정확한 EM update가 아니라 proximal EM-type update다. 이유는 다음과 같다.

1. vMF likelihood에는 Bessel normalizing constant가 들어가서 M-step이 단순 closed-form maximization이 아니다.
2. centered eta group lasso penalty는 nonsmooth penalty이므로 한 번에 정확한 최대점을 구하는 구조가 아니다.
3. 실제 구현에서는 현재 위치에서 proximal shrinkage candidate를 만든 뒤 objective를 확인한다.

따라서 update step이 너무 크면 목적함수가 감소할 수 있다. 직관적으로는 산 정상 방향으로 걷는 중 보폭이 너무 커서 정상을 지나 반대편 내리막으로 넘어가는 overshooting과 같다.

### 3.2 구현상 safeguard: step-halving line search

이 문제를 막기 위해 구현에는 step-halving line search를 넣었다. candidate update에서 penalized objective가 감소하면 step size를 절반으로 줄이고 다시 평가한다. 이 과정은 objective decrease를 막기 위한 numerical safeguard다.

안전한 표현:

- The update is a proximal EM-type update, not a closed-form exact M-step update.
- Monotone increase is not automatic from EM theory because the penalized M-step is not solved exactly.
- A step-halving line search is used as a safeguard against objective decrease.

피해야 할 표현:

- 정확한 EM이다.
- 전역 최적점을 보장한다.
- 단조 수렴이 이론적으로 완전히 증명되었다.

## 4. Feedback 2: Eta-group이 불리한 상황

### 4.1 기존 결과에서 이미 확인된 불리한 상황

Eta-group은 항상 Rossi / Separate보다 좋은 방법이 아니다. 기존 결과에서 확인된 불리한 상황은 다음과 같다.

| Setting | 불리한 지표 | Eta-group | Rossi / Separate | 해석 |
|:---|:---|---:|---:|:---|
| K=4 strong, refit 전 | ARI | 0.625 | Rossi 0.680 / Separate 0.684 | penalty shrinkage가 refit 전 clustering을 낮출 수 있음 |
| $w=0.35$ | ARI | 0.505 | Separate 0.528 | signal 약화 시 ARI 손실 가능 |
| $w=0.25$ | $\kappa$ MSE | 약 $4.999\times 10^9$ | Separate 60.743 | concentration instability 확인 필요 |
| d=200 basic | selected q / FPR | 120.06 / 0.552 | true q=22 | sparse recovery 불충분 |
| d=400 basic | selected q / FPR | 262.95 / 0.642 | true q=22 | high-dimensional limitation |
| d=400 path+adaptive | selected q / FPR | 308.00 / 0.760 | long path Eta q=68.75 / FPR=0.146 | adaptive penalty는 official로 부적절 |
| SPLADE BBC5 | ARI / selected q | 0.817 / 500 | Rossi 0.857 / 500 | harder real-data benchmark에서는 Eta가 불리 |
| SPLADE 20NG4 | ARI | 0.461 | Rossi 0.715 | sparse tuning 시 clustering 붕괴 |

해석은 명확하다. Eta-group의 핵심 장점은 universal ARI improvement가 아니라, posterior decision parameter 기반의 sparse support 해석성이다. 따라서 signal이 약하거나, true support가 dense하거나, path/tuning이 부족하거나, real-data representation이 맞지 않으면 Eta-group도 불리해질 수 있다.

## 5. Rossi / Separate가 유리할 수 있는 새 simulation 설계

이번 후속 simulation의 목적은 Eta-group을 더 좋게 보이는 setting을 찾는 것이 아니라, Rossi / Separate가 유리할 수 있는 환경을 의도적으로 만들어 방법의 적용 범위를 확인하는 것이다.

### 5.1 Setting A: direction-sparse, concentration-homogeneous

목적: Rossi가 유리할 수 있는 환경을 확인한다. component 차이가 주로 $\mu$의 sparse direction 차이로 발생하고, $\kappa$는 모든 component에서 동일하거나 거의 동일한 경우다.

설계:

| 항목 | 값 |
|:---|:---|
| K | 4 |
| n | 1000 |
| d | 100 |
| true active q | 20 |
| direction structure | component별 서로 다른 sparse $\mu_k$ pattern |
| concentration | $\kappa=(60,60,60,60)$ 또는 $(55,60,65,60)$ |
| signal structure | component-specific direction difference가 명확함 |

예상:

- $\kappa$가 거의 같으면 $\eta$ contrast는 사실상 $\mu$ contrast와 비슷하다.
- Rossi의 $\mu$ penalty가 Eta-group과 비슷하거나 더 유리할 수 있다.
- Eta-group의 concentration-aware 장점이 크게 필요하지 않은 setting이다.

주요 비교 지표:

- ARI
- selected q
- FPR, Precision, F1
- MSE_mu
- MSE_centered_eta

### 5.2 Setting B: dense eta contrast / weak sparsity truth

목적: Eta-group의 sparse support assumption이 틀린 환경을 확인한다. 실제로 많은 coordinate가 약하게 cluster separation에 기여하는 경우다.

설계:

| 항목 | 값 |
|:---|:---|
| K | 4 |
| n | 1000 |
| d | 100 |
| true active q | 70 또는 80 |
| signal pattern | 많은 coordinate에 작은 effect가 분산됨 |
| concentration | $\kappa=(30,45,65,90)$ |
| expected structure | sparse recovery보다 dense directional fit이 유리할 수 있음 |

예상:

- Eta-group이 과도하게 shrink하면 ARI 또는 MSE_centered_eta가 나빠질 수 있다.
- Rossi / Separate의 dense selection이 clustering에는 더 유리할 수 있다.
- support recovery metric은 Eta-group에 불리하게 설계되므로, ARI와 parameter MSE도 함께 봐야 한다.

주요 비교 지표:

- ARI loss
- selected q
- MSE_centered_eta
- support F1
- dense support가 실제 truth와 얼마나 맞는지

### 5.3 Setting C: weak signal / low concentration instability

목적: 기존 $w=0.25$에서 보인 $\kappa$ instability를 의도적으로 확인한다. Eta-group이 concentration-sensitive하게 실패할 수 있는 상황을 점검한다.

설계:

| 항목 | 값 |
|:---|:---|
| K | 4 |
| n | 1000 |
| d | 100 |
| true active q | 22 |
| own-specific loading | $w=0.20$ 또는 $w=0.25$ |
| concentration | $\kappa=(25,30,35,40)$ 또는 작은 concentration gap |
| expected difficulty | cluster separation이 약함 |

예상:

- Eta-group의 $\eta$ norm 또는 $\kappa$ 추정이 불안정해질 수 있다.
- Separate penalty가 $\mu$와 $\kappa$를 따로 조정하기 때문에 더 안정적일 가능성이 있다.
- objective trace와 $\kappa$ upper-bound 근처 추정 비율을 같이 확인해야 한다.

주요 비교 지표:

- ARI
- MSE_kappa
- kappa_hat_mean
- kappa ratio
- MSE_centered_eta
- objective trace

## 6. Smoke simulation plan

full simulation 전에 smoke만 먼저 확인한다. 이번 문서에서는 실행하지 않고, 실행 계획만 제안한다.

| Setting | Label | rep | nstart | max_iter | 목적 |
|:---|:---|---:|---:|---:|:---|
| A | `negative_control_direction_sparse_smoke_260708` | 5 | 3 또는 5 | 80 | Rossi-favorable direction-sparse 구조 확인 |
| B | `negative_control_dense_eta_smoke_260708` | 5 | 3 또는 5 | 80 | dense truth에서 Eta shrinkage 손해 확인 |
| C | `negative_control_weak_signal_smoke_260708` | 5 | 3 또는 5 | 80 | low signal / concentration instability 확인 |

실행 전 확인할 것:

1. 기존 `k4_specific_effect_run.r`에서 환경변수만으로 Setting A/B/C를 만들 수 있는지 확인한다.
2. 환경변수만으로 어렵다면 R core algorithm은 건드리지 않고 별도 simulation wrapper를 설계한다.
3. smoke에서 ERROR row, zero support, dense support, objective decrease, $\kappa$ blow-up 여부를 먼저 확인한다.
4. smoke가 정상일 때만 rep=50 또는 rep=100 full diagnostic을 결정한다.

## 7. Negative-control 실행 업데이트

Setting B는 rep=50까지 실행했다. dense true support setting에서 Eta-group + refit은 selected q=52.82로 support를 줄였지만 ARI=0.368, F1=0.726으로 Rossi/Separate보다 낮았다. Separate + refit은 ARI=0.378, F1=0.890이고, MSE_centered_eta도 Eta-group보다 낮았다. 따라서 Setting B는 Eta-group이 불리한 negative-control로 적합하다.

Setting C2는 weak signal을 완화해 smoke로 실행했다. Rossi/Separate의 ARI는 약 0.13-0.15 수준으로 살아났지만, Eta-group BIC는 selected q=0을 선택해 refit이 invalid가 되었다. positive-support diagnostic을 쓰면 selected q=15.80, F1=0.577이지만 ARI=0.136으로 낮다. 이 setting은 Rossi/Separate 우위라기보다 Eta BIC zero-support failure diagnostic에 가깝다.

Setting A는 재설계가 필요하다. 현재 coordinate union support metric은 decision-support sparsity를 보므로 Eta-group에 유리하다. Rossi가 유리한 환경을 보려면 prototype/direction support recovery와 decision support recovery를 분리해서 평가해야 한다.

Setting A2는 equal concentration과 direction-sparse 구조로 smoke 실행했다. Rossi/Separate는 ARI=0.999로 clustering은 거의 완벽했지만 selected q=100, FPR=1.000으로 거의 모든 coordinate를 선택했다. Eta-group은 ARI=0.998을 유지하면서 selected q=40.60, F1=0.658로 union-support 기준에서는 여전히 더 sparse했다. 다만 Rossi/Separate는 entry-level prototype support에서 entry_TPR=1.000을 보였으므로, Rossi류 방법을 공정하게 평가하려면 prototype support metric과 posterior decision support metric을 분리해야 한다.

추가로 공유 좌표가 없는 fragmented block-like setting도 smoke로 확인했다. 저차원 설정($d=60$, true union q=40)에서는 모든 방법이 ARI가 거의 1이었고, Eta-group은 selected q=42.80으로 더 sparse했지만 Separate는 MSE_mu와 MSE_centered_eta가 더 낮았다. 고차원 설정($d=400$, true union q=80)에서는 Rossi/Separate도 selected q=400으로 dense해졌고 Eta-group도 selected q=368.33으로 dense했다. 따라서 현재 generator만으로는 "Rossi/Separate가 압도적으로 유리한 block-diagonal setting"이 아직 만들어지지 않았고, 이를 보려면 dedicated block-diagonal 또는 binary-style generator와 prototype-support metric이 필요하다.

자세한 표는 `results/negative_control_summary_260708/negative_control_summary_260708.md`에 정리했다.

## 8. 260708 미팅에서 교수님께 확인할 질문

1. $\eta$ 유일성 설명을 parameterization-level result로 Methods에 넣는 것이 충분한가?
2. negative-control simulation을 direction-sparse, dense-eta, weak-signal 세 축으로 설계하는 것이 적절한가?
3. Eta-group main claim을 strong sparse support recovery 중심으로 제한해도 되는가?
4. high-dimensional과 hard real-data benchmark는 limitation 또는 appendix로 두는 것이 좋은가?

## 9. 다음 작업 TODO

| Priority | 작업 | 목적 |
|:---|:---|:---|
| P0 | 260708 문서 검토 | 교수님 피드백 답변 구조 확정 |
| P0 | negative-control simulation script 가능성 확인 | 환경변수만으로 구현 가능한지 판단 |
| P1 | Setting B negative-control 정리 | dense true support에서 Eta-group 한계 명시 |
| P1 | prototype support metric 설계 | Rossi/Separate의 자연스러운 목표를 별도로 평가 |
| P1 | block-diagonal generator 설계 | Rossi/Separate가 구조적으로 유리한 setting을 분리해서 확인 |
| P2 | methods note에 uniqueness 문단 반영 | 피드백 1 논문화 |
| P2 | simulation note에 failure-mode section 추가 | 피드백 2 논문화 |

## 10. 현재 결론

260624 피드백에 대한 현재 답변은 다음과 같다.

- $\eta_k=\kappa_k\mu_k$는 $\kappa_k>0$에서 $\mu_k$와 $\kappa_k$를 유일하게 복원할 수 있는 natural decision parameter다.
- 그러나 이 유일성은 전역 식별성 증명이 아니라 parameterization-level statement다.
- Eta-group은 strong sparse support setting에서는 설득력 있지만, 항상 Rossi / Separate보다 좋은 것은 아니다.
- dense true support setting에서는 Eta-group이 과도하게 shrink하여 ARI/F1과 MSE_centered_eta에서 손해를 볼 수 있다.
- A2 smoke에서도 union-support metric은 Eta-group에 유리하게 작동했다. 따라서 다음 단계는 prototype support recovery와 posterior decision support recovery를 분리하는 평가 지표를 만드는 것이다.
- fragmented block-like smoke는 현재 generator만으로는 Rossi/Separate 압도 우위를 만들지 못했다. 해당 질문에 답하려면 별도 block-diagonal generator가 필요하다.
