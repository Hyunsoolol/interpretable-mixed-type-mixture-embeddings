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

$\eta=(0,0,0)$이면 $\kappa=0$이고, 이때 $\mu$ 방향은 식별되지 않는다. 또한 mixture model에서는 label switching이 항상 남아 있으므로, 본 연구에서는 유일성을 component-level parameterization 설명으로 제한한다.

## 3. Feedback 1 추가: proximal EM-type update와 단조증가

본 연구의 추정은 closed-form M-step이 아니라 proximal EM-type update다. vMF normalizing constant와 centered eta group lasso penalty 때문에 penalized M-step을 한 번에 닫힌형태로 풀기 어렵다.

따라서 각 iteration에서는 현재 위치에서 proximal shrinkage candidate를 만들고 objective를 확인한다. 보폭이 너무 크면 objective가 감소할 수 있으므로, 구현에서는 objective decrease가 생길 때 step size를 줄이는 step-halving line search를 사용한다.

정리하면 이 부분은 "자동 단조증가 정리"가 아니라 optimization safeguard로 설명하는 것이 안전하다.

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

### 5.1 Setting B: dense true support

Setting B는 현재까지 가장 명확한 Eta-group failure mode다. Dense true support setting에서 Eta-group + refit은 selected q=52.82로 support를 줄였지만 ARI=0.368, F1=0.726으로 Rossi/Separate보다 낮았다. Separate + refit은 ARI=0.378, F1=0.890이고 MSE_centered_eta도 더 낮았다.

해석: 많은 coordinate가 약하게 separation에 기여하는 경우에는 Eta-group이 과도하게 shrink하여 TPR/F1, ARI, MSE_centered_eta에서 손해를 볼 수 있다.

### 5.2 A2: direction-sparse / equal concentration

A2는 equal concentration과 direction-sparse 구조로 smoke 실행했다. Rossi/Separate는 ARI=0.999로 clustering은 거의 완벽했지만 selected q=100, FPR=1.000으로 거의 모든 coordinate를 선택했다. Eta-group은 ARI=0.998을 유지하면서 selected q=40.60, F1=0.658로 union-support 기준에서는 더 sparse했다.

다만 Rossi/Separate는 entry-level prototype support에서 entry_TPR=1.000을 보였다. 따라서 Rossi류 방법을 공정하게 평가하려면 prototype support metric과 posterior decision support metric을 분리해야 한다.

### 5.3 Fragmented block-like smoke

공유 좌표가 없는 fragmented block-like setting도 smoke로 확인했다.

- 저차원 설정: $d=60$, true union q=40. 모든 방법의 ARI가 거의 1이었다. Eta-group은 selected q=42.80으로 더 sparse했지만, Separate는 MSE_mu와 MSE_centered_eta가 더 낮았다.
- 고차원 설정: $d=400$, true union q=80. Rossi/Separate도 selected q=400으로 dense해졌고, Eta-group도 selected q=368.33으로 dense했다.

따라서 현재 generator만으로는 "Rossi/Separate가 압도적으로 유리한 block-diagonal setting"이 아직 만들어지지 않았다. 이 질문에 답하려면 dedicated block-diagonal 또는 binary-style generator와 prototype-support metric이 필요하다.

### 5.4 C2: weak signal

C2는 weak signal을 완화한 smoke다. Rossi/Separate의 ARI는 약 0.13-0.15 수준으로 살아났지만, Eta-group BIC는 selected q=0을 선택해 refit이 invalid가 되었다. Positive-support diagnostic을 쓰면 selected q=15.80, F1=0.577이지만 ARI=0.136으로 낮다.

해석: 이 setting은 Rossi/Separate 우위라기보다 Eta BIC zero-support failure diagnostic에 가깝다.

자세한 표는 [negative_control_summary_260708.md](../../results/negative_control_summary_260708/negative_control_summary_260708.md)에 정리했다.

## 6. 교수님께 확인할 질문

1. $\eta$ 유일성 설명을 component-level parameterization result로 Methods에 넣는 것이 충분한가?
2. Eta-group main claim을 "ARI 향상"이 아니라 sparse posterior decision support recovery로 제한해도 되는가?
3. dense true support setting을 negative-control limitation으로 본문 또는 appendix에 넣을지?
4. Rossi/Separate와 공정 비교를 위해 prototype support metric과 block-diagonal generator를 추가 설계할지?

## 7. 다음 작업

| Priority | 작업 | 목적 |
|:---|:---|:---|
| P0 | 260708 미팅에서 claim 범위 확인 | Eta-group의 주장 수준 확정 |
| P0 | prototype support metric 정의 | Rossi/Separate의 자연스러운 목표를 별도 평가 |
| P1 | block-diagonal generator 설계 | Rossi/Separate가 구조적으로 유리한 setting 확인 |
| P1 | Setting B negative-control 정리 | dense true support에서 Eta-group 한계 명시 |
| P2 | methods note에 parameterization 설명 반영 | 피드백 1 논문화 |
| P2 | simulation note에 failure-mode section 추가 | 피드백 2 논문화 |

## 8. 현재 결론

- $\eta_k=\kappa_k\mu_k$는 $\kappa_k>0$에서 $\mu_k$와 $\kappa_k$를 복원할 수 있는 natural decision parameter다.
- 이 유일성은 mixture 전체 식별성 증명이 아니라 component-level parameterization 설명이다.
- Eta-group은 strong sparse support setting에서는 설득력 있지만, 항상 Rossi / Separate보다 좋은 것은 아니다.
- Dense true support setting에서는 Eta-group이 과도하게 shrink하여 ARI/F1과 MSE_centered_eta에서 손해를 볼 수 있다.
- 다음 단계는 prototype support recovery와 posterior decision support recovery를 분리하고, 필요하면 block-diagonal generator를 별도로 설계하는 것이다.
