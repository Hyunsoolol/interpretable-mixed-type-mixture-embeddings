# Thesis Meeting 260708

## 1. 이번 미팅 목적

이번 문서는 2026-06-24 연구미팅에서 받은 피드백에 대한 답변을 정리하고, 다음 실험 방향을 확인받기 위한 자료다. 핵심은 다음 세 가지다.

- 피드백 1: $\eta$, $\mu$, $\kappa$의 의존성과 유일성
- 피드백 2: Eta-group penalty가 불리한 상황
- 추가 정리: proximal EM-type update와 단조증가 claim의 범위

전체 결과표는 [negative_control_summary_260708.md](../../results/negative_control_summary_260708/negative_control_summary_260708.md)에 따로 정리하였다.

## 2. 6월 24일 피드백에 대한 답변

### 2.1 의존성과 유일성

vMF mixture에서 posterior decision score에 직접 들어가는 자연모수는

$$
\eta_k=\kappa_k\mu_k
$$

이다. 여기서 $\mu_k$는 평균 방향, $\kappa_k$는 그 방향의 집중도 또는 decision strength를 나타낸다. Posterior score에는 다음 항이 들어간다.

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

### 2.2 Eta-group penalty가 불리한 상황

Eta-group의 핵심 장점은 ARI를 일괄적으로 높이는 것이 아니라, posterior decision parameter에 들어가는 coordinate support를 sparse하게 해석하는 데 있다. 따라서 true decision support가 조밀하거나, 신호가 약하거나, 평가하려는 support target이 prototype sparsity인 경우에는 Eta-group이 불리하거나 해석이 섞일 수 있다.

#### 2.2.0 한눈에 보는 결과 요약

| 진단 | 어떤 상황인가 | Eta-group 결과 | 비교 대상 결과 | 차이 | 결론 |
|:---|:---|:---|:---|:---|:---|
| 조밀 support 음성대조 | true union q=80으로 decision support가 조밀함 | Eta + refit: ARI=0.368, q=52.82, F1=0.726, MSE_eta=2.721 | Separate + refit: ARI=0.378, q=99.74, F1=0.890, MSE_eta=2.150 | Eta가 q를 약 47개 적게 선택하지만 F1 -0.164, MSE_eta +0.571 | support를 과도하게 줄여 손해를 보는 가장 명확한 limitation |
| 약한 신호 튜닝 실패 | true union q=22, weak signal | Eta BIC: 평균 q=2.68, 50회 중 43회 q=0; Eta positive-support + refit: ARI=0.137, q=15.52, F1=0.531 | Rossi/Separate: ARI 약 0.13-0.14, q 약 98-100 | Eta BIC는 zero support로 무너지고, Rossi/Separate는 거의 full support로 감 | clean Rossi 우위라기보다 tuning failure diagnostic |
| support 목표 차이 | union support와 prototype entry support가 서로 다른 목표 | union 기준: q=91.00, F1=0.936 | Separate entry_F1=0.438, Rossi entry_F1=0.399 | Eta는 decision support에서 유리하지만 prototype entry support와는 목표가 다름 | main claim은 posterior decision support recovery로 제한 |
| 분절 support 진단 | 공유 좌표 없는 fragmented/block-like 구조 | d=60: q=42.80, F1=0.967; d=400: q=368.33, F1=0.357 | d=60 Separate: q=56.80, F1=0.827; d=400 Rossi/Separate: q=400, F1=0.333 | 현재 generator로는 Rossi/Separate가 명확히 유리한 구조가 안 나옴 | dedicated block-diagonal generator 필요 |

#### 2.2.1 조밀 support 음성대조

이 설정은 “진짜 필요한 decision coordinate가 많은 경우”를 보는 음성대조다.

| 항목 | 내용 |
|:---|:---|
| 상황 | 많은 coordinate가 약하게 함께 separation에 기여함 |
| 환경 | $K=4$, $n=1000$, $d=100$, rep=50 |
| True support | true union q=80, common q=20, specific q/component=15 |
| 집중도 | $\kappa=(30,45,65,90)$ |
| 선택 기준 | BIC |

- Eta-group은 selected q=52.82로 support를 강하게 줄였지만, F1과 MSE_eta에서 Separate + refit보다 나빠졌다.
- 해석: 필요한 좌표가 많이 있는 상황에서는 group penalty가 필요한 coordinate까지 줄일 수 있다.
- 용도: Eta-group의 가장 명확한 limitation / negative-control 결과.

#### 2.2.2 약한 신호 튜닝 실패 진단

이 설정은 weak signal에서 BIC가 Eta-group support를 너무 강하게 줄일 수 있음을 보여준다.

| 항목 | 내용 |
|:---|:---|
| 상황 | 신호가 약해 support 선택과 clustering이 모두 어려운 경우 |
| 환경 | $K=4$, $n=1000$, $d=100$, rep=50 |
| True support | true union $q=22$, common $q=6$, specific $q$/component=4 |
| 집중도 | $\kappa=(35,45,55,65)$ |
| 선택 기준 | BIC |

- Eta BIC는 평균 q=2.68만 선택했고, 50회 중 43회에서 q=0을 선택했다.
- Positive-support diagnostic은 q=15.52까지 회복하지만 ARI=0.137로 낮다.
- 해석: Rossi/Separate가 좋은 결과를 낸다기보다, Eta BIC의 zero-support tuning failure를 보여주는 결과다.

#### 2.2.3 support 목표 차이 진단

이 설정들은 “어떤 support를 맞히는 것이 목표인가”에 따라 결론이 달라질 수 있음을 보여준다.

| 진단 | 상황 | 결과 | 해석 |
|:---|:---|:---|:---|
| 방향 희소성 지표 진단 | equal concentration과 direction-sparse 구조 | Rossi/Separate ARI=0.999, selected q=100, FPR=1.000; Eta-group ARI=0.998, selected q=40.60, F1=0.658 | coordinate union support 기준과 prototype entry support 기준의 결론이 다를 수 있음 |
| 성분별 희소 / 합집합 조밀 진단 | 각 component가 서로 다른 active coordinate를 갖는 구조 | 모든 방법 ARI=0.999; Eta-group selected q=91.00, F1=0.936; Separate BIC entry_F1=0.438, Rossi BIC entry_F1=0.399 | union support에서는 Eta-group이 좋아 보이지만, prototype entry support에서는 Separate가 더 좋아 보임 |
| 분절 support 진단 | 공유 좌표가 없는 fragmented 구조 | 현재 generator로는 Rossi/Separate가 명확히 유리한 block-diagonal setting을 만들지 못함 | dedicated block-diagonal 또는 binary-style generator가 필요 |

- Eta-group은 coordinate-level posterior decision support를 직접 선택한다.
- Rossi/Separate는 prototype 또는 component-entry sparsity 관점에서 따로 평가할 필요가 있다.
- 따라서 본문 claim은 posterior decision support recovery로 제한하는 것이 안전하다.

#### 2.2.4 support metric 정리

| Metric | 정의 | 주된 의미 | 한계 |
|:---|:---|:---|:---|
| Coordinate union support | $S_{\mathrm{union}}=\{j:\exists k,\ active_{kj}\}$ | coordinate-level variable selection. 모든 방법에 공통으로 계산 가능 | Rossi/Separate의 component별 sparsity 구조를 하나로 합친다 |
| Prototype entry support | $S_{\mathrm{entry}}=\{(k,j):\mu_{kj}\ne0\}$ | Rossi/Separate처럼 direction/prototype sparsity를 목표로 하는 방법에 자연스러움 | Eta-group은 coordinate-level centered eta group penalty라서 같은 방식의 직접 비교가 어렵다 |
| Posterior decision support | $S_{\eta}=\{j:\|c_{\cdot j}\| _ 2>0\}$, $c_{kj}=\eta_{kj}-K^{-1}\sum_\ell \eta_{\ell j}$ | posterior decision boundary에 들어가는 coordinate. Eta-group의 main claim에 가장 적합 | Rossi/Separate의 prototype sparsity와는 목표가 다르다 |

논문 main claim은 prototype sparsity가 아니라 posterior decision support recovery로 두는 것이 안전하다. Rossi/Separate와의 공정 비교에는 prototype entry support를 보조 지표로 추가하는 것이 필요하다.

> 핵심 결론. Eta-group이 불리한 경우는 크게 두 가지다. 첫째, true decision support가 조밀하면 penalty가 필요한 좌표까지 줄여 F1과 MSE가 나빠질 수 있다. 둘째, weak signal에서는 BIC가 zero support를 선택해 tuning failure가 생길 수 있다. 따라서 논문 claim은 ARI 향상이 아니라 posterior decision support recovery로 제한하는 것이 안전하다.

### 2.3 proximal EM-type update와 단조증가

본 방법의 추정은 닫힌형 M-step이 아니라 proximal EM-type update다. vMF normalizing constant와 centered eta group penalty 때문에 penalized M-step을 한 번에 닫힌형태로 풀기 어렵다.

따라서 각 iteration에서는 현재 위치에서 proximal shrinkage candidate를 만들고 objective를 확인한다. 보폭이 너무 크면 objective가 감소할 수 있으므로, 구현에서는 objective decrease가 생길 때 step size를 줄이는 step-halving line search를 사용한다.

이 부분은 자동 단조증가 정리나 전역 수렴 보장이 아니라 optimization safeguard로 설명하는 것이 안전하다.

## 3. 현재 결론

- $\eta_k=\kappa_k\mu_k$는 posterior decision score에 직접 들어가는 자연모수다.
- $\eta_k\ne0$이고 $\kappa_k>0$이면 $\mu_k$와 $\kappa_k$는 component-level에서 유일하게 복원된다.
- Eta-group은 일괄적으로 더 좋은 방법이 아니라 posterior decision support recovery에 강점이 있는 방법이다.
- 조밀 support 또는 약한 신호에서는 Eta-group이 불리하거나 BIC tuning failure가 생길 수 있다.
- 다음 단계는 support target을 분리하고, 필요하면 block-diagonal negative-control generator를 별도로 설계하는 것이다.
