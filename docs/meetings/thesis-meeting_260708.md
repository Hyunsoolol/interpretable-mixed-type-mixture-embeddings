# Thesis Meeting 260708

## 1. 이번 미팅 목적

이번 문서는 2026-06-24 연구미팅에서 받은 피드백에 대한 답변을 정리하고, 다음 실험 방향을 확인받기 위한 자료다. 핵심은 다음 세 가지다.

- 피드백 1: $\eta$, $\mu$, $\kappa$의 의존성과 유일성
- 피드백 2: Eta-group penalty가 불리한 상황
- 추가 정리: Eta penalty ablation 결과와 optimization safeguard 설명

전체 결과표는 [negative_control_summary_260708.md](../../results/negative_control_summary_260708/negative_control_summary_260708.md)에 따로 정리하였다.

## 2. 의존성과 유일성

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

### 2.1 왜 Eta-group인가?

vMF mixture의 posterior classification score는 $\mu$와 $\kappa$가 따로 작동하는 구조가 아니다. 실제 score에는

$$
\eta_k^\top x_i=(\kappa_k\mu_k)^\top x_i
$$

형태의 내적이 들어간다. 따라서 decision boundary에 직접 들어가는 parameter는 $\mu$ 단독이나 $\kappa$ 단독이 아니라 자연모수(natural parameter)

$$
\eta_k=\kappa_k\mu_k
$$

이다.

Clustering에서 중요한 것은 각 component의 절대적 $\eta_k$가 아니라, component 사이에서 어떤 coordinate가 posterior score 차이를 만드는가이다. 그래서 centered eta contrast

$$
c_{kj}=\eta_{kj}-\bar{\eta}_j,\qquad
\bar{\eta}_j=K^{-1}\sum_{\ell=1}^K\eta_{\ell j}
$$

를 본다. Eta-group penalty

$$
\lambda\sum_{j=1}^d \|c_{\cdot j}\|_2
$$

는 coordinate $j$가 component 간 posterior decision boundary를 만드는지 직접 선택하는 penalty다.

즉, Eta-group은 단순한 sparsity penalty가 아니라 posterior decision score에 들어가는 centered natural parameter contrast를 coordinate 단위로 선택하는 penalty다.

$\mu$만 penalize하면 concentration $\kappa$ 차이를 반영하지 못한다. 반대로 $\kappa$만 보면 coordinate-level 해석이 불가능하다. 또한 raw eta 또는 entrywise L1 penalty는 coordinate 전체가 decision에 필요한지보다 component-entry 단위의 작은 차이를 남기기 쉬워 dense support로 갈 수 있다.

따라서 현재 claim은 일괄적 우월성이 아니라, posterior decision support recovery라는 목표에는 centered eta contrast와 coordinate group penalty가 가장 직접적인 구조라는 것이다.

## 3. Eta-group penalty가 불리한 상황

Eta-group은 posterior decision support recovery에 강점이 있지만, 모든 sparsity 구조에서 유리한 것은 아니다. 음성대조 진단에서는 아래 세 가지 한계를 확인했다. 세부 수치와 전체 표는 [negative_control_summary_260708.md](../../results/negative_control_summary_260708/negative_control_summary_260708.md)에 남긴다.

### 3.1 핵심 결과

| 상황 | Eta-group에서 나타난 문제 | 비교 기준 | 핵심 차이 | 해석 |
|:---|:---|:---|:---|:---|
| 조밀 support | 필요한 좌표까지 축소 | Separate + refit | Eta F1=0.726, MSE_eta=2.721 vs Separate F1=0.890, MSE_eta=2.150 | true decision support가 조밀하면 Eta-group이 과소선택할 수 있음 |
| 약한 신호 | BIC가 zero support를 반복 선택 | Rossi/Separate | Eta BIC 평균 selected q=2.68, q=0이 43/50회 | weak signal에서는 BIC tuning failure 가능 |
| support 목표 차이 | union support와 entry support의 결론이 다름 | Rossi/Separate | Eta union F1=0.936, Separate entry_F1=0.438, Rossi entry_F1=0.399 | 어떤 support를 목표로 둘지 분리해야 함 |

### 3.2 해석

- Eta-group의 claim은 ARI의 일괄적 향상이 아니라 posterior decision support recovery다.
- true decision support가 조밀하면 group penalty가 필요한 좌표까지 줄일 수 있다.
- weak signal에서는 BIC가 너무 강하게 작동해 zero support를 선택할 수 있다.
- Rossi/Separate는 dense support를 선택해 ARI를 유지할 수 있지만, sparse support 해석성은 약해질 수 있다.
- 따라서 main result는 positive setting 중심으로 두고, 위 결과는 limitation 또는 appendix diagnostic으로 정리하는 것이 안전하다.

### 3.3 support metric 정리

| Metric | 정의 | 주된 의미 | 한계 |
|:---|:---|:---|:---|
| Coordinate union support | $S_{\mathrm{union}}=\{j:\exists k,\ active_{kj}\}$ | coordinate-level variable selection. 모든 방법에 공통으로 계산 가능 | Rossi/Separate의 component별 sparsity 구조를 하나로 합친다 |
| Prototype entry support | $S_{\mathrm{entry}}=\{(k,j):\mu_{kj}\ne0\}$ | Rossi/Separate처럼 direction/prototype sparsity를 목표로 하는 방법에 자연스러움 | Eta-group은 coordinate-level centered eta group penalty라서 같은 방식의 직접 비교가 어렵다 |
| Posterior decision support | $S_{\eta}=\{j:\|c_{\cdot j}\|_2>0\}$, $c_{kj}=\eta_{kj}-K^{-1}\sum_\ell \eta_{\ell j}$ | posterior decision boundary에 들어가는 coordinate. Eta-group의 main claim에 가장 적합 | Rossi/Separate의 prototype sparsity와는 목표가 다르다 |

논문 main claim은 prototype sparsity가 아니라 posterior decision support recovery로 두는 것이 안전하다. Rossi/Separate와의 공정 비교에는 prototype entry support를 보조 지표로 추가하는 것이 필요하다.

## 4. Eta penalty ablation 진단

앞 절의 이론적 직관은 posterior decision score에 직접 들어가는 centered eta contrast를 coordinate 단위로 선택하는 것이 자연스럽다는 것이다. 이 직관을 확인하기 위해 Eta-group과 두 가지 진단용 변형을 rep=20 기준으로 비교했다. `Eta ANOVA L1`과 `Rossi-mu group`은 공식 방법이 아니라 ablation diagnostic이다.

| 비교 목적 | method | penalty / model | reps | selected q | FPR | Precision | F1 | MSE_eta | 해석 |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|:---|
| Proposed reference | Eta-group + refit | $\lambda\sum_j\lVert c_{\cdot j}\rVert_2$ | 20 | 25.45 | 0.046 | 0.867 | 0.925 | 0.191 | true q=22 근처 support를 선택 |
| Same eta, no group | Eta ANOVA L1 + refit | $\lambda\sum_{k,j}\lvert c_{kj}\rvert$ | 20 | 99.90 | 0.999 | 0.220 | 0.361 | 0.581 | 같은 eta라도 entrywise L1은 거의 dense support |
| Rossi direction group | Rossi-mu group + refit | $\lambda_\mu\sum_j\lVert\mu_{\cdot j}\rVert_2$ | 20 | 29.10 | 0.091 | 0.813 | 0.883 | 0.192 | $\mu$-space group은 개선되지만 Eta-group보다 F1이 낮음 |

표의 차이는 각 penalty가 선택하는 target이 다르기 때문에 나온다. Posterior decision boundary는

$$
\log \frac{P(z=k\mid x)}{P(z=\ell\mid x)}=
\log\frac{\alpha_k}{\alpha_\ell}
+\log\frac{C_d(\kappa_k)}{C_d(\kappa_\ell)}
+(\eta_k-\eta_\ell)^\top x
$$

로 결정된다. 따라서 실제 boundary에 들어가는 coordinate는 $\mu$ 자체보다 $\eta_k-\eta_\ell$ 또는 centered eta contrast에 더 직접적으로 연결된다. 세 진단이 겨냥하는 support target은 다음처럼 다르다.

$$
S_\eta=\{j:\lVert c_{\cdot j}\rVert_2>0\},
\qquad
S_{\mathrm{ANOVA}}=\{(k,j):\lvert c_{kj}\rvert>0\},
\qquad
S_\mu=\{j:\lVert\mu_{\cdot j}\rVert_2>0\}.
$$

Eta-group은 $S_\eta$를 직접 선택하므로 posterior decision support에 가장 가깝다. Eta ANOVA L1은 component-entry 단위 효과가 따로 남아 coordinate support가 조밀해지기 쉽다. Rossi-mu group은 direction sparsity에는 자연스럽지만, posterior score에 들어가는 $\kappa_k\mu_{kj}$ scale과 centered contrast를 penalty 단계에서 직접 보지는 않는다.

- `Eta ANOVA L1`은 같은 eta 자연모수라도 entrywise L1이면 거의 dense support로 가므로, coordinate group penalty가 중요하다.
- `Rossi-mu group + refit`은 Rossi 계열에 더 가까운 $\mu$-space group penalty다. Dense support는 피했지만 Eta-group보다 FPR이 크고 F1이 낮았다.
- 현재 diagnostic 기준에서는 eta 자연모수만으로 충분하지 않고, group penalty만으로도 충분하지 않으며, `centered eta contrast + coordinate group penalty` 조합이 support recovery에서 가장 안정적으로 보인다.
- `MSE_eta`는 `MSE_centered_eta`를 뜻한다.
- 세 행 모두 기존 strong scenario rep=20 summary 기준이다.
- `Eta ANOVA L1`과 `Rossi-mu group`은 diagnostic variant이며 official method가 아니다. 특히 `Rossi-mu group`은 not Rossi 2022 official baseline이다.

## 5. proximal EM-type update와 단조증가

본 방법의 추정은 닫힌형 M-step이 아니라 proximal EM-type update다. vMF normalizing constant와 centered eta group penalty 때문에 penalized M-step을 한 번에 닫힌 형태로 풀기 어렵다.

따라서 각 iteration에서는 현재 위치에서 proximal shrinkage candidate를 만들고 objective를 확인한다. 보폭이 너무 크면 objective가 감소할 수 있으므로, 구현에서는 objective decrease가 생길 때 step size를 줄이는 step-halving line search를 사용한다.

이 부분은 자동 단조증가 정리나 전역 수렴 보장이 아니라 optimization safeguard로 설명하는 것이 안전하다.

## 6. 현재 결론

- $\eta_k=\kappa_k\mu_k$는 posterior decision score에 직접 들어가는 자연모수다.
- $\eta_k\ne0$이고 $\kappa_k>0$이면 $\mu_k$와 $\kappa_k$는 component-level에서 유일하게 복원된다.
- Eta-group의 이론적 동기는 posterior decision score에 직접 들어가는 centered eta contrast를 coordinate 단위로 선택한다는 점이다.
- ablation diagnostic에서는 centered eta contrast와 coordinate group penalty의 조합이 support recovery 안정성에 중요해 보인다.
- 조밀 support 또는 약한 신호에서는 Eta-group이 불리하거나 BIC tuning failure가 생길 수 있다.
- 다음 단계는 posterior decision support를 main claim으로 둘지, prototype entry support를 보조 지표로 둘지 교수님께 확인받는 것이다.
