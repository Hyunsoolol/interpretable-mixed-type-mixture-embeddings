# Thesis Meeting 260624

업데이트: 2026-06-23

## 1. 요약

Eta-group의 핵심 주장은 ARI 향상이 아니라, vMF mixture에서 posterior decision parameter인 $\eta=\kappa\mu$의 component contrast를 sparse하게 만들어 clustering을 유지하면서 coordinate support 해석성을 높이는 것이다.

Main evidence:

1. K=2 toy setting에서는 모든 방법이 ARI=1.000이지만, Eta-group + refit만 selected q=13.20, FPR=0.036으로 true q=10에 가장 가깝다.
2. K=4 strong common+specific setting에서는 Eta-group + refit이 ARI=0.686을 유지하면서 selected q=24.75, FPR=0.037로 Rossi/Separate보다 훨씬 sparse하다.

Limitations:

1. High-dimensional d=400에서는 sparse recovery가 안정적이지 않고, adaptive penalty weighting도 dense support로 악화됨.


관련 문서:

- 자세한 방법론: [docs/methods/thesis-methods_260624.md](../methods/thesis-methods_260624.md)
- 구현 상세: [docs/implementation/thesis-implementation_260624.md](../implementation/thesis-implementation_260624.md)
- 전체 simulation summary: [docs/simulations/thesis-simulation_260624.md](../simulations/thesis-simulation_260624.md)

## 2. 방법론 요약

vMF mixture model:

$$p(x_i;\Theta)=\sum_{k=1}^K \alpha_k C_d(\kappa_k)\exp(\kappa_k\mu_k^\top x_i), \qquad \|\mu_k\|_2=1,\quad \kappa_k>0$$

Natural parameter:

$$\eta_k=\kappa_k\mu_k$$

Posterior decision에는 $\mu_k$ 단독이 아니라 $\eta_k$가 직접 들어간다.

$$\log\frac{\tau_{i2}}{\tau_{i1}}=\mathrm{const}+(\eta_2-\eta_1)^\top x_i$$

K>2에서는 coordinate별 centered eta group penalty를 사용한다.

$$P_{\mathrm{Eta\text{-}group}}(\Theta)=\lambda_\eta\sum_{j=1}^d\left\|c_{\cdot j}\right\|_2,\qquad c_{kj}=\eta_{kj}-\frac{1}{K}\sum_{\ell=1}^K\eta_{\ell j}$$

Estimation은 proximal EM-type update, tuning은 path+BIC, refit은 selected support fixed unpenalized refit으로 둔다. 세부 알고리즘, df approximation, reference는 methods/implementation 문서에 분리했다.

## 3. 핵심 simulation 결과

상세한 6개 모형 비교, 모수 추정 결과, signal sensitivity, high-dimensional diagnostic 표는 [docs/simulations/thesis-simulation_260624.md](../simulations/thesis-simulation_260624.md)로 분리했다.

### 3.1 Toy setting: K=2

평균방향은 같고 concentration 차이만 있는 단순 setting에서 eta contrast penalty가 sparse support를 회복하는지 확인했다.

| 항목 | 값 |
|:---|:---|
| 데이터 크기 | K = 2, n = 1000, d = 100, rep = 20 |
| 활성 변수 구조 | 두 component가 같은 10개 active 좌표 사용 = true q 10 |
| 평균방향 | $\mu_1=\mu_2$ |
| concentration | $\kappa=(20,200)$ |

| Method | ARI | Selected q | TPR | FPR | Precision | F1 |
|:---|---:|---:|---:|---:|---:|---:|
| Rossi BIC | 1.000 | 23.300 | 1.000 | 0.148 | 0.443 | 0.610 |
| Rossi BIC + refit | 1.000 | 23.300 | 1.000 | 0.148 | 0.443 | 0.610 |
| Separate BIC | 1.000 | 23.300 | 1.000 | 0.148 | 0.443 | 0.610 |
| Separate BIC + refit | 1.000 | 23.300 | 1.000 | 0.148 | 0.443 | 0.610 |
| Eta-group BIC | 1.000 | 13.200 | 1.000 | 0.036 | 0.792 | 0.875 |
| Eta-group BIC + refit | 1.000 | 13.200 | 1.000 | 0.036 | 0.792 | 0.875 |

| Method | MSE_mu | MSE_kappa | MSE_Delta_eta | kappa ratio | eta contrast norm |
|:---|---:|---:|---:|---:|---:|
| Rossi BIC | 0.000176 | 1.276 | 0.245 | 10.062 | 181.179 |
| Rossi BIC + refit | 0.000061 | 1.410 | 0.378 | 9.951 | 180.821 |
| Separate BIC | 0.000176 | 1.276 | 0.245 | 10.062 | 181.179 |
| Separate BIC + refit | 0.000061 | 1.410 | 0.378 | 9.951 | 180.821 |
| Eta-group BIC | 0.000180 | 7.415 | 0.292 | 8.559 | 175.542 |
| Eta-group BIC + refit | 0.000041 | 1.185 | 0.216 | 9.960 | 180.630 |

해석: clustering은 모든 방법에서 쉽지만, variable selection은 Eta-group + refit이 가장 좋다. 이 setting은 eta contrast를 직접 penalize하는 이유를 보여주는 idea check다.

### 3.2 Main setting: K=4 strong common+specific

| Method | ARI | Selected q | FPR | Precision | F1 |
|:---|---:|---:|---:|---:|---:|
| Rossi BIC | 0.680 | 98.52 | 0.981 | 0.223 | 0.365 |
| Rossi BIC + refit | 0.653 | 98.52 | 0.981 | 0.223 | 0.365 |
| Separate BIC | 0.684 | 86.46 | 0.826 | 0.258 | 0.409 |
| Separate BIC + refit | 0.657 | 86.46 | 0.826 | 0.258 | 0.409 |
| Eta-group BIC | 0.625 | 24.75 | 0.037 | 0.890 | 0.937 |
| Eta-group BIC + refit | 0.686 | 24.75 | 0.037 | 0.890 | 0.937 |

해석: Eta-group + refit은 ARI를 유지하면서 selected q=24.75로 true union q=22에 가깝고, FPR=0.037로 Rossi/Separate보다 훨씬 낮다. 이 setting이 현재 main evidence다.

### 3.3 Robustness / limitation summary

| Setting | Role | Key result | Interpretation |
|:---|:---|:---|:---|
| Weak concentration | robustness | Eta-group + refit: ARI=0.575, selected q=24.09, FPR=0.027 | 결과는 양호하지만 main claim보다 robustness evidence로 두는 편이 안전하다. |
| d=200 basic path | high-dimensional robustness | Eta-group + refit: selected q=120.06, FPR=0.552 | dense baseline보다는 낫지만 true q=22 근처 sparse recovery는 무너진다. |
| d=200 long path + adaptive | diagnostic | selected q=40.98, FPR=0.127, F1=0.715 | d=200에서는 path density와 adaptive weighting 결합이 가장 좋았다. |
| d=400 basic path | stress limitation | Eta-group + refit: selected q=262.95, FPR=0.642 | high-dimensional success case로 보기 어렵다. |
| d=400 long path | path diagnostic | selected q=68.75, FPR=0.146, F1=0.441 | long path는 개선하지만 true q=22 회복은 여전히 제한적이다. |
| d=400 long path + adaptive | failed diagnostic | selected q=308.00, FPR=0.760, F1=0.127 | adaptive penalty는 d=400에서 dense support로 악화됐다. |

## 4. Real-data diagnostic 후보

- SPLADE sparse lexical representation을 BBC3 text diagnostic에 적용했다. d=500 + EBIC 기준에서 Eta-group + refit은 ARI=0.911, selected q=206으로 Rossi EBIC(ARI=0.903, selected q=489)보다 support를 줄이면서 clustering을 유지했다. 반면 matched TF-IDF에서는 Eta-group EBIC + refit이 selected q=101로 sparse해졌지만 ARI=0.344로 clustering이 무너졌다.

- Eta-group의 selected token은 부호까지 해석할 수 있다. BBC3 결과에서 entertainment cluster는 `+film`, `+actor`를 갖고, sport cluster는 `-film`, tech cluster는 `-actor`를 갖는다. 이는 단순 빈도 차이가 아니라 centered eta score 기준의 soft posterior contrast이며, hard rule로 해석하면 안 된다.

- 따라서 SPLADE BBC3 결과는 본문 핵심 결과가 아니라 appendix/diagnostic real-data 후보로 두는 것이 안전하다. SPLADE token은 learned lexical expansion token이며 반드시 원문 단어는 아니다. 자세한 표는 [`docs/realdata/260624_splade_bbc3_diagnostic/splade_bbc3_realdata_conclusion_260624.md`](../realdata/260624_splade_bbc3_diagnostic/splade_bbc3_realdata_conclusion_260624.md)에 분리했다.
