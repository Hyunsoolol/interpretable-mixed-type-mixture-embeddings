# Sparse vMF Mixture via Natural Parameter Shrinkage

## 1. Formulation & Penalty Structures
vMF 혼합 모형의 확률밀도함수와 자연 모수(Natural Parameter)는 다음과 같다.
$$p(x_i \mid \Theta) = \sum_{k=1}^{K} \alpha_k C_d(\kappa_k) \exp(\kappa_k \mu_k^\top x_i)$$
$$\eta_k = \kappa_k \mu_k$$

**[비교 모형의 목적 함수]**
* **Rossi (2022) [$\mu$-penalty]**: 
  $$\ell_{\mathrm{Rossi}} = \ell(\Theta) - \beta \sum_k \|\mu_k\|_1$$
* **Baseline [Split-penalty]**: 
  $$\ell_{\mathrm{sep}} = \ell(\Theta) - \lambda_\mu \sum_k \|\mu_k\|_1 - \lambda_\kappa \sum_k \kappa_k$$
* **Proposed [$\eta$-penalty]**: 
  $$\ell_{\eta} = \ell(\Theta) - \lambda_\eta \|\eta_2-\eta_1\|_1$$

---

## 2. $\eta$-Penalty 모형의 수리적 타당성 (Theoretical Justification)

### 2.1. 베이즈 결정 경계 (Decision Boundary) 수축
$$\log \frac{\tau_{i2}}{\tau_{i1}} = \log \frac{\alpha_2 C_d(\kappa_2)}{\alpha_1 C_d(\kappa_1)} + (\eta_2 - \eta_1)^\top x_i$$
* 사후 확률을 결정하는 선형 판별 계수는 $\mu$가 아닌 **$\eta$의 대조(Contrast)**임.
* $\eta_2 - \eta_1$에 $L_1$ 패널티 부여 시, 노이즈 차원의 판별 계수를 0으로 강제하여 가장 직접적인 변수 선택 수행.

### 2.2. 집중도 주도 (Concentration-driven) 환경 식별
조건: $\mu_1 = \mu_2$, $\kappa_1 \ll \kappa_2$ (방향 동일, 집중도 상이)
* **Rossi**: $\|\mu_2 - \mu_1\| = 0$ $\rightarrow$ 군집 식별 및 변수 선택 불가.
* **Proposed**: $\|\eta_2 - \eta_1\| = \kappa_2 - \kappa_1 \neq 0$ $\rightarrow$ 좌표별 분리 효과 포착 성공.

### 2.3. 내재적 정규화 (Built-in Regularization)
$$\|\eta_k\|_2 = \|\kappa_k \mu_k\|_2 = \kappa_k$$
* 자연 모수의 $L_2$ 노름이 집중도 $\kappa$.
* $\eta$에 대한 $L_1$ 패널티는 필연적으로 $\kappa$ 스케일 수축 유도.
* 고차원 모형의 $\kappa_k \rightarrow \infty$ 발산 현상을 `shared kappa` 제약 없이 원천 차단.

### 2.4. Select-then-Refit (수축 편향 제거)
$$\hat{S}_\eta = \{j : |\hat{\eta}_{2j} - \hat{\eta}_{1j}| > 0 \}$$
* Phase 1: $\eta$-penalty를 통해 Support $\hat{S}_\eta$ 도출.
* Phase 2 (Refit): $\mu_{kj} = 0 \text{ for } j \notin \hat{S}_\eta$ 제약 하에 unpenalized EM 재학습.
* $L_1$ 수축 편향(Shrinkage bias)을 제거하여 $\kappa$ 및 $\eta$ 모수 추정치 복원.

---

## 3. 핵심 시뮬레이션 결과 요약

**[Setting: Concentration-driven]**
$K=2, n=1000, d=100$. True active $q=10$.
$\mu_1 = \mu_2$, $\kappa_1=20, \kappa_2=200$. 
(True $\kappa$ ratio = 10, True $\|\eta_2-\eta_1\| = 180$)

| Method | Selected $q$ | FPR | F1 | $\kappa$ ratio | $\|\eta_2-\eta_1\|$ |
|:---|---:|---:|---:|---:|---:|
| **Rossi** | 21.93 | 0.133 | 0.635 | 10.15 | 181.37 |
| **Rossi + Refit** | 21.93 | 0.133 | 0.635 | 10.04 | 181.02 |
| **Proposed ($\eta$)** | 11.80 | **0.020** | **0.920** | 8.43 *(Shrink)* | 174.77 *(Shrink)* |
| **Proposed + Refit**| **11.80** | **0.020** | **0.920** | **10.05** | **180.83** |

*모든 모형 `TPR = 1.0`, `ARI = 1.0` 달성.*

**[Conclusion]**
* **Rossi**: False Positive 통제 실패. Refit 수행해도 Support 회복 불가.
* **Proposed ($\eta$)**: FPR 최소화 및 최적의 변수 선택(F1 0.920) 달성.
* **Proposed + Refit**: Support 유지하며 축소된 모수값(kappa ratio, $\eta$ contrast)을 True value 스케일로 완벽히 복원.
