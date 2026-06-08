# 연구미팅 내용 공부 정리

## 1. 기본 모형: vMF 분포

방향자료 `x_i`는 단위구면 위의 벡터다.

```math
x_i \in S^{d-1}, \qquad \|x_i\|_2 = 1
```

vMF density는 다음과 같다.

```math
f(x_i \mid \mu_k, \kappa_k)
=
C_d(\kappa_k)
\exp(\kappa_k \mu_k^\top x_i)
```

여기서

- `mu_k`: k번째 군집의 평균 방향
- `kappa_k`: 집중도, 값이 클수록 `mu_k` 주변에 자료가 강하게 모임
- `C_d(kappa_k)`: 정규화 상수

참고: Banerjee et al. (2005), Rossi & Barbaro (2022)

## 2. vMF Mixture Model

군집이 `K`개일 때 mixture likelihood는 다음과 같다.

```math
f(x_i)
=
\sum_{k=1}^{K}
\alpha_k C_d(\kappa_k)
\exp(\kappa_k \mu_k^\top x_i)
```

추정해야 할 값은 다음이다.

```math
\Theta =
\{\alpha_k, \mu_k, \kappa_k\}_{k=1}^{K}
```

EM algorithm에서는 posterior responsibility를 계산한다.

```math
\tau_{ik}
=
\frac{
\alpha_k C_d(\kappa_k)\exp(\kappa_k \mu_k^\top x_i)
}{
\sum_{\ell=1}^{K}
\alpha_\ell C_d(\kappa_\ell)
\exp(\kappa_\ell \mu_\ell^\top x_i)
}
```

그리고

```math
N_k = \sum_i \tau_{ik},
\qquad
r_k = \sum_i \tau_{ik} x_i
```

를 이용해 M-step을 수행한다.

## 3. Rossi & Barbaro (2022) Sparse vMF

Rossi & Barbaro (2022)는 `mu_k`에 L1 penalty를 둔다.

```math
\ell_p(\Theta)
=
\ell(\Theta)
-
\beta
\sum_{k=1}^{K}
\|\mu_k\|_1
```

핵심 update는 soft-thresholding 형태다.

```math
\mu_{kj}
\propto
\mathrm{sign}(r_{kj})
\left(\kappa_k |r_{kj}| - \beta\right)_+
```

그 후 정규화한다.

```math
\|\mu_k\|_2 = 1
```

집중도는 다음 근사식을 이용한다.

```math
\rho_k =
\frac{\mu_k^\top r_k}{N_k}
```

```math
\kappa_k
\approx
\frac{d\rho_k - \rho_k^3}{1-\rho_k^2}
```

해석:

- Rossi 방법은 sparse prototype을 만든다.
- 즉, 각 군집의 평균 방향 `mu_k`에서 어떤 coordinate가 중요한지 보여준다.
- 하지만 posterior decision에는 `mu_k`만 들어가는 것이 아니라 `eta_k = kappa_k mu_k`가 들어간다.

## 4. Posterior Decision과 Eta

vMF mixture에서 군집 선택은 다음 값을 비교하는 것과 관련된다.

```math
\log \alpha_k
+
\log C_d(\kappa_k)
+
\kappa_k \mu_k^\top x_i
```

여기서 핵심 선형항은

```math
\eta_k^\top x_i
```

이고,

```math
\eta_k = \kappa_k \mu_k
```

이다.

따라서 군집 구분에 직접 들어가는 값은 `mu_k`가 아니라 `eta_k`다.

이 연구의 핵심 문제의식:

```text
Rossi 2022는 mu_k에 penalty를 둔다.
하지만 실제 posterior decision에는 eta_k = kappa_k mu_k가 들어간다.
따라서 concentration 차이가 중요한 상황에서는 eta 기준 sparsity가 더 자연스럽다.
```

## 5. Refit

Refit은 variable selection 후 선택된 support만 고정하고, penalty 없이 다시 vMF mixture를 추정하는 절차다.

절차:

```text
1. penalized model 적합
2. 선택된 active coordinate S 결정
3. S 밖의 coordinate는 0으로 고정
4. S 안에서 penalty 없이 vMF mixture 재추정
```

해석:

- refit은 선택된 변수 집합을 바꾸지 않는다.
- 따라서 `TPR`, `FPR`, `Precision`, `F1`은 그대로다.
- 대신 penalty shrinkage를 줄여 `kappa`, `eta` 추정값을 개선할 수 있다.

## 6. 분리 패널티 EM

교수님 제안에 따라 `mu_k`와 `kappa_k`에 penalty를 분리해서 둔 방법이다.

```math
Q_{\mathrm{pen}}
=
\ell(\Theta)
-
\lambda_\mu
\sum_k \|\mu_k\|_1
-
\lambda_\kappa
\sum_k \kappa_k
```

`mu_k` update는 다음과 같다.

```math
z_{kj}
=
\mathrm{sign}(r_{kj})
\left(\kappa_k |r_{kj}| - \lambda_\mu\right)_+
```

```math
\mu_k =
\frac{z_k}{\|z_k\|_2}
```

`kappa_k` update는 다음 형태다.

```math
\rho_k =
\frac{\mu_k^\top r_k - \lambda_\kappa}{N_k}
```

```math
\kappa_k = A_d^{-1}(\rho_k)
```

해석:

- `mu_k` sparsity와 `kappa_k` shrinkage를 분리해서 볼 수 있다.
- 하지만 `kappa_k`는 coordinate별 값이 아니라 component-level scalar다.
- 따라서 concentration 차이가 어떤 coordinate에서 발생했는지를 직접 선택하지는 못한다.

## 7. Eta Penalty EM

제안 방향은 posterior decision에 직접 들어가는 `eta_k`에 penalty를 두는 것이다.

```math
\eta_k = \kappa_k \mu_k
```

K=2일 때는 두 군집의 차이를 다음으로 본다.

```math
\Delta \eta =
\eta_2 - \eta_1
```

penalty는 다음과 같다.

```math
\ell_{\mathrm{pen}}
=
\ell(\Theta)
-
\lambda_\eta
\|\eta_2 - \eta_1\|_1
```

해석:

- `eta_2j - eta_1j`가 0이면 coordinate `j`는 군집 구분에 직접 기여하지 않는다고 본다.
- concentration-driven setting에서는 이 기준이 `mu` 기준보다 자연스럽다.

## 8. K=4에서 Eta Penalty 일반화

Rossi 2022 paper-like setting은 `K=4`다. 이 경우 `eta_2 - eta_1`만으로는 부족하다.

그래서 coordinate별 centered eta contrast를 쓴다.

```math
\bar{\eta}_{j}
=
\frac{1}{K}
\sum_{k=1}^{K}
\eta_{kj}
```

```math
\eta_{kj}^{c}
=
\eta_{kj}
-
\bar{\eta}_{j}
```

coordinate `j`의 component 간 차이는 다음으로 본다.

```math
\left(
\sum_{k=1}^{K}
(\eta_{kj}^{c})^2
\right)^{1/2}
```

K=4 eta penalty는 다음 방향이다.

```math
\ell_{\mathrm{pen}}
=
\ell(\Theta)
-
\lambda_\eta
\sum_j
\left(
\sum_{k=1}^{K}
(\eta_{kj} - \bar{\eta}_{j})^2
\right)^{1/2}
```

해석:

- 네 군집의 `eta`가 coordinate `j`에서 거의 같으면 제거한다.
- 군집 간 `eta` 차이가 큰 coordinate는 선택한다.
- 이는 posterior decision에 직접 관련된 coordinate selection이다.

## 9. 비교한 방법

현재 비교 구조는 다음 6가지다.

| 방법 | penalty 기준 | refit |
|---|---|---|
| Rossi | `mu_k` | 없음 |
| Rossi + refit | Rossi support | 있음 |
| 분리 패널티 | `mu_k`, `kappa_k` | 없음 |
| 분리 패널티 + refit | 분리 패널티 support | 있음 |
| 에타 패널티 | `eta_k` contrast | 없음 |
| 에타 패널티 + refit | 에타 support | 있음 |

## 10. 평가지표

Clustering:

```math
ARI
```

Variable selection:

```text
selected q
TPR
FPR
Precision
F1
```

Parameter estimation:

```math
\mathrm{MSE}_{\mu}
=
\frac{1}{Kd}
\sum_{k,j}
(\hat{\mu}_{kj} - \mu_{kj})^2
```

```math
\mathrm{MSE}_{\kappa}
=
\frac{1}{K}
\sum_k
(\hat{\kappa}_k - \kappa_k)^2
```

K=2에서는

```math
\mathrm{MSE}_{\Delta \eta}
=
\frac{1}{d}
\sum_j
\left[
(\hat{\eta}_{2j}-\hat{\eta}_{1j})
-
(\eta_{2j}-\eta_{1j})
\right]^2
```

K=4에서는 centered eta 기준을 쓴다.

```math
\mathrm{MSE}_{\eta^c}
=
\frac{1}{Kd}
\sum_{k,j}
(\hat{\eta}_{kj}^{c} - \eta_{kj}^{c})^2
```

## 11. 현재까지의 해석

핵심 결론은 다음이다.

```text
Rossi 2022는 sparse prototype을 제공하지만,
penalty와 해석 대상이 mu_k에 있다.

하지만 posterior decision에는
eta_k = kappa_k mu_k가 직접 들어간다.

따라서 concentration 차이가 중요한 경우에는
eta 기준 variable selection이 더 자연스럽다.
```

시뮬레이션 결과상:

- concentration-driven setting에서는 에타 패널티가 가장 좋은 variable selection을 보였다.
- Rossi는 clustering은 잘하지만 false positive가 많았다.
- 분리 패널티는 Rossi보다 개선되지만, `kappa_k`가 coordinate-specific하지 않다는 한계가 있다.
- 에타 패널티는 FPR을 크게 줄이고 Precision/F1이 좋았다.
- refit은 support를 바꾸지는 않지만, shrinkage를 줄여 `kappa`와 `eta` 추정을 개선했다.

## 12. 공부할 때 핵심 질문

```text
1. 왜 posterior decision에서 eta_k가 중요한가?
2. Rossi 2022의 penalty target은 왜 mu_k인가?
3. concentration 차이가 있을 때 mu 기준 sparsity가 왜 부족한가?
4. 분리 패널티에서 kappa penalty는 왜 coordinate selection을 직접 못 하는가?
5. eta penalty는 어떤 의미에서 coordinate-level decision effect를 선택하는가?
6. refit은 왜 support는 그대로 두고 parameter estimation만 개선하는가?
```

## 13. R 코드 구현 구조

현재 구현은 Rossi & Barbaro (2022) 재현 코드를 공통 기반으로 두고, 그 위에 분리 패널티와 에타 패널티를 비교하는 구조다.

| 주제 | 파일 | 주요 함수 |
|---|---|---|
| 공통 vMF 함수 | `rossi_barbaro_2022_reproduction.r` | `estimate_kappa()`, `e_step_vmf()`, `fit_svMF_em()` |
| Rossi 2022 sparse vMF | `rossi_barbaro_2022_reproduction.r` | `update_mu_kappa_one()`, `fit_svMF_path()` |
| 정보기준 계산 | `rossi_barbaro_2022_reproduction.r` | `model_df()`, `model_ic()` |
| Rossi + refit | `rossi_refit_limit_run.r` | `fit_support_constrained_vmf()` |
| 분리 패널티 EM | `separate_penalty_vmf_run.r` | `fit_separate_penalty_em()`, `update_mu_kappa_separate_one()` |
| 에타 패널티 EM | `eta_penalty_vmf_run.r` | `fit_eta_penalty_em()`, `prox_eta_contrast_k2()` |
| K=4 pilot 비교 | `rb2022_k4_pilot_compare_run.r` | `fit_eta_centered_em()`, `prox_eta_centered()` |

구현상 공통 흐름은 다음과 같다.

```text
1. 초기값 생성 또는 이전 lambda 결과로 warm start
2. E-step에서 posterior probability tau_ik 계산
3. M-step에서 alpha, mu, kappa 또는 eta 업데이트
4. penalized objective 수렴 확인
5. lambda grid 또는 beta path 중 BIC가 가장 작은 모형 선택
6. 선택된 support를 고정하고 필요하면 penalty 없이 refit
```

## 14. 패널티가 있는 EM 구현

공통 E-step은 모든 방법에서 같다.

```math
\tau_{ik}^{(t)}
=
\frac{
\alpha_k^{(t)}
C_d(\kappa_k^{(t)})
\exp((\eta_k^{(t)})^\top x_i)
}{
\sum_{\ell=1}^{K}
\alpha_\ell^{(t)}
C_d(\kappa_\ell^{(t)})
\exp((\eta_\ell^{(t)})^\top x_i)
}
```

여기서

```math
\eta_k = \kappa_k \mu_k
```

이다.

E-step 이후에는 다음 sufficient statistics를 계산한다.

```math
N_k = \sum_i \tau_{ik}
```

```math
r_k = \sum_i \tau_{ik}x_i
```

```math
\alpha_k = N_k/n
```

### Rossi 2022 구현

Rossi 방법은 `mu_k`에 L1 penalty를 둔다.

```math
\ell_{\mathrm{pen}}(\Theta)
=
\ell(\Theta)
-
\beta
\sum_k \|\mu_k\|_1
```

코드에서는 각 component마다 `mu_k`와 `kappa_k`를 안쪽 반복문에서 번갈아 업데이트한다.

```math
z_{kj}
=
\mathrm{sign}(r_{kj})
\left(\kappa_k |r_{kj}|-\beta\right)_+
```

```math
\mu_k = z_k / \|z_k\|_2
```

그 다음

```math
\rho_k = \frac{\mu_k^\top r_k}{N_k}
```

를 계산하고 `estimate_kappa()`로 `kappa_k`를 업데이트한다.

### 분리 패널티 구현

분리 패널티는 교수님 제안에 따라 `mu_k`와 `kappa_k`에 penalty를 따로 둔 baseline이다.

```math
\ell_{\mathrm{pen}}(\Theta)
=
\ell(\Theta)
-
\lambda_\mu
\sum_k \|\mu_k\|_1
-
\lambda_\kappa
\sum_k \kappa_k
```

`mu_k` 업데이트는 Rossi와 같은 soft-thresholding 형태이고, penalty parameter만 `lambda_mu`로 바뀐다.

```math
z_{kj}
=
\mathrm{sign}(r_{kj})
\left(\kappa_k |r_{kj}|-\lambda_\mu\right)_+
```

```math
\mu_k = z_k / \|z_k\|_2
```

`kappa_k` 업데이트에서는 concentration에 직접 penalty가 들어간다.

```math
\rho_k
=
\frac{\mu_k^\top r_k-\lambda_\kappa}{N_k}
```

```math
\kappa_k = A_d^{-1}(\rho_k)
```

즉 `lambda_kappa`는 concentration 자체를 줄이는 역할을 한다. 다만 `kappa_k`는 component-level scalar이므로, 어떤 coordinate가 concentration 차이에 기여했는지를 직접 선택하지는 못한다.

### 에타 패널티 구현

에타 패널티는 2022 논문에 있는 방법이 아니라, 현재 연구에서 비교 중인 제안 방향이다. 목적은 posterior decision에 직접 들어가는 natural parameter 차이를 penalize하는 것이다.

K=2에서는 다음 penalty를 둔다.

```math
\ell_{\mathrm{pen}}(\Theta)
=
\ell(\Theta)
-
\lambda_\eta
\|\eta_2-\eta_1\|_1
```

현재 R 구현은 exact closed-form M-step이 아니라 proximal EM prototype이다. 절차는 다음과 같다.

```text
1. 현재 tau_ik로 unpenalized vMF M-step을 수행한다.
2. eta_k = kappa_k mu_k를 계산한다.
3. delta = eta_2 - eta_1에 coordinate-wise soft-thresholding을 적용한다.
4. shrink된 eta를 다시 mu_k, kappa_k로 변환한다.
```

수식으로 쓰면,

```math
\bar{\eta}
=
(\eta_1+\eta_2)/2
```

```math
\delta
=
\eta_2-\eta_1
```

```math
\delta_\lambda
=
\mathrm{sign}(\delta)
\left(|\delta|-\lambda_\eta\right)_+
```

```math
\eta_1
=
\bar{\eta}-\delta_\lambda/2
```

```math
\eta_2
=
\bar{\eta}+\delta_\lambda/2
```

마지막으로

```math
\kappa_k = \|\eta_k\|_2
```

```math
\mu_k = \eta_k / \|\eta_k\|_2
```

로 되돌린다.

K=4 pilot에서는 단순한 pairwise contrast 대신, component 평균에서 벗어난 centered eta를 shrink했다.

```math
\bar{\eta}_j
=
K^{-1}
\sum_k \eta_{kj}
```

```math
c_{kj}
=
\eta_{kj}-\bar{\eta}_j
```

coordinate `j` 전체에 대해 group shrinkage를 적용한다.

```math
c_{\cdot j,\lambda}
=
\left(
1-\frac{\lambda_\eta}{\|c_{\cdot j}\|_2}
\right)_+
c_{\cdot j}
```

이 방식은 K=2의 `eta_2 - eta_1` penalty를 K>2 상황으로 확장하기 위한 pilot 구현이다.

## 15. 집중도 kappa 추정과 근사식 출처

vMF 분포에서 `kappa`의 MLE는 보통 닫힌형으로 바로 나오지 않는다. 정확한 식은 다음 방정식을 푸는 것이다.

```math
A_d(\kappa_k)
=
\rho_k
```

여기서

```math
A_d(\kappa)
=
\frac{I_{d/2}(\kappa)}
{I_{d/2-1}(\kappa)}
```

이고 `I`는 modified Bessel function이다.

unpenalized M-step에서는 보통

```math
\rho_k
=
\frac{\|r_k\|_2}{N_k}
```

또는 `mu_k`가 먼저 정해진 경우

```math
\rho_k
=
\frac{\mu_k^\top r_k}{N_k}
```

를 사용한다.

현재 코드의 `estimate_kappa()`는 다음 근사식을 사용한다.

```math
\kappa_k
\approx
\frac{d\rho_k-\rho_k^3}
{1-\rho_k^2}
```

이 근사식은 vMF mixture EM에서 널리 쓰이는 Banerjee et al. (2005)의 kappa update approximation에 기반한다. 방향통계의 기본 MLE 식은 Mardia and Jupp (2000)의 directional statistics 문헌에서 표준적으로 다루고, kappa inverse approximation 관련 논의는 Sra (2012)에서도 다룬다.

현재 코드에서 이 근사식을 쓰는 이유는 다음과 같다.

- 매 EM iteration마다 `A_d^{-1}`를 numerical root finding으로 풀면 계산이 느려진다.
- `d=100` 같은 고차원 simulation에서는 Bessel function 계산이 불안정할 수 있다.
- Rossi & Barbaro (2022) 재현 목적상 Banerjee et al. (2005) 계열 근사식을 쓰는 것이 자연스럽다.

분리 패널티에서는 `lambda_kappa` 때문에 `rho_k`가 다음처럼 바뀐다.

```math
\rho_k
=
\frac{\mu_k^\top r_k-\lambda_\kappa}{N_k}
```

하지만 그 이후 `kappa_k` 계산은 같은 `estimate_kappa()` 근사식을 사용한다.

## 16. Lambda 선택 방식

현재 simulation에서 lambda 선택은 정보기준 기반으로 했다. 핵심 기준은 BIC이다.

```math
\mathrm{BIC}
=
\log(n)\mathrm{df}
-
2\ell(\hat{\Theta})
```

고차원 variable selection을 더 강하게 penalize하려면 EBIC도 같이 볼 수 있다.

```math
\mathrm{EBIC}
=
\left(\log(n)+2\gamma\log(d)\right)
\mathrm{df}
-
2\ell(\hat{\Theta})
```

### Rossi 방법의 beta 선택

Rossi 방법은 단순 fixed grid가 아니라 path-following 방식으로 `beta` 후보를 만든다. 코드에서는 `beta=0`에서 시작하고, 현재 nonzero coordinate가 0이 되는 다음 threshold로 이동한다.

```math
\beta_{\mathrm{next}}
=
\min_{k,j:\ \kappa_k |r_{kj}|>\beta}
\kappa_k |r_{kj}|
```

각 `beta` 후보에서 EM을 적합한 뒤 BIC가 가장 작은 모형을 선택했다. 즉 Rossi에서 `beta`는 tuning parameter이고, 현재 보고 결과는 path 위에서 BIC로 선택한 값이다.

### 분리 패널티의 lambda 선택

분리 패널티는 현재 이론적 path를 따로 유도하지 않았기 때문에 grid search로 선택했다.

K=2 concentration-driven simulation에서 기본 grid는 다음이었다.

```text
lambda_mu    = {0, 100, 200, 300, 400, 500, 600}
lambda_kappa = {0, 10, 25, 50, 75}
```

각 조합에 대해 EM을 돌리고, BIC가 가장 작은 조합을 선택했다.

K=4 paper-like pilot에서는 계산량을 줄이기 위해 더 작은 grid를 썼다.

```text
lambda_mu    = {0, 100, 200, 300, 400, 600}
lambda_kappa = {0, 5, 10, 25}
```

### 에타 패널티의 lambda 선택

에타 패널티도 현재는 grid search로 선택했다.

K=2 concentration-driven simulation의 기본 grid는 다음이었다.

```text
lambda_eta = {0, 1, 2, 5, 10, 15, 20, 30, 40, 50}
```

K=4 pilot에서는 다음 grid를 사용했다.

```text
lambda_eta = {0, 1, 2, 5, 10, 20, 30}
```

현재 단계에서는 BIC로 lambda를 선택하고, 선택된 support에 대해 refit을 붙여 parameter estimation을 비교했다.

### 앞으로의 lambda 선택 방향

논문으로 정리하려면 lambda 선택을 다음처럼 더 체계화하는 것이 좋다.

```text
1. K를 고정한 비교에서는 모든 방법에 대해 BIC 또는 EBIC로 lambda를 선택한다.
2. high-dimensional variable selection 성격을 강조할 때는 BIC보다 EBIC를 우선 검토한다.
3. Rossi는 논문 방식과 맞추기 위해 beta path를 유지한다.
4. 분리 패널티와 에타 패널티는 lambda_max를 계산한 뒤 adaptive grid를 만든다.
5. 최종 보고에서는 BIC 선택 결과와 EBIC 선택 결과가 같은지 sensitivity check를 제시한다.
```

adaptive grid의 기준은 다음처럼 잡을 수 있다.

- `lambda_mu`: `kappa_k |r_kj|`의 큰 값부터 작은 값까지 coordinate가 사라지는 threshold를 기준으로 잡는다.
- `lambda_kappa`: `mu_k^T r_k`보다 커지면 `rho_k`가 0 이하가 되어 component가 무너질 수 있으므로, 그 전 범위에서 잡는다.
- `lambda_eta`: K=2에서는 `|eta_2j - eta_1j|`의 threshold를 기준으로 잡는다.

따라서 현재는 “계산 가능한 fixed grid + BIC” 단계이고, 다음 단계는 “method별 lambda_max 기반 adaptive grid + BIC/EBIC 비교”로 가는 것이 자연스럽다.

## 17. 참고문헌 위치

현재 공부 자료에서 직접 연결되는 문헌은 다음과 같다.

| 내용 | 참고문헌 |
|---|---|
| vMF mixture EM과 kappa 근사 | Banerjee et al. (2005), `Clustering on the unit hypersphere using von Mises-Fisher distributions` |
| 방향자료와 vMF MLE 기본식 | Mardia and Jupp (2000), `Directional Statistics` |
| kappa inverse approximation 관련 논의 | Sra (2012), vMF parameter approximation 관련 note |
| sparse vMF mixture와 `mu_k` L1 penalty | Rossi and Barbaro (2022), `Mixture of von Mises-Fisher distribution with sparse prototypes` |

연구미팅에서 말할 때는 다음처럼 정리하면 된다.

```text
Rossi & Barbaro (2022)는 mu_k에 L1 penalty를 두는 sparse prototype 방식이다.
EM 안에서 mu_k는 soft-thresholding으로 업데이트하고,
kappa_k는 Banerjee et al. (2005)에서 쓰이는 vMF concentration approximation으로 계산했다.

우리 구현은 이 구조를 기준으로 두고,
교수님 제안인 mu/kappa 분리 패널티와
posterior decision에 직접 들어가는 eta = kappa mu 패널티를 비교했다.

현재 lambda는 BIC로 선택했고,
논문 단계에서는 EBIC와 adaptive grid를 같이 검토할 필요가 있다.
```
