# Sparse vMF Mixture 및 Eta Penalty 정리

## 1. vMF 분포

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

참고: Banerjee et al. (2005), Mardia and Jupp (2000), Rossi & Barbaro (2022)

## 2. vMF Mixture Model과 EM

군집이 `K`개일 때 mixture likelihood는 다음과 같다.

```math
f(x_i)
=
\sum_{k=1}^{K}
\alpha_k C_d(\kappa_k)
\exp(\kappa_k \mu_k^\top x_i)
```

추정해야 할 모수는 다음이다.

```math
\Theta =
\{\alpha_k, \mu_k, \kappa_k\}_{k=1}^{K}
```

E-step에서는 posterior responsibility를 계산한다.

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

M-step에서는 다음 sufficient statistics를 사용한다.

```math
N_k = \sum_i \tau_{ik}
```

```math
r_k = \sum_i \tau_{ik}x_i
```

```math
\alpha_k = N_k/n
```

unpenalized vMF M-step에서는 보통

```math
\mu_k = r_k / \|r_k\|_2
```

이고,

```math
\rho_k = \|r_k\|_2 / N_k
```

로부터 `kappa_k`를 추정한다.

## 3. Posterior Decision과 Eta

vMF mixture에서 관측치 `x_i`가 어느 군집에 속하는지는 다음 log posterior score 비교로 결정된다.

```math
\log \alpha_k
+
\log C_d(\kappa_k)
+
\kappa_k \mu_k^\top x_i
```

여기서 선형항을 다음처럼 쓸 수 있다.

```math
\eta_k^\top x_i
```

단,

```math
\eta_k = \kappa_k \mu_k
```

이다.

즉 `mu_k`는 평균 방향이고, `kappa_k`는 집중도지만, posterior decision에 직접 들어가는 방향별 효과는 두 값의 곱인 `eta_k`다.

K=2에서는 log posterior odds가 다음처럼 정리된다.

```math
\log
\frac{\tau_{i2}}{\tau_{i1}}
=
\log
\frac{\alpha_2}{\alpha_1}
+
\log
\frac{C_d(\kappa_2)}{C_d(\kappa_1)}
+
(\eta_2-\eta_1)^\top x_i
```

따라서 coordinate `j`가 군집 구분에 직접 기여하는지는 `mu_2j - mu_1j`보다

```math
\eta_{2j} - \eta_{1j}
```

를 보는 것이 더 직접적이다.

이 점이 에타 패널티를 제안하는 핵심 이유다.

## 4. 집중도 kappa 추정과 근사식 출처

vMF 분포에서 `kappa`의 MLE는 다음 방정식을 푸는 문제다.

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

정확히는 `A_d^{-1}`를 numerical method로 풀 수 있지만, EM iteration마다 이를 반복하면 계산이 무겁고 Bessel function 계산도 불안정할 수 있다. 그래서 현재 R 구현에서는 다음 근사식을 사용한다.

```math
\kappa_k
\approx
\frac{d\rho_k-\rho_k^3}
{1-\rho_k^2}
```

이 근사식은 vMF mixture EM에서 널리 쓰이는 Banerjee et al. (2005)의 concentration update approximation에 기반한다. 방향자료 MLE의 기본식은 Mardia and Jupp (2000)에서 표준적으로 다루고, vMF concentration inverse approximation 관련 논의는 Sra (2012)에서도 볼 수 있다.

현재 코드에서는 이 근사식이 `rossi_barbaro_2022_reproduction.r`의 `estimate_kappa()`에 구현되어 있다.

## 5. Rossi & Barbaro (2022) Sparse vMF

Rossi & Barbaro (2022)는 component direction `mu_k`에 L1 penalty를 둔다.

```math
\ell_{\mathrm{pen}}(\Theta)
=
\ell(\Theta)
-
\beta
\sum_{k=1}^{K}
\|\mu_k\|_1
```

E-step에서 `tau_ik`를 계산한 뒤, M-step에서 `N_k`, `r_k`를 만든다. 그 다음 각 component마다 `mu_k`와 `kappa_k`를 안쪽 반복문에서 번갈아 업데이트한다.

`mu_k` 업데이트는 soft-thresholding 형태다.

```math
z_{kj}
=
\mathrm{sign}(r_{kj})
\left(\kappa_k |r_{kj}|-\beta\right)_+
```

```math
\mu_k = z_k / \|z_k\|_2
```

`mu_k`가 정해지면

```math
\rho_k
=
\frac{\mu_k^\top r_k}{N_k}
```

를 계산하고, `estimate_kappa()` 근사식으로 `kappa_k`를 업데이트한다.

해석은 다음과 같다.

- Rossi 방법은 sparse prototype을 제공한다.
- 선택된 coordinate는 `mu_k`의 nonzero coordinate다.
- 그러나 posterior decision에는 `eta_k = kappa_k mu_k`가 들어간다.
- 따라서 concentration 차이가 중요한 상황에서는 `mu_k` sparsity만으로는 decision effect를 충분히 설명하지 못할 수 있다.

## 6. Rossi 방법의 한계가 나타나는 상황

현재 연구에서 만든 limitation setting은 다음이다.

```text
평균 방향 mu는 거의 같거나 동일하다.
하지만 집중도 kappa가 다르다.
```

대표적인 K=2 구조는 다음이다.

```math
\mu_1 = \mu_2
```

```math
\kappa_1 < \kappa_2
```

이 경우 `mu_2 - mu_1`만 보면 군집 간 차이가 거의 없다. 하지만

```math
\eta_2 - \eta_1
=
\kappa_2\mu_2 - \kappa_1\mu_1
```

는 0이 아니다.

즉 군집 차이는 평균 방향 자체가 아니라 concentration이 반영된 natural parameter에서 나타난다. 이 상황에서 Rossi는 clustering은 잘할 수 있지만, variable selection에서는 false positive가 늘거나 실제 decision effect를 덜 직접적으로 설명할 수 있다.

## 7. 분리 패널티 EM

교수님 제안에 따라 `mu_k`와 `kappa_k`에 penalty를 분리해서 둔 baseline이다.

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

`mu_k` update는 Rossi와 같은 soft-thresholding 형태다.

```math
z_{kj}
=
\mathrm{sign}(r_{kj})
\left(\kappa_k |r_{kj}|-\lambda_\mu\right)_+
```

```math
\mu_k = z_k / \|z_k\|_2
```

`kappa_k` update에는 concentration penalty가 들어간다.

```math
\rho_k
=
\frac{\mu_k^\top r_k-\lambda_\kappa}{N_k}
```

```math
\kappa_k = A_d^{-1}(\rho_k)
```

현재 코드에서는 `A_d^{-1}` 대신 `estimate_kappa()` 근사식을 쓴다.

분리 패널티의 장점은 `mu_k` sparsity와 `kappa_k` shrinkage를 분리해서 볼 수 있다는 점이다. 그러나 `kappa_k`는 coordinate별 parameter가 아니라 component-level scalar다. 따라서 `lambda_kappa`를 두어도 어떤 coordinate가 concentration 차이에 기여했는지는 직접 선택하지 못한다.

## 8. 제안 모형: Eta Penalty

에타 패널티는 2022 논문에 있는 방법이 아니라, 현재 연구에서 비교 중인 제안 방향이다. 핵심은 vMF mixture를 `mu_k`, `kappa_k` 대신 natural parameter

```math
\eta_k = \kappa_k \mu_k
```

관점에서 보는 것이다.

`eta_k`를 쓰면 vMF density는 다음처럼 표현된다.

```math
f(x_i \mid \eta_k)
=
C_d(\|\eta_k\|_2)
\exp(\eta_k^\top x_i)
```

이때

```math
\kappa_k = \|\eta_k\|_2
```

```math
\mu_k = \eta_k / \|\eta_k\|_2
```

이다.

따라서 mixture likelihood는 다음처럼 쓸 수 있다.

```math
\ell(\alpha,\eta)
=
\sum_i
\log
\left[
\sum_{k=1}^{K}
\alpha_k
C_d(\|\eta_k\|_2)
\exp(\eta_k^\top x_i)
\right]
```

이 표현의 장점은 posterior decision의 coordinate-level 효과가 `eta_k`에 직접 나타난다는 점이다.

### K=2 Eta Contrast Model

K=2에서는 두 군집을 구분하는 핵심 quantity가 다음이다.

```math
\delta
=
\eta_2-\eta_1
```

posterior log odds에서 coordinate `j`의 효과는 `delta_j x_ij`로 들어간다. 따라서

```math
\delta_j = 0
```

이면 coordinate `j`는 두 군집을 구분하는 선형 decision term에 직접 기여하지 않는다.

이를 기준으로 support를 정의한다.

```math
S_\eta
=
\{j:\delta_j \ne 0\}
```

K=2 에타 패널티 모형은 다음 목적함수를 사용한다.

```math
\ell_{\mathrm{pen}}(\alpha,\eta_1,\eta_2)
=
\ell(\alpha,\eta_1,\eta_2)
-
\lambda_\eta
\|\eta_2-\eta_1\|_1
```

동일한 내용을 평균 eta와 contrast eta로 다시 쓰면

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

이고,

```math
\eta_1
=
\bar{\eta}-\delta/2
```

```math
\eta_2
=
\bar{\eta}+\delta/2
```

이다. 이 관점에서 에타 패널티는 common effect `bar eta`는 유지하고, 군집 간 차이인 `delta`만 sparse하게 만드는 방법이다.

해석은 다음과 같다.

- `mu_k`는 평균 방향이다.
- `kappa_k`는 전체 concentration scalar다.
- `eta_k = kappa_k mu_k`는 posterior decision에 들어가는 natural parameter다.
- `eta_2j - eta_1j`는 coordinate `j`가 두 군집을 구분하는 직접적인 decision effect다.
- 따라서 concentration-driven setting에서는 `mu` 기준보다 `eta` 기준 variable selection이 더 자연스럽다.

### Eta Penalty EM 구현

에타 패널티의 exact penalized M-step은 닫힌형으로 바로 쓰기 어렵다. 이유는 likelihood 안에

```math
C_d(\|\eta_k\|_2)
```

가 들어가고, 동시에 `eta_2 - eta_1`에 L1 penalty가 걸리기 때문이다.

현재 R 구현은 proximal EM prototype이다. 절차는 다음과 같다.

```text
1. 현재 theta로 E-step을 수행해 tau_ik를 계산한다.
2. tau_ik를 고정하고 unpenalized vMF M-step을 수행한다.
3. unpenalized M-step 결과를 eta_k = kappa_k mu_k로 바꾼다.
4. delta = eta_2 - eta_1에 soft-thresholding을 적용한다.
5. shrink된 eta를 다시 mu_k, kappa_k로 변환한다.
6. penalized objective가 수렴할 때까지 반복한다.
```

unpenalized M-step 결과를 `eta_1`, `eta_2`라고 하면,

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

를 계산한다.

그 다음 coordinate-wise soft-thresholding을 적용한다.

```math
\delta_\lambda
=
\mathrm{sign}(\delta)
\left(|\delta|-\lambda_\eta\right)_+
```

그리고 다시

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

로 재구성한다.

마지막으로

```math
\kappa_k = \|\eta_k\|_2
```

```math
\mu_k = \eta_k / \|\eta_k\|_2
```

를 사용해 vMF parameter로 되돌린다.

현재 구현에서 이 부분은 `eta_penalty_vmf_run.r`의 `fit_eta_penalty_em()`과 `prox_eta_contrast_k2()`에 해당한다.

### K>2 Eta Penalty 일반화

K=4 같은 경우에는 `eta_2 - eta_1` 하나만으로 전체 군집 차이를 설명할 수 없다. 그래서 coordinate별로 component 간 `eta` 차이가 있는지를 본다.

coordinate `j`의 component 평균은 다음이다.

```math
\bar{\eta}_j
=
K^{-1}
\sum_{k=1}^{K}
\eta_{kj}
```

centered eta는 다음이다.

```math
c_{kj}
=
\eta_{kj}-\bar{\eta}_j
```

coordinate `j`의 군집 간 차이는 벡터 `c_{\cdot j}`의 norm으로 본다.

```math
\|c_{\cdot j}\|_2
=
\left(
\sum_{k=1}^{K}
c_{kj}^2
\right)^{1/2}
```

K>2 eta penalty는 group-lasso 형태로 쓸 수 있다.

```math
\ell_{\mathrm{pen}}(\alpha,\eta)
=
\ell(\alpha,\eta)
-
\lambda_\eta
\sum_j
\|c_{\cdot j}\|_2
```

현재 K=4 pilot 구현에서는 unpenalized eta M-step 이후 다음 shrinkage를 적용했다.

```math
c_{\cdot j,\lambda}
=
\left(
1-\frac{\lambda_\eta}{\|c_{\cdot j}\|_2}
\right)_+
c_{\cdot j}
```

그리고

```math
\eta_{kj,\lambda}
=
\bar{\eta}_j
+
c_{kj,\lambda}
```

로 복원한다.

해석은 다음과 같다.

- 모든 component의 `eta_kj`가 거의 같으면 coordinate `j`는 군집 구분에 기여하지 않는다고 본다.
- component 간 `eta_kj` 차이가 큰 coordinate는 선택된다.
- K=2의 `eta_2 - eta_1` penalty를 K>2로 확장한 형태다.

## 9. Refit

Refit은 variable selection 후 선택된 support만 고정하고, penalty 없이 다시 vMF mixture를 추정하는 절차다.

```text
1. penalized model 적합
2. 선택된 active coordinate S 결정
3. S 밖의 coordinate는 0으로 고정
4. S 안에서 penalty 없이 vMF mixture 재추정
```

즉 선택되지 않은 coordinate에서는 다음 제약을 둔다.

```math
\mu_{kj} = 0,
\qquad
j \notin S
```

해석은 다음과 같다.

- refit은 선택된 변수 집합을 바꾸지 않는다.
- 따라서 `TPR`, `FPR`, `Precision`, `F1`은 그대로다.
- 대신 penalty shrinkage를 줄여 `kappa`, `eta` 추정값을 개선할 수 있다.
- 에타 패널티에서는 refit 후 `eta contrast`, `kappa ratio`, `MSE_kappa`가 true value에 더 가까워지는지 확인하는 것이 중요하다.

## 10. Lambda 선택 방식

현재 simulation에서 tuning parameter는 정보기준 기반으로 선택했다. 기본 기준은 BIC이다.

```math
\mathrm{BIC}
=
\log(n)\mathrm{df}
-
2\ell(\hat{\Theta})
```

고차원 variable selection을 더 강하게 penalize하려면 EBIC도 함께 볼 수 있다.

```math
\mathrm{EBIC}
=
\left(\log(n)+2\gamma\log(d)\right)
\mathrm{df}
-
2\ell(\hat{\Theta})
```

### Rossi의 beta 선택

Rossi 방법은 단순 fixed grid가 아니라 path-following 방식으로 `beta` 후보를 만든다. 코드에서는 `beta=0`에서 시작하고, 현재 nonzero coordinate가 0이 되는 다음 threshold로 이동한다.

```math
\beta_{\mathrm{next}}
=
\min_{k,j:\ \kappa_k |r_{kj}|>\beta}
\kappa_k |r_{kj}|
```

각 `beta` 후보에서 EM을 적합하고, BIC가 가장 작은 모형을 선택했다.

### 분리 패널티의 lambda 선택

분리 패널티는 현재 이론적 path를 따로 유도하지 않았으므로 grid search를 사용했다.

K=2 concentration-driven simulation의 기본 grid는 다음이었다.

```text
lambda_mu    = {0, 100, 200, 300, 400, 500, 600}
lambda_kappa = {0, 10, 25, 50, 75}
```

K=4 paper-like pilot에서는 다음 grid를 사용했다.

```text
lambda_mu    = {0, 100, 200, 300, 400, 600}
lambda_kappa = {0, 5, 10, 25}
```

각 조합에 대해 EM을 돌리고, BIC가 가장 작은 조합을 선택했다.

### 에타 패널티의 lambda 선택

에타 패널티도 현재는 grid search를 사용했다.

K=2 concentration-driven simulation의 기본 grid는 다음이었다.

```text
lambda_eta = {0, 1, 2, 5, 10, 15, 20, 30, 40, 50}
```

K=4 pilot에서는 다음 grid를 사용했다.

```text
lambda_eta = {0, 1, 2, 5, 10, 20, 30}
```

현재 구현 기준에서는 K=2에서 `lambda_eta`가 커질수록 `delta_j = eta_2j - eta_1j`가 더 많이 0이 된다. 따라서 adaptive grid를 만든다면 다음 값들을 기준으로 잡을 수 있다.

```math
|\delta_j|
```

즉 dense 또는 weakly penalized fit에서 얻은 `|eta_2j - eta_1j|`의 분위수나 threshold를 사용해 `lambda_eta` 후보를 만들 수 있다.

K>2에서는 다음 값이 기준이 된다.

```math
\|c_{\cdot j}\|_2
```

논문 단계에서는 다음 순서가 자연스럽다.

```text
1. K를 고정한 비교에서는 모든 방법에 대해 BIC 또는 EBIC로 lambda를 선택한다.
2. high-dimensional variable selection 성격을 강조할 때는 EBIC를 함께 보고한다.
3. Rossi는 논문 방식과 맞추기 위해 beta path를 유지한다.
4. 분리 패널티와 에타 패널티는 lambda_max 기반 adaptive grid를 추가한다.
5. BIC 선택 결과와 EBIC 선택 결과가 같은지 sensitivity check를 한다.
```

## 11. R 코드 구현 구조

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

## 12. 비교한 방법

현재 비교 구조는 다음 6가지다.

| 방법 | penalty 기준 | selected support | refit |
|---|---|---|---|
| Rossi | `mu_k` | `mu_k`의 nonzero coordinate | 없음 |
| Rossi + refit | `mu_k` | Rossi support | 있음 |
| 분리 패널티 | `mu_k`, `kappa_k` | `mu_k`의 nonzero coordinate | 없음 |
| 분리 패널티 + refit | `mu_k`, `kappa_k` | 분리 패널티 support | 있음 |
| 에타 패널티 | `eta_k` contrast | `eta_2 - eta_1` 또는 centered eta support | 없음 |
| 에타 패널티 + refit | `eta_k` contrast | 에타 support | 있음 |

## 13. 평가지표

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

true active coordinate set을 `S`, 추정된 active coordinate set을 `S_hat`이라고 두면 다음과 같다.

```math
\hat{q}
=
|\hat{S}|
```

```math
\mathrm{TPR}
=
\frac{|S \cap \hat{S}|}{|S|}
```

```math
\mathrm{FPR}
=
\frac{|\hat{S} \setminus S|}{d-|S|}
```

```math
\mathrm{Precision}
=
\frac{|S \cap \hat{S}|}{|\hat{S}|}
```

```math
\mathrm{F1}
=
\frac{2\cdot \mathrm{Precision}\cdot \mathrm{TPR}}
{\mathrm{Precision}+\mathrm{TPR}}
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

K=2에서는 eta contrast MSE를 본다.

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

K>2에서는 centered eta 기준을 쓴다.

```math
\mathrm{MSE}_{\eta^c}
=
\frac{1}{Kd}
\sum_{k,j}
(\hat{\eta}_{kj}^{c} - \eta_{kj}^{c})^2
```

추정값 해석을 위해 다음도 같이 본다.

```text
kappa_1_hat
kappa_2_hat
kappa ratio
eta contrast norm
```

MSE 계산 전에는 label switching을 정리한다. K=2 concentration-driven simulation에서는 true parameter가 `kappa_1 < kappa_2` 구조이므로, 추정 component도 `kappa`가 작은 component와 큰 component 순서로 정렬한 뒤 비교한다. K>2 simulation에서는 true `mu_k`와 추정 `mu_k` 사이의 cosine similarity가 가장 커지는 permutation을 사용한다.

## 14. 현재까지의 해석

시뮬레이션 결과는 다음 방향을 보였다.

- concentration-driven setting에서는 에타 패널티가 가장 좋은 variable selection을 보였다.
- Rossi는 clustering은 잘하지만 false positive가 많았다.
- 분리 패널티는 Rossi보다 개선되지만, `kappa_k`가 coordinate-specific하지 않다는 한계가 있다.
- 에타 패널티는 FPR을 크게 줄이고 Precision/F1이 좋았다.
- refit은 support를 바꾸지는 않지만, shrinkage를 줄여 `kappa`와 `eta` 추정을 개선했다.

## 15. 참고문헌

본문에서 직접 연결되는 문헌은 다음과 같다.

| 내용 | 참고문헌 |
|---|---|
| vMF mixture EM과 kappa 근사 | Banerjee et al. (2005), `Clustering on the unit hypersphere using von Mises-Fisher distributions` |
| 방향자료와 vMF MLE 기본식 | Mardia and Jupp (2000), `Directional Statistics` |
| kappa inverse approximation 관련 논의 | Sra (2012), vMF parameter approximation 관련 note |
| sparse vMF mixture와 `mu_k` L1 penalty | Rossi and Barbaro (2022), `Mixture of von Mises-Fisher distribution with sparse prototypes` |
