# 연구미팅 내용 공부 정리

## 1. 기본 모형: vMF 분포

방향자료 `x_i`는 단위구면 위의 벡터다.

$$
x_i \in S^{d-1}, \qquad \|x_i\|_2 = 1
$$

vMF density는 다음과 같다.

$$
f(x_i \mid \mu_k, \kappa_k)
=
C_d(\kappa_k)
\exp(\kappa_k \mu_k^\top x_i)
$$

여기서

- `mu_k`: k번째 군집의 평균 방향
- `kappa_k`: 집중도, 값이 클수록 `mu_k` 주변에 자료가 강하게 모임
- `C_d(kappa_k)`: 정규화 상수

참고: Banerjee et al. (2005), Rossi & Barbaro (2022)

## 2. vMF Mixture Model

군집이 `K`개일 때 mixture likelihood는 다음과 같다.

$$
f(x_i)
=
\sum_{k=1}^{K}
\alpha_k C_d(\kappa_k)
\exp(\kappa_k \mu_k^\top x_i)
$$

추정해야 할 값은 다음이다.

$$
\Theta =
\{\alpha_k, \mu_k, \kappa_k\}_{k=1}^{K}
$$

EM algorithm에서는 posterior responsibility를 계산한다.

$$
\tau_{ik}
=
\frac{
\alpha_k C_d(\kappa_k)\exp(\kappa_k \mu_k^\top x_i)
}{
\sum_{\ell=1}^{K}
\alpha_\ell C_d(\kappa_\ell)
\exp(\kappa_\ell \mu_\ell^\top x_i)
}
$$

그리고

$$
N_k = \sum_i \tau_{ik},
\qquad
r_k = \sum_i \tau_{ik} x_i
$$

를 이용해 M-step을 수행한다.

## 3. Rossi & Barbaro (2022) Sparse vMF

Rossi & Barbaro (2022)는 `mu_k`에 L1 penalty를 둔다.

$$
\ell_p(\Theta)
=
\ell(\Theta)
-
\beta
\sum_{k=1}^{K}
\|\mu_k\|_1
$$

핵심 update는 soft-thresholding 형태다.

$$
\mu_{kj}
\propto
\mathrm{sign}(r_{kj})
\left(\kappa_k |r_{kj}| - \beta\right)_+
$$

그 후 정규화한다.

$$
\|\mu_k\|_2 = 1
$$

집중도는 다음 근사식을 이용한다.

$$
\rho_k =
\frac{\mu_k^\top r_k}{N_k}
$$

$$
\kappa_k
\approx
\frac{d\rho_k - \rho_k^3}{1-\rho_k^2}
$$

해석:

- Rossi 방법은 sparse prototype을 만든다.
- 즉, 각 군집의 평균 방향 `mu_k`에서 어떤 coordinate가 중요한지 보여준다.
- 하지만 posterior decision에는 `mu_k`만 들어가는 것이 아니라 `eta_k = kappa_k mu_k`가 들어간다.

## 4. Posterior Decision과 Eta

vMF mixture에서 군집 선택은 다음 값을 비교하는 것과 관련된다.

$$
\log \alpha_k
+
\log C_d(\kappa_k)
+
\kappa_k \mu_k^\top x_i
$$

여기서 핵심 선형항은

$$
\eta_k^\top x_i
$$

이고,

$$
\eta_k = \kappa_k \mu_k
$$

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

$$
Q_{\mathrm{pen}}
=
\ell(\Theta)
-
\lambda_\mu
\sum_k \|\mu_k\|_1
-
\lambda_\kappa
\sum_k \kappa_k
$$

`mu_k` update는 다음과 같다.

$$
z_{kj}
=
\mathrm{sign}(r_{kj})
\left(\kappa_k |r_{kj}| - \lambda_\mu\right)_+
$$

$$
\mu_k =
\frac{z_k}{\|z_k\|_2}
$$

`kappa_k` update는 다음 형태다.

$$
\rho_k =
\frac{\mu_k^\top r_k - \lambda_\kappa}{N_k}
$$

$$
\kappa_k = A_d^{-1}(\rho_k)
$$

해석:

- `mu_k` sparsity와 `kappa_k` shrinkage를 분리해서 볼 수 있다.
- 하지만 `kappa_k`는 coordinate별 값이 아니라 component-level scalar다.
- 따라서 concentration 차이가 어떤 coordinate에서 발생했는지를 직접 선택하지는 못한다.

## 7. Eta Penalty EM

제안 방향은 posterior decision에 직접 들어가는 `eta_k`에 penalty를 두는 것이다.

$$
\eta_k = \kappa_k \mu_k
$$

K=2일 때는 두 군집의 차이를 다음으로 본다.

$$
\Delta \eta =
\eta_2 - \eta_1
$$

penalty는 다음과 같다.

$$
\ell_{\mathrm{pen}}
=
\ell(\Theta)
-
\lambda_\eta
\|\eta_2 - \eta_1\|_1
$$

해석:

- `eta_2j - eta_1j`가 0이면 coordinate `j`는 군집 구분에 직접 기여하지 않는다고 본다.
- concentration-driven setting에서는 이 기준이 `mu` 기준보다 자연스럽다.

## 8. K=4에서 Eta Penalty 일반화

Rossi 2022 paper-like setting은 `K=4`다. 이 경우 `eta_2 - eta_1`만으로는 부족하다.

그래서 coordinate별 centered eta contrast를 쓴다.

$$
\bar{\eta}_{j}
=
\frac{1}{K}
\sum_{k=1}^{K}
\eta_{kj}
$$

$$
\eta_{kj}^{c}
=
\eta_{kj}
-
\bar{\eta}_{j}
$$

coordinate `j`의 component 간 차이는 다음으로 본다.

$$
\left(
\sum_{k=1}^{K}
(\eta_{kj}^{c})^2
\right)^{1/2}
$$

K=4 eta penalty는 다음 방향이다.

$$
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
$$

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

$$
ARI
$$

Variable selection:

```text
selected q
TPR
FPR
Precision
F1
```

Parameter estimation:

$$
\mathrm{MSE}_{\mu}
=
\frac{1}{Kd}
\sum_{k,j}
(\hat{\mu}_{kj} - \mu_{kj})^2
$$

$$
\mathrm{MSE}_{\kappa}
=
\frac{1}{K}
\sum_k
(\hat{\kappa}_k - \kappa_k)^2
$$

K=2에서는

$$
\mathrm{MSE}_{\Delta \eta}
=
\frac{1}{d}
\sum_j
\left[
(\hat{\eta}_{2j}-\hat{\eta}_{1j})
-
(\eta_{2j}-\eta_{1j})
\right]^2
$$

K=4에서는 centered eta 기준을 쓴다.

$$
\mathrm{MSE}_{\eta^c}
=
\frac{1}{Kd}
\sum_{k,j}
(\hat{\eta}_{kj}^{c} - \eta_{kj}^{c})^2
$$

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
