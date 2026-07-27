# CSDA 투고 논문 구성안 (2026-07-29)

## 논문 중심

| 모형 | 추정 대상 | 논문 내 역할 |
|---|---|---|
| E-CGL | posterior-score heterogeneity $S_\eta$ | 주 제안 모형 |
| M-CGL | directional heterogeneity $S_\mu$ | matched directional companion |
| E-ACGL, M-ACGL | adaptive weighted support | 보조 확장 |

논문의 중심 증거는 target-specific support recovery, oracle-support gap,
통계적·수치적 안정성이다. Dense vMF와 Rossi sparse prototype은 최소
비교군으로 유지한다.

## 1. Introduction

### 1.1 High-dimensional directional clustering
### 1.2 Limitation of sparse prototype selection
### 1.3 Directional and posterior-score heterogeneity
### 1.4 Main contributions

서론에서는 support 기호를 공식 정의하지 않고 세 추정 대상을 개념적으로
설명한다.

## 2. vMF Mixtures and Coordinate Estimands

### 2.1 vMF mixture and natural parameters
### 2.2 Pairwise posterior log-scores
### 2.3 Prototype, directional, and posterior-score supports
### 2.4 Common baselines and centered contrasts

$$
\eta_k=\kappa_k\mu_k,
\qquad
c_{kj}^{(\mu)}=\mu_{kj}-\bar\mu_j,
\qquad
c_{kj}^{(\eta)}=\eta_{kj}-\bar\eta_j
$$

## 3. Centered Group Regularization

### 3.1 E-CGL objective
### 3.2 E-ACGL adaptive extension
### 3.3 M-CGL directional companion
### 3.4 Target-preserving refit estimand and constraints
### 3.5 Stabilized objective and well-posedness

M-CGL의 estimand, 목적함수와 단위구면 제약은 본문에 유지하고 전체
ADMM·manifold 계산은 Supplement에 둔다.

## 4. Theoretical Properties of the Estimands

### 4.1 Posterior-score cancellation
### 4.2 Common-concentration support equivalence
### 4.3 Heterogeneous-concentration support divergence
### 4.4 Direction–concentration decomposition
### 4.5 Pairwise-dispersion identity
### 4.6 Label invariance

$$
\kappa_1=\cdots=\kappa_K
\quad\Longrightarrow\quad
S_\mu=S_\eta
$$

이 절에는 proximal map, ADMM 또는 line search를 포함하지 않는다.

## 5. Computation and Model Selection

### 5.1 E-CGL guarded proximal generalized EM
### 5.2 Closed-form centered group proximal update
### 5.3 Regularization path, initialization, and warm starts
### 5.4 Step-halving, stopping criteria, and numerical safeguards
### 5.5 Support-constrained refit algorithm and BIC-after-refit
### 5.6 M-CGL computation overview
### 5.7 Computational complexity and stationarity criteria

M-CGL의 전체 반복식과 상세 residual 진단은 Supplement에 두고 본문에는
고수준 계산 절차와 정지조건을 제시한다. 수렴 성공률·실패율·대표
실행시간은 6.4에, residual·objective trace와 Rcpp 동일성은 Supplement에
둔다.

## 6. Simulation Studies

### 6.1 Design, comparators, and evaluation criteria

- Dense vMF
- Rossi sparse prototype
- M-CGL
- E-CGL
- Oracle-$S_\mu$와 Oracle-$S_\eta$
- Target-specific $F_1$, ARI, test NLL, parameter MSE

### 6.2 Estimand-specific recovery

- Common $\kappa$
- Pure concentration heterogeneity
- Shared canonical background
- Crossed support

### 6.3 Statistical recovery and oracle benchmark

- 표본 크기별 exact support recovery
- 모수 MSE
- Oracle-support gap
- 공통·이질적 $\kappa$ 분리 결과

### 6.4 Benchmarks, computation, and limitations

- 기존 방법과의 최소 비교
- Dense weak-support negative control
- 수렴 성공률과 실패율
- 대표 실행시간

## 7. Real-Data Analysis

### 7.1 Classic3 data and SPLADE preprocessing
### 7.2 Clustering and support-selection results
### 7.3 Estimated concentrations
### 7.4 Directional and posterior-score contrasts
### 7.5 Selected tokens and common baselines
### 7.6 Contrasting data analysis

## 8. Discussion and Conclusion

### 8.1 Main findings
### 8.2 When to use M-CGL or E-CGL
### 8.3 Limitations under dense support and weak signals
### 8.4 Limitations of selecting $K$ and information criteria
### 8.5 Future methodological extensions

## Supplement

### S1. Proofs
### S2. Full M-CGL and M-ACGL algorithms
### S3. ADMM, manifold, and proximal convergence diagnostics
### S4. Full simulation results
### S5. Adaptive-penalty results
### S6. BIC, EBIC, degrees-of-freedom, and path sensitivity
### S7. Selection of $K$ and misspecification
### S8. Rcpp equality and runtime validation
### S9. Additional real-data and negative-control results

## 본문과 Supplement의 경계

| 본문 | Supplement |
|---|---|
| E-CGL 주 알고리즘 | 전체 tolerance와 objective trace |
| M-CGL estimand와 목적함수 | M-CGL ADMM·manifold 세부 |
| 공통·이질적 $\kappa$ 핵심 결과 | 전체 cell과 adaptive 결과 |
| Oracle-support gap | path-oracle과 selector sensitivity |
| 수렴률·실패율·대표 실행시간 | Rcpp 동일성과 상세 benchmark |
| Dense vMF와 Rossi 최소 비교 | 외부 clustering baseline |
