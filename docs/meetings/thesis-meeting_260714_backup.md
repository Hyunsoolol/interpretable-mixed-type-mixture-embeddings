# 연구미팅 백업자료: 모형, 선택 기준, 지표 및 계산 검증 (2026-07-14)

이 문서는 연구미팅 본문에서 생략한 모형 정의와 구현 세부사항을 정리한다. 현재 결과의 주 모형은 E-CGL이며, E-ACGL은 adaptive 보조 확장이다.

## 1. 기호

$$
\eta_k=\kappa_k\mu_k,\qquad
\bar\eta_j=\frac{1}{K}\sum_{k=1}^K\eta_{kj},\qquad
c_{kj}=\eta_{kj}-\bar\eta_j.
$$

M 계열은 $\mu$의 prototype support를, E 계열은 centered eta contrast의 posterior decision support를 선택한다.

$$
S_M=\{j:\lVert\mu_{\cdot j}\rVert_2>0\},\qquad
S_E=\{j:\lVert c_{\cdot j}\rVert_2>0\}.
$$

## 2. 전체 비교 모형

### 2.1 내부 비교 모형

| 모형 | 패널티 대상 | 패널티 | group | adaptive | 선택 대상 | 역할 |
|:---|:---|:---|:---:|:---:|:---|:---|
| M-L | $\mu_{kj}$ | $\lambda_\mu\sum_{k,j}|\mu_{kj}|$ | 아니오 | 아니오 | prototype entry/union support | Rossi-type sparse prototype 기준 |
| M-GL | $\mu_{\cdot j}$ | $\lambda_\mu\sum_j\lVert\mu_{\cdot j}\rVert_2$ | 예 | 아니오 | prototype coordinate support | $\mu$ 공간 group 효과 비교 |
| M-AGL | $\mu_{\cdot j}$ | $\lambda_\mu\sum_jw_j^{(M)}\lVert\mu_{\cdot j}\rVert_2$ | 예 | 예 | prototype coordinate support | adaptive $\mu$-group 비교 |
| E-CL | $c_{kj}$ | $\lambda_\eta\sum_{k,j}|c_{kj}|$ | 아니오 | 아니오 | decision entry/union support | centered eta에서 group 효과 분리 |
| **E-CGL** | $c_{\cdot j}$ | $\lambda_\eta\sum_j\lVert c_{\cdot j}\rVert_2$ | 예 | 아니오 | posterior decision support | **주 제안 모형** |
| E-ACGL | $c_{\cdot j}$ | $\lambda_\eta\sum_jw_j^{(E)}\lVert c_{\cdot j}\rVert_2$ | 예 | 예 | posterior decision support | adaptive 보조 확장 |

Adaptive weight는 다음과 같다.

$$
w_j^{(M)}=
\left(\lVert\mu_{\cdot j}^{\mathrm{init}}\rVert_2+\epsilon\right)^{-\gamma},
\qquad
w_j^{(E)}=
\left(\lVert c_{\cdot j}^{\mathrm{init}}\rVert_2+\epsilon\right)^{-\gamma}.
$$

현재 실험은 $\gamma=1$, $\epsilon=10^{-6}$을 사용하고, weight의 median이 1이 되도록 정규화한다. 비적응형 M-GL과 E-CGL은 $w_j\equiv1$이다.

### 2.2 외부 비교 모형

| 모형 | 핵심 정의 | $\kappa$ | support | 현재 역할 |
|:---|:---|:---:|:---:|:---|
| Spherical k-means | cosine similarity 기반 hard clustering | 없음 | 없음 | clustering-only 기준 |
| Dense vMF mixture | $\sum_k\alpha_k C_d(\kappa_k)\exp(\kappa_k\mu_k^Tx)$ | component별 free | 없음 | 비희소 likelihood 기준 |
| Sparse k-means | clustering 목적함수에 feature weight 부여 | 없음 | feature support | 비모형 기반 feature-selection 기준 |
| dbmovMFs | vMF 기반 row/column co-clustering | 모형 의존 | column block | Rossi-style 보조 비교 후보 |

Sparse k-means와 dbmovMFs의 support는 centered eta decision support와 정의가 다르므로 ARI/NMI와 각 방법 고유의 support 지표를 구분해서 해석한다.

## 3. BIC 자유도와 refit 절차

### 3.1 현재 선택 기준

현재 simulation runner는 penalized path의 observed log-likelihood로 BIC를 계산한다.

$$
\mathrm{BIC}^{\mathrm{pen}}(\lambda)
=-2\ell(\widehat\Theta_{\lambda}^{\mathrm{pen}})
+\log(n)\,\mathrm{df}_{\lambda}.
$$

패널티를 뺀 observed log-likelihood $\ell$을 사용하며, BIC가 최소인 path row의 support를 선택한다.

### 3.2 현재 구현의 자유도

$s_k=|\{j:\widehat\mu_{kj}\ne0\}|$, $m=|\widehat S|$, $r_j=|\{k:\widehat c_{kj}\ne0\}|$로 둔다.

| 모형 | path 선택에 사용되는 구현 자유도 |
|:---|:---|
| M-L | $(2K-1)+\sum_k\max(1,s_k-1)$ |
| M-GL, M-AGL | $(K-1)+K+K\max(m-1,1)$ |
| E-CL | $(K-1)+d+\sum_j\max(r_j-1,0)$ |
| E-CGL, E-ACGL | $(K-1)+d+(K-1)m$ |

E-CGL/E-ACGL의 자유도는 다음 세 항으로 해석한다.

$$
\underbrace{K-1}_{\text{mixing proportions}}
+\underbrace{d}_{\text{common eta baseline}}
+\underbrace{(K-1)m}_{\text{active centered contrasts}}.
$$

이는 exact effective degrees of freedom가 아니라 path 선택을 위한 implementation-level approximation이다. $m>0$에서는

$$
d+(K-1)m+(K-1)\mathbf1(m>0)
$$

와 같고, $m=0$에서만 $K-1$만큼 차이가 난다.

### 3.3 선택과 refit 순서

1. 각 $\lambda$에서 penalized model을 적합한다.
2. path row별 support, observed log-likelihood와 BIC를 계산한다.
3. $\widehat\lambda=\arg\min_\lambda\mathrm{BIC}^{\mathrm{pen}}(\lambda)$를 선택한다.
4. $\widehat S_{\widehat\lambda}$를 고정한다.
5. 선택 좌표 밖을 0으로 고정하고 penalty 없이 $\alpha_k,\mu_k,\kappa_k$를 다시 추정한다.
6. ARI, support 지표와 MSE는 refit 결과로 보고한다.

따라서 현재 규칙은 **BIC-before-refit 선택 후 support refit**이다. Refit은 support를 바꾸지 않고 shrinkage bias를 줄이는 단계다.

### 3.4 구현 감사에서 확인된 주의점

- Path 선택용 centered eta BIC와 final refit row에 다시 기록되는 df/BIC는 동일하지 않을 수 있다.
- Centered eta path의 df는 common baseline $d$를 포함하지만, 현재 support refit은 inactive coordinate를 0으로 고정한다.
- 기존 rep=50 결과에는 path candidate가 저장되지 않아 BIC-after-refit으로 사후 재선택할 수 없다.
- 논문 제출 전 BIC-before-refit, BIC-after-refit, combinatorial EBIC와 df sensitivity를 분리해 검증한다.

현재 main 결과는 BIC-before-refit으로 명시한다. Adaptive weight의 $\gamma=1$과 일부 EBIC 구현의 $\gamma=0.5$는 서로 다른 매개변수다.

## 4. Rossi와 Li 논문의 역할

| 항목 | Rossi and Barbaro (2022) | Li et al. (2022) | 본 연구 |
|:---|:---|:---|:---|
| 자료/모형 | directional data, vMF mixture | finite mixture regression | directional data, vMF mixture |
| 핵심 모수 | $\mu_k$ | scale-adjusted regression effect | $\eta_k=\kappa_k\mu_k$ |
| 구조 | sparse prototype | common effect + sum-to-zero cluster deviation | common eta + centered eta contrast |
| 패널티 | $\beta\sum_k\lVert\mu_k\rVert_1$ | $\lambda\sum_{j,k}w_{jk}|\beta_{jk}|$ | $\lambda_\eta\sum_jw_j\lVert c_{\cdot j}\rVert_2$ |
| group 선택 | 아니오 | 아니오, entry-wise $L_1$ | coordinate-wise group $L_2$ |
| adaptive | 사용하지 않음 | 사용, 수치실험 $\gamma=1$ | E-ACGL에서 선택적 사용, $\gamma=1$ |
| 선택 목표 | sparse prototype 해석 | source of heterogeneity | posterior decision support |
| 모형 선택 | path following과 BIC, $K$/sparsity 분리 논의 | BIC로 component와 tuning 탐색 | $K$와 $\lambda_\eta$의 2단계 선택 검토 |

Rossi and Barbaro는 vMF mixture와 sparse $\mu$ 추정의 직접 비교 기준이다. Li et al.은 common effect와 cluster-specific deviation의 효과모형 분해 및 adaptive weight 구성의 구조적 근거다. 본 연구는 Li et al.의 회귀모형이나 이론을 그대로 적용한 것이 아니라, 이 분해를 vMF 자연모수의 centered contrast에 맞게 구성한다.

참고문헌:

- [Rossi, F. and Barbaro, F. (2022), *Mixture of von Mises-Fisher distribution with sparse prototypes*](https://doi.org/10.1016/j.neucom.2022.05.118)
- [Li, Y. et al. (2022), *Pursuing Sources of Heterogeneity in Modeling Clustered Population*](https://academic.oup.com/biometrics/article/78/2/716/7460076)

## 5. Conditional F1과 unconditional F1

반복 $r$에서 $TP_r,FP_r,FN_r$를 계산한다. 전체 반복 기준 micro F1은 zero-support 반복을 포함한다.

$$
F_{1,\mathrm{all}}
=\frac{2\sum_{r=1}^RTP_r}
{2\sum_{r=1}^RTP_r+\sum_{r=1}^RFP_r+\sum_{r=1}^RFN_r}.
$$

Nonzero support가 선택된 반복 집합 $V=\{r:\widehat q_r>0\}$만 사용하면

$$
F_{1,\mathrm{valid}}
=\frac{2\sum_{r\in V}TP_r}
{2\sum_{r\in V}TP_r+\sum_{r\in V}FP_r+\sum_{r\in V}FN_r}.
$$

Zero-support 반복에서는 $TP_r=FP_r=0$, $FN_r=q_{\mathrm{true}}$이므로 $F_{1,\mathrm{all}}$에는 선택 실패가 반영된다. 기존 summary의 `mean(..., na.rm=TRUE)`는 F1이 NA인 zero-support 반복을 제외할 수 있으므로 어려운 설정에서는 conditional 결과가 된다.

S4-N의 E-ACGL 예시는 다음과 같다.

| 반복 수 | valid support | zero support | selected q (all) | F1 all | F1 valid | ARI valid |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 10 | 40 | 16.700 | 0.331 | 0.979 | 0.629 |

$F_{1,\mathrm{valid}}=0.979$만 제시하면 40회의 zero-support 선택이 제외된다. 따라서 zero-support가 있는 결과는 `F1 all`, `F1 valid`, valid-support rate를 함께 보고한다. ARI와 MSE가 refit 성공 반복에서만 정의되면 `valid refit 기준`으로 표시한다.

## 6. 계산 시간과 Rcpp 동일성 검증

### 6.1 여섯 모형 1회 실행 시간

설정은 $K=4$, $n=1000$, $d=200$, nstart=10, path length=240, max_iter=100이며 Rcpp helper를 사용했다. 단일 반복의 diagnostic timing이다.

| 모형 | elapsed sec | selected q | ARI | F1 | MSE_eta |
|:---|---:|---:|---:|---:|---:|
| M-L | 3.750 | 200 | 0.859 | 0.148 | 0.698 |
| M-GL | 8.820 | 20 | 0.895 | 0.889 | 0.074 |
| M-AGL | 8.530 | 20 | 0.895 | 0.889 | 0.074 |
| E-CL | 8.530 | 200 | 0.868 | 0.148 | 0.697 |
| **E-CGL** | **5.620** | **16** | **0.897** | **1.000** | **0.054** |
| E-ACGL | 5.530 | 16 | 0.897 | 1.000 | 0.054 |

단일 반복 시간이므로 모형의 이론적 계산복잡도 순위로 해석하지 않는다. Runtime은 df보다 EM 반복 수, path 길이, line search와 선택 support에 더 직접적으로 영향을 받는다.

### 6.2 R-only와 Rcpp-helper 결과 동일성

Guarded switch 검증은 동일 알고리즘에서 low-level helper만 교체했다. Rcpp는 새로운 추정법이 아니다.

| 검증 항목 | 결과 |
|:---|---:|
| rep=20 raw rows | 120 vs 120 |
| rep=20 summary rows | 6 vs 6 |
| character columns | identical |
| raw maximum numeric difference | $1.019\times10^{-10}$ |
| summary maximum numeric difference | $9.823\times10^{-11}$ |
| difference $>10^{-8}$ | 0 |
| 판정 | PASS |

### 6.3 반복 runtime benchmark

설정은 $K=4$, $n=300$, $d=60$, rep=50, nstart=3, max_iter=50, path steps=40이다. `sourceCpp(cacheDir=..., rebuild=FALSE)` warm cache를 사용했고 cache warm-up 6.310초는 측정에서 분리했다.

| mode | repeats | mean sec | sd sec | median sec | min-max sec |
|:---|---:|---:|---:|---:|:---|
| R-only | 3 | 59.737 | 0.342 | 59.860 | 59.35-60.00 |
| Rcpp-helper | 3 | 25.387 | 0.031 | 25.380 | 25.36-25.42 |

Median OFF/ON ratio는 2.359이며, 이 설정에서 elapsed time은 약 57.6% 감소했다. 이는 diagnostic benchmark이고 논문용 일반 속도 향상 주장으로 사용하지 않는다. R-only fallback은 유지한다.

## 7. 질문별 요약

| 질문 | 답변 |
|:---|:---|
| 왜 M 계열과 E 계열의 selected q가 다른가? | M 계열은 prototype support, E 계열은 component 간 posterior score 차이를 만드는 decision support를 선택한다. |
| 왜 E-CGL이 주 모형인가? | 자연모수의 centered contrast를 coordinate 단위로 직접 선택하며 adaptive 초기값 의존성이 없다. |
| E-ACGL은 왜 보조인가? | 신호별 차등 shrinkage가 유리할 수 있지만 초기 dense fit과 weight 구성에 추가로 의존한다. |
| BIC는 refit 후 계산하는가? | 현재 support 선택은 penalized path의 BIC-before-refit이며, 선택 후 support-constrained refit을 수행한다. |
| BIC df는 exact한가? | 아니다. 현재 모형 선택을 위한 implementation-level approximation이다. |
| S4-N의 F1이 0.979인가 0.331인가? | 0.979는 valid 10회 기준, 0.331은 zero-support 40회를 포함한 전체 50회 기준이다. |
| Rcpp가 결과를 바꾸는가? | 검증 허용오차 $10^{-8}$에서 동일하며, 최대 차이는 약 $10^{-10}$이다. |

## 8. 근거 파일

- 모형 및 simulation runner: `r/simulation/paper_eta_first_s1_run_260702.r`
- 공통 fitting/df/refit 함수: `r/methods/rb2022_k4_pilot_compare_run.r`
- Rossi reproduction: `r/methods/rossi_barbaro_2022_reproduction.r`
- BIC/df 감사: `results/bic_df_audit_260708/bic_df_posthoc_audit_notes.md`
- S4-N 지표 재검산: `results/paper_eta_s4n_metric_audit_260708/s4n_metric_recheck_notes.md`
- 여섯 모형 timing: `results/runtime_six_methods_one_rep_260703/six_methods_one_rep_timing.md`
- Rcpp equality: `results/rcpp_switch_validation_rep20_260708/rcpp_switch_rep20_validation_notes.md`
- Rcpp 반복 benchmark: `results/rcpp_vs_r_runtime_benchmark_rep50_260708/runtime_benchmark_notes.md`
