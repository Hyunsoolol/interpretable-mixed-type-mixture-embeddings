# 고정 반복 홀드아웃 실자료 분석

## 1. 분석 범위

실자료 분석의 목적은 centered coordinate regularization이 홀드아웃 군집
구조를 유지하면서 어휘 정렬 방향 표현의 좌표를 줄일 수 있는지 확인하는
것이다. 선택된 좌표를 참 유의 변수로 해석하지 않는다.

데이터셋, 표현 방법, 상위 $d$ 설정은 탐색적 검토 이후 고정하였다. 따라서
5개의 고정 분할은 독립적으로 사전등록된 확증 연구가 아니라, 고정된 분석
조건에서 수행한 반복 홀드아웃 검증이다.

- **Classic3:** 희소하고 해석 가능한 posterior-score contrast의 주 사례
- **BBCSport:** 유용한 정보가 보다 조밀하게 분포한 대조 사례
- **CSTR:** Rossi 방식 구현의 재현성을 확인하는 문헌 연결 사례. 이전
  E 계열 진단 결과는 이번 고정 분석과 통합하지 않음

## 2. 고정 분석 설계

| 데이터셋 | $K$ | 원자료 $n$ | 완전 중복 제거 | 근접 중복 제거 | 최종 $n$ | 분할별 학습/평가 | $d$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classic3 | 3 | 3,890 | 2 | 5 | 3,883 | 3,105 / 778 | 2,000 |
| BBCSport | 5 | 737 | 13 | 20 | 704 | 561 / 143 | 500 |

Classic3는 [Banerjee et al. (2005)](https://jmlr.org/papers/v6/banerjee05a.html)의
방향 군집 연구에서 사용된 CISI, CRAN, MED 문서 집합으로 구성된다.
BBCSport는 [Greene and Cunningham (2006)](https://doi.org/10.1145/1143844.1143892)의
문서 군집 벤치마크와 연결되는 육상, 크리켓, 축구, 럭비, 테니스 기사로
구성된다.

두 데이터셋 모두 계층화된 80/20 분할 5개를 고정하여 사용하였다. 근접
중복 문서는 분할 전에 제거하였고, 어휘 순위는 각 분할의 학습자료에서만
계산하였다. 평가 문서와 학습 문서 사이의 최대 코사인 유사도가 0.98 이상인
경우는 없었다.

문서는 `naver/splade-cocondenser-ensembledistil`의 revision
`49cf4c7b0db5b870a401ddf5e2669993ef3699c7`로 생성한 비음수 SPLADE
좌표로 표현한 뒤 단위구면 위에서 행 정규화하였다. 어휘는 학습자료의
분산을 기준으로 Classic3에서는 상위 2,000개, BBCSport에서는 상위
500개 좌표를 사용하였다.

적합 과정에서는 벤치마크의 군집 수를 고정하였다. 제공된 class label은
계층화 분할, $K$ 확인, 적합 후 ARI/NMI 평가, 성분 명명 및 시각화에만
사용하였다.

## 3. 비교 방법과 추정 대상

| 방법 | 집중도 | 좌표 추정 대상 |
|---|---|---|
| Spherical $k$-means | 추정하지 않음 | 조밀한 군집 기준 |
| Dense vMF | 공통 $\kappa$ | 조밀한 우도 기준 |
| Dense vMF | 자유로운 $\kappa_k$ | 조밀한 우도 기준 |
| M-L | 공통 $\kappa$ | Rossi 방식 prototype support |
| M-CGL | 자유로운 $\kappa_k$ | centered directional support |
| M-ACGL | 자유로운 $\kappa_k$ | adaptive centered directional support |
| E-CGL | 자유로운 $\kappa_k$ | centered posterior-score support |
| E-ACGL | 자유로운 $\kappa_k$ | adaptive centered posterior-score support |

세 희소 모형의 추정 대상은 서로 다르다.

$$
S_P = \{j : \|\mu_{\cdot j}\|_2 > 0\}.
$$

$$
S_{\mu} = \{j : \|\mu_{\cdot j} - \bar{\mu}_j\mathbf{1}_K\|_2 > 0\}.
$$

$$
S_{\eta} = \{j : \|\eta_{\cdot j} - \bar{\eta}_j\mathbf{1}_K\|_2 > 0\}.
$$

따라서 selected $q$는 M-L, M-CGL, E-CGL에 공통된 하나의 estimand가
아니라 각 방법의 추정 대상 안에서 해석한다.

## 4. 추정 및 평가

- Dense 및 spherical 초기화에는 30개의 무작위 시작값을 사용하였다.
  Penalized path는 선택된 matched dense fit에서 warm start하였다.
- Centered M/E path는 240개 후보를 사용하였다.
- M-L path는 600개 후보를 사용하였다.
- 희소 모형은 target-preserving support-constrained refit 이후 학습자료
  observed log-likelihood로 계산한 방법별 BIC를 사용하여 선택하였다.
- Paired comparison의 dense 기준은 M-L에는 shared-$\kappa$, M-CGL,
  M-ACGL, E-CGL, E-ACGL에는 free-$\kappa_k$를 사용하였다.
- 평가 지표는 held-out ARI, held-out NMI, 문서당 held-out negative
  log-likelihood, selected $q$, 조건부 support Jaccard, 실행시간, warning,
  수렴 여부, path 경계 선택 여부이다.

활성 집합의 크기를 $m$이라고 할 때 centered-$\eta$의 명목 차원은 다음과
같다.

$$
\mathrm{df}_{\eta} = d + (K-1)m + (K-1)\mathbf{1}(m>0).
$$

Centered-$\mu$의 명목 차원은 다음과 같다.

$$
\mathrm{df}_{\mu} =
\begin{cases}
d + 2K - 2, & m=0,\\
d + (K-1)m + K - 1, & m>0.
\end{cases}
$$

M-L은 Rossi 방식 구현에 따라 활성 prototype entry 수를 사용하였다.
비활성 centered-$\eta$ 좌표는 공통 자연모수 baseline을 유지하고, 비활성
centered-$\mu$ 좌표는 행 단위노름 제약 아래 공통 방향 좌표를 유지한다.
이 차원은 비정규 finite mixture에서 support 선택을 위한 실용적 명목
차원이며, 정확한 marginal-likelihood 자유도가 아니다.

5개 홀드아웃은 서로 중복된다. 따라서 평균, 표준편차, paired
win/tie/loss는 기술통계로 제시하며 독립 반복을 가정한 $t$-검정은
수행하지 않았다.

실행시간은 분할별로 기록된 전체 pipeline 시간의 평균이다. Dense
초기화, path 구성, refit을 모두 포함한다. Intel i7-11700K, 32 GB
memory, R 4.2.1 환경에서 최대 6개 분할을 병렬 실행하였다. M-L과
centered path는 path 길이와 최적화 방법이 다르므로, 실행시간은 계산
복잡도의 직접 비교가 아니라 실제 workload 기록이다.

## 5. Classic3

| 방법 | Held-out ARI | Held-out NMI | Selected $q$ | $q/d$ | NLL/doc | Pipeline (초) |
|---|---:|---:|---:|---:|---:|---:|
| Spherical $k$-means | 0.970 (0.007) | 0.946 (0.010) | 2,000.0 | 1.000 | NA | 18.7 |
| Dense shared-$\kappa$ | 0.970 (0.007) | 0.946 (0.010) | 2,000.0 | 1.000 | -4,872.890 | 47.6 |
| Dense free-$\kappa_k$ | 0.973 (0.006) | 0.953 (0.011) | 2,000.0 | 1.000 | -4,873.968 | 60.6 |
| M-L | 0.970 (0.009) | 0.947 (0.013) | 1,924.6 | 0.962 | -4,872.239 | 2,320.7 |
| M-CGL | 0.970 (0.007) | 0.949 (0.011) | 1,376.8 | 0.688 | -4,873.337 | 6,063.2 |
| M-ACGL | 0.971 (0.007) | 0.950 (0.011) | 1,345.4 | 0.673 | -4,873.298 | 6,669.8 |
| **E-CGL** | **0.970 (0.007)** | **0.947 (0.012)** | **1,343.0** | **0.671** | **-4,873.297** | **2,429.6** |
| E-ACGL | 0.973 (0.006) | 0.953 (0.011) | 2,000.0 | 1.000 | -4,873.967 | 2,325.6 |

Dense free-$\kappa_k$와 비교하면 E-CGL은 공통 baseline을 유지하면서
추정된 posterior-score contrast support에서 좌표의 32.9%를 제외하였다.
평균 paired difference는 다음과 같다.

$$
\Delta\mathrm{ARI} = -0.0036,\qquad
\Delta\mathrm{NLL/doc} = 0.6705.
$$

ARI는 5개 분할 중 1개에서 동률이고 4개에서 낮았으며, NLL은 5개
분할에서 모두 높았다. 평균 ARI 감소는 0.0036, 문서당 held-out NLL
증가는 0.6705였다. 두 학습 어휘에 모두 존재하는 좌표에 한정한 평균
support Jaccard는 0.933이었다. E-ACGL은 모든 분할에서 dense support를
선택하였으므로 주 적합 결과가 아니라 adaptive sensitivity 결과로
유지하였다.

E-CGL의 centered-$\eta$ contrast는 고정 분할 전체에서 안정적이었다.
적합 후 class label과 성분을 정렬하여 이름과 시각화에 사용하였다. 5개
분할에서 모두 존재하고 선택된 주요 token은 다음과 같다.

| Class | 양의 contrast가 큰 token |
|---|---|
| CISI | library, information, librarian, libraries, retrieval |
| CRAN | flow, mach, pressure, heat, theory |
| MED | tumor, inhibitor, rat, dose, cancer |

![Classic3 centered-Eta contrast](figures/classic3_locked_ecgl_centered_eta_heatmap_260730.png)

## 6. BBCSport 대조 사례

| 방법 | Held-out ARI | Held-out NMI | Selected $q$ | $q/d$ | NLL/doc | Pipeline (초) |
|---|---:|---:|---:|---:|---:|---:|
| Spherical $k$-means | 0.894 (0.022) | 0.897 (0.021) | 500.0 | 1.000 | NA | 1.5 |
| Dense shared-$\kappa$ | 0.898 (0.024) | 0.901 (0.020) | 500.0 | 1.000 | -922.870 | 15.2 |
| Dense free-$\kappa_k$ | 0.877 (0.048) | 0.878 (0.033) | 500.0 | 1.000 | -922.622 | 15.8 |
| M-L | 0.907 (0.020) | 0.907 (0.019) | 498.2 | 0.996 | -921.605 | 116.7 |
| M-CGL | 0.880 (0.050) | 0.882 (0.035) | 303.2 | 0.606 | -921.086 | 1,516.6 |
| M-ACGL | 0.880 (0.050) | 0.882 (0.035) | 302.4 | 0.605 | -920.979 | 1,533.0 |
| **E-CGL** | **0.875 (0.053)** | **0.874 (0.042)** | **308.6** | **0.617** | **-921.211** | **491.6** |
| E-ACGL | 0.877 (0.048) | 0.878 (0.033) | 285.2 | 0.570 | -920.796 | 530.4 |

E-CGL은 추정된 posterior-score contrast support에서 좌표의 38.3%를
제외하였다. Dense free-$\kappa_k$ 대비 평균 paired difference는 다음과
같다.

$$
\Delta\mathrm{ARI} = -0.0026,\qquad
\Delta\mathrm{NLL/doc} = 1.4105.
$$

모든 희소 방법은 5개 분할 모두에서 각 방법과 대응되는 dense model보다
held-out NLL이 높았다. M-L의 평균 ARI는 0.907이었고, 평가된 path
candidate 중 99.6%의 좌표를 유지하였다. 따라서 BBCSport는
BIC로 선택된 희소 적합이 모든 분할에서 held-out density 손실을 보인
대조 사례이다.

M-L은 shared $\kappa$, centered M/E 방법은 free $\kappa_k$를
사용하므로 family 간 단순 순위만으로 penalty 효과를 분리할 수 없다.
따라서 sparse-minus-matched-dense paired difference를 주 비교로
사용한다.

## 7. 수치 검증

| 데이터셋 | 완료 행 | 오류 | 미수렴 | 경계 후보 선택 | Path fit 중단 | 알려진 Bessel warning 행 | 기타 warning 행 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classic3 | 40/40 | 0 | 0 | 0 | 0 | 10 | 0 |
| BBCSport | 40/40 | 0 | 0 | 0 | 5 | 0 | 0 |

Classic3의 M-CGL/M-ACGL에서는 고차 Bessel 정밀도 warning이 발생하였다.
선택된 적합 범위인 $d=2{,}000$, $\kappa\in[607,818]$에서 별도 감사를
수행하여 production direct/fallback 계산과 고차원 reference 계산을
비교하였다.

$$
\max |\Delta A_d(\kappa)| = 8.983\times10^{-11}.
$$

$$
\max \{\mathrm{relative\ error}\} = 2.850\times10^{-10}.
$$

관측 범위는 사전에 정한 상대오차 허용 기준 $10^{-6}$을 통과하였다.
이는 관측된 범위에 대한 수치 감사 결과이며 일반적인 오차 상한은 아니다.

BBCSport M-L은 다음으로 강한 penalty에서 component별 prototype support가
붕괴할 때 path를 중단하였다. 모든 분할에서 BIC 선택 support는 실패한
endpoint가 아니라 path 내부의 후보였다. 민감도 분석에서는 최대 path
update를 600에서 1,200으로 늘리고 최소 상대 penalty 증가율을 0.02에서
0.005로 줄였다. 평균 path 행은 331.8에서 771.2로 증가했지만 selected
$q$ 차이는 최대 한 좌표였고 held-out ARI/NMI는 모든 분할에서 같았다.
마지막 평가 support의 BIC는 선택 support보다 최소 19,497.94 높았다.
조밀한 path는 평균 M-L 실행시간을 1.76배로 증가시켰으며 이후의
fit failure를 제거하지 못했다. 따라서 M-L의 near-dense 선택은 검토한
path resolution에서 안정적이지만, 평가된 path 이후에 대한 추론은
지원되지 않는다.

![Held-out ARI와 좌표 유지율](figures/realdata_locked_ari_retention_boxplots_260730.png)

## 8. CSTR 문헌 재현

Rossi and Barbaro의 CSTR 설정에서 구현 결과는 보고된 ARI를 재현하였다.

| 방법 | 논문 ARI | 구현 ARI |
|---|---:|---:|
| Dense shared-$\kappa$ vMF | 0.804 | 0.8023 (0.0087) |
| Rossi M-L, BIC | 0.808 | 0.8083 (0.0079) |

이전 centered-$\eta$ CSTR 진단은 refit 이전 BIC를 사용하였으므로 현재
고정 BIC-after 분석과 수치적으로 통합하지 않았다. CSTR은 Rossi 방식
구현의 재현성 확인에만 사용하였다.

## 9. 경험적 결론

Classic3는 주 해석 사례이다. E-CGL은 추정된 posterior-score contrast
support에서 전체 좌표의 67.1%를 유지하였고, 조건부 support Jaccard는
0.933이었으며, 적합 후 class alignment를 거친 성분 contrast는 각
문서 집합의 주제와 일관되었다. 대응되는 dense model과 비교하면
held-out ARI는 평균적으로 낮았고 NLL은 높았다. BBCSport에서는 모든
희소 방법이 모든 분할에서 held-out NLL 손실을 보였다. 따라서 실자료 결과는
학습자료에서 선택된 좌표의 축약, 분할 조건부 안정성, 사후적 해석을
보여주며, support recovery 또는 예측·군집 성능의 보편적 우월성을
의미하지 않는다.
