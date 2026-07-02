# S1-N~S4-N dense decision support pilot 요약

- Diagnostic only: pilot rep=5, n=1000, d=200, K=4, nstart=10, path length=240, BIC selection.
- Dense decision support 설정: common q=4, decision q=80, noise q=116, true decision q=80.
- Rcpp helper guarded switch를 사용했지만, 알고리즘은 동일하고 low-level helper만 교체한 실행이다.
- 전체 method별 값은 `paper_eta_negative_control_s1n_s4n_pilot5_summary.csv`에 있다.

## 핵심 비교: D-AGL vs E-AGL

| scenario | method | ARI | selected q | decision q | noise q | TPR | FPR | F1 | MSE_eta | note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| S1-N | D-AGL | 0.863 | 85.8 | 80.0 | 1.8 | 1.000 | 0.048 | 0.965 | 0.331 | direction-group baseline |
| S1-N | E-AGL | 0.866 | 82.2 | 80.0 | 1.8 | 1.000 | 0.018 | 0.987 | 0.312 | true q=80 근처 |
| S2-N | D-AGL | 0.893 | 85.2 | 80.0 | 1.2 | 1.000 | 0.043 | 0.969 | 0.306 | direction-group baseline |
| S2-N | E-AGL | 0.895 | 81.8 | 80.0 | 1.8 | 1.000 | 0.015 | 0.989 | 0.290 | true q=80 근처 |
| S3-N | D-AGL | 0.583 | 91.6 | 75.0 | 12.6 | 0.938 | 0.138 | 0.875 | 1.831 | support F1 우위 |
| S3-N | E-AGL | 0.584 | 77.0 | 66.6 | 10.2 | 0.833 | 0.087 | 0.848 | 1.215 | dense decision support 과소선택 |
| S4-N | D-AGL | 0.001 | 4.2 | 0.2 | 0.0 | 0.003 | 0.033 | 0.024 | 4.078 | decision signal 거의 상실 |
| S4-N | E-AGL | NA | 0.0 | 0.0 | 0.0 | 0.000 | 0.000 | NA | NA | BIC zero support |

## 해석

- S1-N/S2-N에서는 E-AGL이 dense decision support q=80에서도 안정적이었다.
- S3-N에서는 E-AGL이 decision variable을 과소선택했고, D-AGL이 support F1에서는 더 좋았다.
- S4-N에서는 E-GL/E-AGL이 zero support를 선택해 refit이 유효하지 않았다.
- pilot 결과 기준으로 rep=50 확장이 필요하다. 논문 수준 결론은 rep=50 이후에 판단한다.