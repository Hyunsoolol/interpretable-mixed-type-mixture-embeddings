# S1-N~S4-N dense decision support rep50 요약

- Diagnostic only: rep=50, n=1000, d=200, K=4, nstart=10, path length=240, BIC selection.
- Dense decision support 설정: common q=4, decision q=80, noise q=116, true decision q=80.
- Rcpp helper guarded switch를 사용했지만, 알고리즘은 동일하고 low-level helper만 교체한 실행이다.
- 전체 method별 값은 `paper_eta_negative_control_s1n_s4n_rep50_summary.csv`에 있다.

## 핵심 비교: D-AGL vs E-AGL

| scenario | method | valid reps | ARI | selected q | decision q | noise q | TPR | FPR | F1 | MSE_eta | note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| S1-N | D-AGL | 50 | 0.857 | 85.480 | 79.960 | 1.520 | 1.000 | 0.046 | 0.966 | 0.326 | direction-group baseline |
| S1-N | E-AGL | 50 | 0.857 | 82.400 | 79.980 | 2.280 | 1.000 | 0.020 | 0.985 | 0.318 | true q=80 근처 |
| S2-N | D-AGL | 50 | 0.898 | 84.580 | 80.000 | 0.580 | 1.000 | 0.038 | 0.972 | 0.295 | direction-group baseline |
| S2-N | E-AGL | 50 | 0.897 | 81.820 | 80.000 | 1.780 | 1.000 | 0.015 | 0.989 | 0.292 | true q=80 근처 |
| S3-N | D-AGL | 50 | 0.568 | 85.840 | 72.620 | 9.220 | 0.908 | 0.110 | 0.877 | 2.113 | support F1 우위 |
| S3-N | E-AGL | 50 | 0.565 | 76.060 | 65.440 | 10.240 | 0.818 | 0.088 | 0.840 | 1.603 | dense decision support 과소선택 |
| S4-N | D-AGL | 50 | 0.000 | 4.020 | 0.020 | 0.000 | 0.000 | 0.033 | 0.024 | 3.945 | decision signal 거의 상실 |
| S4-N | E-AGL | 10 | 0.629 | 16.700 | 16.000 | 0.700 | 0.200 | 0.006 | 0.979 | 0.388 | 40회 zero-support, refit valid 10회 |

## 해석

- S1-N/S2-N에서는 E-AGL이 dense decision support q=80에서도 TPR이 거의 1이고 selected q가 true q에 가깝다.
- S3-N에서는 E-AGL이 decision variable을 과소선택한다. D-AGL의 F1은 0.877이고 E-AGL의 F1은 0.840이다.
- S4-N에서는 E-GL이 zero support를 선택했고, E-AGL도 valid refit이 10/50회에 그친다. 이는 weak mean + dense support에서 tuning failure가 생길 수 있음을 보여준다.
- 이 결과는 Eta-group이 항상 유리하다는 주장이 아니라, posterior decision support recovery의 장점과 한계를 함께 보여주는 negative-control diagnostic이다.