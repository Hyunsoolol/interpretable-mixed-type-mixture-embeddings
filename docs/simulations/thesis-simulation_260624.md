# Thesis Simulation Summary 260624

업데이트: 2026-06-17

이 문서는 2026-06-24 연구미팅 기준 시뮬레이션 결과를 정리한 clean summary다. 기존 상세 기록 `thesis-simulation_260615.md`는 archive 성격으로 보존한다.

## 1. Simulation Map

| Setting | Purpose | Source result folder | Meeting role |
|:---|:---|:---|:---|
| K=2 toy | eta contrast penalty가 가장 단순한 환경에서 작동하는지 확인 | `results/fair_path_grid_rep20_260622/` | idea check |
| K=4 strong common+specific | main claim 검증 | `results/k4_specific_effect_w050_rep100_260624/` | main evidence |
| Controlled concentration | 평균방향이 매우 유사하고 concentration 차이가 큰 경우 확인 | `results/k4_controlled_concdom_rep20_260622/` | supporting check |
| Signal strength sensitivity | component-specific loading $w$ 변화 확인 | `results/k4_specific_effect_w025/w035/w050_rep100_260624/` | sensitivity |
| Weak concentration | concentration 차이가 약할 때 robustness 확인 | `results/eta_meeting_weak_n1000_d100_rep100_260622/` | robustness |
| High-dimensional d=200 | moderate high-dimensional robustness 확인 | `results/eta_meeting_highdim_d200_n1000_rep50_260624/` | robustness/limitation |
| High-dimensional d=400 | stronger high-dimensional stress 확인 | `results/eta_meeting_highdim_d400_n1000_rep20_260624/` | limitation |
| Tuning sensitivity | EBIC/RIC-like criteria가 dense selection을 줄이는지 확인 | `results/highdim_tuning_sensitivity_260624/` | diagnostic |
| Long path diagnostic | path density/range가 high-dimensional 결과에 미치는 영향 확인 | `results/eta_highdim_d200_longpath240_rep50_260624/`, `results/eta_highdim_d400_longpath240_rep20_260624/` | diagnostic |

## 2. Common Metrics

| Metric | Meaning |
|:---|:---|
| ARI | clustering accuracy |
| Selected q | selected coordinate support size |
| TPR/FPR | support recovery true/false positive rates |
| Precision/F1 | support selection precision and F1 |
| MSE_mu | mean squared error for component directions |
| MSE_kappa | mean squared error for concentration parameters |
| MSE_centered_eta | mean squared error for centered eta contrasts |

All MSE values are reported on the raw scale.

Method names are standardized as follows.

| Document name | Source CSV name |
|:---|:---|
| Rossi BIC | Rossi path BIC |
| Rossi BIC + refit | Rossi path BIC + refit |
| Separate BIC | Separate 2D path/grid BIC |
| Separate BIC + refit | Separate 2D path/grid BIC + refit |
| Eta-group BIC | Eta centered path BIC |
| Eta-group BIC + refit | Eta centered path BIC + refit |

## 3. Main Results

### 3.1 K=2 Toy Setting

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

### 3.2 K=4 Strong Common+Specific

Setting: $K=4$, $n=1000$, $d=100$, rep=100, true union $q=22$, common loading 1.0, own-specific loading $w=0.5$, $\kappa=(30,45,65,90)$.

| Method | ARI | Selected q | TPR | FPR | Precision | F1 |
|:---|---:|---:|---:|---:|---:|---:|
| Rossi BIC | 0.680 | 98.520 | 1.000 | 0.981 | 0.223 | 0.365 |
| Rossi BIC + refit | 0.653 | 98.520 | 1.000 | 0.981 | 0.223 | 0.365 |
| Separate BIC | 0.684 | 86.460 | 1.000 | 0.826 | 0.258 | 0.409 |
| Separate BIC + refit | 0.657 | 86.460 | 1.000 | 0.826 | 0.258 | 0.409 |
| Eta-group BIC | 0.625 | 24.750 | 0.994 | 0.037 | 0.890 | 0.937 |
| Eta-group BIC + refit | 0.686 | 24.750 | 0.994 | 0.037 | 0.890 | 0.937 |

| Method | MSE_mu | MSE_kappa | MSE_centered_eta | kappa_hat_mean |
|:---|---:|---:|---:|---:|
| Rossi BIC | 0.000153 | 2.989 | 0.314 | 58.661 |
| Rossi BIC + refit | 0.000326 | 3.427 | 0.594 | 58.735 |
| Separate BIC | 0.000084 | 8.762 | 0.179 | 56.089 |
| Separate BIC + refit | 0.000302 | 3.064 | 0.552 | 58.599 |
| Eta-group BIC | 0.000291 | 14.485 | 0.424 | 58.468 |
| Eta-group BIC + refit | 0.000102 | 1.901 | 0.185 | 58.040 |

### 3.3 Signal Strength Sensitivity

| w | Method | ARI | Selected q | TPR | FPR | Precision | F1 |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 0.25 | Rossi BIC | 0.394 | 98.07 | 1.000 | 0.975 | 0.225 | 0.367 |
| 0.25 | Rossi BIC + refit | 0.368 | 98.07 | 1.000 | 0.975 | 0.225 | 0.367 |
| 0.25 | Separate BIC | 0.401 | 93.50 | 0.999 | 0.917 | 0.236 | 0.382 |
| 0.25 | Separate BIC + refit | 0.367 | 93.50 | 0.999 | 0.917 | 0.236 | 0.382 |
| 0.25 | Eta-group BIC | 0.358 | 32.13 | 0.813 | 0.183 | 0.621 | 0.682 |
| 0.25 | Eta-group BIC + refit | 0.399 | 32.13 | 0.813 | 0.183 | 0.621 | 0.682 |
| 0.35 | Rossi BIC | 0.514 | 98.82 | 1.000 | 0.985 | 0.223 | 0.364 |
| 0.35 | Rossi BIC + refit | 0.481 | 98.82 | 1.000 | 0.985 | 0.223 | 0.364 |
| 0.35 | Separate BIC | 0.528 | 90.70 | 1.000 | 0.881 | 0.247 | 0.394 |
| 0.35 | Separate BIC + refit | 0.488 | 90.70 | 1.000 | 0.881 | 0.247 | 0.394 |
| 0.35 | Eta-group BIC | 0.458 | 29.81 | 0.919 | 0.123 | 0.723 | 0.794 |
| 0.35 | Eta-group BIC + refit | 0.505 | 29.81 | 0.919 | 0.123 | 0.723 | 0.794 |
| 0.50 | Rossi BIC | 0.680 | 98.52 | 1.000 | 0.981 | 0.223 | 0.365 |
| 0.50 | Rossi BIC + refit | 0.653 | 98.52 | 1.000 | 0.981 | 0.223 | 0.365 |
| 0.50 | Separate BIC | 0.684 | 86.46 | 1.000 | 0.826 | 0.258 | 0.409 |
| 0.50 | Separate BIC + refit | 0.657 | 86.46 | 1.000 | 0.826 | 0.258 | 0.409 |
| 0.50 | Eta-group BIC | 0.625 | 24.75 | 0.994 | 0.037 | 0.890 | 0.937 |
| 0.50 | Eta-group BIC + refit | 0.686 | 24.75 | 0.994 | 0.037 | 0.890 | 0.937 |

| w | Method | MSE_mu | MSE_kappa | MSE_centered_eta | kappa_hat_mean |
|---:|:---|---:|---:|---:|---:|
| 0.25 | Rossi BIC | 0.000466 | 53.570 | 1.288 | 61.007 |
| 0.25 | Rossi BIC + refit | 0.000878 | 78.474 | 2.351 | 61.228 |
| 0.25 | Separate BIC | 0.000289 | 60.743 | 0.935 | 58.975 |
| 0.25 | Separate BIC + refit | 0.000731 | 66.807 | 1.967 | 60.640 |
| 0.25 | Eta-group BIC | 0.000543 | 4.999e9 | 3.352e7 | 5062.159 |
| 0.25 | Eta-group BIC + refit | 0.000682 | 4.999e9 | 3.352e7 | 5062.191 |
| 0.35 | Rossi BIC | 0.000285 | 16.756 | 0.656 | 59.694 |
| 0.35 | Rossi BIC + refit | 0.000556 | 20.778 | 1.167 | 59.713 |
| 0.35 | Separate BIC | 0.000149 | 12.341 | 0.318 | 57.198 |
| 0.35 | Separate BIC + refit | 0.000460 | 12.747 | 0.927 | 59.190 |
| 0.35 | Eta-group BIC | 0.000405 | 99.000 | 1.283 | 61.973 |
| 0.35 | Eta-group BIC + refit | 0.000359 | 43.882 | 0.969 | 60.408 |
| 0.50 | Rossi BIC | 0.000153 | 2.989 | 0.314 | 58.661 |
| 0.50 | Rossi BIC + refit | 0.000326 | 3.427 | 0.594 | 58.735 |
| 0.50 | Separate BIC | 0.000084 | 8.762 | 0.179 | 56.089 |
| 0.50 | Separate BIC + refit | 0.000302 | 3.064 | 0.552 | 58.599 |
| 0.50 | Eta-group BIC | 0.000291 | 14.485 | 0.424 | 58.468 |
| 0.50 | Eta-group BIC + refit | 0.000102 | 1.901 | 0.185 | 58.040 |

### 3.4 Weak Concentration

Setting: $K=4$, $n=1000$, $d=100$, rep=100, true union $q=22$, $w=0.5$, $\kappa=(40,50,60,70)$.

| Method | ARI | Selected q | TPR | FPR | Precision | F1 |
|:---|---:|---:|---:|---:|---:|---:|
| Rossi BIC | 0.542 | 99.95 | 1.000 | 0.999 | 0.220 | 0.361 |
| Rossi BIC + refit | 0.527 | 99.95 | 1.000 | 0.999 | 0.220 | 0.361 |
| Separate BIC | 0.543 | 99.67 | 1.000 | 0.996 | 0.221 | 0.362 |
| Separate BIC + refit | 0.526 | 99.67 | 1.000 | 0.996 | 0.221 | 0.362 |
| Eta-group BIC | 0.568 | 24.09 | 1.000 | 0.027 | 0.918 | 0.956 |
| Eta-group BIC + refit | 0.575 | 24.09 | 1.000 | 0.027 | 0.918 | 0.956 |

| Method | MSE_mu | MSE_kappa | MSE_centered_eta | kappa_hat_mean |
|:---|---:|---:|---:|---:|
| Rossi BIC | 0.000219 | 3.437 | 0.519 | 56.249 |
| Rossi BIC + refit | 0.000285 | 3.623 | 0.668 | 56.247 |
| Separate BIC | 0.000188 | 8.596 | 0.415 | 54.429 |
| Separate BIC + refit | 0.000286 | 3.536 | 0.667 | 56.209 |
| Eta-group BIC | 0.000172 | 7.409 | 0.355 | 54.506 |
| Eta-group BIC + refit | 0.000075 | 1.824 | 0.183 | 55.497 |

### 3.5 High-dimensional d=200

| Method | ARI | Selected q | TPR | FPR | Precision | F1 |
|:---|---:|---:|---:|---:|---:|---:|
| Rossi BIC | 0.393 | 200.00 | 1.000 | 1.000 | 0.110 | 0.198 |
| Rossi BIC + refit | 0.389 | 200.00 | 1.000 | 1.000 | 0.110 | 0.198 |
| Separate BIC | 0.398 | 200.00 | 1.000 | 1.000 | 0.110 | 0.198 |
| Separate BIC + refit | 0.388 | 200.00 | 1.000 | 1.000 | 0.110 | 0.198 |
| Eta-group BIC | 0.459 | 120.06 | 0.989 | 0.552 | 0.208 | 0.331 |
| Eta-group BIC + refit | 0.430 | 120.06 | 0.989 | 0.552 | 0.208 | 0.331 |

| Method | MSE_mu | MSE_kappa | MSE_centered_eta | kappa_hat_mean |
|:---|---:|---:|---:|---:|
| Rossi BIC | 0.000964 | 123.189 | 2.407 | 63.286 |
| Rossi BIC + refit | 0.001040 | 123.889 | 2.572 | 63.223 |
| Separate BIC | 0.000898 | 73.406 | 1.823 | 58.812 |
| Separate BIC + refit | 0.001033 | 104.516 | 2.420 | 62.509 |
| Eta-group BIC | 0.000434 | 62.041 | 0.905 | 60.350 |
| Eta-group BIC + refit | 0.000691 | 93.797 | 1.818 | 62.299 |

### 3.6 High-dimensional d=400

| Method | ARI | Selected q | TPR | FPR | Precision | F1 |
|:---|---:|---:|---:|---:|---:|---:|
| Rossi BIC | 0.142 | 400.00 | 1.000 | 1.000 | 0.055 | 0.104 |
| Rossi BIC + refit | 0.141 | 400.00 | 1.000 | 1.000 | 0.055 | 0.104 |
| Separate BIC | 0.146 | 399.95 | 1.000 | 1.000 | 0.055 | 0.104 |
| Separate BIC + refit | 0.144 | 399.95 | 1.000 | 1.000 | 0.055 | 0.104 |
| Eta-group BIC | 0.206 | 262.95 | 0.920 | 0.642 | 0.080 | 0.146 |
| Eta-group BIC + refit | 0.182 | 262.95 | 0.920 | 0.642 | 0.080 | 0.146 |

| Method | MSE_mu | MSE_kappa | MSE_centered_eta | kappa_hat_mean |
|:---|---:|---:|---:|---:|
| Rossi BIC | 0.001252 | 411.108 | 4.692 | 71.872 |
| Rossi BIC + refit | 0.001300 | 412.507 | 4.854 | 71.886 |
| Separate BIC | 0.001246 | 306.790 | 4.196 | 67.977 |
| Separate BIC + refit | 0.001330 | 397.506 | 4.860 | 71.287 |
| Eta-group BIC | 0.000701 | 216.991 | 1.942 | 63.839 |
| Eta-group BIC + refit | 0.001107 | 343.629 | 4.016 | 70.811 |

### 3.7 Tuning/path Diagnostics

EBIC/RIC-like/log(d)-slope 재선택은 BIC 결과를 거의 바꾸지 못했다.

| setting | criterion | Selected q | Zero rate | Dense rate | ARI | FPR | Precision | F1 |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| d=200 | BIC_current | 120.06 | 0.00 | 0.20 | 0.459 | 0.552 | 0.208 | 0.331 |
| d=200 | EBIC_gamma_0.5 | 119.98 | 0.00 | 0.20 | 0.459 | 0.552 | 0.209 | 0.331 |
| d=200 | EBIC_gamma_1.0 | 119.98 | 0.00 | 0.20 | 0.459 | 0.552 | 0.209 | 0.331 |
| d=200 | RIC_like | 119.98 | 0.00 | 0.20 | 0.459 | 0.552 | 0.209 | 0.331 |
| d=400 | BIC_current | 262.95 | 0.00 | 0.30 | 0.206 | 0.642 | 0.080 | 0.146 |
| d=400 | EBIC_gamma_0.5 | 262.95 | 0.00 | 0.30 | 0.206 | 0.642 | 0.080 | 0.146 |
| d=400 | EBIC_gamma_1.0 | 262.95 | 0.00 | 0.30 | 0.206 | 0.642 | 0.080 | 0.146 |
| d=400 | RIC_like | 262.90 | 0.00 | 0.30 | 0.206 | 0.642 | 0.080 | 0.146 |

Long path는 official algorithm 변경이 아니라 path density/range가 성능에 미치는 영향을 확인하기 위한 diagnostic이다.

| setting | path | reps | near22 rate | q<=50 rate | Selected q | Dense rate | ARI | TPR | FPR | Precision | F1 |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| d=200 | basic path | 50 | 0.02 | 0.02 | 120.06 | 0.20 | 0.459 | 0.989 | 0.552 | 0.208 | 0.331 |
| d=200 | long path 240 | 50 | 0.34 | 0.46 | 62.14 | 0.04 | 0.447 | 0.920 | 0.235 | 0.507 | 0.584 |
| d=400 | basic path | 20 | 0.00 | 0.00 | 262.95 | 0.30 | 0.206 | 0.920 | 0.642 | 0.080 | 0.146 |
| d=400 | long path 240 | 20 | 0.40 | 0.55 | 68.75 | 0.00 | 0.214 | 0.620 | 0.146 | 0.491 | 0.441 |

### 3.8 Adaptive penalty diagnostic

Adaptive Eta-group penalty는 현재 official algorithm이 아니라 penalty weighting sensitivity다. Gamma=1.0에서는 initial centered eta norm으로 coordinate weight를 만들고, median weight가 1이 되도록 normalize했다.

| setting | Method | reps | ARI | Selected q | FPR | Precision | F1 | MSE_centered_eta | 판단 |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| strong official | Eta-group BIC + refit | 100 | 0.686 | 24.75 | 0.037 | 0.890 | 0.937 | 0.185 | official baseline |
| strong adaptive diagnostic, gamma=1.0 | Eta-group adaptive BIC + refit | 100 | 0.690 | 22.47 | 0.008 | 0.977 | 0.984 | 0.147 | clear improvement |
| strong adaptive diagnostic, gamma=0.5 | Eta-group adaptive BIC + refit | 100 | 0.689 | 23.62 | 0.022 | 0.946 | 0.967 | 0.160 | improvement, weaker than gamma=1.0 |
| weak official | Eta-group BIC + refit | 100 | 0.575 | 24.09 | 0.027 | 0.918 | 0.956 | 0.183 | official weak baseline |
| weak adaptive diagnostic, gamma=1.0 | Eta-group adaptive BIC + refit | 100 | 0.578 | 22.58 | 0.007 | 0.975 | 0.987 | 0.153 | clear improvement |
| d=200 basic path | Eta-group BIC + refit | 50 | 0.430 | 120.06 | 0.552 | 0.208 | 0.331 | 1.818 | dense support problem |
| d=200 long path 240 | Eta-group BIC + refit | 50 | 0.455 | 62.14 | 0.235 | 0.507 | 0.584 | 1.376 | path improves but selected q remains high |
| d=200 adaptive diagnostic, gamma=1.0 | Eta-group adaptive BIC + refit | 50 | 0.415 | 116.72 | 0.534 | 0.205 | 0.334 | 1.925 | adaptive alone fails |
| d=200 path+adaptive diagnostic, gamma=1.0 | Eta-group adaptive BIC + refit | 50 | 0.461 | 40.98 | 0.127 | 0.741 | 0.715 | 1.148 | best d=200 diagnostic |
| d=400 basic path | Eta-group BIC + refit | 20 | 0.182 | 262.95 | 0.642 | 0.080 | 0.146 | 4.016 | dense stress limitation |
| d=400 long path 240 | Eta-group BIC + refit | 20 | 0.238 | 68.75 | 0.146 | 0.491 | 0.441 | 2.248 | best d=400 among tested |
| d=400 path+adaptive diagnostic, gamma=1.0 | Eta-group adaptive BIC + refit | 20 | 0.155 | 308.00 | 0.760 | 0.068 | 0.127 | 4.518 | fails; dense support returns |

Adaptive penalty diagnostic은 strong과 weak d=100 setting에서 selected q와 FPR을 줄이고 Precision/F1을 개선했다. 그러나 d=200에서는 adaptive penalty alone이 dense support 문제를 해결하지 못했고, long path 240과 결합해야 의미 있는 개선이 나타났다. 더 중요하게는 d=400 stress에서 path+adaptive diagnostic이 selected q=308.00, FPR=0.760으로 악화되어 dense support 문제를 다시 만들었다. 따라서 adaptive penalty는 not official이며, current official은 Eta-group path+BIC + refit으로 유지하는 것이 안전하다. Adaptive penalty는 diagnostic only 또는 appendix-level candidate로 두고, d=400 stress limitation은 path construction, screening, update/MM 개선 문제로 정리한다.

## 4. Main Takeaways

1. Eta-group은 strong common+specific setting에서 support recovery가 가장 설득력 있다.
2. Weak setting은 결과가 양호하지만 robustness evidence로 두는 편이 안전하다.
3. High-dimensional setting에서는 basic path가 dense support로 가기 쉽다.
4. EBIC/RIC-like criteria는 basic path 후보가 부족하면 선택을 거의 바꾸지 못한다.
5. Long path는 FPR, Precision, F1을 개선하지만 true q=22 회복은 제한적이다.
6. Adaptive penalty는 strong/weak d=100 setting에서는 promising diagnostic candidate이며, d=200에서는 long path와 결합할 때 의미가 있지만 d=400 stress에서는 악화된다.
7. 다음 보강 후보는 path construction, MM/coordinate update, screening, adaptive penalty weighting이다.
