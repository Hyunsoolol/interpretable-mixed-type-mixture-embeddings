# Code Organization and Rcpp Plan 260708

이 문서는 연구미팅 전 코드 구조를 정리하기 위한 감사 결과다. 이번 단계에서는 새 simulation을 실행하지 않았고, R core algorithm도 수정하지 않았다. 목적은 주요 모형 코드, simulation runner, real-data pipeline을 분리해 볼 수 있는 정리 방향과 Rcpp 변환 후보를 식별하는 것이다.

## 1. 확인한 코드 구조

### 1.1 현재 디렉토리 역할

| 위치 | 현재 역할 | 관찰 |
|:---|:---|:---|
| `r/methods` | Rossi reproduction, Eta-group, Separate, tuning, real-data fit runner가 섞여 있음 | core fitting 함수와 실험 runner가 같은 파일 안에 들어 있는 경우가 많음 |
| `r/simulation` | K=4 main simulation, ablation, negative-control diagnostic, search runner | generator, method wrapper, summary writer가 한 파일에 함께 들어 있는 경우가 많음 |
| `r/realdata` | text/gene real-data screening, profiling, baseline 비교 | selected-term profiling과 fit selection helper가 중복됨 |
| `r/data_prep` | BBC/20NG/Reuters/PBMC/TCGA 등 payload 생성 | env parsing과 tokenize/TF-IDF helper가 반복됨 |
| `scripts/realdata` | sparse embedding payload 변환/검사 script | real-data schema helper로 분리 가능 |

요청 목록 중 `r/simulation/negative_control_setting_a_redesign_260708.r`는 현재 작업 트리에서 확인되지 않았다.

### 1.2 핵심 파일 역할

| 파일 | 주요 역할 | 분류 |
|:---|:---|:---|
| `r/methods/rossi_barbaro_2022_reproduction.r` | vMF utility, Rossi sparse vMF EM, initialization, E-step, path fit | core model + reproduction runner |
| `r/methods/rb2022_k4_pilot_compare_run.r` | Eta-group, Separate, Rossi 비교의 중심 helper와 K=4 runner | core model + simulation runner |
| `r/methods/cstr_eta_compare_run.r` | real-data Eta/Rossi 비교 runner, Eta-group fit helper 복제 | core model + real-data runner |
| `r/simulation/k4_specific_effect_run.r` | K=4 common/specific generator, official simulation runner | simulation generator + runner |
| `r/simulation/eta_group_vs_anova_l1_current_sim_260616.r` | Eta-group vs Eta entry-wise L1 diagnostic | ablation runner |
| `r/simulation/rossi_group_diagnostic_smoke_260708.r` | natural-scale group diagnostic runner | ablation runner |
| `r/simulation/rossi_mu_group_diagnostic_rep20_260708.r` | Rossi mu-group diagnostic runner | ablation runner |
| `r/realdata/text_fit_profile_words.r` | selected token/term profiling | real-data profiling |
| `r/realdata/realdata_screening_run.r` | real-data result screening/scoring | real-data post-processing |
| `r/data_prep/text_prepare_utils.r` | tokenization and TF-IDF payload helper | data prep utility |
| `scripts/realdata/convert_sparse_embedding_to_payload_260624.r` | sparse embedding CSV to Eta payload RDS conversion | real-data preprocessing utility |

## 2. 분리 상태 진단

### 2.1 모형 코드와 runner가 섞인 부분

- `rb2022_k4_pilot_compare_run.r` 안에 `fit_eta_centered_em`, `fit_separate_penalty_em`, `fit_support_refit`, support metric, K=4 `run_one`이 함께 있다.
- `cstr_eta_compare_run.r`에도 Eta fitting helper가 복제되어 real-data runner와 섞여 있다.
- `k4_specific_effect_run.r`는 generator와 method-specific runner, adaptive diagnostic naming, summary row 생성이 함께 들어 있다.
- ablation scripts는 diagnostic 목적상 독립 runner로 이해 가능하지만, `make_specific_effect_params`, `simulate_from_params`, `safe_mean`, `fmt` 같은 helper가 반복된다.

### 2.2 중복 또는 정리 후보 함수

| 함수/패턴 | 중복 정도 | 정리 방향 |
|:---|:---:|:---|
| `safe_mean`, `safe_se`, `fmt` | 매우 높음 | `r/utils/summary_helpers.r` 후보 |
| `getenv_num`, `getenv_int`, `parse_num_grid` | 높음 | `r/utils/env_config.r` 후보 |
| `eta_matrix`, `center_eta`, `active_eta_centered`, `active_mu_coord` | 높음 | `r/core/eta_utils.r` 후보 |
| `support_metrics`, `support_df`, `support_ic` | 높음 | `r/core/model_selection.r` 또는 `r/utils/metrics.r` 후보 |
| `mask_and_normalize_mu`, `fit_support_refit` | 여러 파일 | `r/core/refit.r` 후보 |
| `unpenalized_eta_mstep`, `eta_to_theta`, `prox_eta_centered`, `fit_eta_centered_em` | 여러 파일/variant | `r/core/eta_group_fit.r` 후보 |
| `make_specific_effect_params`, `simulate_from_params` | simulation/ablation 반복 | `r/simulation/generators/k4_specific_effect.r` 후보 |

현재는 결과 재현성을 위해 함수 이동/rename을 바로 하지 않는 것이 안전하다. 먼저 공통 helper를 새 파일에 복사하고, 기존 runner가 optional source로 이를 쓰도록 바꾸는 단계적 접근이 필요하다.

## 3. 권장 디렉토리 구조

실제 이동은 미팅 후 별도 branch에서 진행하는 것이 안전하다.

```text
r/
  core/
    vmf_utils.r
    eta_utils.r
    eta_group_fit.r
    rossi_fit.r
    separate_fit.r
    refit.r
    model_selection.r
  simulation/
    generators/
      k4_specific_effect.r
      negative_control_generators.r
    runners/
      run_k4_specific_effect.r
      run_negative_control.r
      run_ablation_diagnostics.r
    summaries/
      simulation_summary_helpers.r
  realdata/
    payload_schema.r
    fit_text_pipeline.r
    profile_selected_terms.r
    screen_results.r
  utils/
    env_config.r
    metrics.r
    io_helpers.r
    summary_helpers.r
```

권장 순서는 `utils`와 `core` helper를 먼저 만들고, runner 이동은 나중에 한다. 기존 파일명을 바로 바꾸면 과거 결과 재현 script가 깨질 수 있다.

## 4. Rcpp 변환 후보

Rcpp는 설치되어 있고 Rtools `make`도 확인되었다. 하지만 이번 단계에서는 변환을 적용하지 않았다.

| 후보 | 위치 | 역할 | 병목 가능성 | 난이도 | 기대 개선 | numerical risk | 테스트 필요성 | 지금 변환 추천 |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---|:---:|
| `row_logsumexp`, `e_step_vmf` | `r/methods/rossi_barbaro_2022_reproduction.r` | tau/loglik 계산 | 높음 | 중간 | 높음 | 중간 | R vs Rcpp tau/loglik tolerance test | 2순위 |
| `log_vmf_const_one`, `log_vmf_const` | `r/methods/rossi_barbaro_2022_reproduction.r` | vMF normalizing constant | 중간 | 중간 | 중간 | 높음, Bessel edge case | kappa/d grid test | 2순위 |
| `prox_eta_centered` | `r/methods/rb2022_k4_pilot_compare_run.r`, `cstr_eta_compare_run.r` | centered eta group shrinkage | 중간 | 낮음 | 중간 | 낮음 | exact equality/tolerance test | 1순위 |
| `eta_centered_penalty_value` | `rb2022_k4_pilot_compare_run.r` | objective penalty helper | 중간 | 낮음 | 낮음-중간 | 낮음 | penalty value equality test | 1순위 |
| `mask_and_normalize_mu` / `normalize_rows_masked` | 여러 파일 | support refit row normalization | 중간 | 낮음 | 중간 | 낮음 | zero-row/fallback test | 1순위 |
| `fit_support_refit` inner loop | `rb2022_k4_pilot_compare_run.r`, `cstr_eta_compare_run.r` | support fixed refit | 높음 | 중간-높음 | 높음 | 중간 | loglik monotonicity + selected support test | 2순위 |
| `update_mu_kappa_one` | `rossi_barbaro_2022_reproduction.r` | Rossi mu/kappa inner update | 높음 | 중간 | 높음 | 중간 | convergence and failure parity test | 2순위 |
| path candidate loop | `fit_eta_pair`, `fit_separate_pair`, ablation runners | lambda path evaluation | 높음 | 높음 | 중간 | 중간 | same selected lambda/IC test | 3순위 |
| entire optimizer loop | `fit_eta_centered_em`, `fit_svMF_em` | full EM/proximal EM | 높음 | 높음 | 높음 | 높음 | extensive regression tests | 보류 |

## 5. Rcpp 환경 확인

확인 명령은 설치를 수행하지 않고 availability만 확인했다.

```text
R = R version 4.2.1 (2022-06-23 ucrt)
Rcpp = TRUE
inline = TRUE
Rtools make = C:\RBUILD~1\4.2\usr\bin\make.exe
R CMD config CXX = g++ -std=gnu++11
```

Windows에서 Rcpp 컴파일은 가능한 상태로 보인다. 단, OneDrive 경로와 한글 경로는 컴파일/임시 파일 문제를 일으킬 수 있으므로, Rcpp 테스트는 짧은 standalone file로 먼저 확인하는 것이 안전하다.

## 6. 추천 리팩토링 순서

### 6.1 미팅 전 가능한 안전한 정리

1. 코드맵 문서 유지 및 보완.
2. 공통 함수 후보 목록 확정.
3. 기존 runner는 그대로 두고, 이동/rename은 하지 않음.
4. parse check만 수행하고 source 실행은 피함.

### 6.2 미팅 후 1단계

1. `r/utils/env_config.r`, `r/utils/summary_helpers.r` 생성.
2. `getenv_*`, `parse_*_grid`, `safe_mean`, `safe_se`, `fmt`부터 공통화.
3. 결과가 바뀌지 않는 parse/unit test만 먼저 수행.

### 6.3 미팅 후 2단계

1. `r/core/eta_utils.r`, `r/core/model_selection.r`, `r/core/refit.r` 생성.
2. `eta_matrix`, `center_eta`, `active_eta_centered`, `support_metrics`, `support_ic`, `fit_support_refit` 공통화.
3. 기존 main summary를 하나 골라 smoke-free regression check.

### 6.4 미팅 후 3단계

1. `prox_eta_centered`, `eta_centered_penalty_value`, row normalization helper부터 Rcpp prototype 작성.
2. R output과 Rcpp output equality test 작성.
3. 통과하면 E-step/loglik 후보로 확장.

### 6.5 보류할 작업

- `fit_eta_centered_em` 전체 Rcpp화.
- lambda path 전체 Rcpp화.
- 기존 file 이동/rename.
- official simulation runner의 function signature 변경.

## 7. 검증 결과

- 주요 확인 대상 R 파일은 `parse()` 기준 문법 오류 없이 통과했다.
- 새 simulation은 실행하지 않았다.
- R core algorithm은 수정하지 않았다.
- Rcpp 변환은 적용하지 않았다.

Parse check 대상:

```text
r/methods/rb2022_k4_pilot_compare_run.r
r/methods/cstr_eta_compare_run.r
r/simulation/k4_specific_effect_run.r
r/simulation/rossi_group_diagnostic_smoke_260708.r
r/simulation/eta_group_vs_anova_l1_current_sim_260616.r
r/realdata/text_fit_profile_words.r
r/realdata/realdata_screening_run.r
r/data_prep/text_prepare_utils.r
scripts/realdata/convert_sparse_embedding_to_payload_260624.r
```

## 8. 결론

현재 코드는 연구 과정에서 빠르게 확장된 runner 중심 구조다. 핵심 fitting 함수는 존재하지만, simulation runner와 real-data runner 안에 helper가 복제되어 있다. 미팅 전에는 문서화와 코드맵 정리까지만 하고, 미팅 후 별도 branch에서 `utils -> core helpers -> runner cleanup -> Rcpp prototype` 순서로 진행하는 것이 가장 안전하다.

## 9. Rcpp conversion target details

Rcpp 변환의 목적은 알고리즘을 바꾸는 것이 아니라, 동일한 numerical output을 더 빠르게 계산하는 것이다. 따라서 기존 R wrapper와 함수 signature는 유지하고, 내부의 pure numeric helper만 C++ 구현으로 교체하는 순서가 안전하다.

### 9.1 변환 원칙

- 기존 R 함수 wrapper는 유지하고 내부 계산 helper만 Rcpp로 대체한다.
- Rcpp 함수는 matrix/vector 입력과 numeric output이 명확한 pure numeric helper로 제한한다.
- random seed, initialization, lambda path construction, BIC/EBIC/RICc decision logic은 R에 남긴다.
- 첫 단계에서는 output equality 또는 tolerance test를 통과해야 한다.
- simulation 결과를 바꾸는 목적이 아니라 runtime을 줄이는 목적이다.
- Windows/Rtools/OneDrive 경로 이슈를 피하기 위해 standalone Rcpp prototype과 test harness부터 시작한다.

### 9.2 상세 후보 표

| Priority | R function | File(s) | 기능 | Rcpp로 바꿀 단위 | 기대 효과 | 위험도 | 동일성 테스트 | 지금 권장 여부 |
|:---:|:---|:---|:---|:---|:---|:---|:---|:---:|
| 1 | `prox_eta_centered` | `r/methods/rb2022_k4_pilot_compare_run.r`; `r/methods/cstr_eta_compare_run.r` | centered eta contrast에 coordinate-wise group soft-thresholding 적용 | centered matrix, column norm, scale 계산 | 반복 path/iteration에서 중간 정도 속도 개선 기대 | 낮음 | random eta/lambda grid에서 max abs diff `< 1e-10` | 예 |
| 1 | `eta_centered_penalty_value` | `r/methods/rb2022_k4_pilot_compare_run.r` | centered eta group penalty 값 계산 | centered column norm 합산 | objective check 반복부를 가볍게 함 | 낮음 | R penalty value와 Rcpp value 비교 | 예 |
| 1 | `mask_and_normalize_mu` | `r/methods/rb2022_k4_pilot_compare_run.r`; `r/methods/cstr_eta_compare_run.r`; related copies | selected support만 남기고 row-wise L2 normalize | masking, row norm, fallback 처리 | refit 반복부와 profiling helper 안정화 | 낮음-중간 | zero-row, all-FALSE support, fallback case 비교 | 예 |
| 1 | `normalize_rows` | `r/methods/rossi_barbaro_2022_reproduction.r`; `scripts/realdata/convert_sparse_embedding_to_payload_260624.r` | row-wise L2 normalization | row norm 계산과 division | 단순하고 테스트 쉬움 | 낮음 | zero/near-zero/normal rows row norm 비교 | 예 |
| 1 | `soft_threshold` | `r/simulation/eta_group_vs_anova_l1_current_sim_260616.r`; `r/methods/eta_centered_group_vs_anova_l1_smoke_260616.r`; other Eta L1 diagnostics | scalar/vector soft threshold | sign/pmax vector operation | Eta entry-wise L1 diagnostic에서 단순 가속 | 낮음 | scalar/vector edge case 비교 | 예, 단 diagnostic helper |
| 2 | `row_logsumexp` | `r/methods/rossi_barbaro_2022_reproduction.r` | log-sum-exp 안정화 | row max, exp sum, log 계산 | E-step 전반에 영향이 커 속도 개선 가능 | 중간 | extreme negative/positive matrix tolerance test | 조건부 |
| 2 | `e_step_vmf` | `r/methods/rossi_barbaro_2022_reproduction.r` | tau와 loglik 계산 | log density, row logsumexp, tau 계산 | 가장 반복되는 EM 병목 후보 | 중간 | tau row sum, loglik, no NA/Inf 비교 | 조건부 |
| 2 | `log_vmf_const_one`, `log_vmf_const` | `r/methods/rossi_barbaro_2022_reproduction.r` | vMF normalizing constant 계산 | kappa vector별 Bessel/log constant 계산 | E-step 반복부 일부 개선 | 높음 | kappa/d grid에서 R output과 tolerance 비교 | 신중 |
| 2 | `unpenalized_eta_mstep` | `r/methods/rb2022_k4_pilot_compare_run.r`; `r/methods/cstr_eta_compare_run.r` | tau 기준 natural parameter target 계산 | `t(tau) %*% X`, row norm, kappa loop | Eta-group iteration 반복부 개선 | 중간 | alpha/eta/kappa target 비교 | 조건부 |
| 2 | `update_mu_kappa_one` | `r/methods/rossi_barbaro_2022_reproduction.r` | Rossi `mu/kappa` inner update | inner shrinkage loop와 kappa update | Rossi/Separate 반복부 개선 | 중간 | convergence flag, mu/kappa, failure parity 비교 | 조건부 |
| 3 | `fit_support_refit` | `r/methods/rb2022_k4_pilot_compare_run.r`; `r/methods/cstr_eta_compare_run.r`; real-data copies | support-fixed refit loop | 전체 refit loop 내부 | 효과는 크지만 상태 관리가 복잡 | 중간-높음 | loglik monotonicity, selected support, final theta 비교 | 보류 |
| 3 | `fit_eta_centered_em` | `r/methods/rb2022_k4_pilot_compare_run.r`; `r/methods/cstr_eta_compare_run.r` | full proximal EM-type update | 전체 optimizer loop | 잠재 효과 큼 | 높음 | objective trace와 final fit regression test | 보류 |
| 3 | `fit_svMF_em` | `r/methods/rossi_barbaro_2022_reproduction.r` | Rossi sparse vMF EM | 전체 EM loop | 잠재 효과 큼 | 높음 | path selected model과 metrics 비교 | 보류 |
| 3 | lambda path loops | `k4_specific_effect_run.r`; `fit_eta_pair`; `fit_separate_pair`; diagnostic runners | path candidate 반복 평가 | loop orchestration | 병렬/캐싱이 더 적절할 수 있음 | 중간-높음 | selected lambda/IC 동일성 | 보류 |
| 3 | full simulation runner | `r/simulation/*.r` | 반복 실험 orchestration | runner 전체 | C++ 이점보다 재현성 위험이 큼 | 높음 | full regression required | 아니오 |

## 10. Rcpp testing plan

Rcpp 변환은 함수별 equality test를 먼저 통과한 뒤 fit-level regression으로 넘어간다. tolerance는 double precision 연산 순서 차이를 고려해 helper별로 정한다.

| 후보 | 최소 테스트 | 통과 기준 |
|:---|:---|:---|
| `prox_eta_centered` | random eta matrix 10개, `lambda = 0`, small, medium, max, adaptive weights 포함/미포함 | R vs Rcpp max abs diff `< 1e-10`; zeroed coordinate set 동일 |
| `eta_centered_penalty_value` | random eta, all-zero eta, one active coordinate, adaptive weights | penalty value diff `< 1e-10` |
| `normalize_rows` | normal rows, zero row, near-zero row, mixed magnitude rows | row norm and zero-row handling identical |
| `mask_and_normalize_mu` | active all TRUE/FALSE, sparse active, fallback available/unavailable | row norm, fallback coordinate, selected support identical |
| `soft_threshold` | scalar/vector input, lambda 0, small, large, signs mixed | exact or near-exact vector equality |
| `row_logsumexp` | normal matrix, very negative matrix, very positive matrix, mixed rows | finite output and tolerance `< 1e-10` where feasible |
| `e_step_vmf` | fixed small X/theta, K=2/4, d=10/100 | tau row sum = 1, loglik tolerance, no NA/Inf |
| `log_vmf_const` | kappa small/medium/large, d=10/100/400 | tolerance against R implementation; edge warnings documented |
| `unpenalized_eta_mstep` | fixed X/tau, balanced and near-empty components | alpha, eta target, kappa tolerance; empty component behavior identical |
| `update_mu_kappa_one` | normal `r_k`, near-zero `r_k`, high beta, low beta | `failed` flag, mu norm, kappa tolerance, iteration behavior compatible |
| `fit_support_refit` | one controlled small fit with fixed init/support | selected support fixed, loglik trajectory compatible, final theta close |

Fit-level tests should not rely only on ARI because two fits can have similar clustering with different parameter estimates. For Eta-group, compare at least loglik, selected q, active coordinate set, `MSE_centered_eta`, and objective trace when available.

## 11. Suggested Rcpp rollout order

### Step 0: skeleton only

- Create a standalone Rcpp prototype file and a small test harness.
- Do not source it from official runners.
- Confirm Windows/Rtools compilation in a short path if OneDrive or Korean path causes issues.

### Step 1: safe low-level helpers

- Implement `prox_eta_centered`, `eta_centered_penalty_value`, `normalize_rows`.
- Run unit tests only.
- Keep official R functions as wrappers or fallback.

### Step 2: E-step helpers

- Implement `row_logsumexp` first.
- Then wrap `e_step_vmf` only after log-sum-exp tests pass.
- Compare one fixed E-step and one fixed EM iteration, not full simulation.

### Step 3: M-step helpers

- Implement `unpenalized_eta_mstep` and possibly `update_mu_kappa_one`.
- Run smoke-fit regression on one deterministic small dataset.
- Check loglik, selected support, selected q, and parameter MSE.

### Step 4: defer high-level loops

- Keep lambda path loops, `fit_eta_centered_em`, `fit_svMF_em`, and full simulation runners in R until helper-level equivalence is well established.
- If runtime is still limiting after helper conversion, consider path-level parallelism before full optimizer Rcpp conversion.

### Step 5: integration rule

- Rcpp integration should happen on a separate branch.
- A switch such as `USE_RCPP_HELPERS=1` can be added later, but not before helper tests exist.
- The default path should remain the current R implementation until regression checks pass.

## 12. Rcpp helper validation status

As of 260708, the first Eta low-level Rcpp prototype is diagnostic-only and is not connected to the official fitting pipeline.

Validation completed:

- Standalone helper equality: PASS for `prox_eta_centered_cpp`, `eta_centered_penalty_value_cpp`, and `normalize_rows_cpp`.
- Fit-level smoke equality: PASS for a small synthetic K=4 Eta fit using R-only helpers versus Rcpp helpers.
- Extended grid equality: PASS for 24 deterministic comparisons over seeds, dimensions, and lambda scales.
- Maximum observed fit-level differences in the extended grid were numerical-roundoff scale: loglik `9.09e-13`, objective `9.09e-13`, eta `1.78e-14`, tau `3.00e-15`, objective trace `1.36e-12`.

Before official integration:

- Add a one-iteration equality test against the exact official update loop.
- Add path/BIC equality tests with R-only and Rcpp-helper modes.
- Add a rep=1 end-to-end equality test with identical seed, initialization, and lambda path.
- Add a runtime benchmark on a representative Eta-group path.
- Add a fallback switch such as `USE_RCPP_HELPERS=0/1`.
- Perform the official connection in a separate reviewed commit or branch.

## 13. Rcpp pre-integration diagnostic status

Additional diagnostic-only checks were completed before any official wiring:

- One-iteration equality test: PASS for 4 lambda scales.
- Small path/BIC equality test: PASS for 4 lambda scales.
- Best BIC index matched between R-only and Rcpp-helper modes.
- Maximum observed one-iteration differences were numerical-roundoff scale: loglik `4.55e-13`, objective `4.55e-13`, eta `8.88e-15`, tau `1.72e-15`.
- Maximum observed path/BIC differences were numerical-roundoff scale: loglik `4.55e-13`, objective `1.36e-12`, BIC `9.09e-13`, EBIC `9.09e-13`.

Remaining before official wiring:

- Rep=1 end-to-end equality test through a runner-compatible interface.
- Runtime benchmark on a representative Eta-group path.
- Optional runtime fallback switch, e.g. `USE_RCPP_HELPERS=0/1`.

## 14. Runner-compatible equality and runtime benchmark

The next diagnostic-only check used a runner-compatible Eta path interface, still without modifying official method or simulation files.

Completed:

- Rep=1 runner-compatible path equality: PASS for 4 lambda scales.
- Best BIC index matched between R-only and Rcpp-helper modes: index `3` in both.
- Best BIC selected support matched: selected q `32` in both modes.
- Best BIC fit-level differences were numerical-roundoff scale: loglik `9.09e-13`; objective `9.09e-13`.
- Small runtime benchmark, 3 repeats per mode: median R-only `0.18` sec; median Rcpp-helper `0.14` sec; median ratio R/Rcpp `1.286`.

Interpretation:

- The current Eta low-level Rcpp helpers pass equality checks through standalone, fit-level, grid, one-iteration, path/BIC, and runner-compatible diagnostics.
- The runtime benchmark is small and should be treated as directional only.
- Official wiring should still be guarded by a fallback switch such as `USE_RCPP_HELPERS=0/1`.
- If larger speed gains are needed, the next Rcpp targets should be `row_logsumexp` and `e_step_vmf`, because they dominate repeated EM likelihood calculations.

## 15. vMF E-step Rcpp prototype status

A second diagnostic-only Rcpp prototype was added for the repeated E-step computations:

- `row_logsumexp_cpp`: row-wise log-sum-exp.
- `e_step_vmf_cpp`: tau/loglik core using R-computed `log_vmf_const` values.

Completed:

- Row log-sum-exp equality: PASS on random, very negative, very positive, and mixed-scale matrices.
- E-step equality: PASS on four small cases with K=3/4/5 and d=8/36/80/120.
- Maximum E-step tau difference: `1.18e-14`.
- Maximum E-step loglik difference: `5.46e-12`.
- Tau row sums remained within numerical tolerance.
- Micro-benchmark, n=300, d=80, K=4, 250 repeated E-steps: R-only `0.19` sec; Rcpp `0.05` sec; ratio R/Rcpp `3.80`.

Important limitation:

- The Rcpp E-step prototype still receives `log_vmf_const` from the existing R implementation. This avoids changing Bessel-function numerical behavior at this stage.
- The prototype is not connected to official fitting code.
- Any official wiring should be guarded by a fallback switch and validated by end-to-end equality tests.

## 16. Eta path diagnostic with Rcpp E-step

The E-step prototype was also tested inside a diagnostic Eta path runner, still without modifying official method or simulation files.

Modes compared:

- R-only.
- Eta low-level helpers in Rcpp.
- Eta low-level helpers plus E-step core in Rcpp.

Completed:

- Eta path equality: PASS for both R-only vs Eta-helper Rcpp and R-only vs Eta-helper + E-step Rcpp.
- Best BIC index matched for R-only and Eta-helper + E-step Rcpp: index `3`.
- Best BIC selected support matched: selected q `73` in both modes.
- Best BIC fit-level differences were numerical-roundoff scale: loglik `0`, objective `0` in the diagnostic run.
- Small runtime benchmark, 3 repeats per mode: median R-only `0.07` sec; median Eta-helper Rcpp `0.06` sec; median Eta-helper + E-step Rcpp `0.03` sec.
- Median speed ratio R/Eta-helper Rcpp `1.167`; R/Eta-helper+E-step Rcpp `2.333`.

Interpretation:

- The E-step helper appears to provide a larger speed gain than the Eta prox/penalty helpers alone.
- Runtime results are still small-path diagnostics and should not be reported as final speedup.
- The next safe integration step is a guarded runtime switch, not unconditional replacement.

## 17. Guarded official switch smoke

A guarded runtime switch was added to the official R files:

- `USE_RCPP_HELPERS=0` or unset: default R-only behavior.
- `USE_RCPP_HELPERS=1`: load Rcpp helper prototypes with `Rcpp::sourceCpp()` and use them where available.

Files touched:

- `r/methods/rossi_barbaro_2022_reproduction.r`: optional Rcpp E-step helper path.
- `r/methods/rb2022_k4_pilot_compare_run.r`: optional Eta prox/penalty/normalization helper path.

Validation:

- Rep=1 official runner smoke was run with `USE_RCPP_HELPERS=0` and `USE_RCPP_HELPERS=1`.
- Setting: K=4, n=120, d=24, nstart=2, max_iter=30, max_path_steps=20.
- Summary outputs matched up to numerical roundoff.
- Maximum numeric difference: `1.02e-12`.
- Character columns matched exactly.
- Result: PASS.

Important:

- The default remains R-only.
- Rcpp helpers are only used when `USE_RCPP_HELPERS=1`.
- Larger production runs should still keep the fallback switch available.

## 18. Guarded switch rep=3 smoke

The guarded switch was also checked in a slightly larger official runner smoke.

Setting:

- K=4, n=300, d=60, rep=3.
- nstart=3, max_iter=50, max_path_steps=40.
- Same seed and lambda grids for `USE_RCPP_HELPERS=0` and `USE_RCPP_HELPERS=1`.

Validation:

- Raw rows matched: 18 vs 18.
- Summary rows matched: 6 vs 6.
- Character columns matched exactly.
- Maximum numeric difference in raw output: `1.02e-10`.
- Maximum numeric difference in summary output: `1.00e-10`.
- Largest difference was in BIC, consistent with accumulated floating point roundoff.
- Result: PASS at tolerance `1e-8`.

Timing caveat:

- OFF elapsed: `4.44` sec.
- ON elapsed: `9.08` sec.
- The ON run includes `Rcpp::sourceCpp()` startup/compile overhead, which dominates this small smoke. This timing should not be used as a final runtime claim.
- For production speed claims, use a precompiled Rcpp path or benchmark larger runs where startup cost is amortized.

## 19. Rcpp guarded switch validation summary

The current guarded Rcpp route is optional. The default remains:

- `USE_RCPP_HELPERS=0` or unset: use the original R helper functions.
- `USE_RCPP_HELPERS=1`: load the diagnostic Rcpp helpers with `Rcpp::sourceCpp()` and use them where available.

Current helper coverage:

- Eta low-level helpers: `prox_eta_centered_cpp`, `eta_centered_penalty_value_cpp`, `normalize_rows_cpp`.
- vMF E-step helper: `e_step_vmf_cpp`, with the vMF normalizing constants still computed by the existing R implementation.

Validation summary:

| Check | Setting | Result | Maximum numeric difference | Timing note |
|:---|:---|:---|:---|:---|
| Standalone helper equality | Unit-style helper tests | PASS | `< 1e-10` target | No runtime claim |
| Fit-level smoke | Small deterministic Eta fit | PASS | roundoff scale | No official connection |
| Extended grid | 24 deterministic comparisons | PASS | loglik/objective about `9.09e-13` | No official connection |
| Official switch rep=3 | K=4, n=300, d=60, rep=3 | PASS | raw `1.0186e-10`, summary `1.0004e-10` | OFF `4.44`s, ON `9.08`s |
| Official switch rep=10 | K=4, n=300, d=60, rep=10 | PASS | raw `1.0186e-10`, summary `2.9310e-14` | OFF `12.916`s, ON `12.280`s |
| Official switch rep=20 | K=4, n=300, d=60, rep=20 | PASS | raw `1.0186e-10`, summary `9.8225e-11` | OFF `26.415`s, ON `17.755`s |

Interpretation:

- Equality is the primary validation target. All guarded-switch diagnostics passed at tolerance `1e-8`.
- The observed numerical differences are in BIC/EBIC/log-likelihood scale and are consistent with floating point roundoff.
- Runtime values are diagnostic only because `Rcpp::sourceCpp()` loading/compilation behavior is not the same as a precompiled production setup.
- The default fallback should stay OFF until the loading strategy and repeated benchmark policy are finalized.

## 20. Precompiled Rcpp loading plan

The current `sourceCpp()` route is useful for development, but it is not the cleanest way to make runtime claims. The loading options are:

| Option | Description | Advantages | Risks / costs | Recommendation |
|:---|:---|:---|:---|:---|
| A. Keep `Rcpp::sourceCpp()` | Compile/load helper files at script startup | Minimal code; easy debugging; already validated | Startup cost in each fresh `Rscript`; less clean for benchmark claims | Good for development only |
| B. `R CMD SHLIB` + `dyn.load()` | Compile a DLL and load it directly | Can remove compile overhead | Rcpp attributes/export glue is less convenient; Windows toolchain handling is fragile | Not first choice |
| C. Small local R package | Put helpers in a local package, e.g. `r/rcpp/pkg/etahelpers` | Cleanest reproducibility; standard Rcpp exports; install/build tests possible | More setup work; package metadata and build flow needed | Best long-term option |
| D. `sourceCpp(cacheDir=..., rebuild=FALSE)` | Keep sourceCpp but use an explicit repo-local cache | Smallest change; avoids repeated compilation within compatible cache | Cache behavior can be environment-specific; still not as clean as a package | Best short-term option |

Toolchain status checked:

- R version: `4.2.1`.
- Rcpp availability: TRUE, version `1.0.12`.
- Rtools/compiler path indicator: TRUE.
- `R CMD config CXX`: `g++ -std=gnu++11`.
- `R CMD SHLIB --help`: available.
- `Rcpp::sourceCpp()` supports both `cacheDir` and `rebuild`.

Small cache probe:

- Probe folder: `r/rcpp/precompile_probe_260708`.
- Probe command: `Rcpp::sourceCpp(cacheDir = ..., rebuild = FALSE)`.
- First helper loads were about `3.12` seconds in the probe.
- Second helper loads in the same R session were effectively `0.00` seconds.
- This supports using an explicit `cacheDir` as the practical short-term route before a local package is built.

Recommended sequence:

1. Keep the current default OFF guarded switch.
2. For research-meeting preparation, use option D if another runtime diagnostic is needed.
3. For thesis/paper reproducibility, move to option C: a small local Rcpp package.
4. Avoid option B unless package setup becomes impractical.

## 21. Runtime benchmark policy

Runtime results should be reported only after equality and loading behavior are controlled.

Current policy:

- Treat rep=10/20 timings as diagnostic only.
- Do not use `sourceCpp()` timings as a publication speed claim.
- Keep equality checks as the first acceptance criterion:
  - same raw/summary row counts,
  - same selected support and selected q,
  - same best IC choice,
  - max numeric difference below `1e-8`,
  - no NA/Inf or fallback warnings.
- Runtime benchmarks should use:
  - repeated runs,
  - the same seed and lambda path,
  - warmed/precompiled helper loading,
  - separate reporting of compile/load time and fit time.

Before any official Rcpp integration claim:

1. Choose the loading route: short-term `sourceCpp(cacheDir=...)` or long-term local package.
2. Run repeated benchmark trials with and without Rcpp helpers.
3. Run a larger smoke with identical seeds and compare raw/summary outputs.
4. Add a CI-like equality script that can be run before any simulation batch.
5. Confirm rollback by setting `USE_RCPP_HELPERS=0`.
6. Document that the R-only path remains the reference implementation.

## 23. R-only vs Rcpp-helper rep=50 runtime diagnostic

This diagnostic benchmark compares the original R-only implementation against
the guarded Rcpp-helper route. The algorithm, seed, lambda grids, and runner
settings are fixed; only low-level helpers are replaced when
`USE_RCPP_HELPERS=1`.

Setting:

- K=4, n=300, d=60.
- rep=50, nstart=3, max_iter=50, max_path_steps=40.
- base seed `20260708`.
- lambda grids: mu `0,50,100,200`; kappa `0,5,10`; eta `0,0.5,1,2,4,8`.
- 3 repeated OFF/ON timing pairs.
- Loading route: `Rcpp::sourceCpp(cacheDir=..., rebuild=FALSE)` with cache warm-up separated from timed runs.

Equality check:

- Raw rows matched: 300 vs 300 in each repeat.
- Summary rows matched: 6 vs 6 in each repeat.
- Character columns matched exactly.
- Maximum raw numeric difference: `1.0186e-10`.
- Maximum summary numeric difference: `9.8225e-11`.
- Differences above `1e-8`: 0.
- Result: PASS at tolerance `1e-8`.

Runtime summary:

| Mode | Repeats | Mean elapsed | SD | Median elapsed | Min | Max |
|:---|---:|---:|---:|---:|---:|---:|
| R-only OFF | 3 | `59.737` sec | `0.342` | `59.860` sec | `59.35` | `60.00` |
| Rcpp-helper ON | 3 | `25.387` sec | `0.031` | `25.380` sec | `25.36` | `25.42` |

Median OFF/ON elapsed ratio: `2.359`.

Interpretation:

- The Rcpp-helper route is stable and faster in this rep=50 diagnostic setting.
- The result is not a publication speed claim yet, because the loading route is still a diagnostic `sourceCpp(cacheDir)` setup rather than a local package or precompiled shared library.
- The R-only path remains the reference implementation and the default fallback through `USE_RCPP_HELPERS=0`.
- Next, if a formal runtime claim is needed, repeat the benchmark after moving helpers into a small local Rcpp package.

## 24. Rcpp benchmark artifact policy

Rcpp runtime benchmark outputs should be split into small documentation artifacts and generated artifacts.

Keep as documentation candidates:

- `runtime_timing_summary.csv`
- `runtime_off_on_comparison_repeats.csv`
- `runtime_benchmark_notes.md`
- benchmark driver scripts under `r/rcpp/`
- this implementation plan

Do not stage raw/generated artifacts:

- `repeat_*` directories
- `sourcecpp_cache` directories
- `temp_methods` directories
- raw CSV files
- log files
- compiled objects or shared libraries such as `.o`, `.so`, and `.dll`

Policy:

- Rcpp benchmark results are diagnostic benchmark outputs, not scientific simulation results.
- Small summary/notes files may be kept to document timing and equality checks.
- Raw repeat outputs, temporary runner copies, and sourceCpp cache/build outputs should remain untracked.
- The partial failed child-process benchmark folder `results/rcpp_sourcecpp_cache_benchmark_260708` was removed after confirming that the final benchmark output is stored in `results/rcpp_vs_r_runtime_benchmark_rep50_260708`.
