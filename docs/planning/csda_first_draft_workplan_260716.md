# CSDA 원고 초판 작업 계획

- 작성일: 2026-07-16
- 최종 갱신일: 2026-07-19
- 현재 단계: B4-4. Study B1 Batch B 완료
- 다음 작업: `B4-5. Study B1 Batch C 실행`
- 목표: CSDA 투고용 원고 초판과 Supplement 초안 완성

## 상태 표시

- `[x]` 완료
- `[~]` 진행 중
- `[ ]` 대기
- `[!]` 결과 검토 또는 의사결정 필요

작업이 완료될 때마다 다음 네 항목을 함께 갱신한다.

1. 해당 작업의 상태와 완료일
2. 핵심 결과와 해석 범위
3. 관련 코드, 결과 및 문서 경로
4. 다음 실행 작업과 의사결정 사항

---

## 0. 현재까지 완료된 기반

### 0.1 방법론과 추정 대상

- [x] vMF mixture의 자연모수 정의
  $$
  \eta_k=\kappa_k\mu_k.
  $$
- [x] centered natural-parameter contrast 정의
  $$
  c_{kj}=\eta_{kj}-\bar\eta_j,
  \qquad
  \bar\eta_j=K^{-1}\sum_{\ell=1}^K\eta_{\ell j}.
  $$
- [x] posterior decision support 정의
  $$
  S_{\mathrm{dec}}
  =
  \{j:\|c_{\cdot j}\|_2>0\}.
  $$
- [x] E-CGL을 주 제안 모형으로 지정
  $$
  P_{\mathrm{CGL}}(\eta)
  =
  \lambda_\eta\sum_{j=1}^d\|c_{\cdot j}\|_2.
  $$
- [x] E-ACGL을 adaptive 확장으로 지정
  $$
  P_{\mathrm{ACGL}}(\eta)
  =
  \lambda_\eta\sum_{j=1}^d w_j\|c_{\cdot j}\|_2,
  \qquad
  w_j=(\|c_{\cdot j}^{\mathrm{init}}\|_2+\epsilon)^{-\gamma}.
  $$

### 0.2 구현과 계산

- [x] Eta-group low-level Rcpp helper 구현
- [x] R-only와 Rcpp helper의 output equality 검증
- [x] guarded engine switch 구현
  - `eta_engine="current"`
  - `eta_engine="true_pg"`
- [x] 기본 fallback을 `current`로 유지
- [x] exact centered-support refit 구현
- [x] BIC-after-refit 선택 절차 구현
- [x] KKT 기반 true-PG lambda path 구현
- [x] fixed-\(K\) near-empty component의 `retain and record` 정책 구현
- [x] 중단 복구용 checkpoint runner 구현

### 0.3 true-PG 알고리즘 검증

- [x] gradient 계산 검증
- [x] group proximal update 검증
- [x] majorization condition 검증
- [x] accepted-step penalized objective 비감소 검증
- [x] path endpoint 도달 검증
- [x] path length 61과 120 비교
- [x] path length 120 채택
- [x] \(e_B=5\%\) four-cell current/true-PG rep=20 비교
  - 완료 행: 320/320
  - support 일치: 158/160
  - majorization 통과율: 1.000
  - objective monotonicity 통과율: 1.000

관련 문서:

- `docs/theory/eta_group_algorithmic_properties_260715.md`
- `docs/theory/eta_true_pg_guarded_validation_260715.md`
- `docs/theory/eta_true_pg_rep20_adoption_decision_260716.md`

### 0.4 Study B 난이도 확장

- [x] true-PG, path=120, rep=20 실행
- [x] \(e_B\in\{2.5\%,10\%\}\)
- [x] \(n\in\{300,1000\}\)
- [x] equal/heterogeneous \(\kappa\)
- [x] E-CGL/E-ACGL
- [x] 자동 무결성 검증
  - 완료 행: 320/320
  - 오류: 0
  - cell별 반복 수: 20
  - calibration: PASS
  - endpoint: 1.000
  - majorization: 1.000
  - objective monotonicity: 1.000

관련 파일:

- `r/simulation/eta_true_pg_studyb_difficulty_rep20_260716.r`
- `r/simulation/validate_eta_true_pg_studyb_difficulty_rep20_260716.r`
- `results/eta_true_pg_studyb_difficulty_rep20_path120_260716/eta_true_pg_studyb_difficulty_summary.csv`
- `results/eta_true_pg_studyb_difficulty_rep20_path120_260716/eta_true_pg_studyb_difficulty_notes.md`

---

## A. 알고리즘 최종 확정

### A1. 어려운 Study B cell 원인 진단

- 상태: `[x]` 완료 (2026-07-16)

- [x] 대상:
  $$
  e_B=10\%,\quad n=300,\quad
  \kappa=(30,40,50,60).
  $$
- [x] E-CGL과 E-ACGL의 반복별 support 분포 확인
- [x] common, decision 및 noise 선택 좌표 확인
- [x] selected q, F1, ARI 및 MSE 분포 확인
- [x] selected-model effective component count 확인
- [x] path-level near-empty component와 최종 선택 모형의 관계 확인
- [x] 선택된 \(\lambda_\eta\), BIC 및 support key 확인
- [x] 다음 원인을 구분
  1. 약한 분리와 작은 표본에 따른 통계적 한계
  2. BIC support 선택 문제
  3. near-empty component 또는 initialization 문제
  4. nonadaptive와 adaptive weight의 차이

진단 결과:

| method | selected q | decision q | noise q | F1 | ARI | selected near-empty |
|---|---:|---:|---:|---:|---:|---:|
| E-CGL | 20.75 | 14.55 | 6.10 | 0.808 | 0.613 | 14/20 |
| E-ACGL | 14.80 | 14.45 | 0.35 | 0.937 | 0.679 | 1/20 |

- 두 방법 모두 약한 첫 번째 decision block을 일부 놓쳤다.
  - block 1 centered norm: 9.900
  - E-CGL 선택률: 0.638
  - E-ACGL 선택률: 0.613
  - blocks 2--4 선택률: 두 방법 모두 1.000
- E-CGL의 noise 선택 중앙값은 4, E-ACGL은 0이었다.
- E-ACGL exact-refit BIC는 paired replicate 20/20에서 더 작았다.
- E-ACGL은 F1 19/20, ARI 17/20에서 E-CGL보다 높았다.
- 대표 replicate 4개를 동일 seed로 재현했다.
  - E-ACGL 선택 support는 3/4회 E-CGL 경로에도 존재했다.
  - 세 경우 모두 E-CGL 경로의 해당 support refit은
    `empty_component`로 부적격 처리됐다.
  - E-CGL 선택 support는 4/4회 E-ACGL 경로에 없었다.
- endpoint, majorization 및 objective monotonicity는 20/20에서 통과했다.

원인 판정:

1. true-PG descent 또는 majorization 실패가 아니다.
2. component 1의 decision contrast가 상대적으로 약한 유한표본 환경이다.
3. E-CGL의 uniform path는 산발적 noise coordinate를 포함했다.
4. 일부 유용한 support에도 collapsed-component 상태로 도달하여 exact refit이
   `empty_component`로 실패했다.
5. E-ACGL의 frozen adaptive weights는 noise를 억제하고 안정적인
   support-specific 시작점을 제공했다.

완료 기준:

- [x] E-CGL 성능 저하의 주된 원인을 반복별 결과로 구분
- [x] true-PG core 수정은 하지 않기로 결정
- [x] 해당 cell을 nonadaptive E-CGL의 small-sample weak-contrast limitation으로 기록
- [x] E-CGL은 주 비적응 모형, E-ACGL은 adaptive robustness 확장으로 유지

관련 파일:

- `r/simulation/eta_true_pg_studyb_hard_cell_diagnostic_260716.r`
- `results/eta_true_pg_studyb_hard_cell_diagnostic_260716/hard_cell_diagnostic_notes.md`
- `results/eta_true_pg_studyb_hard_cell_diagnostic_260716/hard_cell_method_summary.csv`
- `results/eta_true_pg_studyb_hard_cell_diagnostic_260716/hard_cell_signal_blocks.csv`
- `results/eta_true_pg_studyb_hard_cell_diagnostic_260716/hard_cell_path_overlap.csv`

### A2. \(e_B=5\%\) 조건 일치 재실행

- 상태: `[x]` 완료 (2026-07-16)

- [x] true-PG engine
- [x] path points=120
- [x] nstart=10
- [x] rep=20
- [x] \(n=300,1000\)
- [x] equal/heterogeneous \(\kappa\)
- [x] E-CGL/E-ACGL
- [x] BIC-after exact support refit
- [x] 동일 calibration 및 seed 규칙 적용

목적:

- \(e_B=2.5\%,5\%,10\%\) 결과를 동일한 실행 조건에서 비교한다.
- 기존 \(e_B=5\%\) 결과의 nstart와 calibration 차이를 제거한다.

검증 결과:

- 완료 행: 160/160
- 오류 행: 0
- cell별 valid repetitions: 20
- endpoint coverage: 1.000
- majorization pass: 1.000
- objective monotonicity: 1.000
- achieved oracle error:
  - equal \(\kappa\): 0.05194
  - heterogeneous \(\kappa\): 0.04698

주요 결과:

| n | \(\kappa\) | method | selected q | decision q | noise q | F1 | ARI |
|---:|---|---|---:|---:|---:|---:|---:|
| 300 | equal | E-CGL | 16.15 | 16.00 | 0.10 | 0.996 | 0.854 |
| 300 | equal | E-ACGL | 16.15 | 16.00 | 0.10 | 0.996 | 0.854 |
| 300 | heterogeneous | E-CGL | 16.85 | 16.00 | 0.85 | 0.976 | 0.861 |
| 300 | heterogeneous | E-ACGL | 16.20 | 16.00 | 0.20 | 0.994 | 0.863 |
| 1000 | equal | E-CGL/E-ACGL | 16.00 | 16.00 | 0.00 | 1.000 | 0.856 |
| 1000 | heterogeneous | E-CGL/E-ACGL | 16.10 | 16.00 | 0.10 | 0.997 | 0.877 |

관련 파일:

- `results/eta_true_pg_studyb_eb05_rep20_path120_nstart10_260716/eta_true_pg_studyb_difficulty_summary.csv`
- `results/eta_true_pg_studyb_eb05_rep20_path120_nstart10_260716/eta_true_pg_studyb_difficulty_notes.md`
- `results/eta_true_pg_studyb_difficulty_combined_rep20_path120_260716/studyb_true_pg_difficulty_summary.csv`
- `results/eta_true_pg_studyb_difficulty_combined_rep20_path120_260716/studyb_true_pg_method_contrasts.csv`
- `results/eta_true_pg_studyb_difficulty_combined_rep20_path120_260716/studyb_true_pg_difficulty_notes.md`

### A3. 최종 알고리즘 규칙 고정

- 상태: `[x]` 논문용 실행 규칙 확정 (2026-07-16)

- [x] E-CGL을 주 비적응 제안 모형으로 유지
- [x] E-ACGL을 adaptive robustness 확장으로 유지
- [x] true-PG를 논문용 confirmatory E-series engine으로 채택
- [x] 패키지 기본 engine은 all-model runner와 public API 검증 전까지
  `current`로 유지
- [x] path length=120 고정
- [x] BIC-after exact refit을 main selector로 확정
- [x] near-empty component는 fixed-\(K\)에서 `retain and record`
- [x] 최종 hyperparameter 표 작성

| 항목 | 논문용 설정 |
|---|---|
| E-series engine | `true_pg` |
| lambda path | KKT-geometric |
| path points | 120 |
| `pg_lambda_min_ratio` | \(10^{-2}\) |
| `pg_lambda_max_factor` | 1.05 |
| dense nstart | 10 |
| outer max iterations | 100 |
| PG inner max iterations | 200 |
| exact refit max iterations | 160 |
| exact refit retry iterations | 440 |
| inner optimizer max iterations | 80 |
| E-ACGL adaptive exponent | \(\gamma=1\) |
| E-ACGL stabilizer | \(\epsilon=10^{-6}\) |
| support selector | BIC-after exact refit |
| refit shortlist | 0, 모든 고유 support refit |
| near-empty policy | fixed-\(K\), retain and record |
| package default | 임시로 `current` 유지 |

의사결정 경계:

- 기존 `current` 결과를 true-PG 결과로 소급해 표기하지 않는다.
- true-PG를 채택하면 논문 핵심 simulation의 E 계열을 동일 engine으로 다시 계산한다.

---

## B. 논문용 시뮬레이션 확정

### B1. 비교 모형 구성 고정

#### 본문 모형

- [x] Dense vMF
- [x] M-L
- [x] M-GL
- [x] E-CGL
- [x] E-ACGL

#### Supplement 또는 ablation 모형

- [x] M-AGL
- [!] E-CL: matched true-PG 구현 후 추가
- [ ] spherical k-means
- [ ] sparse k-means
- [ ] dbmovMFs

### B2. 최종 통합 runner

- [x] 모든 비교 모형이 같은 simulated data와 seed를 사용
- [x] 동일 dense initialization을 모든 방법에 제공
- [x] E-CGL/E-ACGL에 true-PG 적용
- [x] M 계열에 method-specific fixed-support refit 적용
- [x] E 계열에 exact centered-Eta support refit 적용
- [x] 모형별 IC/df의 차이를 명시
- [x] checkpoint 및 resume 지원
- [x] raw, summary 및 notes 출력 구분
- [x] rep=5 pilot 실행

대표 pilot 설정:

$$
e_B=5\%,\quad n=300,\quad d=200,\quad K=4,\quad
\kappa=(30,40,50,60).
$$

모든 방법은 같은 표본, seed 및 `nstart=10` dense initialization을 사용했다.
경로 길이는 120이며, 각 방법의 모수 제약에 맞는 refit 뒤 BIC를 계산했다.

| method | valid | selected q | common q | decision q | noise q | F1 | ARI | MSE eta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense vMF | 5/5 | 200 | 4 | 16 | 180 | 0.148 | 0.737 | 2.985 |
| M-L | 5/5 | 200 | 4 | 16 | 180 | 0.148 | 0.745 | 2.934 |
| M-GL | 5/5 | 20 | 4 | 16 | 0 | 0.889 | 0.861 | 0.255 |
| M-AGL | 5/5 | 20 | 4 | 16 | 0 | 0.889 | 0.861 | 0.255 |
| E-CGL | 5/5 | 16 | 0 | 16 | 0 | 1.000 | 0.867 | 0.198 |
| E-ACGL | 5/5 | 16 | 0 | 16 | 0 | 1.000 | 0.867 | 0.198 |

해석 범위:

- M-GL/M-AGL은 prototype에 존재하는 common 좌표 4개와 decision 좌표
  16개를 함께 선택했다.
- E-CGL/E-ACGL은 centered decision support 16개를 선택했다.
- M-L은 component-entry sparsity를 추정하므로 coordinate-union q만으로
  성능을 결론 내리지 않는다. Study A에서 prototype entry support와
  decision support를 분리해 평가한다.
- M-GL/M-AGL은 Rossi 공식 방법이 아닌 diagnostic group variants이다.
- 이 결과는 통합 실행과 비교 규칙을 확인한 rep=5 pilot이며 논문 최종
  성능표가 아니다.
- 공유 dense initialization과 선택된 모든 method-specific refit은
  5/5회 수렴했다.

관련 파일:

- `r/simulation/studyb_all_model_helpers_260716.r`
- `r/simulation/studyb_all_model_pilot_260716.r`
- `results/studyb_all_model_pilot2_rep5_260716/studyb_all_model_summary.csv`
- `results/studyb_all_model_pilot2_rep5_260716/studyb_all_model_notes.md`

### B3. Study A: Rossi-style bridge

- [x] pilot: \(K^\ast=4,d=100,n=200\)
- [x] pilot overlap 5%
- [x] prototype zero fraction 10%, nonzero fraction 90%
- [x] prototype entry support와 decision support를 분리해 평가
- [x] M-L Rossi BIC-before와 matched BIC-after를 분리
- [x] spherical k-means 및 all-model bridge 비교
- [ ] full grid: \(n=200,1000\)
- [ ] full grid: overlap \(=2.5\%,5\%\)
- [ ] full grid: prototype zero fraction \(=5\%,10\%,15\%\)

Rossi 논문의 sparsity 5%, 10%, 15%는 각 directional mean의 zero-entry
fraction이다. 따라서 pilot의 10% sparsity는 component별 nonzero fraction
90%에 해당한다.

Pilot 설정:

$$
K=4,\quad n=200,\quad d=100,\quad
\text{target overlap}=5\%,\quad
\text{prototype zero fraction}=10\%.
$$

| method | valid | ARI | decision q | decision recall | decision F1 | selected entries | nonzero-entry F1 | zero-entry precision | zero-entry recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense vMF | 5/5 | 0.824 | 100.0 | 1.000 | 1.000 | 400.0 | 0.947 | NA | 0.000 |
| Spherical k-means | 5/5 | 0.750 | NA | NA | NA | NA | NA | NA | NA |
| M-L Rossi BIC | 5/5 | 0.814 | 100.0 | 1.000 | 1.000 | 326.8 | 0.860 | 0.119 | 0.220 |
| M-L BIC-after | 5/5 | 0.814 | 100.0 | 1.000 | 1.000 | 326.8 | 0.860 | 0.119 | 0.220 |
| M-GL | 5/5 | 0.421 | 17.4 | 0.174 | 0.286 | 69.6 | 0.285 | 0.104 | 0.850 |
| M-AGL | 5/5 | 0.386 | 15.6 | 0.156 | 0.260 | 62.4 | 0.259 | 0.103 | 0.865 |
| E-CGL | 5/5 | 0.591 | 46.4 | 0.464 | 0.583 | NA | NA | NA | NA |
| E-ACGL | 5/5 | 0.563 | 30.8 | 0.308 | 0.442 | NA | NA | NA | NA |

검증 결과:

- achieved oracle overlap은 평균 4.70%, 범위 4.42%--5.26%였다.
- M-L 공식 BIC의 평균 ARI 0.814는 기존 동일 cell 재현 결과와 같은
  범위이며, 원문에서 보고한 zero-entry 회복의 낮은 precision 경향도
  재현되었다.
- 무작위 component별 zero pattern의 coordinate union은 100개이므로
  centered-Eta decision support도 사실상 dense하다.
- Dense vMF의 decision F1=1은 모든 좌표를 유지한 결과이며 변수선택
  성능을 의미하지 않는다. M-L도 이 setting에서는 coordinate union이
  100이므로 prototype entry 지표가 주 평가 기준이다.
- E-CGL/E-ACGL의 과소선택은 이 DGP가 sparse decision support가 아니라
  sparse prototype entry를 목표로 설계됐다는 차이를 반영한다.
- M-GL/M-AGL은 Rossi 공식 방법이 아니며 dense coordinate support에서
  BIC가 강한 과소선택을 보인 diagnostic variants이다.
- M-L BIC-before와 BIC-after는 이 pilot에서 같은 support를 선택했다.
- 이 결과는 rep=5 bridge pilot이며 최종 성능표가 아니다.

관련 파일:

- `r/simulation/studya_rossi_bridge_pilot_260717.r`
- `r/simulation/studyb_all_model_helpers_260716.r`
- `results/studya_rossi_bridge_pilot_rep5_260717/studya_bridge_summary.csv`
- `results/studya_rossi_bridge_pilot_rep5_260717/studya_bridge_notes.md`

### B4. Study B: Oracle Bayes error

- [x] \(e_B=2.5\%,5\%,10\%\) 통합
- [x] \(n=300,1000\)
- [x] equal/heterogeneous \(\kappa\)
- [x] common q=4, decision q=16, noise q=180
- [x] 모든 본문 비교 모형을 같은 replicate에서 평가
- [x] 최종 반복 수 결정
  - all-cell integration pilot: rep=3
  - one-cell integration pilot: rep=5
  - hard-cell validation: rep=20
  - final: 12개 cell 모두 rep=100
- [x] final calibration seed와 data seed 분리
- [x] 같은 `(e_B, kappa)`에서 `n=300/1000` DGP calibration 고정
- [x] 최종 실행을 A/B/C checkpoint batch로 분할
- [x] final raw에 true-PG majorization, line-search, endpoint 및
  near-empty diagnostic 저장
- [x] final Batch A: `n=300` 6개 cell, cell별 rep=100
- [x] final Batch B: `n=1000`, `e_B=2.5%,5%` 4개 cell, cell별 rep=100
- [x] final batch validation runner 작성 및 parse 검사
- [x] Batch A validation 이후 B2 pilot을 잇는 guarded continuation 시작
- [x] Batch A validation gate 확인

All-model grid pilot 설정:

$$
K=4,\quad d=200,\quad
n\in\{300,1000\},\quad
e_B\in\{2.5\%,5\%,10\%\}.
$$

$$
\kappa\in
\{(45,45,45,45),(30,40,50,60)\},
\quad
(q_C,q_D,q_N)=(4,16,180).
$$

검증 결과:

- 12/12 cell 완료
- 216/216 method-replicate 행 유효
- 오류 및 비수렴 0
- 중복 `(cell, rep, method)` key 0
- oracle-error calibration 최대 절대오차 0.00328
- E-CGL과 E-ACGL은 각각 10/12 cell에서
  `(common q, decision q, noise q)=(0,16,0)`을 정확히 회복
- M-GL/M-AGL은 대부분의 cell에서 `(4,16,0)`을 선택하여
  common coordinate를 유지
- 예외 cell:
  $$
  n=300,\quad e_B=10\%,\quad
  \kappa=(30,40,50,60).
  $$
  - E-CGL: selected q=25.33, noise q=10.00, F1=0.738, ARI=0.567
  - E-ACGL: selected q=15.00, noise q=0.67, F1=0.925, ARI=0.629
- 위 예외 방향은 기존 E-series rep=20 hard-cell 진단과 일치하지만,
  현재 all-model grid는 rep=3 integration pilot이므로 최종 성능표로
  사용하지 않는다.

Hard-cell all-model rep=20 재검증:

$$
n=300,\quad e_B=10\%,\quad
\kappa=(30,40,50,60).
$$

- 120/120 method-replicate 행 유효
- 오류, 비수렴, 중복 key, zero-support 행 0
- achieved oracle error=0.10328 (target 0.10000)
- M-GL: selected q=21.20, common q=4.00, decision q=15.35,
  noise q=1.85, F1=0.830, ARI=0.671
- M-AGL: selected q=19.55, common q=4.00, decision q=15.05,
  noise q=0.50, F1=0.848, ARI=0.681
- E-CGL: selected q=21.65, common q=0.15, decision q=14.80,
  noise q=6.70, F1=0.796, ARI=0.625
- E-ACGL: selected q=15.40, common q=0.00, decision q=14.80,
  noise q=0.60, F1=0.942, ARI=0.682
- E-CGL의 common-coordinate 제거는 유지됐으나 noise 과대선택 변동이
  남았다. E-ACGL은 이 조건에서 support recovery와 centered-Eta MSE를
  안정화했다.
- 위 결과는 가장 어려운 단일 cell에 대한 targeted diagnostic이며,
  Study B 전체의 최종 결과로 사용하지 않는다.

관련 파일:

- `r/simulation/studyb_all_model_pilot_260716.r`
- `r/simulation/studyb_all_model_grid_pilot_260717.r`
- `results/studyb_all_model_grid_pilot_rep3_260717/studyb_all_model_grid_summary.csv`
- `results/studyb_all_model_grid_pilot_rep3_260717/studyb_all_model_grid_calibration.csv`
- `results/studyb_all_model_grid_pilot_rep3_260717/studyb_all_model_grid_notes.md`
- `results/studyb_all_model_hard_cell_rep20_260717/studyb_all_model_summary.csv`
- `results/studyb_all_model_hard_cell_rep20_260717/studyb_all_model_hard_cell_rep20_interpretation.md`
- `r/simulation/studyb_all_model_final_run_260717.r`
- `docs/planning/studyb_final_execution_plan_260717.md`

### B5. Study B2: ambient-dimension sensitivity

- [x] 목적 분리: (d) 증가에 따른 noise-coordinate 부담 평가
- [x] primary grid: ((n,d)=(300,100),(300,200),(300,500))
- [x] sample-size recovery: ((n,d)=(1000,200),(1000,500))
- [x] (e_B=5\%), equal/heterogeneous concentration shape
- [x] (q_C=4,q_D=16,q_N=d-20) 고정
- [x] dimension-specific global Eta scale calibration 채택
- [x] 기존 fixed-kappa (d=500) 결과는 stress diagnostic으로 분리
- [x] scale-calibration runner 작성 및 parse 검사
- [x] 활성 좌표 방향과 relative kappa scale 불변성 수치 검증
- [x] fixed-DGP all-model cell wrapper 작성 및 parse 검사
- [x] 신규 6개 cell checkpoint pilot orchestrator 작성 및 parse 검사
- [x] 신규 6개 cell rep=5 calibration/path pilot
- [x] (d=500) path 120/240 coverage 비교
- [ ] pilot 통과 후 신규 6개 cell rep=50
- [ ] F1, noise selection rate, log10(MSE Eta) boxplot 생성

### B6. Shared-background

- [ ] prototype support와 decision support가 다른 setting 구성
- [ ] M 계열의 common-coordinate 선택 확인
- [ ] E 계열의 common-coordinate 제거 확인
- [ ] support decomposition figure 생성

### B7. Dense-support negative control

- [ ] dense decision support에서 E-CGL의 과소선택 가능성 확인
- [ ] clustering accuracy와 support recovery를 구분
- [ ] limitation으로 보고할 결과와 main claim을 분리

### B8. \(K\) 선택 및 misspecification

- [ ] main support simulation에서는 \(K=K^\ast\) 고정
- [ ] 별도 \(K\)-selection study 구성
- [ ] candidate \(K=2,\ldots,8\)
- [ ] dense model에서 EBIC/RICc로 \(K\) 탐색
- [ ] 선택된 \(K\)에서 BIC-after-refit으로 \(\lambda_\eta\) 선택
- [ ] all-in-one selection과 two-stage selection 비교

### B9. 민감도 분석

- [ ] BIC-before와 BIC-after 비교
- [ ] EBIC \(\gamma=0.25,0.5,1\)
- [ ] df-A, df-B 및 df-C
- [ ] path length sensitivity
- [ ] zero-support rate
- [ ] valid-refit rate
- [ ] runtime

완료 기준:

- 본문에 사용할 모든 숫자가 frozen summary 파일과 연결되어야 한다.
- simulation 설계, 모형, seed 및 tuning 규칙이 변경되지 않아야 한다.

---

## C. 이론 절 완성

### C1. Structural properties

- [ ] centered decomposition의 유일성
- [ ] posterior decision support 정의와 해석
- [ ] common coordinate에서 centered contrast가 0이 되는 조건
- [ ] label permutation invariance
- [ ] prototype support와 decision support의 관계

### C2. Algorithmic properties

- [ ] coordinate group proximal map의 closed form
- [ ] true-PG majorization condition
- [ ] accepted-step penalized objective monotonicity
- [ ] 제한된 모수 공간에서 objective value convergence
- [ ] near-empty component 처리 가정

### C3. 주장하지 않을 성질

- [x] global optimum을 주장하지 않음
- [x] complete parameter sequence의 무조건적 수렴을 주장하지 않음
- [x] 일반적인 stationary-point convergence를 현재 단계에서 주장하지 않음
- [x] BIC의 df를 exact effective df로 주장하지 않음

완료 기준:

- proposition, assumption 및 proof sketch가 구분되어야 한다.
- 본문에 둘 결과와 Appendix로 이동할 증명이 구분되어야 한다.

---

## D. 모형 선택 절차 확정

### D1. \(\lambda_\eta\) 선택

- [ ] main selector:
  $$
  \mathrm{BIC}^{\mathrm{refit}}(\lambda)
  =
  -2\ell(\widehat\Theta_\lambda^{\mathrm{refit}})
  +
  \log(n)\,\mathrm{df}_\lambda.
  $$
- [ ] practical df:
  $$
  \mathrm{df}_\lambda
  =
  d+(K-1)m_\lambda+(K-1)I(m_\lambda>0).
  $$
- [ ] exact effective df가 아니라 practical approximation임을 명시

### D2. \(K\) 선택

- [ ] dense 또는 weakly penalized vMF에서 \(K\) 탐색
- [ ] EBIC/RICc를 중심으로 비교
- [ ] \(K\) 고정 후 E-CGL path에서 \(\lambda_\eta\) 선택
- [ ] all-in-one information criterion의 한계를 설명

---

## E. Classic3 실자료 분석 확정

### E1. 자료와 전처리

- [ ] 원자료 출처와 정답 label 확인
- [ ] document-term matrix 생성 규칙 고정
- [ ] 용어 빈도 필터와 차원 결정
- [ ] TF, TF-IDF 또는 binary 표현 결정
- [ ] row-wise unit normalization
- [ ] data leakage가 없는 전처리 확인

### E2. 비교 분석

- [ ] Dense vMF
- [ ] spherical k-means
- [ ] sparse k-means
- [ ] M-L 또는 대표 M 계열
- [ ] E-CGL
- [ ] E-ACGL
- [ ] 동일 \(K\)와 initialization budget 적용

### E3. 평가와 해석

- [ ] ARI와 NMI
- [ ] observed log-likelihood 또는 held-out NLL
- [ ] selected q
- [ ] E-CGL decision terms
- [ ] M 계열 common/prototype terms
- [ ] 선택 단어의 component contrast 시각화

### E4. CSTR 처리

- [ ] 본문에서 상세 CSTR 분석 제거
- [ ] Supplement에 재현성 및 limitation 표 1개 유지
- [ ] prototype-oriented data에서 M-L이 유리할 수 있음을 Discussion에 명시

---

## F. CSDA 원고 초판 작성

### F1. 원고 구조

- [ ] 1. Introduction
- [ ] 2. Model and Posterior Decision Support
- [ ] 3. Centered Group Regularization
- [ ] 4. Structural and Algorithmic Properties
- [ ] 5. Simulation Studies
- [ ] 6. Classic3 Analysis
- [ ] 7. Discussion

### F2. Introduction

- [ ] 고차원 directional mixture의 문제 제시
- [ ] sparse prototype과 sparse decision support의 차이
- [ ] \(\mu\)와 \(\eta=\kappa\mu\)의 역할 차이
- [ ] centered group regularization의 필요성
- [ ] contribution을 3~4개 항목으로 정리
- [ ] 과장된 novelty 또는 superiority 표현 제거

### F3. Methods

- [ ] vMF mixture와 posterior score
- [ ] centered Eta decomposition
- [ ] E-CGL objective
- [ ] E-ACGL adaptive extension
- [ ] true-PG update와 backtracking
- [ ] exact support refit
- [ ] BIC-after-refit과 practical df
- [ ] practical \(K\)-selection procedure

### F4. Simulation Studies

- [ ] simulation 질문을 study별로 명확히 구분
- [ ] Study A: literature bridge
- [ ] Study B: main decision-support recovery
- [ ] shared-background
- [ ] negative control
- [ ] \(K\) selection
- [ ] main 결과와 Supplement 결과 구분

### F5. Classic3 Analysis

- [ ] 자료 및 전처리
- [ ] 비교 방법과 tuning
- [ ] clustering 결과
- [ ] selected decision terms
- [ ] prototype-support 방법과의 차이

### F6. Discussion

- [ ] posterior decision support 중심 claim
- [ ] dense support, weak signal 및 small-\(n\) limitation
- [ ] \(K\) 선택과 sparsity 선택의 분리
- [ ] 계산 비용
- [ ] adaptive 확장의 역할
- [ ] 향후 연구

### F7. Abstract

- [ ] 목적
- [ ] 방법
- [ ] 주요 simulation 결과
- [ ] Classic3 결과
- [ ] claim 범위와 결론

Abstract는 본문 초판이 완성된 후 작성한다.

---

## G. 본문 표와 그림

### G1. 본문 표

- [ ] Table 1. 비교 모형과 penalty 정의
- [ ] Table 2. Simulation design
- [ ] Table 3. Study B main performance
- [ ] Table 4. Shared-background와 negative control
- [ ] Table 5. Classic3 결과

### G2. 본문 그림

- [ ] Figure 1. Support decomposition
- [ ] Figure 2. F1 by oracle Bayes error
- [ ] Figure 3. Selected noise q
- [ ] Figure 4. \(\log(\mathrm{MSE}_{\eta^c})\)
- [ ] Figure 5. Classic3 selected terms 또는 contrast heatmap

### G3. Supplement

- [ ] 전체 simulation tables
- [ ] \(K\)-selection figures
- [ ] EBIC/df/path sensitivity
- [ ] convergence and failure diagnostics
- [ ] runtime
- [ ] CSTR 재현성 결과

---

## H. 제출 형식과 재현성

- [ ] Elsevier/CSDA LaTeX 구조 적용
- [ ] 표 5~6개, 그림 5~7개 이내로 본문 축소
- [ ] 수동 절 번호 제거
- [ ] 모든 equation, table 및 figure cross-reference 확인
- [ ] bibliography 정리
- [ ] data availability statement
- [ ] code availability statement
- [ ] conflict of interest
- [ ] funding statement
- [ ] author contribution
- [ ] AI-use statement
- [ ] Supplement 별도 컴파일
- [ ] 최종 PDF에서 표·그림 크기 검증

---

## I. 원고 초판 완료 기준

- [ ] 모든 핵심 수식과 기호가 일관됨
- [ ] E-CGL과 E-ACGL의 역할이 구분됨
- [ ] main claim이 posterior decision support로 제한됨
- [ ] 본문 숫자가 frozen summary와 일치함
- [ ] simulation과 실자료의 tuning rule이 명시됨
- [ ] 수렴 성질의 주장 범위가 구현과 일치함
- [ ] limitation과 negative-control 결과가 포함됨
- [ ] 표와 그림이 본문 흐름에 맞게 배치됨
- [ ] LaTeX가 오류 없이 컴파일됨
- [ ] 본문과 Supplement 초안이 모두 존재함
- [ ] 내부 TODO가 별도 목록으로 정리됨

---

## J. 실행 순서

| 순서 | 작업 | 상태 | 선행 조건 |
|---:|---|---|---|
| 1 | A1 어려운 Study B cell 진단 | 완료 | 없음 |
| 2 | A2 \(e_B=5\%\) 조건 일치 rep=20 | 완료 | A1 |
| 3 | A3 최종 알고리즘 규칙 확정 | 완료 | A1, A2 |
| 4 | B1~B2 비교 모형 및 통합 runner 고정 | 완료 | A3 |
| 5 | C 이론 절 초안 | 대기 | A3, 방법론 수식 |
| 6 | B3~B9 논문용 simulation 확정 | 진행 중: B3 완료, B4 Batch A 다음 | B2 |
| 7 | E Classic3 최종 분석 | 대기 | A3, D |
| 8 | F1~F3 Introduction/Methods 작성 | 대기 | A3 |
| 9 | F4~F6 Results/Discussion 작성 | 대기 | B, E |
| 10 | G 표·그림 정리 | 대기 | B, E |
| 11 | F7 Abstract 작성 | 대기 | F1~F6 |
| 12 | H 제출 형식과 재현성 검수 | 대기 | F, G |
| 13 | I 원고 초판 완료 검수 | 대기 | 전체 |

병렬 진행 가능 항목:

- A와 B가 진행되는 동안 Introduction의 문제 정의와 관련 연구를 작성할 수 있다.
- 최종 simulation 실행 중 Methods와 이론 절을 작성할 수 있다.
- Classic3 전처리 확정 후 simulation과 병렬로 실자료 분석을 수행할 수 있다.

---

## K. 작업 기록

### 2026-07-19

- Study B1 Batch B를 완료하고 validation gate를 통과했다.
  - 범위: `n=1000`, `e_B=2.5%,5%`, equal/heterogeneous kappa
  - 4/4 cell, 2400/2400행, 방법별 100회
  - 중복 key, 오류, 비수렴 및 zero-support 모두 0
  - oracle-error 최대 절대오차 0.001640
  - E 계열 majorization, line-search acceptance 및 endpoint rate 1.000
  - 총 wall time 약 16.0시간
- `n=1000`에서 E-CGL과 E-ACGL은 네 cell 모두 decision q=16을
  유지하고 common q=0, noise q 평균 0--0.02를 기록했다.
- Batch B summary와 validation 결과를 별도 파일로 고정했다.
  - `results/studyb_all_model_final_rep100_260717/studyb_final_batch_b_n1000_summary.csv`
  - `results/studyb_all_model_final_rep100_260717/studyb_final_batch_b_n1000_validation_notes.md`

### 2026-07-18

- Study B1 Batch A와 validation gate를 완료했다.
  - 6/6 cell, 3600/3600행, 중복 key 0, 오류 0
  - 비수렴 2행은 `e_B=10%`, equal kappa의 Dense vMF에서 발생
  - zero-support 0, oracle-error 최대 절대오차 0.001640
  - E 계열 majorization, line-search acceptance 및 endpoint rate 1.000
- Study B2 dimension-scale calibration을 독립 검증 표본으로 강화했다.
  - calibration n=100,000, validation n=200,000
  - 허용 절대오차 0.0025를 명시적 실행 gate로 사용
  - 6/6 dimension-scenario calibration PASS
  - achieved oracle error 범위: 0.04794--0.05103
- 첫 Study B2 pilot은 isolated environment의 helper scope 오류로 M 계열이
  `sb_e_step`을 찾지 못해 중단됐다. Fitting update는 바꾸지 않고 helper를
  runner 환경에 source하도록 수정했으며, 오류 checkpoint 행은 재계산하도록
  보완했다.
- 새 출력 폴더에서 Study B2 all-model rep=5 pilot을 다시 실행했다.
  - 출력: `results/studyb_dimension_all_model_pilot_rep5_validated_260718`
  - 6/6 cell, 180/180행, 방법별 5회, 오류·비수렴·zero-support 0
  - E 계열 majorization, line-search acceptance 및 endpoint rate 1.000
  - `n=1000,d=500`에서는 E-CGL/E-ACGL 모두 `(q_C,q_D,q_N)=(0,16,0)` 회복
  - `n=300,d=500`, equal kappa에서는 E-CGL의 noise 과대선택이 남았고
    E-ACGL이 이를 줄였다. 이 결과는 rep=5 pilot 진단으로만 해석한다.
- `n=300,d=500`에서 path 120/240을 동일 seed로 비교했다.
  - 두 경로 모두 E 계열 endpoint, majorization 및 line-search gate 통과
  - path 240은 unique support 후보를 늘렸으나 selected q, F1 및 ARI를
    개선하지 못함
  - path 240의 wall time은 path 120보다 약 64% 증가
  - Study B2 본 실행 path는 120으로 고정하고 240은 sensitivity로 유지
  - 결과: `results/studyb_dimension_path_sensitivity_rep5_260718`

### 2026-07-17

- Study B를 B1 core difficulty와 B2 ambient-dimension sensitivity로 분리했다.
- B1은 `d=200`의 12개 cell, cell별 rep=100 규칙을 유지한다.
- B2는 `(q_C,q_D)=(4,16)`을 고정하고 `d=100,200,500`에서 noise
  dimension의 효과를 평가한다.
- `d=500`에서는 absolute kappa를 고정하지 않고 relative kappa shape를
  유지한 dimension-specific global Eta scale calibration을 사용한다.
- Study B1 Batch A를 2026-07-17 10:45 KST에 시작했다.
  - 범위: `n=300`, `e_B=2.5%,5%,10%`, equal/heterogeneous kappa
  - 출력: `results/studyb_all_model_final_rep100_260717`
  - 첫 checkpoint에서 replicate 1의 Dense vMF와 M-L 행이 오류 없이 기록됐다.
- Study B2 scale calibration runner를 작성했다.
  - 파일: `r/simulation/studyb_dimension_scale_calibration_260717.r`
  - 활성 좌표의 direction-geometry 최대 차이: (2.78\times10^{-17})
  - relative kappa scale 최대 차이: (4.44\times10^{-16})
  - Batch A와 CPU가 경합하지 않도록 full Monte Carlo calibration 실행은
    Batch A validation 이후로 보류했다.
- Study B2 all-model pilot 실행 구조를 준비했다.
  - fixed-DGP cell wrapper:
    `r/simulation/studyb_dimension_all_model_cell_260717.r`
  - six-cell orchestrator:
    `r/simulation/studyb_dimension_all_model_pilot_260717.r`
  - 기존 all-model fitting, path, refit 및 checkpoint 로직은 변경하지 않고
    dimension-calibrated DGP만 isolated runner environment에 주입한다.
- Study B1 final batch validator를 작성했다.
  - 파일: `r/simulation/validate_studyb_final_batch_260717.r`
  - cell별 행 수, method/rep key 중복, 오류, 비수렴, zero support,
    calibration 오차, E-series majorization, line-search, endpoint 및
    near-empty path diagnostic을 함께 검사한다.
- guarded continuation을 시작했다.
  - script: `scripts/run_studyb_after_batch_a_260717.ps1`
  - watcher PID: 27176
  - Batch A 종료 후 validation PASS인 경우에만 dimension scale calibration과
    six-cell rep=5 pilot을 순차 실행한다.
  - 각 단계의 nonzero exit status에서 체인을 중단한다.

### 2026-07-16

- true-PG Study B \(e_B=2.5\%,10\%\) rep=20 실행 완료
- 320/320행, 오류 0
- calibration, endpoint, majorization 및 objective monotonicity 검증 통과
- 어려운 cell 확인:
  $$
  e_B=10\%,\quad n=300,\quad
  \kappa=(30,40,50,60).
  $$
- 다음 작업을 A1 반복별 원인 진단으로 지정
- A1 어려운 Study B cell 진단 완료
  - E-CGL selected near-empty: 14/20
  - E-ACGL selected near-empty: 1/20
  - E-ACGL BIC 우세: 20/20
  - true-PG endpoint, majorization 및 monotonicity: 모두 통과
  - 원인을 weak block, nonadaptive support path 및
    collapsed-component refit initialization의 결합으로 판정
- true-PG core는 수정하지 않고 A2 조건 일치 \(e_B=5\%\) 실행으로 이동
- A2 \(e_B=5\%\) 조건 일치 rep=20 완료
  - 완료 행 160/160, 오류 0
  - endpoint, majorization 및 monotonicity 모두 1.000
  - \(n=300\), heterogeneous \(\kappa\)에서 E-ACGL이 E-CGL보다
    noise q를 0.65 줄이고 F1을 0.018 높임
- \(e_B=2.5\%,5\%,10\%\) 통합 summary 완료
  - 총 480/480행, 오류 0
  - \(n=1000\)에서는 모든 난이도에서 두 방법 모두 거의 정확한 support 회복
  - adaptive 확장의 차이는 주로 작은 표본의 heterogeneous \(\kappa\)에서 나타남
- A3 논문용 알고리즘 규칙 확정
  - true-PG, KKT path 120, BIC-after exact refit
  - E-CGL main, E-ACGL adaptive extension
  - 패키지 기본값 변경은 B2 이후로 보류
- B1~B2 비교 모형 및 통합 runner 완료
  - 동일 표본, seed 및 dense initialization 적용
  - Dense vMF, M-L, M-GL, M-AGL, E-CGL 및 E-ACGL rep=5 실행
  - 30/30 method-replicate 유효, 오류 0
  - M 계열은 fixed-M-support refit, E 계열은 exact centered-Eta refit
  - 대표 cell에서 M-group은 common+decision q=20, E-group은
    decision q=16을 선택
  - E-CL은 matched true-PG 구현 전까지 Supplement 대기
- B3 Study A Rossi-style bridge pilot 완료
  - 논문 sparsity 정의를 zero fraction으로 정정
  - 목표 overlap 5%, achieved overlap 평균 4.70%
  - M-L Rossi BIC ARI 0.814, zero-entry precision 0.119
  - E-CGL/E-ACGL은 dense decision support에서 과소선택
  - Study A는 literature comparability 및 target mismatch 결과로 유지
