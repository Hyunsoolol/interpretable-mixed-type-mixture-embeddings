# CSDA 원고 초판 작업 계획

- 작성일: 2026-07-16
- 최종 갱신일: 2026-08-03
- 현재 단계: Classic3 전체 자료 본 분석과 five-split Supplement 분석 분리 확정
- 다음 작업: Classic3 전체 자료 payload 검증 후 final guarded true-PG 적합
- 목표: CSDA 투고용 원고 초판과 Supplement 초안 완성

## 상태 표시

- `[x]` 완료
- `[~]` 진행 중
- `[ ]` 대기
- `[!]` 결과 검토 또는 의사결정 필요

최종 원고 용어는 `posterior-score contrast support` 또는
`posterior-score coordinate selection`으로 통일한다. 아래 작업 기록의
`posterior decision support`는 같은 추정 대상을 가리키는 과거 명칭이다.

## 계획 갱신 규칙

앞으로 simulation, 코드 검증, 이론 정리, 실자료 분석 및 원고 작업을
수행할 때마다 이 문서에 다음 항목을 함께 갱신한다.

- 문서 상단의 `현재 단계`와 `다음 작업`
- 해당 작업의 체크박스와 완료일
- 핵심 설정, 검증 결과 및 해석 범위
- 생성한 summary, notes, figure 및 원고 파일 경로
- 재실행 금지 또는 추가 검토가 필요한 항목

작업이 완료될 때마다 다음 네 항목을 함께 갱신한다.

1. 해당 작업의 상태와 완료일
2. 핵심 결과와 해석 범위
3. 관련 코드, 결과 및 문서 경로
4. 다음 실행 작업과 의사결정 사항

---

## 교수님 피드백 후 비교 구조 재검토 (2026-07-22)

핵심 질문은 E-CGL의 효과를 raw-\(\mu\) sparsity와 비교하는 것이 아니라,
동일한 centering 및 coordinate-group 구조에서 penalty target만 비교하는 것이다.

- [x] M-L의 논문 목적함수, 구면 제약, 고정점 M-step, path 및 IC를 현재
  구현과 대조
- [x] \(\|\mu_k\|_2=1\)에서도 \(\|\mu_k\|_1\)은 상수가 아니므로
  M-L penalty 자체는 수학적으로 유효함을 확인
- [x] M-L의 threshold-normalize 식이 임의 후처리가 아니라 Rossi 식
  (22)--(26)의 constrained fixed-point update임을 확인
- [x] M-CGL의 estimand와 고정 support 제약을 정의
  $$H=I_K-K^{-1}\mathbf 1\mathbf 1^\top,\qquad
  \mathcal P_{M\text{-}CGL}(\mu)=\lambda_\mu\sum_{j=1}^d
  \|(H\mu)_{\cdot j}\|_2.$$
- [x] M-CGL diagnostic optimizer와 objective/constraint trace 구현
  - exact fixed-support refit 구현 및 단위 검증 완료
  - one-lambda ADMM penalized fit 단위 검증 완료
  - 17-point penalized support path와 BIC-after 단위 검증 완료
- [x] fixed-support refit의 row norm, inactive-coordinate equality,
  likelihood nondecrease 및 inner convergence 단위 검증
- [x] penalized diagnostic의 \(\lambda_\mu=0\) dense equivalence, row norm,
  ADMM residual 및 observed penalized-objective 비감소 검증
- [x] common-\(\kappa\), heterogeneous-\(\kappa\), same-\(\mu\)/different-\(\kappa\)
  세 진단에서 rep=1 실행
  - common-\(\kappa\): M-CGL/E-CGL 모두 q=8, 두 support \(F_1=1.000\)
  - heterogeneous-\(\kappa\): population \(q_{H\mu}=8\), \(q_{H\eta}=12\)로 분리
  - same-\(\mu\)/different-\(\kappa\): E-CGL은 \(q_{H\eta}=12\)를 회복,
    M-CGL은 q=1을 선택; 성능 결론이 아닌 estimand 차이 진단
- [x] M-CGL Rcpp/ADMM 구현, 다중해상도 path 및 runtime 검증 완료
  - 최종 1,100회에서 중앙 실행시간은 312.18초/rep
  - 60점과 120점 path의 support 합집합을 refit하고, path 240은 민감도로 검증
- [x] M-CGL을 사전 지정한 11개 estimand/sample-size cell에서 rep=100 평가
  - common-\(\kappa\)에서는 M-CGL과 E-CGL의 target-specific 결과가 유사
  - heterogeneous-\(\kappa\)에서는 \(S_\mu\)와 \(S_\eta\)의 차이를 확인
- [x] 최종 원고 역할 확정: E-CGL은 주 방법, E-ACGL은 adaptive sensitivity,
  M-CGL은 directional companion이며 M-ACGL은 제외
- [x] 서로 다른 estimand의 \(F_1\)을 직접적인 우열 지표로 사용하지 않음

감사 및 설계 문서:

- `docs/planning/m_cgl_matched_comparison_plan_260722.md`

---

## 전문가 검토 반영 계획 (2026-07-22)

### 즉시 수행: 원고 일관성 보정

- [x] 본문이 Supplement에 실제로 제공되는 결과만 가리키도록 수정
  - cellwise uncertainty 제공 범위
  - 외부 비교 모형 및 수렴 진단 제공 범위
- [x] posterior-score contrast support의 좌표계 의존성 명시
  - component label permutation에는 불변
  - 임의의 feature-axis rotation에는 불변이 아님
  - token-aligned SPLADE 해석과 dense embedding 해석의 차이 명시
- [x] \(\kappa\) 상한의 구현 범위 감사 및 최종 package에 반영
  - `kappa_cap=1e6`을 초기값, accepted true-PG proposal 및 exact refit에 적용
  - 상한을 넘는 proposal은 수용하지 않고 경계 도달 여부를 진단에 기록
  - finite concentration bound를 사용하되 mixture 전역 최적성이나 전체
    parameter sequence의 수렴은 주장하지 않음
- [x] support-recovery 결과가 fixed \(K\)에 조건부라는 범위를 Abstract,
  Methods, Discussion에서 일관되게 유지
- [x] 본문과 Supplement 재컴파일, reference·citation·layout 경고 0 확인

즉시 수정의 완료 조건:

1. 본문이 Supplement에 없는 표나 진단을 제공한다고 주장하지 않는다.
2. 좌표 선택 결과의 feature-axis 의존성이 Model 및 Discussion에 명시된다.
3. \(\kappa\) 상한과 true-PG 반복의 실제 구현 범위가 원고 설명과 일치한다.
4. fixed-\(K\) support recovery와 별도 \(K\)-selection diagnostic이 구분된다.
5. 본문·Supplement가 fatal error, undefined reference/citation 및
   overfull box 없이 컴파일된다.

### 추가 계산: 제출 전 보강

- [x] E-CL/E-CGL/E-ACGL matched ablation을 final true-PG 기준으로 재구성
  - 기존 rep=20은 projected-update diagnostic engine 결과임을 확인
  - exact centered-support refit과 post-refit BIC는 동일하지만 penalty path
    optimizer가 final Study B와 다르므로 직접 성능 근거로 사용하지 않음
  - 새 true-PG diagnostic의 proximal/objective test와 rep=5 smoke 통과
  - entry fragmentation 또는 within-coordinate selection stability 지표 추가
  - group \(L_2\)의 장점은 결과 확인 전까지 구조적 직관과 rep=20 진단으로 제한
- [x] matched ablation rep=100 확장 및 Supplement 표 교체
  - path 120, `nstart=10`, S1 및 Study B hard cell에서 600/600
    method-repetition 행 유효, 오류와 수치 검증 실패 0
  - wall time 7.82시간
  - S1에서는 세 방법 모두 decision q=16, noise q=0, F1=1.000
  - hard cell에서 E-CL/E-CGL/E-ACGL의 noise q는 11.20/5.48/0.47,
    F1은 0.737/0.822/0.948
  - 기존 projected-update rep=20과 혼합하지 않고 final true-PG 결과로 교체
- [ ] Rossi-style bridge를 rep=50 이상으로 확장하거나 본문 claim을 pilot 수준으로 축소
- [ ] unbalanced mixing proportions 및 compressible-support robustness 추가
- [x] spherical \(k\)-means를 공통 simulation panel에 포함하고, 상세
  convergence/runtime table은 Supplement에 배치

### 재현성 및 제출 패키지

- [x] 공개 entry point의 `eta_engine` 정책 확정
  - 최종 실험과 package default는 `true_pg`
  - `current` engine은 과거 결과 재현 및 회귀 진단용으로만 유지
  - true-PG 단위·통합·Rcpp 일치성 검증 완료
- [~] Classic3 source, near-duplicate rule, SPLADE model/revision 및 vocabulary
  filter provenance 문서화 완료; 전체 자료 payload checksum 고정 대기
- [ ] frozen result checksum, 실행 명령 및 공개 code release 범위 정리
- [ ] 교신 연락처, CRediT, data/code availability, AI-use disclosure는
  제출 직전 저자 확인 후 반영

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
- [x] posterior-score contrast support 정의
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
- [x] package default를 `true_pg`로 전환하고 `current`는 명시적 호환 옵션으로 유지
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

### A3. 2026-07-16 중간 알고리즘 규칙 기록

- 상태: `[x]` 당시 실행 규칙 기록; A4의 최종 규칙으로 대체됨

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

### A4. 원고용 최종 실행 규칙 (2026-08-03)

- [x] E-CGL을 주 비적응 방법, E-ACGL을 adaptive sensitivity로 고정
- [x] package와 원고 실행의 E-series engine을 `true_pg`로 고정
- [x] M-L 및 E-series path를 240점으로 고정
- [x] M-CGL은 60점과 120점 path의 support 합집합을 refit하고,
  path 240을 민감도 분석으로 사용
- [x] simulation dense initialization을 `nstart=10`으로 고정
- [x] 모든 고유 support에 estimand-preserving exact refit을 적용
- [x] `exact`는 support constraint의 정확한 적용을 뜻하며 global mixture
  optimum을 뜻하지 않음을 명시
- [x] 방법별 generic nominal dimension을 사용한 BIC-after-refit을 main
  practical selector로 고정
- [x] `kappa_cap=1e6`을 초기값, accepted true-PG proposal 및 refit에 적용
- [x] fixed-\(K\) near-empty component는 `retain and record`로 처리
- [x] E-ACGL은 \(\gamma=1\), \(\epsilon=10^{-6}\), median-normalized
  frozen adaptive weights를 사용

원고의 최종 성능 수치는 A4 규칙과 일치하는 2026-08-03 frozen results만
사용한다. A3 이전의 path 120 및 `current` engine 결과는 개발·민감도 기록으로
보존하며 최종 성능표와 혼합하지 않는다.

---

## B. 논문용 시뮬레이션 확정

### B1. 비교 모형 구성 재검토

#### 본문 공통 패널

- [x] spherical \(k\)-means: 외부 clustering 기준
- [x] Dense vMF, shared/free \(\kappa\): density 기준
- [x] M-L: Rossi sparse-prototype reference
- [x] E-CGL: posterior-score contrast support의 주 제안 방법
- [x] E-ACGL: adaptive sensitivity; 일률적 우월성은 주장하지 않음

#### 사전 지정 directional panel

- [x] M-CGL: E-CGL과 centering/group 구조를 맞춘 directional companion
- [x] common/heterogeneous \(\kappa\), estimand separation 및 표본크기 조건을
  포함한 11개 고유 cell에서 rep=100 평가

#### Supplement, ablation 또는 제외

- [x] E-CL: matched true-PG rep=100 ablation
- [x] M-GL/M-AGL: 과거 raw-\(\mu\) diagnostic으로만 보존
- [x] sparse \(k\)-means: 실자료 외부 기준으로 사용
- [x] M-ACGL: 최종 원고 비교군과 simulation에서 제외
- [x] dbmovMFs: 최종 공통 패널에서 제외

### B1.1 원고용 최종 simulation evidence freeze (2026-08-03)

- [x] 24개 고유 DGP cell, cell당 rep=100 완료
- [x] main jobs 236/236, method-repetition rows 15,500/15,500
- [x] selector groups 5,900/5,900, oracle paired rows 1,600/1,600
- [x] missing, duplicate, unexpected key 및 ERROR row 모두 0
- [x] M-L/E 계열 path 240, M-CGL 60·120 support 합집합 및 path 240 민감도
- [x] BIC-after-refit, 방법별 nominal dimension 및 target-specific support 적용
- [x] E-CGL 주 specification과 E-ACGL adaptive sensitivity 역할 확정

핵심 범위:

- sparse posterior-score 조건에서 E-CGL은 대부분 참 \(q_\eta=16\)에 근접
- \(e_B=0.10,n=300\), heterogeneous \(\kappa\)에서 E-CGL
  \(F_{1,\eta}=0.768\), E-ACGL \(F_{1,\eta}=0.948\)
- \(n=2000\)에서 M-CGL과 E-CGL은 해당 표본크기 panel의 모든 조건에서
  target-specific \(F_1=1\), exact-support rate 1
- dense/high-dimensional 조건은 exact recovery가 성립하지 않는 적용 범위로 유지

최종 근거 문서:

- `docs/simulations/csda-final-simulation-results_260803.md`
- `docs/meetings/csda-manuscript-draft-sections1-4_260803.md`

### B2. 통합 runner의 초기 pilot 기록

아래 rep=5 표는 2026-07-16의 실행 구조 점검 기록이며, 원고 성능 근거는
B1.1의 rep=100 결과로 대체한다.

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
- [x] final Batch C: `n=1000`, `e_B=10%` 2개 cell, cell별 rep=100
- [x] final 12-cell consolidation 및 validation
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
- [x] pilot 통과 후 신규 6개 cell rep=50
- [x] F1, noise selection rate, log10(MSE Eta) boxplot 생성
- [x] 기존 Study B1의 d=200 rep=100 결과와 결합
- [x] 10개 cell, 4,200개 방법별 raw row 통합 검증
  - error row=0, duplicate key=0, zero-support row=0
  - E-series majorization/line-search/path-endpoint rate=1.0
- [x] n=300에서 d=500으로 증가할 때 E-CGL의 noise 선택 증가 확인
  - equal: noise q=3.00, F1=0.970
  - heterogeneous: noise q=0.90, F1=0.974
- [x] n=1000,d=500에서 E-CGL/E-ACGL의 회복 확인
  - selected q=16.02~16.08, noise q=0.02~0.08, F1=0.998~0.999

완료 산출물:

- `results/studyb_dimension_all_model_rep50_260719/studyb_dimension_final_summary.csv`
- `results/studyb_dimension_all_model_rep50_260719/studyb_dimension_final_validation.csv`
- `results/studyb_dimension_all_model_rep50_260719/studyb_dimension_final_notes.md`
- `docs/simulations/figures/studyb_dimension_boxplot_f1_260719.*`
- `docs/simulations/figures/studyb_dimension_boxplot_noise_rate_260719.*`
- `docs/simulations/figures/studyb_dimension_boxplot_log10_mse_eta_260719.*`

### B6. Shared-background

- [x] prototype support와 decision support가 다른 setting 구성
- [x] M 계열의 common-coordinate 선택 pilot 확인
- [x] E 계열의 common-coordinate 제거 pilot 확인
- [x] confirmatory rep=50 실행 및 검증
- [x] support decomposition figure 생성

Pilot 설정 및 결과 (`rep=5`, 2026-07-20):

- 설정: `K=4`, `n=1000`, `d=200`, `(q_C,q_D,q_N)=(80,20,100)`,
  heterogeneous `kappa=(30,40,50,60)`, target oracle error 5%
- 비교: Dense vMF, M-L, M-GL, M-AGL, E-CGL, E-ACGL
- 공통 규칙: nstart=10, path=120, BIC-after exact refit;
  E-CGL/E-ACGL은 true-PG 사용
- calibration: achieved oracle error 0.050175 (MCSE 0.000488)
- 30/30 method-replicate rows 완료, error row=0
- E-CGL/E-ACGL: 평균 `(common,decision,noise)=(0,20,0.2)`, F1=0.995
- M-GL: 평균 `(3,20,0)`, F1=0.939
- M-AGL: 평균 `(19.8,20,0.2)`, F1=0.745
- E 계열 majorization, line-search acceptance 및 path endpoint rate=1.000;
  boundary warning=0
- pilot은 방향성 확인용이며, 아래 confirmatory rep=50에서 같은 패턴을 재검증함

산출물:

- `results/studyd_shared_background_all_model_pilot_rep5_260720/`

Confirmatory 결과 (`rep=50`, 2026-07-20):

- 300/300 method-replicate rows, method별 valid rep=50, error row=0
- calibration: achieved oracle error 0.050175
- E-CGL/E-ACGL: 평균 `(common,decision,noise)=(0,20,0.02)`, F1=1.000,
  ARI=0.868, MSE eta=0.071
- M-GL: 평균 `(15.26,20,0)`, F1=0.766
- M-AGL: 평균 `(21.16,20,0.02)`, F1=0.695
- E 계열 true-PG majorization, line-search acceptance 및 endpoint rate=1.000;
  selected zero-support=0, boundary warning=0
- rep=5 pilot과 동일한 방향을 확인했으며, prototype support와 posterior
  decision support의 차이를 본문 구조 진단 결과로 사용한다.

완료 산출물:

- `results/studyd_shared_background_all_model_rep50_260720/studyb_all_model_summary.csv`
- `results/studyd_shared_background_all_model_rep50_260720/studyd_shared_background_rep50_validation_260720.md`
- `docs/simulations/figures/studyd_shared_background_support_decomposition_rep50_260720.png`
- `docs/simulations/figures/studyd_shared_background_support_decomposition_rep50_260720.pdf`

### B7. Dense-support negative control

- [x] dense decision support에서 E-CGL의 과소선택 가능성 확인
- [x] clustering accuracy와 support recovery를 구분
- [x] limitation으로 보고할 결과와 main claim을 분리
- [x] modern all-model four-cell pilot 실행 및 검증
  - `K=4`, `d=200`, `(q_C,q_D,q_N)=(4,80,116)`
  - `e_B=10%`, `n=300,1000`, equal/heterogeneous kappa
  - Dense vMF, M-L, M-GL, M-AGL, E-CGL, E-ACGL
  - nstart=10, path=120, BIC-after method-specific refit, E-series true-PG
- [x] pilot 120/120 rows, error=0, E-series true-PG gate 통과
- [x] E-CGL zero-support 4건을 `n=300`, equal-kappa에서 확인
- [x] `n=1000`, equal-kappa에서 E-CGL/E-ACGL의 q=80 회복 확인
- [x] 동일 four-cell confirmatory rep=50 실행 및 통합 검증
  - 4/4 cell, 1,200/1,200 method rows, error=0, duplicate=0
  - E-series true-PG 400/400, majorization/line-search/endpoint gate 통과
  - `n=300`, equal kappa: E-CGL/E-ACGL decision q=17.70/21.78,
    zero-support=31/17, ARI=0.134/0.198
  - 같은 cell에서 Dense vMF/M-L ARI=0.419/0.429로 E 계열보다 높음
  - `n=1000`, equal kappa: E-CGL/E-ACGL decision q=79.50/78.86,
    F1=0.994/0.992로 회복
  - heterogeneous kappa에서 E-CGL은 더 많은 decision과 noise를 유지하고,
    E-ACGL은 noise를 줄이면서 더 많은 weak decision을 제외함
  - dense support 자체보다 표본 크기와 coordinate-level contrast의 결합이
    sparse group selection의 limitation을 결정함

완료 산출물:

- `results/studyd_dense_support_all_model_rep50_260720/studyd_dense_support_rep50_summary.csv`
- `results/studyd_dense_support_all_model_rep50_260720/studyd_dense_support_rep50_validation.csv`
- `results/studyd_dense_support_all_model_rep50_260720/studyd_dense_support_rep50_notes.md`

### B8. \(K\) 선택 및 misspecification

- [x] main support simulation에서는 \(K=K^\ast\) 고정
- [x] 별도 \(K\)-selection study 설계 감사 완료 (2026-07-20)
- [x] candidate \(K=2,\ldots,8\) 및 modern six-model 비교 구조 확정
- [x] modern K-grid runner와 독립 validator 구현
- [x] `rep=1`, \(K=\{2,4,6\}\) smoke validation 통과
- [x] `rep=5`, \(K=2,\ldots,8\) pilot 실행 및 validation
- [x] dense model의 AIC/BIC/EBIC/RIC/RICc/ICL/test NLL 비교
- [x] dense stage-1 criterion rep=50 확증 및 독립 validation (2026-07-21)
- [x] 선택된 \(K\)에서 BIC-after-refit으로 support 선택
- [x] all-in-one selection과 two-stage selection pilot 비교
- [!] 고정 stage-1 criterion은 미확정: AIC/test NLL과 IC penalty 결과가 상충
- [x] repeated-holdout NLL rep=1 smoke 및 rep=5 pilot 검증 (2026-07-21)
- [x] resampling two-stage와 all-in-one E-series 선택 pilot 비교
- [x] focused E-CGL/E-ACGL K-grid rep=10 확증 및 독립 validation
- [x] repeated-holdout NLL rep=10 확장 및 two-stage 비교

구현 감사:

- 과거 `paper_eta_k_selection_diag_260714.r`는 구형 Eta engine과 selector를
  사용하므로 최종 B8 결과로 재사용하지 않는다.
- 최신 `studyb_all_model_pilot_260716.r`의 true-PG, Rcpp 및 모형별
  BIC-after-refit 구조를 K-grid diagnostic runner에 재사용한다.
- pilot은 `n=1000`, `d=200`, `e_B=5%`, equal/heterogeneous kappa,
  `rep=5`, path 120으로 구성한다.
- 예상 결과는 420행이며 wall time은 약 3.5--5시간이다.
- 세부 감사 문서:
  `docs/planning/b8_k_selection_implementation_audit_260720.md`
- smoke 결과는 36/36행, 오류와 중복 key 0으로 validation gate를 통과했다.
  E-CGL/E-ACGL은 모두 true-PG를 사용했고, 모든 penalized 모형은
  method-specific BIC-after-refit을 사용했다. BIC 재계산 최대 오차는
  \(1.281\times10^{-9}\)였다.
- smoke wall time은 1,915.1초였다. rep=1에서 dense stage-1 기준은
  equal kappa의 BIC가 \(K=4\), heterogeneous kappa의 BIC가 \(K=2\)를
  선택했다. 반면 방법별 all-in-one BIC/EBIC에서 E-CGL, E-ACGL,
  M-GL 및 M-AGL은 두 시나리오 모두 \(K=4\)를 선택했다. 반복 1회
  결과이므로 선택 빈도에 대한 결론은 rep=5 pilot 이후 내린다.
- rep=5 pilot은 420/420행, 오류와 중복 key 0으로 15개 validation gate를
  통과했다. runner wall time은 31,415.9초(8.73시간)였다.
- Dense stage-1에서 AIC와 independent test NLL은 두 시나리오 모두
  \(K=4\)를 5/5 선택했다. BIC의 \(K=4\) 선택률은 equal 3/5,
  heterogeneous 0/5였고, EBIC/RIC/RICc는 모두 \(K=2\)를 선택했다.
  따라서 dense EBIC/RICc를 고정된 1단계 규칙으로 채택하지 않는다.
- 방법별 all-in-one BIC/EBIC에서는 E-CGL, E-ACGL, M-GL 및 M-AGL이
  두 시나리오 모두 \(K=4\)를 5/5 선택했다. M-L은 equal 2/5,
  heterogeneous 0/5였다.
- \(K=4\)에서 E-CGL/E-ACGL은 `(common,decision,noise)=(0,16,0)`,
  M-GL/M-AGL은 `(4,16,0)`을 평균적으로 선택했다. \(K>4\)에서는
  E-CGL의 noise 과대선택이 증가했고 E-ACGL이 상대적으로 안정적이었다.
- 세부 결과:
  `results/studyc_k_selection_all_model_pilot_rep5_260720/`

Dense stage-1 rep=50 확증:

- 결과: `results/studyc_k_stage1_dense_rep50_260721/`
- 700/700행, 오류 0, 중복 0, 13/13 validation PASS
- AIC와 independent test NLL은 equal/heterogeneous kappa에서 모두
  \(K=4\)를 50/50 선택했다.
- BIC의 \(P(\widehat K=4)\)는 equal에서 0.78, heterogeneous에서
  0.00이었다. EBIC/RIC/RICc는 두 조건 모두 \(K=2\)를 50/50 선택했다.
- 따라서 strong-penalty IC를 고정된 dense stage-1 규칙으로 채택하지
  않는다. 실자료 적용 전 resampling NLL과 focused E-series K-grid를
  별도로 검증한다.

Repeated-holdout NLL rep=5 pilot:

- 결과: `results/studyc_k_resampling_nll_pilot_rep5_260721/`
- 350/350행, 오류·중복·비수렴 0, 19/19 validation PASS
- 각 data repetition에서 무작위 80/20 holdout을 5회 구성하고,
  \(K=2,\ldots,8\) dense fit의 validation NLL을 비교했다.
- minimum NLL과 1-SE 규칙은 equal/heterogeneous kappa 모두
  \(K=4\)를 5/5 선택했다. 같은 5개 repetition에서 dense BIC의
  \(K=4\) 선택은 equal 3/5, heterogeneous 0/5였다.
- holdout NLL로 \(K=4\)를 선택한 뒤 기존 true-PG E-grid 결과를 연결하면,
  E-CGL/E-ACGL 모두 `(common,decision,noise)=(0,16,0)`을 선택했다.
  평균 ARI는 equal 0.874, heterogeneous 0.867--0.868이었다.
- 이 결과는 실자료용 predictive K-selection의 가능성을 지지하지만,
  data repetition이 5회이고 penalized path를 holdout마다 재적합한 결과는
  아니므로 최종 규칙으로 확정하지 않는다.

Focused E-series 및 repeated-holdout rep=10 확증:

- focused K-grid 결과:
  `results/studyc_k_selection_focused_e_rep10_260721/`
- 420/420행, 오류·중복 0, 15/15 validation PASS
- E-CGL/E-ACGL의 BIC와 EBIC는 equal/heterogeneous kappa 모두
  \(K=4\)를 10/10 선택했다.
- \(K=4\)에서 두 방법 모두
  `(common,decision,noise)=(0,16,0)`, F1=1.000을 유지했다.
- repeated-holdout 결과:
  `results/studyc_k_resampling_nll_rep10_260721/`
- 700/700행, 오류·중복·비수렴 0, 19/19 validation PASS
- minimum NLL은 두 kappa 조건 모두 \(K=4\)를 10/10 선택했다.
  1-SE 규칙은 equal 10/10, heterogeneous 9/10이었고 나머지 1회는
  \(K=3\)을 선택했다.
- dense BIC의 \(K=4\) 선택은 equal 7/10, heterogeneous 0/10이었다.
- 따라서 main support-recovery 결과는 true \(K\) 고정을 유지한다.
  실자료의 1단계 \(K\) 선택은 minimum repeated-holdout NLL을 현재
  practical candidate로 두되, rep=10 진단을 일반적 일관성 주장으로
  확대하지 않는다.

### B9. 민감도 분석

- [x] BIC-before와 BIC-after 비교
  - current true-PG, n=300/1000, easy/hard cells, E-CGL/E-ACGL, rep=5
  - BIC-before는 E-CGL에서 noise를 더 남겼고 hard n=300에서 refit 실패 1회
- [x] EBIC \(\gamma=0.25,0.5,1\)
  - df-A를 고정하고 동일 exact-refit candidate table에서 비교
  - n=1000에서는 BIC-after와 동일 support
- [x] df-A, df-B 및 df-C
  - BIC를 고정하고 동일 exact-refit candidate table에서 비교
  - n=300 hard E-CGL에서 sparsity/decision recovery trade-off 확인
- [x] path length sensitivity
  - `d=200`: 61/120, `d=500`: 120/240 비교 완료
  - main path=120 유지
- [x] zero-support rate
  - Study B1 7,200행과 Study B2 통합 4,200행에서 0
- [x] selected valid-refit rate
  - Study B1 6,000/6,000, Study B2 신규 cells 1,500/1,500
- [x] runtime
  - Study B1/B2 method-replicate elapsed time 감사 완료

감사 문서:

- `docs/planning/b9_tuning_sensitivity_audit_260721.md`
- `results/studyb_b9_selector_sensitivity_combined_rep5_260721/`

완료 기준:

- 본문에 사용할 모든 숫자가 frozen summary 파일과 연결되어야 한다.
- simulation 설계, 모형, seed 및 tuning 규칙이 변경되지 않아야 한다.

---

## C. 이론 절 완성

### C1. Structural properties

- [x] centered decomposition의 유일성
- [x] posterior linear-score support 정의와 해석
- [x] common coordinate에서 centered contrast가 0이 되는 조건
- [x] label permutation equivariance와 support invariance
- [x] prototype support와 decision support의 관계

### C2. Algorithmic properties

- [x] coordinate group proximal map의 closed form
- [x] true-PG majorization condition
- [x] accepted-step penalized objective monotonicity
- [x] zero acceptance tolerance와 bounded objective 아래의 조건부 objective-value convergence
- [x] near-empty component의 fixed-\(K\) retain-and-record 규칙

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

- [x] main selector:
  $$
  \mathrm{BIC}^{\mathrm{refit}}(\lambda)
  =
  -2\ell(\widehat\Theta_\lambda^{\mathrm{refit}})
  +
  \log(n)\,\mathrm{df}_\lambda.
  $$
- [x] practical df:
  $$
  \mathrm{df}_\lambda
  =
  d+(K-1)m_\lambda+(K-1)I(m_\lambda>0).
  $$
- [x] exact effective df가 아니라 generic nominal dimension에 기초한
  practical approximation임을 원고에 명시

### D2. \(K\) 선택

- [x] main support-recovery claim은 참값에 고정한 \(K=K^\ast=4\)에 조건부
- [x] \(K\)-selection은 support recovery와 분리한 diagnostic으로 수행
- [x] AIC/BIC/EBIC/RIC/RICc/ICL과 independent test NLL rep=50 비교
- [x] criterion 충돌 기록: AIC/test NLL은 \(K=4\), BIC/ICL/EBIC/RIC는
  concentration 구조에 따라 과소선택
- [x] repeated 80/20 holdout NLL rep=5 pilot 및 rep=10 확장 검증
- [x] split 민감도와 1-SE/minimum-NLL 차이를 기록하고 일반적 선택 규칙으로
  확대하지 않음
- [x] \(K\) 고정 후 E-CGL path에서 \(\lambda_\eta\)를 BIC-after-refit으로 선택
- [x] all-in-one information criterion이 \(K\)와 sparsity를 교환할 수 있는
  한계를 설명
- [!] 본문 실자료는 benchmark \(K=3\) 고정 분석이며, 일반적인 data-driven
  \(K\) 선택 규칙은 현재 논문의 주 기여로 주장하지 않음

---

## E. Classic3 실자료 분석 확정

분석 원칙:

- 본문은 전처리 규칙을 고정한 **전체 자료 1회 적합**을 주 분석으로 한다.
- 외부 label은 모형 적합, tuning 및 support 선택에 사용하지 않고,
  benchmark \(K=3\)의 확인과 사후 ARI/NMI 및 component 명명에만 사용한다.
- 기존 five locked 80/20 splits는 삭제하지 않고 Supplement의
  **prespecified five-split stability and held-out assessment**로 유지한다.
- 실자료에는 true coordinate support가 없으므로 TPR, FPR 및 support
  \(F_1\)은 보고하지 않는다.

### E1. 전체 자료 분석 규칙

- [x] 원자료 출처, near-duplicate 제거 및 benchmark label 확인
- [x] SPLADE 표현, 상위 \(d=2{,}000\) 좌표 및 row-wise unit normalization 고정
- [ ] 동일 규칙으로 전체 자료 payload \(n=3{,}883,\ d=2{,}000\) 생성·검증
- [ ] \(K=3\) 고정 분석과 별도 label-free \(K\)-selection diagnostic을 구분
- [ ] final guarded true-PG, nstart 30, path 240 및 exact
  BIC-after-refit 적용 확인

### E2. 전체 자료 비교 적합

- [ ] Dense vMF
- [ ] spherical k-means
- [ ] sparse k-means
- [ ] M-L 및 대표 directional comparator M-CGL
- [ ] E-CGL: 주 방법
- [ ] E-ACGL: adaptive extension
- [ ] 동일 \(K\)와 initialization budget 적용
- [ ] method-specific support에서 exact refit 후 BIC 비교

### E3. 본문 평가와 해석

- [ ] ARI와 NMI
- [ ] selected \(q\)와 \(q/d\)
- [ ] cluster size, observed log-likelihood 및 BIC-after-refit
- [ ] E-CGL decision terms
- [ ] M 계열 common/prototype terms
- [ ] 선택 단어의 component contrast 시각화
- [ ] convergence, numerical warning 및 boundary diagnostic 보고
- [ ] true support 부재로 TPR/FPR/support \(F_1\)을 사용하지 않음을 명시

### E4. Supplement 안정성 분석

- [x] 기존 five locked 80/20 splits와 held-out NLL 결과 보존
- [ ] 전체 자료 최종 적합과 동일한 방법명·tuning 정의로 표 재정렬
- [ ] ARI/NMI, \(q/d\), held-out NLL 및 support Jaccard/selection frequency 보고
- [ ] overlapping splits의 평균·표준편차는 기술통계로만 해석
- [ ] 본문에는 핵심 안정성 결과만 1--2문장으로 요약

### E5. CSTR 처리

- [x] 본문에서 상세 CSTR 분석 제거
- [x] Supplement에 재현성 및 limitation 표 1개 유지
- [x] prototype-oriented data에서 M-L이 유리할 수 있음을 Discussion에 명시

---

## F. CSDA 원고 초판 작성

### F1. 원고 구조

- [x] 1. Introduction
- [x] 2. Model and methodology
- [x] 3. Properties of the support target and algorithm
- [x] 4. Numerical studies
- [x] 5. Real-data applications
- [x] 6. Discussion
- [x] Proposition 증명은 Appendix A로 분리

### F2. Introduction

- [x] 고차원 directional mixture의 문제 제시
- [x] sparse prototype과 posterior-score contrast support의 차이
- [x] \(\mu\)와 \(\eta=\kappa\mu\)의 역할 차이
- [x] centered group regularization의 필요성
- [x] contribution을 3개 항목으로 정리
- [x] 과장된 novelty 또는 superiority 표현 제거

### F3. Methods

- [x] vMF mixture와 posterior score
- [x] centered Eta decomposition
- [x] E-CGL objective
- [x] E-ACGL adaptive extension
- [x] true-PG update와 backtracking
- [x] exact support refit
- [x] BIC-after-refit과 practical df
- [x] fixed-\(K\) claim과 별도 \(K\)-selection diagnostic의 범위

### F4. Simulation Studies

- [x] 4.1 Design and evaluation criteria
- [x] 4.2 Posterior-score support recovery
- [x] 4.3 Directional versus posterior-score estimands
- [x] 4.4 Sample size, oracle benchmarks, and selector sensitivity
- [x] 4.5 Stress conditions and computation
- [x] Rossi bridge, 전체 selector/path 진단, 반복별 분포 및 \(K\)-selection을
  Supplement로 분리

### F5. Classic3 Analysis

- [~] 기존 locked repeated-holdout 초안 존재
- [ ] 전체 자료 1회 적합을 본 분석으로 재작성
- [ ] 비교 방법과 BIC-after-refit tuning을 전체 자료 기준으로 갱신
- [ ] clustering 결과와 selected decision terms 갱신
- [ ] 기존 five-split 결과를 Supplement 안정성 분석으로 이동
- [ ] prototype-support 방법과의 차이를 전체 자료 결과에 맞춰 재검수

### F6. Discussion

- [x] posterior-score contrast support 중심 claim
- [x] dense support, weak signal 및 small-\(n\) limitation
- [x] \(K\) 선택과 sparsity 선택의 분리
- [x] 계산 비용
- [x] adaptive 확장의 역할
- [x] 향후 연구

### F7. Abstract

- [x] 목적
- [x] 방법
- [x] 주요 simulation 결과
- [ ] Classic3 전체 자료 최종 결과 반영
- [x] claim 범위와 결론

현재 초록은 E-CGL을 주 방법으로, E-ACGL을 adaptive extension으로
구분한다. Classic3 문장은 전체 자료 최종 적합 후 갱신하며, 기존 held-out
density 결과는 Supplement 안정성 분석의 범위에서 기술한다.

---

## G. 본문 표와 그림

### G1. 본문 표

- [~] Table 1. 방법별 estimand, penalty 및 concentration 구조
- [~] Table 2. Simulation design과 참 support
- [~] Table 3. 주요 support recovery, estimand separation 및 표본크기 결과
- [ ] Table 4. Classic3 전체 자료의 방법별 적합·좌표 선택 결과

### G2. 본문 그림

- [ ] Figure 1. Support decomposition
- [ ] Figure 2. concentration 구조별 target-specific \(F_1\)
- [ ] Figure 3. 표본크기별 oracle benchmark gap
- [ ] Figure 4. Classic3 전체 자료 E-CGL centered-\(\eta\) contrast

기존 Study B boxplot과 full cell grid는 삭제하지 않고 Supplement figure로
이동한다. 본문 표와 그림에는 같은 수치를 중복하여 제시하지 않는다.

### G3. Supplement

- [~] 전체 simulation tables
- [x] \(K\)-selection figures
- [~] EBIC/df/path sensitivity
- [~] convergence and failure diagnostics
- [x] runtime
- [x] CSTR 재현성 결과
- [ ] Classic3/BBCSport five-split stability and held-out assessment 이동

---

## H. 제출 형식과 재현성

- [x] Elsevier/CSDA LaTeX 구조 적용
- [!] 아래 LaTeX/PDF 검수 완료 표시는 2026-07-22 패키지 기준이다.
  2026-08-03의 새 1--4절과 Classic3 전체 자료 절을 이관한 뒤 다시 검수한다.
- [~] 최종 본문 목표는 표 4개, 그림 4개이며 Classic3 전체 자료 결과 후
  최종 중복·분량을 검수
- [x] 수동 절 번호 제거
- [~] 모든 equation, table 및 figure cross-reference 확인; 최신 원고 이관 후 재검수
- [x] bibliography와 author-year citation 정리
- [!] data availability statement: Classic3 배포 가능 범위 확인 필요
- [!] code availability statement: 공개 URL, release tag/DOI 및 license 필요
- [x] conflict of interest: 없음으로 확인; 최종 제출본 반영 대기
- [x] funding statement: specific grant 없음으로 확인; 최종 제출본 반영 대기
- [!] author contribution: CRediT 역할 확인 필요
- [!] AI-use statement: 투고 시점 Elsevier 정책과 저자 승인 필요
- [~] Supplement 별도 컴파일; 최신 원고 이관 후 재컴파일
- [ ] 최종 PDF에서 표·그림·수식 크기와 잘림 재검증

---

## I. 원고 초판 완료 기준

- [~] 1--4절의 핵심 수식과 기호는 일관됨; 5절 이관 후 전체 재검수
- [x] E-CGL과 E-ACGL의 역할이 구분됨
- [x] main claim이 posterior-score contrast support로 제한됨
- [x] simulation 숫자는 2026-08-03 frozen summary와 일치함
- [x] simulation tuning과 최종 24-cell 실행 범위 확정
- [x] 수렴 성질의 주장 범위가 구현과 일치함
- [x] limitation과 negative-control 결과가 포함됨
- [~] simulation 본문 표·그림 inventory 확정; 최종 figure 제작과 Classic3
  표·그림 갱신 대기
- [~] 이전 LaTeX는 오류 없이 컴파일됨; 최신 1--5절 이관 후 재컴파일
- [~] 이전 본문·Supplement와 최신 1--4절 초안이 존재함; 통합본 갱신 대기
- [x] 내부 TODO와 저자 확인 항목이 별도 목록으로 정리됨

---

## J. 실행 순서

| 순서 | 작업 | 상태 | 선행 조건 |
|---:|---|---|---|
| 1 | A1 어려운 Study B cell 진단 | 완료 | 없음 |
| 2 | A2 \(e_B=5\%\) 조건 일치 rep=20 | 완료 | A1 |
| 3 | A3 최종 알고리즘 규칙 확정 | 완료 | A1, A2 |
| 4 | B1~B2 비교 모형 및 통합 runner 고정 | 완료 | A3 |
| 5 | C 이론 절 초안 | 완료: 본문 proposition, Appendix A 증명 | A3, 방법론 수식 |
| 6 | B3~B9 논문용 simulation 확정 | 완료: B3~B9 실행·검증 | B2 |
| 7 | E Classic3 최종 분석 | 재개: 전체 자료 적합 대기 | A4, D |
| 8 | F1~F3 Introduction/Methods 작성 | 완료: 6절 구조와 Algorithm 1 반영 | A3 |
| 9 | F4~F6 Results/Discussion 작성 | simulation 완료; Real data 재작성 대기 | B, E |
| 10 | G 표·그림 정리 | 진행 | B, E |
| 11 | F7 Abstract 작성 | Classic3 전체 자료 결과 반영 대기 | F1~F6 |
| 12 | H 제출 형식과 재현성 검수 | 1차 완료; 저자·release 정보 대기 | F, G |
| 13 | I 원고 초판 완료 검수 | Classic3 재작성 후 재검수 | 전체 |
| 14 | M-L 논문-코드 구현 감사 | 완료 | 교수님 피드백 |
| 15 | M-CGL estimand·support·refit 명세 | 완료 | 14 |
| 16 | M-CGL diagnostic optimizer 단위 검증 | 완료: exact refit·one-lambda·path PASS | 15 |
| 17 | M-CGL/E-CGL matched rep=1 구조 진단 | 완료 | 16 |
| 18 | M-CGL 다중해상도 runtime 및 사전 지정 panel rep=100 | 완료 | 17 |
| 19 | 비교 구조와 원고 claim 재결정 | 완료: E-CGL 주방법, M-CGL companion | 18 |
| 20 | Classic3 전체 자료 payload 검증 | 대기 | E1 |
| 21 | Classic3 전체 자료 final guarded true-PG 적합 | 대기 | 20 |
| 22 | 본문 Real data 절·Table 4·Figure 4 갱신 | 대기 | 21 |
| 23 | five-split 결과를 Supplement 안정성 분석으로 이동 | 대기 | 22 |
| 24 | Abstract·Discussion·최종 PDF 재검수 | 대기 | 22, 23 |

병렬 진행 가능 항목:

- A와 B가 진행되는 동안 Introduction의 문제 정의와 관련 연구를 작성할 수 있다.
- 최종 simulation 실행 중 Methods와 이론 절을 작성할 수 있다.
- Classic3 전처리 확정 후 simulation과 병렬로 실자료 분석을 수행할 수 있다.

---

## K. 작업 기록

### 2026-08-03

- 최종 원고용 simulation evidence를 동결했다.
  - 24개 고유 DGP cell, cell당 100회 반복
  - main jobs 236/236, method-repetition rows 15,500/15,500
  - selector groups 5,900/5,900, oracle paired rows 1,600/1,600
  - missing, duplicate, unexpected key 및 ERROR row 0
  - E-CGL은 주 방법, E-ACGL은 adaptive sensitivity, M-CGL은 11개
    사전 지정 cell의 directional companion으로 확정
  - 최종 근거:
    `docs/simulations/csda-final-simulation-results_260803.md`
  - 1--4절 원고 초안:
    `docs/meetings/csda-manuscript-draft-sections1-4_260803.md`
- Classic3 실자료 분석의 본문 구조를 통상적인 전체 자료 적용 방식으로
  변경하기로 확정했다.
  - 본 분석: near-duplicate 제거 후 전체 자료
    \(n=3{,}883,\ d=2{,}000,\ K=3\) 1회 적합
  - fitting/tuning: final guarded true-PG, nstart 30, path 240,
    method-specific exact BIC-after-refit
  - 평가: ARI, NMI, selected \(q\), \(q/d\), cluster size,
    observed log-likelihood 및 BIC
  - 해석: E-CGL selected terms와 centered-\(\eta\) contrast
- 외부 label은 fitting, tuning 및 support 선택에 사용하지 않고,
  benchmark \(K\) 확인과 사후 평가·component 명명에만 사용한다.
- 기존 five locked 80/20 splits는 삭제하지 않고 Supplement의
  prespecified stability and held-out assessment로 이동한다.
- 기존 2026-07-11 전체 자료 결과는 이전 계산 경로의 참고치로만 유지한다.
  최종 본문 수치는 final guarded true-PG와 현재 BIC-after-refit 규칙으로
  재적합한 결과에서 확정한다.
- 다음 작업은 전체 자료 payload 검증, 최종 all-method 적합, 본문 Table 4와
  contrast figure 갱신 순서로 진행한다.

### 2026-07-20

- B6 shared-background modern all-model rep=50을 완료했다.
- B7 dense-support negative-control modern all-model rep=50을 완료했다.
- 두 작업공간을 최종 운영 경로로 통합하고 B6/B7 결과, true-PG 코드,
  Rcpp source 및 `etaVmf` local package 보존을 확인했다.
- B8 K-selection 및 misspecification 구현 감사를 완료했다.
  - 과거 all-model K 진단은 구형 Eta engine과 selector를 사용하므로 참고 결과로만 유지한다.
  - 최신 six-model true-PG/BIC-after-refit runner를 K-grid로 확장한다.
  - `n=1000`, `d=200`, `e_B=5%`, 두 kappa 구조, `K=2,...,8`,
    rep=5 pilot은 420행이며 예상 wall time은 약 3.5--5시간이다.
  - `rep=1`, `K={2,4,6}` modern smoke를 실행해 36/36행 validation을
    통과했다. 오류와 중복 key는 0이며 E 계열은 모두 true-PG였다.
  - 모든 penalized selector의 after-refit 적용, BIC 재계산, wrong-K MSE
    미보고, test NLL 및 calibration을 포함한 15개 검증 항목이 통과했다.
  - smoke wall time은 1,915.1초였다.
  - rep=1에서 dense two-stage K 기준은 criterion에 따라 K=2 과소선택이
    나타났고, 방법별 all-in-one E/M-group BIC와 EBIC는 K=4를 선택했다.
  - `rep=5`, `K=2,...,8` pilot을 420/420행으로 완료했다. 오류와 중복은
    0이고, 15개 validation gate가 모두 통과했다.
  - runner wall time은 31,415.9초(8.73시간)였다.
  - Dense AIC와 independent test NLL은 K=4를 10/10 cell-replicate에서
    선택했지만, EBIC/RIC/RICc는 K=2를 10/10 선택했다. Dense BIC도
    heterogeneous kappa에서 K=4를 선택하지 못했다.
  - 방법별 all-in-one E-CGL/E-ACGL 및 M-group BIC/EBIC는 두 시나리오
    모두 K=4를 5/5 선택했다. E 계열의 fixed-K=4 support F1은 1.000이었다.
  - 다음 단계는 dense stage-1 criterion 반복 확대와 고비용 penalized
    E-series K-grid confirmatory 범위를 분리해 결정하는 것이다.

### 2026-07-21

- B8 dense stage-1 K-selection을 rep=50으로 확증했다.
  - 결과: `results/studyc_k_stage1_dense_rep50_260721`
  - 700/700행, 오류 0, 중복 0, 13/13 validation PASS
  - wall time 133.6분, BIC 재계산 최대 오차 \(1.455\times10^{-9}\)
  - AIC와 independent test NLL은 두 kappa 조건에서 \(K=4\)를 50/50 선택
  - BIC는 equal 39/50, heterogeneous 0/50; EBIC/RIC/RICc는 모두 과소선택
- 다음 B8 작업은 고비용 all-model 반복이 아니라 focused E-CGL/E-ACGL
  K-grid와 실자료에 적용 가능한 resampling NLL 규칙의 비교로 제한한다.
- B8 repeated-holdout NLL pilot을 완료했다.
  - 결과: `results/studyc_k_resampling_nll_pilot_rep5_260721`
  - 350/350행, 오류·중복·비수렴 0, 19/19 validation PASS
  - 5 data repetitions 각각에서 5회의 80/20 split을 사용
  - minimum NLL과 1-SE 규칙 모두 equal/heterogeneous kappa에서
    \(K=4\)를 5/5 선택
  - holdout으로 선택한 \(K=4\)에서 E-CGL/E-ACGL은 decision 16개를
    선택하고 common/noise는 선택하지 않음
- 다음 판단은 focused E-CGL/E-ACGL K-grid를 얼마나 반복할지와,
  repeated holdout의 data-repetition 및 split 민감도를 어디까지
  확증할지에 한정한다.
- B8 focused E-series K-grid와 repeated-holdout NLL을 rep=10으로 확장했다.
  - focused K-grid: 420/420행, 15/15 validation PASS
  - E-CGL/E-ACGL BIC·EBIC의 \(K=4\) 선택: 두 kappa 조건에서 모두 10/10
  - repeated holdout: 700/700행, 19/19 validation PASS
  - minimum NLL의 \(K=4\) 선택: equal/heterogeneous 모두 10/10
  - 1-SE 규칙은 heterogeneous에서 1/10회 \(K=3\)을 선택
  - B8의 현재 결론은 fixed-\(K\) support claim과 practical predictive
    K-selection diagnostic을 분리하는 것이다.
- 다음 작업은 B9 tuning sensitivity의 기존 실행 결과를 먼저 감사하고,
  중복 실행을 제외한 최소 보강 범위를 확정하는 것이다.
- B9 tuning sensitivity의 구현과 기존 근거를 감사했다.
  - 감사 문서: `docs/planning/b9_tuning_sensitivity_audit_260721.md`
  - 현재 E 계열은 true-PG path의 모든 고유 support를 exact refit한 뒤
    df-A 기반 BIC-after-refit으로 선택함을 확인했다.
  - `eta_support_ic()`의 df-A/BIC/EBIC 공식과 exact-refit/public-fit
    testthat 검증이 통과했다.
  - current true-PG 근거로 path=120, zero-support, selected-refit 유효성,
    runtime 항목을 완료 처리했다.
  - 기존 df/EBIC 및 before/after 표는 legacy projected engine 결과이므로
    현재 원고의 직접 근거에서 제외했다.
  - 다음 실행은 `n=300,d=200`의 easy-equal 및 hard-heterogeneous 두 cell,
    E-CGL/E-ACGL, rep=1 smoke로 제한한다.
- B9 true-PG selector sensitivity를 rep=5로 완료했다.
  - 실행 범위: `n=300,1000`, easy-equal 및 hard-heterogeneous,
    E-CGL/E-ACGL, 7개 IC/df 규칙
  - `n=300` 및 `n=1000`에서 각각 140/140 rule rows를 생성했으며,
    중복 key와 fitting error는 없었다.
  - main `BIC-after exact refit + df-A`의 선택 모형은 모든 반복에서
    유효했고 zero support는 없었다.
  - `n=300` hard E-CGL의 비교용 BIC-before에서 q=13 support 한 건이
    empty component로 exact refit되지 않았다. 이 실패는 main selector에서
    발생하지 않았다.
  - `n=1000`에서는 모든 after-refit EBIC/df 규칙이 main BIC-after와
    동일한 support를 선택했다.
  - `n=300` hard E-CGL에서 강한 EBIC/df 규칙은 noise 선택을 줄였지만
    decision coordinate도 함께 줄이고 ARI를 낮췄다.
  - 따라서 BIC-after exact refit + df-A를 main practical selector로
    유지하고, EBIC 및 df-B/df-C는 Supplement sensitivity로 둔다.
  - 통합 결과:
    `results/studyb_b9_selector_sensitivity_combined_rep5_260721`
- C1 Structural and Algorithmic Properties 초안과 확정 사양을 작성했다.
  - 확정 사양: `docs/theory/final_algorithm_specification_260721.md`
  - 원고: `docs/submission_csda_draft/manuscript.tex`
  - centered/pairwise support 등가성, pairwise-dispersion identity,
    proximal map, label invariance, fixed-responsibility majorization 및
    accepted generalized-M monotonicity를 다섯 명제로 정리했다.
  - `true_pg`, KKT-geometric path 120, all-support exact refit 및
    BIC-after df-A를 원고 알고리즘과 일치시켰다.
  - 수리통계, 최적화 및 CSDA 편집 관점의 독립 검수를 통과했다.
  - 수치 허용오차가 있는 trace에는 objective convergence를 주장하지 않고,
    zero tolerance의 exact-arithmetic sequence에 한해 조건부 수렴 문장을
    사용한다.
  - LaTeX는 fatal error와 undefined reference 없이 PDF로 컴파일됐다.
    기존 대형 표의 overfull 경고는 원고 슬림화 단계에서 정리한다.
- F2--F3 Introduction 및 방법론 절을 재구성했다.
  - Introduction을 여섯 개 소절에서 일곱 개 연속 문단으로 축약했다.
  - `Model and Posterior Decision Support`, `Centered Group Regularization`,
    `Estimation and Model Selection`을 독립 절로 분리했다.
  - posterior log-odds의 선형 support, raw-Eta/prototype support와의
    estimand 차이 및 E-CGL/E-ACGL 역할을 명시했다.
  - 중복된 proximal map과 pairwise identity 증명은 Structural 절에만
    유지하고, 비교 모형표는 Simulation 절로 이동했다.
  - vMF normalizer의 zero-concentration 연속 확장, 초기
    fixed-responsibility KKT endpoint 범위 및 finite-mixture identifiability
    가정을 독립 전문가 검수에 따라 보완했다.
  - 최종 LaTeX 검증: fatal error 0, undefined reference 0, 46쪽.
    overfull 경고는 51건으로, 대부분 시뮬레이션 대형 표에 남아 있다.
- 다음 작업은 Simulation Studies를 main-text 핵심 결과와 Supplement
  진단으로 분리하고 표 규격을 정리하는 것이다.

### 2026-07-22

- F4 Simulation Studies를 CSDA 본문 흐름에 맞게 재구성했다.
  - 수동 소절 번호와 내부 작업 메모를 제거했다.
  - 본문은 Study B 설계, frozen rep=100 핵심 결과, shared-background,
    dense-support 범위 진단 및 별도 \(K\)-selection 진단으로 축약했다.
  - Study A bridge, matched E-CL ablation, df/EBIC, ambient dimension,
    runtime 및 상세 검증은 `docs/submission_csda_draft/supplement.tex`로
    분리했다.
- frozen Study B 모형 구성을 다시 감사했다.
  - 최종 six-model output은 Dense vMF, M-L, M-GL, M-AGL, E-CGL,
    E-ACGL이다.
  - 기존 원고의 E-CL 행과 boxplot은 과거 실행에 연결되어 있어 본문
    비교에서 제거하고, E-CL은 matched ablation으로만 유지했다.
- 최종 12-cell raw 7,200행에서 boxplot 세 개를 재생성했다.
  - `scripts/plot_studyb_final_boxplots_260722.R`
  - `docs/simulations/figures/studyb_final_boxplot_f1_260722.*`
  - `docs/simulations/figures/studyb_final_boxplot_noiseq_260722.*`
  - `docs/simulations/figures/studyb_final_boxplot_logmse_eta_260722.*`
  - (n)과 concentration pattern을 분리한 12-panel 구성으로 수정했다.
  - pooled/cellwise SD와 Monte Carlo standard error를 자동 산출한다.
- 독립 통계·편집 검수 결과를 반영했다.
  - 본문의 dense-support 진단은 equal-(\kappa)에서 (n=300)과
    (n=1000)을 함께 제시해 finite-sample under-selection과 표본 증가에
    따른 회복을 구분했다.
  - 전체 네 dense-support cell과 selector, ambient-dimension, runtime
    진단은 Supplement에 유지했다.
  - E-CGL/E-ACGL의 cellwise MCSE 최댓값은 F1 0.0081, ARI 0.0064,
    centered-Eta MSE 0.047이었다.
- LaTeX 최종 검증:
  - 본문 39쪽, Supplement 7쪽
  - fatal error 0, undefined reference/citation 0
  - 남은 큰 overfull 경고는 기존 Real-Data 절의 표와 파일 경로에
    집중되어 있어 F5에서 제거한다.
- F5 Classic3 Analysis를 본문 핵심 실자료 분석으로 재구성했다.
  - SPLADE top-2,000, 학습 3,111건·평가 779건, 조건부 \(K=3\) 분석으로
    자료와 평가 규칙을 고정했다.
  - 본문은 비교 모형 성능표 1개와 centered-Eta contrast heatmap 1개로
    축약했다.
  - E-CGL은 1,347개 좌표를 선택해 dense free-\(\kappa_k\)보다 653개를
    제외하면서 같은 ARI 0.9927과 NMI 0.9863을 보였다.
  - 20회 support reselection에서 E-CGL의 Nogueira stability는 0.8840,
    mean Jaccard는 0.9267이었다.
  - vocabulary-size 및 \(K\)-selection 민감도는 Supplement로 분리했다.
- CSTR과 BBC5를 Additional real-data scope analyses로 이동했다.
  - CSTR에서는 M-L의 평균 ARI가 0.8083, E-CGL refit이 0.6153으로,
    prototype-oriented representation에서 decision-support 축소의 한계를
    그대로 보고했다.
  - CSTR에는 true coordinate support가 없으므로 support-recovery 비교로
    해석하지 않는다.
- LaTeX 및 시각 검증:
  - 본문 32쪽, Supplement 12쪽
  - fatal error 0, undefined reference/citation 0
  - 새 Classic3/CSTR 표와 그림의 잘림·겹침 및 overfull 경고 0
- F6 Discussion을 독립 통계 검수와 CSDA 편집 검수에 따라 재구성했다.
  - 9개 수동 번호 소절을 4개 소절로 통합하고 Conclusion을 독립 9절로
    분리했다.
  - Discussion과 Conclusion 분량을 약 1,280단어에서 828단어로 줄이고 Methods와
    Results의 반복 수치·설명을 제거했다.
  - E-CGL을 main estimator, E-ACGL을 adaptive extension으로 유지했다.
  - group penalty의 근거를 일률적 성능 우위가 아니라 coordinate-level
    decision-support estimand와 penalty unit의 일치로 제한했다.
  - dense/weak-signal·small-\(n\), practical BIC df, fixed-\(K\) claim,
    initialization, real-data support 부재 및 SPLADE 의존성을 명시했다.
  - accepted-step objective control과 Rcpp benchmark를 각각 수렴 정리와
    통계적 성능 주장으로 확대하지 않았다.
- LaTeX 및 시각 검증:
  - 본문 30쪽, Supplement 12쪽
  - fatal error 0, undefined reference/citation 0
  - Discussion/Conclusion의 overfull, 겹침 및 수동 번호 중복 0
- F7 Abstract 및 본문 claim 일치 검수를 완료했다.
  - CSDA 편집 관점과 통계 관점의 독립 검수를 사용해 초록을 201단어로
    축약했다.
  - posterior decision support를 pairwise posterior log-score contrast의
    선형 좌표 support로 한정하고, E-CGL을 주 방법, E-ACGL을 adaptive
    extension으로 구분했다.
  - 큰 표본에서의 sparse support 회복만을 일반화하지 않고, 어려운
    작은 표본에서의 false selection과 dense-support에서의 under-selection을
    함께 명시했다.
  - Classic3 결과는 dense free-\(\kappa_k\) 모형과 동일한 held-out
    ARI/NMI, 1,347/2,000 좌표 선택 및 modest held-out density loss로
    정리했다.
  - refit은 exact support constraint 아래의 unpenalized numerical fit으로
    기술했고, 반복 종료는 `convergence` 대신 numerical stopping criterion으로
    표현했다.
  - natural-parameter block은 convex criterion의 exact minimizer가 아니라
    이를 대상으로 하는 proximal-gradient update임을 명확히 하고,
    generalized-M proposition 앞에 penalized auxiliary function을 정의했다.
- LaTeX 및 시각 검증:
  - 본문 31쪽, abstract 201단어
  - fatal error 0, undefined reference/citation 0
  - 첫 페이지의 초록 잘림·겹침 0
- 다음 작업은 제출 형식·재현성 검수와 참고문헌·declaration 정리이다.

- H1 제출 형식·재현성 1차 검수를 완료했다.
  - 본문과 Supplement에 `Hyunsoo Shin`, `Byungtae Seo` 및
    `Sungkyunkwan University, Republic of Korea`를 반영했다.
  - 교신저자는 Byungtae Seo로 표시했으며, 이메일·세부 소속·주소는
    제공되지 않아 임의로 기입하지 않았다.
  - 초록에서 E-CGL, E-ACGL, SPLADE, ARI 및 NMI를 처음 등장할 때
    정의하고, five-item highlights를 작성했다.
  - 본문 인용을 author-year BibTeX 방식으로 통일하고, 정의되지 않은
    citation/cross-reference와 overfull box가 없음을 확인했다.
  - 최종 frozen calibration CSV에 따라 Study B achieved oracle error를
    수정했다. 최대 절대 오차는 0.164 percentage points이다.
  - Supplement에 Study B DGP, calibration/validation 표본 수, bisection,
    seed map, method-specific df, support metric 및 label alignment를
    추가했다.
  - 본문 31쪽·Supplement 13쪽을 재컴파일했다. fatal error,
    undefined citation/reference 및 overfull box는 모두 0이다.
  - 첫 페이지, 본문 핵심 표·그림·참고문헌, Supplement DGP·Table S1·
    K 진단·참고문헌 페이지를 렌더링해 잘림과 겹침이 없음을 확인했다.
  - 제출 전 저자 확인 항목은
    `docs/submission_csda_draft/submission_metadata_pending_260722.md`,
    전체 readiness는 `submission_readiness_audit_260722.md`에 분리했다.
  - 재현성 manifest의 Study B 기준을 true-PG, path 120, 12 cells,
    rep=100 최종 실행으로 갱신했다. 공개 코드에서는 `true_pg` 엔진을
    명시적으로 고정해야 한다.
- 다음 작업은 교신 이메일·우편주소, CRediT, data/code availability,
  AI-use disclosure 및 공개 release 정보를 저자 확인 후 반영하는 것이다.

- 저자 확인에 따라 소속을 `Department of Statistics, Sungkyunkwan
  University, Seoul, Republic of Korea`로 구체화했다.
- specific grant 없음과 competing interests 없음 문구는 declaration 초안에
  보존하고, 현재 review draft 본문에서는 최종 제출 단계까지 제외했다.
- 남은 저자 메타데이터는 교신 이메일, 선택적 ORCID, CRediT 및 AI-use
  disclosure이다.
- CSDA 원고를 여섯 개 본문 절로 재구성했다.
  - 2절은 모형과 centered regularization을 통합하고, 3절은 추정 알고리즘,
    pathwise selection, exact refit 및 structural/algorithmic properties를
    함께 제시한다.
  - 4절은 Numerical studies, 5절은 Real data analysis, 6절은 Discussion으로
    정리하고 독립 Conclusion의 내용을 Discussion 말미로 통합했다.
  - guarded proximal generalized-EM, KKT path, support refit 및 after-refit
    BIC를 13단계 Algorithm 1로 요약했다. E-CGL과 E-ACGL의 차이는 고정된
    adaptive weights로 한정했다.
  - proposition은 본문에 유지하고 전체 증명은 Appendix A로 이동했다.
  - abstract를 171단어로 축약하고, 본문 수치 표와 결과 해석은 변경하지 않았다.
  - TinyTeX 최종 PDF는 32쪽이며 fatal error, undefined reference/citation,
    overfull 및 underfull box가 모두 0이다. 첫 페이지, 알고리즘, 핵심 표,
    실자료 그림, Discussion 및 Appendix의 잘림·겹침이 없음을 확인했다.
  - 최신 원고를 포함하도록
    `docs/submission_csda_draft/overleaf_csda_draft_260722.zip`을 갱신했다.
  - Overleaf ZIP의 10개 소스·그림 파일을 SHA-256으로 검증하고
    원격 `main`에 커밋 `fe44c3e5`로 게시했다.
    - ZIP SHA-256: `C24DD052567D0531850271A428D90BA1C7DF3B332B0008E775EC979144CBE97A`

- CSDA 전문가 검토 후 과학적 일관성 보정을 완료했다.
  - 본문에서 Supplement에 실제로 포함된 결과만 안내하도록 uncertainty,
    ablation, selector, runtime 및 numerical-check 범위를 맞췄다.
  - posterior decision support가 component label permutation에는 불변이지만
    임의의 feature-axis rotation에는 불변이 아님을 Model과 Discussion에
    명시했다.
  - 2026-07-22 감사 당시 `kappa_cap=1e6`은 dense initialization 및
    unpenalized M-step target에만 적용되었다. 이후 final package에서는
    초기값, accepted true-PG proposal 및 exact refit에 같은 상한 검사를
    적용하도록 보완했다. 이 유한 상한을 넘어 mixture 전역 최적성이나
    전체 parameter-sequence convergence는 주장하지 않는다.
  - E-CGL support recovery는 fixed \(K\)에 조건부이고 component-number
    selection은 별도 diagnostic이라는 범위를 다시 고정했다.
  - Classic3 초록 문장을 1,347/2,000 coordinate 유지로 정량화했다.
  - 본문 32쪽과 Supplement 13쪽을 재컴파일했으며 fatal error,
    undefined reference/citation, overfull 및 underfull box는 0이다.
  - 수정 페이지 1, 5, 12, 17, 18, 23, 29를 렌더링해 잘림·겹침과
    부자연스러운 page break가 없음을 확인했다.
- E-CL/E-CGL/E-ACGL matched ablation runner 감사를 완료했다.
  - 기존 rep=20 runner의 exact refit과 post-refit BIC는 세 방법에 동일하게
    적용됐지만, path는 final true-PG가 아닌 projected-update 진단 엔진이었다.
  - Supplement에 이 차이를 명시하고 기존 표를 structural diagnostic으로
    한정했다.
  - 감사 기록:
    `docs/planning/matched_ablation_engine_audit_260722.md`
- final true-PG matched penalty ablation 구현과 검증을 완료했다.
  - E-CL, E-CGL, E-ACGL이 같은 smooth objective, paired data, dense
    initialization, path resolution, exact centered-support refit 및 df-A
    BIC-after-refit을 공유한다.
  - Rcpp E-CL constrained prox는 R 기준과 최대 오차 `2.4e-13`으로
    일치했고, hard-cell integration 결과의 support·F1·ARI가 유지됐다.
  - hard reduced smoke의 합산 방법 시간은 1167.37초에서 72.62초로
    줄었으며, 이는 diagnostic implementation speedup으로만 기록한다.
  - `S1`과 `StudyB_hard`의 rep=5, path 120, `nstart=10` smoke는
    30/30행 유효, 오류 0, 전체 validation gate PASS였다.
  - S1에서는 세 방법 모두 `(common,decision,noise)=(0,16,0)`을
    회복했다. hard cell에서 평균 noise q는 E-CL 13.4, E-CGL 7.0,
    E-ACGL 0.0이고 F1은 각각 0.707, 0.806, 0.981이었다.
  - 이 결과는 방향성 확인용 rep=5이며 원고의 확정 성능표로 사용하지 않는다.
  - 결과:
    `results/eta_true_pg_matched_ablation_smoke_rep5_path120_260722`
  - 구현·해석 감사:
    `docs/planning/matched_ablation_engine_audit_260722.md`
- final true-PG matched penalty ablation rep=100을 완료했다.
  - 실행 시간 7.82시간, 600/600행 유효, 오류 및 수치 검증 실패 0
  - S1에서는 세 방법 모두 `(common,decision,noise)=(0,16,0)`과 F1=1.000
  - hard cell의 `(noise q,F1)`은 E-CL `(11.20,0.737)`, E-CGL
    `(5.48,0.822)`, E-ACGL `(0.47,0.948)`
  - E-CGL-E-CL paired difference는 noise q -5.72 (MCSE 0.82),
    F1 +0.085 (MCSE 0.010)
  - E-ACGL-E-CGL paired difference는 noise q -5.01 (MCSE 0.53),
    F1 +0.126 (MCSE 0.009)
  - Supplement의 기존 projected-update rep=20 표를 final-engine rep=100
    표로 교체했으며 두 결과를 합산하지 않았다.
  - 본문 32쪽과 Supplement 14쪽을 재컴파일했고 fatal error, undefined
    reference/citation, overfull 및 underfull box는 0이었다. Supplement
    Table S2와 본문 Discussion 페이지의 시각 검수도 통과했다.
  - 결과:
    `results/eta_true_pg_matched_ablation_rep100_path120_260722`

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
- Study B1 Batch C를 완료하고 validation gate를 통과했다.
  - 범위: `n=1000`, `e_B=10%`, equal/heterogeneous kappa
  - 2/2 cell, 1200/1200행, 방법별 100회
  - 중복 key, 오류, 비수렴 및 zero-support 모두 0
  - oracle-error 최대 절대오차 0.001175
  - E 계열 majorization, line-search acceptance 및 endpoint rate 1.000
  - wall time 약 9시간 56분
- Study B1 12개 cell을 재계산 없이 통합하고 전체 validation을 통과했다.
  - 12/12 cell, 7200/7200행, 오류 0, zero-support 0
  - 비수렴 2행은 Batch A의 `e_B=10%`, equal-kappa Dense vMF 결과
  - calibration 최대 절대오차 0.001640
  - 같은 `(e_B,kappa)`에서 `n=300/1000`의 calibration parameter 차이 0
  - 최종 summary:
    `results/studyb_all_model_final_rep100_260717/studyb_final_all12_summary.csv`
  - 전체 validation:
    `results/studyb_all_model_final_rep100_260717/studyb_final_all12_final_validation_notes.md`

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
