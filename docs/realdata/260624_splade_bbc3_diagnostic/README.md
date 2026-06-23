# SPLADE BBC3 실제자료 진단 요약

이 폴더는 BBC 3-class text 자료에 대해 SPLADE sparse lexical representation을 사용한 Eta-group real-data diagnostic 결과를 교수님께 보여줄 수 있는 작은 요약본으로 정리한 것이다.

## 목적

목표는 SPLADE sparse lexical expansion이 Eta-group의 coordinate-level support 해석에 적합한 real-data representation 후보인지 확인하는 것이다. 이 결과는 아직 공식 real-data 결과가 아니며, meeting 또는 appendix 수준의 diagnostic 후보로만 해석한다.

## 포함 파일

- `splade_bbc3_realdata_conclusion_260624.md`: SPLADE BBC3 diagnostic의 핵심 결론과 주요 표.
- `splade_d500_ebic_selected_tokens_overall.csv`: centered eta norm 기준 전체 selected token ranking.
- `splade_d500_ebic_selected_tokens_by_class.csv`: cluster-majority class 기준 signed centered eta score.
- `splade_d500_ebic_cluster_class_table.csv`: cluster와 실제 class의 contingency table.

CSV 파일은 후속 분석 호환성을 위해 영문 컬럼명과 원래 token/class label을 유지했다. 해석은 `splade_bbc3_realdata_conclusion_260624.md`에 한국어로 정리했다.

## 전처리 요약

BBC 원자료에서 `sport`, `entertainment`, `tech` 세 class를 사용하고 class당 최대 120개 문서를 선택해 총 360개 문서를 구성했다. 각 문서는 SPLADE sparse lexical expansion으로 변환했고, `alpha_min3` token filter로 특수 토큰, subword token, 숫자/기호 token, 길이 3 미만 token을 제거했다. 이후 class label을 쓰지 않고 token weight variance 기준으로 상위 500개 coordinate를 고른 뒤, row-wise L2 normalization으로 vMF mixture 입력 행렬을 만들었다.

예를 들어 sport 문서는 `team`, `football`, `match` 같은 expansion token이 커질 수 있고, entertainment 문서는 `film`, `actor`, `award`, tech 문서는 `software`, `computer`, `internet` 같은 token이 커질 수 있다. 단, SPLADE token은 learned lexical expansion token이므로 원문에 그대로 등장한 단어라고 단정하지 않는다.

## 핵심 결론

SPLADE d=500에서 Eta-group EBIC/RICc는 Rossi EBIC보다 훨씬 작은 support를 선택하면서도 BBC3 clustering 성능을 유지했다. 반면 BIC는 여전히 dense support를 선택하므로 real-data sparse support 해석에는 약하다.

SPLADE token은 learned lexical expansion token이며 반드시 원문에 직접 등장한 단어는 아니다. 따라서 selected coordinate는 "원문 단어 목록"이 아니라 "SPLADE가 학습한 lexical/expansion feature"로 해석해야 한다.

## Git에 포함하지 않은 큰 파일

- Payload RDS: `data/bbc/processed/splade_sparse_bbc3_alpha_min3_variance_top500_payload.rds`
- Fit object: `results/splade_bbc3_eta_smoke_top500_260624/cstr_eta_compare_fit.rds`
- SPLADE sparse matrix: `results/splade_sparse_bbc3_smoke_260624/*matrix_top*.csv.gz`
