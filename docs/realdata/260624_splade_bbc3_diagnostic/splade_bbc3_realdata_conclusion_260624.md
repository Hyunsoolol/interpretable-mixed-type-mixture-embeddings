# SPLADE BBC3 실제자료 진단 결과 260624

## 1. 상태

- 이 결과는 real-data 본문 결과가 아니라 diagnostic 결과다.
- 이 요약을 만들기 위해 새 embedding이나 새 Eta-group fit은 실행하지 않았다.
- token table은 저장된 SPLADE d=500 EBIC Eta path fit에서 계산했다. `cstr_eta_compare_fit.rds`에는 refit fit object가 별도로 저장되어 있지 않기 때문이다.
- SPLADE coordinate는 learned lexical expansion token이며, 반드시 원문에 직접 등장한 단어는 아니다.
- dense embedding coordinate는 해석 대상으로 사용하지 않는다.

## 2. 입력과 분석 프로토콜

| 항목 | 값 |
|:---|:---|
| Dataset | BBC 3-class subset |
| Classes | `sport`, `entertainment`, `tech` |
| n | 360 |
| d | 500 |
| Representation | SPLADE sparse lexical expansion |
| Token filter | `alpha_min3` |
| Feature ranking | variance |
| Main diagnostic criterion | EBIC |
| Secondary diagnostic | RICc |
| Baseline | Rossi shared-kappa path, matched TF-IDF diagnostic |

해석 목적은 "SPLADE sparse lexical representation + Eta-group"이 token-level support 해석에 쓸 만한지 확인하는 것이다. 이 단계에서는 official tuning 변경이나 최종 real-data claim을 하지 않는다.

### 데이터 전처리 과정

BBC 원자료에서 `sport`, `entertainment`, `tech` 세 범주를 사용하고, 각 class에서 최대 120개 문서를 선택해 총 360개 문서를 구성했다. 이후 문서를 SPLADE sparse lexical expansion으로 변환한 뒤, 해석 가능한 token coordinate만 남기고 vMF mixture 입력에 맞게 row-wise L2 normalization을 적용했다.

전처리 흐름은 다음과 같다.

1. 원문 문서 선택: `data/bbc/raw/bbc`에서 `sport`, `entertainment`, `tech` 폴더의 문서를 읽는다.
2. SPLADE sparse weight 계산: `naver/splade-cocondenser-ensembledistil`의 masked-LM logits에 `log(1 + ReLU(logits))`와 max-pooling을 적용해 document-token sparse weight를 만든다.
3. token filtering: `alpha_min3` 규칙으로 특수 토큰, subword token, 숫자/기호 token, 길이 3 미만 token을 제거한다.
4. top-d screening: class label을 쓰지 않고 token weight의 variance 기준으로 상위 500개 coordinate를 선택한다.
5. spherical projection: 각 문서 vector를 L2 norm 1로 정규화해 vMF mixture 입력 행렬 `X`를 만든다.
6. payload 변환: `X`, `y`, class label, vocabulary/token table, document metadata를 RDS payload로 저장한다.

예를 들어 sport 문서에서 경기/팀 관련 문맥이 강하면 SPLADE는 `team`, `football`, `match`, `champion` 같은 lexical expansion token에 큰 weight를 줄 수 있다. entertainment 문서에서는 `film`, `actor`, `award`, `oscar` 같은 token이, tech 문서에서는 `software`, `computer`, `internet`, `mobile` 같은 token이 커질 수 있다. 단, 이 token들은 SPLADE가 학습한 lexical expansion feature이므로, 반드시 원문에 그대로 등장한 단어라고 해석하면 안 된다.

이 전처리의 핵심은 주관적 keyword 선택을 하지 않는다는 점이다. 사람이 sport/tech/entertainment 키워드를 미리 고르는 것이 아니라, SPLADE sparse weight와 unsupervised variance screening으로 후보 coordinate를 만든 뒤 Eta-group이 posterior decision contrast 기준으로 support를 다시 선택한다.

## 3. Rossi vs Eta-group 요약

| Method | ARI | Selected q | Kappa ratio | Cluster size |
|:---|---:|---:|---:|:---|
| Rossi EBIC | 0.903 | 489 | 1.000 | 121;121;118 |
| Eta-group EBIC | 0.903 | 206 | 1.034 | 121;121;118 |
| Eta-group EBIC + refit | 0.911 | 206 | 1.092 | 121;120;119 |
| Eta-group BIC + refit | 0.919 | 500 | 1.112 | 121;119;120 |
| Eta-group RICc + refit | 0.903 | 181 | 1.076 | 121;121;118 |

핵심은 ARI 자체의 큰 개선이 아니다. Eta-group EBIC + refit은 Rossi EBIC와 비슷하거나 약간 높은 ARI를 유지하면서 selected q를 489에서 206으로 줄였다. RICc에서도 selected q=181로 같은 방향이 확인된다.

반대로 BIC는 Eta-group에서도 selected q=500, 즉 full support를 선택한다. 따라서 이 BBC3 smoke에서는 BIC가 real-data sparse support 해석 기준으로 너무 약하다. EBIC/RICc는 현재 diagnostic criterion이며, official tuning replacement로 확정하지 않는다.

## 4. Cluster-majority mapping 결과

| Cluster | Majority class | Sport | Entertainment | Tech | Cluster size | Purity |
|---:|:---|---:|---:|---:|---:|---:|
| 1 | sport | 119 | 0 | 2 | 121 | 0.983 |
| 2 | tech | 0 | 6 | 115 | 121 | 0.950 |
| 3 | entertainment | 1 | 114 | 3 | 118 | 0.966 |

cluster-majority mapping은 안정적으로 해석 가능하다. 세 cluster 모두 purity가 0.95 이상이며, sport/tech/entertainment class와 잘 대응된다.

## 5. Class-mapped selected token 해석

아래 표는 EBIC Eta-group fit의 centered eta score를 cluster-majority class 기준으로 정렬한 것이다. `+`는 해당 class 방향으로 eta contrast가 큰 token, `-`는 해당 class와 반대 방향의 token을 뜻한다.

`+`와 `-` 부호는 단순한 token 빈도 차이가 아니라 centered eta score

$$
c_{kj}=\eta_{kj}-\bar{\eta}_j
$$

의 부호다. 예를 들어 `+film`은 entertainment cluster의 posterior decision score를 상대적으로 높이는 방향이고, `-film`은 sport cluster의 score를 상대적으로 낮추는 방향이다.

실제 BBC3 결과에서도 entertainment cluster의 `+film`, `+actor`, sport cluster의 `-film`, tech cluster의 `-actor`, entertainment cluster의 `-champion`처럼 class를 설명하는 positive token과 다른 class와 구분하는 negative contrast token이 함께 확인된다. 다만 이는 hard rule이 아니라 posterior decision score 안의 soft contrast로 해석해야 한다.

### sport cluster

| Rank | Signed token | Centered eta score | Abs score |
|---:|:---|---:|---:|
| 1 | `+champion` | 29.318 | 29.318 |
| 2 | `+match` | 24.747 | 24.747 |
| 3 | `+team` | 23.730 | 23.730 |
| 4 | `+football` | 23.696 | 23.696 |
| 5 | `-film` | -17.344 | 17.344 |
| 6 | `+win` | 17.277 | 17.277 |
| 7 | `+club` | 17.252 | 17.252 |
| 8 | `+player` | 17.141 | 17.141 |
| 9 | `+coach` | 13.937 | 13.937 |
| 10 | `+won` | 13.266 | 13.266 |
| 11 | `-tech` | -12.733 | 12.733 |
| 12 | `+england` | 11.413 | 11.413 |

sport cluster는 `champion`, `match`, `team`, `football`, `club`, `player` 같은 스포츠 관련 token으로 설명된다.

### tech cluster

| Rank | Signed token | Centered eta score | Abs score |
|---:|:---|---:|---:|
| 1 | `+tech` | 24.684 | 24.684 |
| 2 | `+software` | 21.813 | 21.813 |
| 3 | `+computer` | 20.778 | 20.778 |
| 4 | `-actor` | -19.928 | 19.928 |
| 5 | `+internet` | 18.979 | 18.979 |
| 6 | `-won` | -16.649 | 16.649 |
| 7 | `-champion` | -16.094 | 16.094 |
| 8 | `+computers` | 15.651 | 15.651 |
| 9 | `+people` | 14.587 | 14.587 |
| 10 | `+mobile` | 14.280 | 14.280 |
| 11 | `-win` | -14.207 | 14.207 |
| 12 | `+users` | 13.796 | 13.796 |

tech cluster는 `tech`, `software`, `computer`, `internet`, `computers`, `mobile`, `users` 같은 기술 관련 token으로 설명된다.

### entertainment cluster

| Rank | Signed token | Centered eta score | Abs score |
|---:|:---|---:|---:|
| 1 | `+film` | 30.143 | 30.143 |
| 2 | `+award` | 23.095 | 23.095 |
| 3 | `+actor` | 21.928 | 21.928 |
| 4 | `+awards` | 19.639 | 19.639 |
| 5 | `+oscar` | 17.457 | 17.457 |
| 6 | `+movie` | 16.632 | 16.632 |
| 7 | `+singer` | 15.276 | 15.276 |
| 8 | `+actors` | 15.080 | 15.080 |
| 9 | `-champion` | -13.224 | 13.224 |
| 10 | `-player` | -12.888 | 12.888 |
| 11 | `-match` | -12.512 | 12.512 |
| 12 | `+movies` | 12.396 | 12.396 |

entertainment cluster는 `film`, `award`, `actor`, `awards`, `oscar`, `movie`, `singer` 같은 연예/영화 관련 token으로 설명된다.

## 6. 전체 selected token 상위 목록

| Rank | Token | Centered eta norm | Document frequency |
|---:|:---|---:|---:|
| 1 | `film` | 37.057 | 118 |
| 2 | `champion` | 35.965 | 145 |
| 3 | `match` | 30.310 | 104 |
| 4 | `tech` | 30.237 | 82 |
| 5 | `actor` | 29.697 | 212 |
| 6 | `team` | 29.065 | 131 |
| 7 | `football` | 29.022 | 106 |
| 8 | `award` | 28.317 | 99 |
| 9 | `software` | 26.715 | 69 |
| 10 | `computer` | 25.448 | 80 |
| 11 | `awards` | 24.054 | 64 |
| 12 | `internet` | 23.255 | 72 |
| 13 | `win` | 22.578 | 146 |
| 14 | `player` | 21.864 | 148 |
| 15 | `won` | 21.555 | 177 |
| 16 | `oscar` | 21.381 | 56 |
| 17 | `club` | 21.132 | 135 |
| 18 | `movie` | 20.471 | 68 |
| 19 | `computers` | 19.168 | 77 |
| 20 | `singer` | 18.722 | 76 |

상위 selected token은 세 class의 핵심 의미축을 비교적 잘 반영한다. sport는 경기/팀/선수, tech는 소프트웨어/컴퓨터/인터넷, entertainment는 영화/배우/시상식 관련 token이 주로 선택된다.

## 7. TF-IDF matched baseline과의 비교

같은 BBC3 subset에서 TF-IDF top500 matched baseline도 확인했다. TF-IDF에서는 Eta-group EBIC + refit이 selected q=101까지 줄어 sparse support는 만들었지만, ARI=0.344로 clustering이 크게 무너졌다.

따라서 이 smoke에서는 단순히 sparse representation이면 충분한 것이 아니라, vMF mixture와 Eta-group objective에 맞는 representation 품질이 중요하다. SPLADE d=500은 TF-IDF보다 현재 Eta-group real-data diagnostic에 더 잘 맞는 후보로 보인다.

## 8. 해석상 주의점

- 이 결과는 최종 real-data validation이 아니다.
- EBIC/RICc는 diagnostic criterion이며, official tuning 변경으로 확정하지 않는다.
- SPLADE token은 learned lexical expansion token이다. 원문에 실제로 등장한 단어 목록으로 과장하면 안 된다.
- dense LLM embedding은 semantic geometry robustness check로는 쓸 수 있지만, coordinate-level selected support 해석에는 쓰지 않는다.
- 본문 결과로 쓰기 전에는 protocol 고정, top-d sensitivity, tuning sensitivity, stability check가 추가로 필요하다.

## 9. 결론

SPLADE sparse lexical representation은 Eta-group real-data 분석 후보로 충분히 가능성이 있다. BBC3 d=500 diagnostic에서 Eta-group EBIC/RICc는 Rossi보다 훨씬 작은 support를 선택하면서 clustering을 유지했고, selected token도 class별 의미와 잘 맞았다.

다만 현재 결과는 meeting 또는 appendix 수준의 diagnostic이다. 논문 본문 real-data 결과로 사용하려면 representation, top-d, tuning criterion, stability check를 고정한 후 재현 가능한 protocol로 다시 정리해야 한다.

## 10. 이 폴더의 파일

- `splade_d500_ebic_selected_tokens_overall.csv`: centered eta norm 기준 EBIC selected token ranking.
- `splade_d500_ebic_selected_tokens_by_class.csv`: class-mapped signed centered eta score.
- `splade_d500_ebic_cluster_class_table.csv`: cluster-majority class table.
- `splade_bbc3_realdata_conclusion_260624.md`: 이 diagnostic 결론 문서.
