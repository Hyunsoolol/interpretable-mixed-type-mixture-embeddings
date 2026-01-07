# 해석 가능한 혼합형 데이터 혼합 모형 (IM3: Interpretable Mixed-type Mixture Modeling)

![Repo Name](https://img.shields.io/badge/Repo-interpretable--mixed--mixture-blueviolet)
![Status](https://img.shields.io/badge/Status-Research%20Proposal-blue)
![Python](https://img.shields.io/badge/Python-3.9%2B-green)
![Topic](https://img.shields.io/badge/Topic-Mixture%20Models%20%7C%20NLP%20%7C%20XAI-orange)

## 📌 개요 (Overview)

본 프로젝트는 범주형 변수(예: 인구통계학적 정보)와 고차원 텍스트 데이터로 구성된 **혼합형 데이터(Mixed-type data)**를 군집화하기 위한 새로운 통계적 프레임워크를 제안합니다.

기존 연구들이 단어 가방 모형(Bag-of-Words, BoW)이나 강력한 독립성 가정(나이브 베이즈)에 의존했던 것과 달리, 본 연구는 **거대 언어 모델(LLM) 임베딩**을 활용하여 텍스트의 의미론적 문맥(Semantic Context)을 포착합니다. 또한, 임베딩의 "블랙박스" 특성과 "차원의 저주" 문제를 해결하기 위해, 모델 해석을 위한 **디베딩(De-embedding)** 전략과 안정적인 가우시안 혼합 모형(GMM) 추정을 위한 차원 축소 단계를 도입했습니다.

## 📖 연구 배경 (Motivation)

### 기존 방법론의 한계
[Shi et al. (2024)](https://doi.org/10.1214/24-AOAS1893)와 같은 최근 연구에서는 사법적 판결에서 법 외적 요인(Extralegal factors)을 추정하기 위해 **조건부 혼합 회귀(MCR)**를 사용했습니다. 그러나 이러한 방법들은 텍스트 데이터 $Z$를 **나이브 베이즈 가정** 하에 이진(Binary) 벡터로 모델링한다는 한계가 있습니다.

$$
P(Z_i | K_i = k) = \prod_{j=1}^{p} P(Z_{ij} | K_i = k)
$$

**비판점 (Critique):**
1.  **의미 손실 (Loss of Semantics):** 이진 지표(Indicator)는 단어의 순서, 문맥, 그리고 미세한 의미론적 뉘앙스를 무시합니다.
2.  **비현실적 가정 (Unrealistic Assumption):** 자연어에서 단어 간의 독립성 가정은 현실적으로 위배되는 경우가 많습니다.

### 제안하는 접근법 (Our Approach)
우리는 이진 특성 벡터를 LLM(예: SBERT, OpenAI)이 생성한 **밀집 임베딩(Dense Embeddings)**으로 대체할 것을 제안합니다. 이를 통해 문제는 **혼합형 혼합 모형(범주형 + 연속형)**으로 변환됩니다. 해석 가능성과 계산 효율성을 보장하기 위해, 차원 축소 기법과 사후적 "디베딩(De-embedding)" 분석을 통합했습니다.

## 🛠️ 방법론 (Methodology)

제안된 프레임워크는 크게 세 가지 단계로 구성됩니다:

```mermaid
graph LR
    A[원본 데이터] --> B{전처리};
    B -->|"범주형 X_cat"| C[범주형 변수 처리];
    B -->|"텍스트 Z"| D[LLM 임베딩];
    D -->|"고차원 벡터"| E["차원 축소 (PCA/UMAP)"];
    E -->|"축소된 벡터 X_cont"| F[결합 혼합 모형링];
    C --> F;
    F --> G[잠재 클래스 식별];
    G --> H["디베딩을 통한 해석"];
