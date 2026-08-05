# Real Data Analysis

## 1. 목적

$$\text{Dense vMF}\quad\longrightarrow\quad\text{posterior-score contrast sparsification}$$

$$S_\eta=\{j:\Vert{}H_K\eta_{\cdot j}\Vert{}_2>0\},\qquad S_\mu=\{j:\Vert{}H_K\mu_{\cdot j}\Vert{}_2>0\}$$

확인 사항:

$$\text{clustering 유지}+\vert{}S_\eta\vert{}\text{ 감소}+\text{common baseline 보존}$$

---

## 2. 데이터 및 전처리

### Classic3
- 사용 배경: 방향성 문서 군집화(directional document clustering) 알고리즘 검증을 위해 널리 활용되는 벤치마크 데이터셋입니다.
- 레퍼런스: Banerjee et al. (2005), Greene & Cunningham (2006).

$$K=3\quad(\mathrm{CISI},\mathrm{CRAN},\mathrm{MED})$$

$$n_0=3{,}890$$

$$n=3{,}890-2(\text{exact duplicates})-5(\text{near duplicates})=3{,}883$$

$$(n_{\mathrm{CISI}},n_{\mathrm{CRAN}},n_{\mathrm{MED}})=(1454,1397,1032)$$

### SPLADE representation

- 고정된 리비전의 SPLADE CoCondenser 모델(Formal et al., 2021)을 사용하여 문서를 어휘 기반(vocabulary-aligned) 벡터로 인코딩
- SPLADE는 문맥이 반영된 vocabulary 단위의 sparse표현을 제공하므로, 피처 수준의 의미 해석에 적합
- 사용 타당성: SPLADE의 출력값 $y_i$는 non-negative 특성을 가집니다. 이를 $L2$-정규화한 변환값 $x_i$는 단위 초구면($S^{p−1}$) 위에 위치하므로, vMF 분포 및 구면 k-평균(Spherical k-means) 모델의 입력 가정을 충족

$$y_i=\mathrm{SPLADE}(t_i)\in\mathbb R_+^V$$

$$J=\mathrm{TopVar}_{2000}\left(Y_{\mathrm{clean}}\right)$$

$$x_i=\frac{y_{i,J}}{\lVert y_{i,J}\rVert_2}\in\mathbb S^{1999}$$

최종 자료:

$$X\in\mathbb R^{3883\times2000},\qquad\lVert x_i\rVert_2=1$$

검증:

$$\text{zero row}=0,\quad\text{duplicate transformed row}=0,\quad\text{nonfinite value}=0$$

Label 사용:

$$\text{fitting / tuning / support selection}\perp\text{class label}$$

$$\text{label}\rightarrow K=3,\mathrm{ARI},\mathrm{NMI},\text{post-fit component naming}$$

---

## 3. 비교 방법

| Method | Target |
| --- | --- |
| Spherical $k$-means | cosine clustering |
| Sparse $k$-means | weighted coordinates |
| Dense vMF-S | $\kappa_1=\cdots=\kappa_K$ |
| Dense vMF-F | component-specific $\kappa_k$ |
| M-L | prototype-union support $S_P$ |
| M-CGL | directional support $S_\mu$ |
| E-CGL | posterior-score contrast support $S_\eta$ |
| E-ACGL | adaptive $S_\eta$ |

$$\eta_k=\kappa_k\mu_k$$

$$S_P\neq S_\mu\neq S_\eta$$

따라서 방법 간 $q$는 동일한 estimand가 아니며,

$$q_{\text{small}}\not\Rightarrow\text{uniform superiority}$$

---

## 4. 적합 및 선택

$$\text{dense starts}=30$$

$$\text{centered path candidates}=240$$

$$\text{M-L updates}\le600$$

Sparse vMF 계열:

$$\text{penalized path}\rightarrow\text{distinct supports}\rightarrow\text{target-preserving refit}\rightarrow\underset{S}{\arg\min}\,\mathrm{BIC}^{\mathrm{refit}}(S)$$

$$\mathrm{BIC}^{\mathrm{refit}}(S)=-2\ell\left(\widehat\Theta_S^{\mathrm{refit}}\right)+\log(n)\,\mathrm{df}_{\mathrm{nom}}(S)$$

평가:

$$\mathrm{ARI},\quad\mathrm{NMI},\quad\ell,\quad\mathrm{BIC},\quad q,\quad q/d$$

---

## 5. Full-data 결과

| Method | $q$ | $q/d$ | ARI | NMI | $\widehat\kappa_{\mathrm{CISI/CRAN/MED}}$ |
| --- | --- | --- | --- | --- | --- |
| Spherical $k$-means | 2000 | 1.000 | 0.970 | 0.944 | — |
| Sparse $k$-means | 985 | 0.492 | 0.137 | 0.313 | — |
| Dense vMF-S | 2000 | 1.000 | 0.970 | 0.944 | 732.2 / 732.2 / 732.2 |
| Dense vMF-F | 2000 | 1.000 | 0.976 | 0.954 | 738.5 / 818.0 / 615.3 |
| M-L | 1942 | 0.971 | 0.970 | 0.944 | 730.0 / 730.0 / 730.0 |
| M-CGL | 1462 | 0.731 | 0.976 | 0.954 | 738.8 / 816.4 / 609.9 |
| **E-CGL** | **1421** | **0.711** | **0.976** | **0.954** | **738.3 / 815.9 / 610.4** |
| E-ACGL | 2000 | 1.000 | 0.976 | 0.954 | 738.4 / 818.0 / 615.3 |

### E-CGL 좌표 감소

$$d-\vert{}S_\eta\vert{}=2000-1421=579$$

$$\text{reduction}=\frac{579}{2000}=28.95\text{\%}$$

### Dense vMF-F 대비

$$\Delta\mathrm{ARI}\approx0,\qquad\Delta\mathrm{NMI}\approx0$$

$$\vert{}S_\eta\vert{}=1421<\vert{}S_\mu\vert{}=1462<\vert{}S_P\vert{}=1942$$

### Adaptive extension

$$\vert{}S_{\eta,\mathrm{ACGL}}\vert{}=2000$$

$$\Rightarrow\text{E-ACGL은 Classic3에서 추가 sparsity를 생성하지 못함}$$

---

## 6. 좌표 해석

### Centered natural-parameter contrast

$$c_{kj}^{(\eta)}=\eta_{kj}-\bar\eta_j,\qquad\bar\eta_j=\frac1K\sum_{h=1}^K\eta_{hj}$$

$$c_{kj}^{(\eta)}>0\Rightarrow\text{component }k\text{의 상대 posterior score 증가}$$

$$c_{\cdot j}^{(\eta)}=0\Rightarrow x_j\text{가 모든 pairwise linear score에서 소거}$$

### 주요 positive contrast

$$\begin{aligned} \mathrm{CISI}:&\quad\texttt{library,\ information,\ librarian,\ libraries,\ retrieval}\\ \mathrm{CRAN}:&\quad\texttt{flow,\ mach,\ pressure,\ theory,\ heat}\\ \mathrm{MED}:&\quad\texttt{tumor,\ inhibitor,\ rat,\ dose,\ cancer} \end{aligned}$$

### Common background

$$j\notin S_\eta,\qquad\bar\eta_j\neq0,\qquad H_K\eta_{\cdot j}=0$$

예:

$$\texttt{comparison,\ quantitative,\ behavior,\ patterns,\ frequency}$$

핵심:

$$\boxed{\text{inactive}\neq\text{density에서 제거}}$$

$$\boxed{\text{inactive}=\text{component 간 posterior-score contrast가 없음}}$$

<img width="1058" height="888" alt="image" src="https://github.com/user-attachments/assets/3ab2afb8-8116-4b77-b3e2-fc622ec9cb8f" />

## 7. 다음 계획

**1. Theory**

- **Theorem 1 (Estimator 존재성):** Finite cap($\kappa_{\max}$) 파라미터 공간($\Theta_{\max}$)을 정의하여 penalized estimator의 존재성 및 likelihood collapse 방지 증명.
    
- **Theorem 2 (Proximal 수렴성):** Backtracking proximal-gradient sequence가 고정 responsibility 하에서 유일한 조건부 전역 최적해로 수렴함(수렴 속도 포함)을 명시.
    
- **Theorem 3 (GEM Stationarity):** Guarded GEM 알고리즘이 수렴하며, 모든 accumulation point가 KKT stationary point임을 증명 (Global optimum 등 과도한 주장은 배제).
    
- **Theorem 4 & Corollary (권장 통계 이론):** 실제 사용되는 refit 및 BIC 선택 절차에 맞춰 Candidate-path 조건부 BIC consistency 및 selected refit의 Oracle efficiency를 이론적으로 뒷받침.
    

**2. 투고 전 필수 원고 수정**

- **초록(Abstract) 작성
    
- **결과 수치 일괄 업데이트:** 단일 결과(CSV) 기반으로 Tables, Figures, Discussion 수치(예: Classic3 28.95% 등)를 자동 생성하여 하드코딩 오류 제거.
    
- **기호 및 알고리즘 정의 명확화:**
    
    - 행렬($E, M$) 및 Support Set($S_P, S_\mu, S_\eta$) 수식 초기 명시.
        
    - 알고리즘 수식을 본문(stabilized $\pi_k^+$ update)과 통일하고, backtracking 실패 조건 추가.
        
- **Path 및 파라미터 설정 통일:** Path 개수 $L=240$ 통일, 후보군에 Null support($\varnothing$) 반드시 포함, Estimator 정의에 finite-cap $\Theta_{\max}$ 반영.
    
**지표 및 프로토콜 보완:** 평가 지표를 전체 $\eta$가 아닌 centered contrast 기준인 $\mathrm{MSE}_{\eta^c}$ 로 통일. SPLADE++, Sparse $k$-means 인용 추가 및 실데이터 프로토콜(중복 문서 처리 기준 등) 구체화.

**3. Real Data**
- 데이터 사례 추가
