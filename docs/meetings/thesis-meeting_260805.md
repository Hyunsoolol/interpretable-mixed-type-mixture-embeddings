# Real Data Analysis

## 1. 목적

$$\text{Dense vMF}\quad\longrightarrow\quad\text{posterior-score contrast sparsification}$$

$$S_\eta=\{j:\Vert{}H_K\eta_{\cdot j}\Vert{}_2>0\},\qquad S_\mu=\{j:\Vert{}H_K\mu_{\cdot j}\Vert{}_2>0\}$$

확인 사항:

$$\text{clustering 유지}+\vert{}S_\eta\vert{}\text{ 감소}+\text{common baseline 보존}$$

---

## 2. 데이터 및 전처리

### Classic3

$$K=3\quad(\mathrm{CISI},\mathrm{CRAN},\mathrm{MED})$$

$$n_0=3{,}890$$

$$n=3{,}890-2\;(\text{exact duplicates})-5\;(\text{near duplicates})=3{,}883$$

$$(n_{\mathrm{CISI}},n_{\mathrm{CRAN}},n_{\mathrm{MED}})=(1454,1397,1032)$$

### SPLADE representation

$$y_i=\mathrm{SPLADE}(t_i)\in\mathbb R_+^V$$

$$J=\mathrm{TopVar}_{2000}\left(Y_{\mathrm{clean}}\right)$$

$$x_i=\frac{y_{i,J}}{\lVert y_{i,J}\rVert_2}\in\mathbb S^{1999}$$

최종 자료:

$$X\in\mathbb R^{3883\times2000},\qquad\lVert x_i\rVert_2=1$$

검증:

$$\text{zero row}=0,\quad\text{duplicate transformed row}=0,\quad\text{nonfinite value}=0$$

Label 사용:

$$\text{fitting / tuning / support selection}\;\perp\;\text{class label}$$

$$\text{label}\rightarrow K=3,\;\mathrm{ARI},\;\mathrm{NMI},\;\text{post-fit component naming}$$

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

---

## 7. 보조 결과

Classic3 repeated splits:

$$\mathrm{Jaccard}\left(\widehat S_\eta^{(a)},\widehat S_\eta^{(b)}\right)=0.933\quad\text{(mean conditional)}$$

BBCSport:

$$\mathrm{NLL}_{\mathrm{sparse}}>\mathrm{NLL}_{\mathrm{dense}}$$

$$\Rightarrow\text{sparsification이 항상 density fit을 개선하지는 않음}$$

CSTR:

$$\text{Rossi-style implementation bridge}$$

---

## 8. 현재 해석

$$\boxed{\text{Classic3에서 E-CGL은}28.95\text{\%의 posterior-score contrast 좌표를 제거}}$$

$$\boxed{\mathrm{ARI},\mathrm{NMI}\text{는 dense vMF-F와 동일한 수준}}$$

$$\boxed{\text{선택 좌표는 component-specific lexical contrast를 제공}}$$

$$\boxed{\text{비선택 좌표의 common baseline은 fitted density에 유지}}$$
