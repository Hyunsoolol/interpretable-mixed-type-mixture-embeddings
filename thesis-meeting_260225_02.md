# 연구 미팅 보고: 연속형 혼합모형 HP 1차 시뮬레이션 결과

## 1. 실험 목적

텍스트/범주형을 붙이기 전에, **연속형 혼합모형에서 heterogeneity pursuit(HP)**가 다음 세 가지를 달성하는지 먼저 확인하는 디버깅/검증 실험입니다.

1. 군집을 가르는 차원(변수)을 얼마나 정확히 선택하는지
    
2. 그 선택이 군집 성능(ARI)에 실제로 도움이 되는지
    
3. 페널티 때문에 평균 추정이 수축되어 ARI가 떨어질 수 있는데, **post-selection refit(페널티 제거 재적합)**이 이를 복구하는지
    

---

## 2. 시뮬레이션 세팅 (Data Generating Process)

### 2.1 기본 설정

- **표본 크기:** $n=500$
    
- **군집 수:** $K=3$
    
- **차원:** $p=100$
    
- **이질(신호) 차원 수:** $q=5$ (즉, $S={1,\dots,5}$만 군집 간 평균 차이 존재)
    
- **신호 크기:** $a=1$
    
- **반복 횟수:** $R=10$
    

### 2.2 군집 라벨 생성

$$c_i \sim \mathrm{Cat}(\pi_1,\pi_2,\pi_3),\quad \pi_k=\frac{1}{3}$$

### 2.3 평균 구조 (희소 mean heterogeneity)

$$X_i \mid (c_i=k) \sim N(\mu_k,\Psi)$$

$$\mu_k=\mu_0+\delta_k,\quad \mu_0=\mathbf{0}$$

- $j\in S={1,\dots,q}$인 좌표에서만 $\delta_{kj}\neq 0$
    
- 그 외 $j>q$는 $\delta_{kj}=0$ (공통/배경 차원)
    
- 이번 실험은 단순한 난이도를 위해 군집별 평균 패턴을 대칭 형태로 구성(예: $-a,0,+a$ 형태)했고, 식별 제약 $\sum_{k}\pi_k\delta_k=\mathbf{0}$를 만족하도록 생성 및 정렬했습니다.
    

### 2.4 공분산 구조

- **생성:** $\Psi=I_p$ (공통 대각, 분산 1)
    
- 적합도 동일하게 공통 대각을 가정 (이번 1차 실험은 mean-heterogeneity만 확인하기 위함)
    

### 2.5 전처리

- **각 좌표별 표준화:** `x <- scale(X)`
    
- (페널티가 스케일에 민감한 문제를 줄이기 위한 기본 처리)
    

---

## 3. 적합 모델(제안 모형) 및 비교 모델

### 3.1 제안 모형: HP mixture

공통 대각 $\Psi$, mean-heterogeneity에 $\ell_1$ 페널티를 부여한 연속형-only mixture 모델입니다.

$$f(X_i)=\sum_{k=1}^K \pi_k \phi\big(X_i;\mu_0+\delta_k,\Psi\big),\quad \Psi=\mathrm{diag}(\sigma_1^2,\dots,\sigma_p^2) \text{ (공통)}$$

**벌점화 추정(Penalized Estimation):**

$$\max_{\Theta}\{\sum_{i=1}^n\log f(X_i\mid\Theta) - n\lambda \sum_{k=1}^K\sum_{j=1}^p|\delta_{kj}|\}$$


- **추정 (penalized EM):**
    
    - **E-step:** $\gamma_{ik}$ 계산
        
    - **M-step:** $\delta_{kj}$는 soft-thresholding으로 업데이트
        
    - $\sum_k \pi_k\delta_k=\mathbf{0}$ 제약은 매 반복마다 재정렬로 enforce
        
    - $\Psi$는 공통 대각 분산으로 업데이트 (floor 포함)
        

### 3.2 HP + refit (post-selection debiasing; 최종 군집 성능 개선 목적)

1. **HP로 선택 집합 도출:**
    
    $$\widehat{S}=\{j:\exists k,\ |\hat\delta_{kj}|>0\}$$
    
2. **$\widehat{S}$만 사용하여 $\lambda=0$인 unpenalized GMM 재적합(refit):**
    
    - 목적: 선택은 유지하면서 $\ell_1$ 수축 바이어스를 제거하여 군집을 더 날카롭게 만듦
        

### 3.3 비교 모델들 (benchmark)

- **GMM0(all dims):** $\lambda=0$ unpenalized GMM (공통 대각) on $p=100$
    
- **k-means(all dims):** $p=100$
    
- **Oracle GMM(true dims):** 진짜 신호 차원 $S={1,\dots,q}$만 알고 있다고 가정, $\lambda=0$ GMM
    
- **Oracle k-means(true dims):** $S$만으로 k-means
    
    _(Oracle 모델들은 이 세팅에서 달성 가능한 상한(reference) 보고용으로 포함)_
    

---

## 4. 튜닝/구현 세부 (미팅 시 예상 질문 포인트)

- **$\lambda$ 선택:** 그리드 탐색 후 BIC 최소 선택 (이번 실험에서 선택된 $\lambda$ 평균: 약 0.052)
    
- **초기값/로컬해 대응:**
    
    - HP: `nstart_hp=5`
        
    - refit 및 oracle GMM: `nstart_refit=30` (좀 더 강하게 탐색)
        
- **수치 안정화:** $\sigma^2$ floor, $\pi$ floor 적용 (퇴화 방지)
    
- **반복 수렴:** objective 상대 변화 기준 `tol` 사용
    

---

## 5. 평가 기준 (Evaluation metrics)

### 5.1 군집 성능

- **ARI (Adjusted Rand Index):** 추정 라벨 $\hat{c}_i=\arg\max_k \gamma_{ik}$ vs true $c_i$
    

### 5.2 선택 성능 (heterogeneity selection)

- **선택 집합:** $\widehat{S}=\{j:\exists k,\ |\hat\delta_{kj}|>0\}$
    
- **진짜 집합:** $S=\{1,\dots,q\}$
    
- **TPR (True Positive Rate):**
    
    $$\mathrm{TPR}=\frac{|S\cap \widehat{S}|}{|S|}$$
    
- **FPR (False Positive Rate):**
    
    $$\mathrm{FPR}=\frac{|\widehat{S}\setminus S|}{p-|S|}$$
    
- **Shat:** $|\widehat{S}|$



---

## 6. 결과 요약 (10회 반복 평균 결과)

### 6.1 평균 결과 요약표

| 방법 | 사용 차원 | 페널티 | 선택 여부 | ARI | TPR | FPR | $\vert \widehat{S} \vert$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| k-means | 100 | 없음 | 없음 | 0.296 | - | - | - |
| GMM0 | 100 | 없음 | 없음 | 0.370 | - | - | - |
| HP | 100 | 있음 | 있음 | 0.415 | 1.000 | 0.002 | 5.2 |
| HP+refit | $\widehat{S}$ | 없음(refit) | 있음 | 0.535 | (HP와 동일) | (HP와 동일) | 5.2 |
| Oracle k-means | 5 (진짜 $S$) | 없음 | (진짜) | 0.539 | - | - | - |
| Oracle GMM | 5 (진짜 $S$) | 없음 | (진짜) | 0.537 | - | - | - |


### 6.2 결과에서 가장 중요한 관찰 2가지

1. **HP는 "변수선택"을 거의 완벽하게 수행함**
    
    - $\mathrm{TPR}=1$, $\mathrm{FPR}$ 거의 0, $|\widehat{S}|\approx q$
        
    - $\rightarrow$ 군집을 가르는 차원을 찾아내는 HP 메커니즘이 제대로 동작한다는 것을 확인.
        
2. **HP 단독의 ARI가 낮았던 원인은 "선택 실패"가 아니라 "$\ell_1$ 수축 바이어스"**
    
    - HP는 $\delta$를 0 쪽으로 당기므로 군집 평균 차이가 줄어들어 분류가 덜 날카로워질 수 있음.
        
    - 선택된 $\widehat{S}$에서 $\lambda=0$으로 refit하면 수축이 사라져 ARI가 급상승함.
        
    - 실제로 $\text{ARI}_{\text{refit}} \approx \text{ARI}_{\text{oracle GMM}}$
        
    - $\rightarrow$ **"HP는 차원선택용, 최종 군집은 선택 후 refit이 최적"**이라는 결론이 자연스럽게 도출됨.


## 7. 특이사항
1. 연속형-only 환경에서 HP는 군집을 가르는 차원 선택을 매우 정확히 수행했다($\mathrm{TPR}=1$, $\mathrm{FPR}\approx 0$).
    
2. 고차원 잡음($p-q=95$) 때문에 일반 GMM/k-means의 군집 성능이 낮아지지만, HP는 선택을 통해 성능을 개선했다($\text{ARI}: \text{HP} > \text{GMM0} > \text{k-means}$).
    
3. $\ell_1$ 페널티로 인해 $\delta$가 수축되어 HP 단독 ARI가 제한되지만, **HP+refit(선택 후 페널티 제거 재적합)**을 사용하면 oracle 수준의 ARI까지 회복했다.
    
4. 따라서 이후 실험/논문에서는 **"HP(선택) + refit(최종 추정)"**을 표준 파이프라인으로 두는 게 합리적이다.
