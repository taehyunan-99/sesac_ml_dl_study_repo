# 통계 분석 템플릿

## 전체 흐름
```
1. 데이터 척도 파악 → 2. X-Y 척도 조합 확인 → 3. 분석 방법 선택 → 4. 전제조건 검정 → 5. 분석 실행 → 6. 결과 해석
```

---

## 1단계: 데이터 척도 분류

| 척도 | 설명 | 예시 | 판별법 |
|:----:|------|------|--------|
| **명목(Nominal)** | 순서 없는 범주 | 성별, 혈액형, 지역 | 순서 의미 없음 |
| **서열(Ordinal)** | 순서 있는 범주 | 학점(A/B/C), 만족도(상/중/하) | 순서 있으나 간격 불균등 |
| **등간(Interval)** | 균등 간격, 절대0 없음 | 온도(℃), 연도, IQ | 0이 "없음" 아님 |
| **비율(Ratio)** | 균등 간격, 절대0 있음 | 나이, 소득, 키, 무게 | 0 = 실제로 없음 |

### 실무 단순화
- **범주형** = 명목 + 서열
- **연속형** = 등간 + 비율

---

## 2단계: X-Y 척도 조합별 분석 방법

| X (독립변수) | Y (종속변수) | 분석 방법 | 함수 |
|:------------:|:------------:|-----------|------|
| 범주형 | 범주형 | **카이제곱 검정** | `pg.chi2_independence()` |
| 범주형 (2그룹) | 연속형 | **독립표본 t-test** | `pg.ttest()` |
| 범주형 (3+그룹) | 연속형 | **ANOVA** | `pg.anova()` |
| 연속형 | 범주형 | **로지스틱 회귀** | `sm.Logit()` |
| 연속형 | 연속형 | **상관분석 / 선형회귀** | `pg.corr()` / `sm.OLS()` |

---

## 3단계: 각 분석별 상세 가이드

---
### 3.1 카이제곱 검정 (χ² Test)

**사용 조건**: 범주형 vs 범주형

#### 가설
| | 내용 |
|:--:|------|
| **H₀ (귀무가설)** | 두 변수는 **독립**이다 (연관 없음) |
| **H₁ (대립가설)** | 두 변수는 **연관**이 있다 |

#### 핵심 코드
```python
# 교차표 생성
pd.crosstab(df['X'], df['Y'], margins=True)

# 카이제곱 검정
pg.chi2_independence(data=df, x='X', y='Y', correction=True)
```

#### 결과 해석
- **p < 0.05** → H₀ 기각 → 두 변수 간 **유의한 연관성 있음**
- **p ≥ 0.05** → H₀ 채택 → 두 변수는 **독립적**

#### 효과크기: Cramer's V
| V 값 | 해석 |
|:----:|------|
| < 0.1 | 거의 없음 |
| 0.1 ~ 0.3 | 약함 |
| 0.3 ~ 0.5 | 중간 |
| > 0.5 | 강함 |

#### 시각화
```python
# 히트맵
sns.heatmap(pd.crosstab(df['X'], df['Y']), annot=True, fmt='d', cmap='Blues')

# 막대 그래프
sns.countplot(data=df, x='X', hue='Y')
```

---
### 3.2 독립표본 t-검정 (Independent t-test)

**사용 조건**: 범주형(2그룹) vs 연속형

#### 가설
| | 내용 |
|:--:|------|
| **H₀ (귀무가설)** | 두 그룹의 평균은 **같다** (μ₁ = μ₂) |
| **H₁ (대립가설)** | 두 그룹의 평균은 **다르다** (μ₁ ≠ μ₂) |

#### 전제조건
1. **정규성**: 각 그룹이 정규분포를 따름
2. **등분산성**: 두 그룹의 분산이 동일

#### 핵심 코드
```python
# 기술통계
df.groupby('Group')['Value'].agg(['count', 'mean', 'std'])

# 정규성 검정
pg.normality(data=df, dv='Value', group='Group')

# 등분산성 검정
pg.homoscedasticity(data=df, dv='Value', group='Group')

# t-검정 (등분산 불충족 시 correction=True)
pg.ttest(x=group1, y=group2, correction=True)
```

#### 결과 해석
- **p < 0.05** → H₀ 기각 → 두 그룹 간 **유의한 차이 있음**
- **p ≥ 0.05** → H₀ 채택 → 두 그룹 간 **차이 없음**

#### 효과크기: Cohen's d
| d 값 | 해석 |
|:----:|------|
| < 0.2 | 작음 |
| 0.2 ~ 0.5 | 작음~중간 |
| 0.5 ~ 0.8 | 중간~큼 |
| > 0.8 | 큼 |

#### 비모수 대안 (정규성 불충족 시)
```python
# Mann-Whitney U 검정
pg.mwu(x=group1, y=group2)
```

#### 시각화
```python
# 박스플롯
sns.boxplot(data=df, x='Group', y='Value')

# 바이올린 플롯
sns.violinplot(data=df, x='Group', y='Value')
```

---
### 3.3 대응표본 t-검정 (Paired t-test)

**사용 조건**: 동일 대상의 전/후 비교

#### 가설
| | 내용 |
|:--:|------|
| **H₀ (귀무가설)** | 전후 평균 차이는 **0이다** (μ_d = 0) |
| **H₁ (대립가설)** | 전후 평균 차이는 **0이 아니다** (μ_d ≠ 0) |

#### 핵심 코드
```python
# 정규성 검정 (차이값)
pg.normality(df['after'] - df['before'])

# 대응표본 t-검정
pg.ttest(x=df['before'], y=df['after'], paired=True)
```

#### 비모수 대안 (정규성 불충족 시)
```python
# Wilcoxon signed-rank 검정
stats.wilcoxon(df['before'], df['after'])
```

#### 시각화
```python
# 전후 비교 KDE
sns.kdeplot(df['before'], label='Before')
sns.kdeplot(df['after'], label='After')
plt.legend()
```

---
### 3.4 일원분산분석 (One-way ANOVA)

**사용 조건**: 범주형(3+그룹) vs 연속형

#### 가설
| | 내용 |
|:--:|------|
| **H₀ (귀무가설)** | 모든 그룹의 평균은 **같다** (μ₁ = μ₂ = μ₃ = ...) |
| **H₁ (대립가설)** | 최소 하나의 그룹 평균이 **다르다** |

#### 전제조건
1. **정규성**: 각 그룹이 정규분포를 따름
2. **등분산성**: 모든 그룹의 분산이 동일

#### 핵심 코드
```python
# 기술통계
df.groupby('Group')['Value'].agg(['count', 'mean', 'std'])

# 정규성 검정
pg.normality(data=df, dv='Value', group='Group')

# 등분산성 검정
pg.homoscedasticity(data=df, dv='Value', group='Group')

# ANOVA (등분산 충족)
pg.anova(data=df, dv='Value', between='Group')

# Welch's ANOVA (등분산 불충족)
pg.welch_anova(data=df, dv='Value', between='Group')
```

#### 사후검정 (Post-hoc)
ANOVA에서 유의하면 → 어떤 그룹이 다른지 확인

```python
import scikit_posthocs as sp

# Tukey HSD (등분산 충족)
sp.posthoc_tukey(a=df, val_col='Value', group_col='Group')

# Tamhane T2 (등분산 불충족)
sp.posthoc_tamhane(a=df, val_col='Value', group_col='Group')
```

#### 효과크기: Eta-squared (η²)
| η² 값 | 해석 |
|:-----:|------|
| < 0.01 | 매우 작음 |
| 0.01 ~ 0.06 | 작음 |
| 0.06 ~ 0.14 | 중간 |
| > 0.14 | 큼 |

#### 비모수 대안 (정규성 불충족 시)
```python
# Kruskal-Wallis 검정
pg.kruskal(data=df, dv='Value', between='Group')

# 사후검정: Nemenyi
sp.posthoc_nemenyi(a=df, val_col='Value', group_col='Group')
```

#### 시각화
```python
# 박스플롯
sns.boxplot(data=df, x='Group', y='Value')

# KDE (그룹별 분포)
sns.kdeplot(data=df, x='Value', hue='Group', fill=True)
```

---
### 3.5 상관분석 (Correlation)

**사용 조건**: 연속형 vs 연속형

#### 가설
| | 내용 |
|:--:|------|
| **H₀ (귀무가설)** | 두 변수는 **상관이 없다** (ρ = 0) |
| **H₁ (대립가설)** | 두 변수는 **상관이 있다** (ρ ≠ 0) |

#### 상관계수 종류
| 방법 | 사용 조건 |
|------|----------|
| **Pearson** | 정규분포, 선형 관계 |
| **Spearman** | 비정규, 단조 관계, 서열 변수 |

#### 핵심 코드
```python
# Pearson 상관분석
pg.corr(x=df['X'], y=df['Y'], method='pearson')

# Spearman 상관분석 (비모수)
pg.corr(x=df['X'], y=df['Y'], method='spearman')

# 상관행렬
df.corr(method='pearson')
```

#### 효과크기: 상관계수 (r) 자체
| |r| 값 | 해석 |
|:------:|------|
| < 0.1 | 거의 없음 |
| 0.1 ~ 0.3 | 약함 |
| 0.3 ~ 0.5 | 중간 |
| 0.5 ~ 0.7 | 강함 |
| > 0.7 | 매우 강함 |

#### 결과 해석
- **r > 0**: 양의 상관 (X↑ → Y↑)
- **r < 0**: 음의 상관 (X↑ → Y↓)
- **R² (결정계수)**: X가 Y 변동의 몇 %를 설명하는가

#### 시각화
```python
# 산점도 + 회귀선
sns.regplot(data=df, x='X', y='Y')

# 상관행렬 히트맵
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)

# 페어플롯
sns.pairplot(df)
```

---
### 3.6 선형회귀 (Linear Regression)

**사용 조건**: 연속형 X → 연속형 Y 예측

#### 가설 (각 계수에 대해)
| | 내용 |
|:--:|------|
| **H₀** | 회귀계수는 **0이다** (β = 0, 영향 없음) |
| **H₁** | 회귀계수는 **0이 아니다** (β ≠ 0, 영향 있음) |

#### 핵심 코드
```python
import statsmodels.api as sm

# 독립변수에 상수항 추가
X = sm.add_constant(df[['X1', 'X2']])
y = df['Y']

# 회귀모델 적합
model = sm.OLS(y, X).fit()
print(model.summary())
```

#### 결과 해석
| 지표 | 의미 |
|------|------|
| **R²** | 모델이 설명하는 분산 비율 (0~1) |
| **Adj. R²** | 변수 수 보정된 R² |
| **계수 (coef)** | X가 1 증가할 때 Y 변화량 |
| **p-value** | 계수의 유의성 (< 0.05면 유의) |

#### 다중공선성 검사 (VIF)
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
# VIF < 5: 양호 / 5~10: 주의 / > 10: 문제
```

#### 시각화
```python
# 잔차 플롯
sns.residplot(x=model.fittedvalues, y=model.resid)

# 실제 vs 예측
plt.scatter(y, model.fittedvalues)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
```

---
### 3.7 로지스틱 회귀 (Logistic Regression)

**사용 조건**: 연속형 X → 범주형(이항) Y 예측

#### 가설 (각 계수에 대해)
| | 내용 |
|:--:|------|
| **H₀** | 회귀계수는 **0이다** (OR = 1, 영향 없음) |
| **H₁** | 회귀계수는 **0이 아니다** (OR ≠ 1, 영향 있음) |

#### 핵심 코드
```python
import statsmodels.api as sm

X = sm.add_constant(df[['X1', 'X2']])
y = df['Y']  # 0 또는 1

model = sm.Logit(y, X).fit()
print(model.summary())

# Odds Ratio
np.exp(model.params)
```

#### 결과 해석: Odds Ratio (OR)
| OR 값 | 의미 |
|:-----:|------|
| OR = 1 | 영향 없음 |
| OR > 1 | X↑ → Y=1 확률↑ |
| OR < 1 | X↑ → Y=1 확률↓ |

#### 시각화
```python
# 로지스틱 곡선
sns.regplot(data=df, x='X', y='Y', logistic=True)
```

---
## 4단계: 전제조건 검정 요약

| 검정 | 목적 | 코드 | 귀무가설 |
|------|------|------|----------|
| **정규성** | 데이터가 정규분포 따르는가 | `stats.shapiro(x)` | H₀: 정규분포 따름 |
| **등분산성** | 그룹 간 분산이 같은가 | `pg.homoscedasticity()` | H₀: 분산이 같음 |
| **다중공선성** | 독립변수 간 상관 높은가 | VIF 계산 | VIF < 5 권장 |

### 정규성 검정
```python
# Shapiro-Wilk (n ≤ 5000)
stats.shapiro(data)
# p > 0.05 → 정규분포 따름

# 왜도, 첨도 확인
stats.skew(data)     # 0에 가까우면 대칭
stats.kurtosis(data) # 0에 가까우면 정규
```

### 등분산성 검정
```python
pg.homoscedasticity(data=df, dv='Value', group='Group')
# p > 0.05 → 등분산 충족
```

---

## 5단계: 비모수 검정 대안

**정규성 불충족 시** 아래 대안 사용:

| 모수 검정 | 비모수 대안 | 코드 |
|-----------|-------------|------|
| 독립표본 t-test | **Mann-Whitney U** | `pg.mwu(x, y)` |
| 대응표본 t-test | **Wilcoxon signed-rank** | `stats.wilcoxon(x, y)` |
| One-way ANOVA | **Kruskal-Wallis** | `pg.kruskal()` |
| Pearson 상관 | **Spearman 상관** | `pg.corr(method='spearman')` |

---

## 6단계: 효과크기 요약

**p-value**는 표본 크기에 영향받음 → **효과크기**로 실질적 의미 판단

| 분석 | 효과크기 | 작음 | 중간 | 큼 |
|------|----------|:----:|:----:|:----:|
| t-test | Cohen's d | 0.2 | 0.5 | 0.8 |
| ANOVA | η² (Eta-squared) | 0.01 | 0.06 | 0.14 |
| 카이제곱 | Cramer's V | 0.1 | 0.3 | 0.5 |
| 상관분석 | r (상관계수) | 0.1 | 0.3 | 0.5 |

---

## 7단계: 시각화 가이드

| 분석 유형 | 추천 시각화 | 코드 |
|-----------|-------------|------|
| 범주 vs 범주 | 히트맵, 막대그래프 | `sns.heatmap()`, `sns.countplot()` |
| 범주 vs 연속 | 박스플롯, 바이올린 | `sns.boxplot()`, `sns.violinplot()` |
| 연속 vs 연속 | 산점도, 회귀선 | `sns.regplot()`, `sns.scatterplot()` |
| 분포 확인 | 히스토그램, Q-Q plot | `sns.histplot()`, `stats.probplot()` |

### 분포 확인
```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 히스토그램
sns.histplot(data, kde=True, ax=axes[0])

# Q-Q plot
stats.probplot(data, dist='norm', plot=axes[1])
```

---

## 빠른 참조: 분석 선택 플로우차트

```
시작: X와 Y의 척도 확인
│
├─ X: 범주형, Y: 범주형
│   └─→ 카이제곱 검정
│
├─ X: 범주형, Y: 연속형
│   ├─ 2그룹 → t-test (정규성 불충족 시 Mann-Whitney)
│   └─ 3+그룹 → ANOVA (정규성 불충족 시 Kruskal-Wallis)
│
├─ X: 연속형, Y: 범주형
│   └─→ 로지스틱 회귀
│
└─ X: 연속형, Y: 연속형
    ├─→ 상관분석 (관계 파악)
    └─→ 선형회귀 (예측)
```

---

## 라이브러리 Import

```python
import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg
import scikit_posthocs as sp
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
```
