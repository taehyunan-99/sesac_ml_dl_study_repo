# 📚 Machine Learning & Deep Learning Study

머신러닝 및 딥러닝 학습을 위한 레포지토리입니다.

## 📂 폴더 구조
```
project/
├── code/              # 코드 및 노트북
│   ├── desc_stats.ipynb
│   ├── eda_used_cars.ipynb
│   ├── hypotest.ipynb
│   ├── linear_regression.ipynb
│   ├── plt_rce.py
│   └── statistics.ipynb
└── data/              # 데이터 파일
    ├── Used_Cars.pkl
    ├── Used_Cars_Prep.pkl
    └── (기타 데이터셋)
```

## 📂 학습 내용

### 📊 통계학 기초
- **정규분포**: 정규분포 무작위 값 생성, 확률밀도함수(PDF), 누적분포함수(CDF)
- **부트스트래핑**: 표본 통계량 추정 및 중심극한정리 검증
- **표준화**: Z-score 변환 및 이상치 탐지
- **정규성 검정**: Shapiro-Wilk, Anderson-Darling 검정

<br/>

### 🔍 가설검정
#### 상관분석
- **피어슨 상관분석**: scipy.stats, pingouin 라이브러리 활용
- **apply 메서드 활용**: 다변량 상관분석 자동화

<br/>

#### 독립표본 검정
- **기술통계량 확인**: 집단별 평균, 표준편차 비교
- **정규성 검정**: Shapiro-Wilk 검정, pingouin.normality
- **등분산성 검정**: Levene 검정
- **t-검정**: 독립표본 t-test (Welch's correction 포함)
- **맨-휘트니 U 검정**: 비모수 검정 (정규성 미충족 시)

<br/>

### 📈 탐색적 데이터 분석 (EDA)
- **Used Cars 데이터셋**: 중고차 가격 예측을 위한 데이터 탐색
- **기술통계량 분석**: 수치형 변수의 분포 특성 파악
- **시각화**: 이상치 탐지 및 시각화

<br/>

### 📉 선형 회귀 분석
#### 데이터 전처리
- **더미 변수 변환**: 범주형 변수의 원-핫 인코딩 (pd.get_dummies)
- **데이터 분리**: 입력변수(X)와 목표변수(y) 분리

<br/>

#### 회귀 모형 적합
- **OLS 회귀**: statsmodels를 활용한 최소제곱법 회귀분석
- **회귀계수 해석**: 각 변수의 유의성 및 영향력 분석
- **결정계수(R²)**: 모형 설명력 평가

<br/>

#### 회귀 진단
- **잔차 분석**: 잔차 그래프를 통한 회귀 가정 검증
- **정규성 검정**: Shapiro-Wilk 검정으로 잔차의 정규성 확인
- **등분산성 검정**: Breusch-Pagan 검정

<br/>

---
*마지막 업데이트: 2026-02-05*
