# 📚 Machine Learning & Deep Learning Study

머신러닝 및 딥러닝 학습을 위한 레포지토리입니다.

## 📂 폴더 구조
```
project/
├── code/
│   ├── stat/          # 통계학 기초 및 회귀분석
│   ├── ml/            # 머신러닝 모델링
│   ├── ml_part2/      # 머신러닝 심화 (이상탐지, 예측 등)
│   └── dl/            # 딥러닝 기초 및 MNIST
├── data/              # 데이터 파일
kaggle_data/
├── code/              # Kaggle 대회 1 (분류)
├── data/
kaggle_data_2/
├── code/              # Kaggle 대회 2 (예측 정비)
├── data/
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
- **Encar 데이터 전처리**: 차량 색상 표준화, 이상치 제거, 피처 엔지니어링

<br/>

### 📉 회귀 분석
#### 통계 기반 회귀
- **OLS 회귀**: statsmodels를 활용한 최소제곱법 회귀분석
- **회귀계수 해석**: 각 변수의 유의성 및 영향력 분석
- **회귀 진단**: 잔차 분석, Shapiro-Wilk 정규성 검정, Breusch-Pagan 등분산성 검정

<br/>

#### 로지스틱 회귀 (통계)
- **이항 로지스틱 회귀**: GRE/GPA 입학 데이터 기반 모형
- **통계 검정**: t-검정, 카이제곱 검정, 다중공선성 확인

<br/>

### 🤖 머신러닝
#### 데이터 전처리
- **스케일링**: StandardScaler, MinMaxScaler
- **인코딩**: 원-핫 인코딩, 라벨 인코딩, 순서형 인코딩, 타겟 인코딩
- **결측치 처리**: 반복 대입법(Iterative Imputation)
- **거리 측정**: 유클리드, 맨해튼, 체비셰프 거리 및 스케일링 효과

<br/>

#### 지도학습 - 회귀
- **선형 회귀**: Ridge, Lasso 정규화 및 피처 중요도 비교
- **KNN 회귀**: 거리 가중치 및 최적 이웃 수 튜닝
- **의사결정나무 회귀**: 비용-복잡도 가지치기
- **랜덤 포레스트 회귀**: OOB 스코어 추적 및 피처 중요도
- **그래디언트 부스팅 회귀**: n_estimators 튜닝
- **XGBoost 회귀**: 조기 종료 활용

<br/>

#### 지도학습 - 분류
- **KNN 분류**: 거리 가중치, 최적 k 탐색, F1 최적화, SMOTE 적용
- **로지스틱 회귀**: Ridge/Lasso 변형, ROC/PR 곡선 분석
- **의사결정나무 분류**: 비용-복잡도 가지치기, 클래스 불균형 처리 (cutoff 조정, SMOTE, class_weight)
- **랜덤 포레스트 분류**: OOB 스코어, 리샘플링
- **그래디언트 부스팅 분류**: 조기 종료, SMOTE 적용
- **XGBoost 분류**: 조기 종료, scale_pos_weight

<br/>

#### 비지도학습
- **K-Means 클러스터링**: WCSS, 실루엣 분석
- **계층적 클러스터링**: 다양한 연결 방법
- **DBSCAN**: 밀도 기반 클러스터링
- **PCA**: 주성분 분석, 분산 설명력 시각화
- **t-SNE**: 차원 축소 시각화

<br/>

#### 모델 튜닝 및 해석
- **하이퍼파라미터 최적화**: Grid Search, Random Search, Bayesian Optimization
- **모델 해석**: Feature Importance, Permutation Importance, PDP, ICE, LIME, SHAP

<br/>

### 🔬 머신러닝 심화
- **은행 고객 이탈 예측**: 불균형 분류, 랜덤 포레스트, PDP/ICE, LIME, SHAP 해석
- **이상 탐지**: 이상치 탐지 기법
- **설비 센서 분석**: 센서 데이터 기반 분석
- **예측 정비**: 설비 고장 예측 모델링

<br/>

### 🧠 딥러닝
#### 수학적 기초
- **함수 합성 및 미분**: 편미분, 내적, 연쇄법칙
- **활성화 함수**: ReLU, Softmax
- **손실 함수**: 교차 엔트로피

<br/>

#### PyTorch 기초
- **텐서 연산**: 0D~2D 텐서 생성, 인덱싱, 슬라이싱, 브로드캐스팅
- **차원 조작**: unsqueeze, squeeze, stack, permute, transpose
- **이미지 배치 처리**: 배치 차원 다루기

<br/>

#### MNIST 분류
- **데이터 전처리**: MNIST 데이터셋 준비 및 전처리
- **단층 퍼셉트론 (SLP)**: 단일 레이어 모델
- **다층 퍼셉트론 (MLP)**: 다중 은닉층 모델
- **Trainer 유틸리티**: 학습 루프, 손실/정확도 추적, 혼동 행렬, 오분류 시각화

<br/>

### 🏆 Kaggle 대회
- **분류 대회**: Kaggle 분류 문제 풀이
- **예측 정비 대회**: 5-모델 앙상블 (LightGBM, XGBoost, RandomForest, CatBoost, ExtraTrees), 50 시드 × 5-Fold CV, F1 98.65%

<br/>

---
*마지막 업데이트: 2026-03-16*
