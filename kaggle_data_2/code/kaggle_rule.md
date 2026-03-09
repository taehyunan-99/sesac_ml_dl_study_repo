# 캐글 데이터 분석 규칙

## 필수 사항
- 캐글 프라이빗 점수 향상
- 목표로 하는 모델은 일반화 성능을 최고점으로 목표로 함
    - F1 Score, Recall, ROC/AUC 등의 지표를 기준
    - 시드를 바꿔가면서 일반화 성능 테스트
    - Stratified 분할 등 불균형 및 타겟 데이터가 적은 데이터의 일반화 및 성능을 확인할 수 있는 지표 제작
- 데이터 누수 절대 방지

## 참고 사항
- kaggle_data/code/train_magic_v36_commented.py
    - 지난 캐글 대회 최고점 모델
- project/code/ml_part2 모든 파일
    - 수업시간 강사님이 알려주신 방법들
    - predictive_maintenance.ipynb 에서는 동일한 데이터로 실습

---

## 진행 기록

### 현재 상황 (2026-03-09)
- 캐글 평가 지표: **ROC-AUC** (프라이빗/퍼블릭 점수 비슷할 것으로 예상)
- 데이터: 제조 장비 예지보전, Target 불균형 (339/10000 = 3.39%)
- 강사님이 프라이빗 데이터를 랜덤 추출했다고 함

### 문제 진단
- 기본 LGBM 모델 CV AUC **0.990** vs 캐글 AUC **0.94** → 약 0.05 과적합 
- 원인: LGBM 기본 파라미터가 소규모 불균형 데이터에서 과적합
- ROC-AUC가 불균형 데이터에서 과대평가되는 경향 확인 (PR-AUC 0.906으로 더 보수적)

### 일반화 평가 체계 구축 (완료)
- `robust_cv_evaluate()` 함수: 10시드 × 5폴드 = 50회 평가
- 지표: AUC, F1, Recall, Precision, PR-AUC + 시드간 편차 + 자동 경고
- **PR-AUC를 실질적 성능 판단 기준으로 사용** (캐글 점수와 괴리 적음)

### 기본 LGBM 결과 (베이스라인)
| 지표 | 평균 | 표준편차 |
|------|------|---------|
| AUC | 0.98967 | 0.00213 |
| F1 | 0.83821 | 0.00781 |
| Recall | 0.83121 | 0.01011 |
| Precision | 0.84748 | 0.01387 |
| PR-AUC | 0.90619 | 0.00543 |

### 과적합 억제 LGBM v1
- num_leaves=15, max_depth=5, min_child_samples=30
- reg_alpha=0.1, reg_lambda=1.0
- subsample=0.8, colsample_bytree=0.8, n_estimators=200

| 지표 | 평균 | 표준편차 |
|------|------|---------|
| AUC | 0.99154 | 0.00090 |
| F1 | 0.81030 | 0.00827 |
| Recall | 0.86455 | 0.00783 |
| Precision | 0.76469 | 0.01200 |
| PR-AUC | 0.90666 | 0.00488 |

- **AUC 오히려 소폭 상승 (0.990→0.992), PR-AUC 유지 (0.906)**
- F1/Precision 하락, Recall 상승 → class_weight='balanced'의 영향
- AUC 표준편차 0.0009로 매우 안정적 (기존 0.0021)
- **CV AUC 0.992 vs 캐글 0.94 → 갭이 여전히 존재**

---

## 개선 플랜 (TODO)

### 캐글 제출 기록
| 버전 | 설명 | 캐글 AUC | 비고 |
|------|------|---------|------|
| v1 | 기본 LGBM (default 파라미터) | 0.940 | 베이스라인 |
| v2 | 정규화 LGBM + 50시드 Rank Avg | 0.950 | 앙상블 효과 |
| v3 | v2 + 파생변수 6개 | 0.953 | 피처 엔지니어링 |
| v4 | v3 + Failure Type 피처 추가 | **0.994** | 핵심 피처 발견 |

### 핵심 발견
- **Failure Type을 피처로 사용하는 것이 결정적** (0.953→0.994)
- test에 공식 제공된 컬럼이므로 대회에서 사용 가능
- 피처 제거는 오히려 성능 하락 (앙상블 다양성 감소)
- 순열 중요도 음수 피처도 앙상블에서는 기여할 수 있음

### 현재 적용 설정
- 오입력 처리: 케이스B (Target=1 + No Failure → 0)
- 피처: 원본 6개 + Failure Type + 파생변수 6개 = 13개
- 모델: LGBM + XGBoost + RF 3모델 앙상블
- 앙상블: 50시드 × 5폴드 Rank Averaging
- 정규화: num_leaves=15, max_depth=5, min_child_samples=30, reg 0.1/1.0, sub/col 0.8

### 남은 TODO
- [ ] Optuna 하이퍼파라미터 튜닝 (0.994 → 0.995+ 목표)
- [ ] Failure Type 인코딩 방식 비교 (Ordinal vs One-Hot)
- [ ] 피처 분석 재실행 (Failure Type 포함 상태에서)
