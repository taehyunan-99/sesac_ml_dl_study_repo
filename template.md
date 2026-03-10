# 머신러닝 통계 이론 정리

## 참고 경로 및 파일
- 폴더내 코드 경로
    - project/code/ml
    - project/code/ml_part2
- 노션 정리 데이터
    - 노션 -> Study Book -> 새싹 Course -> 05. 머신러닝/딥러닝 -> 01~08(주로 07~08을 참조)
    - 노션 -> Study Book -> 새싹 Course -> 07. 머신러닝/딥러닝 2 -> 01, 02 참조
- 정리 구조 파악
    - 노션 -> Study Book -> Practice -> Template -> 데이터 통계 분석 워크플로우, 데이터 전처리 템플릿, 기초 통계 이론 참조

## **필수 사항**
해당 필수 사항들은 무조건 지키기
- 코드 및 내용 작성은 "참고 경로 및 파일"의 "폴더내 코드 경로", "노션 정리 데이터"에서 다루고 있는 내용들만 사용
    - 사용자 요청시에는 임의로 추가 가능
- 정리 구조 및 스타일은 "참고 경로 및 파일"의 "정리 구조 파악" 파일들을 참고하여 최대한 해당 파일들에 근접한 구조와 스타일을 사용
- 모호하거나 확실하지 않은 내용이 있다면 먼저 사용자에게 확인 받은 뒤 진행
- 내용을 정리하면서 실제 개념과 다르게 정리되어 있는 내용이 있다면 사용자에게 보고 후 잘못된 내용 수정

## 내용
머신러닝에서 다루고 있는 통계 이론들을 노션 페이지에 정리를 할거야
이번 페이지에서는 코드관련 내용들은 제외하고 머신러닝에서 다루는 통계의 이론에만 집중하여 정리할거야 예를 들어 중성분 분석 등 처럼

## 요청사항
바로 내용을 작성하지말고 먼저 사용자와 방향성에 대한 토의를 한 뒤,
방향성에 맞게 계획을 구축하고 계획 단계별로 실행을 하면서 단계별로 사용자의 피드백을 듣고 수정 진행

---

## 진행 상황 (2026-03-10)

### 타겟 노션 페이지
- ID: `31dfbc6b-f206-809d-bb04-f73286dd0d86`
- 제목: 📈 머신러닝 통계 이론
- 스타일 참조: 📈 기초 통계 이론 페이지 (gray_bg 대제목, table header-row, callout 💡 등)

### 전체 계획 (총 9개 섹션, 4단계)
| 단계 | 내용 | 상태 |
|------|------|------|
| Step 1 | 전체 목차 + 섹션 1~3 (거리 메트릭, 회귀 분석, 분류 평가 지표) | ✅ 완료 |
| Step 2 | 섹션 4~6 (PCA, 클러스터링, 정규화) | ✅ 완료 |
| Step 3 | 섹션 7~9 (이상치 탐지, 모델 해석, 하이퍼파라미터 최적화) | ✅ 완료 |
| Step 4 | 최종 검수 및 수정 | ✅ 완료 |

### 소스 노트북 (리서치 완료)
- `project/code/ml/distance.ipynb` → 섹션 1
- `project/code/ml/linear_regression_ml.ipynb` → 섹션 2, 6
- `project/code/ml/logistic_regression_ml.ipynb` → 섹션 3, 6
- `project/code/ml/knn_clf.ipynb` → 섹션 3
- `project/code/ml/pca.ipynb` → 섹션 4
- `project/code/ml/clustering.ipynb` → 섹션 5
- `project/code/ml_part2/anomaly_detection.ipynb` → 섹션 7
- `project/code/ml_part2/predictive_maintenance.ipynb` → 섹션 8
- `project/code/ml/hyperparameter_optimization.ipynb` → 섹션 9

### 특이사항
- 노션 API에서 기존 button 블록(비우기/초기화 버튼 추정)은 replace_content로 대체됨
- 기초 통계 이론 페이지 스타일을 정확히 따름: `# 제목 {color="gray_bg"}`, `<table header-row="true">`, `:::callout {icon="💡" color="gray_bg"}`, `$\`...\`$` 인라인 LaTeX
- 코드 내용은 제외하고 이론만 정리 (필수 사항 준수)
- 각 섹션의 이론 내용은 소스 노트북에서 다룬 내용만 사용 (필수 사항 준수)
- 사용자 요청으로 경사하강법 섹션에 역전파(Backpropagation) 내용 추가 → 이후 사용자가 직접 제거
- 분류 성능 지표: 소스 노트북에서 다루는 지표(정확도, 정밀도, 재현율, F1, ROC/AUC, PR)만 포함 (추가 지표 없음 확인)
