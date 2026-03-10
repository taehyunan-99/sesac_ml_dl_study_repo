
---

## 진행 기록

### 현재 상황 (2026-03-10)
- 캐글 평가 지표: **ROC-AUC**
- 데이터: 제조 장비 예지보전, Target 불균형 (339/10000 = 3.39%)
- **FType이 사실상 답지** (FType != "No Failure" ↔ Target=1)

---

## 핵심 발견: 오입력 패턴

### Case B: Target=1 + No Failure (9건)
- FType은 "No Failure"인데 Target이 1 (고장)
- **FType 라벨링 오류, Target=1이 정답**
- ~~Target을 0으로 수정~~ → **수정하면 안 됨** (0.99451 → 0.99931)
- 모델이 "No Failure여도 고장일 수 있다"를 학습해야 함

### Case C: Target=0 + Failure FType (18건)
- **전부 Random Failures** (다른 FType에서는 오입력 0건)
- FType은 고장인데 Target이 0 (정상)
- **노이즈 데이터 → train에서 제거**

### FType별 오입력 현황 (train)
| FType | 건수 | 오입력 | 비율 |
|-------|------|--------|------|
| No Failure | 9652 | 0건 | 0% |
| Heat Dissipation | 112 | 0건 | 0% |
| Overstrain | 78 | 0건 | 0% |
| Power Failure | 95 | 0건 | 0% |
| Tool Wear | 45 | 0건 | 0% |
| **Random Failures** | **18** | **18건** | **100%** |

### test 후처리
- RF 7건 → 예측값 0.0으로 강제 보정 (train RF 100% 오입력 근거)

---

## 캐글 제출 기록

| 버전 | 설명 | 캐글 AUC | 비고 |
|------|------|---------|------|
| v1 | 기본 LGBM (default) | 0.940 | 베이스라인 |
| v2 | 정규화 LGBM + 50시드 Rank Avg | 0.950 | 앙상블 효과 |
| v3 | v2 + 파생변수 6개 | 0.953 | 피처 엔지니어링 |
| v4 | v3 + FType 피처 추가 | 0.994 | 핵심 피처 발견 |
| v7b | 3모델 50시드 + Case B→0 + RF보정 | 0.99451 | RF 보정 효과 |
| v10b | 3모델 + Case B→0 + RF제거 + RF보정 | 0.99486 | RF 제거 효과 |
| v10b | 3모델 + **Case B 유지** + RF제거 + RF보정 | 0.99931 | **Case B 유지가 핵심** |
| **v11b** | **5모델 + Case B 유지 + RF제거 + RF보정** | **0.99968** | **최종 최고점** |

### 효과 없었던 시도
| 시도 | 점수 | 이유 |
|------|------|------|
| LGBM 1000시드 | 0.99306 | 모델 다양성이 시드 수보다 중요 |
| 이상 탐지 피처 추가 | 0.99286 | FType과 중복 정보 |
| 순수 규칙 기반 (0/1) | 0.988 | 순위 세분화 불가 |
| 규칙+모델 혼합 | 0.99367 | 모델 자체보다 낮음 |
| No-FType 모델 블렌딩 | 0.984~0.994 | FType 안 쓰는 것과 동일 |
| Case C 보정 (Target→1) | 0.990 | 점수 하락 |

---

## 최종 모델 설정 (v11b, 0.99968)

### 전처리
- Case B (Target=1 + No Failure): **수정 안 함** (유지)
- Random Failures 18건: **train에서 제거**

### 피처 (13개)
- 원본: Type, AirTmp, ProcTmp, RotSpd, Torque, ToolWear
- FType: Ordinal 인코딩 (No Failure=0, Heat=1, Overstrain=2, Power=3, RF=4, TWF=5)
- 파생: temp_diff, power, wear_torque, wear_speed, temp_per_torque, temp_per_power

### 모델 (5모델 × 50시드 × 5폴드 = 1250개)
- **LGBM**: num_leaves=15, max_depth=5, min_child_samples=30, reg 0.1/1.0, sub/col 0.8
- **XGBoost**: max_depth=4, min_child_weight=30, reg 0.1/1.0, sub/col 0.8
- **RF**: max_depth=8, min_samples_leaf=20
- **CatBoost**: depth=5, l2_leaf_reg=3.0, subsample=0.8
- **ExtraTrees**: max_depth=8, min_samples_leaf=20
- 전체 보수적 세팅 (과적합 억제), class_weight='balanced'

### 앙상블
- Rank Averaging (5모델 동일 비중)

### 후처리
- test RF 7건 → 예측값 0.0으로 보정

---

## 핵심 교훈

1. **데이터 이해가 모델 튜닝보다 중요** — 오입력 패턴 분석으로 0.994 → 0.999
2. **Case B를 수정하면 안 된다** — 0.99451 → 0.99931 점프의 원인
3. **Random Failures는 100% 노이즈** — 제거 + 후처리 보정
4. **모델 다양성 > 시드 수** — 3모델 50시드(0.994) > LGBM 1000시드(0.993)
5. **보수적 세팅에서 0.99968** — 과적합 위험 없이 달성한 점수
6. **FType이 답지인 상황에서 추가 피처/튜닝은 효과 없음**
