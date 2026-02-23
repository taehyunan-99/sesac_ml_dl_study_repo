"""
=============================================================
v36_zscore: 글로벌 재질 불량률(Material Failure Rate) 모델
=============================================================
핵심 아이디어:
  - attribute_0, attribute_1(재질 정보)를 train 데이터로 불량률 매핑
  - v21 Huber imputation + 1000-Seed Rank Averaging 기반
  - Global Material Rate를 새 피처로 추가

Public Score: 0.58921 (재질 불량률이 F-I 제품에 과적합)
교훈: train(A-E)의 재질 불량률이 test(F-I)에 일반화되지 않음
=============================================================
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "../../data/train.csv")
TEST_PATH = os.path.join(BASE_DIR, "../../data/test.csv")
SUB_PATH = os.path.join(BASE_DIR, "../../data/sample_submission.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "../../data/v36_zscore.csv")


def run():
    # ─────────────────────────────────────────
    # Step 1: 데이터 로딩 & 결측치 플래그
    # ─────────────────────────────────────────
    # m3, m5의 결측 여부를 이진 플래그로 변환
    # → EDA에서 이 두 열의 결측이 failure와 상관관계가 높음을 발견
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample_sub = pd.read_csv(SUB_PATH)

    for df in [train, test]:
        df['m3_nan'] = df['measurement_3'].isna().astype(int)
        df['m5_nan'] = df['measurement_5'].isna().astype(int)

    data = pd.concat([train, test], ignore_index=True)

    # ─────────────────────────────────────────
    # Step 2: 글로벌 재질 불량률 매핑 (핵심 아이디어)
    # ─────────────────────────────────────────
    # attribute_0/attribute_1은 제품 재질 정보 (e.g. material_7)
    # train 데이터에서 각 재질별 평균 failure율을 계산하여 피처로 사용
    # 예: material_7 → 평균 불량률 0.21 등
    mat0_df = train[['attribute_0', 'failure']].rename(columns={'attribute_0': 'material'})
    mat1_df = train[['attribute_1', 'failure']].rename(columns={'attribute_1': 'material'})
    global_mats = pd.concat([mat0_df, mat1_df], ignore_index=True)
    mat_rate_map = global_mats.groupby('material')['failure'].mean().to_dict()

    # train(A-E)의 재질 불량률을 전체 데이터에 매핑
    # ⚠️ 문제: test(F-I)는 train에 없던 재질 조합 → 과적합 위험
    data['global_material_rate_0'] = data['attribute_0'].map(mat_rate_map)
    data['global_material_rate_1'] = data['attribute_1'].map(mat_rate_map)

    # 매핑 안 된 경우 전체 평균으로 대체
    global_mean_fail = train['failure'].mean()
    data['global_material_rate_0'].fillna(global_mean_fail, inplace=True)
    data['global_material_rate_1'].fillna(global_mean_fail, inplace=True)

    # ─────────────────────────────────────────
    # Step 3: measurement_17 Huber 회귀 복원
    # ─────────────────────────────────────────
    # m17은 다른 measurement와 선형 관계 → Huber 회귀로 결측 복원
    # 제품(product_code)별로 상관관계가 다르므로 개별 모델 학습
    m_candidates = [f'measurement_{i}' for i in range(3, 17)]
    m_target = 'measurement_17'

    for pc in data['product_code'].unique():
        sub = data[data['product_code'] == pc].copy()
        clean = sub.dropna(subset=m_candidates + [m_target])
        if len(clean) > 20:
            # m17과 상관관계 상위 4개 피처 선택
            corr = clean[m_candidates + [m_target]].corr()[m_target].abs().sort_values(ascending=False)
            significant_m = corr.index[1:5].tolist()

            hr = HuberRegressor(max_iter=1000)  # 이상치에 강건한 회귀
            hr.fit(clean[significant_m], clean[m_target])

            target_mask = (data['product_code'] == pc) & data[m_target].isna()
            valid_target = data[target_mask].dropna(subset=significant_m)
            if not valid_target.empty:
                data.loc[valid_target.index, m_target] = hr.predict(valid_target[significant_m])

    # 나머지 결측치: 제품별 중앙값으로 대체
    for col in [f'measurement_{i}' for i in range(3, 18)] + ['loading']:
        data[col] = data[col].fillna(data.groupby('product_code')[col].transform('median'))

    # ─────────────────────────────────────────
    # Step 4: 피처 엔지니어링
    # ─────────────────────────────────────────
    data['loading'] = np.log1p(data['loading'])          # 로딩 로그 변환 (왜도 감소)
    data['area'] = data['measurement_0'] * data['measurement_1']  # 제품 면적
    data['l_m17_inter'] = data['loading'] * data['measurement_17']  # 로딩 × m17 상호작용
    data['m3_m5_nan'] = data['m3_nan'] * data['m5_nan']  # 둘 다 결측인 경우 시너지

    # 총 12개 피처: v21 기본 10개 + 재질 불량률 2개
    features = [
        'loading', 'measurement_17',
        'measurement_0', 'measurement_1', 'measurement_2',
        'm3_nan', 'm5_nan', 'm3_m5_nan',
        'area', 'l_m17_inter',
        'global_material_rate_0', 'global_material_rate_1'  # ← v36 추가 피처
    ]

    X = data.iloc[:len(train)][features]
    y = train['failure']
    X_test = data.iloc[len(train):][features]

    # ─────────────────────────────────────────
    # Step 5: 스케일링 + 1000-Seed 훈련
    # ─────────────────────────────────────────
    # StandardScaler: 평균 0, 분산 1로 정규화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    # 1000개 시드 × 5-Fold → 5000번 예측 후 Rank 평균
    # 목적: 시드 노이즈 제거, 안정적인 예측 확보
    final_ranks = np.zeros(len(X_test))
    num_seeds = 1000

    for seed in range(num_seeds):
        current_seed = 36000 + seed
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=current_seed)
        seed_preds = np.zeros(len(X_test))

        for train_idx, val_idx in skf.split(X_scaled, y):
            # L2 정규화, C=0.001 (강한 정규화 → 일반화 우선)
            model = LogisticRegression(C=0.001, penalty='l2', solver='liblinear', random_state=current_seed)
            model.fit(X_scaled[train_idx], y.iloc[train_idx])
            seed_preds += model.predict_proba(X_test_scaled)[:, 1] / 5

        # 확률값 → 순위(Rank)로 변환 후 누적
        final_ranks += pd.Series(seed_preds).rank(pct=True).values / num_seeds

        if (seed + 1) % 200 == 0:
            print(f"  {seed + 1}/{num_seeds} seeds done...")

    sample_sub['failure'] = final_ranks
    sample_sub.to_csv(OUTPUT_PATH, index=False)
    print(f"v36 저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
