# 데이터 불러오기
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("train shape:", train.shape)
print("test shape :", test.shape)

# 케이스 b
# target = 1 은 고장으로 인식
# failure type = no failure 고장 아님 -> 기계에서만 감지된 고장 신호?

# 케이스 c
# train 에서 제거 test에서 0으로 고정


case_b_mask = (train["Target"] == 1) & (train["Failure Type"] == "No Failure")
case_c_train_mask = (train["Failure Type"] == "Random Failures")
case_c_test_mask = (test["Failure Type"] == "Random Failures")

print("Case B count:", int(case_b_mask.sum()))
print("Case C train count:", int(case_c_train_mask.sum()))
print("Case C test count :", int(case_c_test_mask.sum()))

# train에서 Random Failures 제거
train = train.loc[~case_c_train_mask].reset_index(drop=True)
print("train shape after removing Random Failures:", train.shape)

# 원 핫 인코딩 진행

failure_map = {
    "No Failure": 0,
    "Heat Dissipation Failure": 1,
    "Power Failure": 2,
    "Overstrain Failure": 3,
    "Tool Wear Failure": 4,
    "Random Failures": 5
}

train["Failure_Type_Enc"] = train["Failure Type"].map(failure_map)
test["Failure_Type_Enc"] = test["Failure Type"].map(failure_map)

# 고장 유형 있는지 확인
# no failure 아니면 1
# no_failure_flag 에서 no failure 이면 1
train["failure_flag"] = (train["Failure Type"] != "No Failure").astype(int)
test["failure_flag"] = (test["Failure Type"] != "No Failure").astype(int)

train["no_failure_flag"] = (train["Failure Type"] == "No Failure").astype(int)
test["no_failure_flag"] = (test["Failure Type"] == "No Failure").astype(int)

train["case_b_flag"] = ((train["Target"] == 1) & (train["Failure Type"] == "No Failure")).astype(int)
test["case_b_flag"] = 0

# product id를 문자와 숫자를 나눔 왜 나눴는지는 잘 모르겠음...
train["Product_Prefix"] = train["Product ID"].str[0]
test["Product_Prefix"] = test["Product ID"].str[0]

# 숫자 부분 추출 함수
def extract_num_id(series):
    return series.str.extract(r"(\d+)").astype(int)[0]

train["Product_Num"] = extract_num_id(train["Product ID"])
test["Product_Num"] = extract_num_id(test["Product ID"])

# 라벨링 진행
type_le = LabelEncoder()
prefix_le = LabelEncoder()

train["Type"] = type_le.fit_transform(train["Type"])
test["Type"] = type_le.transform(test["Type"])

train["Product_Prefix"] = prefix_le.fit_transform(train["Product_Prefix"])
test["Product_Prefix"] = prefix_le.transform(test["Product_Prefix"])

test_id = test["UDI"].copy()

# 파생변수 만들기 
def add_features(df):
    df = df.copy()

    df["temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["temp_sum"] = df["Process temperature [K]"] + df["Air temperature [K]"]
    df["temp_ratio"] = df["Process temperature [K]"] / (df["Air temperature [K]"] + 1e-6)

    df["power_like"] = df["Rotational speed [rpm]"] * df["Torque [Nm]"]
    df["wear_torque"] = df["Tool wear [min]"] * df["Torque [Nm]"]
    df["rpm_per_wear"] = df["Rotational speed [rpm]"] / (df["Tool wear [min]"] + 1)
    df["torque_per_rpm"] = df["Torque [Nm]"] / (df["Rotational speed [rpm]"] + 1)

    df["temp_wear"] = df["temp_diff"] * df["Tool wear [min]"]
    df["torque_temp"] = df["Torque [Nm]"] * df["Process temperature [K]"]
    df["rpm_tempdiff"] = df["Rotational speed [rpm]"] * df["temp_diff"]
    df["tempdiff_torque"] = df["temp_diff"] * df["Torque [Nm]"]

    df["wear_temp_ratio"] = df["Tool wear [min]"] / (df["temp_diff"].abs() + 1)
    df["temp_wear_ratio"] = df["temp_diff"] / (df["Tool wear [min]"] + 1)
    df["torque_wear_ratio"] = df["Torque [Nm]"] / (df["Tool wear [min]"] + 1)
    df["wear_rpm_ratio"] = df["Tool wear [min]"] / (df["Rotational speed [rpm]"] + 1)

    df["rpm_log"] = np.log1p(df["Rotational speed [rpm]"])
    df["torque_log"] = np.log1p(df["Torque [Nm]"])
    df["wear_log"] = np.log1p(df["Tool wear [min]"])

    df["rpm_square"] = df["Rotational speed [rpm]"] ** 2
    df["torque_square"] = df["Torque [Nm]"] ** 2
    df["wear_square"] = df["Tool wear [min]"] ** 2
    df["tempdiff_square"] = df["temp_diff"] ** 2

    df["load_factor"] = (
        df["Torque [Nm]"] *
        df["Tool wear [min]"] *
        df["Rotational speed [rpm]"]
    )

    df["thermal_load"] = (
        df["temp_diff"] *
        df["Torque [Nm]"] *
        df["Rotational speed [rpm]"]
    )

    df["wear_stress"] = (
        df["Tool wear [min]"] *
        df["Torque [Nm]"] *
        df["temp_diff"]
    )

    df["rotational_stress"] = (
        df["Rotational speed [rpm]"] *
        df["Tool wear [min]"] *
        df["temp_diff"]
    )

    df["load_per_wear"] = (
        df["Torque [Nm]"] * df["Rotational speed [rpm]"]
    ) / (df["Tool wear [min]"] + 1)

    df["wear_per_load"] = (
        df["Tool wear [min]"]
    ) / (df["Torque [Nm]"] * df["Rotational speed [rpm]"] + 1)

    df["thermal_efficiency_like"] = (
        df["Rotational speed [rpm]"] /
        (df["Torque [Nm]"] * (df["temp_diff"].abs() + 1) + 1)
    )

    df["prefix_torque"] = df["Product_Prefix"] * df["Torque [Nm]"]
    df["prefix_rpm"] = df["Product_Prefix"] * df["Rotational speed [rpm]"]
    df["prefix_wear"] = df["Product_Prefix"] * df["Tool wear [min]"]

    df["prefix_load"] = (
        df["Product_Prefix"] *
        df["Torque [Nm]"] *
        df["Rotational speed [rpm]"]
    )

    df["type_torque"] = df["Type"] * df["Torque [Nm]"]
    df["type_rpm"] = df["Type"] * df["Rotational speed [rpm]"]
    df["type_wear"] = df["Type"] * df["Tool wear [min]"]


    df["failure_torque"] = df["failure_flag"] * df["Torque [Nm]"]
    df["failure_rpm"] = df["failure_flag"] * df["Rotational speed [rpm]"]
    df["failure_wear"] = df["failure_flag"] * df["Tool wear [min]"]
    df["failure_tempdiff"] = df["failure_flag"] * df["temp_diff"]

    df["failure_load"] = (
        df["failure_flag"] *
        df["Torque [Nm]"] *
        df["Rotational speed [rpm]"]
    )

    df["caseb_torque"] = df["case_b_flag"] * df["Torque [Nm]"]
    df["caseb_rpm"] = df["case_b_flag"] * df["Rotational speed [rpm]"]
    df["caseb_tempdiff"] = df["case_b_flag"] * df["temp_diff"]

    return df

train = add_features(train)
test = add_features(test)

drop_cols = ["UDI", "Product ID", "Failure Type", "Target", "Num ID"]

X = train.drop(columns=drop_cols, errors="ignore").copy()
y = train["Target"].copy()

X_test = test.drop(columns=["UDI", "Product ID", "Failure Type", "Num ID"], errors="ignore").copy()
X_test = X_test[X.columns]

print("X shape:", X.shape)
print("X_test shape:", X_test.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 사용할 모델 정의
models = {
    "rf": RandomForestClassifier(
        n_estimators=800,
        max_depth=12,
        min_samples_leaf=2,
        min_samples_split=4,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "et": ExtraTreesClassifier(
        n_estimators=900,
        max_depth=12,
        min_samples_leaf=2,
        min_samples_split=4,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "lr": LogisticRegression(
        C=0.5,
        class_weight="balanced",
        max_iter=3000,
        random_state=42
    ),
    "gnb": GaussianNB(),
    "knn": KNeighborsClassifier(
        n_neighbors=25,
        weights="distance",
        metric="minkowski"
    )
}

# k-fold 검증 진행
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# OOF 예측 및 테스트 저장 공간 만들기
oof_preds = {name: np.zeros(len(X)) for name in models.keys()}
test_preds = {name: np.zeros(len(X_test)) for name in models.keys()}
cv_scores = {name: [] for name in models.keys()}

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n========== Fold {fold} ==========")

    X_train_raw = X.iloc[train_idx]
    X_valid_raw = X.iloc[valid_idx]
    y_train = y.iloc[train_idx]
    y_valid = y.iloc[valid_idx]

    X_train_scaled = X_scaled[train_idx]
    X_valid_scaled = X_scaled[valid_idx]

    for name, model in models.items():
        # 스케일링 필요한 모델만 scaled 사용
        if name in ["lr", "knn", "gnb"]:
            model.fit(X_train_scaled, y_train)
            valid_prob = model.predict_proba(X_valid_scaled)[:, 1]
            test_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train_raw, y_train)
            valid_prob = model.predict_proba(X_valid_raw)[:, 1]
            test_prob = model.predict_proba(X_test)[:, 1]

        oof_preds[name][valid_idx] = valid_prob
        test_preds[name] += test_prob / skf.n_splits

        fold_auc = roc_auc_score(y_valid, valid_prob)
        cv_scores[name].append(fold_auc)

        print(f"[{name}] Fold {fold} AUC: {fold_auc:.6f}")

# 모델 성능 요약
for name in models.keys():
    oof_auc = roc_auc_score(y, oof_preds[name])
    oof_pr = average_precision_score(y, oof_preds[name])
    mean_auc = np.mean(cv_scores[name])

    print(f"{name} | Mean Fold AUC: {mean_auc:.6f} | OOF AUC: {oof_auc:.6f} | OOF PR-AUC: {oof_pr:.6f}")

# 앙상블
ensemble_weights = {
    "rf": 0.28,
    "et": 0.24,
    "lr": 0.20,
    "gnb": 0.14,
    "knn": 0.14
}

oof_ensemble = np.zeros(len(X))
test_ensemble = np.zeros(len(X_test))

for name, weight in ensemble_weights.items():
    oof_ensemble += oof_preds[name] * weight
    test_ensemble += test_preds[name] * weight

# 앙상블 성능 평가
ensemble_auc = roc_auc_score(y, oof_ensemble)
ensemble_pr = average_precision_score(y, oof_ensemble)

print("\n========== Ensemble ==========")
print("Ensemble OOF ROC-AUC:", ensemble_auc)
print("Ensemble OOF PR-AUC :", ensemble_pr)

# random failures는 0으로 고정
# 최종적으로 고장 확률이 0이라고 생각
test_ensemble[case_c_test_mask.values] = 0.0

known_failure_mask = (test["Failure Type"] != "No Failure") & (test["Failure Type"] != "Random Failures")
test_ensemble[known_failure_mask.values] = np.maximum(test_ensemble[known_failure_mask.values], 0.995)

no_failure_mask = (test["Failure Type"] == "No Failure")
test_ensemble[no_failure_mask.values] = np.clip(test_ensemble[no_failure_mask.values], 0.0, 0.999)

submission = pd.DataFrame({
    "UDI": test_id,
    "Target": test_ensemble
})

submission.to_csv("ensemble12.csv", index=False)

print("파일 저장 완료")
print(submission.head(20))

final_rf = RandomForestClassifier(
    n_estimators=1000,
    max_depth=12,
    min_samples_leaf=2,
    min_samples_split=4,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
final_rf.fit(X, y)

importance = pd.Series(final_rf.feature_importances_, index=X.columns).sort_values(ascending=False)

print("\n========== Top 25 Feature Importance ==========")
print(importance.head(25))

print("\n========== Diagnostic ==========")
print("Known failure in test count:", int(known_failure_mask.sum()))
print("Random Failures in test count:", int(case_c_test_mask.sum()))
print("Submission target summary:")
print(submission["Target"].describe())