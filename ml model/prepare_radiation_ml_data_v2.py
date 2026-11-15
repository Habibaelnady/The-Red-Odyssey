
import pandas as pd
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "radiation_ml_training_data_v2.csv"
MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "target_high_radiation"

FEATURES = [
    "radiation_median",
    "radiation_mean",
    "radiation_max",
    "measurement_count",
    "median_lag_1h",
    "median_lag_2h",
    "mean_lag_1h",
    "max_lag_1h",
    "median_rolling_mean_3h",
    "max_rolling_max_3h",
    "radiation_change_1h",
    "radiation_change_2h",
    "hour_of_day",
    "day_of_year",
    "month"
]

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n========== LOADING DATA ==========")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Training dataset not found: {DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE)

required_columns = FEATURES + [TARGET, "timestamp"]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df["timestamp"] = pd.to_datetime(
    df["timestamp"], errors="coerce"
)

df = df.dropna(subset=required_columns)

df = df.sort_values("timestamp").reset_index(drop=True)

X = df[FEATURES]
y = df[TARGET].astype(int)

if y.nunique() < 2:
    raise ValueError("Dataset must contain both classes.")

print("Total samples:", len(df))
print("Number of features:", len(FEATURES))
print("Normal radiation:", (y == 0).sum())
print("High radiation:", (y == 1).sum())


# ============================================================
# 3. CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

print("\n========== TRAIN / TEST SPLIT ==========")

split_index = int(len(df) * (1 - TEST_SIZE))

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining period:")
print(df["timestamp"].iloc[0])
print("to", df["timestamp"].iloc[split_index - 1])

print("\nTesting period:")
print(df["timestamp"].iloc[split_index])
print("to", df["timestamp"].iloc[-1])

if y_train.nunique() < 2 or y_test.nunique() < 2:
    raise ValueError(
        "Both training and testing sets must contain "
        "normal and high-radiation samples."
    )


# ============================================================
# 4. DEFINE MODELS
# ============================================================

print("\n========== DEFINING MODELS ==========")

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}


# ============================================================
# 5. TRAIN AND TEST
# ============================================================

print("\n========== TRAINING AND EVALUATION ==========")

for name, model in models.items():

    print(f"\n{'=' * 15} {name} {'=' * 15}")

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # Evaluate
    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    pr_auc = average_precision_score(
        y_test, probabilities
    )

    roc_auc = roc_auc_score(
        y_test, probabilities
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    print("\nEvaluation Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives:  {tp}")

    # Save trained model
    filename = (
        name.lower().replace(" ", "_") + "_classifier.joblib"
    )

    model_path = MODELS_DIR / filename

    joblib.dump(model, model_path)

    print("\nSaved model:", model_path)


# ============================================================
# 6. FINISH
# ============================================================

print("\n========== TRAINING COMPLETED ==========")
print("Both models were trained, tested and saved.")
