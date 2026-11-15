
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    HistGradientBoostingClassifier
)

from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    average_precision_score,
    roc_auc_score
)

from sklearn.utils.class_weight import compute_sample_weight


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_FILE = "radiation_ml_training_data_v2.csv"

OUTPUT_DIR = "radiation_v2_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

MEDIAN_TARGET = "target_next_hour_median"

MAX_TARGET = "target_next_hour_max"

CLASSIFICATION_TARGET = "target_high_radiation"

TEST_SIZE = 0.20

CLASSIFICATION_THRESHOLD = 0.50

RANDOM_STATE = 42


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n========== LOADING DATA ==========")

df = pd.read_csv(DATA_FILE, index_col=0)

df.index = pd.to_datetime(df.index)

df = df.sort_index()

print("Total dataset rows:", len(df))

print("First timestamp:", df.index.min())

print("Last timestamp:", df.index.max())


# Check required columns.

required_columns = FEATURES + [
    MEDIAN_TARGET,
    MAX_TARGET,
    CLASSIFICATION_TARGET
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in dataset: {missing_columns}"
    )


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def chronological_split(data):

    data = data.sort_index()

    split_index = int(
        len(data) * (1 - TEST_SIZE)
    )

    train = data.iloc[:split_index].copy()

    test = data.iloc[split_index:].copy()

    return train, test


def regression_metrics(name, actual, predictions):

    mae = mean_absolute_error(
        actual,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(actual, predictions)
    )

    r2 = r2_score(
        actual,
        predictions
    )

    print(f"\n{name}")
    print(f"MAE:  {mae:.4f} μGy/hour")
    print(f"RMSE: {rmse:.4f} μGy/hour")
    print(f"R2:   {r2:.4f}")

    return {
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# ============================================================
# 4. REGRESSION EXPERIMENT
# ============================================================

def run_regression_experiment(
    target_name,
    baseline_feature,
    experiment_name
):

    print("\n")
    print("=" * 65)
    print(f"REGRESSION: {experiment_name}")
    print("=" * 65)

    data = df.dropna(
        subset=FEATURES + [target_name]
    ).copy()

    train, test = chronological_split(data)

    X_train = train[FEATURES]
    X_test = test[FEATURES]

    y_train = train[target_name]
    y_test = test[target_name]

    print("\n========== DATA SPLIT ==========")

    print("Total samples:", len(data))

    print("Training samples:", len(train))

    print("Testing samples:", len(test))

    print("Training period:", train.index.min(), "to", train.index.max())

    print("Testing period:", test.index.min(), "to", test.index.max())

    results = []

    predictions = {}

    # --------------------------------------------------------
    # A. PERSISTENCE BASELINE
    # --------------------------------------------------------

    baseline_predictions = test[baseline_feature].values

    predictions["Persistence Baseline"] = baseline_predictions

    results.append(
        regression_metrics(
            "Persistence Baseline",
            y_test,
            baseline_predictions
        )
    )

    # --------------------------------------------------------
    # B. RANDOM FOREST
    # --------------------------------------------------------

    print("\nTraining Random Forest Regressor...")

    rf_model = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    rf_model.fit(X_train, y_train)

    rf_predictions = rf_model.predict(X_test)

    predictions["Random Forest"] = rf_predictions

    results.append(
        regression_metrics(
            "Random Forest",
            y_test,
            rf_predictions
        )
    )

    joblib.dump(
        {
            "model": rf_model,
            "features": FEATURES,
            "target": target_name,
            "model_name": "Random Forest"
        },
        os.path.join(
            OUTPUT_DIR,
            f"random_forest_{experiment_name}.joblib"
        )
    )

    # --------------------------------------------------------
    # C. HISTOGRAM GRADIENT BOOSTING
    # --------------------------------------------------------

    print("\nTraining HistGradientBoosting Regressor...")

    gb_model = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE
    )

    gb_model.fit(X_train, y_train)

    gb_predictions = gb_model.predict(X_test)

    predictions["HistGradientBoosting"] = gb_predictions

    results.append(
        regression_metrics(
            "HistGradientBoosting",
            y_test,
            gb_predictions
        )
    )

    joblib.dump(
        {
            "model": gb_model,
            "features": FEATURES,
            "target": target_name,
            "model_name": "HistGradientBoosting"
        },
        os.path.join(
            OUTPUT_DIR,
            f"hist_gradient_boosting_{experiment_name}.joblib"
        )
    )

    # --------------------------------------------------------
    # D. SAVE COMPARISON
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values("RMSE")

    print("\n========== REGRESSION COMPARISON ==========")

    print(results_df.to_string(index=False))

    results_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{experiment_name}_comparison.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # E. SAVE TEST PREDICTIONS
    # --------------------------------------------------------

    prediction_df = pd.DataFrame(
        {
            "actual": y_test,
            "persistence": predictions["Persistence Baseline"],
            "random_forest": predictions["Random Forest"],
            "hist_gradient_boosting": predictions[
                "HistGradientBoosting"
            ]
        },
        index=y_test.index
    )

    prediction_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{experiment_name}_predictions.csv"
        )
    )

    return results_df


# ============================================================
# 5. RUN BOTH REGRESSION EXPERIMENTS
# ============================================================

median_results = run_regression_experiment(
    target_name=MEDIAN_TARGET,
    baseline_feature="radiation_median",
    experiment_name="median_radiation"
)

max_results = run_regression_experiment(
    target_name=MAX_TARGET,
    baseline_feature="radiation_max",
    experiment_name="maximum_radiation"
)


# ============================================================
# 6. CLASSIFICATION EXPERIMENT
# ============================================================

print("\n")
print("=" * 65)
print("HIGH-RADIATION EVENT CLASSIFICATION")
print("=" * 65)

classification_data = df.dropna(
    subset=FEATURES + [CLASSIFICATION_TARGET]
).copy()

classification_data[CLASSIFICATION_TARGET] = (
    classification_data[CLASSIFICATION_TARGET].astype(int)
)

train, test = chronological_split(
    classification_data
)

X_train = train[FEATURES]
X_test = test[FEATURES]

y_train = train[CLASSIFICATION_TARGET]
y_test = test[CLASSIFICATION_TARGET]

print("\n========== CLASSIFICATION DATA SPLIT ==========")

print("Total samples:", len(classification_data))

print("Training samples:", len(train))

print("Testing samples:", len(test))

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

print("\nTraining period:", train.index.min(), "to", train.index.max())

print("Testing period:", test.index.min(), "to", test.index.max())

if y_train.nunique() < 2:
    raise ValueError(
        "Training data contains only one class. "
        "Cannot train a classifier."
    )

if y_test.nunique() < 2:
    raise ValueError(
        "Test data contains only one class. "
        "Classification evaluation would be incomplete."
    )


# ============================================================
# 7. CLASSIFICATION EVALUATION FUNCTION
# ============================================================

classification_results = {}

def evaluate_classifier(
    name,
    actual,
    predicted,
    probabilities
):

    precision = precision_score(
        actual,
        predicted,
        zero_division=0
    )

    recall = recall_score(
        actual,
        predicted,
        zero_division=0
    )

    f1 = f1_score(
        actual,
        predicted,
        zero_division=0
    )

    average_precision = average_precision_score(
        actual,
        probabilities
    )

    roc_auc = roc_auc_score(
        actual,
        probabilities
    )

    matrix = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1]
    )

    tn, fp, fn, tp = matrix.ravel()

    print(f"\n========== {name} ==========")

    print(f"Precision: {precision:.4f}")

    print(f"Recall:    {recall:.4f}")

    print(f"F1-score:  {f1:.4f}")

    print(f"PR-AUC:    {average_precision:.4f}")

    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nConfusion Matrix:")

    print("                 Predicted Normal | Predicted High")

    print(f"Actual Normal:       {tn:6d}       | {fp:6d}")

    print(f"Actual High:         {fn:6d}       | {tp:6d}")

    print("\nClassification Report:")

    print(
        classification_report(
            actual,
            predicted,
            labels=[0, 1],
            target_names=["Normal", "High Radiation"],
            zero_division=0
        )
    )

    return {
        "model": name,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "PR_AUC": average_precision,
        "ROC_AUC": roc_auc,
        "True_Positive": tp,
        "False_Positive": fp,
        "False_Negative": fn,
        "True_Negative": tn
    }


# ============================================================
# 8. BASELINE CLASSIFIER
# ============================================================

print("\nTraining Dummy Classifier baseline...")

dummy_model = DummyClassifier(
    strategy="most_frequent"
)

dummy_model.fit(X_train, y_train)

dummy_predictions = dummy_model.predict(X_test)

dummy_probabilities = dummy_model.predict_proba(X_test)[:, 1]

classification_results["Dummy Baseline"] = (
    evaluate_classifier(
        "Dummy Baseline",
        y_test,
        dummy_predictions,
        dummy_probabilities
    )
)


# ============================================================
# 9. RANDOM FOREST CLASSIFIER
# ============================================================

print("\nTraining Random Forest Classifier...")

rf_classifier = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf_classifier.fit(X_train, y_train)

rf_probabilities = rf_classifier.predict_proba(X_test)[:, 1]

rf_predictions = (
    rf_probabilities >= CLASSIFICATION_THRESHOLD
).astype(int)

classification_results["Random Forest"] = (
    evaluate_classifier(
        "Random Forest Classifier",
        y_test,
        rf_predictions,
        rf_probabilities
    )
)

joblib.dump(
    {
        "model": rf_classifier,
        "features": FEATURES,
        "target": CLASSIFICATION_TARGET,
        "model_name": "Random Forest Classifier",
        "threshold": CLASSIFICATION_THRESHOLD
    },
    os.path.join(
        OUTPUT_DIR,
        "random_forest_classifier.joblib"
    )
)


# ============================================================
# 10. HISTOGRAM GRADIENT BOOSTING CLASSIFIER
# ============================================================

print("\nTraining HistGradientBoosting Classifier...")

gb_classifier = HistGradientBoostingClassifier(
    max_iter=200,
    learning_rate=0.05,
    max_leaf_nodes=15,
    l2_regularization=1.0,
    random_state=RANDOM_STATE
)

# Give rare high-radiation samples greater training weight.
sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

gb_classifier.fit(
    X_train,
    y_train,
    sample_weight=sample_weights
)

gb_probabilities = gb_classifier.predict_proba(X_test)[:, 1]

gb_predictions = (
    gb_probabilities >= CLASSIFICATION_THRESHOLD
).astype(int)

classification_results["HistGradientBoosting"] = (
    evaluate_classifier(
        "HistGradientBoosting Classifier",
        y_test,
        gb_predictions,
        gb_probabilities
    )
)

joblib.dump(
    {
        "model": gb_classifier,
        "features": FEATURES,
        "target": CLASSIFICATION_TARGET,
        "model_name": "HistGradientBoosting Classifier",
        "threshold": CLASSIFICATION_THRESHOLD
    },
    os.path.join(
        OUTPUT_DIR,
        "hist_gradient_boosting_classifier.joblib"
    )
)


# ============================================================
# 11. SAVE CLASSIFICATION COMPARISON
# ============================================================

classification_comparison = pd.DataFrame(
    list(classification_results.values())
)

classification_comparison = (
    classification_comparison.sort_values(
        "PR_AUC",
        ascending=False
    )
)

print("\n========== CLASSIFICATION COMPARISON ==========")

print(
    classification_comparison.to_string(index=False)
)

classification_comparison.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "classification_comparison.csv"
    ),
    index=False
)


# ============================================================
# 12. SAVE CLASSIFICATION PREDICTIONS
# ============================================================

classification_predictions = pd.DataFrame(
    {
        "actual": y_test,
        "dummy_prediction": dummy_predictions,
        "random_forest_probability": rf_probabilities,
        "random_forest_prediction": rf_predictions,
        "gradient_boosting_probability": gb_probabilities,
        "gradient_boosting_prediction": gb_predictions
    },
    index=y_test.index
)

classification_predictions.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "classification_predictions.csv"
    )
)


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 65)
print("EXPERIMENT 2 COMPLETED")
print("=" * 65)

print("\nResults folder:", OUTPUT_DIR)

print("\nGenerated files:")

for filename in sorted(os.listdir(OUTPUT_DIR)):
    print("-", filename)

print("\nAll regression and classification experiments completed.")
print("No final model has been selected yet.")
