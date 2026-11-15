
from pathlib import Path

import joblib
import onnxmltools
import onnx

from onnxmltools.convert.common.data_types import FloatTensorType


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "xgboost_classifier.joblib"

OUTPUT_PATH = (
    BASE_DIR / "models" / "xgboost_radiation_model.onnx"
)


# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================

print("Loading trained XGBoost model...")

model = joblib.load(MODEL_PATH)


# ============================================================
# 3. DEFINE FEATURES IN TRAINING ORDER
# ============================================================

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

print(f"Number of features: {len(FEATURES)}")


# ============================================================
# 4. RENAME XGBOOST FEATURE NAMES
# ============================================================

print("Preparing feature names...")

booster = model.get_booster()

booster.feature_names = [
    f"f{i}" for i in range(len(FEATURES))
]


# ============================================================
# 5. DEFINE ONNX INPUT
# ============================================================

initial_types = [
    (
        "float_input",
        FloatTensorType([None, len(FEATURES)])
    )
]


# ============================================================
# 6. CONVERT MODEL
# ============================================================

print("Converting XGBoost model to ONNX...")

onnx_model = onnxmltools.convert_xgboost(
    model,
    initial_types=initial_types,
    target_opset=15
)


# ============================================================
# 7. VALIDATE MODEL
# ============================================================

print("Validating ONNX model...")

onnx.checker.check_model(onnx_model)


# ============================================================
# 8. SAVE MODEL
# ============================================================

onnx.save_model(
    onnx_model,
    str(OUTPUT_PATH)
)

print("\n========== CONVERSION COMPLETED ==========")
print(f"ONNX model saved at:\n{OUTPUT_PATH}")
print(f"Input features: {len(FEATURES)}")
print(f"File size: {OUTPUT_PATH.stat().st_size / 1024:.2f} KB")
