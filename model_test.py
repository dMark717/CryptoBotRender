import pandas as pd
import numpy as np
from xgboost import Booster, DMatrix

# === USER CONFIGURABLE PARAMETERS ===
SHAP_BUCKETS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]
PRED_BUCKETS = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

# === DATA LOADING ===
df_test = pd.read_csv("btc_data_test_labeled_features.csv")
df_test = df_test.dropna()

X_test = df_test.drop(columns=["Score"])
y_test = df_test["Score"].values
dtest = DMatrix(X_test)

model = Booster()
model.load_model("btc_xgb_model.bin")

# === PREDICTIONS ===
preds = model.predict(dtest)

predicted_signs = np.sign(preds)
actual_signs = np.sign(y_test)

# === CONDITIONAL SHAP CALCULATION ===
pred_mask_for_shap = np.abs(preds) > 0.25
dtest_shap = DMatrix(X_test[pred_mask_for_shap])
shap_values_subset = model.predict(dtest_shap, pred_contribs=True)
shap_strengths = np.zeros(len(preds))
shap_strengths[pred_mask_for_shap] = np.sum(np.abs(shap_values_subset), axis=1)

# === BUCKETED EVALUATION ===
print(f"{'SHAP ↓ / PRED →':<20}", end='')
for i in range(len(PRED_BUCKETS) - 1):
    print(f"[{PRED_BUCKETS[i]:.2f}-{PRED_BUCKETS[i+1]:.2f}]".center(18), end='')
print()

for i in range(len(SHAP_BUCKETS) - 1):
    shap_low, shap_high = SHAP_BUCKETS[i], SHAP_BUCKETS[i+1]
    print(f"[{shap_low:.2f}-{shap_high:.2f}]".ljust(20), end='')

    shap_mask = (shap_strengths > shap_low) & (shap_strengths <= shap_high)

    for j in range(len(PRED_BUCKETS) - 1):
        pred_low, pred_high = PRED_BUCKETS[j], PRED_BUCKETS[j+1]
        pred_mask = (np.abs(preds) > pred_low) & (np.abs(preds) <= pred_high)

        combined_mask = shap_mask & pred_mask
        filtered_preds = predicted_signs[combined_mask]
        filtered_actuals = actual_signs[combined_mask]

        total = len(filtered_preds)
        correct = np.sum(filtered_preds == filtered_actuals)
        accuracy = correct / total * 100 if total > 0 else 0

        print(f"{accuracy:6.2f}% ({total:4})", end='   ')
    print()
