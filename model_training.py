import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import ta

# Load and train
df_train = pd.read_csv("btc_data_train_labeled_features.csv")
X_train = df_train.drop(columns=["Score"])
y_train = df_train["Score"]

model = XGBRegressor(
    n_estimators=100000,
    learning_rate=0.0025,
    max_depth=10,
    gamma=0.25,
    reg_alpha=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',
    random_state=42,
    n_jobs=-1
)


model.fit(X_train, y_train)

# Save the model as a binary file
model.save_model("ai_model0.bin")

# Print all feature importances
booster = model.get_booster()
importances = booster.get_score(importance_type='weight')  # or use 'gain', 'cover', 'total_gain', 'total_cover'

# Convert to DataFrame for better readability
importance_df = pd.DataFrame({
    "Feature": list(importances.keys()),
    "Importance": list(importances.values())
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importances:")
print(importance_df.to_string(index=False))
