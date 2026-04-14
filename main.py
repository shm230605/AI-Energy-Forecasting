import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------
# LOAD DATA
# ----------------------------
data = pd.read_csv("data/energy.csv")

data["Datetime"] = pd.to_datetime(data["Datetime"])
data = data.set_index("Datetime")

# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
data["hour"] = data.index.hour
data["day"] = data.index.dayofweek

# ----------------------------
# FEATURES / TARGET
# ----------------------------
X = data[["hour", "day"]]
y = data["Energy"]

# ----------------------------
# TRAIN TEST SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ----------------------------
# MODEL TRAINING
# ----------------------------
model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000)
model.fit(X_train, y_train)

# ----------------------------
# PREDICTION
# ----------------------------
predictions = model.predict(X_test)

# ----------------------------
# EVALUATION
# ----------------------------
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n===== MODEL PERFORMANCE =====")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)

# ----------------------------
# SAVE MODEL
# ----------------------------
joblib.dump(model, "models/energy_model.pkl")

# ----------------------------
# VISUALIZATION
# ----------------------------
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual")
plt.plot(predictions, label="Predicted")
plt.title("Energy Forecasting")
plt.legend()
plt.show()