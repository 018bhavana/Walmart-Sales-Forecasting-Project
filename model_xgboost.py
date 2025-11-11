# model_xgboost.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# Step 1: Load your dataset
# -------------------------------
# Make sure 'walmart_10000.csv' is in the same folder
data = pd.read_csv("walmart_10000.csv")

print("✅ Dataset loaded successfully!")
print("Dataset shape:", data.shape)
print(data.head())

# -------------------------------
# Step 2: Handle missing values
# -------------------------------
data = data.dropna()

# -------------------------------
# Step 3: Select features and target
# -------------------------------
# Example: Assuming 'Weekly_Sales' is the target
X = data.drop(columns=['Weekly_Sales'])
y = data['Weekly_Sales']

# If categorical columns exist, convert them
X = pd.get_dummies(X, drop_first=True)

# -------------------------------
# Step 4: Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# Step 5: Train the XGBoost Regressor
# -------------------------------
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# Step 6: Make Predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# Step 7: Evaluate the Model
# -------------------------------
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 XGBoost Regression Performance:")
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R² Score:", r2)

# -------------------------------
# Step 8: Visualization
# -------------------------------
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted - XGBoost")
plt.grid(True)
plt.show()

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(8,6))
plt.hist(residuals, bins=30, color='orange', edgecolor='black')
plt.title("Residual Distribution - XGBoost")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.show()

# -------------------------------
# Step 9: Save Results to File
# -------------------------------
with open("model_results.txt", "a") as f:
    f.write("\n\nXGBoost Regression Results:\n")
    f.write(f"MSE: {mse}\n")
    f.write(f"RMSE: {rmse}\n")
    f.write(f"MAE: {mae}\n")
    f.write(f"R²: {r2}\n")
    f.write("--------------------------------------------")
print("\n✅ Results saved in 'model_results.txt'")
