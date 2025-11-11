import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

# -------------------------------
# STEP 1: Load Dataset
# -------------------------------
print("✅ Loading Walmart dataset...")
data = pd.read_csv("walmart_10000.csv")

# If there is no date column, create one automatically
if 'date' not in data.columns:
    print("⚠️ No 'date' column found — generating synthetic date range...")
    data['date'] = pd.date_range(start='2015-01-01', periods=len(data), freq='D')

data.set_index('date', inplace=True)
print("✅ Data Loaded Successfully!\n", data.head())

# -------------------------------
# STEP 2: Use Weekly Sales as target
# -------------------------------
sales = data['Weekly_Sales']


# -------------------------------
# STEP 3: Split into train and test
# -------------------------------
split_index = int(len(sales) * 0.8)
train, test = sales[:split_index], sales[split_index:]

print(f"\nTraining samples: {len(train)}, Testing samples: {len(test)}")

# -------------------------------
# STEP 4: Fit ARIMA model
# -------------------------------
print("\n✅ Fitting ARIMA model (2,1,2)...")
model = ARIMA(train, order=(2, 1, 2))
model_fit = model.fit()
print(model_fit.summary())

# -------------------------------
# STEP 5: Forecast for test period
# -------------------------------
predictions = model_fit.forecast(steps=len(test))
predictions.index = test.index

# -------------------------------
# STEP 6: Evaluate model
# -------------------------------
mse = mean_squared_error(test, predictions)
rmse = sqrt(mse)
mae = mean_absolute_error(test, predictions)
r2 = r2_score(test, predictions)

print("\n📊 Model Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# -------------------------------
# STEP 7: Visualization
# -------------------------------
plt.figure(figsize=(10,5))
plt.plot(train, label='Train Data')
plt.plot(test, label='Actual Sales', color='green')
plt.plot(predictions, label='Predicted Sales', color='red')
plt.title('ARIMA Model - Walmart Weekly Sales Forecast')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.legend()
plt.show()

# -------------------------------
# STEP 8: Save metrics to text file
# -------------------------------
with open("arima_evaluation.txt", "w") as f:
    f.write("ARIMA Model Evaluation\n")
    f.write("=========================\n")
    f.write(f"MSE: {mse:.2f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"MAE: {mae:.2f}\n")
    f.write(f"R² Score: {r2:.2f}\n")
    f.write("\nModel Summary:\n")
    f.write(str(model_fit.summary()))

print("\n✅ Model evaluation saved to 'arima_evaluation.txt'")
