# model_prophet.py

import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

model_name = "Prophet"

print("✅ Loading Walmart dataset...")
data = pd.read_csv("walmart_10000.csv")

# Ensure column names are lowercase for consistency
data.columns = [c.lower() for c in data.columns]

# Fix the date column
if 'date' not in data.columns:
    data['date'] = pd.date_range(start="2015-01-01", periods=len(data), freq='D')

# Prepare data for Prophet
df = pd.DataFrame({
    'ds': pd.to_datetime(data['date']),
    'y': data['weekly_sales']
})

print("✅ Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print(df.head())

# Train-test split
train_size = int(len(df) * 0.8)
train = df.iloc[:train_size]
test = df.iloc[train_size:]

# Fit Prophet model
model = Prophet(daily_seasonality=True)
model.fit(train)

# Make future dataframe and forecast
future = model.make_future_dataframe(periods=len(test))
forecast = model.predict(future)

# Get predictions corresponding to test period
predictions = forecast['yhat'].iloc[-len(test):].values
test_values = test['y'].values

# Evaluate
mse = mean_squared_error(test_values, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_values, predictions)
r2 = r2_score(test_values, predictions)

# --- Save consistent results ---
results = f"""
{model_name} Model Results:
MSE: {mse}
RMSE: {rmse}
MAE: {mae}
R2: {r2}
--------------------------------------------
"""
with open("model_results.txt", "a") as f:
    f.write(results)

print("\n📊 Prophet Model Performance:")
print(results)
print("✅ Results saved in 'model_results.txt'")
