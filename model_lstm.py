# model_lstm.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# STEP 1: Load dataset
print("✅ Loading Walmart dataset...")
data = pd.read_csv("walmart_10000.csv")

# If 'date' column missing, create one
if 'date' not in data.columns:
    print("⚠️ No 'date' column found — generating synthetic date range...")
    data['date'] = pd.date_range(start='2015-01-01', periods=len(data), freq='D')

# Set date index
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# STEP 2: Use Weekly_Sales as target
sales_col = [c for c in data.columns if c.lower() == 'weekly_sales'][0]
sales = data[sales_col].values.reshape(-1, 1)

# STEP 3: Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
sales_scaled = scaler.fit_transform(sales)

# STEP 4: Create sequences for LSTM
def create_dataset(dataset, look_back=7):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 7
X, y = create_dataset(sales_scaled, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# STEP 5: Train/test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# STEP 6: Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

print("✅ Training LSTM model...")
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# STEP 7: Predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# STEP 8: Inverse transform predictions
train_pred = scaler.inverse_transform(train_pred)
test_pred = scaler.inverse_transform(test_pred.reshape(-1, 1))
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# STEP 9: Evaluation Metrics
mse = mean_squared_error(y_test_actual, test_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_actual, test_pred)
r2 = r2_score(y_test_actual, test_pred)

# STEP 10: Save evaluation
with open("lstm_evaluation.txt", "w") as f:
    f.write(f"LSTM Model Evaluation\n")
    f.write(f"MSE: {mse:.2f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"MAE: {mae:.2f}\n")
    f.write(f"R² Score: {r2:.2f}\n")

print("✅ Evaluation saved to 'lstm_evaluation.txt'")
print(f"MSE={mse:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.2f}")

# STEP 11: Plot predictions
plt.figure(figsize=(10,6))
plt.plot(y_test_actual, label='Actual Sales')
plt.plot(test_pred, label='Predicted Sales', color='red')
plt.title('LSTM - Actual vs Predicted Weekly Sales')
plt.xlabel('Time')
plt.ylabel('Weekly Sales')
plt.legend()
plt.tight_layout()
plt.show()
