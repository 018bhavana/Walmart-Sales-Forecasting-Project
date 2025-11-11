import pandas as pd
import plotly.graph_objects as go

# Load dataset
df = pd.read_csv("walmart_10000.csv")
df['ds'] = pd.to_datetime(df['ds'])
df = df.sort_values('ds')

# Load predictions
try:
    prophet_pred = pd.read_csv("prophet_predictions.csv")
    arima_pred = pd.read_csv("arima_predictions.csv")
    lstm_pred = pd.read_csv("lstm_predictions.csv")
    xgb_pred = pd.read_csv("xgboost_predictions.csv")
except FileNotFoundError as e:
    print("❌ Missing prediction file:", e)
    exit()

# Plot interactive comparison
fig = go.Figure()

# Actual
fig.add_trace(go.Scatter(
    x=df['ds'][-len(prophet_pred):],
    y=df['y'][-len(prophet_pred):],
    mode='lines',
    name='Actual Sales',
    line=dict(color='black', width=3)
))

# Each Model
fig.add_trace(go.Scatter(x=df['ds'][-len(prophet_pred):], y=prophet_pred['yhat'], mode='lines', name='Prophet', line=dict(dash='dash')))
fig.add_trace(go.Scatter(x=df['ds'][-len(arima_pred):], y=arima_pred['forecast'], mode='lines', name='ARIMA', line=dict(dash='dot')))
fig.add_trace(go.Scatter(x=df['ds'][-len(lstm_pred):], y=lstm_pred['forecast'], mode='lines', name='LSTM', line=dict(dash='dashdot')))
fig.add_trace(go.Scatter(x=df['ds'][-len(xgb_pred):], y=xgb_pred['forecast'], mode='lines', name='XGBoost', line=dict(dash='solid')))

# Layout
fig.update_layout(
    title="📈 Walmart Sales Forecast Comparison (Actual vs. Predicted)",
    xaxis_title="Date",
    yaxis_title="Sales",
    template="plotly_white",
    legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0)')
)

fig.show()
