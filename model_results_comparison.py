import pandas as pd

# 1️⃣ Load the CSV
df = pd.read_csv("walmart_10000.csv")

# 2️⃣ Strip spaces and lowercase column names
df.columns = [c.strip().lower() for c in df.columns]

# 3️⃣ Identify date and sales columns dynamically
date_col = [c for c in df.columns if 'date' in c][0]      # column containing 'date'
sales_col = [c for c in df.columns if 'sales' in c][0]    # column containing 'sales'

# 4️⃣ Rename to Prophet/XGBoost expected names
df.rename(columns={date_col: 'ds', sales_col: 'y'}, inplace=True)

# 5️⃣ Convert 'ds' to datetime
df['ds'] = pd.to_datetime(df['ds'], errors='coerce')

# 6️⃣ Drop rows where date conversion failed
df = df.dropna(subset=['ds'])

# 7️⃣ Sort by date
df = df.sort_values('ds')

# ✅ Check first few rows
print("Columns after fix:", df.columns.tolist())
print(df.head())
