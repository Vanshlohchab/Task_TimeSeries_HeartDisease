"""
Combined Project: Time Series Analysis (Sales Forecasting with ARIMA)
and
Heart Disease Prediction (Logistic Regression)

"""
# %%
# -----------------------
# Imports and Utilities
# -----------------------
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Statsmodels for ARIMA
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# For classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

import os

# Utility to save outputs
OUTPUT_DIR = Path("project_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Simple plotting helper
def show_plot():
    plt.tight_layout()
    plt.show()

# Helper to determine seasonal period from inferred freq
def period_from_freq(freq):
    if freq is None:
        return None
    f = freq.upper()
    # Monthly or yearly
    if f.startswith('M') or f.startswith('A') or 'M' in f or 'A' in f:
        return 12
    # Weekly/daily
    if f.startswith('D') or f.startswith('B'):
        return 7
    if f.startswith('W'):
        return 52
    # Hourly
    if f.startswith('H'):
        return 24
    return None

# %%
# =======================
# Part 1 — Time Series: Sales Forecasting with ARIMA
# =======================

SALES_CSV = "sales.csv"  # must contain columns: Date, Sales

# Auto-generate dummy dataset if file not found
if not Path(SALES_CSV).exists():
    print(f"{SALES_CSV} not found. Generating dummy dataset...")
    dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="D")
    np.random.seed(42)
    # base sales + small trend + weekly seasonality + noise
    base = 100 + np.linspace(0, 20, len(dates))
    weekly = 10 * np.sin(2 * np.pi * (np.arange(len(dates)) % 7) / 7)
    noise = np.random.normal(0, 8, len(dates))
    sales = (base + weekly + noise).astype(int)
    dummy_df = pd.DataFrame({"Date": dates, "Sales": sales})
    dummy_df.to_csv(SALES_CSV, index=False)
    print(f"Dummy {SALES_CSV} created with {len(dummy_df)} rows.")

# Load data
print("\nLoading sales data from:", SALES_CSV)
try:
    sales_df = pd.read_csv(SALES_CSV, parse_dates=["Date"])
except Exception as e:
    raise SystemExit(f"Error loading {SALES_CSV}: {e}")

# Basic checks
if "Sales" not in sales_df.columns:
    raise SystemExit("Expected a column named 'Sales' in sales.csv")

# Prepare series
sales_df = sales_df.copy()
sales_df.set_index("Date", inplace=True)
sales_df = sales_df.sort_index()
# aggregate duplicates
sales_df = sales_df.groupby(sales_df.index).agg({"Sales": "sum"})

print("Data range:", sales_df.index.min(), "to", sales_df.index.max())
print("Number of observations:", len(sales_df))

# Exploratory plot
plt.figure(figsize=(12,4))
plt.plot(sales_df.index, sales_df['Sales'], label='Sales')
plt.title('Sales over time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
show_plot()

# Moving averages
sales_df['MA_7'] = sales_df['Sales'].rolling(window=7, min_periods=1).mean()
sales_df['MA_30'] = sales_df['Sales'].rolling(window=30, min_periods=1).mean()

plt.figure(figsize=(12,4))
plt.plot(sales_df['Sales'], alpha=0.6, label='Sales')
plt.plot(sales_df['MA_7'], label='7-day MA')
plt.plot(sales_df['MA_30'], label='30-day MA')
plt.title('Sales with moving averages')
plt.legend()
show_plot()

# Decomposition if possible
try:
    inferred = pd.infer_freq(sales_df.index)
    print("Inferred frequency:", inferred)
    period = period_from_freq(inferred)
    if period is not None and period > 1 and len(sales_df) >= period*2:
        decomposition = sm.tsa.seasonal_decompose(sales_df['Sales'], model='additive', period=period)
        fig = decomposition.plot()
        fig.set_size_inches(12,8)
        show_plot()
    else:
        print("Skipping seasonal decomposition (insufficient data or unknown frequency).")
except Exception as e:
    print("Decomposition skipped due to error:", e)

# Train/Test split
n_obs = len(sales_df)
if n_obs >= 36:
    test_len = 12
else:
    test_len = max(1, int(np.ceil(n_obs * 0.2)))

train = sales_df['Sales'][:-test_len]
test = sales_df['Sales'][-test_len:]

print(f"Train observations: {len(train)}, Test observations: {len(test)}")

# ARIMA order selection (simple grid)

def select_arima_order(series, p_range=(0,2), d_range=(0,1), q_range=(0,2)):
    best_aic = np.inf
    best_order = None
    best_res = None
    for p in range(p_range[0], p_range[1]+1):
        for d in range(d_range[0], d_range[1]+1):
            for q in range(q_range[0], q_range[1]+1):
                try:
                    mod = sm.tsa.ARIMA(series, order=(p,d,q))
                    res = mod.fit()
                    aic = res.aic
                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p,d,q)
                        best_res = res
                except Exception:
                    continue
    return best_order, best_res

print("Searching for best ARIMA order (this can take time)...")
order, model_fit = select_arima_order(train, p_range=(0,3), d_range=(0,1), q_range=(0,3))
print("Selected order:", order)

if model_fit is None:
    # fallback
    try:
        model_fit = sm.tsa.ARIMA(train, order=(1,1,1)).fit()
        order = (1,1,1)
        print("Fallback to ARIMA(1,1,1)")
    except Exception as e:
        raise SystemExit("Could not fit ARIMA model. Error: " + str(e))

# Forecast test period
pred_test = model_fit.forecast(steps=len(test))

# Evaluation
rmse = np.sqrt(mean_squared_error(test, pred_test))
mape = mean_absolute_percentage_error(test, pred_test)
print(f"Forecast evaluation on test set — RMSE: {rmse:.3f}, MAPE: {mape:.3f}")

# Plot predictions vs actual
plt.figure(figsize=(12,5))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test', marker='o')
plt.plot(test.index, pred_test, label='Predicted', marker='o')
plt.title(f'ARIMA Forecast (order={order}) — Test RMSE={rmse:.2f}, MAPE={mape:.2f}')
plt.legend()
show_plot()

# Retrain on full data and forecast future
horizon = 12
full_model = sm.tsa.ARIMA(sales_df['Sales'], order=order).fit()
future_forecast = full_model.forecast(steps=horizon)

# Build future index
freq = pd.infer_freq(sales_df.index)
if freq is not None:
    try:
        last_date = sales_df.index.max()
        offset = pd.tseries.frequencies.to_offset(freq)
        future_index = pd.date_range(start=last_date + offset, periods=horizon, freq=freq)
    except Exception:
        future_index = range(len(sales_df), len(sales_df) + horizon)
else:
    future_index = range(len(sales_df), len(sales_df) + horizon)

forecast_df = pd.DataFrame({"Forecast": future_forecast}, index=future_index)
forecast_df.to_csv(OUTPUT_DIR / "sales_forecast.csv")
print("Saved forecast CSV to:", OUTPUT_DIR / "sales_forecast.csv")

# Plot history + forecast
plt.figure(figsize=(12,5))
plt.plot(sales_df.index, sales_df['Sales'], label='Historical')
plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', marker='o')
plt.title('Historical Sales and Forecast')
plt.legend()
show_plot()

# Save ARIMA summary
with open(OUTPUT_DIR / "arima_summary.txt", "w") as f:
    f.write(full_model.summary().as_text())
print("Saved ARIMA model summary to:", OUTPUT_DIR / "arima_summary.txt")

# Try to save the fitted ARIMA model using joblib if available
try:
    import joblib
    joblib.dump(full_model, OUTPUT_DIR / "arima_full_model.joblib")
    print("Saved ARIMA model with joblib.")
except Exception as e:
    print("Could not save ARIMA model via joblib:", e)

# %%
# =======================
# Part 2 — Heart Disease Prediction (Logistic Regression)
# =======================

HEART_CSV = "heart_disease.csv"

# Auto-generate dummy dataset if file not found
if not Path(HEART_CSV).exists():
    print(f"{HEART_CSV} not found. Generating dummy dataset...")
    np.random.seed(42)
    n = 500
    age = np.random.randint(25, 80, size=n)
    gender = np.random.choice(["Male", "Female"], size=n)
    chol = np.random.randint(150, 300, size=n)
    bp = np.random.randint(80, 180, size=n)
    # create a somewhat realistic risk score
    risk = (0.03*(age-30) + 0.02*(chol-150) + 0.02*(bp-90))
    prob = 1 / (1 + np.exp(- (risk - 2)))  # logisticize
    heart = (np.random.rand(n) < prob).astype(int)
    df_dummy = pd.DataFrame({
        "Age": age,
        "Gender": gender,
        "Cholesterol": chol,
        "Blood Pressure": bp,
        "Heart Disease": heart
    })
    df_dummy.to_csv(HEART_CSV, index=False)
    print(f"Dummy {HEART_CSV} created with {len(df_dummy)} rows.")

print("\nLoading heart disease dataset from:", HEART_CSV)
try:
    heart_df = pd.read_csv(HEART_CSV)
except Exception as e:
    raise SystemExit(f"Error loading {HEART_CSV}: {e}")

print("Columns found:", heart_df.columns.tolist())

# Clean dataset
heart_df = heart_df.drop_duplicates().dropna().copy()
heart_df.columns = [c.strip() for c in heart_df.columns]

# Detect target column name (flexible)
possible_targets = ["Heart Disease", "Heart_Disease", "heart_disease", "target", "HasDisease"]
target_col = None
for t in possible_targets:
    if t in heart_df.columns:
        target_col = t
        break
if target_col is None:
    # fallback: last column
    target_col = heart_df.columns[-1]
    print(f"No standard target column found — using last column: {target_col}")

# Choose features
features = []
for c in ["Age", "age"]:
    if c in heart_df.columns:
        features.append(c)
for c in ["Cholesterol", "cholesterol"]:
    if c in heart_df.columns:
        features.append(c)
if "Blood Pressure" in heart_df.columns:
    features.append("Blood Pressure")
elif "Blood_Pressure" in heart_df.columns:
    features.append("Blood_Pressure")

# Gender
if "Gender" in heart_df.columns:
    heart_df['Gender_bin'] = heart_df['Gender'].apply(lambda x: 1 if str(x).strip().lower().startswith('m') else 0)
    features.append('Gender_bin')

# If not enough features, take numeric columns except target
if len(features) < 2:
    numeric_cols = heart_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]
    if len(numeric_cols) >= 2:
        features = numeric_cols
    else:
        raise SystemExit("Could not identify enough numeric features for modeling. Please provide a dataset with at least two numeric predictors.")

print("Using features:", features)
print("Target column:", target_col)

X = heart_df[features]
y = heart_df[target_col].copy()

# Normalize target if needed
if y.dtype == object:
    y = y.apply(lambda v: 1 if str(v).strip().lower() in ["yes","y","1","true","t"] else 0)

# Train-test split with stratify if possible
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
except Exception:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_scaled, y_train)

# Predictions
y_pred = clf.predict(X_test_scaled)

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\nClassification results:")
print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"F1-score: {f1:.3f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# Save predictions and report
results_df = X_test.copy()
results_df['y_true'] = y_test.values
results_df['y_pred'] = y_pred
results_df.to_csv(OUTPUT_DIR / "heart_disease_predictions.csv", index=False)

with open(OUTPUT_DIR / "heart_classification_report.txt", "w") as f:
    f.write(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1: {f1:.4f}\n\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n\nClassification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")

# Save model and scaler
try:
    import joblib
    joblib.dump(clf, OUTPUT_DIR / "logistic_model.joblib")
    joblib.dump(scaler, OUTPUT_DIR / "scaler.joblib")
    print("Saved logistic model and scaler to:", OUTPUT_DIR)
except Exception as e:
    print("Could not save logistic model/scaler via joblib:", e)

# %%
# Final summary
print('\nAll done. Files saved in the folder:', OUTPUT_DIR)
print('- Sales forecast: project_outputs/sales_forecast.csv')
print('- ARIMA summary: project_outputs/arima_summary.txt')
print('- Heart disease predictions: project_outputs/heart_disease_predictions.csv')
print('- Heart classification report: project_outputs/heart_classification_report.txt')

print('\nIf you want this as a Jupyter Notebook (.ipynb), a requirements.txt, or a README.md, tell me and I will create them for you.')
