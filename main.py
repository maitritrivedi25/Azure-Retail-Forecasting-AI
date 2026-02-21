import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 1. MOCK DATA GENERATION
# Simulating 3 years of retail sales data for the project
def generate_mock_data():
    print("[INFO] Generating 3 years of historical retail sales data...")
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(1095)]
    
    # Simulating seasonality + noise
    base_sales = 50
    seasonal_pattern = [np.sin(2 * np.pi * i / 365) * 20 for i in range(1095)]
    noise = np.random.normal(0, 5, 1095)
    sales = [int(max(0, base_sales + s + n)) for s, n in zip(seasonal_pattern, noise)]
    
    df = pd.DataFrame({'ds': dates, 'y': sales})
    return df

# 2. PROPHET FORECASTING MODEL
def train_and_forecast(df):
    print("[INFO] Initializing Prophet Model for Demand Forecasting...")
    model = Prophet(yearly_seasonality=True, daily_seasonality=False)
    model.fit(df)
    
    # Predicting next 30 days
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    print("[SUCCESS] Forecasting complete for the next 30 days.")
    return model, forecast

# 3. FINANCIAL RISK SCORING (BBA Perspective)
def calculate_financial_risk(current_stock, item_value, predicted_demand):
    """
    Business Logic: Risk Score = (Excess Stock * Capital Value) / Predicted Demand
    Higher score indicates 'Dead Stock' risk (Capital is locked).
    """
    excess_stock = max(0, current_stock - predicted_demand)
    risk_score = (excess_stock * item_value) / max(1, predicted_demand)
    return round(risk_score, 2)

# 4. MAIN EXECUTION (Simulation)
if __name__ == "__main__":
    # Generate Data
    sales_data = generate_mock_data()
    
    # Run Forecast
    model, forecast_results = train_and_forecast(sales_data)
    next_month_demand = forecast_results['yhat'].tail(30).sum()
    
    # Financial Simulation
    STOCK_LEVEL = 1200
    UNIT_PRICE = 45.0  # $
    
    risk = calculate_financial_risk(STOCK_LEVEL, UNIT_PRICE, next_month_demand)
    
    print("\n" + "="*50)
    print("RETAIL ANALYTICS SUMMARY")
    print("="*50)
    print(f"Projected Demand (Next 30 Days): {int(next_month_demand)} units")
    print(f"Current Stock Level: {STOCK_LEVEL} units")
    print(f"Financial Risk Score: {risk}")
    
    if risk > 50:
        print("ACTION REQUIRED: Liquidation strategy recommended for dead stock.")
    else:
        print("STATUS: Inventory levels healthy. Liquidity maintained.")
    print("="*50)
