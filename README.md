# Store Item Demand Forecasting

## Objective
Forecast future product demand using historical store-level sales data and machine-learning techniques.

## Business Problem
Accurate demand forecasts can help retailers plan inventory, reduce stock-outs and avoid excessive inventory.

## Dataset
Store Item Demand Forecasting dataset from Kaggle. The dataset contains daily sales information for 10 stores and 50 products over 5 years (2013-2017).

## Tools
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

## Analysis
1. Data cleaning
2. Time-series exploratory analysis
3. Store and item analysis
4. Seasonality analysis
5. Lag feature engineering
6. Rolling statistics
7. Time-based validation
8. Forecasting
9. Error analysis

## Models
- Naive baseline
- Moving average
- Linear Regression
- Random Forest
- XGBoost

## Evaluation
Models were evaluated using time-based validation and SMAPE (Symmetric Mean Absolute Percentage Error).

## Key Business Insights
- **Seasonality**: Sales exhibit strong yearly seasonality, typically peaking during the summer months (July) and hitting a low in January.
- **Weekly Patterns**: Weekends consistently generate higher sales volume across all stores and items compared to weekdays.
- **Store Performance**: Store 2 is consistently the top-performing store, indicating high foot traffic or better location dynamics.
- **Product Demand**: Item 15 and Item 28 are the highest-selling products across all stores, meaning inventory for these must be strictly monitored to prevent stock-outs.
- **Forecasting Utility**: Our XGBoost model robustly captures these trends, significantly outperforming simple baseline methods and providing actionable accuracy for 3-month demand planning.

## Conclusion
The analysis demonstrates how historical sales data can be used to forecast future demand and support inventory planning decisions. The inclusion of temporal lag features and rolling aggregations significantly improves model predictive power.
