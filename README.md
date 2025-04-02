# ERCOT Energy Consumption Analysis and Forecasting

## Project Overview

This project analyzes and forecasts energy consumption in the ERCOT (Electric Reliability Council of Texas) region using historical data from July to December 2023. The goal is to understand patterns in energy usage (`Energy_MW`) influenced by factors like temperature and holidays, and to build predictive models for short-term forecasting. Two machine learning models are implemented: **XGBoost** (a tree-based model) and **LSTM** (a deep learning model for sequential data). The project includes data collection, exploratory data analysis (EDA), feature engineering, model training, and evaluation.

This work demonstrates skills in:
- Time series analysis and forecasting
- Data preprocessing and feature engineering
- Machine learning with XGBoost and LSTM
- Python programming (pandas, scikit-learn, TensorFlow, statsmodels)
- Data visualization (matplotlib, seaborn)

---

## Dataset

The dataset combines:
- **Energy Consumption Data**: Hourly `Energy_MW` for ERCOT, sourced via API from EIA API.
- **Weather Data**: Hourly temperature in Texas, sourced via API from Visual Crossing Weather API.
- **Holiday Data**: US federal holidays, generated using `pandas.tseries.holiday.USFederalHolidayCalendar`.

The final dataset (`merged_data_ercot_2023.csv`) spans July 1, 2023, to December 31, 2023, and includes:
- `Datetime`: Timestamp (hourly)
- `Energy_MW`: Energy consumption in megawatts
- `Temperature`: Temperature in °C
- `IsHoliday`: Binary indicator (1 for holidays, 0 otherwise)

---

## Methodology

### 1. Data Collection
- Fetched hourly energy consumption (`Energy_MW`) and temperature data for ERCOT from July to December 2023.
- Added a binary `IsHoliday` feature using `USFederalHolidayCalendar`, fixed to mark all hours of a holiday day (not just midnight).

### 2. Exploratory Data Analysis (EDA)
- **Descriptive Statistics**: Summarized the dataset with shape, column info, data types, and basic statistics. Confirmed no missing values.
- **Time Series Visualization**: Plotted `Energy_MW` over time to identify trends and fluctuations.
- **Distribution Analysis**: Examined distributions of `Energy_MW` and `Temperature` using summary statistics .
- **Correlation Analysis**: Found a positive correlation between `Energy_MW` and `Temperature`, supporting its use as a predictor.
- **Seasonal Patterns**:
  - Analyzed hourly patterns by month, revealing afternoon peaks.
  - Compared average hourly energy consumption on holidays vs. non-holidays, showing lower usage on holidays.
- **Time Series Decomposition**: Decomposed `Energy_MW` into trend, daily seasonal, and residual components using an additive model.

### 3. Feature Engineering
- **Lags**: Added lagged `Energy_MW` values (e.g., `Lag1`, `Lag24`, `Lag168`) to capture temporal dependencies.
- **Cyclic Features**: Encoded `Hour`, `DayOfWeek`, and `Day` as sine/cosine pairs (e.g., `Hour_sin`, `Hour_cos`) to model periodicity.
- **Rolling Statistics**: Added 24-hour rolling mean and standard deviation.
- **External Features**: Included `Temperature` and `IsHoliday`.

### 4. Modeling
- **XGBoost**:
  - Tree-based model, insensitive to feature scaling.
  - Trained on features mentioned earlier, cyclic encodings, and `Temperature`.
- **LSTM**:
  - Deep learning model for sequential data.
  - Used scaled features and reshaped data for time step sequences.
- **Evaluation**: Models were evaluated using metrics like RMSE.

---

## Key Findings
- **Seasonality**: Strong daily seasonality with afternoon peaks, especially in summer months (July/August), driven by higher temperatures.
- **Holiday Effects**: Lower energy consumption on holidays, with a statistically significant difference between holidays and non-holidays, supporting the `IsHoliday` feature.
- **Temperature Impact**: Positive correlation between `Temperature` and `Energy_MW`, with higher usage during hotter periods.
- **Model Performance**:
  - XGBoost captured short-term patterns effectively, benefiting from engineered features.
  - LSTM excelled at learning sequential dependencies, particularly daily cycles.
  - Achived RMSE: 1012.10 for XGBOOST and 714.31 for LSTM

---

## Future Improvements
- **Expand Data**: Include more years (e.g., 2018-2023) to analyze yearly seasonality and improve model robustness.
- **Additional Features**: Incorporate more weather variables (e.g., humidity, solar radiation) or economic indicators (e.g., electricity prices).
- **Hyperparameter Tuning**: Further optimize XGBoost and LSTM hyperparameters using grid search or Bayesian optimization.
- **Model Comparison**: Test additional models (e.g., Prophet, ARIMA) for benchmarking.

---

## Usage
To explore the project:

- Run `01_data_collection.ipynb` to fetch and merge data (requires API keys).
- Run `01b_exploratory_data_analysis.ipynb` to visualize patterns and validate features.
- Train models with `02_xgboost_training.ipynb` (XGBoost) and `03_lstm_training.ipynb` (LSTM).

---

## Dependencies
See `requirements.txt` for a full list. Key libraries include:

- `pandas`, `numpy`: Data manipulation
- `matplotlib`, `seaborn`: Visualization
- `statsmodels`: Time series decomposition and analysis
- `scikit-learn`, `xgboost`: Machine learning
- `tensorflow`: Deep learning (LSTM)
- `requests`: API calls
