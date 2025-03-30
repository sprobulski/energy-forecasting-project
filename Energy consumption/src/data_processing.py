import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pandas.tseries.holiday import USFederalHolidayCalendar


# The function get_weather() fetches hourly temperature data for the specified location and date range.
# It handles the API's record limit by breaking the date range into smaller chunks and combines the results into a single DataFrame.

def get_weather(api_key, location, start_date, end_date, unit_group="metric", include="hours", content_type="json"):

    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Calculate the number of days per query (to stay under 1,000 records)
    days_per_query = 41  # 41 days * 24 hours = 984 records < 1,000

    all_hourly_data = []

    # Loop through the date range in chunks
    current_start = start_date
    while current_start <= end_date:
        # Calculate the end date for this chunk
        current_end = min(current_start + timedelta(days=days_per_query - 1), end_date)

        start_str = current_start.strftime("%Y-%m-%d")
        end_str = current_end.strftime("%Y-%m-%d")

        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{start_str}/{end_str}?unitGroup={unit_group}&include={include}&key={api_key}&contentType={content_type}"

        print(f"Fetching data for {start_str} to {end_str}...")
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
        else:
            print(f"Error for {start_str} to {end_str}: {response.status_code}, {response.text}")
            continue  

        # Extract hourly temperature data
        hourly_data = []
        for day in data["days"]:
            for hour in day["hours"]:
                timestamp = f"{day['datetime']}T{hour['datetime']}"
                hourly_data.append({
                    "period": timestamp,
                    "Temperature": hour["temp"]
                })

        # Add this chunk to the overall list
        all_hourly_data.extend(hourly_data)

        # Move to the next chunk
        current_start = current_end + timedelta(days=1)


    weather_data = pd.DataFrame(all_hourly_data)
    weather_data["period"] = pd.to_datetime(weather_data["period"])



    return weather_data


# The function get_respondent_data() retrieves hourly electricity data for a specific respondent from the EIA API.
# It constructs the API URL with the provided parameters and returns the data as a DataFrame.

def get_respondent_data(respondent,api_key,start_date,end_date):
    base_url = f"https://api.eia.gov/v2/electricity/rto/region-data/data/?api_key={api_key}&frequency=hourly&data[0]=value&start={start_date}&end={end_date}&sort[0][column]=period&sort[0][direction]=desc&facets[type][]=D&facets[respondent][]={respondent}"
    respondent = respondent
    api_key = api_key  # Replace with your key
    response = requests.get(base_url)
    json_data = response.json()
    data_raw = pd.DataFrame(json_data['response']['data'])
    if response.status_code != 200:
        print(f"Error: {response.status_code}, {json_data}")
        return None
    else:
        return data_raw
    

# The function merge_datasets() merges the raw energy data with weather data and adds a holiday feature.
# It processes the raw data by dropping unnecessary columns, converting the period to datetime, and sorting it.

def merge_datasets(raw_data, weather_data,start_date, end_date):
    cal = USFederalHolidayCalendar()

    raw_data = raw_data.drop(['type','value-units','type-name','respondent-name'], axis = 1)
    raw_data['period'] = pd.to_datetime(raw_data['period'])
    raw_data = raw_data.sort_values(by='period', ascending=True)
    raw_data['value'] = pd.to_numeric(raw_data['value'], errors='coerce')

    holidays = cal.holidays(start=start_date, end=end_date)
    holiday_dates = holidays.date  


    raw_data['IsHoliday'] = raw_data['period'].dt.date.isin(holiday_dates).astype(int)
    data = raw_data.merge(weather_data, on='period', how="left")

    data = data[['period', 'value','Temperature','IsHoliday']].rename(columns={'period': 'Datetime', 'value': 'Energy_MW'})
    data.set_index('Datetime', inplace=True)

    return data


# The function engeneer_features_xgboost() creates additional features for the energy consumption modeling.
# It extracts time-based features (hour, day, month, day of the week) and creates sine and cosine transformations for cyclic features.
# It also generates lagged features and rolling statistics for the energy consumption data.

def engeneer_features_xgboost(df):
    df['Hour'] = df.index.hour
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['DayOfWeek'] = df.index.dayofweek
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    df['Day_sin'] = np.sin(2 * np.pi * df['Day'] / 31)
    df['Day_cos'] = np.cos(2 * np.pi * df['Day'] / 31)
    for lag in [1, 2, 6, 24, 48, 168, 336, 720]:
        df[f'Lag{lag}'] = df['Energy_MW'].shift(lag)
    df["RollingMean_24h"] = df["Energy_MW"].rolling(window=24).mean()
    df["RollingStd_24h"] = df["Energy_MW"].rolling(window=24).std()

    df.drop(['Hour', 'Day','DayOfWeek'], axis=1, inplace=True)
    df.dropna(inplace=True)

    return df