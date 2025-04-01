import os
import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Define cache directory
cache_dir = "/content/f1_cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
fastf1.Cache.enable_cache(cache_dir)

# Load 2024 Australian GP race session
session_2024 = fastf1.get_session(2024, 3, 'R')
session_2024.load()

# Extract weather data with error handling
def get_weather_features(session):
    weather_data = session.weather_data
    features = {}
    
    # Available weather metrics may vary by session
    available_metrics = {
        'AirTemp': ('AirTemp', 20),        # Default 20°C if missing
        'Humidity': ('Humidity', 50),      # Default 50% if missing
        'Rainfall': ('Rainfall', False),   # Default dry if missing
        'TrackTemp': ('TrackTemp', 30)     # Default 30°C if missing
    }
    
    for feature, (col, default) in available_metrics.items():
        if col in weather_data.columns:
            if col == 'Rainfall':
                features[feature] = int(weather_data[col].any())
            else:
                features[feature] = weather_data[col].mean()
        else:
            features[feature] = default
            print(f"Warning: {col} not found in weather data, using default {default}")
    
    return features

weather_features = get_weather_features(session_2024)

# Extract lap times
laps_2024 = session_2024.laps[["Driver", "LapTime"]].copy()
laps_2024.dropna(subset=["LapTime"], inplace=True)
laps_2024["LapTime (s)"] = laps_2024["LapTime"].dt.total_seconds()

# 2025 Qualifying data
qualifying_2025 = pd.DataFrame({
    "Driver": [
        "Lando Norris", "Oscar Piastri", "Max Verstappen", "George Russell",
        "Yuki Tsunoda", "Alexander Albon", "Charles Leclerc", "Lewis Hamilton",
        "Pierre Gasly", "Carlos Sainz", "Lance Stroll", "Fernando Alonso"],
    "QualifyingTime (s)": [
        75.096, 75.103, 75.481, 75.546, 75.670,
        75.737, 75.753, 75.973, 75.980, 76.662, 76.4, 76.51
    ]
})

# Fix driver name typos in mapping
driver_mapping = {
    "Lando Norris": "NOR", "Oscar Piastri": "PIA", "Max Verstappen": "VER", 
    "George Russell": "RUS", "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB", 
    "Charles Leclerc": "LEC", "Lewis Hamilton": "HAM", "Pierre Gasly": "GAS", 
    "Carlos Sainz": "SAI", "Lance Stroll": "STR", "Fernando Alonso": "ALO"
}

qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)

# Merge with 2024 data and add weather features
merged_data = qualifying_2025.merge(
    laps_2024, 
    left_on="DriverCode", 
    right_on="Driver", 
    how="left"
).dropna()

# Add weather features to each row
for feature, value in weather_features.items():
    merged_data[feature] = value

# Feature engineering
merged_data["LapTime"] = merged_data["LapTime"].dt.total_seconds()

# Prepare features and target
features = [
    "QualifyingTime (s)", 
    "AirTemp", 
    "Humidity", 
    "Rainfall",
    "TrackTemp"
]
X = merged_data[features]
y = merged_data["LapTime"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=39
)
model = GradientBoostingRegressor(
    n_estimators=100, 
    learning_rate=0.1, 
    random_state=39
)
model.fit(X_train, y_train)

# Predict for 2025 (with weather features)
qualifying_2025_features = qualifying_2025.copy()
for feature, value in weather_features.items():
    qualifying_2025_features[feature] = value

predicted_lap_times = model.predict(qualifying_2025_features[features])
qualifying_2025["PredictedRaceTime"] = predicted_lap_times

# Rank drivers
qualifying_2025 = qualifying_2025.sort_values(by="PredictedRaceTime")
qualifying_2025["Position"] = range(1, len(qualifying_2025) + 1)

# Results
print("\nPredicted 2025 Australian GP Winner (with Weather Factors)\n")
print(qualifying_2025[["Driver", "PredictedRaceTime", "Position"]])
print("\nWeather Conditions During Race:")
print(f"- Air Temperature: {weather_features['AirTemp']:.1f}°C")
print(f"- Track Temperature: {weather_features['TrackTemp']:.1f}°C")
print(f"- Humidity: {weather_features['Humidity']:.1f}%")
print(f"- Rain: {'Yes' if weather_features['Rainfall'] else 'No'}")

# Evaluation
y_pred = model.predict(X_test)
print(f"\nModel Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")