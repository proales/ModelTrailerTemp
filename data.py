import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from joblib import dump, load
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load and prepare the data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df

# Feature engineering
def create_features(df):
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week

    df['battery_state_lag1'] = df['battery_state'].shift(1)
    df['battery_state_lag2'] = df['battery_state'].shift(2)

    return df

# Define features
FEATURES = ['battery_state_lag1', 'battery_state_lag2', 'temperature', 'humidity', 'wind_speed',
            'hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear']

# Prepare features and target
def prepare_data(df):
    X = df[FEATURES]
    y = df['battery_state']
    return X, y

# Define the DNN model
def create_dnn_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# Define the ensemble model
class EnsembleModel:
    def __init__(self, input_dim):
        self.rf = RandomForestRegressor(n_estimators=100, random_state=42)
        self.gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.svr = SVR(kernel='rbf')
        self.dnn = create_dnn_model(input_dim)
        self.imputer = SimpleImputer(strategy='mean')

    def fit(self, X, y):
        X_imputed = self.imputer.fit_transform(X)
        self.rf.fit(X_imputed, y)
        self.gb.fit(X_imputed, y)
        self.svr.fit(X_imputed, y)
        self.dnn.fit(X_imputed, y, epochs=50, batch_size=32, verbose=0)

    def predict(self, X):
        X_imputed = self.imputer.transform(X)
        rf_pred = self.rf.predict(X_imputed)
        gb_pred = self.gb.predict(X_imputed)
        svr_pred = self.svr.predict(X_imputed)
        dnn_pred = self.dnn.predict(X_imputed).flatten()
        return (rf_pred + gb_pred + svr_pred + dnn_pred) / 4

# Train the model or load if it exists
def train_or_load_model(X, y, model_path='ensemble_model.joblib', scaler_path='scaler.joblib'):
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print("Loading existing model and scaler...")
        model = load(model_path)
        scaler = load(scaler_path)
        X_scaled = scaler.transform(X)
        return model, scaler, X_scaled, y
    else:
        print("Training new ensemble model...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = EnsembleModel(input_dim=X_train_scaled.shape[1])
        model.fit(X_train_scaled, y_train)

        # Save the model and scaler
        dump(model, model_path)
        dump(scaler, scaler_path)

        return model, scaler, X_test_scaled, y_test

# Evaluate the model
def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")

# Make predictions
def predict_battery_state(model, scaler, current_state, weather_data):
    input_data = pd.DataFrame(columns=FEATURES)
    input_data.loc[0, 'battery_state_lag1'] = current_state
    input_data.loc[0, 'battery_state_lag2'] = current_state  # Assuming we don't have the previous state
    input_data.loc[0, 'temperature'] = weather_data['temperature']
    input_data.loc[0, 'humidity'] = weather_data['humidity']
    input_data.loc[0, 'wind_speed'] = weather_data['wind_speed']

    current_time = pd.Timestamp.now()
    input_data.loc[0, 'hour'] = current_time.hour
    input_data.loc[0, 'dayofweek'] = current_time.dayofweek
    input_data.loc[0, 'quarter'] = current_time.quarter
    input_data.loc[0, 'month'] = current_time.month
    input_data.loc[0, 'year'] = current_time.year
    input_data.loc[0, 'dayofyear'] = current_time.dayofyear
    input_data.loc[0, 'dayofmonth'] = current_time.day
    input_data.loc[0, 'weekofyear'] = current_time.isocalendar()[1]

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return prediction[0]

# Main execution
if __name__ == "__main__":
    # Load and prepare data
    df = load_data("fake_battery_weather_data.csv")
    df = create_features(df)
    X, y = prepare_data(df)

    # Train or load the model
    model, scaler, X_test_scaled, y_test = train_or_load_model(X, y)

    # Evaluate the model
    evaluate_model(model, X_test_scaled, y_test)

    # Example prediction
    current_state = 0.75  # Current battery state (75%)
    weather_data = {
        'temperature': 25,
        'humidity': 60,
        'wind_speed': 10
    }
    predicted_state = predict_battery_state(model, scaler, current_state, weather_data)
    print(f"Predicted battery state: {predicted_state:.2f}")