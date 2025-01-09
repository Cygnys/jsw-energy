import os 
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import plot_model

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    # Load the dataset
    data = pd.read_csv("..//datasets//january_weather_energy_data_2020_2025.csv")
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)

    # Feature selection
    features = [
        'energy_consumed_kWh', 'temperature_C', 'humidity_%', 'wind_speed_mps',
        'is_sunny', 'cloud_cover_%', 'solar_irradiance_Wm2',
        'air_density_kgm3', 'precipitation_mm', 'runoff_coefficient'
    ]
    data = data[features]

    # Normalize data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return data, scaler, scaled_data

# Prepare sequences for LSTM
def create_sequences(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback, :])
        y.append(data[i + lookback, 0])  # 'energy_consumed_kWh' is the target
    return np.array(X), np.array(y)

# Build and train LSTM model
@st.cache_resource
def build_and_train_model(scaled_data, lookback, train_split=0.8):
    # Prepare sequences
    X, y = create_sequences(scaled_data, lookback)

    # Train-test split
    train_size = int(train_split * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build the model
    model = Sequential([
        LSTM(64, return_sequences=True,activation='relu' , input_shape=(lookback, X.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)  # Output layer for regression
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    return model, X_test, y_test

# Streamlit app
st.title("Energy Consumption Prediction with LSTM")
st.write("Predict energy consumption based on weather and time data using an LSTM model.")

# Load data and scaler
data, scaler, scaled_data = load_and_preprocess_data()

# Define the lookback period (15-minute intervals, 96 steps = 24 hours)
lookback = 96

# Build and train the model
model, X_test, y_test = build_and_train_model(scaled_data, lookback)

# User inputs for prediction
selected_date = st.date_input("Select a date", pd.Timestamp.now().date())
selected_time = st.time_input("Select a time")
user_timestamp = pd.Timestamp.combine(selected_date, selected_time)

if st.button("Predict"):
    if user_timestamp not in data.index:
        st.error("The selected timestamp is not available in the dataset.")
    else:
        # Find index of user timestamp
        idx = data.index.get_loc(user_timestamp) - lookback

        if idx < 0:
            st.error("Not enough historical data to make predictions for the selected time.")
        else:
            # Prepare input sequence
            input_sequence = scaled_data[idx:idx + lookback, :].reshape(1, lookback, scaled_data.shape[1])

            # Make prediction
            try:
                prediction = model.predict(input_sequence)

                # Inverse transform prediction
                n_features = scaled_data.shape[1]
                prediction_padded = np.concatenate((prediction, np.zeros((1, n_features - 1))), axis=1)
                predicted_energy = scaler.inverse_transform(prediction_padded)[:, 0][0]

                st.success(f"Predicted Energy Consumption: {predicted_energy:.2f} kWh")

                # Visualize results
                st.write("Historical Data for Reference:")
                st.line_chart(data.loc[user_timestamp - pd.Timedelta(hours=1):user_timestamp, "energy_consumed_kWh"])

                st.write("Predicted Data:")
                st.line_chart(data.loc[user_timestamp: user_timestamp + pd.Timedelta(hours=1), "energy_consumed_kWh"])

                # Combine predicted and actual data for comparison
                actual_data = data.loc[user_timestamp: user_timestamp + pd.Timedelta(hours=1), "energy_consumed_kWh"]
                predicted_series = pd.Series(
                    [predicted_energy] * len(actual_data),
                    index=actual_data.index,
                    name="Predicted"
                )
                st.write("Predicted vs Actual:")
                st.line_chart(pd.concat([actual_data, predicted_series], axis=1))
                
                #mean absolute error and mean squared error 
                st.write("Mean Absolute Error (MAE):", mean_absolute_error(actual_data, predicted_series))
                st.write("Mean Squared Error (MSE):", mean_squared_error(actual_data, predicted_series))

                # Save the model weights
                model.save_weights("model.weights.h5")

                # Save the model summary
                # with open("model_summary.txt", "w", encoding="utf-8") as f:
                #     model.summary(print_fn=lambda x: f.write(x + "\n"))

                # Save and display the model structure
                # plot_path = os.path.join(os.getcwd(), "model_structure.png")
                # try:
                #     plot_model(model, to_file=plot_path, show_shapes=True, show_layer_names=True)
                #     if os.path.exists(plot_path):
                #         st.image(plot_path, caption="Model Structure")
                #     else:
                #         st.warning("Model structure plot could not be generated.")
                # except Exception as e:
                #     st.error(f"Error generating model structure plot: {e}")

            except Exception as e:
                st.error(f"Prediction error: {e}")
