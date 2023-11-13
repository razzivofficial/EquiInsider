# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# Function to load or create the model
def load_or_create_model():
    try:
        model = load_model('keras_model.h5')
    except:
        # Create and train the model if not already saved
        model = create_and_train_model()  
    return model

# Function to create and train the model (Fill in this part with your actual model architecture and training logic)
def create_and_train_model():
    model = Sequential()
    # Replace the following lines with your actual model architecture and training logic
    model.add(LSTM(units=50, return_sequences=True, input_shape=(100, 1)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Replace data_training with your actual training data
    data_training = np.random.rand(1000, 1)
    data_training = np.reshape(data_training, (data_training.shape[0], data_training.shape[1], 1))

    model.fit(data_training, data_training, epochs=10, batch_size=32)

    # Save the trained model
    model.save('keras_model.h5')

    return model

# Function to scale and prepare data for predictions
def prepare_data_for_prediction(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    return scaler, data_scaled

# Function to make future predictions
def make_future_predictions(model, data, scaler, days):
    predictions = []

    for i in range(days):
        # Use the last 100 data points for prediction
        x_input = data[-100:]
        x_input = np.reshape(x_input, (1, 100, 1))
        
        # Make a prediction
        y_pred = model.predict(x_input)
        
        # Append the prediction to the result and update input for the next iteration
        predictions.append(y_pred[0, 0])
        x_input = np.append(x_input, y_pred)[1:]

    # Inverse transform the predictions to the original scale
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)[:, 0]
    return predictions

# Streamlit app
st.title("EquiInsider")
user_input = st.text_input('Enter stock name from NSE'+'.NS', 'RELIANCE.NS')
df = yf.download(user_input, start='2013-01-01', end=datetime.now().strftime('%Y-%m-%d'))

# Display raw data
if st.checkbox('Show Raw Data'):
    st.subheader('Raw Data')
    st.write(df)

# Describing the data
st.subheader('Data from 01-04-2013 to Today')
st.write(df.describe())

# Interactive widget for selecting time range
start_date = st.date_input("Select start date", datetime(2013, 1, 1))
end_date = st.date_input("Select end date", datetime.now())

# Filter data based on selected time range
filtered_data = df.loc[start_date:end_date]

# Display closing price vs time chart with 100 MA
st.subheader('Closing Price vs Time chart with 100 MA')
ma100 = filtered_data.Close.rolling(100).mean()
fig1, ax1 = plt.subplots(figsize=(12, 8))
ax1.plot(ma100, label='100 MA')
ax1.plot(filtered_data['Close'], label='Closing Price')
ax1.legend()
st.pyplot(fig1)

# Display closing price vs time chart with 100 MA & 200 MA
st.subheader('Closing Price vs Time chart with 100 MA & 200 MA')
ma200 = filtered_data.Close.rolling(200).mean()
fig2, ax2 = plt.subplots(figsize=(12, 8))
ax2.plot(ma100, 'r', label='100 MA')
ax2.plot(ma200, 'g', label='200 MA')
ax2.plot(filtered_data['Close'], 'b', label='Closing Price')
ax2.legend()
st.pyplot(fig2)

# Divide training and testing data
data_training = pd.DataFrame(filtered_data['Close'][0:int(len(filtered_data) * 0.70)])
data_testing = pd.DataFrame(filtered_data['Close'][int(len(filtered_data) * 0.70):int(len(filtered_data))])

# Display data shapes
st.subheader('Training and Testing Data Shapes')
st.write(f'Training Data Shape: {data_training.shape}')
st.write(f'Testing Data Shape: {data_testing.shape}')

# Load or create the model
model = load_or_create_model()

# Prepare data for prediction
scaler, data_scaled = prepare_data_for_prediction(data_training)

# Make future predictions
future_days = st.slider("Select number of days for future predictions", min_value=1, max_value=30, value=30)
future_predictions = make_future_predictions(model, data_scaled, scaler, future_days)

# Display predicted prices in list format
st.subheader(f'Predicted Prices for the Next {future_days} Days:')
predicted_prices_df = pd.DataFrame({'Date': pd.date_range(start=end_date, periods=future_days), 'Predicted Price': future_predictions})
st.write(predicted_prices_df)

# Display predicted vs actual graph
st.subheader('Prediction vs Actual value graph')
fig3, ax3 = plt.subplots(figsize=(12, 8))
ax3.plot(data_testing.index, data_testing['Close'], 'b', label='Actual Price')
ax3.plot(predicted_prices_df['Date'], predicted_prices_df['Predicted Price'], 'r', label='Predicted Price')
ax3.set_xlabel('Time')
ax3.set_ylabel('Price')
ax3.legend()
st.pyplot(fig3)
