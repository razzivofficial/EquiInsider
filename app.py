import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from keras.models import load_model
import streamlit as st

start = '2002-04-01'
end = datetime.now().strftime('%Y-%m-%d')

st.title("EquiInsider")
user_input= st.text_input('Enter stock ticker from yahoo fin', 'RELIANCE.NS')
df = yf.download(user_input, start=start, end=end)

#Describing the data

st.subheader('Data from 01-04-2002 to todays date')
st.write(df.describe())

st.subheader('Closing Price vs Time chart with 100 MA') #moving avg
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,8))
plt.plot(ma100)
plt.plot(df['Close'])
st.pyplot(fig)


st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,8))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df['Close'] , 'b')
st.pyplot(fig)


#Dividing training and testing data

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

print(data_training.shape)
print(data_testing.shape)


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_training_array= scaler.fit_transform(data_training)

#Splitting data into xtrain n ytrain

# x_train = []
# y_train = []

# for i in range(100, len(data_training_array)):
#     x_train.append(data_training_array[i-100:i])
#     y_train.append(data_training_array[i, 0])
    
# x_train, y_train = np.array(x_train), np.array(y_train)

#Loading model 

model = load_model('keras_model.h5')

#Testing and Pradictions from past 100days data

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)


#Testing 

from keras.preprocessing.sequence import pad_sequences

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i, 0])
    y_test.append(input_data[i, 0])

x_test = np.array(x_test)
y_test = np.array(y_test)

x_test = pad_sequences(x_test, maxlen=100, dtype='float32', padding='post', truncating='post')
y_predicted = model.predict(x_test)


scaler= scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


#prediced final graph
st.subheader('Prediction vs Actual value graph')
fig2 = plt.figure(figsize=(12,8))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
