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

x_train = []
y_train = []

for i in range(100, len(data_training_array)):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i, 0])
    
x_train, y_train = np.array(x_train), np.array(y_train)

#Loading model 

model = load_model('keras_model.h5')