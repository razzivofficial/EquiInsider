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
