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

