# importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st

start='2010-01-01'
end = '2019-12-31'
# loading dataset

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker','AAPL')
df = yf.Ticker(user_input)
df = df.history(period ='max')

#Describing Data
st.subheader('Data from 2010-2023')
st.write(df.describe())

#Visualizations
st.subheader("Closing Price V/S Time chart")
fig = plt.figure(figsize=(20,10))
plt.plot(df.Close)
st.pyplot(fig)

#Plot with 100 days moving average
st.subheader("Closing Price V/S Time chart with 100MA")
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(20,10))
plt.plot(ma100, 'r')
plt.plot(df.Close, 'b')
st.pyplot(fig)

#Plot with 200 days moving average
st.subheader("Closing Price V/S Time chart with 100MA & 200MA")
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(20,10))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)


#load my model
model = load_model('keras_model.h5')


# Tesing Part

past_100_days = data_training.tail(100)

final_df = past_100_days.append(data_testing, ignore_index=True)

input_data = scaler.fit_transform(final_df)

X_test = []
y_test = []

for i in range(100, input_data.shape[0]):
  X_test.append(input_data[i-100: i])
  y_test.append(input_data[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)

y_predicted = model.predict(X_test)

scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor


#Final graph
st.subheader('Prediction V/S Original')
fig2 = plt.figure(figsize=(20,10))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)