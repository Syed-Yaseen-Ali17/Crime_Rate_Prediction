from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd 
from tkinter import simpledialog
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from keras.layers import Dropout
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam
import keras
plt.switch_backend('TkAgg')

main = tkinter.Tk()
main.title("Big Data Analytics and Mining for Effective Visualization and Trends Forecasting of Crime Data") #designing main screen
main.geometry("1000x650")

global filename
global predict_model
global nn_rmse,lstm_rmse
global train_df

crimes = ['LARCENY/THEFT', 'OTHER OFFENSES', 'NON-CRIMINAL', 'ASSAULT', 'DRUG/NARCOTIC', 'VEHICLE THEFT', 'VANDALISM', 'WARRANTS', 'BURGLARY',
          'SUSPICIOUS OCC', 'MISSING PERSON', 'ROBBERY', 'FRAUD', 'FORGERY/COUNTERFEITING', 'SECONDARY CODES', 'WEAPON LAWS', 'PROSTITUTION', 'TRESPASS',
          'STOLEN PROPERTY', 'SEX OFFENSES FORCIBLE', 'DISORDERLY CONDUCT', 'DRUNKENNESS', 'RECOVERED VEHICLE', 'KIDNAPPING', 'DRIVING UNDER THE INFLUENCE',
          'RUNAWAY', 'LIQUOR LAWS', 'ARSON', 'LOITERING', 'EMBEZZLEMENT', 'SUICIDE', 'FAMILY OFFENSES', 'BAD CHECKS', 'BRIBERY', 'EXTORTION',
          'SEX OFFENSES NON FORCIBLE', 'GAMBLING', 'PORNOGRAPHY/OBSCENE MAT', 'TREA']

    
def upload():
    global filename
    filename = filedialog.askopenfilename(initialdir = "dataset")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded')

def visualize():
  global train_df
  train_df = pd.read_csv(filename)
  train_df.Dates.isnull().any()
  train_df.Dates.str.match('\d\d\d\d-\d\d-\d\d').all()

  train_df.rename(columns={'Dates': 'DateTime'}, inplace=True)
  train_df.DateTime = pd.to_datetime(train_df.DateTime)

  labels = train_df.Category.value_counts().index.tolist()
  category = train_df.Category.value_counts()
  print(labels)
  label_X = []
  category_X = []
  for i in range(0,10):
    label_X.append(labels[i])
    category_X.append(category[i])

  plt.pie(category_X,labels=label_X,autopct='%1.1f%%')
  plt.title('Crime Cases')
  plt.axis('equal')
  plt.show()


  
  

def yearlyGraph():
  train_df['Year'] = train_df.DateTime.dt.year
  train_df['Year'].sample(3)

  train_df['Month'] = train_df.DateTime.dt.month
  train_df['Month'].sample(3)

  train_df['Hour'] = train_df.DateTime.dt.hour
  train_df['Hour'].sample(3)

  train_df['Day'] = train_df.DateTime.dt.day
  train_df['Day'].sample(3)

  train_df['Minute'] = train_df.DateTime.dt.minute
  train_df['Minute'].sample(3)
  crime_incidents = train_df.Year.value_counts().sort_index()
  print(crime_incidents)
  plt.figure(figsize=(10,6))
  plt.grid(True)
  plt.xlabel('Year')
  plt.ylabel('Crime')
  plt.plot(crime_incidents, 'ro-', color = 'indigo')
  plt.xticks(crime_incidents.index)
  plt.title('Yearly Crime Graph')
  plt.show()

def hourlyGraph():
  crime_incidents = train_df.Hour.value_counts().sort_index()
  print(crime_incidents)
  plt.figure(figsize=(10,6))
  plt.grid(True)
  plt.xlabel('Hour')
  plt.ylabel('Crime')
  plt.plot(crime_incidents, 'ro-', color = 'indigo')
  plt.xticks(crime_incidents.index)
  plt.title('Hourly Crime Graph')
  plt.show()
  
def neuralNetwork():
  global nn_rmse
  global predict_model
  text.delete('1.0', END)
  balance_data = pd.read_csv('mydata.csv')
  X = balance_data.values[:, 0:7] 
  Y = balance_data.values[:, 0]
  print(Y)
  Y = Y.reshape(-1, 1)
  encoder = OneHotEncoder(sparse=False)
  Y = encoder.fit_transform(Y)

  X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

  model = Sequential()
  model.add(Dense(200, input_shape=(7,), activation='relu', name='fc1'))
  model.add(Dense(200, activation='relu', name='fc2'))
  model.add(Dense(35, activation='softmax', name='output'))    
  optimizer = Adam(lr=0.001)
  model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
  print('CNN Neural Network Model Summary: ')
  print(model.summary())
  model.fit(X_train, Y_train, verbose=2, batch_size=16, epochs=10)

  predicted = model.predict(X_test)
  text.insert(END,'Top 10 next year prediction crimes using Neural Networks\n\n')
  for i in range(0,10):
    text.insert(END,'Next Predicted Crime : '+str(crimes[np.argmax(predicted[i])])+"\n")
  nn_rmse = np.sqrt(mean_squared_error(Y_test, predicted))
  text.insert(END,'\nNeural Network RMSE : '+str(nn_rmse)+"\n\n\n")
  predict_model = model
  
def LSTM():
  global lstm_rmse
  text.delete('1.0', END)

  balance_data = pd.read_csv('mydata.csv')
  X = balance_data.values[:, 0:7] 
  Y = balance_data.values[:, 0]
  print(Y)
  Y = Y.reshape(-1, 1)
  encoder = OneHotEncoder(sparse=False)
  Y = encoder.fit_transform(Y)

  X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)
  X_train, y_train = np.array(X_train), np.array(Y_train)
  X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

  model = Sequential()
  model.add(keras.layers.LSTM(100,input_shape=(X_train.shape[1], X_train.shape[2])))
  model.add(Dropout(0.2))
  model.add(Dense(35))

  model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
  print(model.summary())
  model.fit(X_train, Y_train, epochs = 10, batch_size = 16)

  X_test1, Y_test1 = np.array(X_test), np.array(Y_test)
  X_test1 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
  predicted = model.predict(X_test1)
  lstm_rmse = np.sqrt(mean_squared_error(Y_test1, predicted)) / 100
  predicted = predict_model.predict(X_test)
  text.insert(END,'Top 10 next year prediction crimes using LSTM\n\n')
  for i in range(0,10):
    text.insert(END,'Next Predicted Crime : '+str(crimes[np.argmax(predicted[i])])+"\n")
  
  text.insert(END,'\nLSTM RMSE : '+str(lstm_rmse)+"\n\n\n")

def graph():
  height = [nn_rmse,lstm_rmse]
  bars = ('Neural Network RMSE','LSTM RMSE')
  y_pos = np.arange(len(bars))
  plt.bar(y_pos, height)
  plt.xticks(y_pos, bars)
  plt.show()

def close():
  main.destroy()
   
font = ('times', 16, 'bold')
title = Label(main, text='Big Data Analytics and Mining for Effective Visualization and Trends Forecasting of Crime Data', justify=LEFT)
title.config(bg='lavender blush', fg='navy blue')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Upload Crime Dataset", command=upload)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)  

visualizeButton = Button(main, text="Visualize Dataset", command=visualize)
visualizeButton.place(x=300,y=100)
visualizeButton.config(font=font1)

yearlyButton = Button(main, text="Yearly Crime", command=yearlyGraph)
yearlyButton.place(x=500,y=100)
yearlyButton.config(font=font1)

hourlyButton = Button(main, text="Hourly Crime", command=hourlyGraph)
hourlyButton.place(x=670,y=100)
hourlyButton.config(font=font1)

neuralButton = Button(main, text="Run Neural Network Algorithm", command=neuralNetwork)
neuralButton.place(x=10,y=150)
neuralButton.config(font=font1)

lstmButton = Button(main, text="Run LSTM Algorithm", command=LSTM)
lstmButton.place(x=300,y=150)
lstmButton.config(font=font1)

graphButton = Button(main, text="RMSE Comparison Graph", command=graph)
graphButton.place(x=10,y=200)
graphButton.config(font=font1)

closeButton = Button(main, text="Close Application", command=close)
closeButton.place(x=300,y=200)
closeButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=160)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1) 

main.config(bg='gray')
main.mainloop()
