import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.preprocessing import OneHotEncoder
from keras.layers import Dropout
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam
'''
train_df = pd.read_csv('train.csv')
print(train_df.sample(5))

train_df.Dates.isnull().any()
train_df.Dates.str.match('\d\d\d\d-\d\d-\d\d').all()

train_df.rename(columns={'Dates': 'DateTime'}, inplace=True)
train_df.DateTime = pd.to_datetime(train_df.DateTime)

labels = train_df.Category.value_counts().index.tolist()
category = train_df.Category.value_counts()
print(category)

label_X = []
category_X = []
for i in range(0,10):
    label_X.append(labels[i])
    category_X.append(category[i])

plt.pie(category_X,labels=label_X,autopct='%1.1f%%')
plt.title('Crime Cases')
plt.axis('equal')
plt.show()


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
plt.ylabel('ACF')
plt.plot(crime_incidents, 'ro-', color = 'indigo')
plt.xticks(crime_incidents.index)
plt.title('ALGT Stock. ACF plot')
plt.show()

crime_incidents = train_df.Hour.value_counts().sort_index()
print(crime_incidents)
plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Year')
plt.ylabel('ACF')
plt.plot(crime_incidents, 'ro-', color = 'indigo')
plt.xticks(crime_incidents.index)
plt.title('ALGT Stock. ACF plot')
plt.show()

train_df.drop('DateTime', axis=1, inplace=True)
train_df.drop('Resolution', axis=1, inplace=True)
train_df.drop('Descript', axis=1, inplace=True)

train_df['Block_Ind'] = train_df['Address'].apply(lambda x: 1 if 'block' in x.lower() else 0)
train_df['Street'] = train_df['Address'].apply(lambda x: x.split(" ")[0] if x.split(" ")[0].isnumeric() else 0)
train_df.drop('Address', axis=1, inplace=True)

min_X = train_df['X'].min()
min_Y = train_df['Y'].min()
train_df['X'] = (train_df['X'] - min_X)
train_df['Y'] = (train_df['Y'] - min_Y)

le = LabelEncoder()
train_df['Category'] = pd.Series(le.fit_transform(train_df['Category']))
train_df['DayOfWeek'] = pd.Series(le.fit_transform(train_df['DayOfWeek']))
train_df['PdDistrict'] = pd.Series(le.fit_transform(train_df['PdDistrict']))
train_df['Block_Ind'] = pd.Series(le.fit_transform(train_df['Block_Ind']))
#train_df['Street'] = pd.Series(le.fit_transform(train_df['Street']))

train_df.to_csv('mydata.csv', index=False)
'''

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

print(X_train.shape)
print(X_test.shape)


model = Sequential()
model.add(LSTM(100,input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(35))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print(model.summary())
model.fit(X_train, Y_train, epochs = 10, batch_size = 16)

X_test, Y_test = np.array(X_test), np.array(Y_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


predicted = model.predict(X_test)
for i in range(len(predicted)):
    print(predicted[i])
    print(np.argmax(predicted[i]))


print(predicted.shape)
print(Y_test.shape)
print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_test, predicted)))

'''
model = Sequential()
model.add(Dense(200, input_shape=(7,), activation='relu', name='fc1'))
model.add(Dense(200, activation='relu', name='fc2'))
model.add(Dense(35, activation='softmax', name='output'))    
optimizer = Adam(lr=0.001)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
print('CNN Neural Network Model Summary: ')
print(model.summary())
model.fit(X_train, Y_train, verbose=2, batch_size=16, epochs=10)

#X_test, Y_test = np.array(X_test), np.array(Y_test)

#X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


predicted = model.predict(X_test)
for i in range(len(predicted)):
    print(predicted[i])
    print(np.argmax(predicted[i]))


print(predicted.shape)
print(Y_test.shape)
print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_test, predicted)))
'''
