# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
df = pd.read_csv('diabetes1.csv', header=None)
df.head()

df.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',  
         'BMI', 'DiabetesPedigreeFunction', 'Age', 'Class']
df.head()
# load the dataset
dataset = loadtxt('diabetes1.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]
X_scaled = scale(X) 
print('Scaled_X:\n', X_scaled)

# Split dataset into 'train' & 'test' sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)

# (optional): one hot encoding??
y_train = np_utils.to_categorical(y_train)

print('Y_Train Encoded:\n', y_train)

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
history = model.fit(X_train, y_train, validation_split=0.33, epochs=10, batch_size=10)
# model.fit(X, y, epochs=150, batch_size=10, verbose=0)

# evaluate the keras model
_, accuracy = model.evaluate(X_train, y_train)
print('Accuracy: %.2f' % (accuracy*100))

# make class predictions with the model
predictions = np.argmax(model.predict(X_test), axis=-1)
# summarize the first 5 cases
for i in range(10):
    print('%s => %d (expected %d)' % (X_test[i].tolist(), predictions[i], y[i]))

    y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred, axis=1) 

accuracy_score(y_test, y_pred)