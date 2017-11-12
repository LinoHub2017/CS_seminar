from keras.layers import Input
#from keras.layers.core import dense
from keras.layers import Dense
from keras.models import Model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow
from sklearn import preprocessing
import math


df = pd.read_csv("/home/lino/je/data_set1.csv", header=None, index_col=False,\
                names = ['date','holiday','hour','load'])

# drop column hour and holiday as the 'date' column might be enough to have pattern in the target data 'load'
df = df.drop(['holiday'], axis = 1)
features = df

# Exploring feature in 'date' like week days, month of the year and days of the year
features['week'] = pd.to_datetime(df.date).dt.dayofweek
features['month'] = pd.to_datetime(df.date).dt.month
#features['daysofyear'] = pd.to_datetime(df.date).dt.dayofyear
df['hour'] = pd.to_datetime(df.hour).dt.hour

# adding new colum called temp
#df["temp"] = data1.l

# drop the initial column date
features = features.drop(['date'], axis = 1)
features = features[['hour','week','month','load']]
#print(features)




Xorig =features.as_matrix()

scaler = StandardScaler()
X_scaled= scaler.fit_transform(Xorig)
Xmeans = scaler.mean_
Xstds = scaler.scale_

y = X_scaled[:, 3]
X = np.delete(X_scaled, 3, axis=1)
#print(X)
#print(y)

X_train = X[:55209]
y_train = y[:55209]
X_test = X[55209:61343]
y_test = y[55209:61343]




readings = Input(shape=(3,))
x0 = Dense(1024, activation="relu", kernel_initializer="glorot_uniform")(readings)
x1 = Dense(512, activation="relu", kernel_initializer="glorot_uniform")(x0)
x2 = Dense(64, activation="relu")(x1)
x3 = Dense(64, activation="relu")(x2)
x4 = Dense(128, activation="relu")(x3)
x5 = Dense(64, activation="relu")(x4)

x6 = Dense(64, activation="relu")(x5)
x7 = Dense(64, activation="relu")(x6)
x8 = Dense(128, activation="relu")(x7)
x9 = Dense(64, activation="relu")(x8)

x10 = Dense(64, activation="relu")(x9)
x11 = Dense(64, activation="relu")(x10)
x12 = Dense(128, activation="relu")(x11)
x13 = Dense(64, activation="relu")(x12)


x14 = Dense(128, activation="relu")(x13)

load = Dense(1, kernel_initializer="glorot_uniform")(x14)


model = Model(inputs=[readings], outputs=[load])
model.compile(loss="mse", optimizer="adam",metrics=["accuracy"])


epochs = 200
batch_size = 500

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

score, _ = model.evaluate(X_test,y_test, batch_size=500)
rmse = math.sqrt(score)
print("MSE: {:.3f}, RMSE: {:.3f}".format(score, rmse))


y_target = model.predict(X_test).flatten()
for i in range(20):
    label = (y_test[i] * Xstds[3]) + Xmeans[3]
    prediction = (y_target[i] * Xstds[3]) + Xmeans[3]
    print("Load expected: {:.3f}, predicted: {:.3f}".format(label, prediction))



trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)

#print(trainPredict.shape)
#print(X_train.shape)

#print(testPredict.shape)
#print(X_test.shape)

from sklearn.metrics import r2_score
print('Train Score: %.2f R^2 ' % r2_score(y_train,trainPredict, multioutput='variance_weighted'))
print('Test Score: %.2f R^2 ' % r2_score(y_test,testPredict,multioutput='variance_weighted' ))


"""
# plotting
plt.plot(np.arange(y_test.shape[0]), (y_test * Xstds[3]) / Xmeans[3], color="b", label="actual")
plt.plot(np.arange(y_target.shape[0]), (y_target*Xstds[3]) / Xmeans[3], color="r", label="predicted", alpha=0.5)
plt.xlabel("points")
plt.ylabel("load")
plt.show()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title(' Model Accuracy ')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
    
"""
