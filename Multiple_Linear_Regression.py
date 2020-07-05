# importing libraries
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing datasets
trainData = pd.read_csv("Train.csv")
testData = pd.read_csv("Test.csv")
X_train = trainData.iloc[1:, :-1].values
Y_train = trainData.iloc[1:, :-1].values
X_test = testData.iloc[1:, :].values

# Training the Multiple Linear Regression Model on the Training set
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualizing Training set results
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("AIR QUALITY PREDICTION (TRAINING SET)")
plt.xlabel("FEATURES")
plt.ylabel("AIR QUALITY")
plt.show()

# Visualizing Test set results
plt.scatter(X_test, y_pred, color='red')
plt.plot(X_test, y_pred, color='blue')
plt.title('AIR QUALITY PREDICTION (TEST SET)')
plt.xlabel('FEATURES')
plt.ylabel('AIR QUALITY')
plt.show()
