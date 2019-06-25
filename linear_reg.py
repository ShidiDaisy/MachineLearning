#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:39:35 2019

@author: shidi
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("height.vs.temperature.csv")

#data visualization
plt.figure(figsize=(16, 8))
plt.scatter(data['height'], data['temperature'], c='black')
plt.xlabel("Height")
plt.ylabel("Temperature")
plt.show()

#Train Linear Regression Model
X = data['height'].values.reshape(-1,1)
y = data['temperature'].values.reshape(-1,1)
reg = LinearRegression()
reg.fit(X,y)

print('a = {:.5}'.format(reg.coef_[0][0]))
print('b = {:.5}'.format(reg.intercept_[0]))
print("Linear regression model is: Y = {:.5}X + {:.5}".format(reg.coef_[0][0], reg.intercept_[0]))

#Visualize the trained linear regression model
predictions = reg.predict(X)
plt.figure(figsize=(16, 8))
plt.scatter(data['height'], data['temperature'], c = 'black')
plt.plot(data['height'], predictions, c = 'blue', linewidth = 2)
plt.xlabel("Height")
plt.ylabel("Temperature")
plt.show()

#prediction
predict_100 = reg.predict([[100]])
print('When height is 100 meter, the predicted value of temperature is {:.5}'.format(predict_100[0][0]))