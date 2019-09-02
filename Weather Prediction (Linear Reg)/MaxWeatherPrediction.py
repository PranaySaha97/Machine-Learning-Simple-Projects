# Linear Regression
# Date: 01/09/2019
# Author: Pranay Saha

import pandas as pd

weather_data= pd.read_csv('Summary of Weather.csv')

# print(weather_data.info())
# print(weather_data.describe())

req_data= weather_data[['MinTemp','MaxTemp']].copy()
# print(req_data.info())

from sklearn.model_selection import train_test_split

train_set, test_set= train_test_split(req_data, test_size=0.2)

# print (len(train_set))
# print (len(test_set))

from sklearn.linear_model import LinearRegression

xlabel=train_set.drop('MaxTemp', axis=1)
ylabel=train_set['MaxTemp']
pred_model= LinearRegression()
pred_model.fit(xlabel, ylabel)

x_test_label= test_set.drop("MaxTemp", axis=1)
y_test_label= test_set['MaxTemp']

x_test= x_test_label.iloc[:5]
print("Predicted Values: " , pred_model.predict(x_test))
print("True Values: ", list(y_test_label.iloc[:5]))


from sklearn.metrics import mean_absolute_error

mae= mean_absolute_error(y_test_label.iloc[:5] , pred_model.predict(x_test))
print("Mean Absolute Error: ", mae)
