# Multivariate Regression
# Date: 30/08/2019

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_data():
    return pd.read_csv("housing_cal.csv")
#loading the data
housing= load_data()

# checking out the data
print(housing.head())
print(housing.info())
print(housing.describe())

# visualisation (uncomment to see the graph)
# housing.hist(bins=50, figsize=(20,15))
# plt.show()

# splitting of housing data into train and test data at a ratio of 80% a nd 20% (i.e, 0.2)
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print("Length of train set", len(train_set))
print("Length of test set", len(test_set))

# data_visualisation (uncomment to see the graph)
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
# plt.show()
# housing.plot(kind="scatter", x="longitude",y="latitude", alpha=0.4,
# s=housing["population"]/100,label="population",
# c=housing["median_house_value"], cmap=plt.get_cmap("jet"), colorbar=True)
# plt.legend()
# plt.show()

# prepare train data

housing = train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = train_set["median_house_value"].copy()
housing.dropna(subset=["total_bedrooms"]) # remove or drop empty rows
housing_num = housing.drop('ocean_proximity', axis=1) # ocean_proximity

# training data

from sklearn.linear_model import LinearRegression

model= LinearRegression()
model.fit(housing_num, housing_labels)

# predicting test results
some_data= housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data.dropna(subset=["total_bedrooms"])
some_data_num= some_data.drop('ocean_proximity', axis=1)
print("Predictions: ", model.predict(some_data_num))
print("True Labels: ", list(some_labels))

# validation 

from sklearn.metrics import mean_squared_error
import numpy as np

pred = model.predict(some_data_num)
true_lab= some_labels
mse= mean_squared_error(true_lab, pred)
print("Mean squared error (Linear Regressor):", np.sqrt(mse))
