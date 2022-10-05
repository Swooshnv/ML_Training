from random import random
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

train = pd.read_csv('Desktop/Coding/scikit/s1/train.csv')
test = pd.read_csv('Desktop/coding/scikit/s1/test.csv')

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
y = train.SalePrice
X = train[features]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)
train_model = DecisionTreeRegressor(random_state = 1)
train_model.fit(train_X, train_y)

predicted = train_model.predict(val_X)
errors = mean_absolute_error(predicted, val_y)
print(errors)
