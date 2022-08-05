from statistics import mean
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

train = pd.read_csv('Desktop/coding/scikit/s1/train.csv')
test = pd.read_csv('Desktop/coding/scikit/s1/test.csv')

max_nodes = [50, 100, 150, 200, 250]
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = train[features]
y = train.SalePrice
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)
forest_model = RandomForestRegressor(random_state = 1)
forest_model.fit(train_X, train_y)
preds = forest_model.predict(val_X)
errors = mean_absolute_error(preds, val_y)
print('Error: {}'.format(errors))