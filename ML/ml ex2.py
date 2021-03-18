import sys
sys.path.append(r'/Users/admin/Documents/GitHub/Python/ML')
print('ML ex 2')
import numpy as np
import pandas as pd
data = pd.read_csv("candy-data.csv", encoding = 'utf-8', delimiter = ',', index_col='competitorname')
b = data.index.get_loc('Boston Baked Beans')
p = data.index.get_loc('Peanut M&Ms')
print(b,p)
bb = data.iloc[b]
Y1 = bb.drop(['winpercent', 'Y'])
pp = data.iloc[p]
Y2 = pp.drop(['winpercent', 'Y'])
train_data = data.drop(['Boston Baked Beans','Peanut M&Ms'])
X = pd.DataFrame(train_data.drop(['winpercent', 'Y'], axis=1))
Y = pd.DataFrame(train_data['winpercent'])
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X, Y)
PredictionY1 = reg.predict([Y1])
PredictionY2 = reg.predict([Y2])
Prediction = reg.predict([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0.261, 0.273]])
print('Предсказанное значение winpercent для конфеты Boston Baked Beans:', PredictionY1)
print('Предсказанное значение winpercent для конфеты Peanut M&Ms:', PredictionY2)
print('Предсказание для конфет введеных вручную:', Prediction)
