print('ML ex 3')
import numpy as np
import pandas as pd
data = pd.read_csv("data1.csv",  delimiter = ',', index_col='competitorname')
data_test = pd.read_csv("test1.csv", delimiter = ',', index_col='competitorname')
b = data_test.index.get_loc('Tootsie Roll Midgies')
p = data_test.index.get_loc('Swedish Fish')
print(b,p)
bb = data_test.iloc[b]
Y1 = bb.drop(['Y'])
pp = data_test.iloc[p]
Y2 = pp.drsop(['Y'])
train_data = data.drop(['One dime','Fun Dip, Milky Way'])
X = pd.DataFrame(train_data.drop(['winpercent', 'Y'], axis=1))
Y = pd.DataFrame(train_data['Y'])
from sklearn.linear_model import LogisticRegression
reg = LogisticRegression(random_state=2019, solver='lbfgs').fit(X, Y.values.ravel())
ProbPredictionY1 = reg.predict_proba([Y1])
print(ProbPredictionY1)
ProbPredictionY2 = reg.predict_proba([Y2])
