import os
os.chdir('/Users/admin/Documents/GitHub/Python/IntML')
print('ML ex 3')
import numpy as np
import pandas as pd
import os
os.chdir('/Users/admin/Documents/GitHub/Python/IntML')
data = pd.read_csv("data1.csv",  delimiter = ',', index_col='competitorname')
data_test = pd.read_csv("test1.csv", delimiter = ',', index_col='competitorname')
b = data_test.index.get_loc('Tootsie Roll Midgies')
p = data_test.index.get_loc('Swedish Fish')
print(b,p)
bb = data_test.iloc[b]
Y1 = bb.drop(['Y'])
pp = data_test.iloc[p]
Y2 = pp.drop(['Y'])
train_data = data.drop(['One dime','Fun Dip', 'Milky Way'])
X = pd.DataFrame(train_data.drop(['winpercent','Y'], axis=1))
Y = pd.DataFrame(train_data['Y'])
from sklearn.linear_model import LogisticRegression
reg = LogisticRegression(random_state=2019, solver='lbfgs').fit(X, Y.values.ravel())
X_test = pd.DataFrame(data_test.drop(['Y'], axis=1))
Y_pred = reg.predict(X_test)
Y_pred_probs = reg.predict_proba(X_test)
Y_pred_probs_class_1 = Y_pred_probs[:, 1]
ProbPredictionY1 = reg.predict_proba([Y1])
PredY1 = ProbPredictionY1[:, 1]
print(PredY1)
ProbPredictionY2 = reg.predict_proba([Y2])
PredY2 = ProbPredictionY2[:, 1]
print(PredY2)
Y_true = (data_test['Y'].to_frame().T).values.ravel()
from sklearn import metrics
fpr, tpr, _ = metrics.roc_curve(Y_true, Y_pred)
AUC = metrics.roc_auc_score(Y_true, Y_pred_probs_class_1)
Recall = metrics.recall_score(Y_true, Y_pred)
Precision = metrics.precision_score(Y_true, Y_pred)
print('AUC =', AUC)
print('Recall =', Recall)
print('Precision =', Precision)
import matplotlib.pyplot as plt

metrics.plot_roc_curve(reg, X_test, Y_true, color='darkorange')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.show()
