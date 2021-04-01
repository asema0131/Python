import os
os.chdir('/Users/admin/Documents/GitHub/Python/AdvML')
os.environ["PATH"] += os.pathsep + '/usr/local/Cellar/graphviz/2.47.0/lib'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('diabetes.csv')
df.head()
task_data = df.head(680)
stclass1 = len(task_data[task_data['Outcome'] == 1])
stclass0 = len(task_data[task_data['Outcome'] == 0])
print('1')
print('the number of strings in the obtained sample, which belong to the class 0 =',stclass0)
print('the number of strings in the obtained sample, which belong to the class 1 =',stclass1)
print('2')
train = task_data.head(int(len(task_data)*0.8))
test = task_data.tail(int(len(task_data)*0.2))
features = list(train.columns[:8])
x = train[features]
y = train['Outcome']
print('3')
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy', #splitting criterion
                              min_samples_leaf=10, #minimum number of samples per leaf
                              max_leaf_nodes=15, #maximum number of leaves
                              random_state=2020)
clf=tree.fit(x, y)
print('4')
from sklearn.tree import export_graphviz
import graphviz
columns = list(x.columns)
export_graphviz(clf, out_file='tree.dot',
                feature_names=columns,
                class_names=['0', '1'],
                rounded = True, proportion = False,
                precision = 2, filled = True, label='all')

with open('tree.dot') as f:
    dot_graph = f.read()

graph = graphviz.Source(dot_graph)
graphviz.render('dot','pdf','/Users/admin/Documents/GitHub/Python/AdvML/tree.dot','gd')
the_tree_depth = clf.tree_.max_depth

print('the tree depth =', the_tree_depth)
print('5')
features = list(test.columns[:8])
x = test[features]
y_true = test['Outcome']
y_pred = clf.predict(x)
from sklearn.metrics import accuracy_score
accuracy_score(y_true, y_pred)
