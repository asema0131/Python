pervyekolstrok = 680
trainpercent = 80
testpercent = 20
criterionn='entropy'
minsamplesleaf=10
maxleafnodes=15
randomstate=2020
import os
os.chdir('/Users/admin/Documents/GitHub/Python/AdvML')
import pandas as pd 
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
df = pd.read_csv('diabetes.csv')
df.head()
task_data = df.head(pervyekolstrok)
stclass1 = len(task_data[task_data['Outcome'] == 1])
stclass0 = len(task_data[task_data['Outcome'] == 0])
print('1')
print('the number of strings in the obtained sample, which belong to the class 0 =',stclass0)
print('the number of strings in the obtained sample, which belong to the class 1 =',stclass1)
print('2')
train = task_data.head(int(len(task_data)*0.01*trainpercent))
test = task_data.tail(int(len(task_data)*0.01*testpercent))
features = list(train.columns[:8])
x = train[features]
y = train['Outcome']
print('3')
from sklearn import tree
tre = tree.DecisionTreeClassifier(criterion=criterionn, #splitting criterion
                              min_samples_leaf=minsamplesleaf, #minimum number of samples per leaf
                              max_leaf_nodes=maxleafnodes, #maximum number of leaves
                              random_state=randomstate)
clf=tre.fit(x, y)
print('4')

columns = list(x.columns)

tree.plot_tree(clf, feature_names=columns,  
                   class_names=['0', '1'],
                   filled=True)
print('глубина дерева = ',clf.tree_.max_depth)
features = list(test.columns[:8])
x = test[features]
y_true = test['Outcome']
y_pred = clf.predict(x)
print('доля правильных ответов классификатора', round(accuracy_score(y_true, y_pred),2))
print('Среднее значение метрик  𝐹1(Macro-F1):', round(f1_score(y_true, y_pred, average='macro'),2))
print('Назначенный класс конкретного объекта 737 :',clf.predict([df.loc[737, features].tolist()])[0])
print('Назначенный класс конкретного объекта 740 :',clf.predict([df.loc[740, features].tolist()])[0])
print('Назначенный класс конкретного объекта 763 :',clf.predict([df.loc[763, features].tolist()])[0])
print('Назначенный класс конкретного объекта 702 :',clf.predict([df.loc[702, features].tolist()])[0])


