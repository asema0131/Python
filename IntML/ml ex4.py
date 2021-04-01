import os
os.chdir('/Users/admin/Documents/GitHub/Python/IntML')
import pandas as pd
train_data = pd.read_csv("mlex4.csv", delimiter=',', index_col='id')
X = pd.DataFrame(train_data.drop(['Class'], axis=1))
Y = pd.DataFrame(train_data['Class']).values.ravel()
from sklearn.neighbors import KNeighborsClassifier
print('Евклидово расстояние')
neigh = KNeighborsClassifier(n_neighbors=3, p=2)
neigh.fit(X, Y)
NewObject = [92, 85]
NewClass = neigh.predict([NewObject])
Prob = neigh.predict_proba([NewObject])
Rastdotochki = neigh.kneighbors([NewObject])
print(NewClass)
print(Prob)
print(Rastdotochki)
print('Манхэттенское расстояние')
neigh1 = KNeighborsClassifier(n_neighbors=3, p=1)
neigh1.fit(X, Y)
NewObject = [92, 85]
NewClass = neigh1.predict([NewObject])
Prob = neigh1.predict_proba([NewObject])
Rastdotochki = neigh1.kneighbors([NewObject])
print(NewClass)
print(Prob)
print(Rastdotochki)
