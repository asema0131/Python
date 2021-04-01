import os
os.chdir('/Users/admin/Documents/GitHub/Python/IntML')
import numpy as np
import pandas as pd
DATA = pd.read_csv("mlex5.csv", delimiter=',', index_col='Object')
coords = DATA.drop('Cluster', axis=1)
from sklearn.cluster import KMeans
centroid = np.array([[7.5, 12.0], [11.43, 8.43], [11.25, 9.75]])
kmeans = KMeans(n_clusters=3, init=centroid, max_iter=100, n_init=1)
model = kmeans.fit(coords)
Clusters = model.labels_.tolist()
alldistances = kmeans.fit_transform(coords)
print(Clusters)
print(alldistances)
