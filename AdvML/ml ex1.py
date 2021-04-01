import os
os.chdir('/Users/admin/Documents/GitHub/Python/AdvML')
import numpy as np
import pandas as pd
import csv
print('Quiz 1.3')
Quiz_1_3_data = []
with open('Quiz 1.3.csv') as File:
    reader = csv.reader(File, delimiter=',', quotechar=',',
                        quoting=csv.QUOTE_MINIMAL)
    for row in reader:
         Quiz_1_3_data.append(row)
F = np.asarray(Quiz_1_3_data,dtype=np.int32)
phi = np.array([[0.32],[0.95]])
z = np.dot(F,phi)
print('z =',z)
print('Quiz 1.4')
a = np.array([2,3])
b = np.array([-3,2])
ab = np.dot(a,b)
print('The result of dot product of a,b =' ,ab)
a_length = np.sqrt([np.power(a[0],2)+np.power(a[1],2)])
b_length = np.sqrt([np.power(b[0],2)+np.power(b[1],2)])
a1 = np.divide(a,a_length)
print('The first coordinate of the vector a`=', round(a1[0],2))
print('The second coordinate of the vector a`=', round(a1[1],2))
b1 = np.divide(b,b_length)
print('The first coordinate of the vector b`=', round(b1[0],2))
print('The second coordinate of the vector b` =', round(b1[1],2))
print('Quiz 1.5')
Theta = np.array([[2,1], [1,14/3]])
lambda_1 = 5/3
lambda_2 = 5
print('The value of sample variance of the scores of the first principal component =', lambda_2)
print('The value of sample variance of the scores of the second principal component =', lambda_1)
print('Quiz 1.6')
Fshtrih = np.inner(z,phi)
print('The first coordinate of the first object =',round(Fshtrih[0,0],2))
print('The second coordinate of the first object =',round(Fshtrih[0,1],2))
print('The first coordinate of the second object =',round(Fshtrih[1,0],2))
print('The second coordinate of the second object =',round(Fshtrih[1,1],2))
print('The first coordinate of the third object =',round(Fshtrih[2,0],2))
print('The second coordinate of the third object =',round(Fshtrih[2,1],2))
print('Task 1.1')
from sklearn.decomposition import PCA
from sklearn import datasets
import matplotlib
import matplotlib.pyplot as plt

data_task_1_1 = []
with open('task_1_1_en.csv') as File:
    reader = csv.reader(File, delimiter=',', quotechar=',',
                        quoting=csv.QUOTE_MINIMAL)
    for row in reader:
         data_task_1_1.append(row)
data_task_1_11 = np.asarray(data_task_1_1,dtype=np.float64)
X = data_task_1_11
Y = np.arange(60)
print(X[0])
pca = PCA(n_components=3, svd_solver='full') #PCA class object creation. The number of PCs and optimization method are the parameters
X_transformed = np.round(pca.fit(X).transform(X),3) #X_transformed -- ndarray of the objects, where each object is described by 2 PCs
print(X_transformed[0])
explained_variance = np.round(np.cumsum(pca.explained_variance_ratio_),3)
print(explained_variance)
plt.scatter(X_transformed[:101, 0], X_transformed[:101, 1], c=Y[:101], edgecolor='none', s=40,cmap='winter')

print('Task 1.2')
scores = np.genfromtxt('task_1_2_X_reduced_ru.csv', delimiter=';')
loadings = np.genfromtxt('task_1_2_X_loadings_ru.csv', delimiter=';')
values = np.dot(scores,loadings.T)
plt.imshow(values, cmap='Greys_r')
