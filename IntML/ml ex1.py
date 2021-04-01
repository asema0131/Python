import os
os.chdir('/Users/admin/Documents/GitHub/Python/IntML')
print('ML ex 1')
import numpy as np
import pandas as pd
data = pd.read_csv("salary_and_population.csv", encoding = 'utf-8', delimiter = ',')
data.head()
new_data1 = data[data['Region_RU'] != 'Томская область']
new_data = new_data1[new_data1['Region_RU'] != 'Самарская область']
X = new_data['AVG_Salary'].mean()
Me = new_data['AVG_Salary'].median()
S2 = new_data['AVG_Salary'].var(ddof = 0)
sigma = new_data['AVG_Salary'].std(ddof = 0)
print('Выборочное среднее', '=', X)
print('Выборочная медиана', '=', Me)
print('Оценка дисперсии', '=', S2)
print('Оценка среднеквадратического отклонения', '=', sigma)
