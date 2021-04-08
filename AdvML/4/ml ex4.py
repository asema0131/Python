cvv = 2
Cc = 1.75
criterionn='entropy'
minsamplesleaf=10
maxleafnodes=20
randomstate=195
nestimators = 12
solverr='lbfgs'
import os
os.chdir('/Users/admin/Documents/GitHub/Python/AdvML/4')
from IPython.display import Image
from IPython.display import display
from imutils import paths
import cv2
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
def extract_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()
imagePaths = sorted(list(paths.list_images('train')))
trainData = []
labels = []

for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath, 1)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]
    hist = extract_histogram(image)
    trainData.append(hist)
    labels.append(label)
Y = [1 if x == 'cat' else 0 for x in labels]
print(Y[0])
i = Image(filename=imagePaths[0])
display(i)
print('decision tree')
tree = DecisionTreeClassifier(criterion=criterionn, #критерий разделения
                              min_samples_leaf=minsamplesleaf, #минимальное число объектов в листе
                              max_leaf_nodes=maxleafnodes, #максимальное число листьев
                              random_state=randomstate)
bagging = BaggingClassifier(tree, #базовый алгоритм
                            n_estimators=nestimators, #количество деревьев
                            random_state=randomstate)
bagging.fit(trainData, Y)
print('model of the almost separating hyperplane')
svm = LinearSVC(random_state = randomstate, C = Cc)
svm.fit(trainData, Y)
print('random forest')
forest = RandomForestClassifier(n_estimators=nestimators, #количество деревьев
                             criterion=criterionn, #критерий разделения
                              min_samples_leaf=minsamplesleaf, #минимальное число объектов в листе
                              max_leaf_nodes=maxleafnodes, #максимальное число листьев
                              random_state=randomstate)
forest.fit(trainData, Y)
print('logistic regression')
lr = LogisticRegression(solver=solverr, random_state=randomstate)
base_estimators = [('SVM', svm), ('Bagging DT', bagging), ('DecisionForest', forest)]
sclf = StackingClassifier(estimators=base_estimators, final_estimator=lr, cv=cvv)
sclf.fit(trainData, Y)
print(sclf.score(trainData, Y))

singleImage = cv2.imread('test/dog.1024.jpg')
histt = extract_histogram(singleImage)
histt2 = histt.reshape(1, -1)
prediction = sclf.predict(histt2)
j = Image(filename='test/dog.1024.jpg')
display(j)
print(prediction)
print(sclf.predict_proba(histt2))





















