import numpy as np
import pandas as pd

#Load a dataset
from sklearn.datasets import load_iris
iris = load_iris()
x = iris.data
y = iris.target
feature_names = iris.feature_names

#quick look on the dataset
df = pd.DataFrame(x, columns=feature_names)
df['targer'] = y
print(df.head())

#spliting data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42,stratify=y)
#stratify will keep both train and test set balanced (the same percentage in both classes)

#preprocessing  : Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled =scaler.fit_transform(x_test)

#build and fit KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(
    n_neighbors=5,
    weights='uniform', #uniform choose the majority of votes or distance choose from the nearest neighbors
    metric='minkowski', #distance function / p=2 that's mean Euclidien & p=1 munhutan
    p=1,
    algorithm='auto' # 'auto' lets sklearn choose (kd_tree, ball_tree, brute) depending on data
)
knn.fit(x_train_scaled,y_train)

#predict on test set
y_pred = knn.predict(x_test_scaled)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print("Accuracy : ", accuracy_score(y_test,y_pred))
print("Classification_report : \n", classification_report(y_test,y_pred,target_names=iris.target_names))
print("confusion matrix : \n", confusion_matrix(y_test,y_pred))

#Inspect neighbors of a specific test point
distances,indices = knn.kneighbors(x_test_scaled[0].reshape(1, -1),n_neighbors=5)
print("Distances to 5 nearest neighbors for first test point:", distances)
print("Indices of neighbors (in training set):", indices)

# Cross-validation to estimate generalization & search for best k
from sklearn.model_selection import cross_val_score, GridSearchCV
# check/evaluate the model on 5 folds (split the data on 5 parts) and calculate the mean accuracy
cv_scores = cross_val_score(knn,scaler.transform(x),y,cv=5,scoring='accuracy')
print("5 Fold cv accuracy : ",cv_scores.mean(),cv_scores)

#
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('scaler',StandardScaler()),
    ('knn', KNeighborsClassifier()),
])
param_grid = {
    'knn__n_neighbors': [1,3,5,7,9,11],
    'knn__weights': ['uniform', 'distance'],
    'knn__p': [1, 2]   # p=1 => Manhattan, p=2 => Euclidean
}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(x, y)
print("Best params:", grid.best_params_)
print("Best CV accuracy:", grid.best_score_)

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# choose two features for visualization
X2 = x[:, :2]  # first two features
y2 = y

# pipeline + gridsearch as before but using only X2 if desired
pipe2 = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=5))])
pipe2.fit(X2, y2)

# create mesh
x_min, x_max = X2[:, 0].min() - .5, X2[:, 0].max() + .5
y_min, y_max = X2[:, 1].min() - .5, X2[:, 1].max() + .5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = pipe2.predict(grid_points).reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3)  # predicted region colors
plt.scatter(X2[:,0], X2[:,1], c=y2, edgecolor='k')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title("K-NN decision boundary (k=5)")
plt.show()

#Resume
#The first thing we did was Loading a data set called "Iris flower" it continue images
# of differente types of iris flowers
#We split the data into training set and testing set (train:75%,test:25%) and we used
# stratified to keep the balance between train and test sets for example :
# train :(type A:30%,type B:70%), test :(type A:30%,type B:70%)
#In the preprocessing step we used standardization (Z-score) make all the numbers
# close to each other for example:[-1 ,0 ,0.4 ,2]. But in other cases we can use Normalization
# (Min-Max score), it includes all the data into one range [0 and 1]
#In the training step we used the function KNeighborClassifier which used different
# parameters for training : n_neighbors = k
# weights('uniform': take the majority,'distance': calculate distances)
# matic(we used minkowski which represent either Euclien or munhutan) or you can use other matrices like cosin similariy
# algorithm(we used 'auto' because it's the best way to choose the better algorithm for this specefic case)

#fit: Learn from parameters and calculte result
#transform: apply those results on the original parameters
#fit_transform: it's a shortcut of fit and transform together

#Cross validation CV: the point of using it is to check if the model will respon right
# with a new/unknown data and it work like that : split that data into smaller pieces(folds)
# and train all k-1 folds and test it on the remining one
# repeat the same thing on all folds
#And we use 'pipeline' to make sure the model will do scaling(preprocessing step) every time
# before traning(save sequence)
#Param_gird: will try several methods/parameters to find the best result
#Finaly plot the classes