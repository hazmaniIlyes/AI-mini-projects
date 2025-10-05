import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# Put into a DataFrame for inspection
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
print(df.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

#Train Naive Bayes Model
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)

#Predict and evaluate
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=class_names))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

probs = model.predict_proba(X_test[:5])
print("Predicted probabilities for first 5 samples:\n", probs)

#plot result
import matplotlib.pyplot as plt

# Use only first two features for 2D visualization
X_vis = X[:, :2]
y_vis = y

model2 = GaussianNB().fit(X_vis, y_vis)

x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
Z = model2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_vis, edgecolor='k')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title("Naive Bayes decision regions (first 2 features)")
plt.show()


#We used GaussianNB because it's the best choice for data like Iris (height, weight) numeric data
#The main formula is : P(A∣B)=P(B∣A)×P(A)/P(B)
#P(A∣B)= Posterior , P(A)= prior
#Naive bayes methods : GaussianNB,MultinomialNB,BernoulliNB,ComplementNB,CategoricalNB
#KNN gave better Result on the same dataset