import numpy as np
import pandas as pd

#Load dataset
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data            # shape (569, 30)
y = data.target          # 0 = benign, 1 = malignant
feature_names = data.feature_names
target_names = data.target_names

df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
print(df.head())

#Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Preprocessing step
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit on train only
X_test_scaled = scaler.transform(X_test)

#Build and fit logistic regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='liblinear',
    max_iter=1000,
    class_weight=None,
    random_state=42
)
lr.fit(X_train_scaled, y_train)

# Predict & evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
y_pred = lr.predict(X_test_scaled)
y_proba = lr.predict_proba(X_test_scaled)[:, 1]  # prob of positive class (malignant)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=target_names))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

#ROC / AUC
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
print("ROC AUC:", roc_auc)

import matplotlib.pyplot as plt
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.plot([0,1],[0,1],'--', linewidth=0.7)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

#Inspect coefficients (feature influence)
coefficients = lr.coef_.ravel()   # shape (30,)
coef_df = pd.DataFrame({'feature': feature_names, 'coef': coefficients})
coef_df['abs_coef'] = np.abs(coef_df['coef'])
coef_df = coef_df.sort_values('abs_coef', ascending=False)
print(coef_df.head(10))   # top 10 influential features

#Decision boundary visualization (2 features)
# pick two features for visualization (e.g., mean radius & mean texture)
feat_idx = [0, 1]  # indices of chosen features
X2 = X[:, feat_idx]
y2 = y

# pipeline to scale and fit LR on these two features
from sklearn.pipeline import make_pipeline
pipe2 = make_pipeline(StandardScaler(), LogisticRegression(penalty='l2', C=1.0, solver='liblinear', random_state=42))
pipe2.fit(X2, y2)

# mesh grid for plotting
x_min, x_max = X2[:,0].min() - 1, X2[:,0].max() + 1
y_min, y_max = X2[:,1].min() - 1, X2[:,1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = pipe2.predict(grid).reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.2)
plt.scatter(X2[:,0], X2[:,1], c=y2, edgecolor='k', s=30)
plt.xlabel(feature_names[feat_idx[0]])
plt.ylabel(feature_names[feat_idx[1]])
plt.title('Logistic Regression decision regions (2 features)')
plt.show()

#Calibration / see if predicted probabilities are well calibrated
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
plt.figure(figsize=(6,6))
plt.plot(prob_pred, prob_true, marker='o', label='LR')
plt.plot([0,1],[0,1],'--', label='perfect calibration')
plt.xlabel('Predicted probability')
plt.ylabel('True probability')
plt.title('Calibration plot')
plt.legend()
plt.show()
#Penalty: Type of regularization {l2 for general use,
# l1 for feature selection,elasticnet,none},I choose it
# because I have small number of features
#C: Inverse of regularization strength{0.01: suspect overfitting
# or noisy data,1: balanced case,10: suspect underfitting}
#Solver: Optimazation algorithm(find the best cofficients)
# {liblinear: for small dataset,lbfgs or saga: for large datasets}
#max_iter: Maximum number of iteration for convergence
# class_wight: {None: data is balanced,balanced: dataset
# is imbalanced,{0: 1, 1: 3}: when you know which class is more important}
