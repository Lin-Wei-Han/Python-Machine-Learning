# Logistic Regression

from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data/Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Logistic Regression model
logistic_classifier = LogisticRegression(random_state=0).fit(X_train, y_train)

# Training the K-NN model
knn_classifier = KNeighborsClassifier(
    n_neighbors=5, metric='minkowski', p=2).fit(X_train, y_train)

# Training the SVM model on the Training set
svm_classifier = SVC(kernel='linear', random_state=0,
                     probability=True).fit(X_train, y_train)

# Training the Kernel SVM model
kernelsvm_classifier = SVC(
    kernel='rbf', random_state=0, probability=True).fit(X_train, y_train)

# Training the Naive Bayes model
nb_classifier = GaussianNB().fit(X_train, y_train)

# Training the Decision Tree Classification model
dc_classifier = DecisionTreeClassifier(
    criterion='entropy', random_state=0).fit(X_train, y_train)

# Training the Random Forest Classification model
rf_classifier = RandomForestClassifier(
    n_estimators=10, criterion='entropy', random_state=0).fit(X_train, y_train)

# logistic define metrics
y_pred_logistic_proba = logistic_classifier.predict_proba(X_test)[::, 1]
logistic_fpr, logistic_tpr, _ = metrics.roc_curve(
    y_test,  y_pred_logistic_proba)
logistic_auc = metrics.roc_auc_score(y_test, y_pred_logistic_proba)

# KNN define metrics
y_pred_knn_proba = knn_classifier.predict_proba(X_test)[::, 1]
knn_fpr, knn_tpr, _ = metrics.roc_curve(
    y_test,  y_pred_knn_proba)
knn_auc = metrics.roc_auc_score(y_test, y_pred_knn_proba)

# SVM define metrics
y_pred_svm_proba = svm_classifier.predict_proba(X_test)[::, 1]
svm_fpr, svm_tpr, _ = metrics.roc_curve(
    y_test,  y_pred_svm_proba)
svm_auc = metrics.roc_auc_score(y_test, y_pred_svm_proba)

# Kernel SVM define metrics
y_pred_kernelsvm_proba = kernelsvm_classifier.predict_proba(X_test)[::, 1]
kernelsvm_fpr, kernelsvm_tpr, _ = metrics.roc_curve(
    y_test,  y_pred_kernelsvm_proba)
kernelsvm_auc = metrics.roc_auc_score(y_test, y_pred_kernelsvm_proba)


# NB define metrics
y_pred_nb_proba = nb_classifier.predict_proba(X_test)[::, 1]
nb_fpr, nb_tpr, _ = metrics.roc_curve(
    y_test,  y_pred_nb_proba)
nb_auc = metrics.roc_auc_score(y_test, y_pred_nb_proba)

# DC define metrics
y_pred_dc_proba = dc_classifier.predict_proba(X_test)[::, 1]
dc_fpr, dc_tpr, _ = metrics.roc_curve(
    y_test,  y_pred_dc_proba)
dc_auc = metrics.roc_auc_score(y_test, y_pred_dc_proba)

# RF define metrics
y_pred_rf_proba = rf_classifier.predict_proba(X_test)[::, 1]
rf_fpr, rf_tpr, _ = metrics.roc_curve(
    y_test,  y_pred_rf_proba)
rf_auc = metrics.roc_auc_score(y_test, y_pred_rf_proba)

# create ROC curve
plt.plot(logistic_fpr, logistic_tpr,
         label="Logistic AUC=" + str(round(logistic_auc, 2)))
plt.plot(knn_fpr, knn_tpr, label="KNN AUC=" + str(round(knn_auc, 2)))
plt.plot(svm_fpr, svm_tpr, label="SVM AUC=" + str(round(svm_auc, 2)))
plt.plot(kernelsvm_fpr, kernelsvm_tpr,
         label="Kernel SVM AUC=" + str(round(kernelsvm_auc, 2)))
plt.plot(nb_fpr, nb_tpr, label="Naive Bayes AUC=" + str(round(nb_auc, 2)))
plt.plot(dc_fpr, dc_tpr, label="Decision Tree AUC=" + str(round(dc_auc, 2)))
plt.plot(rf_fpr, rf_tpr, label="Random Forest AUC=" + str(round(rf_auc, 2)))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()
