from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data/50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
print(X)

# Encoding categorical data
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)

# Training the Multiple Linear Regression model on the Training set
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Evaluate Multiple Linear Regression model by MAPE , R score


def mape(y_true, y_pred):
    Accuracy = r2_score(y_true, y_pred)*100
    Mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(" Accuracy of the model is %.2f" % Accuracy)
    print(" MAPE of the model is %.2f" % Mape)


mape(y_test, y_pred)

# Visualising the Test data results
plt.scatter(y_test, y_pred, color='red')
sns.regplot(x=y_test, y=y_pred, ci=None, color='blue')
plt.title('Predicted vs Actual (Test set)')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()
