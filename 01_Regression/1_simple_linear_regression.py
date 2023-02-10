import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Importing the dataset
dataset = pd.read_csv('data/Salary_Data.csv')

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=1/3, random_state=0)

# Training the Simple Linear Regression model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test data results
y_pred = regressor.predict(x_test)


# Evaluate Simple Linear Regression model by MAPE , R score
def mape(y_true, y_pred):
    Accuracy = r2_score(y_true, y_pred)*100
    Mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(" Accuracy of the model is %.2f" % Accuracy)
    print(" MAPE of the model is %.2f" % Mape)


mape(y_test, y_pred)

#  Visualising the Training data results
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test data results
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
