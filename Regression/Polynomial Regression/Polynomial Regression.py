# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
'''from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)'''

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

from sklearn.linear_model import LinearRegression
len_reg = LinearRegression()
len_reg.fit(X,y)

#fittting polynomial dataset into our model
from sklearn.preprocessing import PolynomialFeatures
pol_reg = PolynomialFeatures(degree=2)
X_poly = pol_reg.fit_transform(X)
len_reg_2 = LinearRegression()
len_reg_2.fit(X_poly , y)

# Showing the result via matplotlib
plt.scatter(X , y , color='red')
plt.plot(X , len_reg.predict(X) , color='blue')
plt.title('Truth or Bluff(Simple Linear Regression')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

#SHowing the result of polynomial Regression
X_grid = np.arange(min(X) , max(X) , 0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X , y , color='red')
plt.plot(X_grid , len_reg_2.predict(pol_reg.fit_transform(X_grid)) , color='blue')
plt.title('Truth or Bluff(Polynomial Regression')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

#predit the Specific resultt for linear Regression
print(len_reg.predict(6.5))

#predit the Specific resultt for POlynomial Regression
print(len_reg_2.predict(pol_reg.fit_transform(6.5)))
