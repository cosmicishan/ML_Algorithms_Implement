# Data Preprocessing 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#splitting test set and data set
from sklearn.cross_validation import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 0)

'''#feature Scalling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train , y_train)

y_pred = regressor.predict(X_train)

plt.scatter(X_train , y_train , color='red')
plt.plot(X_train , regressor.predict(X_train) , color='blue')
plt.title('Salary vs expeience')
plt.xlabel('Years of Experience ')
plt.ylabel('Salary')
plt.show()
                     
plt.scatter(X_test , y_test , color='red')
plt.plot(X_train , y_pred , color='blue')
plt.title('Salary vs expeience')
plt.xlabel('Years of Experience ')
plt.ylabel('Salary')
plt.show()