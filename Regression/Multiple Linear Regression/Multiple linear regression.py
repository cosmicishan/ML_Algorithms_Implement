# Data Preprocessing 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#categorising the data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3]= labelencoder_X.fit_transform(X[:, 3]) 
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#Escape the dummy variable trap
X = X[:, 1:]

#splitting test set and data set
from sklearn.cross_validation import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 0)

'''#feature Scalling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''

#perform multiple linear regression
from sklearn.linear_model import LinearRegression
Regressor = LinearRegression()
Regressor = Regressor.fit(X_train , y_train)

#predicting the result 
y_train_pred = Regressor.predict(X_train)
y_test_pred = Regressor.predict(X_test)

#Building optical model using backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones([50 ,1]).astype(int) , values = X , axis=1)
X_opt = X[: , [0,1,2,3,4,5]]
regressor_ols = sm.OLS(endog = y , exog = X_opt).fit()
regressor_ols.summary()
X_opt = X[: , [0,1,3,4,5]]
regressor_ols = sm.OLS(endog = y , exog = X_opt).fit()
regressor_ols.summary()
X_opt = X[: , [0,3,4,5]]
regressor_ols = sm.OLS(endog = y , exog = X_opt).fit()
regressor_ols.summary()
X_opt = X[: , [0,3,4,5]]
regressor_ols = sm.OLS(endog = y , exog = X_opt).fit()
regressor_ols.summary()
X_opt = X[: , [0,3,5]]
regressor_ols = sm.OLS(endog = y , exog = X_opt).fit()
regressor_ols.summary()
X_opt = X[: , [0,3]]
regressor_ols = sm.OLS(endog = y , exog = X_opt).fit()
regressor_ols.summary()

Regressor_2 = LinearRegression()
Regressor_2.fit(X_train[:,2:3] , y_train)
#showing the result of X train dataset
X_grid = np.arange(min(X_train[:,2:3]), max(X_train[:,2:3]), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_train[:, 2] , y_train , color='red')
plt.plot(X_grid , Regressor_2.predict(X_grid) , color='blue')
plt.title('R&D Spend vs Profit')
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.show()

#showing the result of Y train dataset
plt.scatter(X_test[:,2] , y_test , color='red')
plt.plot(X_test[:,2] , Regressor_2.predict(X_test[:,2:3]) , color='blue')
plt.title('R&D Spend vs Profit')
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.show()

