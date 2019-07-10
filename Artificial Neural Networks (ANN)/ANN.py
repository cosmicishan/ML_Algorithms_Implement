import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[: , -1].values

from sklearn.preprocessing import LabelEncoder , OneHotEncoder
LB = LabelEncoder()
X[:,1] = LB.fit_transform(X[:,1])
X[:,2] = LB.fit_transform(X[:,2])
OH = OneHotEncoder(categorical_features=[1])
X = OH.fit_transform(X).toarray()
X = X[ :, 1:]

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
SC = StandardScaler()
X_train = SC.fit_transform(X_train)
X_test = SC.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Dropout

classifier = Sequential()
classifier.add(Dense(units = 6 , init='uniform' , activation= 'relu'))
classifier.add(Dropout(rate = 0.1))
classifier.add(Dense(units = 6 , init='uniform' , activation= 'relu'))
classifier.add(Dropout(rate = 0.1))
classifier.add(Dense(units = 1 , init='uniform' , activation= 'sigmoid'))

classifier.compile(optimizer='adam' , loss = 'binary_crossentropy' , metrics=['accuracy'])

classifier.fit(X_train , y_train , batch_size=10 , epochs=100)

y_pred = classifier.predict(X_test)
y_pred=(y_pred > 0.5)

from sklearn.metrics import confusion_matrix
CM = confusion_matrix(y_test , y_pred)

challenge = classifier.predict(SC.transform(np.array([[0,0,600,1,40,2,60000,2,1,1,50000]])))
challenge = (challenge > 0.5)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6 , init='uniform' , activation= 'relu'))
    classifier.add(Dense(units = 6 , init='uniform' , activation= 'relu'))
    classifier.add(Dense(units = 1 , init='uniform' , activation= 'sigmoid'))
    classifier.compile(optimizer='adam' , loss = 'binary_crossentropy' , metrics=['accuracy'])
    return classifier
KC = KerasClassifier(build_fn=build_classifier , batch_size = 10 , epochs = 100)
accuracies = cross_val_score(estimator = KC , X = X_train , y = y_train , cv = 10 , n_jobs = 1)
mean = accuracies.mean()
variance = accuracies.std()

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def build_classifier(optimizer = 'adam'):
    classifier = Sequential()
    classifier.add(Dense(units = 6 , init='uniform' , activation= 'relu'))
    classifier.add(Dense(units = 6 , init='uniform' , activation= 'relu'))
    classifier.add(Dense(units = 1 , init='uniform' , activation= 'sigmoid'))
    classifier.compile(optimizer=optimizer , loss = 'binary_crossentropy' , metrics=['accuracy'])
    return classifier
KC = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size' : [25,32],
              'epochs' : [100,500],
              'optimizer':['adam','rmsprop']}
grid_search = GridSearchCV(estimator=KC , param_grid=parameters,scoring='accuracy',cv=10)
grid_search.fit(X_train,y_train)
best_parameters = grid_search.best_params_
best_accuracies = grid_search.best_score_



















