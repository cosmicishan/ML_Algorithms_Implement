#importing the libariries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[: , [3,4]].values

#fitting the scipy denogram method
import scipy.cluster.hierarchy as sch
denogram = sch.dendrogram(sch.linkage(X , method='ward'))
plt.title('Denogram')
plt.xlabel('NO. of clusters')
plt.ylabel('Euclidean Distance')
plt.show()

#fitting cluster into dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5)
y_hc = hc.fit_predict(X)

#visualising the reuslt
plt.scatter(X[y_hc == 0 , 0],X[y_hc == 0,1], s = 80 , c = 'red' , label = 'Careful')
plt.scatter(X[y_hc == 1 , 0],X[y_hc == 1,1], s = 80 , c = 'blue' , label = 'Target')
plt.scatter(X[y_hc == 2 , 0],X[y_hc == 2,1], s = 80 , c = 'green' , label = 'Regular')
plt.scatter(X[y_hc == 3 , 0],X[y_hc == 3,1], s = 80 , c = 'magenta' , label = 'Careless')
plt.scatter(X[y_hc == 4 , 0],X[y_hc == 4,1], s = 80 , c = 'cyan' , label = 'Sensilble')
plt.title('Hierarchy Clustering')
plt.xlabel('Amount of salary')
plt.ylabel('Spending score')
plt.legend()
plt.show()