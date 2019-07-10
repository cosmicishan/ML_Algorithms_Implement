
Hello, universe.
I just have completed machine learning A-Z course from udemy by super data science team instructed by kirill eremenko and hadelin de ponteves.
I really enjoyed the course and got to learn many things. in this repository i uploaded the different algorithms of regression, classification and clustering which i learned from the course.
I get the datasets from the kaggle and it was also available on the super data science team official website.
I implement all the different algorithms on the different datasets. 
The library i used is numpy, pandas, matplotlib, scikit-learn for the machine learning section. 
I also done data pre-proccesing and in the end of every file i visualise the data using matplotlib.
In multiple linear regression i used backward elimination approach to find out the column of the data which contribute to the algorithm most and i visualise the data using the best fit column of the dataset.
I also use feature scaling on wherever it was require, some scikit-learn class do feature scaling implicitly but some other purpose i implement it explicitly.
I use label encoder and one-hot encoder to transform some string values into binary and split it into the columns.
At last, i code artificial neural network using keras, i used relu as activation function in the hidden layers of neurons and the sigmoid activation function on the output layers of neurons. i use adam optimizer for gradient descent.
To improve the model more in the last part i implement k-fold cross validation to get more accuracy. 
After that i apply parameter tuning using GridDearchCV to find the best fitting parameters to train the model. it will train the artificial neural network using k-fold cross validation to get a relevant accuracy with a different combination of the values and eventually, in the end we will get best accuracy with the best selection of the values. 
