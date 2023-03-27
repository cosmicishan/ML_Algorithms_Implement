I recently completed the Machine Learning A-Z course on Udemy by the SuperDataScience team, instructed by Kirill Eremenko and Hadelin de Ponteves. In this repository, I have uploaded different algorithms of regression, classification, and clustering that I learned from the course. I obtained the datasets from Kaggle and the SuperDataScience team's official website.

For the machine learning section, I used various libraries such as NumPy, Pandas, Matplotlib, and scikit-learn. I performed data pre-processing, and at the end of every file, I visualized the data using Matplotlib.

In multiple linear regression, I used the backward elimination approach to find out the columns of the data that contribute the most to the algorithm, and I visualized the data using the best-fit column of the dataset. I also used feature scaling wherever required, and in some cases, I explicitly implemented it because some scikit-learn classes do feature scaling implicitly.

To transform some string values into binary and split them into columns, I used label encoder and one-hot encoder. Lastly, I coded an artificial neural network using Keras, where I used ReLU as the activation function in the hidden layers of neurons and the sigmoid activation function on the output layers of neurons. I used the Adam optimizer for gradient descent.

To improve the model further, I implemented k-fold cross-validation to get more accuracy. After that, I applied parameter tuning using GridSearchCV to find the best fitting parameters to train the model. It trained the artificial neural network using k-fold cross-validation to obtain relevant accuracy with different combinations of values. Eventually, I obtained the best accuracy with the best selection of values.
