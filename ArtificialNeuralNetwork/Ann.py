"""
The Ann looks if a customer stayed or left the bank during a 6 months period.
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
Importing the dataset
"""
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

"""
Encoding categorical data. Encoding of each countries
"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()

#Remove Dummy variable trap
X = X[:,1:]


"""
Splitting the dataset into the Training set and Test set
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


"""
 Feature Scaling
"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""
Make the ANN
"""
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialize the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer

# units is the average of node in the input layer and the avererage of node in the output layer.
# The number of node in the input layer is 11 and the number of node in the output layer is 1.
#So the avere is (11+1)/2 = 6

#kernel initializer
# randomly initialize the weight as small number close to zero 
# So we can randomly initialize them with a uniform function
# The uniform fuction will initialize the weights according to a uniform distribution
# It will make sure that the weights are small number close to zero

# activation
# The activation function we want to use in our hidden layer which is the rectifier.
# and the sigmoid activation function for the output layer

# input_dim
# The number of node in the input layer which is 11
#classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
# Don't need to provide input_dim since it's provided from the 1st layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
#The sigmoid function is the heart of this probabilistic approach. 
#From this function we manage to get some probabilities in the logistic regression model.
# SofMas is actually the Sigmoid function but applied to a dependent variable that has more than 2 categories(output_dim)
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN

# Optimizer
# It is simply the algorithm we want to use to find the optimal set of weights
# This algorithm is the Stochastic Gradient Descent. 
# There's several types of Stochastic Gradient Descent and a very efficent one is called Adam

# Loss
# This corresponds to the last function within the Stochastic Gradient Descent which is the adam algorithm
# If we go deeper into the mathematical detail of Stochastic Gradient Descent we will see that it is base
# on the lost function that need to be optimize to find the optimal weights.
# The loss function use here will be the Logarithmic loss, binary crossentropy.
# Binary crossentropy has to do if the dependent variable has a binary outcome

# metrics
# When the weights are dated after each observation or after each batch of many observations the algorithm use
# this accurency criterion to improve the model performance
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
# The batch size is the number of observations after which we want to update the weights.
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

"""
Making predictions and evaluating the model
"""
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# If y_pred is greater than 50% return true else return false
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Accurency
# The accurency is the number of correct predictions divided by the total number
# of prediction in the test set(X_Test)

accu = (1492+228)/2000
# This accurancy is good because it's the same we got from the training set and test set

"""
Use our ANN model to predict if the customer with the following informations will leave the bank: 

Geography: France (dummy variable: 0)
Credit Score: 600
Gender: Male  (dummy variable: 1)
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000
So should we say goodbye to that customer ?
"""
new_customer = np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])
new_customer = sc.fit_transform(new_customer)
new_prediction = classifier.predict(new_customer)
new_prediction = (new_prediction > 0.5)

"""
Evaluating, Improving and Tuning the ANN
"""

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier


# Classifier with K4 Cross-Validation
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)

#estimator
# That's the object to use to fit the data.
#X
# The data to fit, the training set.
#y
#  The target variable y_train

#cv
# It is bascically the number of fold needed for the k4 Cross-Validation.
# 10 folds is usually use. We will get a relevant idea of the accuracy we have 10 accuracies with 10 folds.

# n_jobs
# It is the nummber of CPUs to use to do the computation and -1 means all CPUs.
# This is import because our classifier is going to be trained 10 times (10 folds)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)
mean = accuracies.mean()
variance = accuracies.std()


"""
Improving the ANN
Dropout Regularization to reduce overfitting if needed

Overfitting is when the model was train too much on the training set, too much that
it becomes less performance on the test set. We can observe this when we have a lagrge
difference of accuracies between the trainning set and test set. Generally when overfiting
happens we have a much higher accuracy on the training sets than the test set.
Another way to detect overfitting is when we observe a high variance when 
applying 4 Cross-Validation, the trainin set learn too much.

"""
    
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    #p
    # Fraction of the input units to drop.For example if we have 10 neurons and we choose 0.1(10%) then that means 
    # that at each iteration 1 neuron will be disabled. If it's 0.2(20%) 2 neurons will be disable.
    
    #classifier.add(Dropout(p=0.1))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)

"""
Parameter Tuning with Grid search

"""
    
# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    #p
    # Fraction of the input units to drop.For example if we have 10 neurons and we choose 0.1(10%) then that means 
    # that at each iteration 1 neuron will be disabled. If it's 0.2(20%) 2 neurons will be disable.
    
    #classifier.add(Dropout(p=0.1))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],'epochs': [100, 500],'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,scoring = 'accuracy',cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

