#!/usr/bin/env python
# coding: utf-8

# ## How to Fit Regression Data with NN Model in Python

# In[7]:


import pandas as pd
from sklearn import datasets


# In[8]:


boston = datasets.load_boston()
X = pd.DataFrame(boston['data'], columns=boston['feature_names'])
y = pd.Series(boston['target'])


# In[9]:


X
    # :Attribute Information (in order):
    #     - CRIM     per capita crime rate by town
    #     - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
    #     - INDUS    proportion of non-retail business acres per town
    #     - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    #     - NOX      nitric oxides concentration (parts per 10 million)
    #     - RM       average number of rooms per dwelling
    #     - AGE      proportion of owner-occupied units built prior to 1940
    #     - DIS      weighted distances to five Boston employment centres
    #     - RAD      index of accessibility to radial highways
    #     - TAX      full-value property-tax rate per $10,000
    #     - PTRATIO  pupil-teacher ratio by town
    #     - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    #     - LSTAT    % lower status of the population


# In[10]:


y   # MEDV     Median value of owner-occupied homes in $1000's


# In[11]:


from tensorflow.keras.wrappers.scikit_learn import KerasRegressor # There is also a KerasClassifier class
from sklearn.model_selection import RandomizedSearchCV, KFold


# In[12]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# To tune ANN hyperparameters we will need a function building/returning a model.
# This function's parameters must be our model hyperparameters that we want to tune.

#tuning:
#  - the number of hidden layers
#  - the number of neurons in each hidden layer.


def build_model(number_of_hidden_layers=3, number_of_neurons=2):
    model = Sequential()
    
    # First hidden layer
    model.add(Dense(number_of_neurons, input_shape=(13,), activation='relu'))
    
    # hidden layers
    for hidden_layer_number in range(1, number_of_hidden_layers):
        model.add(Dense(number_of_neurons, activation='relu'))
        
    # output layer
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse')
    
    return model


# In[13]:


tuned_model = KerasRegressor(build_fn=build_model)


# In[16]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


tuned_model = KerasRegressor(build_fn=build_model)

# possible values of parameters - we want to find the best set of them

params = {'number_of_hidden_layers': [2, 3, 4, 5], 'number_of_neurons': [5, 15, 25]}

# Create a randomize search cross validation object, to find the best hyperparameters it will use a KFold cross validation with 5 splits
random_search = RandomizedSearchCV(tuned_model, param_distributions = params, cv = KFold(5))


#Dataset
from tensorflow.keras.datasets import boston_housing
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.20) #80% training and 20% testing

# find the best parameters!
his=random_search.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, verbose=1)


# In[18]:


random_search.best_estimator_.get_params()
# best combination of hyperparameters is:
  #  'number_of_hidden_layers': 4,
  #  'number_of_neurons': 25


# In[29]:


best_found_model = build_model(4, 25)


# In[30]:


best_found_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, verbose=1)


# In[31]:


print(f"The best model minimum mse loss on validation set is:  { min(best_found_model.history.history['val_loss']) } ")


# In[32]:


from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import seaborn as sns
from pylab import rcParams

register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 22, 10


# In[33]:


# load best model
#best_model = tf.keras.models.load_model('best_model.hdf5')

y_pred = best_found_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"The mse of Best found Saved Model is {round(mse, 2)}")


# In[34]:


x_ax = range(len(y_pred))
plt.scatter(x_ax, y_test, s=5, color="blue", label="original") 
plt.plot(x_ax, y_pred, lw=0.8, color="green", label="predicted")
plt.legend()
plt.show()


# In[ ]:




