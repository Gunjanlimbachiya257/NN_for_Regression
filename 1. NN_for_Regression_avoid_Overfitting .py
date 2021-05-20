#!/usr/bin/env python
# coding: utf-8

# ## How to Fit Regression Data with NN Model in Python

# ### Dataset: Housing Values in Suburbs of Boston The medv variable is the target variable.
# 
# - The Boston data frame has 506 rows and 14 columns (please see columns description below) 

# ### Necessary libraries
# * Numpy
# * Pandas
# * Scikit.learn
# * Tensorflow
# * Keras
# * Matplotlib

# ### Table of Contents
# 
# * [Step1: Data Loading](#Data_loading) 
# * [Step2: Preparation of training and testing samples](#Prep_train_test_samples)
# * [Step3: Building Neural Network](#Building_NN)
# * [Step4: Fit model](#fit_model)
# * [Step5: Predictions](#predictions)
# * [Step6: Visualizing the results](#results)

# In[57]:


import pandas as pd
from sklearn import datasets


# ### Step1: Data Loading <a class="anchor" id="Data_loading"></a>

# In[58]:


boston = datasets.load_boston()
X = pd.DataFrame(boston['data'], columns=boston['feature_names'])
y = pd.Series(boston['target'])


# In[59]:


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


# In[60]:


y   # MEDV     Median value of owner-occupied homes in $1000's


# * We are trying to predict MEDV (Median value of owner-occupied homes in $1000's), so we are dealing with a regression task

# ## Simple Regression with Tensorflow's Keras API

# ### Step2: Preparation of training and testing samples <a class="anchor" id="Prep_train_test_samples"></a>

# In[61]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ### Step3: Building Neural Network <a class="anchor" id="Building_NN"></a>

# In[62]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[75]:


model = Sequential()

model.add(Dense(13, input_shape=(13,), activation='relu')) #input layer
model.add(Dense(28, activation='relu')) #hidden layer1
model.add(Dense(13, activation='relu')) #hidden layer2
model.add(Dense(8, activation='relu'))  #hidden layer3

# regression - no activation function in the last layer
model.add(Dense(1)) #output layer        

model.compile(optimizer='adam', loss='mse')
model.summary()


# ### Step4: Fit model <a class="anchor" id="fit_model"></a>, Train the Model Using Callbacks
# 
# ### Here, using the Early Stopping to Halt the Training of Neural Networks at the Right Time

# In[76]:


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#patience: Number of epochs with no improvement after which training will be stopped.
es = EarlyStopping(monitor = 'val_loss', patience=15)

# save the best model ( = lowest mse) to a file 'best_model.hdf5'
mc = ModelCheckpoint('best_model.hdf5', save_best_only = True)

# pass the above callbacks to callbacks parameter while fitting model
print('---------------------------------NN training started----------------------------------')
his=model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[es, mc], epochs=500)
print('---------------------------------NN training Done------------------------------------')


# ### Plot history on test and train sets during training

# In[88]:


import matplotlib.pyplot as plt
plt.figure(figsize=(15,8))

plt.xlabel('epochs')
plt.ylabel('mse')

plt.plot(his.history['loss'][0:])
plt.plot(his.history['val_loss'][0:])
plt.ylim(0,200)


# ### Step5: Predictions <a class="anchor" id="predictions"></a>

# ### MSE on Test Set

# In[85]:


from sklearn.metrics import mean_absolute_error, mean_squared_error

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"The mse on the test set is {round(mse, 2)}")


# ### MSE on Test Set - Based on Saved Best Model

# In[90]:


# load best model
best_model = tf.keras.models.load_model('best_model.hdf5')

y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"The mse of Best Saved Model is {round(mse, 2)}")


# In[87]:


import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import seaborn as sns
from pylab import rcParams


register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 22, 10
x_ax = range(len(y_pred))
plt.scatter(x_ax, y_test, s=5, color="blue", label="original") 
plt.plot(x_ax, y_pred, lw=0.8, color="green", label="predicted")
plt.legend()
plt.show()


# In[ ]:




