from hyperopt import Trials, STATUS_OK, tpe
import optim
from hyperopt.hp import choice
from hyperopt.hp import uniform
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns



tf.enable_eager_execution()

#Create the CNN model
def create_model(normed_train_data, normed_test_data, train_labels, test_labels):
  #Build the CNN
  model= keras.models.Sequential()
  model.add(layers.Reshape((22, 1))) #reshape for use in CNN
  # CNN layers
  model.add(layers.Conv1D(activation='relu', #Options are in the list in {}
		  padding="same", filters={{choice([8, 16, 20, 25, 30, 33, 35, 40, 45, 50, 55, 60, 64])}}, 
		  kernel_size={{choice([6, 8, 10, 12])}}))  
  model.add(layers.Conv1D(activation='relu',
		  padding="same", filters={{choice([8, 16, 20, 25, 30, 33, 35, 40, 45, 50, 55, 60, 64])}}, 
		  kernel_size={{choice([6, 8, 10, 12])}}))  

  # Max pooling layer
  model.add(layers.MaxPooling1D(pool_size={{choice([2, 3,  4, 5,  6, 7, 8])}}))
  model.add(layers.Flatten())#flatten for use in dense layers
  # Dense layers
  model.add(layers.Dense(units={{choice([25, 30, 33, 35, 40])}}, 
		activation='relu'))
  model.add(layers.Dense(units=1, activation="linear")) #output layer
  
  optimizer = tf.train.RMSPropOptimizer(0.001)
  model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

  result = model.fit(normed_train_data, train_labels,
            batch_size={{choice([32, 64, 128])}},
            epochs=500,
            verbose=0,
            validation_split=0.2)

  validation_acc = np.amax(result.history['val_mean_absolute_error']) #Some stats
  print('Best validation acc of epoch:', validation_acc)
  return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


#Loading in my data to use
def data():
  #Load in the stats for normalisation
  stats = np.loadtxt('../CALIFA_stats.dat')
  mean = stats[2:, 0]
  deviation = stats[2:, 1]
  #ALHAMBRA flux bands
  column_names = ['name', 'age', 'ZHL', 'F3655', 'F3965', 'F4275', 'F4585', 'F4895', 'F5205', 'F5515', \
    'F5825', 'F6135', 'F6445', 'F6755', 'F7065', 'F7375', 'F7685', 'F7995', 'F8305', 'F8615', 'F8915', \
    'F9235', 'F9545', 'J', 'H']

  feature_names = column_names[0] #column for metallicity
  label_names = column_names[1:] #columns for the filters

  #Import data
  incoming = pd.read_table('../CALIFA_data.dat', delim_whitespace = True, names = column_names)
  dataset = incoming.copy()
  dataset = dataset.dropna() #Removes lines with inf values

  #Build dataset of galaxies
  GroupA = dataset.copy()
  GroupA = GroupA.drop(columns = 'name') #Hyperas doesn't need gal id

  smalldataset1 = GroupA.sample(frac = 1, random_state = 0) #Randomise order
  smalldataset = smalldataset1.drop(columns = 'age') #ignore age for this NN
  #split dataset into training & testing
  train_dataset = smalldataset.sample(frac=0.8,random_state=0) #Split into training/testing
  test_dataset = smalldataset.drop(train_dataset.index)

  train_labels = train_dataset.pop("ZHL")#Separate labels
  test_labels = test_dataset.pop("ZHL")
  print(train_dataset)
  #Normalise dataset
  normed_train_data = (train_dataset.values - mean )/ deviation #normalise flux bands
  normed_test_data = (test_dataset.values - mean)/deviation
  
  
  
  train_labels = (train_labels - stats[1, 0]) / stats[1, 1] #normalise ages
  test_labels = (test_labels - stats[1, 0]) / stats[1, 1] 
  
  train_labels = train_labels.values #This dislikes pd arrays it seems
  test_labels = test_labels.values

  return normed_train_data, train_labels,  normed_test_data, test_labels



best_run, best_model = optim.minimize(model=create_model,
                                      data=data,
                                      algo=tpe.suggest,
                                      max_evals=5,
                                      trials=Trials())
X_train, Y_train, X_test, Y_test = data()
print("Evalutation of best performing model:")
print(best_model.evaluate(X_test, Y_test))
print("Best performing model chosen hyper-parameters:")
print(best_run)