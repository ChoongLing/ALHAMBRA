import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns

import bokeh
from bokeh.io import output_file, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.models import Range1d


tf.enable_eager_execution()

def norm(x):
  return (x - mean) / deviation


#Load in the stats for normalisation
stats = np.loadtxt('No_error_stats.txt')
mean = stats[:-2, 0]
deviation = stats[:-2, 1]

column_names = ['name', 'F3655', 'F3965', 'F4275', 'F4585', 'F4895', 'F5205', 'F5515', \
    'F5825', 'F6135', 'F6445', 'F6755', 'F7065', 'F7375', 'F7685', 'F7995', 'F8305', 'F8615', 'F8915', \
    'F9235', 'F9545', 'J', 'H', 'K', 'age', 'ZHL']

feature_names = column_names[0] #column for age
label_names = column_names[1:] #columns for the filters

#Import data
#incoming = pd.read_table('Data_no_error.txt', delim_whitespace = True, names = column_names)
grpa = pd.read_table('Grp1.dat', delim_whitespace = True, names = column_names)
grpb = pd.read_table('Grp2.dat', delim_whitespace = True, names = column_names)
grpc = pd.read_table('Grp3.dat', delim_whitespace = True, names = column_names)

incoming = pd.concat([grpa, grpb, grpc]) #Join together 3 groups

dataset = incoming.copy()
dataset = dataset.dropna()
  
GroupA = dataset.drop(columns = 'name') #name not needed here

smalldataset1 = GroupA.sample(frac = 1, random_state = 12345) #randomise order

train_dataset = smalldataset1.drop(columns = 'age')

#Gives the statistics of each band, not particularly relevant now
train_stats = train_dataset.describe()
train_stats.pop("ZHL")
train_stats = train_stats.transpose()

#'True' Z for the training sets
train_labels = train_dataset.pop("ZHL")
train_labels = (train_labels - stats[-1, 0])/stats[-1, 1] #normalise

#Normalise dataset
normed_train_data = norm(train_dataset)


#Build the CNN
input_spec = layers.Input(shape=(23,)) #input spectra
MidLayer = layers.Reshape((23, 1))(input_spec) #reshape for use in CNN
# CNN layers
MidLayer = layers.Conv1D(activation='relu', 
                padding="same", filters=33, kernel_size=12)(MidLayer)
MidLayer = layers.Conv1D(activation='relu',
                padding="same", filters=50, kernel_size=10)(MidLayer)
## Max pooling layer
MidLayer = layers.MaxPooling1D(pool_size=2)(MidLayer)
MidLayer =layers.Flatten()(MidLayer)#flatten for use in dense layers
# Dense layers
MidLayer = layers.Dense(units=30, activation='relu')(MidLayer)
output_label = layers.Dense(units=1, activation="linear", #output layer
                     input_dim=30,)(MidLayer)

model = keras.models.Model(input_spec, output_label)

earlystop = EarlyStopping(monitor='mean_absolute_error', patience=250)
callbacks_list = [earlystop] #use early stopping to prevent overfitting
optimizer = tf.train.RMSPropOptimizer(0.001)

model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
model.summary()

print('CNN starts training')
#Do the training of the CNN
history = model.fit(
  normed_train_data, train_labels, batch_size = 64, callbacks = callbacks_list,
  epochs=5000, validation_split = 0., verbose=0)

#tracks the values of errors, variable values etc.
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

model.save('Z_for_grp4.h5') #Save the model for testing
