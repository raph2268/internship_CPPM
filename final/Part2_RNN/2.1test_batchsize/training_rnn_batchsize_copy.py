import tensorflow as tf
import keras
import numpy as np
import time
import random
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Flatten, LSTM, GRU, SimpleRNN, Conv2D, MaxPooling2D, Flatten, Dropout, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

from collections import deque


from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error as mse
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow import keras
from matplotlib import pyplot as plt
from IPython.display import clear_output
import os
import qkeras
from qkeras import *

import hls4ml
import nnlar
from nnlar.datashaper import DataShaper

from nnlar.datashaper import DataShaper
ds = DataShaper.from_h5("../../../data/rdgap_mu140.h5")

x, x_val, x_test, y, y_val, y_test = ds()

boosted_model =  tf.keras.models.load_model('/atlas/bonnet/Desktop/code/internship_CPPM/pb_file')

boosted_model.summary()


# settings of the networks 
output = 1

nbr_batch = 64
nbr_epoch = 200
lr=0.001

nbr_conv_epoch = 4 #number of epochs for the conversion 

patience_es = 8
patience_rlr = 5
delta = 0.0000001

def version(v): return (v)
versions_range = 8

def units(j): return (j+8)
units_range = 1


# path

#path of the tested models 
def models_path(j,v) : return f"tests/models/models_units={units(j)}_batchsize={nbr_batch}v{version(v)}.h5"

#path of the tested models 
def predicts_path(j,v) : return f"tests/predicts/models_units={units(j)}_batchsize={nbr_batch}v{version(v)}.h5"


def normal_model (units_parameter):

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                    patience=patience_es, 
                                                    restore_best_weights=True, 
                                                    min_delta=delta,
                                                    mode='min')

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                patience=patience_rlr, min_lr=0.000001, min_delta=delta, verbose=1)

    val_loss = 1


    # restart training if the model does not start converging by the end of first epoch
    while val_loss > 0.0005:
        print("testing new weights")
        r_model = Sequential()
        r_model.add(SimpleRNN(units_parameter, activation='relu', input_shape=(5, 1), return_sequences=False, name='SimpleRNN'))
        r_model.add(Dense(output, activation='relu',name='dense'))
        r_model.compile(loss="mean_squared_error", optimizer=Adam(lr))
        history = r_model.fit(x,y,validation_data=(x_val,y_val), epochs=1, batch_size=nbr_batch, shuffle=True, callbacks=[early_stopping, reduce_lr])
        val_loss = history.history['val_loss'][0]


    r_model.summary()

    r_model.fit(x,y,validation_data=(x_val,y_val), epochs=nbr_epoch, batch_size=nbr_batch, shuffle=True, callbacks=[early_stopping, reduce_lr])


    return r_model

def model_training (): 
    models =[]

    for j in range(units_range):
        units_parameter = units(j)
        for v in range(versions_range):

            if (os.path.exists(models_path(j,v))==False):
                model = normal_model(units_parameter)
                models.append(model)
                model.save(models_path(j,v))
                print('number of units ', units_parameter)  
            else : print(f'{models_path(j,v)} already exists')       
    return models

models = model_training()