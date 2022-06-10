import tensorflow as tf
import keras
import numpy as np
import time
import random
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Flatten, LSTM, GRU, SimpleRNN, Conv2D, MaxPooling2D, Flatten, Dropout, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from keras.regularizers import l2, l1, l1_l2
from collections import deque

from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error as mse
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow import keras
from matplotlib import pyplot as plt
from IPython.display import clear_output

import qkeras
from qkeras import *
import pickle
import os

from nnlar.datashaper import DataShaper
ds = DataShaper.from_h5("../../data/rdgap_mu140.h5")

x, x_val, x_test, y, y_val, y_test = ds()

# settings of the networks 
output = 1

nbr_batch = 64
nbr_epoch = 200
lr=0.001

time_step = 5
nbr_conv_epoch = 100 #number of epochs for the conversion 

weight_decay=0.00000001
patience_es = 12
patience_rlr = 3
delta = 0.00000001
regularizer = l2

def version(v): return (v)
versions_range = 10

def units(j): return (j+8)
units_range = 1

integer = 0

def bit_lenghts(i):
    #To change to create models with different bits parameter by defining the smallest bit lenghts 
    #Example : bits 8 ==> (i+4)*4
    return (i+1)*2

def bit_width(i): return {'bits': (i+1)*2, 'integer': integer, 'symmetric': 0, 'alpha':1}

rest_array = [8,10,12]
rest_units_range =  1
bits_range = 7


#path of the qconverted models
def qhist_path(i,k): 
    return f"tests/qhist/qmodels_conv_patiencES={patience_es}_patienceRLR={patience_rlr}_mindelta{delta},rkernel<{bit_lenghts(i)},{integer}>_rest={rest_array[k]}.pkl"

#path of the qtrained models
def qtrained_models_path(i, v, k): 
    return f"tests/qmodels/qmodels_scatch_patiencES={patience_es}_patienceRLR={patience_rlr}_mindelta{delta},rkernel<{bit_lenghts(i)},{integer}>_rest={rest_array[k]}_v{version(v)}.h5"

def qtrained_hist_path(i, v, k): 
    return f"tests/qhist/qmodels_scatch_patiencES={patience_es}_patienceRLR={patience_rlr}_mindelta{delta},rkernel<{bit_lenghts(i)},{integer}>_rest={rest_array[k]}_v{version(v)}.h5"

#path of the qref models 
def qrefmodels_path (i) : 
    return f"qmodels_epoch={nbr_conv_epoch}/qmodels<{bit_lenghts(i)},{integer}>.h5"

def quantized_model (bits, units_parameter,k):  
    rest_bit_width =  {'bits':rest_array[k], 'integer': integer, 'symmetric': 0, 'alpha':1}    

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
        qr_model = Sequential()
        qr_model.add(QSimpleRNN(units_parameter,
                            input_dim= 1,
                            activation='relu',
                            kernel_quantizer=quantized_bits(**rest_bit_width),
                            recurrent_quantizer=quantized_bits(**bits),
                            bias_quantizer=quantized_bits(**rest_bit_width)    
                            )) 
        qr_model.add(QDense(output, 
                            activation='relu',
                            kernel_quantizer=quantized_bits(**rest_bit_width),
                            bias_quantizer=quantized_bits(**rest_bit_width) ))
        qr_model.compile(loss="mse", optimizer=Adam(lr))
        history = qr_model.fit(x, y, validation_data= (x_val,y_val),epochs = 1, batch_size=nbr_batch, shuffle=True, callbacks=[early_stopping, reduce_lr])

        val_loss = history.history['val_loss'][0]

        
    qr_model.summary()

    hist = qr_model.fit(x, y, validation_data= (x_val,y_val),epochs = nbr_epoch, batch_size=nbr_batch, shuffle=True, callbacks=[early_stopping, reduce_lr])
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    return qr_model, hist

def qmodel_training (): 

    qmodels =[]
    for i in range(bits_range):
        bits_parameter = bit_width(i)
        for k in range(rest_units_range):            
            for v in range(8,9):
                if (os.path.exists(qtrained_models_path(i,v,k))==False):
                    units_parameter = 8
                    qmodel = quantized_model(bits_parameter, units_parameter,k)
                    qmodel[0].save(qtrained_models_path(i,v,k))
                    with open(qtrained_hist_path(i,v,k), 'wb') as file_pi:
                        pickle.dump(qmodel[1].history, file_pi)     
                    qmodels.append(qmodel[0])
                else : print(f'{qtrained_models_path(i,v,k)} already exists')
        print( 'bit width ', bits_parameter)        
    return qmodels
        
qmodels = qmodel_training()