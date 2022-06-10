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
lr=0.0001

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

#quantized settings
integer = 0

def bit_lenghts(i):
    #To change to create models with different bits parameter by defining the smallest bit lenghts 
    #Example : bits 8 ==> (i+4)*4
    return (i+1)*2

def bit_width(i): return {'bits': (i+1)*2, 'integer': integer, 'symmetric': 0, 'alpha':1}

rest_array = [8,10,12]
rest_units_range =  1
bits_range = 7


# path

#path of the tested models 
def models_path(v) : return f"tests/models/optimized_model.h5"      #optimized model = model_decay<class 'keras.regularizers.L2'>=1e-07_v0

#path of the qconverted models
def qmodels_path(i,v,k): return f"tests/qmodels/qmodels_conv_patiencES={patience_es}_patienceRLR={patience_rlr}_mindelta{delta},rkernel<{bit_lenghts(i)},{integer}>_rest={rest_array[k]}_v{version(v)}.h5"

#path of the qconverted models
def qhist_path(i,v,k): return f"tests/qhist/qmodels_conv_patiencES={patience_es}_patienceRLR={patience_rlr}_mindelta{delta},rkernel<{bit_lenghts(i)},{integer}>_rest={rest_array[k]}_v{version(v)}.pkl"

#path of the qtrained models
def qtrained_models_path(j, i, v ): return f"tests/qmodels_test/qtrained/qmodels_units={units(j)}_epoch={nbr_epoch}_patiencES={patience_es}_patienceRLR={patience_rlr}_delta={delta}_cp,<{bit_lenghts(i)},{integer}>v{version(v)}.h5"

#path of the qref models 
def qrefmodels_path (i) : return f"qmodels_epoch={nbr_conv_epoch}/qmodels<{bit_lenghts(i)},{integer}>.h5"


from nnlar.datashaper import DataShaper
ds = DataShaper.from_h5("../../data/rdgap_mu140.h5")

x, x_val, x_test, y, y_val, y_test = ds()


def quantized_conv_model (bits, units_parameter, model_to_convert,k):  
    
    rest_bit_width =  {'bits':rest_array[k], 'integer': integer, 'symmetric': 0, 'alpha':1}    
    dense_bit_width =  {'bits':16, 'integer': integer, 'symmetric': 0, 'alpha':1}    
    checkpoint_filepath = '/atlas/bonnet/Desktop/code/internship_CPPM/rnn/model_checkpoint'

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                    save_weights_only=True,
                                                                    monitor='val_loss',
                                                                    mode='min',
                                                                    save_best_only=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                    patience=patience_es, 
                                                    restore_best_weights=True, 
                                                    min_delta=delta,
                                                    mode='min')

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                patience= patience_rlr, min_lr=0.000001, min_delta=delta, verbose=1)
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
                        kernel_quantizer=quantized_bits(**dense_bit_width),
                        bias_quantizer=quantized_bits(**dense_bit_width) ))

    qr_model.compile(loss="mse", optimizer=Adam(lr))
    
    qr_model.summary()

    #using the weight from the classic network as a base
    qr_model.set_weights(model_to_convert.get_weights())
    hist = qr_model.fit(x, y, validation_data= (x_val,y_val),epochs = nbr_conv_epoch, batch_size=nbr_batch, shuffle=True, callbacks=[model_checkpoint_callback, early_stopping, reduce_lr])
    lr_change = []
    for i in range (len(hist.history['lr'])-1):
    
        if (hist.history['lr'][i]==hist.history['lr'][i+1]):
            lr_change.append(None)
        else: 
            lr_change.append(hist.history['val_loss'][i+1])
    plt.plot(lr_change, 'X')
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title(qmodels_path(101,101,k))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['lr_changed','train', 'test'])
    plt.show()
    return qr_model, hist

def qmodel_conv_training (): 

    qmodels =[]
    qhist = []
    for i in range(bits_range):
        bits_parameter = bit_width(i)
        k=2
        v=2
        if (os.path.exists(qmodels_path(i,v,k))==False):
            qmodel = quantized_conv_model(bits_parameter, 8, tf.keras.models.load_model(models_path(0)),k)
            qmodel[0].save(qmodels_path(i,v,k))
            with open(qhist_path(i,v,k), 'wb') as file_pi:
                pickle.dump(qmodel[1].history, file_pi)                
            qmodels.append(qmodel[0])
            qhist.append(qmodel[1])
        else : print(f'{qmodels_path(i,v,k)} already exists')
        v=3
        if (os.path.exists(qmodels_path(i,v,k))==False):
            qmodel = quantized_conv_model(bits_parameter, 8, tf.keras.models.load_model(models_path(0)),k)
            qmodel[0].save(qmodels_path(i,v,k))
            with open(qhist_path(i,v,k), 'wb') as file_pi:
                pickle.dump(qmodel[1].history, file_pi)                
            qmodels.append(qmodel[0])
            qhist.append(qmodel[1])
        else : print(f'{qmodels_path(i,v,k)} already exists')
        print( 'bit width ', bits_parameter)        
    return qmodels
        
qmodels = qmodel_conv_training()