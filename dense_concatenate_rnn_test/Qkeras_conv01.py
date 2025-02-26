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

# settings of the networks 
output = 1

nbr_batch = 64
nbr_epoch = 200
lr=0.001

time_step = 5
nbr_conv_epoch = 100 #number of epochs for the conversion 

weight_decay=0.00000001
patience_es = 7
patience_rlr = 5
delta = 0.00000001

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
def models_path(v) : return f"tests/models/models_patiencES=12_patienceRLR={patience_rlr}_mindelta{delta}_v{version(v)}.h5"      

#path of the tested models 
def hist_path(v) : return f"tests/hist/models_patiencES=12_patienceRLR={patience_rlr}_mindelta{delta}_v{version(v)}.pkl"     

#path of the qconverted models
def qmodels_path(i,v): return f"tests/qmodels/qmodels_lr={lr}_patiencES={patience_es}_patienceRLR={patience_rlr}_mindelta{delta},hetero<{bit_lenghts(i)},{integer}>v{version(v)}.h5"
def qpredicts_path(i,v): return f"tests/qpredicts/qmodels_lr={lr}_patiencES={patience_es}_patienceRLR={patience_rlr}_mindelta{delta},hetero<{bit_lenghts(i)},{integer}>v{version(v)}.npy"

#path of the qconverted models
def qhist_path(i,v): return f"tests/qhist/qmodels_lr={lr}_patiencES={patience_es}_patienceRLR={patience_rlr}_mindelta{delta},hetero<{bit_lenghts(i)},{integer}>v{version(v)}.pkl"

#path of the qtrained models
def qtrained_models_path(j, i, v ): return f"tests/qmodels_test/qtrained/qmodels_units={units(j)}_epoch={nbr_epoch}_patiencES={patience_es}_patienceRLR={patience_rlr}_delta={delta}_cp,<{bit_lenghts(i)},{integer}>v{version(v)}.h5"

#path of the qref models 
refmodels_path = f"tests/models/models_patiencES=12_patienceRLR={patience_rlr}_mindelta{delta}_v4.h5"      

from nnlar.datashaper import DataShaper
ds = DataShaper.from_h5("../data/rdgap_mu140.h5")

X, X_valid, X_test, y, y_valid, y_test = ds(seq_len=30)

X_past = X[:,:25,:]
X_valid_past = X_valid[:,:25,:]
X_test_past = X_test[:,:25,:]

X = X[:,25:,:]
X_valid = X_valid[:,25:,:]
X_test = X_test[:,25:,:]

rest_bit_width =  {'bits':  12, 'integer': integer, 'symmetric': 0, 'alpha':1}

dense_bit_width = {'bits':  16, 'integer': integer, 'symmetric': 0, 'alpha':1}
def quantized_conv_model (bits, units_parameter, model_to_convert):  
    
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
    
    val_loss = 1


    # restart training if the model does not start converging by the end of first epoch
    while val_loss > 0.0005:    
        inputs_seq = keras.Input(shape=(5,1))
        inputs_past = keras.Input(shape=(25,))

        x = qkeras.QDense(1, activation="relu",
                            kernel_quantizer=quantized_bits(**dense_bit_width),
                            bias_quantizer=quantized_bits(**dense_bit_width))(inputs_past)

        x = keras.layers.Reshape((1,1))(x)

        x =keras.layers.Concatenate(axis=1)([x, inputs_seq])
        x = QSimpleRNN(units_parameter, 
                            input_dim= 1,
                            kernel_quantizer=quantized_bits(**rest_bit_width),
                            recurrent_quantizer=quantized_bits(**bits),
                            bias_quantizer=quantized_bits(**rest_bit_width),    
                            activation='relu')(x)

        outputs = QDense(output, 
                            activation='relu',
                            kernel_quantizer=quantized_bits(**dense_bit_width),
                            bias_quantizer=quantized_bits(**dense_bit_width))(x)

        qr_model = keras.Model(inputs=[inputs_past, inputs_seq], outputs=outputs, name="past_data_model_1")

        qr_model.compile(loss='mse', optimizer='adam')
    
        #using the weight from the classic network as a base
        qr_model.set_weights(model_to_convert.get_weights())
        history = qr_model.fit([X_past, X],y,validation_data=([X_valid_past, X_valid],y_valid),epochs = 1, batch_size=nbr_batch, shuffle=True, callbacks=[model_checkpoint_callback, early_stopping, reduce_lr])
        val_loss = history.history['val_loss'][0]

    hist = qr_model.fit([X_past, X],y,validation_data=([X_valid_past, X_valid],y_valid),epochs = nbr_conv_epoch, batch_size=nbr_batch, shuffle=True, callbacks=[model_checkpoint_callback, early_stopping, reduce_lr])
    lr_change = []
    for i in range (len(hist.history['lr'])-1):
    
        if (hist.history['lr'][i]==hist.history['lr'][i+1]):
            lr_change.append(None)
        else: 
            lr_change.append(hist.history['val_loss'][i+1])
    plt.plot(lr_change, 'X')
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title(qmodels_path(101,0))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['lr_changed','train', 'test'])
    plt.show()
    return qr_model, hist
def exist(path,modeltopred):
    if (os.path.exists(path)==False):
        np.save(path, modeltopred.predict([X_test_past, X_test]))
     
    else : print(f'{path} already exists')

def qmodel_conv_training (): 

    qmodels =[]
    qhist = []
    for i in range(bits_range):
        tmp_qmodel=0
        bits_parameter = bit_width(i)
        for j in range(units_range):
            for v in range (versions_range):
                if (os.path.exists(qmodels_path(i,v))==False):
                    print("training of :", qmodels_path(i,v))
                    qmodel = quantized_conv_model(bits_parameter, units(j), tf.keras.models.load_model(refmodels_path))
                    qmodel[0].save(qmodels_path(i,v))
                    with open(qhist_path(i,v), 'wb') as file_pi:
                        pickle.dump(qmodel[1].history, file_pi)                
                    qmodels.append(qmodel[0])
                    qhist.append(qmodel[1])
                    tmp_qmodel = qmodel[0]
                else : 
                    print(f'{qmodels_path(i,v)} already exists')
                    tmp_qmodel=qkeras.utils.load_qmodel(qmodels_path(i,v))
                exist(qpredicts_path(i,v),tmp_qmodel)
        print( 'bit width ', bits_parameter)        
    return qmodels
        
qmodels = qmodel_conv_training()