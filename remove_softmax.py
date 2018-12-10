
import sys
import os
import time
from argparse import ArgumentParser
from array import array
from math import *
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from keras import initializers
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Input, Convolution1D, Concatenate, Flatten
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import SGD,Adam
from keras.regularizers import l1, l2
from keras.regularizers import l1, l2
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization
#from keras.utils.visualize_util import plot
from numpy.lib.recfunctions import stack_arrays
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.cross_validation import train_test_split
import pickle
from sklearn.neural_network import MLPClassifier


from keras import backend as K
from keras.engine.topology import Layer


OutputDir = 'Model_denseBased'

nclasses = 3
dropoutRate = 0.1
adam = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
Inputs = [Input(shape=(8,5)),Input(shape=(3,5)),Input(shape=(3,5)),Input(shape=(13,))]

jets = BatchNormalization(momentum=0.6,name='jets_input_batchnorm') (Inputs[0])
muons = BatchNormalization(momentum=0.6,name='muons_input_batchnorm')     (Inputs[1])
elec = BatchNormalization(momentum=0.6,name='elec_input_batchnorm')     (Inputs[2])
globalvars = BatchNormalization(momentum=0.6,name='globalvars_input_batchnorm')     (Inputs[3])


jets = Flatten()(jets)

muons = Flatten()(muons)

elec = Flatten()(elec)

x = Concatenate()( [globalvars,jets,muons,elec])
x = Dense(200,activation='relu',kernel_initializer='lecun_uniform',name='dense_0')(x)
x = Dropout(dropoutRate)(x)
x = Dense(100,activation='relu',kernel_initializer='lecun_uniform',name='dense_1')(x)
x = Dropout(dropoutRate)(x)
x = Dense(100,activation='relu',kernel_initializer='lecun_uniform',name='dense_2')(x)
x = Dropout(dropoutRate)(x)
pred=Dense(nclasses, activation='linear',kernel_initializer='lecun_uniform',name='ID_pred')(x)

model = Model(inputs=Inputs,outputs=pred)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
print(model.summary())

model_withSoft=load_model(OutputDir+'/model_checkpoint_save.hdf5')
model_withSoft.save_weights(OutputDir+'/training_weights.h5')
model.load_weights(OutputDir+'/training_weights.h5')
model.save(OutputDir+'/Dense_nosoftmax.h5')
