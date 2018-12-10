from ROOT import TFile, TTree, TChain, TCanvas, TH1D, TLegend, gROOT, gStyle
import sys
import ROOT
from ROOT import TCanvas, TGraph
from ROOT import gROOT
import os
import time
from argparse import ArgumentParser
from array import array
from math import *
import numpy as np
from collections import Counter
import root_numpy as rootnp
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Input, Convolution1D, Concatenate, Flatten, LSTM
from keras.utils import np_utils, conv_utils
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
from rootpy.plotting import Hist
from keras import initializers
from rootpy.plotting import Hist2D
import tensorflow as tf
from sklearn.neural_network import MLPClassifier
from keras import backend as K
from keras.engine.topology import Layer
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline


class SortLayer(Layer):

    def __init__(self, kernel_initializer='glorot_uniform', **kwargs):
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_size = conv_utils.normalize_tuple(1, 1, 'kernel_size')
        super(SortLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
	channel_axis = 2
	input_dim = input_shape[channel_axis]	
      	kernel_shape = self.kernel_size + (input_dim, 1)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',trainable=True)
        super(SortLayer, self).build(input_shape)  # Be sure to call this at the end


    def compute_output_shape(self, input_shape):
        outshape=list(input_shape)
        print('compute', tuple(outshape))
        return tuple(outshape)
        
    def call(self, x):
        n_batch = tf.shape(x)[0]
        values = K.conv1d(x, self.kernel, strides = 1, padding = "valid", data_format ='channels_last', dilation_rate = 1)
        values = tf.squeeze(values, axis=2)
        values = tf.nn.softsign(values)+1
        index = tf.nn.top_k(values, x.get_shape()[1]).indices
        values = tf.expand_dims(values,axis=2)
        x = x*values
        index = tf.expand_dims(index, axis=2)
        batch_range = tf.expand_dims(tf.expand_dims(tf.range(0, n_batch), axis=1), axis=1)
        batch_range = tf.tile(batch_range, [1, x.get_shape()[1], 1])
        index_tensor = tf.concat([batch_range,index],axis=2)
        x = tf.gather_nd(x,index_tensor)
        return x


def spit_out_roc(df, label, sm = False):

    newx = np.logspace(-3, 0, 100)
    tprs = pd.DataFrame()
    scores = []
    cs = np.squeeze((label != 0))
    print cs.shape
    print df.shape
    if sm: 
    	df = df[cs[:]]
	blab = (label[cs] != 2)*1.0
    else:
    	blab = cs*1.0
    tmp_fpr, tmp_tpr, _ = roc_curve(blab, df)
    scores.append(
        roc_auc_score(blab, df)
    )
    coords = pd.DataFrame()
    coords['fpr'] = tmp_fpr
    coords['tpr'] = tmp_tpr
    clean = coords.drop_duplicates(subset=['fpr'])
    spline = InterpolatedUnivariateSpline(clean.fpr, clean.tpr,k=1)
    tprs = spline(newx)
    scores = np.array(scores)
    
    return tprs, newx

Y = np.load('numpy_array/truth.npy')    
X_jets = np.load('numpy_array/features_jet.npy')
X_mu = np.load('numpy_array/features_mu.npy')
X_el = np.load('numpy_array/features_el.npy')
X_flat = np.load('numpy_array/features_flat.npy')
print X_jets.shape
print Y.shape
SM = (Y == 0) 
left = ((Y == 1) | (Y == 2))
right = ((Y == 3) | (Y == 4) | (Y == 5))
Y[left] = 1
Y[right] = 2

cut = len(Y[SM])/2
Y = Y[cut:]
labels = Y

X_jets = X_jets[cut:]
X_mu = X_mu[cut:]
X_el = X_el[cut:]
X_flat = X_flat[cut:]

X_jets_train, X_jets_test,X_mu_train, X_mu_test,X_el_train, X_el_test,X_flat_train, X_flat_test, Y_train, Y_test, y_train, y_test = train_test_split(X_jets,X_mu,X_el,X_flat, Y, labels, test_size=0.2, random_state = 930607)
X_train = [X_jets_train,X_mu_train, X_el_train, X_flat_train]
X_test = [X_jets_test,X_mu_test,X_el_test,X_flat_test]
#model = load_model('Model_RNNwithSort/model_checkpoint_save.hdf5',custom_objects={'SortLayer':SortLayer()})
model = load_model('Model_moreComplex_RNN_noSort/model_checkpoint_save.hdf5')
discr_dict = model.predict(X_test)
print discr_dict.shape
print y_test.shape
x1, y1 = spit_out_roc(discr_dict[:,1], y_test, sm = True)
x2, y2 = spit_out_roc(discr_dict[:,1]+discr_dict[:,2], y_test, sm = False)

f = ROOT.TFile("ROCS/RNN_moreComplex_not_sorted.root", "recreate")
gr1 = TGraph( 100, x1, y1 )
gr1.SetName("roccurve_0")
gr2 = TGraph( 100, x2, y2 )
gr2.SetName("roccurve_1")
gr1.Write()
gr2.Write()
f.Write()
