from __future__ import division

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
from numpy import polyfit, diag, sqrt
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.model_selection import train_test_split
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
from scipy.optimize import curve_fit

def func((x, y), a, b, c, d, e, f):
            return a*(1 + b*x + c*y + d*x*x + e*y*y + f*x*y)
            

#samples = ['cQQ1']
out_dir = 'XsecContours/'
samples = ['cQQ1','cQt1']
#samples = ['cQQ1']
couplings = ['-3','-2','-1','0','+1','+2','+3']
model = load_model('model_RNN_leftright_cuts_onLO/model_checkpoint_save.hdf5')
#wp = 0.55  #0.5506506
wp_specific = 0.5
wp_lvsr = 0.5
#wp_cut = 900 #0.6999969090965289
wp = 0.75
wp_cut = 1200
lum = 302.3*1000.0 # pb^-1
#lum = 1.0 # pb^-1

classes_dict = {
        'SM': 0,
        'ctt1' : 3,
        'cQQ1' : 1,
        'cQQ8' : 1,
        'cQt1' : 2,
        'cQt8' : 2
}
x_sec = {'cQQ1': [0.01541,0.003964,0.001164,0.0002886,0.0002575,0.0003095,0.00127,0.004182,0.0155],'cQQ8': [0.001889, 0.0006504, 0.0003482, 0.0002612, 0.0002575, 0.0002679, 0.0003825, 0.0007219, 0.001978],'cQt1': [0.02115, 0.005472, 0.001567, 0.0003145, 0.0002575, 0.0003105, 0.001553, 0.005465, 0.02112], 'cQt8': [0.005063, 0.00141, 0.0005217, 0.0002623,0.0002575, 0.0002842, 0.0006235, 0.001619, 0.005482], 'ctt1': [0.06074, 0.01526, 0.003965, 0.0003928,0.0002575,0.0004321,0.004174,0.01562,0.06168]} # pb
#xsec_error = [0.00001,0.00001,0.00001,0.00001,0.00000000000001,0.00001,0.00001,0.00001,0.00001] # pb
xsec_frac_error = 0.01
n=0
input_dir = 'inference_samples_two_preprocessed/'
frac_syst = 0.5

x_sec = np.load('cross_interference/cross_section.npy')


n=0
uncs = {'xsec': 0.0,'xsec_discrim': 0.0, 'xsec_discrim_target': 0.0, 'xsec_cut': 0.0}
limits = {'xsec': [],'xsec_discrim': [], 'xsec_discrim_target': [],'xsec_cut': []}
SM = {'xsec': 0.0,'xsec_discrim': 0.0, 'xsec_discrim_target': 0.0, 'xsec_cut': 0.0}
events_pure = []
events_cut = []
events_discs = []
events_discs_target = []
events_coupling1 = []
events_coupling2 = []
count1 = -1
count2 = -1
sample = 'cQQ1'
for z in couplings:
            count1 = count1 + 1
            count2 = -1
            for k in couplings:
                        count2 = count2 + 1
                        if z and k is not '0':
                                    n+=1
                        else:
                                    X_jets = np.load('SM_updated_cuts/features_jet.npy')
                                    X_mu = np.load('SM_updated_cuts/features_mu.npy')
                                    X_el = np.load('SM_updated_cuts/features_el.npy')
                                    X_flat = np.load('SM_updated_cuts/features_flat.npy')
                                    X = [X_jets,X_mu,X_el,X_flat]
                                    discr_dict = model.predict(X,batch_size=126)
                                    np.save('SM_updated_cuts/prediction_rightleft_LO.npy',discr_dict)
