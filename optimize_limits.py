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

def func(x, a, b, c):
            return a*(1 + b*x + c*x*x)
            

#samples = ['cQQ1']
out_dir = 'XsecPlots/'
samples = ['cQQ1','cQQ8','cQt1','cQt8','ctt1']
#samples = ['cQQ1']
couplings = ['-20','-10','-5','-1','0','+1','+5','+10','+20']
#model = load_model('m/model_checkpoint_save.hdf5')
wp = 0.78  #0.5506506
wp_specific = 0.385
#wp_cut = 900 #0.6999969090965289
#wp_list = np.arange(0.45,0.95,0.01)

eff = 0.4
#eff = 1.0


wp_list = np.arange(0.05,0.85,0.01)
wp_cut = 1450
#wp_list = np.arange(600,2000,50)
#wp_list = [1200]
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
for sample in samples:
            for n in range(0,len(x_sec[sample])):
                        x_sec[sample][n] = x_sec[sample][n]*eff


best_limit1 = {'cQQ1':-99.0,'ctt1':-99.0,'cQt8':-99.0,'cQt1':-99.0,'cQQ8':-99.0}
best_limit2 = {'cQQ1':99.0,'ctt1':99.0,'cQt8':99.0,'cQt1':99.0,'cQQ8':99.0}
best_wp = {'cQQ1':0.0,'ctt1':0.0,'cQt8':0.0,'cQt1':0.0,'cQQ8':0.0}
xsec_frac_error = 0.001
n=0
input_dir = 'inference_samples_preprocessed_cuts/'
frac_syst = 0.2
wp = 0.7
for wp_specific in wp_list:
   for sample in samples:
              n=0
              uncs = {'xsec': 0.0,'xsec_discrim': 0.0, 'xsec_discrim_target': 0.0, 'xsec_cut': 0.0}
              limits = {'xsec': [],'xsec_discrim': [], 'xsec_discrim_target': [],'xsec_cut': []}
              SM = {'xsec': 0.0,'xsec_discrim': 0.0, 'xsec_discrim_target': 0.0, 'xsec_cut': 0.0}
              events_pure = []
              events_cut = []
              events_discs = []
              events_discs_target = []
              events_coupling = []
              for z in couplings:
                          if z is not '0':
                                      name = sample+z
                                      X_flat = np.load(input_dir+name+'features_flat.npy')
                                      discr_dict = np.load(input_dir+name+'prediction_rightleft_LO.npy')
                                      discr = 1-discr_dict[:,0]
                                      if discr_dict.shape[0] != X_flat.shape[0]:
                                                  print 'LOL'
                                      discr_target = discr_dict[:,classes_dict[sample]]/(discr_dict[:,classes_dict[sample]]+discr_dict[:,0])
                                      sample_size = discr.shape[0]
                                      events_pure.append(x_sec[sample][n])
                                      events_discs.append(x_sec[sample][n]*discr[discr > wp].shape[0]/sample_size)
                                      events_discs_target.append(x_sec[sample][n]*discr[discr_target > wp_specific].shape[0]/sample_size)
                                      events_cut.append(x_sec[sample][n]*discr[X_flat[:,1] > wp_cut].shape[0]/sample_size)
                                      events_coupling.append(int(z))
                                      n+=1
                          else:
                                      X_flat = np.load('SM_updated_cuts/features_flat.npy')
                                      discr_dict =np.load('SM_updated_cuts/prediction_rightleft_LO.npy')
                                      discr = 1-discr_dict[:,0]
                                      discr_target = discr_dict[:,classes_dict[sample]]/(discr_dict[:,classes_dict[sample]]+discr_dict[:,0])
                                      sample_size = discr.shape[0]
                                      events_pure.append(x_sec[sample][n])
                                      events_discs.append(x_sec[sample][n]*discr[discr > wp].shape[0]/sample_size)
                                      events_discs_target.append(x_sec[sample][n]*discr[discr_target > wp_specific].shape[0]/sample_size)
                                      events_cut.append(x_sec[sample][n]*discr[X_flat[:,1] > wp_cut].shape[0]/sample_size)
                                      events_coupling.append(int(z))
                                      uncs['xsec'] = sqrt( (sqrt(lum*x_sec[sample][n])/lum)**2 + (frac_syst*x_sec[sample][n])**2 )
                                      uncs['xsec_discrim'] = sqrt( ( sqrt(lum*x_sec[sample][n]*discr[discr > wp].shape[0]/sample_size)/lum )**2 + (frac_syst*x_sec[sample][n]*discr[discr > wp].shape[0]/sample_size)**2 )
                                      uncs['xsec_discrim_target'] = sqrt( ( sqrt(lum*x_sec[sample][n]*discr[discr_target > wp_specific].shape[0]/sample_size)/lum )**2 + (frac_syst*x_sec[sample][n]*discr[discr_target > wp_specific].shape[0]/sample_size)**2 )
                                      uncs['xsec_cut'] = sqrt( (sqrt(lum*x_sec[sample][n]*discr[X_flat[:,1] > wp_cut].shape[0]/sample_size)/lum)**2 + (frac_syst*x_sec[sample][n]*discr[X_flat[:,1] > wp_cut].shape[0]/sample_size)**2)
                                      SM['xsec'] = x_sec[sample][n]
                                      SM['xsec_discrim'] = x_sec[sample][n]*discr[discr > wp].shape[0]/sample_size
                                      SM['xsec_discrim_target'] = x_sec[sample][n]*discr[discr_target > wp_specific].shape[0]/sample_size
                                      SM['xsec_cut'] = x_sec[sample][n]*discr[X_flat[:,1] > wp_cut].shape[0]/sample_size

                                      n+=1

              xsec_pure = np.asarray(events_pure)        
              xsec_discs = np.asarray(events_discs)
              xsec_discs_target = np.asarray(events_discs_target)
              xsec_cut = np.asarray(events_cut)
              coupling_strengths = np.asarray(events_coupling)
              objects = {'xsec_discrim_target': xsec_discs_target}

              for name_sec,xsec in objects.items():
                          xsec_error = xsec_frac_error*xsec
                          coeff, covmat = curve_fit(func,coupling_strengths,xsec,sigma=xsec_error)
                          errors = sqrt(diag(covmat))
                          xmin = min(coupling_strengths)-1
                          xmax = max(coupling_strengths)+1
                          ymin = min(xsec)-(max(xsec)- min(xsec))/5.
                          ymax = max(xsec)+(max(xsec)- min(xsec))/5.
                          nsteps = 50
                          x = np.arange(xmin, xmax+ float(xmax-xmin)/float(nsteps), float(xmax-xmin)/float(nsteps))
                          #y = coeff[0] + coeff[1]*x + coeff[2]*x*x
                          y =  coeff[0]*(1 + coeff[1]*x + coeff[2]*x*x)

                        
                           #print '%s: sigma=%.4f ( 1 + %.4fC + %.4f C^2 )$'%(sample,coeff[0],coeff[1],coeff[2])
                          y_1 = [(i-coeff[0])**2/((uncs[name_sec])**2) for i in y]                          
                          ymin = -2
                          ymax = 15
                          ba = np.arange(xmin, xmax+ float(xmax-xmin)/float(100000), float(xmax-xmin)/float(100000))
                          for i in ba:
                                      p = coeff[0]*(1 + coeff[1]*i + coeff[2]*i*i)
                                      #test = (p-coeff[0])**2/((uncs[name_sec])**2)
                                      test = (p-SM[name_sec])**2/((uncs[name_sec])**2)
                                      if np.abs(test - 3.84) < 0.01:
                                                  limits[name_sec].append(i)
                          f = open(out_dir+'output'+name_sec+'.txt', 'a')
                          if limits[name_sec][0] > best_limit1[sample]:
                                      best_limit1[sample] = limits[name_sec][0]
                                      best_wp[sample] = wp_specific
                          if limits[name_sec][-1] < best_limit2[sample]:
                                      best_limit2[sample] = limits[name_sec][-1]
                                      best_wp[sample] = wp_specific
                          f.write(sample+','+str(limits[name_sec][0])+','+str(limits[name_sec][-1])+'\n')

print best_limit1
print best_limit2
print best_wp
