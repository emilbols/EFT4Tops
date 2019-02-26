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
#model = load_model('RNN_multiclass/model_checkpoint_save.hdf5')
#wp = 0.55  #0.5506506
wp_specific = 0.55
wp_lvsr = 0.5
#wp_cut = 900 #0.6999969090965289
#wp = 0.55
wp_cut = 1200
lum = 302.3*1000.0 # pb^-1
#lum = 1.0 # pb^-1

classes_dict = {
            'SM': 0,
            'cQQ1' : 1,
            'cQQ8' : 2,
            'cQt1' : 3,
            'cQt8' : 4,
            'ctt1' : 5
}
x_sec = {'cQQ1': [0.01541,0.003964,0.001164,0.0002886,0.0002575,0.0003095,0.00127,0.004182,0.0155],'cQQ8': [0.001889, 0.0006504, 0.0003482, 0.0002612, 0.0002575, 0.0002679, 0.0003825, 0.0007219, 0.001978],'cQt1': [0.02115, 0.005472, 0.001567, 0.0003145, 0.0002575, 0.0003105, 0.001553, 0.005465, 0.02112], 'cQt8': [0.005063, 0.00141, 0.0005217, 0.0002623,0.0002575, 0.0002842, 0.0006235, 0.001619, 0.005482], 'ctt1': [0.06074, 0.01526, 0.003965, 0.0003928,0.0002575,0.0004321,0.004174,0.01562,0.06168]} # pb
#xsec_error = [0.00001,0.00001,0.00001,0.00001,0.00000000000001,0.00001,0.00001,0.00001,0.00001] # pb
xsec_frac_error = 0.01
n=0
input_dir = 'inference_samples_two_preprocessed/'
frac_syst = 0.5

x_sec = np.load('cross_interference/cross_section.npy')


n=0
sample = 'cQQ1'
lol = [0.55,0.6,0.65,0.7,0.75,0.8]
for wp in lol:
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

 for z in couplings:
             count1 = count1 + 1
             count2 = -1
             for k in couplings:
                         count2 = count2 + 1
                         if z and k is not '0':
                                     name = samples[0]+'_'+z+'_'+samples[1]+'_'+k
                                     X_flat = np.load(input_dir+name+'features_flat.npy')
                                     discr_dict = np.load(input_dir+name+'prediction.npy')
                                     discr = 1-discr_dict[:,0]
                                     discr_left = (discr_dict[:,classes_dict['cQQ1']]+discr_dict[:,classes_dict['cQQ8']])/(1-discr_dict[:,0])
                                     discr_right = (discr_dict[:,classes_dict['ctt1']]+discr_dict[:,classes_dict['cQt8']]+discr_dict[:,classes_dict['cQt1']])/(1-discr_dict[:,0])
                                     discr_target = (discr_dict[:,classes_dict['cQQ1']]+discr_dict[:,classes_dict['cQQ8']])/(discr_dict[:,classes_dict['cQQ1']]+discr_dict[:,classes_dict['cQQ8']]+discr_dict[:,0])
                                     sample_size = discr.shape[0]
                                     events_pure.append(x_sec[count1][count2])
                                     events_discs.append(x_sec[count1][count2]*discr[discr > wp].shape[0]/sample_size)
                                     events_discs_target.append(x_sec[count1][count2]*discr[discr_target > wp_specific].shape[0]/sample_size)
                                     events_cut.append(x_sec[count1][count2]*discr[X_flat[:,1] > wp_cut].shape[0]/sample_size)
                                     events_coupling1.append(int(z))
                                     events_coupling2.append(int(k))
                                     n+=1
                         else:
                                     X_flat = np.load('SM_only/features_flat.npy')
                                     discr_dict = np.load('SM_only/prediction.npy')
                                     discr = 1-discr_dict[:,0]
                                     discr_target = (discr_dict[:,classes_dict['cQQ1']]+discr_dict[:,classes_dict['cQQ8']])/(discr_dict[:,classes_dict['cQQ1']]+discr_dict[:,classes_dict['cQQ8']]+discr_dict[:,0])
                                     sample_size = discr.shape[0]
                                     events_pure.append(x_sec[count1][count2])
                                     events_discs.append(x_sec[count1][count2]*discr[discr > wp].shape[0]/sample_size)
                                     events_discs_target.append(x_sec[count1][count2]*discr[discr_target > wp_specific].shape[0]/sample_size)
                                     events_cut.append(x_sec[count1][count2]*discr[X_flat[:,1] > wp_cut].shape[0]/sample_size)
                                     events_coupling1.append(int(z))
                                     events_coupling2.append(int(k))
                                     uncs['xsec'] = sqrt( (sqrt(lum*x_sec[count1][count2])/lum)**2 + (frac_syst*x_sec[count1][count2])**2 )
                                     uncs['xsec_discrim'] = sqrt( ( sqrt(lum*x_sec[count1][count2]*discr[discr > wp].shape[0]/sample_size)/lum )**2 + (frac_syst*x_sec[count1][count2]*discr[discr > wp].shape[0]/sample_size)**2 )
                                     uncs['xsec_discrim_target'] = sqrt( ( sqrt(lum*x_sec[count1][count2]*discr[discr_target > wp_specific].shape[0]/sample_size)/lum )**2 + (frac_syst*x_sec[count1][count2]*discr[discr_target > wp_specific].shape[0]/sample_size)**2 )
                                     uncs['xsec_cut'] = sqrt( (sqrt(lum*x_sec[count1][count2]*discr[X_flat[:,1] > wp_cut].shape[0]/sample_size)/lum)**2 + (frac_syst*x_sec[count1][count2]*discr[X_flat[:,1] > wp_cut].shape[0]/sample_size)**2)
                                     SM['xsec'] = x_sec[count1][count2]
                                     SM['xsec_discrim'] = x_sec[count1][count2]*discr[discr > wp].shape[0]/sample_size
                                     SM['xsec_discrim_target'] = x_sec[count1][count2]*discr[discr_target > wp_specific].shape[0]/sample_size
                                     SM['xsec_cut'] = x_sec[count1][count2]*discr[X_flat[:,1] > wp_cut].shape[0]/sample_size
                                     n+=1

 print 'converting lists'
 xsec_pure = np.asarray(events_pure)        
 xsec_discs = np.asarray(events_discs)
 xsec_discs_target = np.asarray(events_discs_target)
 xsec_cut = np.asarray(events_cut)
 coupling_strengths1 = np.asarray(events_coupling1)
 coupling_strengths2 = np.asarray(events_coupling2)
 #objects = {'xsec': xsec_pure,'xsec_discrim': xsec_discs,'xsec_discrim_target': xsec_discs_target,'xsec_cut': xsec_cut}
 objects = {'xsec_discrim': xsec_discs}
 print 'finished converting lists'

 #xsec_pure = np.reshape(xsec_pure,(7,7))
 #xsec_discs = np.reshape(xsec_discs,(7,7))
 #xsec_discs_target = np.reshape(xsec_discs_target,(7,7))
 #xsec_cut = np.reshape(xsec_cut,(7,7))

 #coupling_strengths1 = np.reshape(coupling_strengths1,(7,7))
 #coupling_strengths2 = np.reshape(coupling_strengths2,(7,7))
 colors_dict = {'xsec_discrim': 'black'}


 for name_sec,xsec in objects.items():
             xsec_error = xsec_frac_error*xsec
             intial = [0.001,1,1,1,1,1]
             coeff, covmat = curve_fit(func,(coupling_strengths1,coupling_strengths2),xsec,p0=intial,sigma=xsec_error)
             errors = sqrt(diag(covmat))
             xmin = min(coupling_strengths1)-1
             xmax = max(coupling_strengths1)+1

             nsteps = 50

             ba = np.arange(xmin, xmax+ float(xmax-xmin)/float(nsteps), float(xmax-xmin)/float(nsteps))
             ka = np.arange(xmin, xmax+ float(xmax-xmin)/float(nsteps), float(xmax-xmin)/float(nsteps))
             pa = np.array([])
             for x in ba:
                         for y in ka:
                                     p =coeff[0]*( 1 + coeff[1]*x + coeff[2]*y + coeff[3]*x*x + coeff[4]*y*y + coeff[5]*x*y ) 
                                     test = (p-coeff[0])**2/((uncs[name_sec])**2)
                                     pa = np.append(pa,test)
                                     if np.abs(test - 2.6896) < 0.1:
                                                 limits[name_sec].append([x,y])

             f = open(out_dir+'output'+name_sec+'.txt', 'a')
             f.write(sample+','+str(limits[name_sec])+','+str(limits[name_sec])+'\n')
             pa = pa.reshape((51,51))
             plt.contour(ba,ka,pa,[2.6896], colors = colors_dict[name_sec])
             print limits[name_sec]

plt.savefig(out_dir+'fit_'+name_sec+'_'+sample+'.png')
plt.savefig(out_dir+'fit_'+name_sec+'_'+sample+'.pdf')
