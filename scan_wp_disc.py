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
#from keras.utils.visualize_util import plot
from numpy.lib.recfunctions import stack_arrays
from numpy import polyfit, diag, sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.model_selection import train_test_split
import pickle
from rootpy.plotting import Hist
from keras import initializers
from rootpy.plotting import Hist2D
from sklearn.neural_network import MLPClassifier
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import curve_fit
from keras.models import load_model
import tensorflow as tf

def func(x, a, b, c):
            return a*(1 + b*x + c*x*x)
            

#samples = ['cQQ1']
samples = ['cQQ1','cQQ8','cQt1','cQt8','ctt1']
couplings = ['-20','-10','-5','-1','0','+1','+5','+10','+20']
wp = 0.80  #0.5506506
#wp_specific = 0.55
wp_list = [0.6,0.7,0.8,0.9,0.95]
#wp_cut = 674.942741394043 #0.6999969090965289
lum = 302.3*1000.0 # pb^-1
#lum = 1.0 # pb^-1
model = load_model('RNN_multiclass/model_checkpoint_save.hdf5')

classes_dict = {
        'SM': 0,
        'ctt1' : 1,
        'cQQ1' : 2,
        'cQQ8' : 3,
        'cQt1' : 4,
        'cQt8' : 5
}

x_sec = {'cQQ1': [0.01541,0.003964,0.001164,0.0002886,0.0003314,0.0003095,0.00127,0.004182,0.0155],'cQQ8': [0.001889, 0.0006504, 0.0003482, 0.0002612, 0.0003314, 0.0002679, 0.0003825, 0.0007219, 0.001978],'cQt1': [0.02115, 0.005472, 0.001567, 0.0003145, 0.0003314, 0.0003105, 0.001553, 0.005465, 0.02112], 'cQt8': [0.005063, 0.00141, 0.0005217, 0.0002623,0.0003314, 0.0002842, 0.0006235, 0.001619, 0.005482], 'ctt1': [0.06074, 0.01526, 0.003965, 0.0003928,0.0003314,0.0004321,0.004174,0.01562,0.06168]} # pb
xsec_error = [0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001] # pb
n=0
input_dir = 'inference_samples_preprocessed/'
frac_syst = 0.5


for sample in samples:
            n=0
            uncs = {'xsec': 0.0,'xsec_discrim': 0.0, 'xsec_discrim_target': 0.0, 'xsec_cut': 0.0}
            limits = {'xsec': [],'xsec_discrim': [], 'xsec_discrim_target': [],'xsec_cut': []}
            SM = {'xsec': 0.0,'xsec_discrim': 0.0, 'xsec_discrim_target': 0.0, 'xsec_cut': 0.0}
            events_discs_target = []
            events_discs = []
            events_cut = []
            events_coupling = []
            events_pure = []
                        for z in couplings:
                                    if z is not '0':
                                                name = sample+z
                                                X_jets = np.load(input_dir+name+'features_jet.npy')
                                                X_mu = np.load(input_dir+name+'features_mu.npy')
                                                X_el = np.load(input_dir+name+'features_el.npy')
                                                X_flat = np.load(input_dir+name+'features_flat.npy')
                                                X = [X_jets,X_mu,X_el,X_flat]
						discr_dict = model.predict(X)
                                                discr = 1-discr_dict[:,0]
                                                discr_target = discr_dict[:,classes_dict[sample]]/(discr_dict[:,classes_dict[sample]]+discr_dict[:,0])
                                                sample_size = discr.shape[0]
                                                for wp in wp_list:
                                                            events_discs.append(x_sec[sample][n]*discr[discr > wp].shape[0]/sample_size)
                                                            events_coupling.append(int(z))
                                                n+=1
                                    else:
                                                X_jets = np.load('SM_only/features_jet.npy')
                                                X_mu = np.load('SM_only/features_mu.npy')
                                                X_el = np.load('SM_only/features_el.npy')
                                                X_flat = np.load('SM_only/features_flat.npy')
                                                X = [X_jets,X_mu,X_el,X_flat]
                                                discr_dict = model.predict(X)
                                                discr = 1-discr_dict[:,0]
                                                sample_size = discr.shape[0]
                                                for wp in wp_list:
                                                            events_coupling.append(int(z))
                                                            events_discs.append(x_sec[sample][n]*discr[discr > wp].shape[0]/sample_size)
                                                            uncs['xsec_discrim'].append(sqrt( ( sqrt(lum*x_sec[sample][n]*discr[discr > wp].shape[0]/sample_size)/lum )**2 + (frac_syst*x_sec[sample][n]*discr[discr > wp].shape[0]/sample_size)**2 ) )
                                                            SM['xsec_discrim'].append(x_sec[sample][n]*discr[discr > wp].shape[0]/sample_size)
                                                n+=1
                        
                        xsec_pure = np.asarray(events_pure)
                        xsec_cut = np.asarray(events_cut)
                        xsec_error = np.asarray(xsec_error)
                        coupling_strengths = np.asarray(events_coupling)

                        objects = {'xsec_cut': xsec_cut}
            
                        for name_sec,xsec in objects.items():
                                    coeff, covmat = curve_fit(func,coupling_strengths,xsec,sigma=xsec_error)
                                    errors = sqrt(diag(covmat))
                                    xmin = min(coupling_strengths)-1
                                    xmax = max(coupling_strengths)+1
                                    ymin = min(xsec)-(max(xsec)- min(xsec))/5.
                                    ymax = max(xsec)+(max(xsec)- min(xsec))/5.
                                    nsteps = 50
                                    x = np.arange(xmin, xmax+ float(xmax-xmin)/float(nsteps), float(xmax-xmin)/float(nsteps))
                                    #y = coeff[0] + coeff[1]*x + coeff[2]*x*x
                                    y = coeff[0]*(1 + coeff[1]*x + coeff[2]*x*x)
                        
                                    ba = np.arange(xmin, xmax+ float(xmax-xmin)/float(2000), float(xmax-xmin)/float(2000))
                                    for i in ba:
                                                p = coeff[0]*(1 + coeff[1]*i + coeff[2]*i*i)
                                                test = (p-SM[name_sec])**2/((uncs[name_sec])**2)
                                                if np.abs(test - 4.0) < 0.1:
                                                            limits[name_sec].append(i)
                                    print wp_cut
                                    print limits[name_sec]
