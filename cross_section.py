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
model = load_model('model_RNN_leftright/model_checkpoint_save.hdf5')
#wp = 0.55  #0.5506506
#wp_specific = {'cQt8': 0.88, 'cQt1': 0.78, 'ctt1': 0.79, 'cQQ1': 0.71, 'cQQ8': 0.77}

#50%
#wp_specific = {'cQt8': 0.8800000000000003, 'cQt1': 0.7800000000000002, 'ctt1': 0.6600000000000001, 'cQQ1': 0.7500000000000002, 'cQQ8': 0.7700000000000002}

#Interference_LO
#wp_specific = {'cQt8': 0.8899999999999996, 'cQt1': 0.8899999999999996, 'ctt1': 0.8399999999999996, 'cQQ1': 0.8899999999999996, 'cQQ8': 0.8699999999999996}
#Inteference_LO eff og 20% syst
#wp_specific = {'cQt8': 0.8300000000000002, 'cQt1': 0.8400000000000002, 'ctt1': 0.7200000000000002, 'cQQ1': 0.8200000000000002, 'cQQ8': 0.8200000000000002}

#Shallow eff og 20% syst
#wp_specific = {'cQt8': 0.7000000000000002, 'cQt1': 0.6000000000000002, 'ctt1': 0.5100000000000001, 'cQQ1': 0.7200000000000002, 'cQQ8': 0.5500000000000002}
wp_specific = {'cQt8': 0.7400000000000002, 'cQt1': 0.6300000000000001, 'ctt1': 0.5900000000000002, 'cQQ1': 0.7100000000000002, 'cQQ8': 0.6600000000000001}

#30%
#wp_specific = {'cQt8': 0.880000000000000, 'cQt1': 0.7800000000000002, 'ctt1': 0.55, 'cQQ1': 0.6500000000000001, 'cQQ8': 0.7700000000000002}
#wp_cut = 900 #0.6999969090965289
#wp = 0.795
#wp = {'cQt8': 0.8100000000000003, 'cQt1': 0.8100000000000003, 'ctt1': 0.7800000000000002, 'cQQ1': 0.8100000000000003, 'cQQ8': 0.8200000000000003}
#wp = {'cQt8': 0.6300000000000003, 'cQt1': 0.5100000000000002, 'ctt1': 0.43000000000000016, 'cQQ1': 0.6300000000000003, 'cQQ8': 0.6300000000000003}

#rightleftLO
#wp = {'cQt8': 0.45999999999999985, 'cQt1': 0.45999999999999985, 'ctt1': 0.15999999999999998, 'cQQ1': 0.2899999999999999, 'cQQ8': 0.2699999999999999}

#with 20% syst and eff 40%
wp = {'cQt8': 0.5200000000000001, 'cQt1': 0.4600000000000001, 'ctt1': 0.33, 'cQQ1': 0.44000000000000006, 'cQQ8': 0.5000000000000001}


wp_cut = 950
lum = 302.3*1000.0 # pb^-1
#lum = 137.3*1000.0 # pb^-1

classes_dict = {
        'SM': 0,
        'ctt1' : 3,
        'cQQ1' : 1,
        'cQQ8' : 1,
        'cQt1' : 2,
        'cQt8' : 2
}
x_sec = {'cQQ1': [0.01541,0.003964,0.001164,0.0002886,0.0002575,0.0003095,0.00127,0.004182,0.0155],'cQQ8': [0.001889, 0.0006504, 0.0003482, 0.0002612, 0.0002575, 0.0002679, 0.0003825, 0.0007219, 0.001978],'cQt1': [0.02115, 0.005472, 0.001567, 0.0003145, 0.0002575, 0.0003105, 0.001553, 0.005465, 0.02112], 'cQt8': [0.005063, 0.00141, 0.0005217, 0.0002623,0.0002575, 0.0002842, 0.0006235, 0.001619, 0.005482], 'ctt1': [0.06074, 0.01526, 0.003965, 0.0003928,0.0002575,0.0004321,0.004174,0.01562,0.06168]} # pb

#x_sec = {'cQQ1': [0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001],'cQQ8': [0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001],'cQt1': [0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001], 'cQt8': [0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001], 'ctt1': [0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001]} # pb

#eff = 0.85
#eff = 0.2
eff = 0.4
for sample in samples:
            for n in range(0,len(x_sec[sample])):
                        x_sec[sample][n] = x_sec[sample][n]*eff

#xsec_error = [0.00001,0.00001,0.00001,0.00001,0.00000000000001,0.00001,0.00001,0.00001,0.00001] # pb
xsec_frac_error = 0.01
n=0
input_dir = 'inference_samples_preprocessed/'
frac_syst = 0.2
M_cut = 3000
open(out_dir+'outputxsec.txt', 'w').close()
open(out_dir+'outputxsec_cut.txt', 'w').close()
open(out_dir+'outputxsec_discrim.txt', 'w').close()
open(out_dir+'outputxsec_discrim_target.txt', 'w').close()

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
            coeff = 1.0
            for z in couplings:
                        if z is not '0':
                                    name = sample+z
                                    X_flat = np.load(input_dir+name+'features_flat.npy')
                                    discr_dict = np.load(input_dir+name+'prediction_rightleft_LO.npy')
                                    discr_dict2 = np.load(input_dir+name+'prediction_highlevel_LO.npy')
                                    #discr = 1-discr_dict[:,0]
                                    discr = discr_dict[:,classes_dict[sample]]/(discr_dict[:,classes_dict[sample]]+discr_dict[:,0])
                                    discr_target = discr_dict2[:,classes_dict[sample]]/(discr_dict2[:,classes_dict[sample]]+discr_dict2[:,0])
                                    sample_size = discr.shape[0]
                                    events_pure.append(x_sec[sample][n]*(discr[X_flat[:,1] < M_cut].shape[0]/sample_size))
                                    events_discs.append(x_sec[sample][n]*discr[(discr > wp[sample]) & (X_flat[:,1] < M_cut)].shape[0]/sample_size)
                                    events_discs_target.append(x_sec[sample][n]*discr[(discr_target > wp_specific[sample]) & (X_flat[:,1] < M_cut)].shape[0]/sample_size)
                                    events_cut.append(x_sec[sample][n]*discr[(X_flat[:,1] > wp_cut) & (X_flat[:,1] < M_cut)].shape[0]/sample_size)
                                    events_coupling.append(int(z))
                                    n+=1
                        else:
                                    X_flat = np.load('SM_LO_only/features_flat.npy')
                                    discr_dict = np.load('SM_LO_only/prediction_rightleft_LO.npy')
                                    discr_dict2 = np.load('SM_LO_only/prediction_highlevel_LO.npy')
                                    #discr = 1-discr_dict[:,0]
                                    discr = discr_dict[:,classes_dict[sample]]/(discr_dict[:,classes_dict[sample]]+discr_dict[:,0])
                                    discr_target = discr_dict2[:,classes_dict[sample]]/(discr_dict2[:,classes_dict[sample]]+discr_dict2[:,0])
                                    sample_size = discr.shape[0]
                                    events_pure.append(coeff*x_sec[sample][n]*(discr[X_flat[:,1] < M_cut].shape[0]/sample_size))
                                    print discr[X_flat[:,1] < M_cut].shape[0]/sample_size
                                    events_discs.append(x_sec[sample][n]*discr[(discr > wp[sample]) & (X_flat[:,1] < M_cut)].shape[0]/sample_size)
                                    events_discs_target.append(x_sec[sample][n]*discr[(discr_target > wp_specific[sample]) & (X_flat[:,1] < M_cut)].shape[0]/sample_size)
                                    events_cut.append(x_sec[sample][n]*discr[(X_flat[:,1] > wp_cut) & (X_flat[:,1] < M_cut)].shape[0]/sample_size)
                                    events_coupling.append(int(z))
                                    uncs['xsec'] = sqrt( (sqrt(lum*coeff*x_sec[sample][n]*(discr[X_flat[:,1] < M_cut].shape[0]/sample_size))/lum)**2 + (frac_syst*coeff*x_sec[sample][n]*(discr[X_flat[:,1] < M_cut].shape[0]/sample_size))**2 )
                                    uncs['xsec_discrim'] = sqrt( ( sqrt(lum*x_sec[sample][n]*discr[(discr > wp[sample]) & (X_flat[:,1] < M_cut)].shape[0]/sample_size)/lum )**2 + (frac_syst*x_sec[sample][n]*discr[(discr > wp[sample]) & (X_flat[:,1] < M_cut)].shape[0]/sample_size)**2 )
				    uncs['xsec_discrim_target'] = sqrt( ( sqrt(lum*x_sec[sample][n]*discr[(discr_target > wp_specific[sample]) & (X_flat[:,1] < M_cut)].shape[0]/sample_size)/lum )**2 + (frac_syst*x_sec[sample][n]*discr[(discr_target > wp_specific[sample]) & (X_flat[:,1] < M_cut)].shape[0]/sample_size)**2 )
                                    uncs['xsec_cut'] = sqrt( (sqrt(lum*x_sec[sample][n]*discr[(X_flat[:,1] > wp_cut) & (X_flat[:,1] < M_cut)].shape[0]/sample_size)/lum)**2 + (frac_syst*x_sec[sample][n]*discr[(X_flat[:,1] > wp_cut) & (X_flat[:,1] < M_cut)].shape[0]/sample_size)**2)
                                    SM['xsec'] = x_sec[sample][n]*coeff*(discr[X_flat[:,1] < M_cut].shape[0]/sample_size)
                                    SM['xsec_discrim'] = x_sec[sample][n]*discr[(discr > wp[sample]) & (X_flat[:,1] < M_cut)].shape[0]/sample_size
                                    SM['xsec_discrim_target'] = x_sec[sample][n]*discr[(discr_target > wp_specific[sample]) & (X_flat[:,1] < M_cut)].shape[0]/sample_size
                                    SM['xsec_cut'] = x_sec[sample][n]*discr[(X_flat[:,1] > wp_cut) & (X_flat[:,1] < M_cut)].shape[0]/sample_size
                                                
                                    n+=1
                        
            xsec_pure = np.asarray(events_pure)        
            xsec_discs = np.asarray(events_discs)
            xsec_discs_target = np.asarray(events_discs_target)
            xsec_cut = np.asarray(events_cut)
            coupling_strengths = np.asarray(events_coupling)
            objects = {'xsec': xsec_pure,'xsec_discrim': xsec_discs,'xsec_discrim_target': xsec_discs_target,'xsec_cut': xsec_cut}
            print SM['xsec']*lum
            print uncs['xsec']/SM['xsec']
            #objects = {'xsec_discrim': xsec_discs,'xsec_discrim_target': xsec_discs_target,'xsec_cut': xsec_cut}
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
                        
                        plt.plot(x,y,c="r")
                        
                                    # plot points with error bars
                        plt.errorbar(coupling_strengths, xsec, yerr = xsec_error, fmt= 'o', color = "b")
                        plt.axis([xmin, xmax, ymin, ymax])
                        plt.grid(True)
                        plt.xlabel(sample, fontsize = 15)
                        plt.ylabel('cross section [pb]', fontsize = 15)
                        
                                    # draw some things on the canvas
            
                        plt.text(xmin, ymax+(ymax-ymin)/float(20), r'$\sigma=%.4f ( 1 + (%.4f) \ C + (%.4f) \ C^2 )$'%(coeff[0],coeff[1],coeff[2]), fontsize=20, color="r")
                        plt.text(xmin + (xmax-xmin)/float(20), ymax-1.5*(ymax-ymin)/float(20), r'%i test'%(xsec[0]), fontsize=17, color="b")
                                    
                        plt.savefig(out_dir+'fit_'+name_sec+'_'+sample+'.png')
                        plt.savefig(out_dir+'fit_'+name_sec+'_'+sample+'.pdf')
            
                        plt.close()

                        #print '%s: sigma=%.4f ( 1 + %.4fC + %.4f C^2 )$'%(coupling,coeff[0],coeff[1]/coeff[0],coeff[2]/coeff[0])
                        print '%s: sigma=%.4f ( 1 + %.4fC + %.4f C^2 )$'%(sample,coeff[0],coeff[1],coeff[2])
                        c = ROOT.TCanvas("c","c",800,700)
                        y_1 = [(i-coeff[0])**2/((uncs[name_sec])**2) for i in y]                          
                        ymin = -2
                        ymax = 15
                        gr_chi2 = ROOT.TGraph(len(x),array('d',x),array('d',y_1))
                        gr_chi2.SetLineWidth(2)
                        gr_chi2.SetLineColor(1)
                        gr_chi2.SetLineStyle(1)
                        gr_chi2.Draw()
                        gr_chi2.SetMinimum(ymin)
                        gr_chi2.SetMaximum(ymax)
                        gr_chi2.GetYaxis().SetTitleSize(0.085)
                        gr_chi2.GetYaxis().SetTitleOffset(0.75)
                        gr_chi2.GetYaxis().SetLabelSize(0.075)
                        gr_chi2.GetXaxis().CenterTitle()
                        gr_chi2.GetXaxis().SetTitleSize(0.085)
                        gr_chi2.GetXaxis().SetTitleOffset(1.3)
                        gr_chi2.GetXaxis().SetLabelSize(0.085)
                        gr_chi2.GetXaxis().SetRangeUser(xmin+1,xmax-1)
                        c.SaveAs(out_dir+'chi2'+name_sec+'_'+sample+'.png')
                        c.SaveAs(out_dir+'chi2'+name_sec+'_'+sample+'.pdf')
                        ba = np.arange(xmin, xmax+ float(xmax-xmin)/float(100000), float(xmax-xmin)/float(100000))
                        for i in ba:
                                    p = coeff[0]*(1 + coeff[1]*i + coeff[2]*i*i)
                                    #test = (p-coeff[0])**2/((uncs[name_sec])**2)
                                    test = (p-SM[name_sec])**2/((uncs[name_sec])**2)
                                    if np.abs(test - 3.84) < 0.01:
                                                limits[name_sec].append(i)
                        f = open(out_dir+'output'+name_sec+'.txt', 'a')
                        f.write(sample+','+str(limits[name_sec][0])+','+str(limits[name_sec][-1])+'\n')
                        print limits[name_sec]
