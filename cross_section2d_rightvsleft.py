
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

def func3d((x, y), a, b, c, d, e, f, h, i, j, k):
            return a*(1 + b*x + c*y + d*x*x + e*y*y + f*x*y + h*x*x*x + i*y*y*y + j*x*x*y + k*x*y*y)


def makeDiscr(discr_dict,outfile,xtitle="discriminator",nbins=30,x_min=0,x_max=1):
    c = ROOT.TCanvas("c","c",800,500)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gPad.SetMargin(0.15,0.1,0.2,0.1)
    #ROOT.gPad.SetLogy(1)
    #ROOT.gPad.SetGrid(1,1)
    ROOT.gStyle.SetGridColor(17)
    l = TLegend(0.17,0.75,0.88,0.88)
    l.SetTextSize(0.055)
    l.SetBorderSize(0)
    l.SetFillStyle(0)
    l.SetNColumns(2)

    colors = [2,4,8,ROOT.kCyan+2]
    counter = 0
    for leg,discr in discr_dict.iteritems():
        a = Hist(nbins, x_min, x_max)
        #fill_hist_with_ndarray(a, discr)
        a.fill_array(discr)
        a.SetLineColor(colors[counter])
        a.SetLineWidth(2)
        a.GetXaxis().SetTitle(xtitle)
        a.GetXaxis().SetLabelSize(0.05)
        a.GetXaxis().SetTitleSize(0.05)
        a.GetXaxis().SetTitleOffset(1.45)
        a.GetYaxis().SetTitle("a.u.")
        a.GetYaxis().SetTickSize(0)
        a.GetYaxis().SetLabelSize(0)
        a.GetYaxis().SetTitleSize(0.06)
        a.GetYaxis().SetTitleOffset(0.9)
        a.Scale(1./a.Integral())
        #a.GetYaxis().SetRangeUser(0.00001,100)
        a.GetYaxis().SetRangeUser(0,0.2)
        if counter == 0: a.draw("hist")
        else: a.draw("same hist")
        l.AddEntry(a,leg,"l")
        counter += 1
        
    l.Draw("same")
    c.SaveAs(outfile)

#samples = ['cQQ1']
out_dir = 'XsecContours_cQQ1_ctt1/'
samples = ['cQQ1','ctt1']
#samples = ['cQQ1']
couplings = ['-3','-2','-1','0','+1','+2','+3']
#couplings = ['-3','-1','0','+1','+3']
#model = load_model('RNN_multiclass/model_checkpoint_save.hdf5')
#wp = 0.55  #0.5506506
wp_specific1 = 0.85
wp_specific2 = 0.85
wp_lvsr = 0.4
#wp_cut = 900 #0.6999969090965289
wp = 0.85
wp_cut = 1200
lum = 302.3*1000.0 # pb^-1
#lum = 1.0 # pb^-1

#classes_dict = {
#                        'SM': 0,
#                        'cQQ1' : 1,
#                        'cQQ8' : 2,
#                        'cQt1' : 3,
#                        'cQt8' : 4,
#                        'ctt1' : 5
#            }

classes_dict = {
                        'SM': 0,
                        'cQQ1' : 1,
                        'cQQ8' : 1,
                        'cQt1' : 2,
                        'cQt8' : 2,
                        'ctt1' : 3
            }

#x_sec = {'cQQ1': [0.01541,0.003964,0.001164,0.0002886,0.0002575,0.0003095,0.00127,0.004182,0.0155],'cQQ8': [0.001889, 0.0006504, 0.0003482, 0.0002612, 0.0002575, 0.0002679, 0.0003825, 0.0007219, 0.001978],'cQt1': [0.02115, 0.005472, 0.001567, 0.0003145, 0.0002575, 0.0003105, 0.001553, 0.005465, 0.02112], 'cQt8': [0.005063, 0.00141, 0.0005217, 0.0002623,0.0002575, 0.0002842, 0.0006235, 0.001619, 0.005482], 'ctt1': [0.06074, 0.01526, 0.003965, 0.0003928,0.0002575,0.0004321,0.004174,0.01562,0.06168]} # pb
#xsec_error = [0.00001,0.00001,0.00001,0.00001,0.00000000000001,0.00001,0.00001,0.00001,0.00001] # pb
xsec_frac_error = 0.01
n=0
input_dir = 'inference_samples_cQQ1_ctt1_preprocessed/'
frac_syst = 0.5
#x_sec = np.load('cross_interference/cross_section.npy')
x_sec = np.load('cQQ1_ctt1_inference_cross/cross_section.npy')


n=0
uncs = {'xsec': 0.0,'xsec_discrim': 0.0, 'xsec_discrim_sr1': 0.0,'xsec_discrim_sr2': 0.0, 'xsec_cut': 0.0}
limits = {'xsec': [],'xsec_discrim': [], 'xsec_discrim_sr1': [],'xsec_discrim_sr2': [],'xsec_cut': []}
SM = {'xsec': 0.0,'xsec_discrim': 0.0, 'xsec_discrim_sr1': 0.0,'xsec_discrim_sr2': 0.0, 'xsec_cut': 0.0}
events_pure = []
events_cut = []
events_discs = []
events_discs_sr1 = []
events_discs_sr2 = []
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
                        if z is not '0' or k is not '0':
                                    name = samples[0]+'_'+z+'_'+samples[1]+'_'+k
                                    X_flat = np.load(input_dir+name+'features_flat.npy')
                                    discr_dict = np.load(input_dir+name+'prediction_inference_model.npy')
				    discr_dict2 = np.load(input_dir+name+'prediction_rightleft.npy')
				    discr_spec = 1-discr_dict2[:,0]
                                    discr = 1-discr_dict[:,0]
                                    discr_left = (discr_dict2[:,classes_dict['cQQ1']])/(1-discr_dict2[:,0]-discr_dict2[:,3])
				    discr_right = ( discr_dict2[:,classes_dict['cQt1']] )/(1-discr_dict2[:,0]-discr_dict2[:,3])
                                    #discr_specific1 = (discr_dict[:,classes_dict['cQQ1']])/(discr_dict[:,classes_dict['cQQ1']]+discr_dict[:,0])
				    #discr_specific2 = ( discr_dict[:,classes_dict['cQt1']] )/(discr_dict[:,classes_dict['cQt1']]+discr_dict[:,0])
                                    SR1 = (discr_left > wp_lvsr) & (discr_spec > wp_specific1)
                                    SR2 = (discr_left < wp_lvsr) & (discr_spec > wp_specific2)
                                    #SR1 = (discr_specific1 > wp_specific1)
                                    #SR2 = (discr_specific2 > wp_specific2)
                                    sample_size = discr.shape[0]
                                    events_pure.append(x_sec[count1][count2])
                                    events_discs.append(x_sec[count1][count2]*discr[discr > wp].shape[0]/sample_size)
                                    events_discs_sr1.append(x_sec[count1][count2]*discr[SR1].shape[0]/sample_size)
                                    events_discs_sr2.append(x_sec[count1][count2]*discr[SR2].shape[0]/sample_size)
                                    events_cut.append(x_sec[count1][count2]*discr[X_flat[:,1] > wp_cut].shape[0]/sample_size)
                                    events_coupling1.append(int(z))
                                    events_coupling2.append(int(k))
                                    print int(z)
                                    print int(k)
                                    n+=1
                        else:
                                    X_flat = np.load('SM_only/features_flat.npy')
                                    discr_dict = np.load('SM_only/prediction_inference_model.npy')
				    discr_dict2 = np.load('SM_only/prediction_rightleft.npy')	
				    discr_spec = 1-discr_dict2[:,0]
                                    discr = 1-discr_dict[:,0]
                                    discr_left = (discr_dict2[:,classes_dict['cQQ1']])/(1-discr_dict2[:,0]-discr_dict2[:,3])
				    discr_right = ( discr_dict2[:,classes_dict['cQt1']] )/(1-discr_dict2[:,0]-discr_dict2[:,3])
                                    discr_specific1 = (discr_dict[:,classes_dict['cQQ1']])/(discr_dict[:,classes_dict['cQQ1']]+discr_dict[:,0])
				    discr_specific2 = ( discr_dict[:,classes_dict['cQt1']] )/(discr_dict[:,classes_dict['cQt1']]+discr_dict[:,0])
                                    SR1 = (discr_left > wp_lvsr) & (discr_spec > wp_specific1)
                                    SR2 = (discr_left < wp_lvsr) & (discr_spec > wp_specific2)
                                    #SR1 = (discr_specific1 > wp_specific1)
                                    #SR2 = (discr_specific2 > wp_specific2)
                                    sample_size = discr.shape[0]
                                    events_pure.append(x_sec[count1][count2])
                                    events_discs.append(x_sec[count1][count2]*discr[discr > wp].shape[0]/sample_size)
                                    events_discs_sr1.append(x_sec[count1][count2]*discr[SR1].shape[0]/sample_size)
                                    events_discs_sr2.append(x_sec[count1][count2]*discr[SR2].shape[0]/sample_size)
                                    events_cut.append(x_sec[count1][count2]*discr[X_flat[:,1] > wp_cut].shape[0]/sample_size)
                                    events_coupling1.append(int(z))
                                    events_coupling2.append(int(k))
                                    uncs['xsec'] = sqrt( (sqrt(lum*x_sec[count1][count2])/lum)**2 + (frac_syst*x_sec[count1][count2])**2 )
                                    uncs['xsec_discrim'] = sqrt( ( sqrt(lum*x_sec[count1][count2]*discr[discr > wp].shape[0]/sample_size)/lum )**2 + (frac_syst*x_sec[count1][count2]*discr[discr > wp].shape[0]/sample_size)**2 )
			            uncs['xsec_discrim_sr1'] = sqrt( ( sqrt(lum*x_sec[count1][count2]*discr[SR1].shape[0]/sample_size)/lum )**2 + (frac_syst*x_sec[count1][count2]*discr[SR1].shape[0]/sample_size)**2 )
                                    uncs['xsec_discrim_sr2'] = sqrt( ( sqrt(lum*x_sec[count1][count2]*discr[SR2].shape[0]/sample_size)/lum )**2 + (frac_syst*x_sec[count1][count2]*discr[SR2].shape[0]/sample_size)**2 )
                                    uncs['xsec_cut'] = sqrt( (sqrt(lum*x_sec[count1][count2]*discr[X_flat[:,1] > wp_cut].shape[0]/sample_size)/lum)**2 + (frac_syst*x_sec[count1][count2]*discr[X_flat[:,1] > wp_cut].shape[0]/sample_size)**2)
                                    SM['xsec'] = x_sec[count1][count2]
                                    SM['xsec_discrim'] = x_sec[count1][count2]*discr[discr > wp].shape[0]/sample_size
                                    SM['xsec_discrim_sr1'] = x_sec[count1][count2]*discr[SR1].shape[0]/sample_size
                                    SM['xsec_discrim_sr2'] = x_sec[count1][count2]*discr[SR2].shape[0]/sample_size
                                    SM['xsec_cut'] = x_sec[count1][count2]*discr[X_flat[:,1] > wp_cut].shape[0]/sample_size
                                    n+=1
                        
print 'converting lists'
xsec_pure = np.asarray(events_pure)        
xsec_discs = np.asarray(events_discs)
xsec_discs_sr1 = np.asarray(events_discs_sr1)
xsec_discs_sr2 = np.asarray(events_discs_sr2)
xsec_cut = np.asarray(events_cut)
coupling_strengths1 = np.asarray(events_coupling1)
coupling_strengths2 = np.asarray(events_coupling2)
objects = {'xsec': xsec_pure,'xsec_discrim': xsec_discs,'xsec_discrim_sr1': xsec_discs_sr1,'xsec_discrim_sr2': xsec_discs_sr2,'xsec_cut': xsec_cut}
#objects = {'xsec': xsec_pure, 'xsec_discrim': xsec_discs,'xsec_cut': xsec_cut} 
print 'finished converting lists'


print xsec_pure
print coupling_strengths1
print coupling_strengths2
print xsec_discs
print xsec_cut

SM_discr_dict = np.load('SM_only/prediction_inference_model.npy')
SM_discr = 1-SM_discr_dict[:,0]
rEFT_discr_dict = np.load('inference_samples_three_preprocessed/cQQ1_0_ctt1_-3prediction_inference_model.npy')
rEFT_discr = 1-rEFT_discr_dict[:,0]
lEFT_discr_dict = np.load('inference_samples_three_preprocessed/cQQ1_-3_ctt1_0prediction_inference_model.npy')
lEFT_discr = 1-lEFT_discr_dict[:,0]

SM_discr_spec = SM_discr_dict[:,1]/(SM_discr_dict[:,1]+SM_discr_dict[:,3])
rEFT_discr_spec = rEFT_discr_dict[:,1]/(rEFT_discr_dict[:,1]+rEFT_discr_dict[:,3])
lEFT_discr_spec = lEFT_discr_dict[:,1]/(lEFT_discr_dict[:,1]+lEFT_discr_dict[:,3])

makeDiscr({"SM":SM_discr_spec,"R_EFT":rEFT_discr_spec,"L_EFT":lEFT_discr_spec}, "discr_SMvsEFT_spec.pdf","discriminator P(t_{L})/(P(t_{L}) + P(t_{R}))")

makeDiscr({"SM":SM_discr,"R_EFT":rEFT_discr,"L_EFT":lEFT_discr}, "discr_SMvsEFT.pdf","discriminator P(t_{L}) + P(t_{R})")

SM_X_flat = np.load('SM_only/features_flat.npy')
SM_ht = SM_X_flat[:,1]
rEFT_X_flat = np.load('inference_samples_three_preprocessed/cQQ1_0_ctt1_-3features_flat.npy')
rEFT_ht = rEFT_X_flat[:,1]
lEFT_X_flat = np.load('inference_samples_three_preprocessed/cQQ1_-3_ctt1_0features_flat.npy')
lEFT_ht = lEFT_X_flat[:,1]

rEFT_discr_dict_test = np.load('inference_samples_preprocessed/ctt1-20prediction.npy')
lEFT_discr_dict_test = np.load('inference_samples_preprocessed/cQt1-20prediction.npy')
rEFT_discr_spec_test = rEFT_discr_dict_test[:,1]/(rEFT_discr_dict_test[:,1]+rEFT_discr_dict_test[:,3])
lEFT_discr_spec_test = lEFT_discr_dict_test[:,1]/(lEFT_discr_dict_test[:,1]+lEFT_discr_dict_test[:,3])


makeDiscr({"SM":SM_discr_spec,"R_EFT":rEFT_discr_spec_test,"L_EFT":lEFT_discr_spec_test}, "discr_SMvsEFT_spec_test.pdf","discriminator P(t_{L})/(P(t_{L}) + P(t_{R}))")

makeDiscr({"SM":SM_ht,"R_EFT":rEFT_ht,"L_EFT":lEFT_ht}, "Ht_SMvsEFT.pdf","H_{t}",nbins=30,x_min=0,x_max=2000)
#xsec_pure = np.reshape(xsec_pure,(7,7))
#xsec_discs = np.reshape(xsec_discs,(7,7))
#xsec_discs_target = np.reshape(xsec_discs_target,(7,7))
#xsec_cut = np.reshape(xsec_cut,(7,7))

#coupling_strengths1 = np.reshape(coupling_strengths1,(7,7))
#coupling_strengths2 = np.reshape(coupling_strengths2,(7,7))
colors_dict = {'xsec': 'red', 'xsec_discrim': 'blue', 'xsec_discrim_sr1': 'green', 'xsec_discrim_sr2': 'orange', 'xsec_cut': 'black','xsec_combined':'purple'}
#colors_dict = {'xsec': 'red', 'xsec_discrim_sr1': 'green', 'xsec_discrim_sr2': 'orange'}
sr1 = np.array([])
sr2 = np.array([])

#plt.scatter(coupling_strengths1,xsec_discs)
#axes = plt.gca()
#axes.set_ylim([-0.0005,0.0005])
#plt.savefig('CQQ1vsDisc.png')

#plt.scatter(coupling_strengths1,xsec_cut)
#axes = plt.gca()
#axes.set_ylim([-0.0005,0.0005])
#plt.savefig('CQQ1vsCut.png')



#plt.scatter(coupling_strengths2,xsec_discs)
#axes = plt.gca()
#axes.set_ylim([-0.0005,0.0005])
#plt.savefig('CQt1vsDisc.png')


#plt.scatter(coupling_strengths2,xsec_cut)
#axes = plt.gca()
#axes.set_ylim([-0.0005,0.0005])
#plt.savefig('CQt1vsCut.png')


for name_sec,xsec in objects.items():
	    xsec_error = xsec_frac_error*xsec
            #intial = [0.001,1,1,1,1,1,1,1,1,1]
            intial = [0.001,1,1,1,1,1]
            print coupling_strengths1.shape
            print coupling_strengths2.shape
            print xsec.shape
            coeff, covmat = curve_fit(func,(coupling_strengths1,coupling_strengths2),xsec,p0=intial,sigma=xsec_error)
            errors = sqrt(diag(covmat))
            xmin = min(coupling_strengths1)-0.2
            xmax = max(coupling_strengths1)+0.2

            nsteps = 50
            
            ba = np.arange(xmin, xmax+ float(xmax-xmin)/float(nsteps), float(xmax-xmin)/float(nsteps))
            ka = np.arange(xmin, xmax+ float(xmax-xmin)/float(nsteps), float(xmax-xmin)/float(nsteps))
            pa = np.array([])
            for x in ba:
                        for y in ka:
                                    #p =coeff[0]*( 1 + coeff[1]*x + coeff[2]*y + coeff[3]*x*x + coeff[4]*y*y + coeff[5]*x*y + coeff[6]*x*x*x + coeff[7]*y*y*y + coeff[8]*x*x*y + coeff[9]*x*y*y )
                                    p =coeff[0]*( 1 + coeff[1]*x + coeff[2]*y + coeff[3]*x*x + coeff[4]*y*y + coeff[5]*x*y )
                                    test = (p-coeff[0])**2/((uncs[name_sec])**2)
                                    if name_sec is 'xsec_discrim_sr1':
                                                sr1 = np.append(sr1,test)
                                    if name_sec is 'xsec_discrim_sr2':
                                                sr2 = np.append(sr2,test)
                                    pa = np.append(pa,test)
                                    if np.abs(test - 3.84) < 0.1:
                                                limits[name_sec].append([x,y])

            f = open(out_dir+'output'+name_sec+'.txt', 'a')
            f.write(sample+','+str(limits[name_sec])+','+str(limits[name_sec])+'\n')
	    pa = pa.reshape((51,51))
            plt.contour(ba,ka,pa,[3.84], colors = colors_dict[name_sec])
            

comb = sr1+sr2
comb = comb.reshape((51,51))
une = np.arange(xmin, xmax+ float(xmax-xmin)/float(nsteps), float(xmax-xmin)/float(nsteps))
deux = np.arange(xmin, xmax+ float(xmax-xmin)/float(nsteps), float(xmax-xmin)/float(nsteps))
plt.contour(une,deux,comb,[5.991], colors = 'violet')


plt.savefig(out_dir+'fit_'+name_sec+'_'+sample+'.png')
plt.savefig(out_dir+'fit_'+name_sec+'_'+sample+'.pdf')
