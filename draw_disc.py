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

def makeDiscr(discr_dict,outfile,xtitle="discriminator"):
    c = ROOT.TCanvas("c","c",800,500)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gPad.SetMargin(0.15,0.1,0.2,0.1)
    #ROOT.gPad.SetLogy(1)
    #ROOT.gPad.SetGrid(1,1)
    ROOT.gStyle.SetGridColor(17)
    l = TLegend(0.17,0.75,0.88,0.88)
    l.SetTextSize(0.025)
    l.SetBorderSize(0)
    l.SetFillStyle(0)
    l.SetNColumns(2)

    colors = [2,4,8,ROOT.kCyan+2,1]
    counter = 0
    for leg,discr in discr_dict.iteritems():
        a = Hist(30, 0, 1)
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


Y = np.load('numpy_array/truth.npy')    
X_jets = np.load('numpy_array/features_jet.npy')
X_mu = np.load('numpy_array/features_mu.npy')
X_el = np.load('numpy_array/features_el.npy')
X_flat = np.load('numpy_array/features_flat.npy')
print X_jets.shape
print Y.shape
SM = (Y == 0) 
left = ((Y == 1) | (Y == 2))
leftright = ((Y == 3) | (Y == 4) )
right = (Y == 5)
#Y[left] = 1
#Y[leftright] = 2
#Y[right] = 3


labels = Y



X_jets_train, X_jets_test,X_mu_train, X_mu_test,X_el_train, X_el_test,X_flat_train, X_flat_test, Y_train, Y_test, y_train, y_test = train_test_split(X_jets,X_mu,X_el,X_flat, Y, labels, test_size=0.2, random_state = 930607)

SM = (Y_test == 0)[:,0]
left = ((Y_test == 1) | (Y_test == 2))[:,0]
leftright = ((Y_test == 3) | (Y_test == 4))[:,0]
right = (Y_test == 5)[:,0]

op1 = (Y_test == 1)[:,0]
op2 = (Y_test == 2)[:,0]
op3 = (Y_test == 3)[:,0]
op4 = (Y_test == 4)[:,0]
op5 = (Y_test == 5)[:,0]

X_train = [X_jets_train,X_mu_train, X_el_train, X_flat_train]
X_test = [X_jets_test,X_mu_test,X_el_test,X_flat_test]
model = load_model('RNN_multiclass/model_checkpoint_save.hdf5')
#model = load_model('model_RNN_leftright/model_checkpoint_save.hdf5')
discr_dict = model.predict(X_test)
print discr_dict.shape
print y_test.shape
discr = discr_dict[:,1]+discr_dict[:,2]
discrTL =(discr_dict[:,1]+discr_dict[:,2])/(discr_dict[:,1]+discr_dict[:,2]+discr_dict[:,3]+discr_dict[:,4]+discr_dict[:,5])

discrTLR =(discr_dict[:,3])/(discr_dict[:,1]+discr_dict[:,2]+discr_dict[:,3]+discr_dict[:,4]+discr_dict[:,5])

discrTLvsSM =discr_dict[:,1]/(discr_dict[:,1]+discr_dict[:,0])


SM_discr = discr[SM]
EFT_discr = discr[~SM]

EFT_tL_discr = discr[left]

EFT_tR_discr = discr[right]


tL_discr = discrTL[left]
tcQt1_discr = discrTL[op3]
tcQt8_discr = discrTL[op4]
tR_discr = discrTL[op5]
SM_left_right_discr = discrTL[SM]

t2L_discr = discrTLR[left]
t2cQt1_discr = discrTLR[op3]
t2cQt8_discr = discrTLR[op4]
t2R_discr = discrTLR[right]
SM2_left_right_discr = discrTLR[SM]


discr_cQQ1 = discr_dict[:,1]/(discr_dict[:,2]+discr_dict[:,1])
cQQ8_discr_cQQ1 = discr_cQQ1[op2]
cQQ1_discr_cQQ1 = discr_cQQ1[op1]

discr_cQt1 = discr_dict[:,3]/(discr_dict[:,3]+discr_dict[:,4])
cQt8_discr_cQt1 = discr_cQt1[op4]
cQt1_discr_cQt1 = discr_cQt1[op3]

discr_cQt1 = discr_dict[:,3]/(discr_dict[:,3]+discr_dict[:,4])
cQt8_discr_cQt1 = discr_cQt1[op4]
cQt1_discr_cQt1 = discr_cQt1[op3]

discr_ctt1 = discr_dict[:,5]/(discr_dict[:,1]+discr_dict[:,5])
ctt1_discr_ctt1 = discr_ctt1[op5]
cQQ1_discr_ctt1 = discr_ctt1[op1]

makeDiscr({"EFT":EFT_discr,"SM":SM_discr}, "discr_SMvsEFT.pdf","discriminator P(t_{L}) + P(t_{R})")

makeDiscr({"#splitline{EFT with}{cQQ1+cQQ8}":tL_discr, "#splitline{EFT with}{ctt1}":tR_discr, "#splitline{EFT with}{cQt1}":tcQt1_discr,"#splitline{EFT with}{cQt8}":tcQt8_discr,"SM":SM_left_right_discr},"discr_tL_withEveryClass.pdf","discriminator #frac{P_{cQQ1+cQQ8}}{P_{cQQ1+cQQ8}+P_{cQt1}+P_{cQt8}+ P_{ctt1}}")


makeDiscr({"#splitline{EFT with}{cQQ1+cQQ8}":t2L_discr, "#splitline{EFT with}{ctt1}":t2R_discr, "#splitline{EFT with}{cQt1}":t2cQt1_discr,"#splitline{EFT with}{cQt8}":t2cQt8_discr,"SM":SM2_left_right_discr},"discr_tcQt1_withEveryClass.pdf","discriminator #frac{P_{cQt1}}{P_{cQQ1+cQQ8}+P_{cQt1}+P_{cQt8}+ P_{ctt1}}")

makeDiscr({"#splitline{EFT with}{cQQ1}":cQQ1_discr_cQQ1, "#splitline{EFT with}{cQQ8}":cQQ8_discr_cQQ1},"discr_cQQ1vsCQQ8.pdf","discriminator #frac{P_{cQQ1}}{P_{cQQ1}+P_{cQQ8}}")

makeDiscr({"#splitline{EFT with}{cQt1}":cQt1_discr_cQt1, "#splitline{EFT with}{cQt8}":cQt8_discr_cQt1},"discr_cQt1vsCQt8.pdf","discriminator #frac{P_{cQt1}}{P_{cQt1}+P_{cQt8}}")

makeDiscr({"#splitline{EFT with}{ctt1}":ctt1_discr_ctt1, "#splitline{EFT with}{cQQ1}":cQQ1_discr_ctt1},"discr_cQQ1vsctt1.pdf","discriminator #frac{P_{ctt1}}{P_{ctt1}+P_{cQQ1}}")


makeDiscr({"#splitline{EFT with}{left-handed top}":EFT_tL_discr, "#splitline{EFT with}{right-handed top}":EFT_tR_discr,"SM":SM_discr}, "discr_SMvstLvstR.pdf","discriminator P(t_{L}) + P(t_{R})")
