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

def makeDiscr(discr_dict,outfile,xtitle="discriminator"):
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

    colors = [2,4,8,ROOT.kCyan+2,ROOT.kBlack,ROOT.kGreen+3]
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
right = ((Y == 3) | (Y == 4) | (Y == 5))



cut = len(Y[SM])/2
Y = Y[cut:]

labels = Y

X_jets = X_jets[cut:]
X_mu = X_mu[cut:]
X_el = X_el[cut:]
X_flat = X_flat[cut:]

X_jets_train, X_jets_test,X_mu_train, X_mu_test,X_el_train, X_el_test,X_flat_train, X_flat_test, Y_train, Y_test, y_train, y_test = train_test_split(X_jets,X_mu,X_el,X_flat, Y, labels, test_size=0.2, random_state = 930607)

SM = (Y_test == 0)[:,0]
op1 = (Y_test == 1)[:,0]
op2 = (Y_test == 2)[:,0]
op3 = (Y_test == 3)[:,0]
op4 = (Y_test == 4)[:,0]
op5 = (Y_test == 5)[:,0]

X_train = [X_jets_train,X_mu_train, X_el_train, X_flat_train]
X_test = [X_jets_test,X_mu_test,X_el_test,X_flat_test]
#model = load_model('Model_RNNwithSort/model_checkpoint_save.hdf5',custom_objects={'SortLayer':SortLayer()})
model = load_model('RNN_multiclass/model_checkpoint_save.hdf5')
discr_dict = model.predict(X_test)
print discr_dict.shape
print y_test.shape
discr = 1-discr_dict[:,0]



SM_discr = discr[SM]
EFT_discr1 = discr[op1]
EFT_discr2 = discr[op2]
EFT_discr3 = discr[op3]
EFT_discr4 = discr[op4]
EFT_discr5 = discr[op5]

discr_cQQ1 = discr_dict[:,1]/(discr_dict[:,0]+discr_dict[:,1])
SM_discr_cQQ1 = discr_cQQ1[SM]
EFT_discr_cQQ1 = discr_cQQ1[op1]

discr_cQQ8 = discr_dict[:,2]/(discr_dict[:,0]+discr_dict[:,2])
SM_discr_cQQ8 = discr_cQQ8[SM]
EFT_discr_cQQ8 = discr_cQQ8[op2]

discr_cQt1 = discr_dict[:,3]/(discr_dict[:,0]+discr_dict[:,3])
SM_discr_cQt1 = discr_cQt1[SM]
EFT_discr_cQt1 = discr_cQt1[op3]

discr_cQt8 = discr_dict[:,4]/(discr_dict[:,0]+discr_dict[:,4])
SM_discr_cQt8 = discr_cQt8[SM]
EFT_discr_cQt8 = discr_cQt8[op4]

discr_ctt1 = discr_dict[:,5]/(discr_dict[:,0]+discr_dict[:,5])
SM_discr_ctt1 = discr_ctt1[SM]
EFT_discr_ctt1 = discr_ctt1[op5]

#tL_discr = discrTL[left]
#tR_discr = discrTL[right]

makeDiscr({"cQQ1":EFT_discr1,"cQQ8":EFT_discr2,"cQt1":EFT_discr3,"cQt8":EFT_discr4,"ctt1":EFT_discr5,"SM":SM_discr}, "discr_SMvsEFT_multi.pdf","discriminator 1-SM)")
makeDiscr({"cQQ1":EFT_discr_cQQ1,"SM":SM_discr_cQQ1}, "discr_SMvscQQ1_multi.pdf","P(cQQ1)/(P(cQQ1)+P(SM))")
makeDiscr({"cQQ8":EFT_discr_cQQ8,"SM":SM_discr_cQQ8}, "discr_SMvscQQ8_multi.pdf","P(cQQ8)/(P(cQQ8)+P(SM))")
makeDiscr({"cQt1":EFT_discr_cQt1,"SM":SM_discr_cQt1}, "discr_SMvscQt1_multi.pdf","P(cQt1)/(P(cQt1)+P(SM))")
makeDiscr({"cQt8":EFT_discr_cQt8,"SM":SM_discr_cQt8}, "discr_SMvscQt8_multi.pdf","P(cQt8)/(P(cQt8)+P(SM))")
makeDiscr({"ctt1":EFT_discr_ctt1,"SM":SM_discr_ctt1}, "discr_SMvsctt1_multi.pdf","P(ctt1)/(P(ctt1)+P(SM))")


