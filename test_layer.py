from ROOT import TFile, TTree, TChain, TCanvas, TH1D, TLegend, gROOT, gStyle
import sys
import ROOT
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
from keras.layers import Dense, Activation, Dropout, Input, Convolution1D, Concatenate, Flatten
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
import tensorflow as tf
from rootpy.plotting import Hist2D
from keras import initializers
from sklearn.neural_network import MLPClassifier
from tensorflow.python.framework.ops import get_gradient_function

from keras import backend as K
from keras.engine.topology import Layer

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
                                      name='kernel')
        super(SortLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        x = K.print_tensor(x,'input = ')
      	values = K.print_tensor(K.conv1d(x, self.kernel, strides = 1, padding = "valid", data_format ='channels_last', dilation_rate = 1),'values =')
        k = values.shape[1]
        values = tf.squeeze(values,axis=2)
	order = K.print_tensor(tf.nn.top_k(values, k = k, sorted=True).indices, 'order =')
	x = K.print_tensor(tf.gather(x,order,axis=1),'gather =')
        x = K.print_tensor(tf.slice(x,[0,0,0,0],[-1,1,-1,-1]),'slice =')
        x = K.print_tensor(tf.squeeze(x,axis=1),'final = ')
        print x.shape
        return x


def draw_roc(df, df2, label, color, draw_unc=False, ls='-', draw_auc=True, flavour = False):
        newx = np.logspace(-3, 0, 100)
        tprs = pd.DataFrame()
        scores = []
        if flavour:
                cs = ( (df['isC'] == 0) & (df['isCC'] == 0) & (df['isGCC'] == 0) )
        else:
                cs = ( (df['isUD'] == 0) & (df['isS'] == 0) & (df['isG'] == 0) )
        df = df[cs]
        df2 = df2[cs]
        tmp_fpr, tmp_tpr, _ = roc_curve(np.clip(df['isB']+df['isBB']+df['isLeptonicB_C']+df['isLeptonicB']+df['isGBB'],0,1), df2['prob_isBB']+df2['prob_isB'])
        scores.append(
        roc_auc_score(np.clip(df['isB']+df['isBB']+df['isLeptonicB_C']+df['isLeptonicB']+df['isGBB'],0,1), df2['prob_isB']+df2['prob_isBB'])
        )
        coords = pd.DataFrame()
        coords['fpr'] = tmp_fpr
        coords['tpr'] = tmp_tpr
        clean = coords.drop_duplicates(subset=['fpr'])
        spline = InterpolatedUnivariateSpline(clean.fpr, clean.tpr,k=1)
        tprs = spline(newx)
        scores = np.array(scores)
        auc = ' AUC: %.3f +/- %.3f' % (scores.mean(), scores.std()) if draw_auc else ''
        plt.plot(tprs, newx, label=label + auc, c=color, ls=ls)


def makeROC(fpr, tpr, thresholds,AUC,outfile,signal_label, background_label):
	
	c = TCanvas("c","c",700,600)
	ROOT.gPad.SetMargin(0.15,0.07,0.15,0.05)
	ROOT.gPad.SetLogy(0)
	ROOT.gPad.SetGrid(1,1)
	ROOT.gStyle.SetGridColor(17)
	
	roc = ROOT.TGraph(len(fpr),tpr,fpr)
	
	roc.SetLineColor(2)
	roc.SetLineWidth(2)
	roc.SetTitle(";Signal efficiency (%s); Background efficiency (%s)"%(signal_label, background_label))
	roc.GetXaxis().SetTitleOffset(1.4)
	roc.GetXaxis().SetTitleSize(0.045)
	roc.GetYaxis().SetTitleOffset(1.4)
	roc.GetYaxis().SetTitleSize(0.045)
	roc.GetXaxis().SetRangeUser(0,1)
	roc.GetYaxis().SetRangeUser(0.000,1)
	roc.Draw("AL")
	
	ROOT.gStyle.SetTextFont(42)
	t = ROOT.TPaveText(0.2,0.84,0.4,0.94,"NBNDC")
	t.SetTextAlign(11)
	t.SetFillStyle(0)
	t.SetBorderSize(0)
	t.AddText('AUC = %.3f'%AUC)
	t.Draw('same')
	
	c.SaveAs(outfile)

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
   
    colors = [2,4,8,ROOT.kCyan+2]
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
    

def drawTrainingCurve(input,output):
    hist = pickle.load(open(input,"rb"))
    tr_acc = hist["acc"]
    tr_loss = hist["loss"]
    val_acc = hist["val_acc"]
    val_loss = hist["val_loss"]
    epochs = range(len(tr_acc))

    plt.figure(1)
    plt.subplot(211)
    plt.plot(epochs, tr_acc,label="training")
    plt.plot(epochs, val_acc, label="validation")
    plt.legend(loc='best')
    plt.grid(True)
    #plt.xlabel("number of epochs")
    plt.ylabel("accuracy")
    plt.subplot(212)
    plt.plot(epochs, tr_loss, label="training")
    plt.plot(epochs, val_loss, label="validation")
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel("number of epochs")
    plt.ylabel("loss")
    plt.savefig(output)

  


gROOT.SetBatch(1)

OutputDir = 'Model'
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
SM = (Y == 0) 
left = ((Y == 1))
right = ((Y == 2))
X_jets = X_jets[cut:]
X_mu = X_mu[cut:]
X_el = X_el[cut:]
X_flat = X_flat[cut:]
print len(Y)
print len(Y[left])
print len(Y[SM])
print len(Y[right])
labels = Y

Y = to_categorical(labels, num_classes=3)
scaler = StandardScaler()

#X_jets = X_jets.reshape(X_jets.shape[0],X_jets.shape[1]*X_jets.shape[2])
#X_jets = scaler.fit_transform(X_jets)
#X_mu = X_mu.reshape(X_mu.shape[0],X_mu.shape[1]*X_mu.shape[2])
#X_mu = scaler.fit_transform(X_mu)
#X_el = X_el.reshape(X_el.shape[0],X_el.shape[1]*X_el.shape[2])
#X_el = scaler.fit_transform(X_el)
#X_flat = scaler.fit_transform(X_flat)


X_jets_train, X_jets_test,X_mu_train, X_mu_test,X_el_train, X_el_test,X_flat_train, X_flat_test, Y_train, Y_test, y_train, y_test = train_test_split(X_jets,X_mu,X_el,X_flat, Y, labels, test_size=0.2)

    
X_test = [X_jets_test[0:1], X_mu_test[0:1], X_el_test[0:1], X_flat_test[0:1]]

adam = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

nclasses = 3
dropoutRate = 0.1

Inputs = [Input(shape=(8,5)),Input(shape=(3,5)),Input(shape=(3,5)),Input(shape=(13,))]
jets = BatchNormalization(momentum=0.6,name='jets_input_batchnorm') (Inputs[0])
muons = BatchNormalization(momentum=0.6,name='muons_input_batchnorm')     (Inputs[1])
elec = BatchNormalization(momentum=0.6,name='elec_input_batchnorm')     (Inputs[2])
globalvars = BatchNormalization(momentum=0.6,name='globalvars_input_batchnorm')     (Inputs[3]) 

#jets = SortLayer()(jets)
muons = SortLayer()(muons)
#elec = SortLayer()(elec)

jets = Flatten()(jets)
muons = Flatten()(muons)
elec = Flatten()(elec)

x = Concatenate()( [globalvars,jets,muons,elec])
pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)

model = Model(inputs=Inputs,outputs=pred)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.predict(X_test)
