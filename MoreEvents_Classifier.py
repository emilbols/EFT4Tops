from ROOT import TFile, TTree, TChain, TCanvas, TH1D, TLegend, gROOT, gStyle
import sys
import ROOT
import os
import time
from argparse import ArgumentParser
from array import array
from math import *
import numpy as np
import tensorflow as tf
from collections import Counter
import root_numpy as rootnp
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Input, Convolution1D, Concatenate, Flatten
from keras.utils import np_utils
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

from rootpy.plotting import Hist2D

from sklearn.neural_network import MLPClassifier


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

OutputDir = 'ConvFun'
if not os.path.exists(OutputDir):
            os.makedirs(OutputDir)

Y = np.load('multi_test_merge/truth.npy')    
X = np.load('multi_test_merge/features.npy')
print X.shape
print Y.shape

SM = (Y == 0) 
a = ((Y == 1))
b = ((Y == 2))
c = ((Y == 3))
d = ((Y == 4))
e = ((Y == 5))
print len(Y[SM])
Y = Y - 1
labels = Y

Y = to_categorical(labels, num_classes=5)
scaler = StandardScaler()

#X_jets = X_jets.reshape(X_jets.shape[0],X_jets.shape[1]*X_jets.shape[2])
#X_jets = scaler.fit_transform(X_jets)
#X_mu = X_mu.reshape(X_mu.shape[0],X_mu.shape[1]*X_mu.shape[2])
#X_mu = scaler.fit_transform(X_mu)
#X_el = X_el.reshape(X_el.shape[0],X_el.shape[1]*X_el.shape[2])
#X_el = scaler.fit_transform(X_el)
#X_flat = scaler.fit_transform(X_flat)


X_train,X_test, Y_train, Y_test, y_train, y_test = train_test_split(X, Y, labels, test_size=0.2)
print X.shape
    

adam = Adam(lr=0.005, beta_1=0.9, beta_2=0.999)

nclasses = 5
dropoutRate = 0.5

Inputs = [Input(shape=(50,83))]

x = BatchNormalization(momentum=0.6,name='jets_input_batchnorm') (Inputs[0])

x  = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='x_conv0')(x)
x = Dropout(dropoutRate)(x)
x  = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='x_conv1')(x)
x = Dropout(dropoutRate)(x)
x  = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='x_conv2')(x)
x = Dropout(dropoutRate)(x)
x  = Convolution1D(5, 1, kernel_initializer='lecun_uniform',  activation='relu', name='x_conv3')(x)
x = Dropout(dropoutRate)(x)
x = Flatten()(x)
x = Dense(100,activation='relu',kernel_initializer='lecun_uniform',name='dense_0')(x)
x = Dropout(dropoutRate)(x)
pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)

model = Model(inputs=Inputs,outputs=pred)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
print model.summary()


train_history = model.fit(X_train, Y_train,
          batch_size=1, epochs=100,
          validation_data=(X_test, Y_test),
          callbacks = [ModelCheckpoint(OutputDir + "/model_checkpoint_save.hdf5")],
          shuffle=True,verbose=1)
#model.save_weights('model.h5')

#for layer in model.layers:
#        if 'input_batchnorm' not in layer.name:
#                layer.trainable = False
#model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
#model.load_weights('model.h5')

#train_history = model.fit(X_train, Y_train,
 #                         batch_size=128, nb_epoch=100,
 #                         validation_data=(X_test, Y_test),
 #                         callbacks = [ModelCheckpoint(OutputDir + "/model_checkpoint_save.hdf5")],
 #                         shuffle=True,verbose=1)

pickle.dump(train_history.history,open(OutputDir + "/loss_and_acc.pkl",'wb'))
drawTrainingCurve(OutputDir+"/loss_and_acc.pkl",OutputDir+"/training_curve.pdf")
discr_dict = model.predict(X_test)

SM_discr = [(discr_dict[jdx,1]+discr_dict[jdx,2]) for jdx in range(0,len(discr_dict[:,0])) if y_test[jdx] == 0]
EFT_discr = [(discr_dict[jdx,1]+discr_dict[jdx,2]) for jdx in range(0,len(discr_dict[:,0])) if y_test[jdx] ==1 or y_test[jdx] == 2]
fpr, tpr, thres = roc_curve(np.concatenate((np.zeros(len(SM_discr)),np.ones(len(EFT_discr)))),np.concatenate((SM_discr,EFT_discr)))
AUC = 1-roc_auc_score(np.concatenate((np.zeros(len(SM_discr)),np.ones(len(EFT_discr)))),np.concatenate((SM_discr,EFT_discr)))
makeROC(fpr, tpr, thres,AUC,OutputDir+"/roc_SMvsEFT.pdf","EFT","SM")
makeDiscr({"EFT":EFT_discr,"SM":SM_discr},OutputDir+"/discr_SMvsEFT.pdf","discriminator P(t_{L}) + P(t_{R})")

tL_discr = [discr_dict[jdx,1]/(discr_dict[jdx,1]+discr_dict[jdx,2]) for jdx in range(0,len(discr_dict[:,0])) if y_test[jdx] == 1]
tR_discr = [discr_dict[jdx,1]/(discr_dict[jdx,1]+discr_dict[jdx,2]) for jdx in range(0,len(discr_dict[:,0])) if y_test[jdx] == 2]
fpr, tpr, thres = roc_curve(np.concatenate((np.zeros(len(tR_discr)),np.ones(len(tL_discr)))),np.concatenate((tR_discr,tL_discr)))
AUC = 1-roc_auc_score(np.concatenate((np.zeros(len(tR_discr)),np.ones(len(tL_discr)))),np.concatenate((tR_discr,tL_discr)))
makeROC(fpr, tpr, thres,AUC,OutputDir+"/roc_tLvstR.pdf","t_{L}","t_{R}")
makeDiscr({"tL":tL_discr,"tR":tR_discr},OutputDir+"/discr_tLvstR.pdf","discriminator #frac{P(t_{L})}{P(t_{L}) + P(t_{R})}")

