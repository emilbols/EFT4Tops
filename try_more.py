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
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import SGD,Adam
from keras.regularizers import l1, l2
from keras.layers import Convolution1D
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

  
def make_model(input_dim, nb_classes, nb_hidden_layers = 3, nb_neurons = 100,momentum_sgd = 0.8, init_learning_rate_sgd = 0.0005, dropout =0.15,nb_epoch = 100, batch_size=128):
    #batch_size = 128
    #nb_epoch = args.n_epochs

    #prepare the optimizer 
    decay_sgd = init_learning_rate_sgd/float(5*nb_epoch) if nb_epoch !=0 else 0.0001
    sgd = SGD(lr=init_learning_rate_sgd, decay=decay_sgd, momentum=momentum_sgd, nesterov=True)
    adam = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model = Sequential()
    model.add(Dense(nb_neurons ,input_shape= input_dim))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    for x in range ( nb_hidden_layers ):
            model.add(Dense(nb_neurons))
            model.add(Activation('relu'))
            model.add(Dropout(dropout))
    # model.add(Dense(nb_neurons))
#     model.add(Activation('relu'))
    #model.add(Dropout(dropout))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
 
    return model

gROOT.SetBatch(1)

InputDir = 'inference_samples'
classes_dict = { #name:class_number 
        'SM': 0,
        #LLLL
        'ctt1' : 1,
        'cQQ1' : 2,
        'cQQ8' : 3,
        'cQt1' : 4,
        'cQt8' : 5
}

couplings = classes_dict.keys()
print couplings
    
nb_classes = len(set(i for j,i in classes_dict.iteritems()))

files = [i for i in os.listdir(InputDir) if ".root" in i]

chain_dict = {}
for c in couplings:
        chain_dict.update({c:TChain("tree")})
        #SM_chain = TChain("")
        #C1tu_chain = TChain("")

namelist = ['cQQ1','cQQ8','cQt1','cQt8','ctt1']
coupling = ['-20','-10','-5','-1','+1','+5','+10','+20']
for gdp in namelist:
        for bbp in coupling:
                name = gdp+bbp
                #chain_dict["SM"].Add(InputDir + "/SM_merge_tag_1_delphes_events.root")
                file_check = ROOT.TChain("tree")
                for f in files:
                        if name in f:
                                file_check.Add(InputDir + "/" + f)
                        
                                branchnames = [i.GetName() for i in file_check.GetListOfBranches()]
                                print branchnames, len(branchnames)
                        
                jetbranch = ['jet_pt','jet_eta','jet_mass','jet_phi','jet_btag']
                mu_branch = ['mu_pt','mu_eta','mu_mt','mu_phi','mu_q']
                el_branch = ['el_pt','el_eta','el_mt','el_phi','el_q']
                flat_branch = ['m_l1j1', 'H_T', 'm_l1j2', 'm_l1l2', 'Nleps', 'H_Tratio', 'Nbtags', 'Nlooseb', 'Ntightb', 'H_Tb', 'Njets', 'MET', 'm_j1j2']
                
                truthbranch = ['class']

                data_dict = {}

                Y = rootnp.tree2array(file_check,branches = truthbranch)
                Z_Y = rootnp.rec2array(Y)

                flat = rootnp.tree2array(file_check,branches = flat_branch)
                Z_flat = rootnp.rec2array(flat)
                #Z_Y = np.zeros(Y.shape[0])
                #for a in range(0,Y.shape):
                #        Z_Y[a] = Z_Y[a].tolist()

                
                X_mu = rootnp.tree2array(file_check, branches = mu_branch)
                X_mu = rootnp.rec2array(X_mu)
                
                X_el = rootnp.tree2array(file_check, branches = el_branch)
                X_el = rootnp.rec2array(X_el)
                
                X_jets = rootnp.tree2array(file_check, branches = jetbranch)
                X_jets = rootnp.rec2array(X_jets)
                
                max_jets = 8
                Z_jets = np.zeros((X_jets.shape[0],max_jets,len(jetbranch)))
                for a in range(0,X_jets.shape[0]):
                        for b in range(0,len(jetbranch)):
                                Z_jets[a,0:len(X_jets[a,b].tolist()),b] = X_jets[a,b][:max_jets].tolist()
                                
                max_el = 3
                Z_el = np.zeros((X_el.shape[0],max_el,len(el_branch)))
                for a in range(0,X_el.shape[0]):
                        for b in range(0,len(el_branch)):
                                Z_el[a,0:len(X_el[a,b].tolist()),b] = X_el[a,b][:max_el].tolist()

                max_mu = 3
                Z_mu = np.zeros((X_mu.shape[0],max_mu,len(mu_branch)))
                for a in range(0,X_mu.shape[0]):
                        for b in range(0,len(mu_branch)):
                                Z_mu[a,0:len(X_mu[a,b].tolist()),b] = X_mu[a,b][:max_mu].tolist()


                np.save('inference_samples_preprocessed/'+name+'features_jet.npy',Z_jets)
                np.save('inference_samples_preprocessed/'+name+'features_mu.npy',Z_mu)
                np.save('inference_samples_preprocessed/'+name+'features_el.npy',Z_el)
                np.save('inference_samples_preprocessed/'+name+'features_flat.npy',Z_flat)
                np.save('inference_samples_preprocessed/'+name+'truth.npy',Z_Y)



