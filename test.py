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
#from keras.utils.visualize_util import plot
from numpy.lib.recfunctions import stack_arrays
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.cross_validation import train_test_split
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

  
def make_model(input_dim, nb_classes, nb_hidden_layers = 1, nb_neurons = 50,momentum_sgd = 0.8, init_learning_rate_sgd = 0.0005, dropout =0.1,nb_epoch = 100, batch_size=128):
    #batch_size = 128
    #nb_epoch = args.n_epochs

    #prepare the optimizer 
    decay_sgd = init_learning_rate_sgd/float(5*nb_epoch) if nb_epoch !=0 else 0.0001
    sgd = SGD(lr=init_learning_rate_sgd, decay=decay_sgd, momentum=momentum_sgd, nesterov=True)


    model = Sequential()
    model.add(Dense(nb_neurons ,input_shape= input_dim))
    model.add(Activation('relu'))
    for x in range ( nb_hidden_layers ):
            model.add(Dense(nb_neurons))
            model.add(Activation('relu'))
            model.add(Dropout(dropout))
    # model.add(Dense(nb_neurons))
#     model.add(Activation('relu'))
    #model.add(Dropout(dropout))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
 
    return model

def main():
    
    gROOT.SetBatch(1)

    parser = ArgumentParser()
    parser.add_argument('--nepoch', type=int, default=100,help='number of epochs to run the training for')
    parser.add_argument('--TrainingFile', default = "", help='path to training')
    parser.add_argument('--InputDir', default = "final", help='path to the converted delphes files')
    #parser.add_argument('--tag', default=time.strftime("%a%d%b%Y_%Hh%Mm%Ss"),help='name of output directory')
    args = parser.parse_args()
    
    
#SM_merged.root  cQQ1_merged.root  cQQ8_merged.root  cQt1_merged.root  cQt8_merged.root  ctt1_merged.root
    classes_dict = { #name:class_number 
            'SM': 0,
            #LLLL
            "cQQ1": 1,
            "cQQ8": 1,
            #LLRR
            "cQt1": 2,
            "cQt8": 2,
            #RRRR
            "ctt1": 2
        }

    couplings = classes_dict.keys()
    print couplings
    
    nb_classes = len(set(i for j,i in classes_dict.iteritems()))
    
    files = [i for i in os.listdir(args.InputDir) if ".root" in i]
    
    chain_dict = {}
    for c in couplings:
        chain_dict.update({c:TChain("tree")})
    #SM_chain = TChain("")
    #C1tu_chain = TChain("")


    for f in files:
        if "SM" in f:
		chain_dict["SM"].Add(args.InputDir + "/" + f)
        else:
		coupling_name = f.split("_")[0]#[:-3]
		if coupling_name in couplings:
			chain_dict[coupling_name].Add(args.InputDir + "/" + f)

    branchnames = [i.GetName() for i in chain_dict["SM"].GetListOfBranches()]
    print branchnames, len(branchnames)
    listbranch = ['m_l1j1', 'deltaPhi_l1j1', 'H_T', 'm_l1j2', 'm_l1l2', 'Nleps', 'H_Tratio', 'deltaEta_l1l2', 'pT_j1', 'pT_j2', 'Nbtags', 'Wcands', 'deltaPhi_j1j2', 'Nlooseb', 'q1', 'Ntightb', 'H_Tb', 'Njets', 'mT_l2', 'mT_l1', 'MET', 'm_j1j2']
    data_dict = {}
    X_SM_ = rootnp.tree2array(chain_dict["SM"], branches = listbranch)
    X_SM_ = rootnp.rec2array(X_SM_)
    data_dict.update({"SM":[X_SM_,np.zeros(len(X_SM_))]})
    #counter = 1
    for name, chain in chain_dict.iteritems():
        if name == "SM": continue
        X_ = rootnp.tree2array(chain,branches = listbranch)
        X_ = rootnp.rec2array(X_)
        y_= np.asarray([classes_dict[name]]*len(X_))
        data_dict.update({name:[X_,y_]})
        #counter += 1
    
    # make sure that all classes have the same number of events
    nb_events_per_class = [0]*nb_classes
    for name, data in data_dict.iteritems():
        nb_events_per_class[classes_dict[name]] += len(data[0])
    smallest_nb_events_per_class = min(nb_events_per_class)
    ratio_per_class = [smallest_nb_events_per_class/float(i) for i in nb_events_per_class]
    
    #shortest_length = min([len(i[0]) for name, i in data_dict.iteritems() ])
    data_dict["SM"][0] = data_dict["SM"][0][0:int(ratio_per_class[classes_dict["SM"]]*len(data_dict["SM"][0]))]
    data_dict["SM"][1] = data_dict["SM"][1][0:int(ratio_per_class[classes_dict["SM"]]*len(data_dict["SM"][1]))]
    X = data_dict["SM"][0]
    y = data_dict["SM"][1]
    for name, data in data_dict.iteritems():
        if name == "SM": continue
        X = np.concatenate((X,data[0][0:int(ratio_per_class[classes_dict[name]]*len(data[0]))]))
        y = np.concatenate((y,data[1][0:int(ratio_per_class[classes_dict[name]]*len(data[1]))]))
    Y = np_utils.to_categorical(y.astype(int), nb_classes)
    
    scaler = StandardScaler()
    scaler.fit(X)
    if not os.path.isdir(args.InputDir + "/training_output"): os.mkdir(args.InputDir + "/training_output")
    pickle.dump(scaler,open(args.InputDir + "/training_output/scaler.pkl",'wb'))
        
    X_train, X_test , y_train, y_test, Y_train, Y_test = train_test_split(X, y, Y, test_size=0.2)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
	
    
if __name__ == "__main__":
    main()
