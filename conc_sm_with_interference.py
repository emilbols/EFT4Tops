import numpy as np

X_jet_EFT = np.load('train_samples_preprocessed/ctt1+20features_jet.npy')
X_mu_EFT = np.load('train_samples_preprocessed/ctt1+20features_mu.npy')
X_el_EFT = np.load('train_samples_preprocessed/ctt1+20features_el.npy')
X_flat_EFT = np.load('train_samples_preprocessed/ctt1+20features_flat.npy')
Y_EFT = np.load('train_samples_preprocessed/ctt1+20truth.npy')

X_jet_SM = np.load('SM_only/features_jet.npy')
X_mu_SM = np.load('SM_only/features_mu.npy')
X_el_SM = np.load('SM_only/features_el.npy')
X_flat_SM = np.load('SM_only/features_flat.npy')
Y_SM = np.load('SM_only/truth.npy')

X_jet = np.concatenate((X_jet_SM,X_jet_EFT),axis=0)
X_mu = np.concatenate((X_mu_SM,X_mu_EFT),axis=0)
X_el = np.concatenate((X_el_SM,X_el_EFT),axis=0)
X_flat = np.concatenate((X_flat_SM,X_flat_EFT),axis=0)
Y = np.concatenate((Y_SM,Y_EFT),axis=0)


np.save('train_samples_preprocessed/features_jet.npy',X_jet)
np.save('train_samples_preprocessed/features_mu.npy',X_mu)
np.save('train_samples_preprocessed/features_el.npy',X_el)
np.save('train_samples_preprocessed/features_flat.npy',X_flat)
np.save('train_samples_preprocessed/truth.npy', Y)
