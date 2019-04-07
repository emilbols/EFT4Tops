import numpy as np

X_jet_EFT = np.load('Sample_train_LO/WithSM_LO_features_jet.npy')
X_mu_EFT = np.load('Sample_train_LO/WithSM_LO_features_mu.npy')
X_el_EFT = np.load('Sample_train_LO/WithSM_LO_features_el.npy')
X_flat_EFT = np.load('Sample_train_LO/WithSM_LO_features_flat.npy')
Y_EFT = np.load('Sample_train_LO/WithSM_LO_truth.npy')

X_jet_SM = np.load('Only_SM_LO/features_jet.npy')
X_mu_SM = np.load('Only_SM_LO/features_mu.npy')
X_el_SM = np.load('Only_SM_LO/features_el.npy')
X_flat_SM = np.load('Only_SM_LO/features_flat.npy')
Y_SM = np.load('Only_SM_LO/truth.npy')

X_jet = np.concatenate((X_jet_SM,X_jet_EFT),axis=0)
X_mu = np.concatenate((X_mu_SM,X_mu_EFT),axis=0)
X_el = np.concatenate((X_el_SM,X_el_EFT),axis=0)
X_flat = np.concatenate((X_flat_SM,X_flat_EFT),axis=0)
Y = np.concatenate((Y_SM,Y_EFT),axis=0)


np.save('Sample_train_LO/WithSM_LO_features_jet.npy',X_jet)
np.save('Sample_train_LO/WithSM_LO_features_mu.npy',X_mu)
np.save('Sample_train_LO/WithSM_LO_features_el.npy',X_el)
np.save('Sample_train_LO/WithSM_LO_features_flat.npy',X_flat)
np.save('Sample_train_LO/WithSM_LO_truth.npy', Y)
