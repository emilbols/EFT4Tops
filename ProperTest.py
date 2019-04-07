from sklearn.model_selection import train_test_split

import numpy as np

Y = np.load('train_samples_preprocessed/truth.npy')
X_jets = np.load('train_samples_preprocessed/features_jet.npy')
X_mu = np.load('train_samples_preprocessed/features_mu.npy')
X_el = np.load('train_samples_preprocessed/features_el.npy')
X_flat = np.load('train_samples_preprocessed/features_flat.npy')



Y = np.load('train_samples_preprocessed/truth.npy')
X_jets = np.load('train_samples_preprocessed/features_jet.npy')
X_mu = np.load('train_samples_preprocessed/features_mu.npy')
X_el = np.load('train_samples_preprocessed/features_el.npy')
X_flat = np.load('train_samples_preprocessed/features_flat.npy')


X_jets_train, X_jets_test,X_mu_train, X_mu_test,X_el_train, X_el_test,X_flat_train, X_flat_test, Y_train, Y_test = train_test_split(X_jets,X_mu,X_el,X_flat, Y, test_size=0.2, random_state = 19930607)

SM = (Y_test == 0)
print X_jets_test.shape
print SM.shape
np.save('OtherCrossCheck/features_jet.npy',X_jets_test[SM[:,0]])
np.save('OtherCrossCheck/features_mu.npy',X_mu_test[SM[:,0]])
np.save('OtherCrossCheck/features_el.npy',X_el_test[SM[:,0]])
np.save('OtherCrossCheck/features_flat.npy',X_flat_test[SM[:,0]])
