import numpy as np


X_jets = np.load('SM_LO_only/features_jet.npy')
X_mu = np.load('SM_LO_only/features_mu.npy')
X_el = np.load('SM_LO_only/features_el.npy')
X_flat = np.load('SM_LO_only/features_flat.npy')
Y = np.load('SM_LO_only/truth.npy')


new_X_jets = np.load('inference_samples_two_preprocessed/cQQ1_0_cQt1_0features_jet.npy')
new_X_mu = np.load('inference_samples_two_preprocessed/cQQ1_0_cQt1_0features_mu.npy')
new_X_el = np.load('inference_samples_two_preprocessed/cQQ1_0_cQt1_0features_el.npy')
new_X_flat = np.load('inference_samples_two_preprocessed/cQQ1_0_cQt1_0features_flat.npy')
Y = np.load('inference_samples_two_preprocessed/cQQ1_0_cQt1_0truth.npy')



