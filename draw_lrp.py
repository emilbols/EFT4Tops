import warnings
warnings.simplefilter('ignore')
import numpy as np
import imp
import time
import matplotlib.pyplot as plt
import os
import tempfile
import h5py
import keras
import keras.backend
from keras.models import load_model
from keras import activations



def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp((x.transpose()-x.max(axis=1)).transpose())
        return e_x / np.sum(e_x,axis=1)[:,None]

def rms(x):
        n = x.shape[0]
        sqsum = ( np.square(x).sum(axis=0) )
        return np.sqrt(sqsum/n)


def sorting(weight,cpf_labels):
        cpf_w = np.sum(weight,axis=1)
        cpf_mean = np.mean(cpf_w,axis=0)
        cpf_w_rms = rms(cpf_w)
        cpf_rms_args = np.argsort(cpf_w_rms)
        cpf_w_rms_sort = cpf_w_rms[cpf_rms_args[:]]
        cpf_arguments = np.argsort(cpf_mean)
        cpf_std = np.std(cpf_w,axis=0)
        cpf_std_args = np.argsort(cpf_std)
        cpf_mean_sorted = cpf_mean[cpf_arguments[:]]
        cpf_std_sorted = cpf_std[cpf_arguments[:]]
        cpf_mean_std_sorted = cpf_mean[cpf_std_args[:]]
        cpf_std_std_sorted = cpf_std[cpf_std_args[:]]
        #cpf_covariance = np.cov(cpf_w.transpose())
        cpf_labels_sorted = []
        cpf_std_labels_sorted = []
        cpf_rms_labels_sorted = []
        for n in range(0, len(cpf_arguments)):
                cpf_labels_sorted.append(cpf_labels[cpf_arguments[n]])
                cpf_std_labels_sorted.append(cpf_labels[cpf_std_args[n]])
                cpf_rms_labels_sorted.append(cpf_labels[cpf_rms_args[n]])
                
        return cpf_mean_sorted, cpf_std_sorted, cpf_mean_std_sorted, cpf_std_std_sorted, cpf_labels_sorted, cpf_std_labels_sorted, cpf_rms_labels_sorted, cpf_w_rms_sort

def sorting2(weight,cpf_labels):
        cpf_w = weight
        cpf_mean = np.mean(cpf_w,axis=0)
        cpf_w_rms = rms(cpf_w)
        cpf_rms_args = np.argsort(cpf_w_rms)
        cpf_w_rms_sort = cpf_w_rms[cpf_rms_args[:]]
        cpf_arguments = np.argsort(cpf_mean)
        cpf_std = np.std(cpf_w,axis=0)
        cpf_std_args = np.argsort(cpf_std)
        cpf_mean_sorted = cpf_mean[cpf_arguments[:]]
        cpf_std_sorted = cpf_std[cpf_arguments[:]]
        cpf_mean_std_sorted = cpf_mean[cpf_std_args[:]]
        cpf_std_std_sorted = cpf_std[cpf_std_args[:]]
        #cpf_covariance = np.cov(cpf_w.transpose())
        cpf_labels_sorted = []
        cpf_std_labels_sorted = []
        cpf_rms_labels_sorted = []
        for n in range(0, len(cpf_arguments)):
                cpf_labels_sorted.append(cpf_labels[cpf_arguments[n]])
                cpf_std_labels_sorted.append(cpf_labels[cpf_std_args[n]])
                cpf_rms_labels_sorted.append(cpf_labels[cpf_rms_args[n]])
                
        return cpf_mean_sorted, cpf_std_sorted, cpf_mean_std_sorted, cpf_std_std_sorted, cpf_labels_sorted, cpf_std_labels_sorted, cpf_rms_labels_sorted, cpf_w_rms_sort


def largest_indices(ary, n):
        """Returns the n largest indices from a numpy array."""
        flat = ary.flatten()
        indices = np.argpartition(flat, -n)[-n:]
        indices = indices[np.argsort(-flat[indices])]
        return np.unravel_index(indices, ary.shape)

def apply_modifications(model, custom_objects=None):
        """Applies modifications to the model layers to create a new Graph. For example, simply changing
    `model.layers[idx].activation = new activation` does not change the graph. The entire graph needs to be updated
    with modified inbound and outbound tensors because of change in layer building function.
    Args:
        model: The `keras.models.Model` instance.
    Returns:
        The modified model with changes applied. Does not mutate the original `model`.
    """
        # The strategy is to save the modified model and load it back. This is done because setting the activation
        # in a Keras layer doesnt actually change the graph. We have to iterate the entire graph and change the
        # layer inbound and outbound nodes with modified tensors. This is doubly complicated in Keras 2.x since
        # multiple inbound and outbound nodes are allowed with the Graph API.
        model_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + '.h5')
        try:
            model.save(model_path)
            return load_model(model_path, custom_objects=custom_objects)
        finally:
            os.remove(model_path)
                                            

import gc
gc.enable()



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

pred = np.load('lrp_weights/prediction.npy') 
pred_soft = softmax(pred)
lrp = True
lrpw2 = False
rnn = False
smoothgrad = False
norm = True



jet_labels = ['jet_pt','jet_eta','jet_mass','jet_phi','jet_btag']
mu_labels = ['mu_pt','mu_eta','mu_mt','mu_phi','mu_q']
el_labels = ['el_pt','el_eta','el_mt','el_phi','el_q']
flat_labels = ['m_l1j1', 'H_T', 'm_l1j2', 'm_l1l2', 'Nleps', 'H_Tratio', 'Nbtags', 'Nlooseb', 'Ntightb', 'H_Tb', 'Njets', 'MET', 'm_j1j2']


full_jet_labels = []
full_mu_labels = []
full_el_labels = []

for i in range(1,9):
        for a in jet_labels:
                full_jet_labels.append(a+str(i))

for i in range(1,4):
        for a in mu_labels:
                full_mu_labels.append(a+str(i))

for i in range(1,4):
        for a in el_labels:
                full_el_labels.append(a+str(i))


jet_w = np.load("lrp_weights/jets_lrp_weights.npy")
mu_w = np.load("lrp_weights/mu_lrp_weights.npy")
el_w = np.load("lrp_weights/el_lrp_weights.npy")
flat_w = np.load("lrp_weights/flat_lrp_weights.npy")



if norm:
        flat_row_sums = np.sum(np.abs(flat_w),axis=1)
        mu_row_sums = np.sum(np.sum(np.abs(mu_w),axis=1),axis=1)
        el_row_sums = np.sum(np.sum(np.abs(el_w),axis=1),axis=1)
        jet_row_sums = np.sum(np.sum(np.abs(jet_w),axis=1),axis=1)
        row_sums = flat_row_sums+el_row_sums+mu_row_sums+jet_row_sums
        mu_w = mu_w / row_sums[:,None,None]
        mu_w = mu_w[~np.isnan(mu_w).any(axis=1).any(axis=1)] 
        el_w = el_w / row_sums[:,None,None]
        el_w = el_w[~np.isnan(el_w).any(axis=1).any(axis=1)]
        jet_w = jet_w / row_sums[:,None,None]
        jet_w = jet_w[~np.isnan(jet_w).any(axis=1).any(axis=1)]
        flat_w = flat_w / row_sums[:,None]
        flat_w = flat_w[~np.isnan(flat_w).any(axis=1)] 



full_X_jets = X_jets.reshape(X_jets.shape[0],X_jets.shape[1]*X_jets.shape[2])
full_X_mu = X_mu.reshape(X_mu.shape[0],X_mu.shape[1]*X_mu.shape[2])
full_X_el = X_el.reshape(X_el.shape[0],X_el.shape[1]*X_el.shape[2])

full_jet_w = jet_w.reshape(jet_w.shape[0],jet_w.shape[1]*jet_w.shape[2])
full_mu_w = mu_w.reshape(mu_w.shape[0],mu_w.shape[1]*mu_w.shape[2])
full_el_w = el_w.reshape(el_w.shape[0],el_w.shape[1]*el_w.shape[2])


full_jet_even = np.linspace(1,40,40)
full_jet_mean_sorted, full_jet_std_sorted, full_jet_mean_std_sorted, full_jet_std_std_sorted, full_jet_labels_sorted, full_jet_std_labels_sorted, full_jet_rms_labels_sorted, full_jet_w_rms_sort = sorting2(full_jet_w,full_jet_labels)

plt.errorbar(full_jet_even,full_jet_mean_sorted,full_jet_std_sorted)
plt.xticks(full_jet_even, full_jet_labels_sorted, rotation=90)
plt.show()
plt.errorbar(full_jet_even,full_jet_mean_std_sorted,full_jet_std_std_sorted)
plt.xticks(full_jet_even, full_jet_std_labels_sorted, rotation=90)
plt.show()
plt.plot(full_jet_even,full_jet_w_rms_sort)
plt.xticks(full_jet_even, full_jet_rms_labels_sorted, rotation=90)
plt.show()


full_mu_even = np.linspace(1,15,15)

full_mu_mean_sorted, full_mu_std_sorted, full_mu_mean_std_sorted, full_mu_std_std_sorted, full_mu_labels_sorted, full_mu_std_labels_sorted, full_mu_rms_labels_sorted, full_mu_w_rms_sort = sorting2(full_mu_w,full_mu_labels)

print full_mu_mean_sorted.shape

plt.errorbar(full_mu_even,full_mu_mean_sorted,full_mu_std_sorted)
plt.xticks(full_mu_even, full_mu_labels_sorted, rotation=90)
plt.show()
plt.errorbar(full_mu_even,full_mu_mean_std_sorted,full_mu_std_std_sorted)
plt.xticks(full_mu_even, full_mu_std_labels_sorted, rotation=90)
plt.show()
plt.plot(full_mu_even,full_mu_w_rms_sort)
plt.xticks(full_mu_even, full_mu_rms_labels_sorted, rotation=90)
plt.show()

full_el_even = np.linspace(1,15,15)

full_el_mean_sorted, full_el_std_sorted, full_el_mean_std_sorted, full_el_std_std_sorted, full_el_labels_sorted, full_el_std_labels_sorted, full_el_rms_labels_sorted, full_el_w_rms_sort = sorting2(full_el_w,full_el_labels)

plt.errorbar(full_el_even,full_el_mean_sorted,full_el_std_sorted)
plt.xticks(full_el_even, full_el_labels_sorted, rotation=90)
plt.show()
plt.errorbar(full_el_even,full_el_mean_std_sorted,full_el_std_std_sorted)
plt.xticks(full_el_even, full_el_std_labels_sorted, rotation=90)
plt.show()
plt.plot(full_el_even,full_el_w_rms_sort)
plt.xticks(full_el_even, full_el_rms_labels_sorted, rotation=90)
plt.show()

inputs = np.concatenate((full_X_jets,full_X_mu,full_X_el),axis=1)
weights = np.concatenate((full_jet_w,full_mu_w,full_el_w),axis=1)
labels = np.concatenate((full_jet_labels,full_mu_labels,full_el_labels))

koop = np.linspace(0,69,70)

correlation = np.abs(np.corrcoef(weights.transpose()))
correlation_in = np.abs(np.corrcoef(inputs.transpose()))
rms_val = rms(weights)
select_rms = rms_val < 0.008
matrix_rms = rms_val*np.transpose(rms_val)

selection = (correlation < 0.1) | (correlation_in > 0.6)

correlation_in = np.abs(np.tril(correlation_in,-1))
correlation = np.abs(np.tril(correlation,-1))
correlation[selection] = 0
correlation[select_rms,:] = 0
correlation[:,select_rms] = 0
rms_corr = np.multiply(matrix_rms,correlation)
idx = largest_indices(rms_corr,10)
print labels[idx[0]]
print labels[idx[1]]
#print labels[idx_in[0]]
#print labels[idx_in[1]]

plt.figure(figsize=(14,14))
plt.imshow(rms_corr, cmap='jet', interpolation='nearest')
plt.xticks(koop, labels, rotation=90)
plt.yticks(koop, labels)
plt.savefig("corr.png")
plt.show()


mu_even = np.linspace(1,5,5)

mu_mean_sorted, mu_std_sorted, mu_mean_std_sorted, mu_std_std_sorted, mu_labels_sorted, mu_std_labels_sorted, mu_rms_labels_sorted, mu_w_rms_sort = sorting(mu_w,mu_labels)

print mu_mean_sorted.shape

plt.errorbar(mu_even,mu_mean_sorted,mu_std_sorted)
plt.xticks(mu_even, mu_labels_sorted, rotation=90)
plt.show()
plt.errorbar(mu_even,mu_mean_std_sorted,mu_std_std_sorted)
plt.xticks(mu_even, mu_std_labels_sorted, rotation=90)
plt.show()
plt.plot(mu_even,mu_w_rms_sort)
plt.xticks(mu_even, mu_rms_labels_sorted, rotation=90)
plt.show()


el_even = np.linspace(1,5,5)

el_mean_sorted, el_std_sorted, el_mean_std_sorted, el_std_std_sorted, el_labels_sorted, el_std_labels_sorted, el_rms_labels_sorted, el_w_rms_sort = sorting(el_w,el_labels)


plt.errorbar(el_even,el_mean_sorted,el_std_sorted)
plt.xticks(el_even, el_labels_sorted, rotation=90)
plt.show()
plt.errorbar(el_even,el_mean_std_sorted,el_std_std_sorted)
plt.xticks(el_even, el_std_labels_sorted, rotation=90)
plt.show()
plt.plot(el_even,el_w_rms_sort)
plt.xticks(el_even, el_rms_labels_sorted, rotation=90)
plt.show()



flat_even = np.linspace(1,13,13)
flat_mean_sorted, flat_std_sorted, flat_mean_std_sorted, flat_std_std_sorted, flat_labels_sorted, flat_std_labels_sorted, flat_rms_labels_sorted, flat_w_rms_sort = sorting2(flat_w,flat_labels)

plt.errorbar(flat_even,flat_mean_sorted,flat_std_sorted)
plt.xticks(flat_even, flat_labels_sorted, rotation=90)
plt.show()
plt.errorbar(flat_even,flat_mean_std_sorted,flat_std_std_sorted)
plt.xticks(flat_even, flat_std_labels_sorted, rotation=90)
plt.show()
plt.plot(flat_even,flat_w_rms_sort)
plt.xticks(flat_even, flat_rms_labels_sorted, rotation=90)
plt.show()


jet_even = np.linspace(1,5,5)
jet_mean_sorted, jet_std_sorted, jet_mean_std_sorted, jet_std_std_sorted, jet_labels_sorted, jet_std_labels_sorted, jet_rms_labels_sorted, jet_w_rms_sort = sorting(jet_w,jet_labels)

plt.errorbar(jet_even,jet_mean_sorted,jet_std_sorted)
plt.xticks(jet_even, jet_labels_sorted, rotation=90)
plt.show()
plt.errorbar(jet_even,jet_mean_std_sorted,jet_std_std_sorted)
plt.xticks(jet_even, jet_std_labels_sorted, rotation=90)
plt.show()
plt.plot(jet_even,jet_w_rms_sort)
plt.xticks(jet_even, jet_rms_labels_sorted, rotation=90)
plt.show()

combine_rms = np.concatenate((jet_w_rms_sort,flat_w_rms_sort,el_w_rms_sort,mu_w_rms_sort))
combine_label = np.concatenate((jet_rms_labels_sorted,flat_rms_labels_sorted,el_rms_labels_sorted,mu_rms_labels_sorted))
combine_even = np.linspace(1,28,28)
arguments_rms = np.argsort(combine_rms)
value_rms = np.sort(combine_rms)
sorted_labels = []
for n in range(0, len(arguments_rms)):
               sorted_labels.append(combine_label[arguments_rms[n]])

plt.plot(combine_even, value_rms)
plt.xticks(combine_even, sorted_labels, rotation=90)
plt.show()
fig, ax1 = plt.subplots(1,1)

c_track_even = np.linspace(1,3,3)
c_track_mean = np.mean(np.sum(mu_w,axis=2),axis=0)
#sort_list_mu = np.argsort(c_track_mean)[:]
#c_track_mean_sort = c_track_mean[sort_list_mu]
#c_track_std_sort = np.std(np.sum(mu_w,axis=2),axis=0)[sort_list_mu]
c_track_std = np.std(np.sum(mu_w,axis=2),axis=0)
plt.errorbar(c_track_even,c_track_mean,c_track_std)
plt.xlabel('Track number')
plt.title('Charged tracks LRP')
plt.show()

el_track_even = np.linspace(1,3,3)
el_track_mean = np.mean(np.sum(el_w,axis=2),axis=0)
el_track_std = np.std(np.sum(el_w,axis=2),axis=0)
plt.errorbar(el_track_even,el_track_mean,el_track_std)
plt.xlabel('Track number')
plt.title('Neutral tracks LRP')
plt.show()


jet_even = np.linspace(1,8,8)
jet_mean = np.mean(np.sum(jet_w,axis=2),axis=0)
jet_std = np.std(np.sum(jet_w,axis=2),axis=0)
plt.errorbar(jet_even,jet_mean,jet_std)
plt.show()

