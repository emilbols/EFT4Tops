import numpy as np

X_EFT = np.load('multi_test_merge/features.npy')
Y_EFT = np.load('multi_test_merge/truth.npy')

X_SM = np.load('multi_test_merge/SM_features.npy')
Y_SM = np.load('multi_test_merge/SM_truth.npy')

X = np.concatenate((X_SM,X_EFT),axis=0)
Y = np.concatenate((Y_SM,Y_EFT),axis=0)


np.save('multi_test_merge/Merged_features.npy',X)
np.save('multi_test_merge/Merged_truth.npy', Y)
