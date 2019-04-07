import os
import numpy as np
import pandas as pd


file_tttt = open("XsecPlots/outputxsec.txt","r")
lines_tttt = file_tttt.readlines()
couplings_tttt = [l.split(",")[0] for l in lines_tttt]
lower_limits_tttt = [float(l.split(",")[1]) for l in lines_tttt]
upper_limits_tttt = [float(l.split(",")[2]) for l in lines_tttt]
tttt_dict = {a:[] for a in couplings_tttt}
valid_dict = {a:[] for a in couplings_tttt}
for idx,c in enumerate(couplings_tttt):
        tttt_dict[c]=[lower_limits_tttt[idx],upper_limits_tttt[idx]]
        
print tttt_dict
M_cut = 3.0
#req = {'cQQ1': 1, 'cQQ8': 1, 'cQt8': 1, 'ctt1': 1, 'cQt1': 1}
req = {'cQQ1': 0.16, 'cQQ8': 0.29, 'cQt8': 0.29, 'ctt1': 0.16, 'cQt1': 0.16}
for idx,c in enumerate(couplings_tttt):
        print np.abs(tttt_dict[c][0])*(M_cut**2)/((4*3.1415)**2)
        print np.abs(tttt_dict[c][1])*(M_cut**2)/((4*3.1415)**2)
        valid_dict[c] = ( np.abs(tttt_dict[c][0])*(M_cut**2)/((4*3.1415)**2) < req[c] ) & ( np.abs(tttt_dict[c][1])*(M_cut**2)/((4*3.1415)**2) < req[c] )
print valid_dict
