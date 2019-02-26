import os
from shutil import copyfile

couplings = ['_-20','_-10','_-5','_-1','_+1','_+5','_+10','_+20']
prefix = '/user/imarches/MAD/MG5_aMC_v2_6_0/'
samples = ['cQQ1','cQQ8','cQt1','cQt8','ctt1']
suffix = '_interf_5/crossx.html'

dest = '/user/ebols/Cross_Section/'

for coupling in couplings:
    for sample in samples:
        name = prefix+sample+coupling+suffix
        dst = dest + sample + coupling + '.html'
        copyfile(name,dst)
