import numpy as np


couplings = ['-3','-2','-1','0','+1','+2','+3']
test = np.zeros((7,7))
n = -1
l = -1
num = 0
for y in couplings:
        n = n + 1
        l = -1
        for z in couplings:
                num = -1
                l = l + 1
                searchfile = open("cQQ1_"+y+"_cQt1_"+z+".html", "r")
                for line in searchfile:
                        if "<a href=\"./HTML/run_01/results.html\"> " in line:
                                num = float(line.split("<a href=\"./HTML/run_01/results.html\"> ",1)[1].split("<font")[0])
                test[n,l] = num 
                searchfile.close()

                

np.save('cross_section.npy',test)
