from ROOT import *
import os
import numpy as np
from array import array


def convertToLatex(c):
    if len(c) == 4: return c[0]+"^{"+c[1]+"}_{"+c[2:4]+"}"
    elif len(c) == 5: return c[0]+"^{"+c[1]+","+c[2]+"}_{"+c[3:5]+"}"
    else: 
        print "couldn't convert"
        return c

# showing_order = {
#     0:'C11Qq',
#     1:'C13Qq',
#     2:'C81Qq',
#     3:'C83Qq',
#     4:'C1tu',
#     5:'C1td',
#     6:'C8tu',
#     7:'C8td',
#     8:'C1Qu',
#     9:'C1Qd',
#     10:'C1tq',
#     11:'C8Qu',
#     12:'C8Qd',
#     13:'C8tq'
# }
#showing_order = {
#    4:'cQt8',
#    3:'cQt1',
#    2:'cQQ8',
#    1:'cQQ1',
#    0:'ctt1'
#}
showing_order = {
    2:'cQt1',
    1:'cQQ1',
    0:'ctt1'
}


file_tttt = open("XsecPlots/outputxsec.txt","r")
lines_tttt = file_tttt.readlines()
couplings_tttt = [l.split(",")[0] for l in lines_tttt]
#couplings_tttt = ['ctt1','cQQ1','cQt1']
lower_limits_tttt = [float(l.split(",")[1]) for l in lines_tttt]
upper_limits_tttt = [float(l.split(",")[2]) for l in lines_tttt]
#make dictionary
tttt_dict={}
for idx,c in enumerate(couplings_tttt):
    tttt_dict[c]=[lower_limits_tttt[idx],upper_limits_tttt[idx]]

file_xsec = open("XsecPlots/outputxsec_cut.txt","r")
lines_xsec = file_xsec.readlines()
couplings_xsec = [l.split(",")[0] for l in lines_xsec]
lower_limits_xsec = [float(l.split(",")[1]) for l in lines_xsec]
upper_limits_xsec = [float(l.split(",")[2]) for l in lines_xsec]
xsec_dict={}
for idx,c in enumerate(couplings_xsec):
    xsec_dict[c]=[lower_limits_xsec[idx],upper_limits_xsec[idx]]

file_NN = open("XsecPlots/outputxsec_discrim_target.txt","r")
lines_NN = file_NN.readlines()
couplings_NN = [l.split(",")[0] for l in lines_NN]
lower_limits_NN = [float(l.split(",")[1]) for l in lines_NN]
upper_limits_NN = [float(l.split(",")[2]) for l in lines_NN]
NN_dict={}
for idx,c in enumerate(couplings_NN):
    NN_dict[c]=[lower_limits_NN[idx],upper_limits_NN[idx]]


file_infer = open("XsecPlots/outputxsec_discrim.txt","r")
lines_infer = file_infer.readlines()
couplings_infer = [l.split(",")[0] for l in lines_infer]
lower_limits_infer = [float(l.split(",")[1]) for l in lines_infer]
upper_limits_infer = [float(l.split(",")[2]) for l in lines_infer]
infer_dict={}
for idx,c in enumerate(couplings_infer):
    infer_dict[c]=[lower_limits_infer[idx],upper_limits_infer[idx]]

    
#n_couplings = len(couplings_tttt)
n_couplings = 3

offset_tttt = 2*(1./float(n_couplings))/6
offset_xsec = 3*(1./float(n_couplings))/6
offset_NN = 4*(1./float(n_couplings))/6
y_pos_tttt = array("d",[float(i)/float(n_couplings)+offset_tttt for i in range(n_couplings)])
y_pos_xsec = array("d",[float(i)/float(n_couplings)+offset_xsec for i in range(n_couplings)])
y_pos_NN = array("d",[float(i)/float(n_couplings)+offset_NN for i in range(n_couplings)])
y_errpos_tttt = array("d",[0]*n_couplings)
y_errpos_xsec = array("d",[0]*n_couplings)
y_errpos_NN = array("d",[0]*n_couplings)

offset_infer = 5*(1./float(n_couplings))/6
y_pos_infer = array("d",[float(i)/float(n_couplings)+offset_infer for i in range(n_couplings)])
y_errpos_infer = array("d",[0]*n_couplings)
#print couplings_tttt


#gSystem.SetBatch(1)
gStyle.SetOptStat(0)

x_tttt = array("d")
ex_tttt = array("d")
for i in range(n_couplings):
    x_tttt.append(np.mean(np.asarray(tttt_dict[showing_order[i]])))
    ex_tttt.append(tttt_dict[showing_order[i]][1]-np.mean(np.asarray(tttt_dict[showing_order[i]])))
# for coupling,limits in tttt_dict.iteritems():
#     x_tttt.append(np.mean(np.asarray(limits)))
#     ex_tttt.append(limits[1]-np.mean(np.asarray(limits)))
x_xsec = array("d")
ex_xsec = array("d")
for i in range(n_couplings):
    x_xsec.append(np.mean(np.asarray(xsec_dict[showing_order[i]])))
    ex_xsec.append(xsec_dict[showing_order[i]][1]-np.mean(np.asarray(xsec_dict[showing_order[i]])))
# for coupling,limits in xsec_dict.iteritems():
#     x_xsec.append(np.mean(np.asarray(limits)))
#     ex_xsec.append(limits[1]-np.mean(np.asarray(limits)))
x_NN = array("d")
ex_NN = array("d")
for i in range(n_couplings):
    x_NN.append(np.mean(np.asarray(NN_dict[showing_order[i]])))
    ex_NN.append(NN_dict[showing_order[i]][1]-np.mean(np.asarray(NN_dict[showing_order[i]])))
# for coupling,limits in NN_dict.iteritems():
#     x_NN.append(np.mean(np.asarray(limits)))
#     ex_NN.append(limits[1]-np.mean(np.asarray(limits)))
x_infer = array("d")
ex_infer = array("d")
for i in range(n_couplings):
    x_infer.append(np.mean(np.asarray(infer_dict[showing_order[i]])))
    ex_infer.append(infer_dict[showing_order[i]][1]-np.mean(np.asarray(infer_dict[showing_order[i]])))

    
c = TCanvas("c","c",300,600)
low_margin = 0.1
up_margin = 0.05
left_margin=0.2
right_margin=0.1
gPad.SetMargin(left_margin,right_margin,low_margin,up_margin)
gr_tttt = TGraphErrors(n_couplings,x_tttt,y_pos_tttt,ex_tttt,y_errpos_tttt)
gr_tttt.SetTitle("")
gr_tttt.SetLineColor(2)
gr_tttt.SetLineWidth(2)
gr_tttt.SetMarkerSize(0)
gr_tttt.SetMarkerColor(2)
gr_tttt.Draw("APE1")
gr_tttt.GetXaxis().SetLabelSize(0.05)
gr_tttt.GetXaxis().SetRangeUser(-15,15)
gr_tttt.GetYaxis().SetRangeUser(-0.2,1.2)

gr_xsec = TGraphErrors(n_couplings,x_xsec,y_pos_xsec,ex_xsec,y_errpos_xsec)
gr_xsec.SetTitle("")
gr_xsec.SetLineColor(4)
gr_xsec.SetLineWidth(2)
gr_xsec.SetMarkerSize(0)
gr_xsec.SetMarkerColor(4)
gr_xsec.Draw("APE1same")
gr_xsec.GetXaxis().SetLabelSize(0.05)
gr_xsec.GetXaxis().SetRangeUser(-15,15)
gr_xsec.GetYaxis().SetRangeUser(-0.2,1.2)

gr_NN = TGraphErrors(n_couplings,x_NN,y_pos_NN,ex_NN,y_errpos_NN)
gr_NN.SetTitle("")
gr_NN.SetLineColor(8)
gr_NN.SetLineWidth(2)
gr_NN.SetMarkerSize(0)
gr_NN.SetMarkerColor(8)
gr_NN.Draw("APE1same")
gr_NN.GetXaxis().SetLabelSize(0.05)
gr_NN.GetXaxis().SetRangeUser(-15,15)
gr_NN.GetYaxis().SetRangeUser(-0.2,1.2)

gr_infer = TGraphErrors(n_couplings,x_infer,y_pos_infer,ex_infer,y_errpos_infer)
gr_infer.SetTitle("")
gr_infer.SetLineColor(kViolet)
gr_infer.SetLineWidth(2)
gr_infer.SetMarkerSize(0)
gr_infer.SetMarkerColor(kViolet)
gr_infer.Draw("APE1same")
gr_infer.GetXaxis().SetLabelSize(0.05)
gr_infer.GetXaxis().SetRangeUser(-15,15)
gr_infer.GetYaxis().SetRangeUser(-0.2,1.2)



mg = TMultiGraph("mg","mg")
mg.SetTitle("")
mg.Add(gr_tttt)
mg.Add(gr_xsec)
mg.Add(gr_NN)
mg.Add(gr_infer)
mg.Draw("AP")
mg.GetYaxis().SetTickSize(0)
mg.GetYaxis().SetLabelSize(0)
mg.GetXaxis().SetLabelSize(0.05)

ylow = -0.1
yup = 1.2
mg.SetMinimum(ylow)
mg.SetMaximum(yup)

offset = low_margin+abs(ylow)*(1./1.3) #+ offset_xsec#abs(ylow)+0.08
y_pos_text = [(float(i)/n_couplings)*(1./1.52)+offset for i in range(len(couplings_tttt))]

latex = TLatex()
#for idx,coup in enumerate(couplings_tttt):
for i in range(n_couplings):
    latex.SetNDC(True)
    latex.SetTextAlign(11)
    latex.SetTextFont(42)
    latex.DrawLatex(.08,y_pos_text[i],convertToLatex(showing_order[i]))
    
#lines

line_pos = [-20,-15,-10,-5,0,5,10,15,20]
line_dict = {}
for pos in line_pos:
    line_dict[pos] = TLine(pos,ylow,pos,yup)
    line_dict[pos].SetLineColor(kBlue)
    line_dict[pos].SetLineStyle(2)
    if pos == 0: line_dict[pos].SetLineWidth(3)
    line_dict[pos].Draw("same")

#Legend
l = TLegend(0.2,0.83,0.9,0.95)
l.SetBorderSize(1)
l.SetFillColor(0)
l.SetFillStyle(1001)
l.SetTextSize(0.037)
l.AddEntry(gr_NN,"Shallow NN (2#sigma)","l")
l.AddEntry(gr_infer,"RNN (2#sigma)","l")
l.AddEntry(gr_xsec,"H_{t} cut (2#sigma)","l")
l.AddEntry(gr_tttt,"pure cross section (2#sigma)","l")
l.Draw("same")


c.SaveAs("./limits_summary_lal.pdf")
c.SaveAs("./limits_summary_lal.png")



