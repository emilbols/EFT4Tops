
void read(){
  //SM.root
  TFile *ntfile = new TFile("datafiles/SM.root","READ");
  //TFile *ntfile = new TFile("cQQ1_-1__merge_tag_1_delphes_events.root","READ");
  TFile *ntfile1 = new TFile("datafiles/cQQ1_merged.root","READ");
  TTree *data1 = (TTree*)ntfile->Get("tree");
  TTree *data2 = (TTree*)ntfile1->Get("tree");

  double Njets;
  double Nbtags;
  double Nleps;
  double mT_l2;
  double m_l1l2;
  double MET;
  double mT_l1;
  double H_Tb;
  double m_j1j2;
  double m_l1j2;
  double deltaPhi_j1j2;
  double deltaPhi_l1j1;
  double pT_j1;
  double q1;
  double deltaEta_l1l2;
  double H_T;
  double H_Tratio;
  double pT_j2;
  double m_l1j1;
  double coupling;
  double classy;
  double Nlooseb;
  double Ntightb;
  double Wcands;
  data1->SetBranchAddress("Njets",&Njets);
  data1->SetBranchAddress("Nbtags",&Nbtags);
  data1->SetBranchAddress("Nleps",&Nleps);
  data1->SetBranchAddress("mT_l2",&mT_l2);
  data1->SetBranchAddress("m_l1l2",&m_l1l2);
  data1->SetBranchAddress("MET",&MET);
  data1->SetBranchAddress("mT_l1",&mT_l1);
  data1->SetBranchAddress("H_Tb",&H_Tb);
  data1->SetBranchAddress("m_j1j2",&m_j1j2);
  data1->SetBranchAddress("m_l1j2",&m_l1j2);
  data1->SetBranchAddress("deltaPhi_j1j2",&deltaPhi_j1j2);
  data1->SetBranchAddress("deltaPhi_l1j1",&deltaPhi_l1j1);
  data1->SetBranchAddress("pT_j1",&pT_j1);
  data1->SetBranchAddress("q1",&q1);
  data1->SetBranchAddress("deltaEta_l1l2",&deltaEta_l1l2);
  data1->SetBranchAddress("H_T",&H_T);
  data1->SetBranchAddress("H_Tratio",&H_Tratio);
  data1->SetBranchAddress("pT_j2",&pT_j2);
  data1->SetBranchAddress("m_l1j1",&m_l1j1);
  data1->SetBranchAddress("coupling",&coupling);
  data1->SetBranchAddress("class",&classy);
  data1->SetBranchAddress("Nlooseb",&Nlooseb);
  data1->SetBranchAddress("Ntightb",&Ntightb);
  data1->SetBranchAddress("Wcands",&Wcands);
  
  TH1F *h_SM_HT= new TH1F("SM_HT","SM_HT", 100,0,3000);
  TH1F *h_cQQ1_HT= new TH1F("cQQ1_HT","cQQ1_HT", 100,0,3000);
  TH1F *h_SM_H_Tb= new TH1F("SM_H_Tb","SM_H_Tb", 100,0,3000);
  TH1F *h_cQQ1_H_Tb= new TH1F("cQQ1_H_Tb","cQQ1_H_Tb", 100,0,3000);
  TH1F *h_SM_m_j1j2= new TH1F("SM_m_j1j2","SM_m_j1j2", 100,0,3000);
  TH1F *h_cQQ1_m_j1j2= new TH1F("cQQ1_m_j1j2","cQQ1_m_j1j2", 100,0,3000);
  TH1F *h_SM_m_l1l2= new TH1F("SM_m_l1l2","SM_m_l1l2", 100,0,3000);
  TH1F *h_cQQ1_m_l1l2= new TH1F("cQQ1_m_l1l2","cQQ1_m_l1l2", 100,0,3000);
  TH1F *h_SM_MET= new TH1F("SM_MET","SM_MET", 100,0,3000);
  TH1F *h_cQQ1_MET= new TH1F("cQQ1_MET","cQQ1_MET", 100,0,3000);

  Int_t nentries = data1->GetEntries();
  for (Int_t i=0;i<nentries;i++) {
    data1->GetEntry(i);
    h_SM_HT->Fill(H_T);
    h_SM_H_Tb->Fill(H_Tb);
    h_SM_m_j1j2->Fill(m_j1j2);
    h_SM_m_l1l2->Fill(m_l1l2);
    h_SM_MET->Fill(MET);
  }
  
  data2->SetBranchAddress("Njets",&Njets);
  data2->SetBranchAddress("Nbtags",&Nbtags);
  data2->SetBranchAddress("Nleps",&Nleps);
  data2->SetBranchAddress("mT_l2",&mT_l2);
  data2->SetBranchAddress("m_l1l2",&m_l1l2);
  data2->SetBranchAddress("MET",&MET);
  data2->SetBranchAddress("mT_l1",&mT_l1);
  data2->SetBranchAddress("H_Tb",&H_Tb);
  data2->SetBranchAddress("m_j1j2",&m_j1j2);
  data2->SetBranchAddress("m_l1j2",&m_l1j2);
  data2->SetBranchAddress("deltaPhi_j1j2",&deltaPhi_j1j2);
  data2->SetBranchAddress("deltaPhi_l1j1",&deltaPhi_l1j1);
  data2->SetBranchAddress("pT_j1",&pT_j1);
  data2->SetBranchAddress("q1",&q1);
  data2->SetBranchAddress("deltaEta_l1l2",&deltaEta_l1l2);
  data2->SetBranchAddress("H_T",&H_T);
  data2->SetBranchAddress("H_Tratio",&H_Tratio);
  data2->SetBranchAddress("pT_j2",&pT_j2);
  data2->SetBranchAddress("m_l1j1",&m_l1j1);
  data2->SetBranchAddress("coupling",&coupling);
  data2->SetBranchAddress("class",&classy);
  data2->SetBranchAddress("Nlooseb",&Nlooseb);
  data2->SetBranchAddress("Ntightb",&Ntightb);
  data2->SetBranchAddress("Wcands",&Wcands); 
  Int_t nentries2 = data2->GetEntries();
  for (Int_t i=0;i<nentries2;i++) {
    data2->GetEntry(i);
    h_cQQ1_HT->Fill(H_T);
    h_cQQ1_H_Tb->Fill(H_Tb);
    h_cQQ1_m_j1j2->Fill(m_j1j2);
    h_cQQ1_MET->Fill(MET);
    h_cQQ1_m_l1l2->Fill(m_l1l2);
  }
  auto legend = new TLegend(0.9,0.7,0.48,0.9);
  legend->AddEntry(h_SM_HT,"SM");
  legend->AddEntry(h_cQQ1_HT,"cQQ1");
  TCanvas *c1 = new TCanvas("c1", "HT SM vs cQQ1",50,50,1000,800);
  h_SM_HT->Scale(1/(h_SM_HT->Integral(1,100)));
  h_SM_HT->Draw("hist");
  h_SM_HT->SetFillColor(2);
  h_cQQ1_HT->Scale(1/(h_cQQ1_HT->Integral(1,100)));
  h_cQQ1_HT->Draw("E0 same");
  h_cQQ1_HT->SetMarkerStyle(20);
  legend->Draw();
  c1->SetLogy();
  c1->Update();

  TCanvas *c2 = new TCanvas("c2", "m_j1j2 SM vs cQQ1",50,50,1000,800);
  h_SM_m_j1j2->Scale(1/(h_SM_m_j1j2->Integral(1,100)));
  h_SM_m_j1j2->Draw("hist");
  h_SM_m_j1j2->SetFillColor(2);
  h_cQQ1_m_j1j2->Scale(1/(h_cQQ1_m_j1j2->Integral(1,100)));
  h_cQQ1_m_j1j2->Draw("E0 same");
  h_cQQ1_m_j1j2->SetMarkerStyle(20);
  legend->Draw();
  c2->SetLogy();
  c2->Update();


  TCanvas *c3 = new TCanvas("c3", "m_l1l2 SM vs cQQ1",50,50,1000,800);
  h_SM_m_l1l2->Scale(1/(h_SM_m_l1l2->Integral(1,100)));
  h_SM_m_l1l2->Draw("hist");
  h_SM_m_l1l2->SetFillColor(2);
  h_cQQ1_m_l1l2->Scale(1/(h_cQQ1_m_l1l2->Integral(1,100)));
  h_cQQ1_m_l1l2->Draw("E0 same");
  h_cQQ1_m_l1l2->SetMarkerStyle(20);
  legend->Draw();
  c3->SetLogy();
  c3->Update();


  TCanvas *c4 = new TCanvas("c4", "MET SM vs cQQ1",50,50,1000,800);
  h_SM_MET->Scale(1/(h_SM_MET->Integral(1,100)));
 
  h_SM_MET->Draw("hist");
  h_SM_MET->SetFillColor(2);
  h_cQQ1_MET->Scale(1/(h_cQQ1_MET->Integral(1,100)));
  h_cQQ1_MET->Draw("E0 same");
  h_cQQ1_MET->SetMarkerStyle(20);
  legend->Draw();
  c4->SetLogy();
  c4->Update();

  TCanvas *c5 = new TCanvas("c5", "H_Tb SM vs cQQ1",50,50,1000,800);
  h_SM_H_Tb->Scale(1/(h_SM_H_Tb->Integral(1,100)));
  h_SM_H_Tb->Draw("hist");
  h_SM_H_Tb->SetFillColor(2);
  h_cQQ1_H_Tb->Scale(1/(h_cQQ1_H_Tb->Integral(1,100)));
  h_cQQ1_H_Tb->Draw("E0 same");
  h_cQQ1_H_Tb->SetMarkerStyle(20);
  legend->Draw();
  c5->SetLogy();
  c5->Update();
  
}
