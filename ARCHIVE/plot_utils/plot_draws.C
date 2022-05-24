



////// ---------------------------------------------------
////// plotting functions

// give me clones in hists ... I will modify them
void resolution_summary(std::vector < TH1* > hists, std::vector < TString > names, TString cname, TString outFold, TString label)
{

    // lists to build graphs
    unsigned int nentries = hists.size();
    double* ys = new double[nentries]; // y values (simple iterator)
    double* mean = new double[nentries]; // mean resolution from h.GetMean()
    double* stdev = new double[nentries]; // error on resolution mean h.GetRMS()
    double* max = new double[nentries]; // maximal or mediam resolution
    double* med_at_98pc_up = new double[nentries]; // up error on median resolution
    double* med_at_98pc_lo = new double[nentries]; // lo error on median resolution
    double* dummy = new double[nentries]; // dummy to have no y-errors
    double maximum = -99.9; // to define plot x-range
    double minimum = 99.9; // to define plot x-range

    // ###########################################
    // // Fill lists
    // iterate over methods
    for (unsigned int ie = 0; ie < nentries; ++ie)
    {
        ys[ie] = 0.5+ie; // put line every 0.5 + iterator (frame defined from 0 to N(methods) with bins of size 1)
        dummy[ie] = 0.000001;
        unsigned int nbins = hists[ie]->GetNbinsX();
        hists[ie]->Scale(1./hists[ie]->Integral(0, nbins+1)); // just to make things easier
        double integral = 1.0; // Integral if hist not scaled

        // // get mean and RMS
        mean[ie] = hists[ie]->GetMean();
        stdev[ie] = hists[ie]->GetRMS();

        // // get max +/- 98% container
        // double maxval=0; // fraction of events at max
        double maxbin = 0; // bin number of max/med

        // get median or max
        for (unsigned int ib = 1; ib < nbins+1; ++ib)
        {
            // // if max :
            // if(maxval > hists[ib]->GetBinContent(ib)) continue;
            // maxval = hists[ib]->GetBinContent(ib);
            // maxbin = ib;

            // if median :
            if (hists[ie]->Integral(0, ib) > 0.5*integral)
            {
                maxbin = ib;
                break;
            }
        }
        max[ie] = (hists[ie]->GetBinLowEdge(maxbin)+hists[ie]->GetBinLowEdge(maxbin+1))/2.0;

        // get up/down error by dessending a line L(y=cst) until 98% events are above L, up(lo) is crossing L distrib above(below) max/med
        unsigned int npoints = 100000;
        for (unsigned int i = 0; i < npoints+1; ++i)
        {
            if (hists[ie]->Integral(1, nbins)/integral < 0.98)
            {
                med_at_98pc_up[ie] = 99;
                med_at_98pc_lo[ie] = 99;
                std::cout << "WARNING::resolution_summary::In " << cname << " " << names[ie] << " 98pc range not found .. use dummy values" << std::endl;
                std::cout << "WARNING::resolution_summary::   -> in case that helps : underflow=" << hists[ie]->GetBinContent(0) << "; overflow=" << hists[ie]->GetBinContent(nbins+1) << "; integral(1,nbins)=" << hists[ie]->Integral(1, nbins) << "; integral(2,nbins-1)=" << hists[ie]->Integral(2, nbins-1) << std::endl;
                break;
            }
            double iy = integral*((double)i/(double)npoints);
            double low_bound = 99;
            double hig_bound = 99;
            for (unsigned int ib = 1; ib < nbins+1; ++ib)
            {
                if (low_bound == 99 && hists[ie]->GetBinContent(ib) > iy)
                    low_bound = ib;
                if (low_bound == 99) continue;
                if (low_bound != 99 && ib < maxbin) continue; // to avoid capturing stat fluc after low_bound
                if (hig_bound == 99 && hists[ie]->GetBinContent(ib) < iy)
                    hig_bound = ib;
                if (hig_bound != 99) break;
            }
            double contain = hists[ie]->Integral(low_bound, hig_bound) / integral;
            if (cname == "summary_0p3_inf_mu140" && names[ie] == "lstm1_u10_seq5_shift5"
                && outFold == "20201008_compMus/reso_comp/sequence_ofmax_eT_isZero") std::cout << iy << " - " << low_bound << "  " << hig_bound << "  " << contain << std::endl;
            if (contain > 0.98) continue;

            med_at_98pc_up[ie] = hists[ie]->GetBinLowEdge(hig_bound-1) - max[ie];
            med_at_98pc_lo[ie] = max[ie] - hists[ie]->GetBinLowEdge(low_bound-1);

            if (maximum < med_at_98pc_up[ie]+max[ie]) maximum = med_at_98pc_up[ie]+max[ie];
            if (minimum > max[ie]-med_at_98pc_lo[ie]) minimum = max[ie]-med_at_98pc_lo[ie];

            break;
        }

        if (maximum < mean[ie]+stdev[ie]) maximum = mean[ie]+stdev[ie];
        if (minimum > mean[ie]-stdev[ie]) minimum = mean[ie]-stdev[ie];

        std::cout << "resolution_summary::Perf for " << names[ie] << " in " << cname << " : " << std::endl;
        std::cout << std::setprecision(22) << "resolution_summary::  -> mean pm stdev = " << mean[ie]
                  << " +/- " << stdev[ie] << std::endl;
        std::cout << std::setprecision(22) << "resolution_summary::  -> max + 98\%_up - 98\%_do = " << max[ie]
                  << " + " << med_at_98pc_up[ie] << " - " << med_at_98pc_lo[ie] << std::endl;

    }

    // ###########################################
    // // Now make the TGraph, dump and plot :)

    TCanvas* c = new TCanvas(cname, cname, 600, 600);
    c->cd();
    c->SetBottomMargin(0.15);
    c->SetTopMargin(0.08);
    c->SetLeftMargin(0.3);
    c->SetRightMargin(0.08);

    int nys = nentries+1;
    if (nentries > 3) nys += 1;

    // base for drawing on top of it
    TH1* base = new TH2F("base", "base", 100, minimum-0.2*abs(minimum), maximum+0.2*abs(maximum), nys, 0, nys);
    base->GetXaxis()->SetTitle("E_{T}^{pred}-E_{T}^{true} [GeV]");
    base->GetYaxis()->SetLabelSize(0.045);
    base->GetXaxis()->SetTitleSize(0.05);
    base->GetXaxis()->SetTitleOffset(1.2);
    for (unsigned int ie = 0; ie < nentries; ++ie)
    {
        base->GetYaxis()->SetBinLabel(ie+1, names[ie]);
    }
    base->Draw();

    // Create TGraphs for error bars
    gStyle->SetEndErrorSize(5);
    TGraphAsymmErrors* g_median = new TGraphAsymmErrors(nentries,
                                                        max, ys,
                                                        med_at_98pc_lo, med_at_98pc_up,
                                                        dummy, dummy);
    g_median->SetLineWidth(2);
    g_median->SetMarkerSize(0);
    g_median->SetLineColor(kBlack);
    g_median->Draw("same p");


    TGraphErrors* g_mean = new TGraphErrors(nentries,
                                            mean, ys,
                                            stdev,
                                            dummy);
    g_mean->SetLineWidth(2);
    g_mean->SetMarkerSize(0);
    g_mean->SetLineColor(kGreen+1);
    g_mean->Draw("same p");

    // Create TGraphs for points (allow to slightly shift points w.r.t line to have one above one below)
    // (Mean and Median central value can differ)
    double shift = 0.055;
    double* yssp = new double[nentries];
    double* yssm = new double[nentries];
    for (unsigned int ie = 0; ie < nentries; ++ie)
    {
        yssp[ie] = ys[ie]-shift;
        yssm[ie] = ys[ie]+shift;
    }
    TGraph* g_median_p = new TGraph(nentries, max, yssp);
    g_median_p->SetLineWidth(0);
    g_median_p->SetMarkerStyle(22);
    g_median_p->SetMarkerSize(1.5);
    g_median_p->SetLineColor(kBlack);
    g_median_p->Draw("same pz");


    TGraph* g_mean_p = new TGraph(nentries, mean, yssm);
    g_mean_p->SetLineWidth(0);
    g_mean_p->SetMarkerStyle(23);
    g_mean_p->SetMarkerSize(1.5);
    g_mean_p->SetMarkerColor(kGreen+1);
    g_mean_p->Draw("same pz");

    // Draw lines around first element errors to guide the eye on the comparison (consider first as reference)
    TLine* line1_up = new TLine(mean[0]+stdev[0], 0, mean[0]+stdev[0], nentries);
    line1_up->SetLineWidth(1);
    line1_up->SetLineStyle(8);
    line1_up->Draw("same");
    TLine* line1_lo = new TLine(mean[0]-stdev[0], 0, mean[0]-stdev[0], nentries);
    line1_lo->SetLineWidth(1);
    line1_lo->SetLineStyle(8);
    line1_lo->Draw("same");

    TLine* line2_up = new TLine(max[0]+med_at_98pc_up[0], 0, max[0]+med_at_98pc_up[0], nentries);
    line2_up->SetLineWidth(1);
    line2_up->SetLineStyle(7);
    line2_up->Draw("same");
    TLine* line2_lo = new TLine(max[0]-med_at_98pc_lo[0], 0, max[0]-med_at_98pc_lo[0], nentries);
    line2_lo->SetLineWidth(1);
    line2_lo->SetLineStyle(7);
    line2_lo->Draw("same");

    TLine* zero_line = new TLine(0, 0, 0, nentries);
    zero_line->SetLineWidth(1);
    zero_line->SetLineStyle(2);
    zero_line->Draw("same");

    // draw legend use dummy graphs to have both point and line
    TLegend* legend = new TLegend(0.5, 0.71, 0.9, 0.81);
    legend->SetFillColor(-1);
    legend->SetBorderSize(0);
    double* one = new double[1];
    one[1] = 1.0;
    TGraph* dummyMean = new TGraph(1, one, one);
    dummyMean->SetMarkerStyle(22);
    dummyMean->SetMarkerColor(kGreen+1);
    dummyMean->SetLineWidth(2);
    dummyMean->SetLineColor(kGreen+1);
    TGraph* dummyMed = new TGraph(1, one, one);
    dummyMed->SetMarkerStyle(23);
    dummyMed->SetMarkerColor(kBlack);
    dummyMed->SetLineWidth(2);
    dummyMed->SetLineColor(kBlack);
    legend->AddEntry(dummyMean, "Mean #pm StDev", "lp");
    legend->AddEntry(dummyMed, "Median #pm 98% range", "lp");
    legend->Draw("same");

    AREUSText(0.32, 0.88, "EM-Middle |#eta|#times#phi=0.5125x0.0125", 1, 0.035);

    myText(0.32, 0.88-0.05, label, 1, 0.035);

    c->SaveAs(outFold+"/"+cname+".png");
    if (PDF_PLOTS) {
        c->SaveAs(outFold+"/"+cname+".pdf");
    }
    
    if (c)
    {
        c->Close();
        gSystem->ProcessEvents();
        delete c;
        c = 0;
    }
    if (dummyMed)
    {
        dummyMed->Delete();
        dummyMed = 0;
    }
    if (dummyMean)
    {
        dummyMean->Delete();
        dummyMean = 0;
    }
    if (g_median)
    {
        g_median->Delete();
        g_median = 0;
    }
    if (g_mean)
    {
        g_mean->Delete();
        g_mean = 0;
    }
    if (g_median_p)
    {
        g_median_p->Delete();
        g_median_p = 0;
    }
    if (g_mean_p)
    {
        g_mean_p->Delete();
        g_mean_p = 0;
    }
    if (base)
    {
        base->Delete();
        base = 0;
    }

}

void compare_mus(TString name, TString outFold, int method_index = 0, int trigger_index = 0, int range_index = 0, int meta_index = 0)
{
    gROOT->Reset();
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);

    TString tmp = outFold;
    TString cname = tmp.ReplaceAll("/", "_")+name;
    TCanvas* c = new TCanvas(cname, cname, 600, 600);
    c->cd();
    c->SetBottomMargin(0.15);
    c->SetTopMargin(0.08);
    c->SetLeftMargin(0.17);
    c->SetRightMargin(0.08);
    c->SetLogy();

    int color[10] = {1, 2, 4, 6, 8, 48, 32, 3, 5, 67};

    TLegend* legend = new TLegend(m_meta[meta_index].legendPos_reso[range_index][0],
                                  m_meta[meta_index].legendPos_reso[range_index][1],
                                  m_meta[meta_index].legendPos_reso[range_index][2],
                                  m_meta[meta_index].legendPos_reso[range_index][3]);
    legend->SetNColumns(2);
    legend->SetFillColor(-1);
    legend->SetBorderSize(0);

    std::vector < TH1*> h(nmu, 0);

    for (unsigned int ih = 0; ih < nmu; ++ih)
    {
        h[ih] = (TH1*)m_histos[trigger_index][method_index].resolution[range_index][ih]->Clone();
        int nbins = h[ih]->GetNbinsX();
        h[ih]->Scale(1./h[ih]->Integral(0, nbins+1));

        if (ih == 0)
        {
            h[ih]->GetXaxis()->SetLabelSize(0.05);
            h[ih]->GetYaxis()->SetLabelSize(0.05);

            h[ih]->GetXaxis()->SetRangeUser(m_meta[meta_index].xranges[range_index][0],
                                            m_meta[meta_index].xranges[range_index][1]);
            h[ih]->GetYaxis()->SetRangeUser(m_meta[meta_index].yranges[range_index][0],
                                            m_meta[meta_index].yranges[range_index][1]);

            h[ih]->SetXTitle(m_meta[meta_index].xname);
            h[ih]->GetXaxis()->SetTitleSize(0.05);
            h[ih]->GetXaxis()->SetTitleOffset(1.1);
            h[ih]->SetYTitle("normalized to unity");
            h[ih]->GetYaxis()->SetTitleSize(0.05);
            h[ih]->GetYaxis()->SetTitleOffset(1.5);

            h[ih]->SetLineColor(kBlack);
            h[ih]->SetLineWidth(2);

            h[ih]->Draw("hist");
        }
        else
        {
            if (ih < 10) h[ih]->SetLineColor(color[ih]);
            else h[ih]->SetLineColor(kGray);
            h[ih]->SetLineWidth(2);

            h[ih]->Draw("histsame");
        }

        legend->AddEntry(h[ih], mu_labels[ih], "l");

    }

    if (m_data.size() > 1) legend->Draw();

    AREUSText(m_meta[meta_index].labelPos[range_index][0],
              m_meta[meta_index].labelPos[range_index][1],
              "EM-Middle |#eta|#times#phi=0.5125x0.0125", 1, 0.035);

    myText(m_meta[meta_index].labelPos[range_index][0],
           m_meta[meta_index].labelPos[range_index][1]-0.05,
           range_labels[range_index], 1, 0.035);

    c->SaveAs(outFold+"/"+name+".png");

    if (c)
    {
        c->Close();
        gSystem->ProcessEvents();
        delete c;
        c = 0;
    }
    for (unsigned int i = 0; i < h.size(); ++i)
    {
        if (!h[i]) continue;
        h[i]->Delete();
        h[i] = 0;
    }
    h.clear();

    // std::cout << "    -> Saved : " << outFold << "/" << name << std::endl;
}

void compare_predictions(TString name, TString outFold, int trigger_index = 0, int range_index = 0, int mu_index = 0, int meta_index = 0)
{
    gROOT->Reset();
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);

    TString tmp = outFold;
    TString cname = tmp.ReplaceAll("/", "_")+name;
    TCanvas* c = new TCanvas(cname, cname, 600, 600);
    c->cd();
    c->SetBottomMargin(0.15);
    c->SetTopMargin(0.08);
    c->SetLeftMargin(0.17);
    c->SetRightMargin(0.08);
    c->SetLogy();

    int color[10] = {1, 2, 4, 6, 8, 48, 32, 3, 5, 67};

    TLegend* legend = new TLegend(m_meta[meta_index].legendPos_reso[range_index][0],
                                  m_meta[meta_index].legendPos_reso[range_index][1],
                                  m_meta[meta_index].legendPos_reso[range_index][2],
                                  m_meta[meta_index].legendPos_reso[range_index][3]);
    legend->SetFillColor(-1);
    legend->SetBorderSize(0);

    std::vector < TH1*> h(m_histos[trigger_index].size(), 0);

    for (unsigned int ih = 0; ih < m_histos[trigger_index].size(); ++ih)
    {
        h[ih] = (TH1*)m_histos[trigger_index][ih].prediction[range_index][mu_index]->Clone();
        int nbins = h[ih]->GetNbinsX();
        h[ih]->Scale(1./h[ih]->Integral(0, nbins+1));

        if (ih == 0)
        {
            h[ih]->GetXaxis()->SetLabelSize(0.05);
            h[ih]->GetYaxis()->SetLabelSize(0.05);

            double xmax = 0;
            for (int ib = 0; ib < nbins; ++ib)
            {
                int bin = nbins-ib;
                if (h[ih]->GetBinContent(bin) == 0) continue;
                xmax = bin;
                break;
            }

            h[ih]->GetXaxis()->SetRangeUser(0, xmax);
            h[ih]->GetYaxis()->SetRangeUser(m_meta[meta_index].yranges_pred[range_index][0],
                                            m_meta[meta_index].yranges_pred[range_index][1]);

            h[ih]->SetXTitle("E_T^{pred} [GeV]");
            h[ih]->GetXaxis()->SetTitleSize(0.05);
            h[ih]->GetXaxis()->SetTitleOffset(1.1);
            h[ih]->SetYTitle("normalized to unity");
            h[ih]->GetYaxis()->SetTitleSize(0.05);
            h[ih]->GetYaxis()->SetTitleOffset(1.5);

            h[ih]->SetLineColor(kBlack);
            h[ih]->SetLineWidth(2);

            h[ih]->Draw("hist");
        }
        else
        {
            if (ih < 10) h[ih]->SetLineColor(color[ih]);
            else h[ih]->SetLineColor(kGray);
            h[ih]->SetLineWidth(2);

            h[ih]->Draw("histsame");
        }

        TString legname = m_histos[trigger_index][ih].method_name;
        for (unsigned int i = 0; i < m_meta[meta_index].rename.size(); ++i)
        {
            if (m_meta[meta_index].rename[i][0] != legname) continue;
            legname = m_meta[meta_index].rename[i][1];
            break;
        }
        legend->AddEntry(h[ih], legname, "l");

    }

    if (m_data.size() > 1) legend->Draw();

    AREUSText(m_meta[meta_index].labelPos[range_index][0],
              m_meta[meta_index].labelPos[range_index][1],
              "EM-Middle |#eta|#times#phi=0.5125x0.0125", 1, 0.035);

    myText(m_meta[meta_index].labelPos[range_index][0],
           m_meta[meta_index].labelPos[range_index][1]-0.05,
           range_labels[range_index], 1, 0.035);

    c->SaveAs(outFold+"/"+name+".png");

    if (c)
    {
        c->Close();
        gSystem->ProcessEvents();
        delete c;
        c = 0;
    }
    for (unsigned int i = 0; i < h.size(); ++i)
    {
        if (!h[i]) continue;
        h[i]->Delete();
        h[i] = 0;
    }
    h.clear();

    // std::cout << "    -> Saved : " << outFold << "/" << name << std::endl;

}

// void compare_resolutions(TString name, TString outFold, int trigger_index = 0, int range_index = 0, int mu_index = 0, int meta_index = 0, bool forGaps = false)
void compare_resolutions(TString name, TString outFold, std::vector < TH1* > h, int trigger_index = 0, int range_index = 0, int mu_index = 0, int meta_index = 0, bool forGaps = false, bool forResoPerTrue = false)
{
    gROOT->Reset();
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);

    TString tmp = outFold;
    TString cname = tmp.ReplaceAll("/", "_")+name;
    TCanvas* c = new TCanvas(cname, cname, 600, 600);
    c->cd();
    c->SetBottomMargin(0.15);
    c->SetTopMargin(0.08);
    c->SetLeftMargin(0.17);
    c->SetRightMargin(0.08);
    c->SetLogy();

    int color[10] = {1, 2, 4, 6, 8, 48, 32, 3, 5, 67};

    TLegend* legend = new TLegend(m_meta[meta_index].legendPos_reso[range_index][0],
                                  m_meta[meta_index].legendPos_reso[range_index][1],
                                  m_meta[meta_index].legendPos_reso[range_index][2],
                                  m_meta[meta_index].legendPos_reso[range_index][3]);
    legend->SetFillColor(-1);
    legend->SetBorderSize(0);
/*
    std::vector < TH1*> h(m_histos[trigger_index].size(), 0);

    for (unsigned int ih = 0; ih < m_histos[trigger_index].size(); ++ih)
    {
        if (forGaps == true) {
            h[ih] = (TH1*)m_histos[trigger_index][ih].resolution_for_gaps[range_index][mu_index]->Clone();
        } else {

            h[ih] = (TH1*)m_histos[trigger_index][ih].resolution[range_index][mu_index]->Clone();

        }

 */
    for (unsigned int ih = 0; ih < m_histos[trigger_index].size(); ++ih)
    {

        int nbins = h[ih]->GetNbinsX();
        h[ih]->Scale(1./h[ih]->Integral(0, nbins+1));

        if (ih == 0)
        {
            h[ih]->GetXaxis()->SetLabelSize(0.05);
            h[ih]->GetYaxis()->SetLabelSize(0.05);

            if (forGaps == false && forResoPerTrue == false) {
                h[ih]->GetXaxis()->SetRangeUser(m_meta[meta_index].xranges[range_index][0],
                                                m_meta[meta_index].xranges[range_index][1]);
                h[ih]->GetYaxis()->SetRangeUser(m_meta[meta_index].yranges[range_index][0],
                                                m_meta[meta_index].yranges[range_index][1]);
            } else if (forResoPerTrue == false) {

                // use resolution ranges for all energies
                h[ih]->GetXaxis()->SetRangeUser(m_meta[meta_index].xranges[0][0],
                                                m_meta[meta_index].xranges[0][1]);
                h[ih]->GetYaxis()->SetRangeUser(m_meta[meta_index].yranges[0][0],
                                                m_meta[meta_index].yranges[0][1]);

            } else {
                // Reso / true range
                h[ih]->GetXaxis()->SetRangeUser(-1.1, 1.0);
            }
            if (!forResoPerTrue) {
                h[ih]->SetXTitle(m_meta[meta_index].xname);
            } else {
                h[ih]->SetXTitle("(E_{T}^{pred} - E_{T}^{true}) /  E_{T}^{true}");
            }

            h[ih]->GetXaxis()->SetTitleSize(0.05);
            h[ih]->GetXaxis()->SetTitleOffset(1.1);
            h[ih]->SetYTitle("normalized to unity");
            h[ih]->GetYaxis()->SetTitleSize(0.05);
            h[ih]->GetYaxis()->SetTitleOffset(1.5);

            h[ih]->SetLineColor(kBlack);
            h[ih]->SetLineWidth(2);

            h[ih]->Draw("hist");
        }
        else
        {
            if (ih < 10) h[ih]->SetLineColor(color[ih]);
            else h[ih]->SetLineColor(kGray);
            h[ih]->SetLineWidth(2);

            h[ih]->Draw("histsame");
        }

        TString legname = m_histos[trigger_index][ih].method_name;
        for (unsigned int i = 0; i < m_meta[meta_index].rename.size(); ++i)
        {
            if (m_meta[meta_index].rename[i][0] != legname) continue;
            legname = m_meta[meta_index].rename[i][1];
            break;
        }
        legend->AddEntry(h[ih], legname, "l");

    }

    if (m_data.size() > 1) legend->Draw();

    AREUSText(m_meta[meta_index].labelPos[range_index][0],
              m_meta[meta_index].labelPos[range_index][1],
              "EM-Middle |#eta|#times#phi=0.5125x0.0125", 1, 0.035);

    if (forGaps == true) {
        myText(m_meta[meta_index].labelPos[range_index][0],
               m_meta[meta_index].labelPos[range_index][1]-0.05,
               gap_range_labels[range_index], 1, 0.035);
    } else {
        myText(m_meta[meta_index].labelPos[range_index][0],
               m_meta[meta_index].labelPos[range_index][1]-0.05,
               range_labels[range_index], 1, 0.035);
    }


    c->SaveAs(outFold+"/"+name+".png");

    if (c)
    {
        c->Close();
        gSystem->ProcessEvents();
        delete c;
        c = 0;
    }
    for (unsigned int i = 0; i < h.size(); ++i)
    {
        if (!h[i]) continue;
        h[i]->Delete();
        h[i] = 0;
    }
    h.clear();

    // std::cout << "    -> Saved : " << outFold << "/" << name << std::endl;

}

void compare_rightzeros(TString name, TString outFold, int trigger_index = 0, int mu_index = 0)
{
    gROOT->Reset();
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);

    TString tmp = outFold;
    TString cname = tmp.ReplaceAll("/", "_")+name;
    TCanvas* c = new TCanvas(cname, cname, 600, 600);
    c->cd();
    c->SetBottomMargin(0.15);
    c->SetTopMargin(0.08);
    c->SetLeftMargin(0.17);
    c->SetRightMargin(0.08);
    c->SetLogy();

    int color[10] = {1, 2, 4, 6, 8, 48, 32, 3, 5, 67};

    TLegend* legend = new TLegend(0.4, 0.65, 0.9, 0.80);
    legend->SetFillColor(-1);
    legend->SetBorderSize(0);

    std::vector < TH1*> h(m_histos[trigger_index].size(), 0);

    double ymax = 0;
    double ymin = 10000000;
    for (unsigned int ih = 0; ih < m_histos[trigger_index].size(); ++ih)
    {
        unsigned int nbins = m_histos[trigger_index][ih].correct_zeros[mu_index]->GetNbinsX();
        for (unsigned int ib = 1; ib < nbins+1; ++ib)
        {
            if (ymax < m_histos[trigger_index][ih].correct_zeros[mu_index]->GetBinContent(ib)) ymax = m_histos[trigger_index][ih].correct_zeros[mu_index]->GetBinContent(ib);
            if (ymin > m_histos[trigger_index][ih].correct_zeros[mu_index]->GetBinContent(ib)
                && m_histos[trigger_index][ih].correct_zeros[mu_index]->GetBinContent(ib) != 0) ymin = m_histos[trigger_index][ih].correct_zeros[mu_index]->GetBinContent(ib);
        }
    }

    for (unsigned int ih = 0; ih < m_histos[trigger_index].size(); ++ih)
    {
        h[ih] = (TH1*)m_histos[trigger_index][ih].correct_zeros[mu_index]->Clone();

        if (ih == 0)
        {
            h[ih]->GetXaxis()->SetLabelSize(0.05);
            h[ih]->GetYaxis()->SetLabelSize(0.05);

            h[ih]->GetYaxis()->SetRangeUser(ymin*0.5, ymax*50);
            //h[ih]->GetYaxis()->SetRangeUser(0.01,ymax*1.4);

            h[ih]->SetYTitle("Counts");
            h[ih]->GetYaxis()->SetTitleSize(0.05);
            h[ih]->GetYaxis()->SetTitleOffset(1.5);

            h[ih]->SetLineColor(kBlack);
            h[ih]->SetLineWidth(2);

            h[ih]->Draw("hist");
        }
        else
        {
            if (ih < 10) h[ih]->SetLineColor(color[ih]);
            else h[ih]->SetLineColor(kGray);
            h[ih]->SetLineWidth(2);

            h[ih]->Draw("histsame");
        }


        TString legname = m_histos[trigger_index][ih].method_name;
        for (unsigned int i = 0; i < m_meta[0].rename.size(); ++i)
        {
            if (m_meta[0].rename[i][0] != legname) continue;
            legname = m_meta[0].rename[i][1];
            break;
        }
        legend->AddEntry(h[ih], legname, "l");

    }

    if (m_data.size() > 1) legend->Draw();

    AREUSText(0.2, 0.85, "EM-Middle |#eta|#times#phi=0.5125x0.0125", 1, 0.035);

    c->SaveAs(outFold+"/"+name+".png");

    if (c)
    {
        c->Close();
        gSystem->ProcessEvents();
        delete c;
        c = 0;
    }
    for (unsigned int i = 0; i < h.size(); ++i)
    {
        if (!h[i]) continue;
        h[i]->Delete();
        h[i] = 0;
    }
    h.clear();

    // std::cout << "    -> Saved : " << outFold << "/" << name << std::endl;

}

void draw_2D_in_canvas(TH2* hin, TString name, TString outFold,
                       TString label, TString xtitle, TString ytitle,
                       float xmin, float xmax, float ymin, float ymax,
                       float xlabel = 0.2, float ylabel = 0.85,
                       bool logx = true, bool logy = false, TString Elabel = "",
                       bool include_profile = false, int rebin_profile = 20)
{
    gROOT->Reset();
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);


    TH2* h = (TH2*)hin->Clone();
    int nbinsx = h->GetNbinsX();
    int nbinsy = h->GetNbinsY();
    h->Scale(1./h->Integral(0, nbinsx+1, 0, nbinsy+1));
    //h->Rebin(4);

    TString tmp = outFold;
    TCanvas* c = new TCanvas("canvas", "canvas", 700, 600);
    c->cd();
    c->SetBottomMargin(0.15);
    c->SetTopMargin(0.08);
    c->SetLeftMargin(0.17);
    c->SetRightMargin(0.26);
    if (logx) c->SetLogx();
    if (logy) c->SetLogy();


    h->GetXaxis()->SetLabelSize(0.05);
    h->GetXaxis()->SetRangeUser(xmin, xmax);

    h->SetXTitle(xtitle);
    h->GetXaxis()->SetTitleSize(0.05);
    h->GetXaxis()->SetTitleOffset(1.1);


    h->GetYaxis()->SetLabelSize(0.05);
    h->GetYaxis()->SetRangeUser(ymin, ymax);

    h->SetYTitle(ytitle);
    h->GetYaxis()->SetTitleSize(0.05);
    h->GetYaxis()->SetTitleOffset(1.5);

    h->GetZaxis()->SetLabelSize(0.05);
    h->GetZaxis()->SetTitle("Normalized to unity");
    h->GetZaxis()->SetTitleSize(0.05);
    h->GetZaxis()->SetTitleOffset(2.0);

    // h->SetLineColor(kBlack);
    // h->SetLineWidth(2);


    h->Draw("COLZ");
    if (include_profile) {
        TProfile* p = ((TH2D*) h->Clone())->RebinX(rebin_profile)->ProfileX("profile", 1, -1, "s");
        p->SetLineColor(kBlack);
        p->SetLineWidth(2);
        p->SetMarkerColor(kBlack);

        p->SetFillColorAlpha(kBlack, 0.8);

        p->SetFillStyle(0);
        p->Draw("HIST E3 SAME");
    }

    AREUSText(xlabel, ylabel,
              "Simulation", 1, 0.04, 0.12);

    //myText(xlabel, ylabel-0.05, "EM-Middle |#eta|#times#phi=0.5125x0.0125", 1, 0.04);
    if (Elabel != "") myText(xlabel, ylabel-0.05, Elabel, 1, 0.04);
    else if (xtitle == "Gap [BC]") myText(xlabel, ylabel-0.05, "<#mu> = 140, E_{T}^{true} > 240 MeV", 1, 0.04);

    TString text = label;
    for (unsigned int i = 0; i < m_meta[0].rename.size(); ++i)
    {
        if (m_meta[0].rename[i][0] != text) continue;
        text = m_meta[0].rename[i][1];
        break;
    }

    if (Elabel != "" || xtitle == "Gap [BC]") myText(xlabel, ylabel-0.1, text, 1, 0.04);
    else myText(xlabel, ylabel-0.05, text, 1, 0.04);

    c->SaveAs(outFold+"/"+name+".png");
    if (PDF_PLOTS) {
        c->SaveAs(outFold+"/"+name+".pdf");
    }


    if (c)
    {
        c->Close();
        gSystem->ProcessEvents();
        delete c;
        c = 0;
    }
    if (h) h->Delete();



    // std::cout << "    -> Saved : " << outFold << "/" << name << std::endl;

    // delete h;
    // delete c;
}

void draw_in_canvas(TH1* hin, TString name, TString outFold, TString label, TString xtitle,
                    //float xmin, float xmax, float ymin, float ymax,
                    float xlabel = 0.2, float ylabel = 0.8)
{
    gROOT->Reset();
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);


    TH1* h = (TH1*)hin->Clone();
    int nbins = h->GetNbinsX();
    h->Scale(1./h->Integral(0, nbins+1));
    //h->Rebin(4);

    TCanvas* c = new TCanvas("canvas", "canvas", 600, 600);
    c->cd();
    c->SetBottomMargin(0.15);
    c->SetTopMargin(0.08);
    c->SetLeftMargin(0.17);
    c->SetRightMargin(0.08);
    c->SetLogy();

    h->GetXaxis()->SetLabelSize(0.05);
    h->GetYaxis()->SetLabelSize(0.05);

    // h->GetXaxis()->SetRangeUser(xmin, xmax);
    // h->GetYaxis()->SetRangeUser(ymin, ymax);

    h->SetXTitle(xtitle);
    h->GetXaxis()->SetTitleSize(0.05);
    h->GetXaxis()->SetTitleOffset(1.1);
    h->SetYTitle("normalized to unity");
    h->GetYaxis()->SetTitleSize(0.05);
    h->GetYaxis()->SetTitleOffset(1.5);

    h->SetLineColor(kBlack);
    h->SetLineWidth(2);

    h->Draw("hist");

    myText(xlabel, ylabel, label, 1, 0.04);

    c->SaveAs(outFold+"/"+name+".png");

    if (c)
    {
        c->Close();
        gSystem->ProcessEvents();
        delete c;
        c = 0;
    }
    if (h) h->Delete();


    // std::cout << "    -> Saved : " << outFold << "/" << name << std::endl;

    // delete h;
    // delete c;
}



void draw_profiles_unc(std::vector < TH2* > h2din, std::vector < TString > names, TString name, TString outFold)
{

    gROOT->Reset();
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);

    TString tmp = outFold;
    TString cname = tmp.ReplaceAll("/", "_")+name;
    TCanvas* c = new TCanvas(cname, cname, 600, 600);
    c->cd();
    c->SetBottomMargin(0.15);
    c->SetTopMargin(0.08);
    c->SetLeftMargin(0.17);
    c->SetRightMargin(0.08);

    int color[10] = {1, 2, 4, 6, 8, 48, 32, 3, 5, 67};

    TLegend* legend = new TLegend(0.3, 0.7, 0.7, 0.9);
    legend->SetFillColor(-1);
    legend->SetBorderSize(0);

    for (unsigned int i = 0; i < h2din.size(); ++i)
    {
        TH2* h2d = (TH2*)h2din[i]->Clone();
        h2d->RebinX(10);

        TProfile* p = h2d->ProfileX(names[i]+"_profile", 1, -1, "s");
        //TProfile* p = h2d->ProfileX();

        int nbins = p->GetNbinsX();
        double xmin = p->GetXaxis()->GetBinLowEdge(1);
        double xmax = p->GetXaxis()->GetBinLowEdge(nbins+1);

        TH1* he = new TH1D(names[i]+"_profile_err", names[i]+"_profile_err", nbins, xmin, xmax);

        for (int i = 1; i < nbins+1; ++i)
        {
            double err = p->GetBinError(i);
            he->SetBinContent(i, err);
        }

        he->SetLineColor(color[i]);
        he->SetMarkerColor(color[i]);

        if (i == 0)
        {

            //he->GetXaxis()->SetTitle("E_{T}^{true}");
            he->GetYaxis()->SetTitle("#sigma(E_{T}^{pred} - E_{T}^{true}) [GeV]");
            he->GetYaxis()->SetTitleSize(0.05);

            he->GetXaxis()->SetTitle("E_{T}^{true} [GeV]");
            he->GetXaxis()->SetTitleSize(0.05);

            he->GetYaxis()->SetRangeUser(0., 0.4);

            he->Draw("hist");
        }
        else
        {
            he->Draw("histsame");
        }

        legend->AddEntry(he, names[i], "lp");

    }
    legend->Draw();

    c->SaveAs(outFold+"/"+name+".png");

}



void draw_profiles(std::vector < TH2* > h2din, std::vector < TString > names, TString name, TString outFold)
{

    gROOT->Reset();
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);

    TString tmp = outFold;
    TString cname = tmp.ReplaceAll("/", "_")+name;
    TCanvas* c = new TCanvas(cname, cname, 600, 600);
    c->cd();
    c->SetBottomMargin(0.15);
    c->SetTopMargin(0.08);
    c->SetLeftMargin(0.17);
    c->SetRightMargin(0.08);

    int color[10] = {1, 2, 4, 6, 8, 48, 32, 3, 5, 67};

    TLegend* legend = new TLegend(0.5, 0.7, 0.85, 0.9);
    legend->SetFillColor(-1);
    legend->SetBorderSize(0);

    for (unsigned int i = 0; i < h2din.size(); ++i)
    {
        TH2* h2d = (TH2*)h2din[i]->Clone();
        h2d->RebinX(10);

        //TProfile* p = h2d->ProfileX(names[i]+"_profile", 1, -1, "s");
        TProfile* p = h2d->ProfileX();

        p->SetLineColor(color[i]);
        p->SetMarkerColor(color[i]);

        if (i == 0)
        {

            p->GetXaxis()->SetTitle("E_{T}^{true} [GeV]");
            p->GetXaxis()->SetTitleSize(0.05);
            //p->GetYaxis()->SetTitle("<E_{T}^{pred} - E_{T}^{true}> +/- #sigma(E_{T}^{pred} - E_{T}^{true})");
            p->GetYaxis()->SetTitle("<E_{T}^{pred} - E_{T}^{true}> [GeV]");
            p->GetYaxis()->SetTitleSize(0.05);

            p->GetYaxis()->SetRangeUser(-0.3, 0.2);

            p->Rebin()->Draw("hist");
        }
        else
        {
            p->Rebin()->Draw("histsame");
        }

        legend->AddEntry(p, names[i], "lp");

    }
    legend->Draw();

    c->SaveAs(outFold+"/"+name+".png");

}



void draw_profiles_unc_E(std::vector < TH2* > h2din, std::vector < TString > names, TString name, TString outFold)
{

    gROOT->Reset();
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);

    TString tmp = outFold;
    TString cname = tmp.ReplaceAll("/", "_")+name;
    TCanvas* c = new TCanvas(cname, cname, 600, 600);
    c->cd();
    c->SetBottomMargin(0.15);
    c->SetTopMargin(0.08);
    c->SetLeftMargin(0.17);
    c->SetRightMargin(0.08);

    int color[10] = {1, 2, 4, 6, 8, 48, 32, 3, 5, 67};

    TLegend* legend = new TLegend(0.3, 0.7, 0.7, 0.9);
    legend->SetFillColor(-1);
    legend->SetBorderSize(0);

    for (unsigned int i = 0; i < h2din.size(); ++i)
    {
        TH2* h2d = (TH2*)h2din[i]->Clone();
        // h2d->RebinX(10);
        // h2d->RebinY(10);

        TProfile* p = h2d->ProfileX(names[i]+"_profile", 1, -1, "s");
        //TProfile* p = h2d->ProfileX();

        int nbins = p->GetNbinsX();
        double xmin = p->GetXaxis()->GetBinLowEdge(1);
        double xmax = p->GetXaxis()->GetBinLowEdge(nbins+1);

        TH1* he = new TH1D(names[i]+"_profile_err", names[i]+"_profile_err", nbins, xmin, xmax);

        for (int i = 1; i < nbins+1; ++i)
        {
            double err = p->GetBinError(i);
            he->SetBinContent(i, err);
        }

        he->SetLineColor(color[i]);
        he->SetMarkerColor(color[i]);

        if (i == 0)
        {

            he->GetXaxis()->SetTitle("E_{T}^{true}");
            he->GetYaxis()->SetTitle("<E_{T}^{pred} - E_{T}^{true}> +/- #sigma(E_{T}^{pred} - E_{T}^{true}) [GeV]");
            //he->GetYaxis()->SetTitle("<E_{T}^{pred}>");

            he->GetYaxis()->SetRangeUser(0., 0.4);
            he->GetXaxis()->SetRangeUser(0., MAX_ENERGY + 0.5);

            he->Draw("hist");
        }
        else
        {
            he->Draw("histsame");
        }

        legend->AddEntry(he, names[i], "lp");

    }
    legend->Draw();

    c->SaveAs(outFold+"/"+name+".png");

}



void draw_profiles_E(std::vector < TH2* > h2din, std::vector < TString > names, TString name, TString outFold)
{

    gROOT->Reset();
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);

    TString tmp = outFold;
    TString cname = tmp.ReplaceAll("/", "_")+name;
    TCanvas* c = new TCanvas(cname, cname, 600, 600);
    c->cd();
    c->SetBottomMargin(0.15);
    c->SetTopMargin(0.08);
    c->SetLeftMargin(0.17);
    c->SetRightMargin(0.08);

    int color[10] = {1, 2, 4, 6, 8, 48, 32, 3, 5, 67};

    TLegend* legend = new TLegend(0.2, 0.6, 0.6, 0.9);
    legend->SetFillColor(-1);
    legend->SetBorderSize(0);

    for (unsigned int i = 0; i < h2din.size(); ++i)
    {
        TH2* h2d = (TH2*)h2din[i]->Clone();
        // h2d->RebinX(10);
        // h2d->RebinY(10);

        //TProfile* p = h2d->ProfileX(names[i]+"_profile", 1, -1, "s");
        TProfile* p = h2d->ProfileX();

        p->SetLineColor(color[i]);
        p->SetMarkerColor(color[i]);

        if (i == 0)
        {

            p->GetXaxis()->SetTitle("E_{T}^{true} [GeV]");
            p->GetXaxis()->SetTitleSize(0.05);

            //p->GetYaxis()->SetTitle("<E_{T}^{pred} - E_{T}^{true}> +/- #sigma(E_{T}^{pred} - E_{T}^{true}) [GeV]");
            p->GetYaxis()->SetTitle("<E_{T}^{pred}> [GeV]");
            p->GetYaxis()->SetTitleSize(0.05);

            p->GetXaxis()->SetRangeUser(0., MAX_ENERGY + 0.5);
            p->GetYaxis()->SetRangeUser(0., MAX_ENERGY + 0.5);

            p->Draw();
        }
        else
        {
            p->Draw("same");
        }

        legend->AddEntry(p, names[i], "lp");

    }
    legend->Draw();

    c->SaveAs(outFold+"/"+name+".png");

}


void draw_profiles_bt(std::vector < TH2* > h2din, std::vector < TString > names, TString name, TString outFold, TString label, float y_range_low = -0.1, float y_range_high = 0.14, float bunchtrain_length = 80)
{

    gROOT->Reset();
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);

    TString tmp = outFold;
    TString cname = tmp.ReplaceAll("/", "_")+name;
    TCanvas* c = new TCanvas(cname, cname, 600, 600);
    c->cd();
    c->SetBottomMargin(0.15);
    c->SetTopMargin(0.08);
    c->SetLeftMargin(0.17);
    c->SetRightMargin(0.08);

    int color[10] = {1, 2, 4, 6, 8, 48, 32, 3, 5, 67};

    TLegend* legend = new TLegend(0.5, 0.7, 0.85, 0.9);
    legend->SetFillColor(-1);
    legend->SetBorderSize(0);


    for (unsigned int i = 0; i < h2din.size(); ++i)
    {
        TH2* h2d = (TH2*)h2din[i]->Clone();
        //h2d->RebinX(10);


        TProfile* p = h2d->ProfileX(names[i]+"_profile", 1, -1, "s");
        //TProfile* p = h2d->ProfileX();

        p->SetLineColor(color[i]);
        //p->SetFillColorAlpha(color[i], 0.4);
        p->SetMarkerColor(color[i]);
        //p->SetFillStyle(3001+i);


        if (i == 0)
        {

            p->GetXaxis()->SetTitle("Bunchtrain [BC]");
            p->GetXaxis()->SetTitleSize(0.05);
            //p->GetYaxis()->SetTitle("<E_{T}^{pred} - E_{T}^{true}> +/- #sigma(E_{T}^{pred} - E_{T}^{true})");
            p->GetYaxis()->SetTitle("#sigma and mean for <E_{T}^{pred} - E_{T}^{true}> [GeV]");
            p->GetYaxis()->SetTitleSize(0.05);

            p->GetYaxis()->SetRangeUser(y_range_low, y_range_high);

            p->SetFillColorAlpha(color[i], 0.2);
            p->SetFillStyle(1001);
            p->Draw("HIST E3");
        }
        else
        {
            p->SetFillColorAlpha(color[i], 0.15);
            p->SetFillStyle(1001);
            p->Draw("HIST E3 SAME");
        }

        legend->AddEntry(p, names[i], "lp");

    }
    legend->Draw();

    float bc_split_pos = bunchtrain_length-8;

    TLine *l_bc_split = new TLine(bc_split_pos, y_range_low, bc_split_pos, y_range_high);
    l_bc_split->SetLineColor(kBlack);
    l_bc_split->Draw();

    myText(0.20, 0.88-0.02, label, 1, 0.035);

    myText(0.05 + bc_split_pos/bunchtrain_length*0.75, 0.3, "Filled #xleftrightarrow Empty", 1, 0.035);

    /*
    TLine *l = new TLine(0, 0, 80, 0);
    l->SetLineColor(kGreen);
    l->Draw();
    */


    c->SaveAs(outFold+"/"+name+".png");

}
