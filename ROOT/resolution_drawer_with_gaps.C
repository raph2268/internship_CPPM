
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <map>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>

#include <TROOT.h>
#include <TSystem.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1.h>
#include <TH2.h>
#include <TGraph.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TString.h>
#include <TStyle.h>
#include <TProfile.h>
#include <TPad.h>
#include <TMultiGraph.h>
#include <TGraphAsymmErrors.h>
#include <TGraphErrors.h>

#include "TLatex.h"
#include "TLine.h"
#include "TPave.h"
#include "TPad.h"
#include "TMarker.h"
#include "TMathText.h"
#include "TArrow.h"



///// ---------------------------------------------------
///// Defining naming convention, bins, scale vars ...

// Flag to create algo comps (if not required setting this to false lowers memory usage)
const bool ALGO_COMPS = false;
const bool PDF_PLOTS = true;
const bool RESO_FOR_GAPS = false;

// Max energy used for bins, can be changed with commandline argument --max
float MAX_ENERGY = 5;

float BUNCHTRAIN_LENGTH = 80;

const TString m_debug = true;
const std::string debug_header = "\033[1;36mDEBUG::";
const std::string debug_close = "\033[0m";


std::vector < std::vector < TString >> pred_names_labels = {{"sequence_ofmax_eT", "OF with MaxFinder"},
                                                            {"lstm_no_dense", "LSTM (sliding)"},
                                                            {"lstm_seq5", "LSTM (sliding)"},
                                                            {"lstm_singlecell", "LSTM (single)"},
                                                            {"lstm_stateful", "LSTM (single)"},
                                                            {"vanilla_8unit", "Vanilla RNN"},
                                                            {"rnn_seq5", "Vanilla RNN"},
                                                            {"vanilla_8unit", "Vanilla RNN"},
                                                            {"rnn_hls", "Vanilla RNN (HLS)"},
                                                            {"rnn_patience=8", "patience=8"},
                                                            {"rnn_patience=12", "patience=12"},
                                                            {"rnn_patience=16", "patience=16"},
                                                            {"rnn_patience=20", "patience=20"},
                                                            {"rnn_ref", "ref"},
                                                            {"time step 30", "time step = 30"},
                                                            {"rnn_64", "Batch size = 64"}};


// Energy ranges for plots
const unsigned int nrange = 5;
std::vector < std::vector < float>> range_vals = {{0., 14000.}, {0, 0}, {0, 0.24}, {0.24, 14000.}, {1.0, 14000.}};      // index 0 is all !!! important !!!
//std::vector < std::vector < float>> range_vals = {{0., 140.}, {0, 0}, {1, 2}, {3, 5}, {6, 15}, {16, 45}, {46, 14000.}};      // index 0 is all !!! important !!!
TString ranges[nrange] = {"all", "e0", "0_0p240", "0p24_inf", "1p0_inf"};
TString range_labels[nrange] = {"All energies", "No deposit E=0", "0 MeV #leq E < 240 MeV", "E #geq 240 MeV", "E #geq 1 GeV"};
// TString range_labels[nrange] = {"All gaps", "No gap ", "1, 2", "3, 5", "6, 15", "16, 45", "46, 14000"};


// Define ranges and labels for resolution per gap bin plots
const unsigned int ngaps = 5;
std::vector < std::vector < float>> gap_vals = {{1., 15.}, {15, 18}, {18, 22}, {22, 30}, {30, 14000}}; // 0-15/18, 15/18-22/25; 22/25-30; 30+
TString gap_ranges[ngaps] = {"0_15", "15_18", "18_22", "22_30", "30_inf"};
TString gap_range_labels[ngaps] = {"Gap 0 - 15", "Gap 16 - 18", "Gap 19 - 22", "Gap 23 - 30", "Gap > 31 "};

// Energy ranges for gap vs resolution
const unsigned int neng_gaps = 3;
std::vector < std::vector < float>> eng_gaps_range_vals = {{0., 1.}, {1, 3}, {3, 14000}};      // index 0 is all !!! important !!!
//std::vector < std::vector < float>> range_vals = {{0., 140.}, {0, 0}, {1, 2}, {3, 5}, {6, 15}, {16, 45}, {46, 14000.}};      // index 0 is all !!! important !!!
TString eng_gaps_ranges[neng_gaps] = {"low", "mid", "high"};
TString eng_gaps_range_labels[neng_gaps] = {"0.24 GeV #leq E < 1 GeV", "1 GeV < E < 3 GeV", "3 GeV #leq E "};


// Pile-up conditions // need that at configuration level (commented part goes with Thomas' stuffs in the main
const unsigned int nmu = 1; //6; //
TString muref = "mu140"; // which of the mu values is to be taken as reference for initialisation (undef methods at muref are skiped, undef methods in others will make it crash)
TString mus[nrange] = {"mu140"}; // {"mu140","mu100","mu120","mu160","mu180","mu200"}; //
TString mu_labels[nrange] = {"#mu = 140"}; // {"#mu = 140","#mu = 100","#mu = 120","#mu = 160","#mu = 180","#mu = 200"}; //

// For comparison-holder - set of names for saving if(a==0) or !=0, if(b==0) or !=0, no-0 or all BCs
const unsigned int nzeroset = 6;
TString zerosets[nzeroset] = {"all", "refnonzero", "refzero", "altnonzero", "altzero", "nozero"};
TString zeroset_labels[nzeroset] = {"All BCs included", "a = 0", "a != 0", "b = 0", "b != 0", "a & b != 0"};

std::vector < std::vector < float>> bt_y_range_vals = {{-0.1, 0.14}, {-0.1, 0.14}, {-0.08, 0.12}, {-0.36, 0.34}, {-0.4, 0.4}};


////// ---------------------------------------------------
////// Data containers
#include "plot_utils/struct_def.h"

std::vector < ResoHolder > m_data;
std::vector < std::vector < HistoHolder>> m_histos;      // full container with two levels - root(inside) is which method - high level which method is zero or non-zero

std::vector < CompHolder > m_comps;
std::vector < HistoMeta > m_meta;

#include "plot_utils/utils.C"
#include "plot_utils/plot_draws.C"


bool align_single_algo(TString name,
                       unsigned int pred_delay, unsigned int pred_cut,
                       unsigned int ref_delay, unsigned int ref_cut)
{

    bool found;
    for (unsigned int i_rh = 0; i_rh < m_data.size(); ++i_rh)
    {
        if (m_data[i_rh].name != name) continue;
        found = true;

        // end at the last available element, ie earlier cut arriving
        unsigned int end = m_data[i_rh].bcid.size() - ref_cut;
        if (ref_cut < pred_cut) end = m_data[i_rh].bcid.size() - pred_cut;
        if (end > m_data[i_rh].bcid.size() - ref_delay) end = m_data[i_rh].bcid.size() - ref_delay;
        if (end > m_data[i_rh].bcid.size() - pred_delay) end = m_data[i_rh].bcid.size() - pred_delay;

        for (unsigned int i_bc = 0; i_bc < m_data[i_rh].bcid.size(); ++i_bc)
        {
            m_data[i_rh].reference[i_bc] = m_data[i_rh].reference[i_bc+ref_delay];
            m_data[i_rh].prediction[i_bc] = m_data[i_rh].prediction[i_bc+pred_delay];
            m_data[i_rh].resolution[i_bc] = m_data[i_rh].prediction[i_bc]-m_data[i_rh].reference[i_bc];
            m_data[i_rh].reso_per_true[i_bc] = (m_data[i_rh].prediction[i_bc]-m_data[i_rh].reference[i_bc])/m_data[i_rh].reference[i_bc];
            if (i_bc == end) break;
        }

        for (unsigned int i_bc = m_data[i_rh].bcid.size()-1; i_bc >= 0; --i_bc)
        {
            remove_resoholder_bcid(m_data.at(i_rh), i_bc);
            if (i_bc == end) break;
        }

    }
    if (!found)
    {
        std::cout << "WARNING::align_single_algo::Algorithm " << name << "not found" << std::endl;
        return false;
    }
    return true;

}


// read from two input txt files
bool read_files(std::string ref_fname, std::string pred_fname, TString name = "", TString mu = "mu140",
                int ref_delay = 0, int ref_cut = 0, int pred_delay = 0, int pred_cut = 0, unsigned int init_bcid = 0)
{

    std::cout << "INFO::read_files::Add new algo " << name << " using "<< pred_fname << " with ref " << ref_fname << std::endl;
    std::cout << "INFO::read_files::  -> delays : ref=" << ref_delay << " pred=" << pred_delay << std::endl;
    std::cout << "INFO::read_files::  -> cuts   : ref=" << ref_cut << " pred=" << pred_cut << std::endl;
    /// build histo holder
    ResoHolder rh;
    rh.name = name;
    rh.mu = mu;
    rh.mu_index = find_mu_index(mu, nmu, mus);

    std::vector < float > e_vs_bc;
    bool success = false;


    /// read reference
    success = fill_vec_from_txtfile(ref_fname, rh.reference);
    //std::cout << "??g " << rh.reference.size() << std::endl;
    if (!success)
    {
        rh.reference.clear();
        return false;
    }


    /// read prediction
    success = fill_vec_from_txtfile(pred_fname, rh.prediction);
    if (!success)
    {
        rh.reference.clear();
        rh.prediction.clear();
        return false;
    }


    if (m_debug)
    {
        std::cout << std::endl << debug_header
                  << "read_files::Dumping for initial lists -----------------------"
                  << std::endl
                  << "Init sizes : ref = " << rh.reference.size() << "  &  pred = " << rh.prediction.size()
                  << std::endl
                  << "Print 10 el before and 10 el after ref_delay and pred_delay in ref and pred lists : "
                  << std::endl;

        int run = 20;
        int start_ref = ref_delay - 10;
        if (start_ref < 0)
        {
            start_ref = 0;
            run = ref_delay+10;
        }
        int start_pred = pred_delay - 10;
        if (start_pred < 0)
        {
            start_pred = 0;
            if (run > pred_delay + 10) run = pred_delay + 10;
        }

        for (int i = 0; i < run; ++i)
        {
            std::cout << "  - el=" << i << " : ref = " << rh.reference[i+ref_delay] << "  &  pred = " << rh.prediction[i+pred_delay]
                      << std::endl;
        }
        std::cout << debug_close << std::endl;
    }


    //// resize predictions
    if (ref_cut > 0) rh.reference.erase(rh.reference.end() - ref_cut, rh.reference.end());
    if (ref_delay > 0) rh.reference.erase(rh.reference.begin(), rh.reference.begin()+ref_delay);
    if (pred_cut > 0) rh.prediction.erase(rh.prediction.end() - pred_cut, rh.prediction.end());
    if (pred_delay > 0) rh.prediction.erase(rh.prediction.begin(), rh.prediction.begin()+pred_delay);

    if (rh.prediction.size() > rh.reference.size())
    {
        std::cout << "Warning::read_files::prediction size > reference sizes ... erasing last predictions" << std::endl;
        rh.prediction.erase(rh.prediction.begin() + rh.reference.size(), rh.prediction.end());
    }
    if (rh.prediction.size() < rh.reference.size())
    {
        std::cout << "Warning::read_files::reference size > prediction sizes ... erasing last references" << std::endl;
        rh.reference.erase(rh.reference.begin() + rh.prediction.size(), rh.reference.end());
    }


    //// create bcid and fill resolution
    rh.bcid.clear();
    rh.resolution.clear();
    for (unsigned int i = 0; i < rh.prediction.size(); ++i)
    {
        rh.bcid.push_back(i + init_bcid);
        rh.resolution.push_back(rh.prediction[i] - rh.reference[i]);
        rh.reso_per_true.push_back((rh.prediction[i] - rh.reference[i])/rh.reference[i]);

    }



    if (m_debug)
    {
        std::cout << std::endl << debug_header
                  << "read_files::Dumping for final lists -----------------------"
                  << std::endl
                  << "Final sizes : ref = " << rh.reference.size() << "  &  pred = " << rh.prediction.size()
                  << std::endl
                  << "Print 10 elements in ref and pred lists : "
                  << std::endl;

        for (int i = 0; i < 10; ++i)
        {
            std::cout << "  - el=" << i << " : ref = " << rh.reference[i] << "  &  pred = " << rh.prediction[i]
                      << " => reso = " << rh.resolution[i] << "  at  bcid = " << rh.bcid[i]
                      << std::endl;
        }
        std::cout << debug_close << std::endl;
    }

    /// save data, clean up and return
    m_data.push_back(rh);

    return true;

}

// read from one root file formated as the output of the training code
bool read_from_tree(TString root_fname, std::vector < TString > pred_names, TString mu = "mu140", unsigned int init_bcid = 0)
{
    std::cout << "INFO::read_from_tree::Extract ttree from " << root_fname << std::endl;
    std::cout << "INFO::read_from_tree::  -> pred_names to get : ";

    TFile* file = new TFile(root_fname, "read");

    // general variables
    unsigned int n_p = pred_names.size();
    unsigned int first_mdata_index = m_data.size();

    // tree reading variables
    Float_t sequence_dig_eT = 0;
    Float_t sequence_hit_eT = 0;
    // Float_t sequence_ofmax_eT = 0; -> now goes in prediction list
    Float_t* predictions = new Float_t[n_p];

    // Store the distance to previous high energy signal
    Float_t sequence_gap_to_signal = 0;
    Float_t sequence_gap_to_signal_for_all_bcs = 0;

    Float_t sequence_bunchtrain = 0;


    for (unsigned int i_p = 0; i_p < n_p; ++i_p)
    {
        ResoHolder rh;
        std::cout << "   " << pred_names[i_p];
        predictions[i_p] = 0;
        rh.name = pred_names[i_p];

        rh.mu   = mu;
        rh.mu_index = find_mu_index(mu, nmu, mus);
        m_data.push_back(rh);
    }
    std::cout << std::endl;

    // get and set tree
    TTree* ttree = (TTree*)file->Get("Events");
    ttree->SetMakeClass(1);
    //ttree->SetBranchStatus("*", 0);
    ttree->SetBranchAddress("sequence_dig_eT", &sequence_dig_eT);
    ttree->SetBranchAddress("sequence_hit_eT", &sequence_hit_eT);
    ttree->SetBranchAddress("sequence_gap_to_signal", &sequence_gap_to_signal);
    ttree->SetBranchAddress("sequence_gap_to_signal_for_all_bcs", &sequence_gap_to_signal_for_all_bcs);

    ttree->SetBranchAddress("sequence_bunchtrain", &sequence_bunchtrain);

    for (unsigned int i_p = 0; i_p < n_p; ++i_p)
    {
        ttree->SetBranchAddress(pred_names[i_p], &predictions[i_p]);
    }

    // loop over events
    Long64_t nentries = ttree->GetEntries();
    for (Long64_t i_e = 0; i_e < nentries; ++i_e)
    {
        if (i_e < init_bcid) continue;
        ttree->GetEntry(i_e);

        // skip invalid bcids
        bool touse = true;
        for (unsigned int i_p = 0; i_p < n_p; ++i_p)
        {
            if (predictions[i_p] < -13999 && predictions[i_p] > -14001)
            {
                touse = false;
                break;
            }
        }
        if (!touse && m_debug)
        {
            std::cout << debug_header << "Remove " << i_e << std::endl << " -> ";
            for (unsigned int i_p = 0; i_p < n_p; ++i_p)
            {
                if (i_p > 0) std::cout << " : ";
                std::cout << pred_names[i_p] << "=" << predictions[i_p];
            }
            std::cout << debug_close << std::endl;
        }

        if (!touse) continue;

        // fill vectors
        for (unsigned int i_p = 0; i_p < n_p; ++i_p)
        {
            // std::cout << "Evt " << i_e << " with E = " << sequence_hit_eT << ", method " << i_p << " (" << pred_names[i_p]
            //      << " ) = " << predictions[i_p] << std::endl;
            unsigned int this_mdata = first_mdata_index + i_p;
            m_data[this_mdata].bcid.push_back((unsigned int)i_e);
            m_data[this_mdata].reference.push_back(sequence_hit_eT);
            m_data[this_mdata].gap_to_signal.push_back(sequence_gap_to_signal);
            m_data[this_mdata].gap_to_sig_all_bcs.push_back(sequence_gap_to_signal_for_all_bcs);
            m_data[this_mdata].bunchtrain.push_back(sequence_bunchtrain);
            m_data[this_mdata].prediction.push_back(predictions[i_p]);
            m_data[this_mdata].resolution.push_back(predictions[i_p] - sequence_hit_eT);
            m_data[this_mdata].reso_per_true.push_back((predictions[i_p] - sequence_hit_eT)/sequence_hit_eT);
        }
    }

    file->Clear();
    file->Close();
    delete file;

    std::cout << "INFO::read_from_tree::Exit" << std::endl;
    return true;

}




////// ---------------------------------------------------
////// function to align all predictions at a given mu
bool align_methods()
{
    // will only plot on bcids where all method outputs are valid
    // different mu => different run with its own bcid sequence - not comparable event by event
    unsigned int n_resoholder = m_data.size();
    unsigned int first_bc[nmu];
    unsigned int last_bc[nmu];
    for (unsigned int i_mu = 0; i_mu < nmu; ++i_mu)
    {
        first_bc[i_mu] = 0;
        last_bc[i_mu] = 0;
    }


    // get first and last valid BC for each mu
    for (unsigned int i_rh = 0; i_rh < n_resoholder; ++i_rh)
    {
        if (m_data[i_rh].bcid.size() < 1)
        {
            std::cout << "Error::align_methods::ResoHolder with empty bcid vector can't handle it" << std::endl;
            return false;
        }
        unsigned int i_mu = m_data[i_rh].mu_index;

        if (m_data[i_rh].bcid[0] > first_bc[i_mu]) first_bc[i_mu] = m_data[i_rh].bcid[0];
        if (last_bc[i_mu] == 0) last_bc[i_mu] = m_data[i_rh].bcid.back();
        else if (last_bc[i_mu] > m_data[i_rh].bcid.back()) last_bc[i_mu] = m_data[i_rh].bcid.back();
    }


    // pop out elements not valid for all
    for (unsigned int i_rh = 0; i_rh < n_resoholder; ++i_rh)
    {
        unsigned int i_mu = m_data[i_rh].mu_index;

        unsigned int first_bc_index = find_bcindex_in_vec(first_bc[i_mu], m_data[i_rh].bcid);
        unsigned int last_bc_index = find_bcindex_in_vec(last_bc[i_mu], m_data[i_rh].bcid);

        if (m_debug)
            std::cout << debug_header << "align_methods::Method " << m_data[i_rh].name << " " << m_data[i_rh].mu
                      << " : first_bc(index)=" << first_bc[i_mu] << "(" << first_bc_index << ")"
                      << " ; last_bc(index)=" << last_bc[i_mu] << "(" << last_bc_index << ")"
                      << debug_close << std::endl;

        if ( int(last_bc_index) - int(first_bc_index) < 0)
        {
            std::cout << "Error::align_methods::Negative n_bc for mu = " << m_data[i_rh].mu << "... check your numbers ... return" << std::endl;
            return false;
        }
        if (last_bc_index < m_data[i_rh].bcid.size()-1)
        {
            if (m_debug) std::cout << debug_header << "align_methods::Remove last " << m_data[i_rh].bcid.size()-last_bc_index << " els" << debug_close << std::endl;
            m_data[i_rh].bcid.erase(m_data[i_rh].bcid.begin() + last_bc_index + 1, m_data[i_rh].bcid.end());
            m_data[i_rh].prediction.erase(m_data[i_rh].prediction.begin() + last_bc_index + 1, m_data[i_rh].prediction.end());
            m_data[i_rh].reference.erase(m_data[i_rh].reference.begin() + last_bc_index + 1, m_data[i_rh].reference.end());
            m_data[i_rh].gap_to_signal.erase(m_data[i_rh].gap_to_signal.begin() + last_bc_index + 1, m_data[i_rh].gap_to_signal.end());
            m_data[i_rh].gap_to_sig_all_bcs.erase(m_data[i_rh].gap_to_sig_all_bcs.begin() + last_bc_index + 1, m_data[i_rh].gap_to_sig_all_bcs.end());
            m_data[i_rh].bunchtrain.erase(m_data[i_rh].bunchtrain.begin() + last_bc_index + 1, m_data[i_rh].bunchtrain.end());
            m_data[i_rh].resolution.erase(m_data[i_rh].resolution.begin() + last_bc_index + 1, m_data[i_rh].resolution.end());
            m_data[i_rh].reso_per_true.erase(m_data[i_rh].reso_per_true.begin() + last_bc_index + 1, m_data[i_rh].reso_per_true.end());

        }
        if (first_bc_index > 0)
        {
            if (m_debug) std::cout << debug_header << "align_methods::Remove first " << first_bc_index << " els" << debug_close << std::endl;
            m_data[i_rh].bcid.erase(m_data[i_rh].bcid.begin(), m_data[i_rh].bcid.begin() + first_bc_index);
            m_data[i_rh].prediction.erase(m_data[i_rh].prediction.begin(), m_data[i_rh].prediction.begin() + first_bc_index);
            m_data[i_rh].reference.erase(m_data[i_rh].reference.begin(), m_data[i_rh].reference.begin() + first_bc_index);
            m_data[i_rh].gap_to_signal.erase(m_data[i_rh].gap_to_signal.begin(), m_data[i_rh].gap_to_signal.begin() + first_bc_index);
            m_data[i_rh].gap_to_sig_all_bcs.erase(m_data[i_rh].gap_to_sig_all_bcs.begin(), m_data[i_rh].gap_to_sig_all_bcs.begin() + first_bc_index);
            m_data[i_rh].bunchtrain.erase(m_data[i_rh].bunchtrain.begin(), m_data[i_rh].bunchtrain.begin() + first_bc_index);
            m_data[i_rh].resolution.erase(m_data[i_rh].resolution.begin(), m_data[i_rh].resolution.begin() + first_bc_index);
            m_data[i_rh].reso_per_true.erase(m_data[i_rh].reso_per_true.begin(), m_data[i_rh].reso_per_true.begin() + first_bc_index);
        }
        if (m_debug)
        {
            std::cout << debug_header << "align_methods::post-removals method " << m_data[i_rh].name << " " << m_data[i_rh].mu
                      << " : len(bcid)=" << m_data[i_rh].bcid.size();
            if (m_data[i_rh].bcid.size() > 0) std::cout << ", first(last) el=" << m_data[i_rh].bcid[0] << "(" << m_data[i_rh].bcid.back() << ")";
            std::cout << debug_close << std::endl;
            std::cout << debug_header << "align_methods::post-removals method " << m_data[i_rh].name << " " << m_data[i_rh].mu
                      << " : len(reference)=" << m_data[i_rh].reference.size();
            if (m_data[i_rh].bcid.size() > 0) std::cout << ", first(last) el=" << m_data[i_rh].reference[0] << "(" << m_data[i_rh].reference.back() << ")";
            std::cout << debug_close << std::endl;
            std::cout << debug_header << "align_methods::post-removals method " << m_data[i_rh].name << " " << m_data[i_rh].mu
                      << " : len(prediction)=" << m_data[i_rh].prediction.size();
            if (m_data[i_rh].bcid.size() > 0) std::cout << ", first(last) el=" << m_data[i_rh].prediction[0] << "(" << m_data[i_rh].prediction.back() << ")";
            std::cout << debug_close << std::endl;
        }
    }

    // Last check : for a given mu make sure all entries are aligned
    for (unsigned int i_mu = 0; i_mu < nmu; ++i_mu)
    {
        std::vector < unsigned int > bcids;
        unsigned int ref_rh = -1;
        for (unsigned int i_rh = 0; i_rh < n_resoholder; ++i_rh)
        {
            if (m_data[i_rh].mu != mus[i_mu]) continue;
            bcids = m_data[i_rh].bcid;
            ref_rh = i_rh;
            break;
        }
        unsigned int bad_rh = 0;
        for (unsigned int i_bc = 0; i_bc < bcids.size(); ++i_bc)
        {
            for (unsigned int i_rh = 0; i_rh < n_resoholder; ++i_rh)
            {
                if (m_data[i_rh].mu != mus[i_mu]) continue;
                if (ref_rh != i_rh) continue;
                if (m_data[i_rh].bcid[i_bc] != bcids[i_bc])
                {
                    bad_rh = i_rh;
                    break;
                }
            }
            if (bad_rh) break;
        }
        if (bad_rh)
        {
            std::cout << "WARNING::align_methods::m_data bcids vectors are different for "
                      << m_data[ref_rh].name << " and " << m_data[bad_rh].name << std::endl;
        }
    }

    return true;

}






////// ---------------------------------------------------
////// make resolution
bool make_all_histograms()
{
    if (m_data.size() == 0)
    {
        std::cout << "Fatal::make_all_histograms:: No data" << std::endl;
        return false;
    }

    bool success = false;
    // first of all, reduce inputs to only bcids where all methods outputs are valid
    // miss-matching is expected to be small and due to slight difference in train/text data-sets
    // done independently for each mu (different AREUS data => different sequences)
    success = align_methods();
    if (!success)
    {
        std::cout << "Fatal::make_all_histograms::Fatal error in prior function ... check it out" << std::endl;
        return false;
    }

    // Initiate global objects
    unsigned int n_resoholder = m_data.size();
    unsigned int maxloop = 0;
    for (unsigned int i_rh = 0; i_rh < n_resoholder; ++i_rh)
    {
        if (maxloop < m_data[i_rh].bcid.size()) maxloop = m_data[i_rh].bcid.size();
    }

    std::vector < TString > method_list;
    std::vector < TString > method_label_list;
    for (unsigned int i_rh = 0; i_rh < n_resoholder; ++i_rh)
    {
        if (tstr_is_in_vec(m_data[i_rh].name, method_list)) continue;

        method_list.push_back(m_data[i_rh].name);
        method_label_list.push_back(m_data[i_rh].label);
        std::cout << "Add method :: " << m_data[i_rh].name << std::endl;
    }

    unsigned int n_method = method_list.size();
    float resMax[nrange][nmu]; // max value for each range
    float resMin[nrange][nmu]; // min value for each range

    // init histos and counts
    for (unsigned int i_er = 0; i_er < nrange; ++i_er)
    {
        for (unsigned int i_mu = 0; i_mu < nmu; ++i_mu)
        {
            resMax[i_er][i_mu] = -28000.;
            resMin[i_er][i_mu] = +28000.;
        }
    }

    // get histo ranges
    for (unsigned int i_bc = 0; i_bc < maxloop; ++i_bc)
    {
        for (unsigned int i_rh = 0; i_rh < n_resoholder; ++i_rh)
        {
            if (i_bc > m_data[i_rh].bcid.back()) continue;
            unsigned int i_mu = m_data[i_rh].mu_index;
            float reso = m_data[i_rh].resolution[i_bc];
            // int gap = m_data[i_rh].gap_to_signal[i_bc];
            float e = m_data[i_rh].reference[i_bc];

            for (unsigned int i_er = 0; i_er < nrange; ++i_er)
            {
                if (range_vals[i_er][0] > e || e > range_vals[i_er][1]) continue;
                if (resMax[i_er][i_mu] < reso) resMax[i_er][i_mu] = reso;
                if (resMin[i_er][i_mu] > reso) resMin[i_er][i_mu] = reso;
            }

        }
    }

    if (m_debug)
    {
        std::cout << std::endl << debug_header
                  << "make_all_histograms::Histo ranges -----------------------"
                  << std::endl;
        for (unsigned int i_mu = 0; i_mu < nmu; ++i_mu)
        {
            std::cout << mus[i_mu] << " :" << std::endl;
            for (unsigned int i_er = 0; i_er < nrange; ++i_er)
            {
                if (resMin[i_er][i_mu] == 0) resMin[i_er][i_mu] = -0.06;
                std::cout << "  -> " << range_labels[i_er] << " : min = " << resMin[i_er][i_mu] << "  &  max = " << resMax[i_er][i_mu] << std::endl;

            }
        }
        std::cout << debug_close << std::endl;
    }


    // start building histograms and holders
    // One HistoHolder per method making a vector<HistoHolder>
    // One vector<HistoHolder> per 2*method+2  making vector< vector<HistoHolder> >:
    // plot when given method is zero or not for all method + all bc + no method giving 0

    // first histoholder
    for (unsigned int i_z = 0; i_z < 2*n_method+2; ++i_z)
    {
        std::vector < HistoHolder > tmp_histos;

        // loop through methods
        for (unsigned int i_md = 0; i_md < n_method; ++i_md)
        {
            // holder
            HistoHolder hh;
            // name
            hh.method_name = method_list[i_md];
            // set the label to use
            hh.method_label = method_label_list[i_md];




            // trigger
            if (i_z == 0) hh.method_trigger = "None";
            else if (i_z == 2*n_method+1) hh.method_trigger = "AllNonZero";
            else if (i_z % 2 == 1) hh.method_trigger = method_list[(i_z-1)/2]+"_isZero";
            else hh.method_trigger = method_list[(i_z-1)/2]+"_isNonZero";

            std::cout << "Build HistoHolder : name=" << hh.method_name << "; trigger=" << hh.method_trigger << std::endl;

            // build histograms
            // loop through energy ranges

            // Loop throught energy values
            for (unsigned int i_er = 0; i_er < nrange; ++i_er)
            {
                // loop through mu values
                for (unsigned int i_mu = 0; i_mu < nmu; ++i_mu)
                {
                    if (range_vals[i_er][1] == 14000) {
                        hh.prediction[i_er][i_mu] = new TH1D("prediction_"+hh.method_name+"_"+hh.method_trigger+"_"+mus[i_mu]+"_"+ranges[i_er],
                                                             "prediction_"+hh.method_name+"_"+hh.method_trigger+"_"+mus[i_mu]+"_"+ranges[i_er],
                                                             1000,
                                                             range_vals[i_er][0],
                                                             MAX_ENERGY);
                    }
                    else {
                        hh.prediction[i_er][i_mu] = new TH1D("prediction_"+hh.method_name+"_"+hh.method_trigger+"_"+mus[i_mu]+"_"+ranges[i_er],
                                                             "prediction_"+hh.method_name+"_"+hh.method_trigger+"_"+mus[i_mu]+"_"+ranges[i_er],
                                                             1000,
                                                             range_vals[i_er][0],
                                                             range_vals[i_er][1]+0.01);

                    }

                    // resolution
                    // use -5/+5 to ensure finding 98pc range within the plot
                    hh.resolution_sum[i_er][i_mu] = new TH1D("resolution_sum_"+hh.method_name+"_"+hh.method_trigger+"_"+mus[i_mu]+"_"+ranges[i_er],
                                                             "resolution_sum_"+hh.method_name+"_"+hh.method_trigger+"_"+mus[i_mu]+"_"+ranges[i_er],
                                                             5000,
                                                             -5.0,
                                                             5.0);

                    hh.reso_per_true[i_er][i_mu] = new TH1D("reso_per_true"+hh.method_name+"_"+hh.method_trigger+"_"+mus[i_mu]+"_"+ranges[i_er],
                                                            "reso_per_true"+hh.method_name+"_"+hh.method_trigger+"_"+mus[i_mu]+"_"+ranges[i_er],
                                                            1000,
                                                            -1.5,
                                                            5);

                    hh.resolution_vs_bunchtrain[i_er][i_mu] = new TH2D("resolutionVSbunchtrain_"+hh.method_name+"_"+hh.method_trigger+"_"+mus[i_mu]+"_"+ranges[i_er],
                                                                       "resolutionVSbunchtrain_"+hh.method_name+"_"+hh.method_trigger+"_"+mus[i_mu]+"_"+ranges[i_er],
                                                                       (unsigned int) BUNCHTRAIN_LENGTH,
                                                                       0.,
                                                                       BUNCHTRAIN_LENGTH,
                                                                       5000,
                                                                       resMin[0][0]*1.1,
                                                                       resMax[0][0]*1.1);


                    hh.resolution_vs_bunchtrain[i_er][i_mu]->Sumw2();


                    // use custom ranges for detailed resolution plots
                    if (ranges[i_er] == "0_0p05" || ranges[i_er] == "0p0001_0p05")
                    {
                        hh.resolution[i_er][i_mu] = new TH1D("resolution_"+hh.method_name+"_"+hh.method_trigger+"_"+mus[i_mu]+"_"+ranges[i_er],
                                                             "resolution_"+hh.method_name+"_"+hh.method_trigger+"_"+mus[i_mu]+"_"+ranges[i_er],
                                                             1000,
                                                             -0.145,
                                                             0.2);
                    }
                    else
                    {
                        hh.resolution[i_er][i_mu] = new TH1D("resolution_"+hh.method_name+"_"+hh.method_trigger+"_"+mus[i_mu]+"_"+ranges[i_er],
                                                             "resolution_"+hh.method_name+"_"+hh.method_trigger+"_"+mus[i_mu]+"_"+ranges[i_er],
                                                             1000,
                                                             resMin[i_er][0]*1.5,
                                                             resMax[i_er][0]*1.5);
                    }

                    hh.prediction[i_er][i_mu]->Sumw2();
                    hh.resolution[i_er][i_mu]->Sumw2();
                    hh.resolution_sum[i_er][i_mu]->Sumw2();
                    hh.reso_per_true[i_er][i_mu]->Sumw2();

                    if (i_er == 0)
                    {
                        hh.correct_zeros[i_mu] = new TH1D(
                            "zeros_trueORfalse_"+hh.method_name+"_"+hh.method_trigger+"_"+mus[i_mu],
                            "zeros_trueORfalse_"+hh.method_name+"_"+hh.method_trigger+"_"+mus[i_mu],
                            5,
                            -0.5,
                            4.5);
                        hh.prediction_vs_e[i_mu] = new TH2D(
                            "predictionVSe_"+hh.method_name+"_"+hh.method_trigger+"_"+mus[i_mu],
                            "predictionVSe_"+hh.method_name+"_"+hh.method_trigger+"_"+mus[i_mu],
                            1000,
                            0.,
                            50.,
                            1000,
                            0.,
                            50.);
                        hh.resolution_vs_e[i_mu] = new TH2D(
                            "resolutionVSe_"+hh.method_name+"_"+hh.method_trigger+"_"+mus[i_mu],
                            "resolutionVSe_"+hh.method_name+"_"+hh.method_trigger+"_"+mus[i_mu],
                            1000,
                            0.,
                            MAX_ENERGY,
                            1000,
                            resMin[0][0]*1.5,
                            resMax[0][0]*1.5);

                        hh.resolution_vs_gap[i_mu] = new TH2D(
                            "resolutionVSgap_"+hh.method_name+"_"+hh.method_trigger+"_"+mus[i_mu],
                            "resolutionVSgap_"+hh.method_name+"_"+hh.method_trigger+"_"+mus[i_mu],
                            100,
                            0.,
                            100.,
                            250,
                            -1.0,
                            1.5);
                        // resMin[0][0]*1.5,
                        // resMax[0][0]*1.5);


                        hh.correct_zeros[i_mu]->Sumw2();
                        hh.correct_zeros[i_mu]->GetXaxis()->SetBinLabel(1, "False-0");
                        hh.correct_zeros[i_mu]->GetXaxis()->SetBinLabel(2, "Missed-0");
                        hh.correct_zeros[i_mu]->GetXaxis()->SetBinLabel(3, "True-0");
                        hh.correct_zeros[i_mu]->GetXaxis()->SetBinLabel(4, "");
                        hh.correct_zeros[i_mu]->GetXaxis()->SetBinLabel(5, "");
                        hh.prediction_vs_e[i_mu]->Sumw2();
                        hh.resolution_vs_e[i_mu]->Sumw2();
                        hh.resolution_vs_gap[i_mu]->Sumw2();

                    }
                }
            }

            // Create histograms contining resolution for gap ranges for different energy ranges
            // Created for all mu values

            for (unsigned int i_egap = 0; i_egap < neng_gaps; ++i_egap)
            {
                // loop through mu values
                for (unsigned int i_mu = 0; i_mu < nmu; ++i_mu) {
                    hh.resolution_vs_gap_erng[i_egap][i_mu] = new TH2D(
                        "resolutionVSgap_erng_"+hh.method_name+"_"+hh.method_trigger+"_"+mus[i_mu],
                        "resolutionVSgap_erng_"+hh.method_name+"_"+hh.method_trigger+"_"+mus[i_mu],
                        200,
                        0.,
                        200.,
                        1000,
                        resMin[0][0]*1.5,
                        resMax[0][0]*1.5);
                    hh.resolution_vs_gap_erng[i_egap][i_mu]->Sumw2();

                }
            }

            // Create histograms contining resolution for gap ranges
            // Created for all mu values
            if (RESO_FOR_GAPS) {
                for (unsigned int i_gap = 0; i_gap < ngaps; ++i_gap)
                {
                    // loop through mu values
                    for (unsigned int i_mu = 0; i_mu < nmu; ++i_mu)
                    {
                        hh.resolution_for_gaps[i_gap][i_mu] = new TH1D(
                            "resolution_for_gaps_"+hh.method_name+"_"+hh.method_trigger+"_"+mus[i_mu]+"_"+gap_ranges[i_gap],
                            "resolution_for_gaps"+hh.method_name+"_"+hh.method_trigger+"_"+mus[i_mu]+"_"+gap_ranges[i_gap],
                            1000,
                            -5,
                            5);

                        hh.resolution_for_gaps[i_gap][i_mu]->Sumw2();

                        // create for energy ranges for all bcs hh.resolution_for_gaps_all_bcs[nrange][ngaps][nmu]

                        for (unsigned int i_er = 0; i_er < nrange; ++i_er) {
                            hh.resolution_for_gaps_all_bcs[i_er][i_gap][i_mu] = new TH1D(
                                "resolution_for_gaps_all_bcs_"+hh.method_name+"_"+hh.method_trigger+"_"+mus[i_mu]+"_"+ranges[i_er]+"_"+gap_ranges[i_gap],
                                "resolution_for_gaps_all_bcs_"+hh.method_name+"_"+hh.method_trigger+"_"+mus[i_mu]+"_"+gap_ranges[i_gap],
                                1000,
                                resMin[i_er][0]*1.5,
                                resMax[i_er][0]*1.5);


                            hh.resolution_for_gaps_all_bcs[i_er][i_gap][i_mu]->Sumw2();
                        }
                    }
                }
            }


            tmp_histos.push_back(hh);
        }
        m_histos.push_back(tmp_histos);
    }

    if (ALGO_COMPS) {
        // then comparison holder
        for (unsigned int i_md = 0; i_md < n_method; ++i_md)
        {
            if (i_md+1 >= n_method) break;

            for (unsigned int i_md2 = i_md+1; i_md2 < n_method; ++i_md2)
            {
                CompHolder ch;

                ch.method1_name = method_list[i_md];
                ch.method2_name = method_list[i_md2];

                for (unsigned int i_er = 0; i_er < nrange; ++i_er)
                {
                    for (unsigned int i_mu = 0; i_mu < nmu; ++i_mu)
                    {
                        for (unsigned int i_zs = 0; i_zs < nzeroset; ++i_zs)
                        {
                            ch.resolution_a_vs_b[i_er][i_mu][i_zs] = new TH1D(
                                "reso_"+ch.method1_name+"_VS_"+ch.method2_name+"_"+mus[i_mu]+"_"+ranges[i_er]+"_"+zerosets[i_zs],
                                "reso_"+ch.method1_name+"_VS_"+ch.method2_name+"_"+mus[i_mu]+"_"+ranges[i_er]+"_"+zerosets[i_zs],
                                1000,
                                -1.5,
                                1.5);
                            ch.resolution_a_vs_resolution_b[i_er][i_mu][i_zs] = new TH2D(
                                "reso_"+ch.method1_name+"_VS_reso_"+ch.method2_name+"_"+mus[i_mu]+"_"+ranges[i_er]+"_"+zerosets[i_zs],
                                "reso_"+ch.method1_name+"_VS_reso_"+ch.method2_name+"_"+mus[i_mu]+"_"+ranges[i_er]+"_"+zerosets[i_zs],
                                1000,
                                resMin[i_er][0],
                                resMax[i_er][0],
                                1000,
                                resMin[i_er][0],
                                resMax[i_er][0]);
                        }

                        if (i_er == 0)
                        {
                            ch.e_a_vs_e_b[i_mu] = new TH2D(
                                "e_"+ch.method1_name+"_VS_e_"+ch.method2_name+"_"+mus[i_mu],
                                "e_"+ch.method1_name+"_VS_e_"+ch.method2_name+"_"+mus[i_mu],
                                1000,
                                0.,
                                5.,
                                1000,
                                0.,
                                5.);
                        }
                    }
                }

                m_comps.push_back(ch);
            }
        }


    }
    // Now finally run an event loop like and fill histos

    /// loop over bc
    for (unsigned int i_bc = 0; i_bc < maxloop; ++i_bc)
    {
        if (i_bc % 1000 == 0)
        {
            std::cout << "hmaker evt count : " << i_bc << "\r";
            std::flush(std::cout);
        }
        /// loop over resoholders to select the method and mu of interest
        for (unsigned int i_rh = 0; i_rh < n_resoholder; ++i_rh)
        {
            if (i_bc > m_data[i_rh].bcid.back()) continue;

            unsigned int i_mu = m_data[i_rh].mu_index;
            unsigned int i_md = tstr_index_in_vec(m_data[i_rh].name, method_list);
            if (i_md == 999999)
            {
                std::cout << "Error::make_all_histograms::i_md=999999 supposed to be default, i.e. method not found .. continue ... ??" << std::endl;
                continue;
            }

            float e = m_data[i_rh].reference[i_bc];
            float gap = m_data[i_rh].gap_to_signal[i_bc];
            float gap_for_all = m_data[i_rh].gap_to_sig_all_bcs[i_bc];
            float bt = m_data[i_rh].bunchtrain[i_bc];
            float pred = m_data[i_rh].prediction[i_bc];
            float reso = m_data[i_rh].resolution[i_bc];
            float reso_per_true = m_data[i_rh].reso_per_true[i_bc];

            if (e == 0 && reso < 0) {
                std::cout << "NEGATIVE RESO FOR ZERO E: " << e << " -> " << reso << std::endl;
            }

            if (pred == 0 && reso == 0 && e != 0) std::cout << std::endl << "WHAT ?! .... " << i_rh << " @ " << i_bc << std::endl;

            for (unsigned int i_er = 0; i_er < nrange; ++i_er)
            {
                if (range_vals[i_er][0] > e || e > range_vals[i_er][1]) continue;

                m_histos[0][i_md].prediction[i_er][i_mu]->Fill(pred);
                m_histos[0][i_md].resolution[i_er][i_mu]->Fill(reso);
                m_histos[0][i_md].resolution_sum[i_er][i_mu]->Fill(reso);
                m_histos[0][i_md].reso_per_true[i_er][i_mu]->Fill(reso_per_true);
                m_histos[0][i_md].resolution_vs_bunchtrain[i_er][i_mu]->Fill(bt, reso);


                if (i_er == 0)
                {
                    if (pred == 0 && e == 0) m_histos[0][i_md].correct_zeros[i_mu]->Fill(2);
                    else if (pred == 0) m_histos[0][i_md].correct_zeros[i_mu]->Fill(0);
                    else if (e == 0) m_histos[0][i_md].correct_zeros[i_mu]->Fill(1);
                    m_histos[0][i_md].prediction_vs_e[i_mu]->Fill(e, pred);
                    m_histos[0][i_md].resolution_vs_e[i_mu]->Fill(e, reso);


                    if (gap != 0) {
                        if (reso < 0.9) m_histos[0][i_md].resolution_vs_gap[i_mu]->Fill(gap, reso);
                    }


                }

                // loop over methods to find 0's or not in other methods at same mu and bc
                int countZeros = 0;
                for (unsigned int i_rh2 = 0; i_rh2 < n_resoholder; ++i_rh2)
                {
                    if (i_mu != m_data[i_rh2].mu_index) continue;
                    if (i_bc > m_data[i_rh2].bcid.back())
                    {
                        std::cout << "Weird::make_all_histograms::continue for method 2 but not for first ... should be aligned ... run checks" << std::endl;
                        continue;
                    }

                    unsigned int i_md2 = tstr_index_in_vec(m_data[i_rh2].name, method_list);

                    if (m_histos[2*i_md2+1][i_md].method_trigger != m_data[i_rh2].name+"_isZero")
                    {
                        std::cout << "Error::make_all_histograms:: I fucked up with method indexing in m_histos" << std::endl;
                        return false;
                    }

                    float pred2 = m_data[i_rh2].prediction[i_bc];
                    float reso2 = m_data[i_rh2].resolution[i_bc];

                    if (pred2 == 0)
                    {
                        m_histos[2*i_md2+1][i_md].prediction[i_er][i_mu]->Fill(pred);
                        m_histos[2*i_md2+1][i_md].resolution[i_er][i_mu]->Fill(reso);
                        m_histos[2*i_md2+1][i_md].resolution_sum[i_er][i_mu]->Fill(reso);
                        m_histos[2*i_md2+1][i_md].resolution_vs_bunchtrain[i_er][i_mu]->Fill(bt, reso);

                        if (i_er == 0)
                        {
                            if (pred == 0 && e == 0) m_histos[2*i_md2+1][i_md].correct_zeros[i_mu]->Fill(2);
                            else if (pred == 0) m_histos[2*i_md2+1][i_md].correct_zeros[i_mu]->Fill(0);
                            else if (e == 0) m_histos[2*i_md2+1][i_md].correct_zeros[i_mu]->Fill(1);
                            m_histos[2*i_md2+1][i_md].prediction_vs_e[i_mu]->Fill(e, pred);
                            m_histos[2*i_md2+1][i_md].resolution_vs_e[i_mu]->Fill(e, reso);
                            m_histos[2*i_md2+1][i_md].resolution_vs_gap[i_mu]->Fill(gap, reso);
                        }
                        ++countZeros;
                    }
                    else
                    {
                        m_histos[2*i_md2+2][i_md].prediction[i_er][i_mu]->Fill(pred);
                        m_histos[2*i_md2+2][i_md].resolution[i_er][i_mu]->Fill(reso);
                        m_histos[2*i_md2+2][i_md].resolution_sum[i_er][i_mu]->Fill(reso);
                        m_histos[2*i_md2+2][i_md].resolution_vs_bunchtrain[i_er][i_mu]->Fill(bt, reso);

                        if (i_er == 0)
                        {
                            if (pred == 0 && e == 0) m_histos[2*i_md2+2][i_md].correct_zeros[i_mu]->Fill(2);
                            else if (pred == 0) m_histos[2*i_md2+2][i_md].correct_zeros[i_mu]->Fill(0);
                            else if (e == 0) m_histos[2*i_md2+2][i_md].correct_zeros[i_mu]->Fill(1);
                            m_histos[2*i_md2+2][i_md].prediction_vs_e[i_mu]->Fill(e, pred);
                            m_histos[2*i_md2+2][i_md].resolution_vs_e[i_mu]->Fill(e, reso);

                            m_histos[2*i_md2+2][i_md].resolution_vs_gap[i_mu]->Fill(gap, reso);
                        }
                    }

                    if (i_md2 <= i_md) continue;

                    if (ALGO_COMPS) {
                        unsigned int i_ch = find_comp_index(m_data[i_md].name, m_data[i_md2].name, m_comps);
                        if (i_ch == 999999)
                        {
                            std::cout << "Error::make_all_histograms::i_ch=999999 supposed to be default, i.e. method not found .. continue ... ??" << std::endl;
                            continue;
                        }

                        m_comps[i_ch].resolution_a_vs_b[i_er][i_mu][0]->Fill(pred2-pred);
                        m_comps[i_ch].resolution_a_vs_resolution_b[i_er][i_mu][0]->Fill(reso2, reso);
                        if (i_er == 0)
                        {
                            m_comps[i_ch].e_a_vs_e_b[i_mu]->Fill(pred2, pred);
                        }

                        if (pred == 0)
                        {
                            m_comps[i_ch].resolution_a_vs_b[i_er][i_mu][1]->Fill(pred2-pred);
                            m_comps[i_ch].resolution_a_vs_resolution_b[i_er][i_mu][1]->Fill(reso2, reso);
                        }
                        else
                        {
                            m_comps[i_ch].resolution_a_vs_b[i_er][i_mu][2]->Fill(pred2-pred);
                            m_comps[i_ch].resolution_a_vs_resolution_b[i_er][i_mu][2]->Fill(reso2, reso);
                        }

                        if (pred2 == 0)
                        {
                            m_comps[i_ch].resolution_a_vs_b[i_er][i_mu][3]->Fill(pred2-pred);
                            m_comps[i_ch].resolution_a_vs_resolution_b[i_er][i_mu][3]->Fill(reso2, reso);
                        }
                        else
                        {
                            m_comps[i_ch].resolution_a_vs_b[i_er][i_mu][4]->Fill(pred2-pred);
                            m_comps[i_ch].resolution_a_vs_resolution_b[i_er][i_mu][4]->Fill(reso2, reso);
                        }

                        if (pred != 0 && pred2 != 0)
                        {
                            m_comps[i_ch].resolution_a_vs_b[i_er][i_mu][5]->Fill(pred2-pred);
                            m_comps[i_ch].resolution_a_vs_resolution_b[i_er][i_mu][5]->Fill(reso2, reso);
                        }
                    }

                }

                if (countZeros == 0)
                {
                    m_histos[2*n_method+1][i_md].prediction[i_er][i_mu]->Fill(pred);
                    m_histos[2*n_method+1][i_md].resolution[i_er][i_mu]->Fill(reso);
                    m_histos[2*n_method+1][i_md].resolution_sum[i_er][i_mu]->Fill(reso);
                    m_histos[2*n_method+1][i_md].resolution_vs_bunchtrain[i_er][i_mu]->Fill(bt, reso);

                    if (i_er == 0)
                    {
                        if (pred == 0 && e == 0) m_histos[2*n_method+1][i_md].correct_zeros[i_mu]->Fill(2);
                        else if (pred == 0) m_histos[2*n_method+1][i_md].correct_zeros[i_mu]->Fill(0);
                        else if (e == 0) m_histos[2*n_method+1][i_md].correct_zeros[i_mu]->Fill(1);
                        m_histos[2*n_method+1][i_md].prediction_vs_e[i_mu]->Fill(e, pred);
                        m_histos[2*n_method+1][i_md].resolution_vs_e[i_mu]->Fill(e, reso);
                        m_histos[2*n_method+1][i_md].resolution_vs_gap[i_mu]->Fill(gap, reso);
                    }
                }

            }
            if (RESO_FOR_GAPS) {
                // Loop through gap ranges
                for (unsigned int i_gap = 0; i_gap < ngaps; ++i_gap)
                {

                    for (unsigned int i_er = 0; i_er < nrange; ++i_er) {

                        if ( e >= range_vals[i_er][0]  && e < range_vals[i_er][1] && gap_for_all >= gap_vals[i_gap][0] && gap_for_all < gap_vals[i_gap][1]) {
                            // resolution_for_gaps_all_bcs[nrange][i_gap][i_mu]
                            // std::cout << "INSERT INTO gap for all " << gap_for_all<< " vals " <<  e<< " vals " << i_er<< " vals " << i_gap<< " vals " << i_mu<< " vals "<< reso << std::endl;
                            m_histos[0][i_md].resolution_for_gaps_all_bcs[i_er][i_gap][i_mu]->Fill(reso);

                        }
                    }

                    if (gap >= gap_vals[i_gap][0] && gap < gap_vals[i_gap][1]) {

                        m_histos[0][i_md].resolution_for_gaps[i_gap][i_mu]->Fill(reso);
                        //break;
                    }


                }

                // Loop through gap ranges
                for (unsigned int i_rng = 0; i_rng < neng_gaps; ++i_rng)
                {
                    if (e >= eng_gaps_range_vals[i_rng][0] && e < eng_gaps_range_vals[i_rng][1]) {

                        m_histos[0][i_md].resolution_vs_gap_erng[i_rng][i_mu]->Fill(gap, reso);
                        break;
                    }
                }

            }
        }


    }


    return true;

}



void fill_meta_data(TString name)
{

    HistoMeta hm;

    if ( name == "ofm_vs_true" )
    {
        hm.xname = "E_{T}^{ofmax} - E_{T}^{true} [GeV]";
    }
    else if ( name == "hls_vs_cpp" )
    {
        hm.xname = "E_{T}^{hls} - E_{T}^{cpp} [GeV]";
    }
    else if ( name == "hlscpp_vs_true" )
    {
        hm.xname = "E_{T}^{pred} - E_{T}^{true} [GeV]";
    }

    // reminder : {"All energies", "No deposit E=0", "0 MeV #leq E < 50 MeV", "0 MeV < E < 50 MeV", "50 MeV #leq E < 100 MeV", "100 MeV #leq E < 240 MeV", "E #geq 240 MeV"};
    hm.xranges_pred = {{0., 5.0}, {0., 0.01}, {0., 0.05}, {0., 0.05}, {0., 0.10}, {0., 0.24}, {0., 5.0}};
    hm.yranges_pred = {{0.000001, 10.}, {0.00002, 10.}, {0.00002, 5.}, {0.00002, 5.}, {0.00008, 0.7}, {0.0002, 0.1}, {0.00006, 1.}};


    hm.xranges = {{-1.5, 1.5}, {-0.06, 0.2}, {-0.25, 0.5}, {-1.0, 1.0}, {-3, 3}, {-0.35, 0.30}, {-1.0, 1.0}};
    // // for +5/-5 range (see resolution plot declaration)
    // hm.yranges = {{0.000001, 50.}, {0.00002, 10.}, {0.00002, 5.}, {0.00002, 5.}, {0.00008, 0.7}, {0.0002, 1.0}, {0.00006, 10.}};
    // for detailed resolution plots (see resolution plot declaration)
    hm.yranges = {{0.000001, 50.}, {0.00002, 10.}, {0.00002, 5.}, {0.00002, 5.}, {0.0000008, 0.7}, {0.0002, 0.1}, {0.00006, 1.}};

    hm.labelPos = {{0.2, 0.85}, {0.2, 0.85}, {0.2, 0.85}, {0.2, 0.85}, {0.2, 0.85}, {0.2, 0.85}, {0.2, 0.85}};

    hm.legendPos_mus = {{0.5, 0.5, 0.9, 0.8}, {0.5, 0.5, 0.9, 0.8}, {0.5, 0.5, 0.9, 0.8}, {0.5, 0.5, 0.9, 0.8}, {0.5, 0.5, 0.9, 0.8}, {0.5, 0.5, 0.9, 0.8}, {0.5, 0.5, 0.9, 0.8}};
    hm.legendPos_reso = {{0.6, 0.65, 0.92, 0.84}, {0.6, 0.65, 0.92, 0.84}, {0.6, 0.65, 0.92, 0.84}, {0.6, 0.65, 0.92, 0.84}, {0.6, 0.65, 0.92, 0.84}, {0.6, 0.65, 0.92, 0.84}, {0.6, 0.65, 0.92, 0.84}};

    hm.rename = pred_names_labels;//{{"sequence_ofmax_eT", "OF with MaxFinder"}, {"lstm_no_dense", "LSTM (sliding)"}, {"lstm_singlecell", "LSTM (single)"}};

    m_meta.push_back(hm);

}






void draw_plots(TString outFold)
{

    /// General variables set for the different set of plots
    TString compFold = "";
    TString hname = "";

    /// First plot simple comparisons of the resolution of all methods in one plot for all cases
    // orga : outFold/reso_comp/triggerFold/resolution_erange_mu.png/pdf/eps
    //        outFold/reso_comp/triggerFold/prediction_vs_e_mu.png/pdf/eps
    //        outFold/reso_comp/triggerFold/resolution_vs_e_mu.png/pdf/eps
    std::cout << "    -> Plot resolution comp per trigger, per mu and ranges" << std::endl;
    compFold = "/reso_comp";
    gSystem->Exec("mkdir -p "+outFold+compFold);
    for (unsigned int i_hhv = 0; i_hhv < m_histos.size(); ++i_hhv)
    {
        // if(i_hhv > 0 ) break;
        if (m_histos[i_hhv].size() < 1)
        {
            std::cout << "Error::draw_plots::m_histos number " << i_hhv << " is empty" <<std::endl;
            continue;
        }

        TString triggerFold = "/"+m_histos[i_hhv][0].method_trigger;
        gSystem->Exec("mkdir -p "+outFold+compFold+triggerFold);



        for (unsigned int i_mu = 0; i_mu < nmu; ++i_mu)
        {
            std::cout << "1D resolution comparisons -----" << std::endl;
            if (i_hhv == 0) compare_rightzeros("zeros_trueORfalse_"+mus[i_mu], outFold+compFold, 0, i_mu);

            for (unsigned int i_rg = 0; i_rg < nrange; ++i_rg)
            {
                std::cout << "resolution_"+ranges[i_rg]+"_"+mus[i_mu] << "   in    " << outFold+compFold+triggerFold << std::endl;



                // void compare_resolutions(TString name,
                //    TString outFold,
                //    int trigger_index = 0,
                //    int range_index = 0,
                //    int mu_index = 0,
                //    int meta_index = 0,
                //    bool forGaps = false)

                std::vector < TH1*> h(m_histos[i_hhv].size(), 0);

                for (unsigned int ih = 0; ih < m_histos[i_hhv].size(); ++ih)
                {
                    h[ih] = (TH1*)m_histos[i_hhv][ih].resolution[i_rg][i_mu]->Clone();
                }


                compare_resolutions("resolution_"+ranges[i_rg]+"_"+mus[i_mu],
                                    outFold+compFold+triggerFold,
                                    h,
                                    i_hhv,
                                    i_rg,
                                    i_mu,

                                    0);
                compare_predictions("prediction_"+ranges[i_rg]+"_"+mus[i_mu],
                                    outFold+compFold+triggerFold,
                                    i_hhv,
                                    i_rg,
                                    i_mu,

                                    0);

                if (i_hhv == 0) {

                    // 1d histo for reso per true
                    std::vector < TH1*> h_rpt(m_histos[i_hhv].size(), 0);

                    for (unsigned int ih = 0; ih < m_histos[i_hhv].size(); ++ih)
                    {
                        h_rpt[ih] = (TH1*)m_histos[i_hhv][ih].reso_per_true[i_rg][i_mu]->Clone();
                    }

                    // i_hhv = trigger_index = 0, i_rg = range_index = 0, i_mu = mu_index = 0, 0 = meta_index = 0
                    compare_resolutions(
                        "resolution_per_true_"+ranges[i_rg]+"_"+mus[i_mu],
                        outFold+compFold+triggerFold,
                        h_rpt,
                        i_hhv,
                        i_rg,
                        i_mu,
                        0,
                        false,
                        true);
                }


                std::vector < TH1*> summaries(m_histos[i_hhv].size(), 0);
                std::vector < TString > names(m_histos[i_hhv].size(), "");
                for (unsigned int i_hv = 0; i_hv < m_histos[i_hhv].size(); ++i_hv)
                {
                    summaries[i_hv] = (TH1*)m_histos[i_hhv][i_hv].resolution_sum[i_rg][i_mu]->Clone();

                    names[i_hv] = m_histos[i_hhv][i_hv].method_name;

                    // Replace label names
                    if (m_meta[i_hhv].rename.size() == pred_names_labels.size()) {
                        for (unsigned int rename_index = 0; rename_index < m_meta[i_hhv].rename.size(); ++rename_index)
                        {
                            if (m_meta[i_hhv].rename[rename_index][0] == m_histos[i_hhv][i_hv].method_name) {

                                names[i_hv] = m_meta[i_hhv].rename[rename_index][1];
                                break;
                            }
                        }
                    } else {
                        std::cout << "Labels to replace size is: " << m_meta[i_hhv].rename.size() << ", should be: " << pred_names_labels.size() << std::endl;
                    }

                }
                resolution_summary(summaries,
                                   names,
                                   "summary_"+ranges[i_rg]+"_"+mus[i_mu],
                                   outFold+compFold+triggerFold,
                                   range_labels[i_rg]+", "+mu_labels[i_mu]);

                for (unsigned int i_hv = 0; i_hv < summaries.size(); ++i_hv)
                {
                    summaries[i_hv]->Delete();
                    summaries[i_hv] = 0;
                }
                summaries.clear();
                names.clear();
                /*
                   std::vector < TH1*> summaries_per_true(m_histos[i_hhv].size(), 0);
                   std::vector < TString > names_per_true(m_histos[i_hhv].size(), "");
                   for (unsigned int i_hv = 0; i_hv < m_histos[i_hhv].size(); ++i_hv)
                   {
                   summaries_per_true[i_hv] = (TH1*)m_histos[i_hhv][i_hv].reso_per_true[i_rg][i_mu]->Clone();
                   names_per_true[i_hv] = m_histos[i_hhv][i_hv].method_label;
                   }
                   resolution_summary(summaries_per_true,
                                   names_per_true,
                                   "summaries_per_true_"+ranges[i_rg]+"_"+mus[i_mu],
                                    outFold+compFold+triggerFold,
                                    range_labels[i_rg]+", "+mu_labels[i_mu],
                                    "(E_{T}^{pred}-E_{T}^{true})/E_{T}^{true}");

                   for (unsigned int i_hv = 0; i_hv < summaries.size(); ++i_hv)
                   {
                   summaries_per_true[i_hv]->Delete();
                   summaries_per_true[i_hv] = 0;
                   }
                   summaries_per_true.clear();
                   names_per_true.clear();
                 */
            }
            std::cout << std::endl;


            if (i_hhv == 0) {

                if (RESO_FOR_GAPS) {
                    std::cout << "CREATING PLOTS FOR GAPS"  << std::endl;

                    for (unsigned int i_gap = 0; i_gap < ngaps; ++i_gap)
                    {
                        std::cout << "   plot: " << i_gap  << std::endl;

                        std::vector < TH1*> h_gap(m_histos[i_hhv].size(), 0);

                        for (unsigned int ih = 0; ih < m_histos[i_hhv].size(); ++ih)
                        {
                            h_gap[ih] = (TH1*)m_histos[i_hhv][ih].resolution_for_gaps[i_gap][i_mu]->Clone();
                        }

                        // i_hhv = trigger_index = 0, i_rg = range_index = 0, i_mu = mu_index = 0, 0 = meta_index = 0
                        compare_resolutions(
                            "resolution_for_gaps_"+gap_ranges[i_gap]+"_"+mus[i_mu],
                            outFold+compFold+triggerFold,
                            h_gap,
                            i_hhv,
                            i_gap,
                            i_mu,
                            0,
                            true);




                        for (unsigned int i_er = 0; i_er < nrange; ++i_er)
                        {
                            std::cout << "   plot: " << i_er  << std::endl;


                            std::vector < TH1*> h_gap_for_eranges(m_histos[i_hhv].size(), 0);
                            for (unsigned int ih = 0; ih < m_histos[i_hhv].size(); ++ih)
                            {
                                h_gap_for_eranges[ih] = (TH1*)m_histos[i_hhv][ih].resolution_for_gaps_all_bcs[i_er][i_gap][i_mu]->Clone();
                            }

                            std::cout << " i_er " << i_er << " val " << ranges[i_er] << " content " << h_gap_for_eranges[0]->GetEntries() << std::endl;
                            // i_hhv = trigger_index = 0, i_rg = range_index = 0, i_mu = mu_index = 0, 0 = meta_index = 0
                            compare_resolutions(
                                "resolution_for_gaps_per_energyrange_"+ranges[i_er]+"_"+gap_ranges[i_gap]+"_"+mus[i_mu],
                                outFold+compFold+triggerFold,
                                h_gap_for_eranges,
                                i_hhv,
                                i_er,
                                i_mu,
                                0,
                                false);


                        }
                    }
                }

            }



            std::cout << "2D prediction VS truth and resolution VS truth" << std::endl;
            std::vector < TH2* > profiles(m_histos[i_hhv].size(), 0);
            std::vector < TH2* > profiles_E(m_histos[i_hhv].size(), 0);
            std::vector < TString > pnames(m_histos[i_hhv].size(), "");
            std::vector < std::vector < TH2* >> profiles_bunchtrain(nrange, std::vector < TH2*> (m_histos[i_hhv].size(), 0));

            for (unsigned int i_md = 0; i_md < m_histos[i_hhv].size(); ++i_md)
            {
                profiles[i_md] = (TH2*)m_histos[i_hhv][i_md].resolution_vs_e[i_mu]->Clone();
                profiles_E[i_md] = (TH2*)m_histos[i_hhv][i_md].prediction_vs_e[i_mu]->Clone();
                for (unsigned int i_rg = 0; i_rg < nrange; ++i_rg) {

                    profiles_bunchtrain[i_rg][i_md] = (TH2*)m_histos[i_hhv][i_md].resolution_vs_bunchtrain[i_rg][i_mu]->Clone();
                }
                pnames[i_md] = m_histos[i_hhv][i_md].method_name;

                draw_2D_in_canvas(m_histos[i_hhv][i_md].prediction_vs_e[i_mu], m_histos[i_hhv][i_md].method_name+"_prediction_vs_e_"+mus[i_mu], outFold+compFold+triggerFold,
                                  m_histos[i_hhv][i_md].method_name, "E_{T}^{true} [GeV]", "E_{T}^{pred} [GeV]",
                                  0.001, MAX_ENERGY, 0.001, 50.,
                                  0.2, 0.85, true, true);
                draw_2D_in_canvas(m_histos[i_hhv][i_md].resolution_vs_e[i_mu], m_histos[i_hhv][i_md].method_name+"_resolution_vs_e_"+mus[i_mu], outFold+compFold+triggerFold,
                                  m_histos[i_hhv][i_md].method_name, "E_{T}^{true} [GeV]", "E_{T}^{pred} - E_{T}^{true} [GeV]",
                                  0.001, MAX_ENERGY, -1.5, 1.5,
                                  0.2, 0.85, true, false, "",
                                  true, 20);
                draw_2D_in_canvas(m_histos[i_hhv][i_md].resolution_vs_gap[i_mu], m_histos[i_hhv][i_md].method_name+"_resolution_vs_gap_"+mus[i_mu], outFold+compFold+triggerFold,
                                  m_histos[i_hhv][i_md].method_name, "Gap [BC]", "E_{T}^{pred} - E_{T}^{true} [GeV]",
                                  1, 100, -1.0, 1.5,
                                  0.2, 0.85, false, false, "",
                                  false, 2);

                /*
                   draw_2D_in_canvas(m_histos[i_hhv][i_md].resolution_vs_bunchtrain[i_mu], m_histos[i_hhv][i_md].method_name+"_resolution_vs_bunchtrain_"+mus[i_mu], outFold+compFold+triggerFold,
                                  m_histos[i_hhv][i_md].method_label, "Bunchtrain [BC]", "E_{T}^{pred} - E_{T}^{true} [GeV]",
                                  1, 80, -1.5, 1.5,
                                  0.2, 0.85, false, false);
                 */
                /*void draw_2D_in_canvas(TH2* hin, TString name, TString outFold,
                                       TString label, TString xtitle, TString ytitle,
                                       float xmin, float xmax, float ymin, float ymax,
                                       float xlabel = 0.2, float ylabel = 0.85,
                                       bool logx = true, bool logy = false)*/










                if (i_hhv == 0 and RESO_FOR_GAPS) {

                    for (unsigned int i_rng = 0; i_rng < neng_gaps; ++i_rng) {

                        draw_2D_in_canvas(m_histos[i_hhv][i_md].resolution_vs_gap_erng[i_rng][i_mu], m_histos[i_hhv][i_md].method_name+"_resolution_vs_gap_erng_"+ eng_gaps_ranges[i_rng] + "_" +mus[i_mu], outFold+compFold+triggerFold,
                                          m_histos[i_hhv][i_md].method_name, "Gap [BC]", "E_{T}^{pred} - E_{T}^{true} [GeV]",
                                          1, 100, -1.2, 2.0,
                                          0.2, 0.85, false, false, eng_gaps_range_labels[i_rng]);

                    }

                }
            }

            draw_profiles(profiles, pnames, "profiles_reso_"+mus[i_mu], outFold+compFold+triggerFold);
            draw_profiles_E(profiles_E, pnames, "profiles_E_"+mus[i_mu], outFold+compFold+triggerFold);
            draw_profiles_unc(profiles, pnames, "profiles_unc_reso_"+mus[i_mu], outFold+compFold+triggerFold);
            draw_profiles_unc_E(profiles_E, pnames, "profiles_unc_E_"+mus[i_mu], outFold+compFold+triggerFold);

            //draw_profiles(profiles_bunchtrain, pnames, "profiles_bt_"+mus[i_mu], outFold+compFold+triggerFold);
            //draw_profiles_unc(profiles_bunchtrain, pnames, "profiles_unc_bt_"+mus[i_mu], outFold+compFold+triggerFold);

            for (unsigned int i_rg = 0; i_rg < nrange; ++i_rg) {

                draw_profiles_bt(profiles_bunchtrain[i_rg], pnames, "profiles_bt_"+mus[i_mu]+"_"+ranges[i_rg],
                                 outFold+compFold+triggerFold,
                                 range_labels[i_rg],
                                 bt_y_range_vals[i_rg][0], bt_y_range_vals[i_rg][1],
                                 BUNCHTRAIN_LENGTH);
            }


            for (unsigned int i_hv = 0; i_hv < profiles.size(); ++i_hv)
            {
                profiles[i_hv]->Delete();
                profiles[i_hv] = 0;
                profiles_E[i_hv]->Delete();
                profiles_E[i_hv] = 0;
                for (unsigned int i_rng = 0; i_rng < nrange; i_rng++) {

                    profiles_bunchtrain[i_rng][i_hv]->Delete();
                    profiles_bunchtrain[i_rng][i_hv] = 0;
                }
            }
            profiles.clear();
            profiles_E.clear();
            profiles_bunchtrain.clear();
            pnames.clear();
            std::cout << std::endl;



        }
    }

    // return;


    /// Second plot CompHolder plots for resolution a to b
    // orga : outFold/algo_comp-a_b/triggerFold/resolution_aVSb_erange_mu.png/pdf/eps
    //        outFold/algo_comp-a_b/triggerFold/prediction_a_vs_prediction_b_mu.png/pdf/eps
    //        outFold/algo_comp-a_b/triggerFold/resolution_a_vs_resolution_b_erange_mu.png/pdf/eps
    // here triggerFold corresponds to nzeroset
    if (ALGO_COMPS) {
        std::cout << std::endl;
        std::cout << "    -> Plot algo comparisons (reso a-b, a VS b, ...)" << std::endl;
        for (unsigned int i_ch = 0; i_ch < m_comps.size(); ++i_ch)
        {
            compFold = "/algo_comp-"+m_comps[i_ch].method1_name+"_"+m_comps[i_ch].method2_name;
            gSystem->Exec("mkdir -p "+outFold+compFold);
            for (unsigned int i_zs = 0; i_zs < nzeroset; ++i_zs)
            {
                TString triggerFold = "/"+zerosets[i_zs];
                gSystem->Exec("mkdir -p "+outFold+compFold+triggerFold);
                for (unsigned int i_mu = 0; i_mu < nmu; ++i_mu)
                {
                    for (unsigned int i_rg = 0; i_rg < nrange; ++i_rg)
                    {
                        draw_in_canvas(m_comps[i_ch].resolution_a_vs_b[i_rg][i_mu][i_zs], "reso_aVSb_"+ranges[i_rg]+"_"+mus[i_mu], outFold+compFold+triggerFold,
                                       zeroset_labels[i_zs], "E_{T}^{"+m_comps[i_ch].method1_name+"} - E_{T}^{"+m_comps[i_ch].method2_name+"} [GeV]");

                        draw_2D_in_canvas(m_comps[i_ch].resolution_a_vs_resolution_b[i_rg][i_mu][i_zs], "resolution_a_vs_resolution_b_"+ranges[i_rg]+"_"+mus[i_mu], outFold+compFold+triggerFold,
                                          zeroset_labels[i_zs], "E_{T}^{"+m_comps[i_ch].method1_name+"} - E_{T}^{true} [GeV]", "E_{T}^{"+m_comps[i_ch].method2_name+"} - E_{T}^{true} [GeV]",
                                          -1.5, 1.5, -1.5, 1.5,
                                          0.2, 0.85, false, false);
                    }
                    if (i_zs == 0) draw_2D_in_canvas(m_comps[i_ch].e_a_vs_e_b[i_mu], "prediction_a_vs_prediction_b_"+mus[i_mu], outFold+compFold+triggerFold,
                                                     "", "E_{T}^{"+m_comps[i_ch].method1_name+"} [GeV]", "E_{T}^{"+m_comps[i_ch].method2_name+"} [GeV]",
                                                     0.001, 5., 0.001, 5.,
                                                     0.2, 0.85, true, true);
                }
            }
        }
    }



    if (nmu < 2) return;
    // Third loop over the property labelled "mu" but can be anything
    std::cout << std::endl;
    std::cout << "    -> Plot property comparison (pile-up) for each method" << std::endl;
    compFold = "/prop_comp-mu";
    gSystem->Exec("mkdir -p "+outFold+compFold);
    for (unsigned int i_hhv = 0; i_hhv < m_histos.size(); ++i_hhv)
    {
        for (unsigned int i_hv = 0; i_hv < m_histos[i_hhv].size(); ++i_hv)
        {
            std::string mtrig = std::string(m_histos[i_hhv][i_hv].method_trigger);
            if (mtrig.find(std::string(m_histos[i_hhv][i_hv].method_name)) == std::string::npos
                && mtrig != "None" ) continue;
            if (mtrig.find("isZero") != std::string::npos) continue;
            TString methodFold = "/"+m_histos[i_hhv][i_hv].method_name;
            gSystem->Exec("mkdir -p "+outFold+compFold+methodFold);

            for (unsigned int i_rg = 0; i_rg < nrange; ++i_rg)
            {
                TString plotname = "resolution_"+m_histos[i_hhv][i_hv].method_name;
                if (mtrig != "None") plotname += "_isNonZero";

                plotname += "_"+ranges[i_rg];
                compare_mus(plotname, outFold+compFold+methodFold, i_hv, i_hhv, i_rg, 0);

                std::vector < TH1*> summaries(nmu, 0);
                std::vector < TString > names(nmu, "");
                for (unsigned int i_mu = 0; i_mu < nmu; ++i_mu)
                {
                    summaries[i_mu] = (TH1*)m_histos[i_hhv][i_hv].resolution_sum[i_rg][i_mu]->Clone();
                    names[i_mu] = mu_labels[i_mu];
                }
                resolution_summary(summaries, names, "summary_"+ranges[i_rg]+"_"+m_histos[i_hhv][i_hv].method_name, outFold+compFold+methodFold, range_labels[i_rg]+", "+m_histos[i_hhv][i_hv].method_name);
                for (unsigned int i_hv = 0; i_hv < summaries.size(); ++i_hv)
                {
                    summaries[i_hv]->Delete();
                    summaries[i_hv] = 0;
                }
                summaries.clear();
                names.clear();

            }
        }
    }




}


/////---------------------------------------------------
void make_zero_ana(TString outFold, unsigned int range_index = 1)
{

    if (nrange-1 < range_index)
    {
        std::cout << "ERROR::make_zero_ana::range_index out of range ... return" << std::endl;
        return;
    }
    if (ranges[range_index] != "e0")
    {
        std::cout << "ERROR::make_zero_ana::range_index does not poin to range \"e0\" ... return" << std::endl;
        return;
    }

    std::cout << "make_zero_ana::start" << std::endl;
    gSystem->Exec("mkdir -p "+outFold+"/reso_comp");
    for (unsigned int i_mu = 0; i_mu < nmu; ++i_mu)
    {
        std::cout << "Table for mu = " << mu_labels[i_mu] << std::endl;

        std::ofstream myfile;
        myfile.open(outFold+"/reso_comp/zero_ana_"+mus[i_mu]+".txt");

        char buffer[30];

        std::string line = "";
        sprintf(buffer, "%25s", "Method");
        line += std::string(buffer)+" | ";
        sprintf(buffer, "%11s", "Pred@mean");
        line += std::string(buffer)+" | ";
        sprintf(buffer, "%11s", "Pred@max");
        line += std::string(buffer)+" | ";
        sprintf(buffer, "%13s", "85pc preds");
        line += std::string(buffer)+" | ";
        sprintf(buffer, "%13s", "95pc preds");
        line += std::string(buffer);
        myfile << line << std::endl;
        std::cout << line << std::endl;

        for (unsigned int i_hhv = 0; i_hhv < m_histos.size(); ++i_hhv)
        {
            if (m_histos[i_hhv].size() == 0) continue;
            for (unsigned int i_hv = 0; i_hv < m_histos[i_hhv].size(); ++i_hv)
            {
                if (m_histos[i_hhv][i_hv].method_trigger != "None") continue;

                TH1* hist = (TH1*)m_histos[i_hhv][i_hv].resolution[range_index][i_mu]->Clone("current");

                // Remove true = pred = 0 i.e correct zeros
                unsigned int nbins = hist->GetNbinsX();
                for (unsigned int ib = 1; ib < nbins+1; ++ib)
                {
                    if (hist->GetBinLowEdge(ib) > 0 || hist->GetBinLowEdge(ib+1) < 0) continue;
                    hist->SetBinContent(ib, 0);
                }

                double pred_at_mean = hist->GetMean();
                double maxVal = 0;
                double pred_at_max = 0;
                for (unsigned int ib = 1; ib < nbins+1; ++ib)
                {
                    if (maxVal > hist->GetBinContent(ib)) continue;
                    maxVal = hist->GetBinContent(ib);
                    pred_at_max = (hist->GetBinLowEdge(ib)+hist->GetBinLowEdge(ib+1))/2.0;
                }

                double contains_85pc = 0;
                double contains_95pc = 0;
                for (unsigned int ib = 1; ib < nbins+1; ++ib)
                {
                    double content = hist->Integral(0, ib)/hist->Integral(0, nbins+1);
                    if (contains_85pc == 0 && content >= 0.85) contains_85pc = hist->GetBinLowEdge(ib);
                    if (contains_95pc == 0 && content >= 0.95) contains_95pc = hist->GetBinLowEdge(ib);
                }


                line = "";
                sprintf(buffer, "%25s", std::string(m_histos[i_hhv][i_hv].method_name).c_str());
                line += std::string(buffer)+" | ";
                sprintf(buffer, "%11f", pred_at_mean);
                line += std::string(buffer)+" | ";
                sprintf(buffer, "%11f", pred_at_max);
                line += std::string(buffer)+" | ";
                sprintf(buffer, "%13f", contains_85pc);
                line += std::string(buffer)+" | ";
                sprintf(buffer, "%13f", contains_95pc);
                line += std::string(buffer);
                myfile << line << std::endl;
                std::cout << line << std::endl;

            }
        }
        std::cout << std::endl;
        myfile.close();
    }
    std::cout << "make_zero_ana::end" << std::endl;

}




/////---------------------------------------------------
///// all default comparisons for hls code VS cpp code
void run_hls_vs_cpp(TString outFold, std::string hls_fname = "../common_io/input_data/ofmax.txt",
                    std::string cpp_fname = "../common_io/input_data/ofmax.txt",
                    std::string true_fname = "../common_io/input_data/true.txt",
                    int cpp_delay = 0, int cpp_cut = 0, int cpp_init_bcid = 0,
                    int hls_delay = 10, int hls_cut = 0, int hls_init_bcid = 0,
                    int cpp_truth_delay = 0, int cpp_truth_cut = 0,
                    int hls_truth_delay = 0, int hls_truth_cut = 0)
{


    bool success = false;


    // run first resolution hls - cpp
    std::cout << "" << std::endl;
    std::cout << "----------------------------------" << std::endl;
    std::cout << "hls-cpp comparison" << std::endl;
    success = read_files(true_fname, cpp_fname, "cpp", "mu140", cpp_truth_delay, cpp_truth_cut, cpp_delay, cpp_cut, cpp_init_bcid);
    if (!success)
    {
        std::cout << "FATAL :: failed reading" << std::endl;
        return;
    }
    success = read_files(true_fname, hls_fname, "hls", "mu140", hls_truth_delay, hls_truth_cut, hls_delay, hls_cut, hls_init_bcid);
    if (!success)
    {
        std::cout << "FATAL :: failed reading" << std::endl;
        return;
    }
    fill_meta_data("hls_vs_cpp");


    /// get resolution histograms
    std::cout << " -> make resolution hists" << std::endl;
    success = make_all_histograms();
    if (!success)
    {
        std::cout << "FATAL :: all failed in make_all_resolutions function" << std::endl;
        return;
    }



    /// plot successes
    std::cout << " -> now ploting" << std::endl;
    draw_plots(outFold);



    // clear data
    m_data.clear();
    m_meta.clear();



}




/////---------------------------------------------------
///// all algo comparisons n(mus) should equal n(infiles)
void run_alg_comp(TString outFold, std::vector < TString > infiles = {"/home/lauri/koulu/cppm/cppm-nnlar/trainAndTest/output_201103_rdgap_full_new_hit/prediction_output_store.root"},
                  std::vector < TString > pred_names = {"sequence_ofmax_eT"}, std::vector < TString > mus = {"mu140"}, unsigned int init_bcid = 0)
{


    bool success = false;

    // run first resolution hls - cpp
    std::cout << "" << std::endl;
    std::cout << "----------------------------------" << std::endl;
    std::cout << "Run comparison for these algorithms: " << std::endl;

    std::cout << " ->";
    for (unsigned int i_a = 0; i_a < pred_names.size(); ++i_a)
    {
        std::cout << "  " << pred_names[i_a];
    }
    std::cout << std::endl;

    for (unsigned int i_f = 0; i_f < infiles.size(); ++i_f)
    {
        success = read_from_tree(infiles[i_f], pred_names, mus[i_f], init_bcid);
        if (!success)
        {
            std::cout << "FATAL :: failed reading" << std::endl;
            return;
        }
    }

    success = align_single_algo("sequence_ofmax_eT", 6, 0, 0, 0);

    fill_meta_data("ofm_vs_true");


    /// get resolution histograms
    std::cout << " -> make resolution hists" << std::endl;
    success = make_all_histograms();
    if (!success)
    {
        std::cout << "FATAL :: all failed in make_all_resolutions function" << std::endl;
        return;
    }



    /// plot successes
    std::cout << " -> now ploting" << std::endl;
    draw_plots(outFold);

    std::cout << " -> some more analysis on plot" << std::endl;
    make_zero_ana(outFold);

    // clear data
    m_data.clear();
    m_meta.clear();



}









/////---------------------------------------------------
///// main function, run the code as: root -l resolution_drawer.C+
int main(int argc, char* argv[])
{

    std::cout << "Welcome to resolution drawer ..." << std::endl;

    TString outFold = "20210429_for_CHEP";

    // // Options :
    // hls_vs_cpp options
    std::string hls_fname = "../common_io/input_data/ofmax.txt";
    std::string cpp_fname = "../common_io/input_data/ofmax.txt";
    std::string true_fname = "../common_io/input_data/true.txt";
    int cpp_delay = 0; int cpp_cut = 0; int cpp_init_bcid = 0;
    int hls_delay = 10; int hls_cut = 0; int hls_init_bcid = 0;
    int cpp_truth_delay = 0; int cpp_truth_cut = 0;
    int hls_truth_delay = 10; int hls_truth_cut = 0;


    // // alg_comp options -- Thomas' stuffs for now ... to be made configurable + requires to change mu vec global def ...

    std::vector < TString > infiles = {"/atlas/bonnet/Desktop/code/internship_CPPM/timestep.root"};

    std::vector < TString > pred_names = {"sequence_ofmax_eT",
                                          "time step 30"};
    std::vector < TString > mus = {"mu140"};

    unsigned int init_bcid = 1000000;

    bool hlsrun = false;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if ((arg == "-h") || (arg == "--help"))
        {
            show_usage(argv[0]);
            return 0;
        }
        else if ((arg == "-o") || (arg == "--outfold") || (arg == "--output"))
        {
            if (i + 1 < argc) outFold = std::string(argv[++i]);
            else std::cout << "Error::main --outout,-o,--outfold option requires one argument" << std::endl;
        }
        else if ((arg == "-i") || (arg == "--infold") || (arg == "--infile"))
        {
            if (i + 1 < argc) infiles = {argv[++i]};
            else std::cout << "Error::main --infold,-i,--infiles option requires one argument" << std::endl;
        }
        else if ((arg == "-p") || (arg == "--predictions"))
        {
            if (i + 1 < argc) {
                pred_names = {};

                std::string s = argv[++i];
                std::cout << "args: " << s << std::endl; // pred_names = {argv[++i]};
                std::string delimiter = ",";

                size_t pos = 0;
                std::string predname;
                while ((pos = s.find(delimiter)) != std::string::npos) {
                    predname = s.substr(0, pos);
                    pred_names.push_back(predname);
                    std::cout << "    push: " << predname << std::endl;
                    s.erase(0, pos + delimiter.length());

                }
                pred_names.push_back(s);
                std::cout << "    push: " << s << std::endl;
                // std::cout << "using pred_names: "<< pred_names << std::endl;



            }
            else std::cout << "Error::main -p option requires arguments" << std::endl;
        }

        else if (arg == "--max")
        {
            if (i + 1 < argc) {
                MAX_ENERGY = std::atof(std::string(argv[++i]).c_str());
            }
            else std::cout << "Error::main --max option requires one argument" << std::endl;
        }
        else if ((arg == "-n") || (arg == "--names"))
        {
            if (i + 1 < argc) {


                std::string s = argv[++i];
                std::cout << "Using labels: " << s << std::endl;
                std::string delimiter = ",";
                std::string delimiter_label = "->";

                size_t pos = 0;
                std::string predlabel;
                std::vector < TString > new_label = {};

                std::string pred_root_name;
                std::string pred_displayname;

                while ((pos = s.find(delimiter)) != std::string::npos) {

                    predlabel = s.substr(0, pos);
                    pred_root_name = predlabel.substr(0, predlabel.find(delimiter_label));
                    pred_displayname = predlabel.substr(predlabel.find(delimiter_label), pos+ delimiter_label.length());

                    new_label = { pred_root_name, pred_displayname };

                    pred_names_labels.push_back(new_label);
                    std::cout << "    push: " << new_label[0] << " -> " << new_label[1] << std::endl;
                    s.erase(0, pos + delimiter.length());

                }

                predlabel = s;
                pred_root_name = predlabel.substr(0, predlabel.find(delimiter_label));
                pred_displayname = predlabel.substr(predlabel.find(delimiter_label), s.length());
                new_label = { pred_root_name, pred_displayname };

                pred_names_labels.push_back(new_label);
                std::cout << "    push: " << new_label[0] << " -> " << new_label[1] << std::endl;
                // std::cout << "using pred_names: "<< pred_names << std::endl;



            }
            else std::cout << "Error::main -n option requires arguments" << std::endl;
        }
        else if (arg == "--mu")
        {
            if (i + 1 < argc) mu_labels[0] = "#mu = " + ((TString) std::string(argv[++i]));
            else std::cout << "Error::main --mu option requires one argument" << std::endl;
        }
        else if (arg == "--bt")
        {
            if (i + 1 < argc) BUNCHTRAIN_LENGTH = std::atof(std::string(argv[++i]).c_str());
            else std::cout << "Error::main --bt option requires one argument" << std::endl;
        }
        else if (arg == "--hlsvscpp") hlsrun = true;
        else if (arg == "--cppin")
        {
            if (i + 1 < argc) cpp_fname = std::string(argv[++i]);
            else std::cout << "Error::main --cppin option requires one argument" << std::endl;
        }
        else if (arg == "--hlsin")
        {
            if (i + 1 < argc) hls_fname = std::string(argv[++i]);
            else std::cout << "Error::main --hlsin option requires one argument" << std::endl;
        }
        else if (arg == "--cpp_delay")
        {
            if (i + 1 < argc) cpp_delay = std::atoi(argv[++i]);
            else std::cout << "Error::main --cpp_delay option requires one argument" << std::endl;
        }
        else if (arg == "--cpp_cut")
        {
            if (i + 1 < argc) cpp_cut = std::atoi(argv[++i]);
            else std::cout << "Error::main --cpp_cut option requires one argument" << std::endl;
        }
        else if (arg == "--cpp_init_bcid")
        {
            if (i + 1 < argc) cpp_init_bcid = std::atoi(argv[++i]);
            else std::cout << "Error::main --cpp_init_bcid option requires one argument" << std::endl;
        }
        else if (arg == "--cpp_truth_delay")
        {
            if (i + 1 < argc) cpp_truth_delay = std::atoi(argv[++i]);
            else std::cout << "Error::main --cpp_truthdelay option requires one argument" << std::endl;
        }
        else if (arg == "--cpp_truth_cut")
        {
            if (i + 1 < argc) cpp_truth_cut = std::atoi(argv[++i]);
            else std::cout << "Error::main --cpp_truthcut option requires one argument" << std::endl;
        }
        else if (arg == "--hls_delay")
        {
            if (i + 1 < argc) hls_delay = std::atoi(argv[++i]);
            else std::cout << "Error::main --hls_delay option requires one argument" << std::endl;
        }
        else if (arg == "--hls_cut")
        {
            if (i + 1 < argc) hls_cut = std::atoi(argv[++i]);
            else std::cout << "Error::main --hls_cut option requires one argument" << std::endl;
        }
        else if (arg == "--hls_init_bcid")
        {
            if (i + 1 < argc) hls_init_bcid = std::atoi(argv[++i]);
            else std::cout << "Error::main --hlsinit_bcid option requires one argument" << std::endl;
        }
        else if (arg == "--hls_truth_delay")
        {
            if (i + 1 < argc) hls_truth_delay = std::atoi(argv[++i]);
            else std::cout << "Error::main --hls_truth_delay option requires one argument" << std::endl;
        }
        else if (arg == "--hls_truth_cut")
        {
            if (i + 1 < argc) hls_truth_cut = std::atoi(argv[++i]);
            else std::cout << "Error::main --hls_truth_cut option requires one argument" << std::endl;
        }
        else if (arg == "--init_bcid")
        {
            if (i + 1 < argc) init_bcid = std::atoi(argv[++i]);
            else std::cout << "Error::main --init_bcid option requires one argument" << std::endl;
        }
    }


    std::cout << "Output target : " << outFold << std::endl;
    gSystem->Exec("mkdir -p "+outFold);

    if (hlsrun) run_hls_vs_cpp(outFold, hls_fname, cpp_fname, true_fname,
                               cpp_delay, cpp_cut, cpp_init_bcid,
                               hls_delay, hls_cut, hls_init_bcid,
                               cpp_truth_delay, cpp_truth_cut,
                               hls_truth_delay, hls_truth_cut);
    else run_alg_comp(outFold, infiles,  pred_names, mus, init_bcid);

    for (unsigned int i_hhv = 0; i_hhv < m_histos.size(); ++i_hhv)
    {
        for (unsigned int i_hv = 0; i_hv < m_histos[i_hhv].size(); ++i_hv)
        {
            std::cout << i_hhv << ", " << i_hv << ": "<<m_histos[i_hhv][i_hv].method_name << std::endl;

            for  (unsigned int asd = 0; asd < nmu; ++asd) {
                std::cout << "    " <<  m_histos[i_hhv][i_hv].resolution_vs_e[nmu]->GetEntries() << std::endl;
            }

        }
    }

    std::cout << "bye bye ..." << std::endl;

    return 0;
}
