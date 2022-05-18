


////// Input data for each method at each mu (expected input format)
struct ResoHolder
{

    TString name;
    TString label;
    TString mu;
    unsigned int mu_index;

    std::vector < unsigned int > bcid;
    std::vector < float > reference;
    std::vector < float > prediction;
    std::vector < unsigned int > gap_to_signal;
    std::vector < unsigned int > gap_to_sig_all_bcs;
    std::vector < unsigned int > bunchtrain;
    std::vector < float > resolution;
    std::vector < float > reso_per_true;

};

////// Resolution histograms - one set per method


struct HistoHolder
{

    TString method_name; // Identify which method is plotted
    TString method_label; // Use label other than method_name
    TString method_trigger; // Identify for which method pred=0 was excluded or required

    TH1* prediction[nrange][nmu];
    TH1* resolution[nrange][nmu];
    TH1* reso_per_true[nrange][nmu];
    TH1* resolution_sum[nrange][nmu];
    TH2* prediction_vs_e[nmu];
    TH2* resolution_vs_e[nmu];
    TH2* resolution_vs_gap[nmu]; // create resolution plots for gap
    TH2* resolution_vs_gap_erng[neng_gaps][nmu]; // create resolution plots for gap with energy range
    TH1* resolution_for_gaps[ngaps][nmu]; // create resolution plots for gap ranges
    TH1* resolution_for_gaps_all_bcs[nrange][ngaps][nmu]; // create resolution plots for all bcs with energy ranges
    TH2* resolution_vs_bunchtrain[nrange][nmu];     // create bunchtrain plot
    TH1* correct_zeros[nmu];

};


////// Method comparator
struct CompHolder
{

    TString method1_name; // ref (a)
    TString method2_name; // alt (b)

    TH1* resolution_a_vs_b[nrange][nmu][nzeroset]; // a is reference
    TH2* e_a_vs_e_b[nmu]; // a is x-axis
    TH2* resolution_a_vs_resolution_b[nrange][nmu][nzeroset]; // a is x-axis

};


////// ---------------------------------------------------
////// Plot informations
struct HistoMeta
{

    TString xname;

    std::vector < std::vector < float>> xranges_pred;
    std::vector < std::vector < float>> yranges_pred;

    std::vector < std::vector < float>> xranges;
    std::vector < std::vector < float>> yranges;

    std::vector < std::vector < float>> labelPos;

    std::vector < std::vector < float>> legendPos_mus;
    std::vector < std::vector < float>> legendPos_reso;

    std::vector < std::vector < TString>> rename;

};
