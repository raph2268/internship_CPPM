

///// ---------------------------------------------------
///// General functions

void AREUSText(Double_t x, Double_t y, const char *text, Color_t color = 1, double tsize = 0, double space = 0.12) {
    //Double_t tsize=18;
    TLatex l; //l.SetTextAlign(12);
    if (tsize > 0) l.SetTextSize(tsize);
    l.SetNDC();
    //l.SetTextFont(62);
    l.SetTextFont(72);
    l.SetTextColor(color);
    l.DrawLatex(x, y, "AREUS");
    l.SetTextFont(42);
    l.DrawLatex(x+space, y, text);
}

void myText(Double_t x, Double_t y, const char *text, Color_t color = 1, double tsize = 0) {
    //Double_t tsize=18;
    TLatex l; //l.SetTextAlign(12);
    if (tsize > 0) l.SetTextSize(tsize);
    l.SetNDC();
    l.SetTextFont(42);
    l.SetTextColor(color);
    l.DrawLatex(x, y, text);
}

static void show_usage(std::string name)
{
    std::cerr << "Usage: " << name << " <option(s)>"
              << "Options:\n"
              << "\t-h,--help\tShow this help message\n"
              << "\t--hlsvscpp\tRun in hls VS cpp mode\n"
              << "\n"
              << "hls VS cpp options (generic for 2 methods with same ref):\n"
              << "\t--ein FILEPATH\t\ttrue energie input file\n"
              << "\t--hlsin FILEPATH\thls prediction input file\n"
              << "\t--cppin FILEPATH\tcpp prediction input file\n"
              << "\t--cpp_delay N\t\tRemove N many entries at the start of the cpp file\n"
              << "\t--cpp_cut N\t\tRemove N many entries at the end of the cpp file\n"
              << "\t--cpp_init_bcid N\tGives to first entry after delay in cpp file bcid=N\n"
              << "\t--hls_delay N\t\tRemove N many entries at the start of the hls file\n"
              << "\t--hls_cut N\t\tRemove N many entries at the end of the hls file\n"
              << "\t--hls_init_bcid N\tGives to first entry after delay in hls file bcid=N\n"
              << "\t--cpp_truth_delay N\tRemove N many entries at the start of the true e file for cpp\n"
              << "\t--cpp_truth_cut N\tRemove N many entries at the end of the true e file for cpp\n"
              << "\t--hls_truth_delay N\tRemove N many entries at the start of the true e file for hls\n"
              << "\t--hls_truth_cut N\tRemove N many entries at the end of the true e file for hls\n"
              << std::endl;
    std::cerr << "Compilation :\n"
              << "g++ resolution_drawer.C -o resdrawer -lm -g -Wall -pthread -m64 -I/home/tcalvet/configs/root/v6.22.02-x86_64-gcc9.3_built/include -L/home/tcalvet/configs/root/v6.22.02-x86_64-gcc9.3_built/lib -lGui -lCore -lImt -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lROOTVecOps -lTree -lTreePlayer -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread -lMultiProc -lROOTDataFrame -pthread -lm -ldl -rdynamic\n"
              <<std::endl;
}



///// ---------------------------------------------------
///// helper functions

unsigned int find_mu_index(TString str, unsigned int nmu, TString *mus)
{
    for (unsigned int i_mu = 0; i_mu < nmu; ++i_mu)
    {
        if (str == mus[i_mu]) return i_mu;
    }
    return 999999;
}


unsigned int find_bcindex_in_vec(unsigned int bcid, std::vector < unsigned int > avector)
{
    for (unsigned int i = 0; i < avector.size(); ++i)
    {
        if (bcid == avector[i]) return i;
    }
    return 999999;
}


unsigned int tstr_index_in_vec(TString str, std::vector < TString > avector)
{
    for (unsigned int i = 0; i < avector.size(); ++i)
    {
        if (str == avector[i]) return i;
    }
    return 999999;
}


bool tstr_is_in_vec(TString str, std::vector < TString > avector)
{
    for (unsigned int i = 0; i < avector.size(); ++i)
    {
        if (str == avector[i]) return true;
    }
    return false;
}






///// ---------------------------------------------------
///// helper functions - 2

unsigned int find_comp_index(TString md1, TString md2, std::vector < CompHolder > m_comps)
{
    for (unsigned int i_ch = 0; i_ch < m_comps.size(); ++i_ch)
    {
        if (md1 != m_comps[i_ch].method1_name) continue;
        if (md2 != m_comps[i_ch].method2_name) continue;
        return i_ch;
    }
    return 999999;
}

void remove_resoholder_bcid(ResoHolder& rh, unsigned int index)
{
    if (rh.bcid.size() > 0) rh.bcid.erase(rh.bcid.begin()+index);
    if (rh.bcid.size() > 0) rh.reference.erase(rh.reference.begin()+index);
    if (rh.bcid.size() > 0) rh.prediction.erase(rh.prediction.begin()+index);
    if (rh.bcid.size() > 0) rh.resolution.erase(rh.resolution.begin()+index);
    if (rh.bcid.size() > 0) rh.reso_per_true.erase(rh.reso_per_true.begin()+index);
}





////// ---------------------------------------------------
////// data collector functions

// fill energy at each BC vector from single txt file
bool fill_vec_from_txtfile(std::string fname, std::vector < float >& e_vs_bc)
{

    std::ifstream inFile;
    inFile.open(fname);
    if (!inFile.is_open())
    {
        std::cout << "Error :: can not open " << fname << std::endl;
        return false;
    }

    e_vs_bc.clear();
    float e;
    while (inFile >> e)
    {
        e_vs_bc.push_back(e);
    }
    inFile.close();

    //std::cout << "??g " << e_vs_bc.size() << std::endl;
    return true;
}
