#!/usr/bin/env python

"""
This script dumps NN results to a TTree that is read by the plotting script

Usage:
import nnDumper_standalone

dump_preds = nnDumper_standalonedumper(start = int(data.shape[0]*(training_set+v_set)), name = "test_out")

dump_preds.set_data(true = hit*16, data = dig*16, sig = sig*16, ofmax = np.concatenate([i['sequence_OFMax_eT'] for i in dataset]))

dump_preds.set_preds('lstm_merge', 16*model.predict([X_test,X_peak_test]).flatten())
dump_preds.set_preds('lstm_seq5', 16*old_model.predict(X_test).flatten())

dump_preds.runme()

"""
# General packages
import ROOT
import numpy as np

# Analysis in root and plotting setup
import uproot  # root
print("uproot version:", uproot.__version__)
if str(uproot.__version__)[0] == 3:
    print("update your uproot to uproot4")
#import uproot_methods.classes.TH1 as uproot_TH1


class Cfg:

    def __init__(self):
        self.asdf = None


class dumper:
    """ collect NN results and dump in root file
    """

    # , cfg=None, datacollection=None) :
    def __init__(self, start, name, threshold=0.24, bt_len=80):
        #self.cfg = config.globalconfig() if cfg is None else cfg
        # ## this root data will be writen as output

        self.name = name
        self.cfg = Cfg()
        self.start = start

        self.threshold = threshold

        self.rootData = None  # rootData(cfg=self.cfg)
        # ## data to save
        # # np.arrays of the relevant add-ons
        self.bcid = None
        self.ofmax = None
        self.edepo = None
        self.adc = None

        # add the signal and the gap to previous signal
        self.sig = None
        self.gap = None
        self.gap_for_all_bcs = None
        self.bunchtrain_position = None
        self.bt_len = bt_len

        # # dict {Id : np.array()} for NN predictions
        self.preds = {}
        # # th1 store
        self.histowner = set()

    def set_data(self, true, data, sig, ofmax):

        self.ofmax = ofmax
        self.edepo = true
        self.adc = data
        self.sig = sig

        self.bcid = np.array([i for i in range(len(self.adc))])

    def set_preds(self, name, pred):

        self.add_prediction(name, pred)

    def get_gap_to_previous(self):

        self.gap = np.zeros_like(self.sig)
        dist_to_prev = 0

        for i in range(0, len(self.edepo)):
            dist_to_prev += 1

            if self.edepo[i] > self.threshold:
                self.gap[i] = dist_to_prev
                dist_to_prev = 0

    def get_gap_to_previous_for_all_bcs(self):

        self.gap_for_all_bcs = np.zeros_like(self.sig)
        dist_to_prev = 0

        for i in range(0, len(self.edepo)):
            dist_to_prev += 1
            self.gap_for_all_bcs[i] = dist_to_prev

            if self.edepo[i] > self.threshold:
                dist_to_prev = 0

    def get_bunchtrain(self):

        self.bunchtrain_position = np.zeros_like(self.edepo)

        for i in range(0, self.edepo.shape[0], self.bt_len):
            if self.edepo.shape[0] - i < self.bt_len:
                asdf = self.edepo.shape[0] - i
                self.bunchtrain_position[i:] = np.arange(1, asdf + 1)
            else:
                self.bunchtrain_position[i:i +
                                         self.bt_len] = np.arange(1, self.bt_len + 1)

    def cleanup(self):
        for p in self.histowner:
            p.Delete()
        self.histowner.clear()

    def gather_results(self, datacollection):
        """ Write here how to build the datacollection and its NNdata
        """
        print("nnDumper::Enter gather_results")

        print("Collect general objects:")

        # create np array containing the gap to previous signal
        self.get_gap_to_previous()
        self.get_gap_to_previous_for_all_bcs()

        # assign the bunchtrain for each BC
        self.get_bunchtrain()

        print("nnDumper::Exit gather_results")
        return True

    def add_prediction(self, name, preds):
        pred = [-14000.0 for i in range(len(self.adc))]

        for i in range(len(preds)):
            pred[i + self.start] = preds[i]

        self.preds[name] = np.array(pred)

    def dump_results(self):
        print("nnDumper::Enter dump_results")

        outName = self.name  # "prediction_output_store"

        print("nnDumper::Make TTree")
        with uproot.recreate(outName + ".root") as outfile:
            """
            treedict = {"sequence_dig_eT": "float32",
                        "sequence_hit_eT": "float32",
                        "sequence_sig_eT": "float32",
                        "sequence_ofmax_eT": "float32",
                        "sequence_gap_to_signal": "float32",
                        "sequence_gap_to_signal_for_all_bcs": "float32",
                        "sequence_bunchtrain": "float32"}

            for Id in self.preds.keys():
                treedict[Id] = "float32"

            outfile["Events"] = uproot.newtree(treedict)
            """
            datadict = {"sequence_dig_eT": self.adc.astype("float32"),
                        "sequence_hit_eT": self.edepo.astype("float32"),
                        "sequence_sig_eT": self.sig.astype("float32"),
                        "sequence_gap_to_signal": self.gap.astype("float32"),
                        "sequence_ofmax_eT": self.ofmax.astype("float32"),
                        "sequence_gap_to_signal_for_all_bcs": self.gap_for_all_bcs.astype("float32"),
                        "sequence_bunchtrain": self.bunchtrain_position.astype("float32")}

            for Id in self.preds.keys():
                datadict[Id] = self.preds[Id].astype("float32")

            outfile["Events"] = datadict

        print("nnDumper::Done with tree, skipping THs")
        """
        nbins = len(self.bcid)
        xi = self.bcid[0]
        xf = self.bcid[-1]

        rf = ROOT.TFile(outName + ".root", 'update')

        hadc = ROOT.TH1F("sequence_dig_eT", "sequence_dig_eT", nbins, xi, xf)
        hadc.SetDirectory(0)
        hedepo = ROOT.TH1F("sequence_hit_eT", "sequence_hit_eT", nbins, xi, xf)
        hedepo.SetDirectory(0)
        hsig = ROOT.TH1F("sequence_sig_eT", "sequence_sig_eT", nbins, xi, xf)
        hsig.SetDirectory(0)

        hgap = ROOT.TH1F("sequence_gap_to_signal",
                         "sequence_gap_to_signal", nbins, xi, xf)
        hgap.SetDirectory(0)
        hgap_for_all = ROOT.TH1F("sequence_gap_to_signal_for_all_bcs",
                                 "sequence_gap_to_signal_for_all_bcs", nbins, xi, xf)
        hgap_for_all.SetDirectory(0)

        hseq_bt = ROOT.TH1F("sequence_bunchtrain",
                            "sequence_bunchtrain", nbins, xi, xf)
        hseq_bt.SetDirectory(0)

        hofmax = ROOT.TH1F("sequence_ofmax_eT",
                           "sequence_ofmax_eT", nbins, xi, xf)
        hofmax.SetDirectory(0)

        for ib, adc in enumerate(self.adc):
            hadc.SetBinContent(ib + 1, adc)
            hedepo.SetBinContent(ib + 1, self.edepo[ib])
            hsig.SetBinContent(ib + 1, self.sig[ib])
            hgap.SetBinContent(ib + 1, self.gap[ib])
            hgap_for_all.SetBinContent(ib + 1, self.gap_for_all_bcs[ib])
            hofmax.SetBinContent(ib + 1, self.ofmax[ib])
            hseq_bt.SetBinContent(ib + 1, self.bunchtrain_position[ib])
        release(hadc, self.histowner)
        release(hedepo, self.histowner)
        release(hsig, self.histowner)
        release(hgap, self.histowner)
        release(hgap_for_all, self.histowner)
        release(hofmax, self.histowner)
        release(hseq_bt, self.histowner)

        hadc.Write()
        hedepo.Write()
        hsig.Write()
        hgap.Write()
        hgap_for_all.Write()
        hofmax.Write()
        hseq_bt.Write()

        for Id in self.preds.keys():
            hpred = ROOT.TH1F("sequence_" + Id + "_eT",
                              "sequence_" + Id + "_eT", nbins, xi, xf)
            hpred.SetDirectory(0)
            for ib, pred in enumerate(self.preds[Id]):
                hpred.SetBinContent(ib + 1, pred)
                release(hpred, self.histowner)
            hpred.Write()

        rf.Close()
        """
        print("nnDumper::Exit dump_results")

    def runme(self, datacollection=None):

        self.gather_results(self.rootData)

        self.dump_results()

        self.cleanup()


#pointers_in_the_wild = set()

def release(obj, container=None):
    """ Tell python that no, we don't want to lose this one when current function returns """
    global pointers_in_the_wild
    if container is None:
        pointers_in_the_wild.add(obj)
    else:
        container.add(obj)
    ROOT.SetOwnership(obj, False)
    return obj
