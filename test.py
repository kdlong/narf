import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--nThreads", type=int, help="number of threads", default=None)
parser.add_argument("--useBoost", action='store_true', help="user boost histograms")
parser.add_argument("-n", "--maxfiles", type=int, default=-1, help="Max number of files to process per dataset")
args = parser.parse_args()

import ROOT
ROOT.gInterpreter.ProcessLine(".O3")
if args.nThreads is not None:
    if args.nThreads > 1:
        ROOT.ROOT.EnableImplicitMT(args.nThreads)
else:
    ROOT.ROOT.EnableImplicitMT()

#ROOT.TTreeProcessorMT.SetTasksPerWorkerHint(1)

import pickle
import gzip


import narf
from datasets import datasets2016
import hist
import lz4.frame
import numba
from narf import numbaIncludes
ROOT.gInterpreter.Declare('#include "narf/include/csframe.h"')

datasets = datasets2016.allDatasets(args.maxfiles)[2:3]

boost_hist_default = ROOT.boost.histogram.use_default
boost_hist_options_none = ROOT.boost.histogram.axis.option.none_t

# standard regular axes

axis_pt = hist.axis.Regular(29, 26., 55., name = "pt")
axis_eta = hist.axis.Regular(48, -2.4, 2.4, name = "eta")

# categorical axes in python bindings always have an overflow bin, so use a regular
# axis for the charge
axis_charge = hist.axis.Regular(2, -2., 2., underflow=False, overflow=False, name = "charge")

# integer axis with no overflow
axis_pdf_idx = hist.axis.Integer(0, 103, underflow=False, overflow=False, name = "pdfidx")

#@numba.njit
#def arrsq(arr):
    #return arr*arr

#@ROOT.Numba.Declare(["RVec<float>"], "RVec<float>")
#def weightsq(w):
    #return arrsq(w)

def build_graph(df, dataset):
    results = []

    if dataset.is_data:
        df = df.Define("weight", "1.0")
    else:
        df = df.Define("weight", "std::copysign(1.0, genWeight)")

    weightsum = df.SumAndCount("weight")

    df = df.Define("vetoMuons", "Muon_pt > 10 && Muon_looseId && abs(Muon_eta) < 2.4 && abs(Muon_dxybs) < 0.05")

    df = df.Filter("Sum(vetoMuons) == 1")

    df = df.Define("goodMuons", "vetoMuons && Muon_mediumId")

    df = df.Filter("Sum(goodMuons) == 1")

    df = df.Define("goodMuons_Pt0", "Muon_pt[goodMuons][0]")
    df = df.Define("goodMuons_Eta0", "Muon_eta[goodMuons][0]")
    df = df.Define("goodMuons_Charge0", "Muon_charge[goodMuons][0]")

    #df = df.Filter("nMuon > 0")
    #df = df.Define("goodMuons_Pt0", "Muon_pt[0]")
    #df = df.Define("goodMuons_Eta0", "Muon_eta[0]")
    #df = df.Define("goodMuons_Charge0", "Muon_charge[0]")


    if args.useBoost:
        #hptetacharge = df.HistoBoost("hptetacharge", [axis_pt, axis_eta, axis_charge], ["goodMuons_Pt0", "goodMuons_Eta0", "goodMuons_Charge0", "weight"])
        hptetacharge = df.HistoBoost("hptetacharge", [axis_pt, axis_eta, axis_charge], ["goodMuons_Pt0", "goodMuons_Eta0", "goodMuons_Charge0", "weight"], var_axis_names = ["pdfidx"])
        #hptetacharge = df.HistoBoost("hptetacharge", [axis_pt], ["goodMuons_Pt0", "weight"])
    else:
        #hptetacharge = df.Histo3D(("hptetacharge", "", 29, 26., 55., 48, -2.4, 2.4, 2, -2., 2.), "goodMuons_Pt0", "goodMuons_Eta0", "goodMuons_Charge0", "weight")
        hptetacharge = df.Histo3DWithBoost(("hptetacharge", "", 29, 26., 55., 48, -2.4, 2.4, 2, -2., 2.), "goodMuons_Pt0", "goodMuons_Eta0", "goodMuons_Charge0", "weight")
        #hptetacharge = df.Histo1DWithBoost(("hptetacharge", "", 29, 26., 55.), "goodMuons_Pt0", "weight")
    results.append(hptetacharge)

    if "Wplus" in dataset.name:
        df = df.Define("genPrefsrLeps", "Numba::prefsrLeptons(GenPart_status, GenPart_statusFlags, GenPart_pdgId, GenPart_genPartIdxMother, GenPart_pt)")
        df = df.Define("genPlusIdx", "GenPart_pdgId[genPrefsrLeps][0] == -13 || GenPart_pdgId[genPrefsrLeps][0] == -14 ? 0 : 1")
        df = df.Define("genlp", "ROOT::Math::PtEtaPhiMVector(GenPart_pt[genPrefsrLeps][genPlusIdx], GenPart_eta[genPrefsrLeps][genPlusIdx], GenPart_phi[genPrefsrLeps][genPlusIdx], GenPart_mass[genPrefsrLeps][genPlusIdx])")
        df = df.Define("genlm", "ROOT::Math::PtEtaPhiMVector(GenPart_pt[genPrefsrLeps][!genPlusIdx], GenPart_eta[genPrefsrLeps][!genPlusIdx], GenPart_phi[genPrefsrLeps][!genPlusIdx], GenPart_mass[genPrefsrLeps][!genPlusIdx])")
        df = df.Define("genV", "ROOT::Math::PxPyPzEVector(genlp)+ROOT::Math::PxPyPzEVector(genlm)")
        df = df.Define("csSineCosThetaPhi", "csSineCosThetaPhi(genlp, genlm)")
        df = df.Define("ptVgen", "genV.pt()")
        df = df.Define("yVgen", "genV.Rapidity()")
        df = df.Define("mVgen", "genV.mass()")
        df = df.Define("minnloQCDWeightsByHelicity", "Numba::qcdUncByHelicity(ptVgen, yVgen, csSineCosThetaPhi[0], csSineCosThetaPhi[1], csSineCosThetaPhi[2], csSineCosThetaPhi[3])*weight")
        df = df.DefinePerSample("helicityScaleIndex", "std::array<int, 54> res; std::iota(res.begin(), res.end(), 0); return res;")
        df = df.DefinePerSample("scaleIndex", "std::array<int, 9> res; std::iota(res.begin(), res.end(), 0); return res;")

        if args.useBoost:
            axis_weights = hist.axis.Regular(100, 0.5, 1.5, name = "weights")
            df = df.Define("helicityWeights", "Eigen::TensorFixedSize<double, Eigen::Sizes<54>> res; auto w = minnloQCDWeightsByHelicity; std::copy(std::begin(w), std::end(w), res.data()); return res;")
            hPtEtaWeights = df.HistoBoost("hPtEtaHelcity", [axis_eta, axis_pt], ["goodMuons_Eta0", "goodMuons_Pt0", "helicityWeights"], var_axis_names=["systIdx"])
            df = df.Define("scaleWeights", "Eigen::TensorFixedSize<double, Eigen::Sizes<9>> res; auto w = weight*LHEScaleWeight; std::copy(std::begin(w), std::end(w), res.data()); return res;")
            hPtEtaScaleWeights = df.HistoBoost("hPtEtaScaleWeight", [axis_eta, axis_pt,], ["goodMuons_Eta0", "goodMuons_Pt0", "scaleWeights"], var_axis_names=["systIdx"])
            hWeights = df.HistoBoost("hHelWeights", [axis_weights], ["minnloQCDWeightsByHelicity"])
            hWeights2D = df.HistoBoost("hHelWeights2D", [axis_weights], ["weight", "helicityWeights"])
        else:
            hGenVPt = df.Histo1D(("hgenVpt", "", 29, 26, 55), "ptVgen", "weight")
            hGenVtheta = df.Histo1D(("hgenVtheta", "", 10, -1.6, 1.6), "thetaVgen", "weight")
            hGenVmass = df.Histo1D(("hgenVmass", "", 20, 71,101), "mVgen", "weight")
            hPtEtaWeights = df.Histo3D(("hPtEtaHelcity", "", 48, -2.4, 2.4, 29, 26, 55, 54, -0.5, 53.5), "goodMuons_Eta0", "goodMuons_Pt0", "helicityScaleIndex", "minnloQCDWeightsByHelicity")
        
        #df.Snapshot("Events", "dump.root", ["ptVgen", "yVgen", "minnloQCDWeightsByHelicity", "goodMuons_Eta0", "goodMuons_Pt0", "helicityScaleIndex"])
        results.extend([hPtEtaScaleWeights, hPtEtaWeights, hWeights, hWeights2D])

    #if not dataset.is_data:

        #df = df.Define("pdfweight", "weight*LHEPdfWeight")

        #df = df.DefinePerSample("pdfidx", "std::array<int, 103> res; std::iota(res.begin(), res.end(), 0); return res;")

        #hptetachargepdf = df.HistoBoost("hptetachargepdf", [axis_pt, axis_eta, axis_charge, axis_pdf_idx], ["goodMuons_Pt0", "goodMuons_Eta0", "goodMuons_Charge0", "pdfidx", "pdfweight"])
        #results.append(hptetachargepdf)

        #df = df.DefinePerSample("pdfidx", "std::array<int, 103> res; std::iota(res.begin(), res.end(), 0); return res;")

    return results, weightsum

resultdict = narf.build_and_run(datasets, build_graph)

fname = "test.pkl.lz4"

print("writing output")
#with gzip.open(fname, "wb") as f:
with lz4.frame.open(fname, "wb") as f:
    pickle.dump(resultdict, f)
