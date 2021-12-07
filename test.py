import ROOT
ROOT.gInterpreter.ProcessLine(".O3")
ROOT.ROOT.EnableImplicitMT(24)

ROOT.gInterpreter.ProcessLine('#include "narf/histHelpers.cc"')

import narf
from datasets import datasets2016

#import gc

datasets = datasets2016.allDatasets()
#datasets = [zmc, data]
#datasets = [zmc]

def build_graph(df, dataset):
    if dataset.is_data:
        df = df.Define("weight", "1.0")
    else:
        df = df.Define("weight", "std::copysign(1.0, genWeight)")

    df = df.Define("one", "1.0")
    hweight = df.Histo1D(("sumweights", "", 1, 0.5, 1.5), "one", "weight")


    df = df.Filter("HLT_IsoTkMu24 || HLT_IsoMu24")
    df = df.Define("vetoMuons", "Muon_pt > 10 && Muon_looseId && abs(Muon_eta) < 2.4 && abs(Muon_dxybs) < 0.05 && abs(Muon_dz)< 0.2")
    df = df.Define("vetoElectrons", "Electron_pt > 10 && Electron_cutBased > 0 && abs(Electron_eta) < 2.4 && abs(Electron_dxy) < 0.05 && abs(Electron_dz)< 0.2")
    df = df.Define("goodMuons", "vetoMuons && Muon_pt > 26 && Muon_mediumId > 0")
    #df.Define("goodTrigObjs", "goodMuonTriggerCandidate(TrigObj_id,TrigObj_pt,TrigObj_l1pt,TrigObj_l2pt,TrigObj_filterBits)")

    df = df.Filter("Sum(vetoMuons) == 2 && Sum(goodMuons) == 2")


    df = df.Define("Muon0_pt", "Muon_pt[0]")
    df = df.Define("Muon0_eta", "Muon_eta[0]")

    hpt = df.Histo1D(("hpt", "", 35, 25., 60.), "Muon0_pt", "weight")
    heta = df.Histo1D(("heta", "", 48, -2.4, 2.4), "Muon0_eta", "weight")

    results = [hpt, heta]
    if not dataset.is_data:
        df = df.Define("Muon0_ptvarsUp", "Numba::calibratedPt(Muon_pt[0], Muon_eta[0], Muon_charge[0], true)")
    #    #df = df.Define("Muon0_ptvarsDown", "Numba::calibratedPt(Muon_pt[0], Muon_eta[0], Muon_charge[0], false)")
        df = df.Define("ptvarIndices", "indices(Muon0_ptvarsUp)")
        hptvar = df.Histo2D(("hptvarUp", "", 35, 25., 60., 288, -0.5, 287.5), "Muon0_pt", "ptvarIndices", "weight")
    #    #hptvar = df.Histo2D(("hptvarDown", "", 35, 25., 60., 288, -0.5, 287.5), "Muon0_ptvarsDown", "ptvarIndices", "weight")
        results.append(hptvar)

    return results, hweight

narf.build_and_run(datasets, build_graph, "testout.root")
