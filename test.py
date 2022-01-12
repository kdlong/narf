import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--nThreads", type=int, help="number of threads", default=None)
parser.add_argument("--useBoost", action='store_true', help="user boost histograms")
parser.add_argument("--test", action='store_true', help="Only run over one file per dataset")
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

datasets = datasets2016.allDatasets(args.test)[2:3]

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
        df = df.Define("genl1", "ROOT::Math::PtEtaPhiMVector(GenPart_pt[genPrefsrLeps][0], GenPart_eta[genPrefsrLeps][0], GenPart_phi[genPrefsrLeps][0], GenPart_mass[genPrefsrLeps][0])")
        df = df.Define("genl2", "ROOT::Math::PtEtaPhiMVector(GenPart_pt[genPrefsrLeps][1], GenPart_eta[genPrefsrLeps][1], GenPart_phi[genPrefsrLeps][1], GenPart_mass[genPrefsrLeps][1])")
        df = df.Define("genV", "ROOT::Math::PxPyPzMVector(genl1)+ROOT::Math::PxPyPzMVector(genl2)")
        df = df.Define("ptVgen", "genV.pt()")
        df = df.Define("yVgen", "genV.Rapidity()")
        df = df.Define("mVgen", "genV.mass()")
        df = df.Define("phiVgen", "genV.phi()")
        df = df.Define("thetaVgen", "genV.theta()")
        df = df.Define("minnloQCDWeightsByHelicity", "Numba::qcdUncByHelicity(ptVgen, yVgen, genV.theta(), genV.phi())*weight")
        df = df.DefinePerSample("helicityScaleIndex", "std::array<int, 54> res; std::iota(res.begin(), res.end(), 0); return res;")
        df = df.Define("etas", "std::array<float, 54> res; res.fill(goodMuons_Eta0); return res;")
        df = df.Define("pts", "std::array<float, 54> res; res.fill(goodMuons_Pt0); return res;")

        if args.useBoost:
            axis_ptV = hist.axis.Regular(50, 0., 50., name = "ptV")
            axis_phi = hist.axis.Regular(10, 0, 7, name = "phi")
            axis_theta = hist.axis.Regular(10, -1.6, 1.6, name = "theta")
            axis_mass = hist.axis.Regular(20, 71, 1.6, name = "mass")
            axis_weight = hist.axis.Regular(10, 0.5, 1.5, name = "weights")
            axis_weightIndices = hist.axis.Regular(54, -0.5, 53.5, name = "weightIndices")
            hGenVPt = df.HistoBoost("hgenVpt", [axis_ptV], ["ptVgen", "weight"])
            hGenVPhi = df.HistoBoost("hgenVphi", [axis_phi], ["phiVgen", "weight"])
            hGenVtheta = df.HistoBoost("hgenVtheta", [axis_theta], ["thetaVgen", "weight"])
            hGenVmass = df.HistoBoost("hgenVmass", [axis_mass], ["mVgen", "weight"])
            hHelicityWeights = df.HistoBoost("hHelicityWeights", [axis_weight], ["minnloQCDWeightsByHelicity"])
            hPtEtaWeights = df.HistoBoost("hPtEtaHelcity", [axis_eta, axis_pt, axis_weightIndices], ["goodMuons_Eta0", "goodMuons_Pt0", "helicityScaleIndex", "minnloQCDWeightsByHelicity"])
            #hPtEtaWeights = df.HistoBoost("hPtEtaHelcity", [axis_eta, axis_pt, axis_weightIndices], ["etas", "pts", "helicityScaleIndex", "minnloQCDWeightsByHelicity"])
        else:
            hGenVPt = df.Histo1D(("hgenVpt", "", 29, 26, 55), "ptVgen", "weight")
            hGenVPhi = df.Histo1D(("hgenVphi", "", 14, 0, 7), "phiVgen", "weight")
            hGenVtheta = df.Histo1D(("hgenVtheta", "", 10, -1.6, 1.6), "thetaVgen", "weight")
            hGenVmass = df.Histo1D(("hgenVmass", "", 20, 71,101), "mVgen", "weight")
            hHelicityWeights = df.Histo1D(("hHelicityWeights", "", 50, 0.5,1.5), "minnloQCDWeightsByHelicity")
            hPtEtaWeights = df.Histo3D(("hPtEtaHelcity", "", 48, -2.4, 2.4, 29, 26, 55, 54, -0.5, 53.5), "goodMuons_Eta0", "goodMuons_Pt0", "helicityScaleIndex", "minnloQCDWeightsByHelicity")
        
        #df.Snapshot("Events", "dump.root", ["ptVgen", "yVgen", "phiVgen", "thetaVgen", "minnloQCDWeightsByHelicity", "goodMuons_Eta0", "goodMuons_Pt0", "helicityScaleIndex"])
        results.extend([hGenVPt, hGenVPhi, hGenVtheta, hGenVmass, hHelicityWeights, hPtEtaWeights])

    if not dataset.is_data:

        #df = df.Define("pdfweight", "weight*LHEPdfWeight")

        #df = df.DefinePerSample("pdfidx", "std::array<int, 103> res; std::iota(res.begin(), res.end(), 0); return res;")

        #hptetachargepdf = df.HistoBoost("hptetachargepdf", [axis_pt, axis_eta, axis_charge, axis_pdf_idx], ["goodMuons_Pt0", "goodMuons_Eta0", "goodMuons_Charge0", "pdfidx", "pdfweight"])
        #results.append(hptetachargepdf)

        #df = df.DefinePerSample("pdfidx", "std::array<int, 103> res; std::iota(res.begin(), res.end(), 0); return res;")


        for i in range(10):

            wname = f"pdfweight_{i}"

            #df = df.Define(wname, "weight*LHEPdfWeight")


            df = df.Define(wname, "Eigen::TensorFixedSize<double, Eigen::Sizes<103>> res; auto w = weight*LHEPdfWeight; std::copy(std::begin(w), std::end(w), res.data()); return res;")

            if args.useBoost:
                #hptetachargepdf = df.HistoBoost(f"hptetachargepdf_{i}", [axis_pt, axis_eta, axis_charge, axis_pdf_idx], ["goodMuons_Pt0", "goodMuons_Eta0", "goodMuons_Charge0", "pdfidx", wname])
                #hptetachargepdf = df.HistoBoost(f"hptetachargepdf_{i}", [axis_pt, axis_eta, axis_charge, axis_pdf_idx], ["goodMuons_Pt0", "goodMuons_Eta0", "goodMuons_Charge0", "pdfidx", wname], storage=hist.storage.Double())
                #hptetachargepdf = df.HistoBoostArr(f"hptetachargepdf_{i}", [axis_pt, axis_eta, axis_charge], ["goodMuons_Pt0", "goodMuons_Eta0", "goodMuons_Charge0", wname])
                hptetachargepdf = df.HistoBoost(f"hptetachargepdf_{i}", [axis_pt, axis_eta, axis_charge], ["goodMuons_Pt0", "goodMuons_Eta0", "goodMuons_Charge0", wname])
                #hptetachargepdf = df.HistoBoost(f"hptetachargepdf_{i}", [axis_pt, axis_eta, axis_charge], ["goodMuons_Pt0", "goodMuons_Eta0", "goodMuons_Charge0", wname], storage=hist.storage.Double())
            else:
                #hptetachargepdf = df.HistoND((f"hptetachargepdf_{i}", "", 4, [29, 48, 2, 103], [26., -2.4, -2., -0.5], [55., 2.4, 2., 102.5]), ["goodMuons_Pt0", "goodMuons_Eta0", "goodMuons_Charge0", "pdfidx", f"pdfweight_{i}"])
                #hptetachargepdf = df.HistoNDWithBoost((f"hptetachargepdf_{i}", "", 4, [29, 48, 2, 103], [26., -2.4, -2., -0.5], [55., 2.4, 2., 102.5]), ["goodMuons_Pt0", "goodMuons_Eta0", "goodMuons_Charge0", "pdfidx", wname])
                hptetachargepdf = df.HistoNDWithBoost((f"hptetachargepdf_{i}", "", 3, [29, 48, 2], [26., -2.4, -2.], [55., 2.4, 2.]), ["goodMuons_Pt0", "goodMuons_Eta0", "goodMuons_Charge0", wname])
            results.append(hptetachargepdf)

    return results, weightsum

resultdict = narf.build_and_run(datasets, build_graph)

fname = "test.pkl.lz4"

print("writing output")
#with gzip.open(fname, "wb") as f:
with lz4.frame.open(fname, "wb") as f:
    pickle.dump(resultdict, f)
