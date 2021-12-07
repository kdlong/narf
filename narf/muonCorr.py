import uproot						
import numpy as np
import ROOT

f = uproot.open("/scratch/shared/MuonCalibration/calibrationJDATA_aftersm.root")
cov,binsx,binsy = f["covariance_matrix"].to_numpy()

w,v = np.linalg.eigh(cov)

@ROOT.Numba.Declare(["float", "float", "int", "bool"], "RVec<double>")
def calibratedPt(pt: float, eta: float, charge: int, isUp: bool) -> float:
    netabins = 48
    nparams = 6
    etamin = -2.4
    etamax = 2.4
    etastep = (etamax-etamin)/netabins
    ieta = int(np.round((eta-etamin)/etastep))

    binlow = nparams*ieta
    # -3 because we're only using 3 of 6 params (others are for resolution)
    binhigh = nparams*(ieta+1)-3

    fac = 1 if isUp else -1
    a,e,m = fac*np.sqrt(w)*v[binlow:binhigh,:]
    k = 1.0/pt
    magnetic = 1. + a
    material = -1*e * k
    alignment = charge * m
    k_corr = (magnetic + material)*k + alignment

    return 1.0/k_corr
