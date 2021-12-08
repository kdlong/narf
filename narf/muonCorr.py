import uproot						
import numpy as np
import ROOT
import numba

f = uproot.open("/scratch/shared/MuonCalibration/calibrationJDATA_aftersm.root")
cov,binsx,binsy = f["covariance_matrix"].to_numpy()

w,v = np.linalg.eigh(cov)

@numba.jit(nopython=True, nogil=True)
def calibratedPt(pt, eta, charge, isUp):
    #return np.array([pt*7., pt*1., pt*3.], dtype='float32')
    netabins = 48
    nparams = 6
    etamin = -2.4
    etamax = 2.4
    etastep = (etamax-etamin)/netabins

    if eta < etamin:
        eta = etamin
    elif eta >= etamax:
        eta = etamax-etastep/2
    ieta = int(np.floor((eta-etamin)/etastep))

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

@ROOT.Numba.Declare(["float", "float", "int"], "RVec<double>")
def calibratedPtUpVar(pt, eta, charge):
    return calibratedPt(pt, eta, charge, True)

@ROOT.Numba.Declare(["float", "float", "int"], "RVec<double>")
def calibratedPtDownVar(pt, eta, charge):
    return calibratedPt(pt, eta, charge, False)

