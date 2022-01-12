import uproot						
import numpy as np
import ROOT
import numba

# Probably should add to the repo
f = uproot.open("/scratch/shared/MuonCalibration/calibrationJDATA_aftersm.root")
cov,binsx,binsy = f["covariance_matrix"].to_numpy()

w,v = np.linalg.eigh(cov)

nparams = 6
netabins = 48

@numba.jit('int64(float64, int64)', nopython=True, nogil=True, debug=True)
def etaBin(eta, nbins):
    etamin = -2.4
    etamax = 2.4
    etastep = (etamax-etamin)/nbins
    if eta < etamin:
        eta = etamin
    elif eta >= etamax:
        eta = etamax-etastep/2
    ieta = int(np.floor((eta-etamin)/etastep))
    return ieta

@numba.jit('float64[:](float64, float64, int32, float64, boolean)', nopython=True, nogil=True, debug=True)
def calibratedPt(pt, eta, charge, scale, isUp):
    ieta = etaBin(eta, netabins)
    binlow = nparams*ieta
    # -3 because we're only using 3 of 6 params (others are for resolution)
    binhigh = nparams*(ieta+1)-3

    fac = scale if isUp else -scale
    a,e,m = fac*np.sqrt(w)*v[binlow:binhigh,:]
    k = 1.0/pt
    magnetic = 1. + a
    material = -1*e * k
    alignment = charge * m
    k_corr = (magnetic + material)*k + alignment

    return 1.0/k_corr

@ROOT.Numba.Declare(["float", "float", "int", "float"], "RVec<double>")
def calibratedPtUpVar(pt, eta, charge, scale):
    return calibratedPt(pt, eta, charge, scale, True)

@ROOT.Numba.Declare(["float", "float", "int", "float"], "RVec<double>")
def calibratedPtDownVar(pt, eta, charge, scale):
    return calibratedPt(pt, eta, charge, scale, False)

@numba.jit("float64[:](float64, int64, float64, int64)", nopython=True, nogil=True, debug=True)
def dummyCalibratedPt(pt, ieta, magnitude, etabins):
    ptvars = np.full(etabins*2, pt, dtype='float64') 
    ptvars[ieta] = pt*(1+magnitude)
    ptvars[ieta+etabins] = pt/(1+magnitude)

    return ptvars

@ROOT.Numba.Declare(["float", "float", "float", "int"], "RVec<double>")
def dummyCalibratedPtFlat(pt, eta, magnitude, nbins):
    ieta = etaBin(eta, nbins)
    return dummyCalibratedPt(pt, ieta, magnitude, nbins)

bnorms = np.zeros(netabins)
etaBins = np.floor(np.argmax(np.abs(v), axis=0) / nparams)
isBfield = (np.argmax(np.abs(v), axis=0) % nparams) == 0

maxVars = np.max(v[:, isBfield], axis=0)
# Reorder so that norms are in the order of eta bins
bnorms[etaBins[isBfield].astype(int)] = np.sqrt(w[isBfield])*maxVars
# There are 3 duplicates and 3 bins that don't get filled, just set them to 
# the average of their neighbors
zeroidxs = np.where(bnorms==0)[0]
bnorms[zeroidxs] = (bnorms[zeroidxs+1]+bnorms[zeroidxs-1])/2

@ROOT.Numba.Declare(["float", "float"], "RVec<double>")
def dummyCalibratedPtApproxBField(pt, eta):
    ieta = etaBin(eta, netabins)
    return dummyCalibratedPt(pt, ieta, bnorms[ieta], netabins)

@ROOT.Numba.Declare(["float", "float", "RVec<float>", "float", "int"], "RVec<float>")
def dummyScaleFromMassWeights(pt, eta, massWeights, scale, bins):
    upWeight = massWeights[10+1]
    downWeight = massWeights[10-1]
    weightsPerEta = np.ones(bins*2, dtype='float32')
    ieta = etaBin(eta, bins)
    weightsPerEta[ieta] = np.exp(scale*np.log(upWeight))
    weightsPerEta[ieta+bins] = np.exp(scale*np.log(downWeight))
    return weightsPerEta

@numba.jit("float64[:](float64, float64, int32, float32[:], boolean)", nopython=True, nogil=True, debug=True)
def calibratedPtMassWeightsProxy(pt, eta, charge, massWeights, isUp):
    relptcorr = (calibratedPt(pt, eta, charge, 1., isUp)-pt)/pt
    # Up-like weight if it increases pt, down otherwise
    upweight = massWeights[10+1]
    downweight = massWeights[10-1]
    weight = np.full_like(relptcorr, upweight)
    weight[relptcorr < 0] = downweight
    # Scale factor of 10 MeV mass shift --> pt percent unc.
    corrfac = np.abs(relptcorr)/1.2482e-4
    return np.exp(corrfac*np.log(np.abs(weight)))*np.sign(weight)

@ROOT.Numba.Declare(["double", "double", "int", "RVec<float>"], "RVec<double>")
def calibratedPtMassWeightsProxyUpVar(pt, eta, charge, massWeights):
    return calibratedPtMassWeightsProxy(pt, eta, charge, massWeights, True)

@ROOT.Numba.Declare(["double", "double", "int", "RVec<float>"], "RVec<double>")
def calibratedPtMassWeightsProxyDownVar(pt, eta, charge, massWeights):
    return calibratedPtMassWeightsProxy(pt, eta, charge, massWeights, False)
