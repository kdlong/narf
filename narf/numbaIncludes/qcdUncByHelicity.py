import uproot
import numpy as np
import numba
import ROOT

def loadXsecs(variation, bins=False):
    hists = []
    coeffnames = [f"a{i}" for i in range(8)]
    unpol,binsx,binsy = f[f"unpol_plus_{variation}"].to_numpy()
    hists.append(np.ones_like(unpol))
    for coeff in coeffnames:
        hist,binsx,binsy = f[f"{coeff}_plus_{variation}"].to_numpy()
        hists.append(hist)
    coeffs = np.stack(hists, axis=-1)
    xsecs = coeffs*np.expand_dims(unpol, axis=-1)
    return (xsecs,binsx,binsy) if bins else xsecs

f = uproot.open("/scratch/shared/angularCoefficients/fractions_plus_2022.root")
nominal,binsy,binspt = loadXsecs("nominal", bins=True)
muRmuFUp = loadXsecs("muRmuFUp")
muRmuFDown = loadXsecs("muRmuFDown")
muRUp = loadXsecs("muRUp")
muRDown = loadXsecs("muRDown")
muFUp = loadXsecs("muFUp")
muFDown = loadXsecs("muFDown")


@numba.jit(nopython=True)
def angularFacs(theta, phi):
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    sin2theta = 2.*sintheta*costheta

    return np.array([1.+costheta*costheta,
        0.5*(1.-3.*costheta*costheta), 
        np.sin(2.*theta)*np.cos(2.*theta), 
        0.5*sintheta*sintheta*cosphi,
        sintheta*cosphi,
        costheta,
        sintheta*sintheta*np.sin(2*phi),
        sin2theta*sinphi,
        sintheta*sinphi
    ])*3./16.*np.pi

@numba.jit(nopython=True)
def makeWeightsFullDim(nominal, variation, theta, phi):
    angular = angularFacs(theta, phi)
    # Annoyingly numba doesn't support a tuple argument for axes
    angexpand = np.expand_dims(np.expand_dims(angular, axis=0), axis=0)
    nom = nominal*angexpand
    var = variation*angexpand

    newshape = (nom.shape[-1], nom.shape[-1])
    # Expand dimension with off diagonal, sum over that to exclude the one index being varied
    nomexp = np.expand_dims(nom, axis=-1)
    offdiag = np.ones(newshape) - np.eye(*newshape)
    offdiagexp = np.expand_dims(np.expand_dims(offdiag, axis=0), axis=0)

    sumnom = np.sum(nomexp*offdiag, axis=-2)
    num = var+sumnom
    denom = np.full(num.shape, np.sum(nom))

    return np.where(denom != 0., num/denom, np.zeros_like(num))

@numba.jit(nopython=True)
# Nominal and variation contain values for 1 pt/y bin
def makeWeights1D(nominal, variation, theta, phi):
    angular = angularFacs(theta, phi)
    nom = nominal*angular
    var = variation*angular

    newshape = (nom.shape[-1], nom.shape[-1])
    # Expand dimension with off diagonal, sum over that to exclude the one index being varied
    nomexp = np.expand_dims(nom, axis=-1)
    offdiag = np.ones(newshape) - np.eye(*newshape)

    sumnom = np.sum(nomexp*offdiag, axis=-2)
    num = var+sumnom
    denom = np.full(num.shape, np.sum(nom))

    #return denom # has nans
    #return num  -- has nans
    vals = np.where(denom != 0., num/denom, np.zeros_like(num))
    #vals[np.isnan(vals)] = 0.
    return vals

#@numba.jit("float64(float64, int64[:])", nopython=True)
@numba.jit(nopython=True)
def findBin(val, bins):
    bindig = np.digitize(np.array(val, dtype=np.float64), bins)
    binnum = bindig-1
    if (binnum == bins.shape[-1]-1):
        binnum = binnum-1
    elif (binnum < 0):
        binnum = 0
    return binnum

@ROOT.Numba.Declare(["double", "double", "double", "double"], "RVec<double>")
def qcdUncByHelicity(ptW, yW, thetaW, phiW):
    biny = findBin(np.abs(yW), binsy)
    binpt = findBin(ptW, binspt)
    nom = nominal[biny,binpt]
    return np.concatenate((makeWeights1D(nom, muRmuFUp[biny,binpt], thetaW, phiW),
        makeWeights1D(nom, muRmuFDown[biny,binpt], thetaW, phiW),
        makeWeights1D(nom, muRUp[biny,binpt], thetaW, phiW),
        makeWeights1D(nom, muRDown[biny,binpt], thetaW, phiW),
        makeWeights1D(nom, muFUp[biny,binpt], thetaW, phiW),
        makeWeights1D(nom, muFDown[biny,binpt], thetaW, phiW),),
    axis=0)


