import numpy as np
from scipy.special import kolmogorov 

def ks_uniformity_test(x, xbin=None):
    """
    test whether sample in 1d array x is uniformly distributed in [xbin[0], xbin[1]]
    
    Parameters:
    -----------
    x - 1d numpy array containing sample values 
    xbin - list of size 2 containing limits of the uniform distribution
        if None, the range is defined as [x.min(), x.max()]
    
    Returns:
    --------
    float - Kolmogorov probability for D_KS statistic computed using sample CDF and uniform cdf
    """
    if xbin is None: 
        xbin = [x.min(), x.max()]
    nx = x.size
    if nx <= 1:
        return 1. # impossible to tell for such small number of samples 
    # cdf for the x sample 
    xcdf = np.arange(1,nx+1,1) / (nx-1)
    # compute D_KS statistic 
    dks = nx**0.5 *  np.max(np.abs(xcdf -(np.sort(x) - xbin[0])/(xbin[1] - xbin[0])))
    return kolmogorov(dks) # return value of Kolmogorov pdf for this D_KS