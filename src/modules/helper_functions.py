import numpy as np
from tqdm.notebook import tqdm
import astropy.units as u
from astropy.cosmology import WMAP7
from .spline import spline, new_spline

def eval_poly(t, coefs):
    n = len(coefs)
    result = coefs[-1]
    for i in range(n-2, -1, -1):
        result *= t
        result += coefs[i]
    return result

def eval_diff_sq_dist(t, diff_X, diff_Y, diff_Z, coef_Rvir_0):
    return eval_poly(t, diff_X)**2 + eval_poly(t, diff_Y)**2 + eval_poly(t, diff_Z)**2 - eval_poly(t, coef_Rvir_0)**2

def eval_der_diff_sq_dist(t, diff_X, diff_Y, diff_Z, coef_Rvir_0):
    arange_0 = np.arange(1, len(diff_X))
    arange_1 = np.arange(1, len(coef_Rvir_0))
    return 2*(eval_poly(t, diff_X)*eval_poly(t, arange_0*diff_X[1:]) + eval_poly(t, diff_Y)*eval_poly(t, arange_0*diff_Y[1:]) + eval_poly(t, diff_Z)*eval_poly(t, arange_0*diff_Z[1:]) - eval_poly(t, coef_Rvir_0)*eval_poly(t, arange_1*coef_Rvir_0[1:]))

def solve(diff_time, diff_X, diff_Y, diff_Z, coef_Rvir_0, stol = 10e-10):
    """
    solve for time t at which the subhalo enters the virial radius
    """
    ctime = 0
    ntime = ctime - eval_diff_sq_dist(ctime, diff_X, diff_Y, diff_Z, coef_Rvir_0)/eval_der_diff_sq_dist(ctime, diff_X, diff_Y, diff_Z, coef_Rvir_0)
    count = 0
    while count < 200 and abs(ntime-ctime) >= stol:
        ctime = ntime
        ntime = ctime - eval_diff_sq_dist(ctime, diff_X, diff_Y, diff_Z, coef_Rvir_0)/eval_der_diff_sq_dist(ctime, diff_X, diff_Y, diff_Z, coef_Rvir_0)
        count += 1
    if count == 200:
        return -1
    else:
        return ctime

def to_spherical(x,y,z):
    """
    find [theta, phi] in [-pi/2,pi/2]x[-pi,pi] where
    x = r*cos(theta)*cos(phi)
    y = r*cos(theta)*sin(phi)
    z = r*sin(theta)
    """
    theta = np.arctan(z/(x**2+y**2)**(1/2))
    phi = np.arctan(y/x)
    if x > 0:
        if phi > 0:
            phi -= np.pi
        else:
            phi += np.pi
    return [theta, phi]

def to_direction(x,y,z):
    return [x/(x**2+y**2+z**2)**0.5,y/(x**2+y**2+z**2)**0.5,z/(x**2+y**2+z**2)**0.5]

def read_elvis_tracks(elvis_dir, elvis_name, 
                      varnames = None):
    """
    first 2 inputs are directory and simulation name
    varnames is a python list of strings of variable names 
    to extract and return. To see available variable names consult 
    corresponding subdirectory
    """
    prefix = elvis_dir + '/' +  elvis_name + '/'
    suffix = '.txt'
    
    vars_info = []
    for iv, var in tqdm(enumerate(varnames)):
        file_path = prefix + var + suffix
        f = np.loadtxt(file_path)
        vars_info.append(f)

    #return the dictionary of properties. Each property can be accessed by varnames
    data = dict(zip(varnames, vars_info))
    return data

def plot_pretty(dpi=175,fontsize=9):
    # import pyplot and set some parameters to make plots prettier
    import matplotlib.pyplot as plt

    plt.rc("savefig", dpi=dpi)
    plt.rc("figure", dpi=dpi)
    plt.rc('text', usetex=True)
    plt.rc('font', size=fontsize)
    plt.rc('xtick', direction='in') 
    plt.rc('ytick', direction='in')
    plt.rc('xtick.major', pad=5) 
    plt.rc('xtick.minor', pad=5)
    plt.rc('ytick.major', pad=5) 
    plt.rc('ytick.minor', pad=5)
    plt.rc('lines', dotted_pattern = [0.5, 1.1])

    return

def get_num_halos(data):
    return len(data['X'])

def get_num_time(data):
    return len(data['X'][0])

def extract_elvis_data(data, row):
    non_zero_index = len(data['X'][0])-1
    while data['Mvir'][row][non_zero_index] == 0:
        non_zero_index -= 1
    non_zero_index += 1
    halo_mvir = np.flip(data['Mvir'][row][:non_zero_index])
    scale = np.flip(data['scale'][row][:non_zero_index])
    X = np.flip(data['X'][row][:non_zero_index])
    Y = np.flip(data['Y'][row][:non_zero_index])
    Z = np.flip(data['Z'][row][:non_zero_index])
    Vx = np.flip(np.array((data['Vx'][row] * u.km/u.s).to(u.Mpc/u.Gyr))[:non_zero_index])/scale
    Vy = np.flip(np.array((data['Vy'][row] * u.km/u.s).to(u.Mpc/u.Gyr))[:non_zero_index])/scale
    Vz = np.flip(np.array((data['Vz'][row] * u.km/u.s).to(u.Mpc/u.Gyr))[:non_zero_index])/scale
    Rvir = np.flip(data['Rvir'][row][:non_zero_index]/1000)
    zreds = (1/scale) - 1
    lookback_time = -np.array(WMAP7.lookback_time(zreds))

    coefs_X = np.array(list(new_spline(lookback_time, X, Vx)))
    coefs_Y = np.array(list(new_spline(lookback_time, Y, Vy)))
    coefs_Z = np.array(list(new_spline(lookback_time, Z, Vz)))
    coefs_Rvir = np.array(list(spline(lookback_time, Rvir)))
    coefs_Mvir = np.array(list(spline(lookback_time, halo_mvir)))

    return [non_zero_index, lookback_time, X, Y, Z, Vx, Vy, Vz, halo_mvir, Rvir, coefs_X, coefs_Y, coefs_Z, coefs_Rvir, coefs_Mvir]