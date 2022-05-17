from itertools import combinations
import random
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde
import scipy.optimize as opt
from modules.helper_functions_v3 import get_MW, read_specific, to_spherical
from modules.stats_v3 import get_D_sph, get_R_med, get_D_rms

from .stats_v3 import conf_interval
from .helper_functions_v3 import normalize, to_degree

def plot_2d_dist(x,y, xlim, ylim, nxbins, nybins, figsize=(5,5), 
                cmin=1.e-4, cmax=1.0, smooth=None, xpmax=None, ypmax=None, 
                log=False, weights=None, xlabel='x', ylabel='y', 
                clevs=None, fig_setup=None, savefig=None):
  """
  construct and plot a binned, 2d distribution in the x-y plane 
  using nxbins and nybins in x- and y- direction, respectively
  
  log = specifies whether logged quantities are passed to be plotted on log-scale outside this routine
  """
  if fig_setup is None:
    fig, ax = plt.subplots(figsize=figsize)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
  else:
    ax = fig_setup
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
  if xlim[1] < 0.: ax.invert_xaxis()

  if weights is None: weights = np.ones_like(x)
  H, xbins, ybins = np.histogram2d(x, y, weights=weights, bins=(np.linspace(xlim[0], xlim[1], nxbins),np.linspace(ylim[0], ylim[1], nybins)))
  
  H = np.rot90(H); H = np.flipud(H); 
            
  X,Y = np.meshgrid(xbins[:-1],ybins[:-1]) 

  if smooth != None:
    from scipy.signal import wiener
    H = wiener(H, mysize=smooth)
      
  H = H/np.sum(H)        
  Hmask = np.ma.masked_where(H==0,H)
  
  if log:
    X = np.power(10.,X); Y = np.power(10.,Y)

  pcol = ax.pcolormesh(X, Y,(Hmask), cmap=plt.cm.BuPu, norm=LogNorm(), linewidth=0., rasterized=False)
  pcol.set_edgecolor('face')
  
  # plot contours if contour levels are specified in clevs 
  if clevs is not None:
    lvls = []
    for cld in clevs:  
      sig = opt.brentq(conf_interval, 0., 1., args=(H,cld) )   
      lvls.append(sig)
    
    ax.contour(X, Y, H, linewidths=(1.0,0.75, 0.5, 0.25), colors='black', levels = sorted(lvls), 
            norm = LogNorm(), extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]])
  if xpmax is not None:
    ax.scatter(xpmax, ypmax, marker='x', c='orangered', s=20)
  if savefig:
    plt.savefig(savefig,bbox_inches='tight')
  if fig_setup is None:
    plt.show()
  return ax

def plot_vectors(vectors, img_name=None, save_dir=None, title=None, size=None, ax=None, saveimage=False):
  """plot vectors in aitoff projection

  Parameters
  ----------
  vectors : array
      vectors in spherical coordinates of shape (n, 2)
  img_name : str, optional
      filename of the plot, by default None
  save_dir : str, optional
      save directory of the plot, by default None
  size : array, optional
      size of quantities of shape (n,), by default None
  title : str, optional
      title of the plot, by default None
  ax : Axes, optional
      Axes of the plot, by default None
  saveimage : bool, optional
      save image if True, by default False
  """
  if img_name is None or save_dir is None:
    saveimage = False

  vectors = normalize(vectors)
  if size is None:
    size = np.ones_like(vectors[:,0], dtype='float')

  if ax is None:
    ax = plt.figure(figsize=(8, 6)).add_subplot(111, projection="aitoff")

  ax.scatter(vectors[:,1], vectors[:,0], marker = '.', 
                          s = (size/np.max(size))**(2/5)*200)
  plt.rcParams['axes.titley'] = 1.1
  if title is not None:
    ax.title(title)
  ax.grid(True)

  if saveimage:
    plt.savefig(f'{save_dir}/{img_name}')

def plot3D(vectors, ax = None):
  """plot 3D vectors

  Parameters
  ----------
  vectors : array
      array of shape (n, 3)
  ax : Axes, optional
      Axes of the plot, by default None
  """
  if ax is None:
    ax = plt.figure().add_subplot(projection='3d')
  ax.set_box_aspect(aspect = (1,1,1))
  ax.scatter(vectors[:,0],vectors[:,1],vectors[:,2])

def plot_distribution_D_rms_dispersion(suite_name, data_dir, data_surv_probs_dir, data, brightest_dir=None, 
                    is_heaviest=False, select_by_Rvir=False, seed=None, save_dir="", ax=None, saveimage=False):
  """plot distribution of D_rms dispersion

  Parameters
  ----------
  suite_name : str
      name of the suite
  data_dir : str
      directory of the generated data
  data_surv_probs_dir : str
      directory of the generated data with surv_probs
  data : dataframe 
      dataframe containing relevant attributes
  brightest_dir : str, optional
      directory of the generated data for brightest subhalos with surv_probs, by default None
  is_heaviest : bool, optional
      plot for heaviest if True, by default False
  select_by_Rvir : bool, optional
      True if choosing subhalos inside Rvir, by default True
  seed : _type_, optional
      set random seed, by default None
  save_dir : str, optional
      save directory of the plot, by default ""
  ax : Axes, optional
      Axes of the plot, by default None
  saveimage : bool, optional
      save image if True, by default False
  """
  df = pd.read_csv(f"{data_dir}/{suite_name}.csv", usecols=['D_rms', 'R_med'])
  df_surv_probs = pd.read_csv(f"{data_surv_probs_dir}/{suite_name}.csv", usecols=['D_rms', 'R_med'])

  if ax is None:
    ax = plt.figure(figsize=(8, 6)).add_subplot()

  D_rms = np.array(df['D_rms'])
  R_med = np.array(df['R_med'])
  D_rms_surv_probs = np.array(df_surv_probs['D_rms'])
  R_med_surv_probs = np.array(df_surv_probs['R_med'])

  points = np.linspace(0, 1, num = 1000)

  kernel_halo = gaussian_kde(D_rms/R_med)
  kernel_halo_surv_probs = gaussian_kde(D_rms_surv_probs/R_med_surv_probs)
  ax.plot(points, kernel_halo(points), label=f"{suite_name} without surv_probs", color='b')
  ax.plot(points, kernel_halo_surv_probs(points), label=f"{suite_name} with surv_probs", color='m')
  
  pos_brightest_without_surv_probs = read_specific(data, type='brightest', is_surv_probs=False, seed=seed, select_by_Rvir=select_by_Rvir)['pos']

  data_MW = get_MW(is_D_rms=True, is_R_med=True, num_D_sph=None, num_D_sph_flipped=None)

  ax.arrow(data_MW['D_rms']/data_MW['R_med'],0,0,0.5, label="MW",color='orangered',head_width=0.01,head_length=0.3)
  ax.arrow(get_D_rms(pos_brightest_without_surv_probs)['D_rms']/get_R_med(pos_brightest_without_surv_probs)['R_med'],0,0,0.5, label="brightest without surv_probs",color='c',head_width=0.01,head_length=0.1)

  if brightest_dir is None:
    pos_brightest = read_specific(data, type='brightest', seed=seed, select_by_Rvir=select_by_Rvir)['pos']
    ax.arrow(get_D_rms(pos_brightest)['D_rms']/get_R_med(pos_brightest)['R_med'],0,0,0.5, label="brightest",color='y',head_width=0.01,head_length=0.1)
  else:
    data_brightest_sampled = pd.read_csv(f'{brightest_dir}/{suite_name}.csv', usecols=['D_rms', 'R_med'])
    kernel_brightest = gaussian_kde(data_brightest_sampled['D_rms']/data_brightest_sampled['R_med'])
    ax.plot(points, kernel_brightest(points), label=f"brightest",color='y')

  if is_heaviest:
    pos_heaviest = read_specific(data, type='heaviest', seed=seed, select_by_Rvir=select_by_Rvir)['pos']
    pos_heaviest_without_surv_probs = read_specific(data, type='heaviest', is_surv_probs=False, seed=seed, select_by_Rvir=select_by_Rvir)['pos']
    ax.arrow(get_D_rms(pos_heaviest)['D_rms']/get_R_med(pos_heaviest)['R_med'],0,0,0.5, label="heaviest",color='g',head_width=0.01,head_length=0.2)
    ax.arrow(get_D_rms(pos_heaviest_without_surv_probs)['D_rms']/get_R_med(pos_heaviest_without_surv_probs)['R_med'],0,0,0.5, label="heaviest without surv_probs",color='k',head_width=0.01,head_length=0.2)

  ax.set_title('The distribution of the D_rms dispersions')
  ax.set_xlabel('$\\Delta_{\\textrm{rms}}/R_{\\textrm{med}}$')
  ax.set_ylabel('P($\\Delta_{\\textrm{rms}}/R_{\\textrm{med}}$)')

  ax.legend()

  if saveimage:
    plt.savefig(f"{save_dir}/distribution_D_rms_over_R_med_dispersion_for_{suite_name}.pdf")

def plot_circle_around_vector(average_pole, d_angle, ax=None, label=""):
  """plot circle around a vector

  Parameters
  ----------
  average_pole : array
      array of shape (3,)
  d_angle : float
      D_sph in rad
  ax : Axes, optional
      Axes of the plot, by default None
  label : str, optional
      label of the plot, by default ""
  """
  if ax is None:
    ax = plt.figure(figsize=(8, 6)).add_subplot(projection='aitoff')
  x, y, z = average_pole
  rho = (x**2 + y**2)**(1/2)
  rot_matrix = np.array([[x*z/rho, y*z/rho, -rho],[-y/rho, x/rho, 0],[x,y,z]])
  phi = np.random.uniform(size=1000)*2*np.pi
  
  X = np.cos(phi)*np.sin(d_angle)
  Y = np.sin(phi)*np.sin(d_angle)
  Z = np.cos(d_angle)* np.ones_like(X)
  
  pos = np.array([X,Y,Z]).T
  
  rot_pos = np.matmul(pos, rot_matrix)
  
  aitoff_phis = []
  aitoff_thetas = []
  
  for i in range(1000):
    cur = rot_pos[i]
    aitoff_theta, aitoff_phi = to_spherical(cur[0],cur[1],cur[2])
    aitoff_thetas.append(aitoff_theta)
    aitoff_phis.append(aitoff_phi)
  
  ax.scatter(aitoff_phis, aitoff_thetas, marker = '.', label=label)
  ax.grid(True)
  plt.rcParams['axes.titley'] = 1.1

def plot_poles_brightest(suite_name, data, select_by_Rvir=False, seed=None, save_dir="", 
                    num_chosen=11, ax=None, saveimage=False):
  """plot brightest poles with D_sph for num_chosen

  Note: without surv_probs

  Parameters
  ----------
  suite_name : str
      name of the suite
  data : dataframe 
      dataframe containing relevant attributes
  select_by_Rvir : bool, optional
      True if choosing subhalos inside Rvir, by default True
  seed : int, optional
      set random seed, by default None
  save_dir : str, optional
      save directory of the plot, by default ""
  num_chosen : int, optional
      number of chosen brightest subhalos, by default 11
  ax : Axes, optional
      Axes of the plot, by default None
  saveimage : bool, optional
      save image if True, by default False
  """
  if ax is None:
    ax = plt.figure(figsize=(8, 6)).add_subplot(111, projection="aitoff")

  poles = read_specific(data, select_by_Rvir=select_by_Rvir, seed=seed)['poles']

  min_average_poles = []
  min_d_angles = []

  for k in range(3, num_chosen+1):
    indices = np.array(list(combinations(np.arange(11),k)))
    chosen_poles = poles[indices]
    dic = get_D_sph(chosen_poles, isAvg=True)
    
    argmin = np.argmin(dic['D_sph'])

    min_d_angles.append(dic['D_sph'][argmin])
    min_average_poles.append(dic['avg'][argmin])

  count = 3
  for average_pole, d_angle in zip(min_average_poles, min_d_angles):
    plot_circle_around_vector(average_pole, d_angle, ax=ax,label=f"{count}")
    count += 1

  plt.rcParams['axes.titley'] = 1.1
  ax.set_title(f"Distribution of brightest poles and uncertainties for {suite_name}")
  ax.legend(loc='upper right', bbox_to_anchor=(0.6, 0., 0.5, 0.3))

  if saveimage:
    plt.savefig(f"{save_dir}/plot_poles_brightest_for_{suite_name}.pdf")

def plot_poles_brightest_with_config(suite_name, data, select_by_Rvir=False, seed=None, save_dir="", 
                    num_chosen=11, ax=None, saveimage=False):
  """plot brightest poles with D_sph for num_chosen with the best config

  Note: without surv_probs

  Parameters
  ----------
  suite_name : str
      name of the suite
  data : dataframe 
      dataframe containing relevant attributes
  select_by_Rvir : bool, optional
      True if choosing subhalos inside Rvir, by default True
  seed : int, optional
      set random seed, by default None
  save_dir : str, optional
      save directory of the plot, by default ""
  num_chosen : int, optional
      number of chosen brightest subhalos, by default 11
  ax : Axes, optional
      Axes of the plot, by default None
  saveimage : bool, optional
      save image if True, by default False
  """
  if ax is None:
    ax = plt.figure(figsize=(8, 6)).add_subplot(111, projection="aitoff")
  
  dic = read_specific(data, select_by_Rvir=select_by_Rvir, seed=seed)
  poles = dic['poles']
  pos = dic['pos']

  indices = np.array(list(combinations(np.arange(11),num_chosen)))
  chosen_poles = poles[indices]
  dic = get_D_sph(chosen_poles, isAvg=True)
  
  argmin = np.argmin(dic['D_sph'])

  d_angle = dic['D_sph'][argmin]
  average_pole = dic['avg'][argmin]

  plot_circle_around_vector(average_pole, d_angle, ax=ax,label=f"{num_chosen}")

  pos_thetas = []
  pos_phis = []

  chosen_poles_ = chosen_poles[argmin]

  for i in range(len(chosen_poles_)):
    cur = chosen_poles_[i]
    aitoff_theta, aitoff_phi = to_spherical(cur[0],cur[1],cur[2])
    pos_thetas.append(aitoff_theta)
    pos_phis.append(aitoff_phi)

  ax.scatter(pos_phis, pos_thetas, marker = '.')
  ax.grid(True)
  plt.rcParams['axes.titley'] = 1.1
  ax.set_title(f"Distribution of brightest poles config for {suite_name}")
  ax.set_xlabel("$\\Delta_{\\textrm{rms}}/R_{\\textrm{med}}$: "+"{:.2f}".format(get_D_rms(pos)['D_rms']/get_R_med(pos)['R_med'])+"-$\\Delta_{\\textrm{sph}}: $"+"{:.2f}$^\circ$".format(to_degree(d_angle)))
  ax.legend(loc='upper right', bbox_to_anchor=(0.6, 0., 0.5, 0.3))

  if saveimage:
    plt.savefig(f"{save_dir}/plot_poles_brightest_with_config_for_{suite_name}.pdf")

def plot_D_sph_vs_k_brightest(suite_name, data, brightest_dir=None, save_dir="", select_by_Rvir=False, seed=None, 
                    num_chosen=11, ax=None, saveimage=False):
  """plot D_sph as a function of k with/without surv_probs

  Parameters
  ----------
  suite_name : str
      name of the suite
  data : dataframe 
      dataframe containing relevant attributes
  brightest_dir : str, optional
      directory of the generated data for brightest subhalos with surv_probs, by default None
  save_dir : str, optional
      save directory of the plot, by default ""
  select_by_Rvir : bool, optional
      True if choosing subhalos inside Rvir, by default True
  seed : int, optional
      set random seed, by default None
  num_chosen : int, optional
      number of chosen brightest subhalos, by default 11
  ax : Axes, optional
      Axes of the plot, by default None
  saveimage : bool, optional
      save image if True, by default False
  """
  if ax is None:
    ax = plt.figure(figsize=(8, 6)).add_subplot()
  
  dic = read_specific(data, select_by_Rvir=select_by_Rvir, seed=seed)
  poles = dic['poles']
  poles_without_surv_probs = read_specific(data, is_surv_probs=False, select_by_Rvir=select_by_Rvir, seed=seed)['poles']

  arr = []
  arr_without_surv_probs = []
  arr_MW = []
  data_MW = get_MW(is_D_rms=False, is_R_med=False, num_D_sph=list(np.arange(3, 12)), num_D_sph_flipped=None)

  for k in range(3, 12):
    arr_MW.append(to_degree(data_MW['D_sph'][k]))

  for k in range(3, num_chosen+1):
    indices = np.array(list(combinations(np.arange(11),k)))
    chosen_poles = poles[indices]
    chosen_poles_without_surv_probs = poles_without_surv_probs[indices]
    
    arr.append(to_degree(np.min(get_D_sph(chosen_poles, isAvg=False)['D_sph'])))
    arr_without_surv_probs.append(to_degree(np.min(get_D_sph(chosen_poles_without_surv_probs, isAvg=False)['D_sph'])))
  
  if brightest_dir is None:
    ax.plot(np.arange(3,num_chosen+1), arr, label='with surv_probs')
  else:
    arr = []
    arr_lowers = [[],[],[]]
    arr_uppers = [[],[],[]]
    data_brightest_sampled = pd.read_csv(f'{brightest_dir}/{suite_name}.csv')

    for k in range(3, num_chosen+1):
      D_sph = to_degree(data_brightest_sampled[f'D_sph_{k}'])

      arr.append(np.median(D_sph))

      arr_lowers[0].append(np.percentile(D_sph, 15.86))
      arr_uppers[0].append(np.percentile(D_sph, 84.14))
      arr_lowers[1].append(np.percentile(D_sph, 2.28))
      arr_uppers[1].append(np.percentile(D_sph, 97.72))
      arr_lowers[2].append(np.percentile(D_sph, 0.13))
      arr_uppers[2].append(np.percentile(D_sph, 99.87))
    
    ax.plot(np.arange(3,num_chosen+1), arr, label='with surv_probs', color='g')
    ax.fill_between(np.arange(3,num_chosen+1), arr_lowers[2], arr_uppers[2], alpha=0.2, color='g')
    ax.fill_between(np.arange(3,num_chosen+1), arr_lowers[1], arr_uppers[1], alpha=0.25, color='g')
    ax.fill_between(np.arange(3,num_chosen+1), arr_lowers[0], arr_uppers[0], alpha=0.3, color='g')

  ax.plot(np.arange(3,num_chosen+1), arr_without_surv_probs, label='without surv_probs', color='b')
  ax.plot(np.arange(3,num_chosen+1), arr_MW, label='MW', color='orangered')
  ax.set_title(f"D_sph(k) vs k brightest for {suite_name}")
  ax.set_xlabel("k")
  ax.set_ylabel("$\\Delta_{\\textrm{sph}}$ "+"($^\\circ$)")

  ax.legend()

  if saveimage:
    plt.savefig(f"{save_dir}/plot_D_sph_vs_k_brightest_3sigma_for_{suite_name}.pdf")

def plot_distribution_D_sph_dispersion(suite_name, data_dir, data_surv_probs_dir, data, brightest_dir=None, 
                    is_heaviest=False, k=11, select_by_Rvir=False, seed=None, save_dir="", ax=None, saveimage=False):
  """plot distribution of D_rms dispersion

  Parameters
  ----------
  suite_name : str
      name of the suite
  data_dir : str
      directory of the generated data
  data_surv_probs_dir : str
      directory of the generated data with surv_probs
  data : dataframe 
      dataframe containing relevant attributes
  brightest_dir : str, optional
      directory of the generated data for brightest subhalos with surv_probs, by default None
  is_heaviest : bool, optional
      plot for heaviest if True, by default False
  k : int, optional
      _description_, by default 11
  select_by_Rvir : bool, optional
      True if choosing subhalos inside Rvir, by default True
  seed : _type_, optional
      set random seed, by default None
  save_dir : str, optional
      save directory of the plot, by default ""
  ax : Axes, optional
      Axes of the plot, by default None
  saveimage : bool, optional
      save image if True, by default False
  """
  df = pd.read_csv(f"{data_dir}/{suite_name}.csv", usecols=[f'D_sph_{k}'])
  df_surv_probs = pd.read_csv(f"{data_surv_probs_dir}/{suite_name}.csv", usecols=[f'D_sph_{k}'])

  d_sphs = to_degree(df[f'D_sph_{k}'])
  d_sphs_surv_probs = to_degree(df_surv_probs[f'D_sph_{k}'])

  if ax is None:
    _, ax = plt.subplots(figsize=(8,6))

  points = np.linspace(0, 180, num = 1000)

  kernel_sph = gaussian_kde(d_sphs)
  kernel_sph_surv_probs = gaussian_kde(d_sphs_surv_probs)

  ax.plot(points, kernel_sph(points), label=f"{suite_name} without surv_probs", color='b')
  ax.plot(points, kernel_sph_surv_probs(points), label=f"{suite_name} with surv_probs", color='m')
  
  poles_brightest_without_surv_probs = read_specific(data, type='brightest', is_surv_probs=False, seed=seed, select_by_Rvir=select_by_Rvir)['poles']
  
  indices = np.array(list(combinations(np.arange(11),k)))
  
  chosen_poles_brightest_without_surv_probs = poles_brightest_without_surv_probs[indices]
  
  data_MW = get_MW(is_D_rms=False, is_R_med=False, num_D_sph=k, num_D_sph_flipped=None)

  if brightest_dir is None:
    poles_brightest = read_specific(data, type='brightest with surv_probs', select_by_Rvir=select_by_Rvir, seed=seed)['poles']
    chosen_poles_brightest = poles_brightest[indices]
    ax.arrow(to_degree(np.min(get_D_sph(chosen_poles_brightest)['D_sph'])),0,0,0.005, label="brightest",color='y',head_width=1,head_length=0.00125)
  else:
    data_brightest_sampled = pd.read_csv(f'{brightest_dir}/{suite_name}.csv', usecols=[f'D_sph_{k}'])
    kernel_brightest = gaussian_kde(to_degree(data_brightest_sampled[f'D_sph_{k}']))
    ax.plot(points, kernel_brightest(points), label=f"brightest",color='y')

  ax.arrow(to_degree(data_MW['D_sph']),0,0,0.005, label="MW",color='orangered',head_width=1,head_length=0.00375)
  ax.arrow(to_degree(np.min(get_D_sph(chosen_poles_brightest_without_surv_probs)['D_sph'])),0,0,0.005, label="brightest without surv_probs",color='c',head_width=1,head_length=0.00125)

  if is_heaviest:
    poles_heaviest = read_specific(data, type='heaviest', select_by_Rvir=select_by_Rvir, seed=seed)['poles']
    poles_heaviest_without_surv_probs = read_specific(data, type='heaviest', is_surv_probs=False, seed=seed, select_by_Rvir=select_by_Rvir)['poles']

    chosen_poles_heaviest = poles_heaviest[indices]
    chosen_poles_heaviest_without_surv_probs = poles_heaviest_without_surv_probs[indices]

    ax.arrow(to_degree(np.min(get_D_sph(chosen_poles_heaviest)['D_sph'])),0,0,0.005, label="heaviest",color='g',head_width=1,head_length=0.0025)
    ax.arrow(to_degree(np.min(get_D_sph(chosen_poles_heaviest_without_surv_probs)['D_sph'])),0,0,0.005, label="heaviest without surv_probs",color='k',head_width=1,head_length=0.0025)

  
  ax.set_xlabel('$\\Delta_{\\textrm{sph}}$ '+"$(^{\\circ})$")
  ax.set_ylabel('P($\\Delta_{\\textrm{sph}}$)')

  ax.legend()

  ax.set_title(f'The distribution of the D_sph({k}) dispersions')
  if saveimage:
    plt.savefig(f"{save_dir}/distribution_D_sph_dispersion_for_{suite_name}.pdf")

def plot_hist_D_rms_over_R_med_vs_D_sph(suite_name, data_dir, data, brightest_dir=None, is_surv_probs=False, 
                    is_heaviest=False, k=11, select_by_Rvir=False, seed=None, save_dir="", ax=None, saveimage=False):
  """histogram of D_rms/R_med vs D_sph

  Parameters
  ----------
  suite_name : str
      name of the suite
  data_dir : str
      directory of the generated data
  data : dataframe 
      dataframe containing relevant attributes
  brightest_dir : str, optional
      directory of the generated data for brightest subhalos with surv_probs, by default None
  is_surv_probs : bool, optional
      True if the sample of size k uses surv_probs, by default False
  is_heaviest : bool, optional
      include  heaviest if True, by default False
  k : int, optional
      number of subhalos, by default 11
  select_by_Rvir : bool, optional
      True if choosing subhalos inside Rvir, by default True
  seed : _type_, optional
      set random seed, by default None
  save_dir : str, optional
      save directory of the plot, by default ""
  ax : Axes, optional
      Axes of the plot, by default None
  saveimage : bool, optional
      save image if True, by default False
  """
  if ax is None:
    fig, ax = plt.subplots(figsize=(8,6))

  dic = pd.read_csv(f"{data_dir}/{suite_name}.csv", usecols=['D_rms','R_med',f'D_sph_{k}'])

  D_rms = dic['D_rms']
  R_med = dic['R_med']
  D_sph = to_degree(dic[f'D_sph_{k}'])

  scaled_rms = D_rms/R_med

  data_MW = get_MW(is_D_rms=True, is_R_med=True, num_D_sph=k, num_D_sph_flipped=None)

  data_brightest_without_surv_probs = read_specific(data, type='brightest', is_surv_probs=False, seed=seed, select_by_Rvir=select_by_Rvir)

  poles_brightest_without_surv_probs = data_brightest_without_surv_probs['poles']

  D_rms_MW = data_MW['D_rms']
  D_sph_MW = to_degree(data_MW['D_sph'])
  R_med_MW = data_MW['R_med']

  xmin = min(np.min(scaled_rms), D_rms_MW/R_med_MW)
  xmax = max(np.max(scaled_rms), D_rms_MW/R_med_MW)
  ymin = min(np.min(D_sph), D_sph_MW)
  ymax = max(np.max(D_sph), D_sph_MW)

  indices = np.array(list(combinations(np.arange(11),k)))
  chosen_poles_brightest_without_surv_probs = poles_brightest_without_surv_probs[indices]

  plot_2d_dist(scaled_rms, D_sph, [xmin, xmax], [ymin, ymax], 50, 50, figsize=(5,5),fig_setup=ax,clevs=[0.6827, 0.9545, 0.9973])
  ax.scatter(D_rms_MW/R_med_MW, D_sph_MW, label='MW', marker='x', c='orangered', s=100)
  ax.scatter(get_D_rms(data_brightest_without_surv_probs['pos'])['D_rms']/get_R_med(data_brightest_without_surv_probs['pos'])['R_med'], 
      to_degree(np.min(get_D_sph(chosen_poles_brightest_without_surv_probs)['D_sph'])), label='brightest without surv_probs', marker='x', c='c', s=100)

  if brightest_dir is None:
    data_brightest = read_specific(data, type='brightest', select_by_Rvir=select_by_Rvir, seed=seed)
    poles_brightest = data_brightest['poles']
    chosen_poles_brightest = poles_brightest[indices]

    ax.scatter(get_D_rms(data_brightest['pos'])['D_rms']/get_R_med(data_brightest['pos'])['R_med'], 
      to_degree(np.min(get_D_sph(chosen_poles_brightest)['D_sph'])), label='brightest', marker='x', c='y', s=100)
    
  else:
    data_brightest_sampled = pd.read_csv(f'{brightest_dir}/{suite_name}.csv', usecols=['D_rms', 'R_med', f'D_sph_{k}'])

    D_rms_over_R_med = data_brightest_sampled['D_rms']/data_brightest_sampled['R_med']
    D_sph = to_degree(data_brightest_sampled[f'D_sph_{k}'])

    D_rms_over_R_med_med = np.median(D_rms_over_R_med)
    D_sph_med = np.median(D_sph)

    D_rms_over_R_med_errorbar = [[np.abs(np.percentile(D_rms_over_R_med, 2.5)-D_rms_over_R_med_med)], [np.abs(np.percentile(D_rms_over_R_med, 97.5)-D_rms_over_R_med_med)]]
    D_sph_errorbar = [[np.abs(np.percentile(D_sph, 2.5)-D_sph_med)], [np.abs(np.percentile(D_sph, 97.5)-D_sph_med)]]

    ax.errorbar(D_rms_over_R_med_med, D_sph_med, xerr=D_rms_over_R_med_errorbar, yerr=D_sph_errorbar, fmt='.', color='y', label='brightest with surv_probs')

  if is_heaviest:
    data_heaviest = read_specific(data, type='heaviest', select_by_Rvir=select_by_Rvir, seed=seed)
    data_heaviest_without_surv_probs = read_specific(data, type='heaviest', is_surv_probs=False, seed=seed, select_by_Rvir=select_by_Rvir)

    poles_heaviest = data_heaviest['poles']
    poles_heaviest_without_surv_probs = data_heaviest_without_surv_probs['poles']

    chosen_poles_heaviest = poles_heaviest[indices]
    chosen_poles_heaviest_without_surv_probs = poles_heaviest_without_surv_probs[indices]

    ax.scatter(get_D_rms(data_heaviest['pos'])['D_rms']/get_R_med(data_heaviest['pos'])['R_med'], 
      to_degree(np.min(get_D_sph(chosen_poles_heaviest)['D_sph'])), label='heaviest', marker='x', c='g', s=30)
    ax.scatter(get_D_rms(data_heaviest_without_surv_probs['pos'])['D_rms']/get_R_med(data_heaviest_without_surv_probs['pos'])['R_med'], 
      to_degree(np.min(get_D_sph(chosen_poles_heaviest_without_surv_probs)['D_sph'])), label='heaviest without surv_probs', marker='x', c='k', s=30)

  
  ax.set_xlabel("$\\Delta_{\\textrm{rms}}/R_{\\textrm{med}}$")
  ax.set_ylabel("$\\Delta_{\\textrm{sph}} $"+"$(^{\\circ})$")

  ax.legend()
  if is_surv_probs:
    ax.set_title(f"histogram of D_rms/D_med and D_sph({k}) with surv_probs of {suite_name}")
    if saveimage:
      plt.savefig(f"{save_dir}/hist_D_rms_over_R_med_vs_D_sph_{k}_surv_probs_for_{suite_name}.pdf")
  else:
    ax.set_title(f"histogram of D_rms/D_med and D_sph({k}) of {suite_name}")
    if saveimage:
      plt.savefig(f"{save_dir}/hist_D_rms_over_R_med_vs_D_sph_{k}_for_{suite_name}.pdf")

def plot_hist_D_rms_vs_D_sph(suite_name, data_dir, data, brightest_dir=None, is_surv_probs=False, 
                    is_heaviest=False, k=11, select_by_Rvir=False, seed=None, save_dir="", ax=None, saveimage=False):
  """histogram of D_rms vs D_sph

  Parameters
  ----------
  suite_name : str
      name of the suite
  data_dir : str
      directory of the generated data
  data : dataframe 
      dataframe containing relevant attributes
  brightest_dir : str, optional
      directory of the generated data for brightest subhalos with surv_probs, by default None
  is_surv_probs : bool, optional
      True if the sample of size k uses surv_probs, by default False
  is_heaviest : bool, optional
      include  heaviest if True, by default False
  k : int, optional
      number of subhalos, by default 11
  select_by_Rvir : bool, optional
      True if choosing subhalos inside Rvir, by default True
  seed : _type_, optional
      set random seed, by default None
  save_dir : str, optional
      save directory of the plot, by default ""
  ax : Axes, optional
      Axes of the plot, by default None
  saveimage : bool, optional
      save image if True, by default False
  """
  fig, ax = plt.subplots(figsize=(8,6))

  dic = pd.read_csv(f"{data_dir}/{suite_name}.csv", usecols=['D_rms','R_med',f'D_sph_{k}'])

  D_rms = dic['D_rms']
  D_sph = to_degree(dic[f'D_sph_{k}'])

  data_MW = get_MW(is_D_rms=True, is_R_med=True, num_D_sph=k, num_D_sph_flipped=None)
  
  data_brightest_without_surv_probs = read_specific(data, type='brightest', is_surv_probs=False, seed=seed, select_by_Rvir=select_by_Rvir)

  poles_brightest_without_surv_probs = data_brightest_without_surv_probs['poles']

  D_rms_MW = data_MW['D_rms']
  D_sph_MW = to_degree(data_MW['D_sph'])

  xmin = min(np.min(D_rms), D_rms_MW)
  xmax = max(np.max(D_rms), D_rms_MW)
  ymin = min(np.min(D_sph), D_sph_MW)
  ymax = max(np.max(D_sph), D_sph_MW)

  indices = np.array(list(combinations(np.arange(11),k)))

  chosen_poles_brightest_without_surv_probs = poles_brightest_without_surv_probs[indices]

  plot_2d_dist(D_rms, D_sph, [xmin, xmax], [ymin, ymax], 50, 50, figsize=(5,5),fig_setup=ax,clevs=[0.6827, 0.9545, 0.9973])
  ax.scatter(D_rms_MW, D_sph_MW, label='MW', marker='x', c='orangered', s=100)
  ax.scatter(get_D_rms(data_brightest_without_surv_probs['pos'])['D_rms'], 
      to_degree(np.min(get_D_sph(chosen_poles_brightest_without_surv_probs)['D_sph'])), label='brightest without surv_probs', marker='x', c='c', s=100)

  if brightest_dir is None:
    data_brightest = read_specific(data, type='brightest', select_by_Rvir=select_by_Rvir, seed=seed)
    poles_brightest = data_brightest['poles']
    chosen_poles_brightest = poles_brightest[indices]

    ax.scatter(get_D_rms(data_brightest['pos'])['D_rms'], 
      to_degree(np.min(get_D_sph(chosen_poles_brightest)['D_sph'])), label='brightest', marker='x', c='y', s=100)
    
  else:
    data_brightest_sampled = pd.read_csv(f'{brightest_dir}/{suite_name}.csv', usecols=['D_rms', f'D_sph_{k}'])

    D_rms = data_brightest_sampled['D_rms']
    D_sph = to_degree(data_brightest_sampled[f'D_sph_{k}'])

    D_rms_med = np.median(D_rms)
    D_sph_med = np.median(D_sph)

    D_rms_errorbar = [[np.abs(np.percentile(D_rms, 2.5)-D_rms_med)], [np.abs(np.percentile(D_rms, 97.5)-D_rms_med)]]
    D_sph_errorbar = [[np.abs(np.percentile(D_sph, 2.5)-D_sph_med)], [np.abs(np.percentile(D_sph, 97.5)-D_sph_med)]]

    ax.errorbar(D_rms_med, D_sph_med, xerr=D_rms_errorbar, yerr=D_sph_errorbar, fmt='.', color='y', label='brightest with surv_probs')

  if is_heaviest:
    data_heaviest = read_specific(data, type='heaviest', select_by_Rvir=select_by_Rvir, seed=seed)
    data_heaviest_without_surv_probs = read_specific(data, type='heaviest', is_surv_probs=False, seed=seed, select_by_Rvir=select_by_Rvir)

    poles_heaviest = data_heaviest['poles']
    poles_heaviest_without_surv_probs = data_heaviest_without_surv_probs['poles']

    chosen_poles_heaviest = poles_heaviest[indices]
    chosen_poles_heaviest_without_surv_probs = poles_heaviest_without_surv_probs[indices]

    ax.scatter(get_D_rms(data_heaviest['pos'])['D_rms'], 
      to_degree(np.min(get_D_sph(chosen_poles_heaviest)['D_sph'])), label='heaviest', marker='x', c='g', s=30)
    ax.scatter(get_D_rms(data_heaviest_without_surv_probs['pos'])['D_rms'], 
      to_degree(np.min(get_D_sph(chosen_poles_heaviest_without_surv_probs)['D_sph'])), label='heaviest without surv_probs', marker='x', c='k', s=30)

  ax.set_xlabel("$\\Delta_{\\textrm{rms}} (\\textrm{kpc})$")
  ax.set_ylabel("$\\Delta_{\\textrm{sph}} $"+("$(^{\\circ})$"))

  ax.legend()
  if is_surv_probs:
    ax.set_title(f"histogram of D_rms and D_sph({k}) with surv_probs of {suite_name}")
    if saveimage:
      fig.savefig(f"{save_dir}/hist_D_rms_vs_D_sph_{k}_surv_probs_for_{suite_name}.pdf")
  else:
    ax.set_title(f"histogram of D_rms and D_sph({k}) of {suite_name}")
    if saveimage:
      fig.savefig(f"{save_dir}/hist_D_rms_vs_D_sph_{k}_for_{suite_name}.pdf")
