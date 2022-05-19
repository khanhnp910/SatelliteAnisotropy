from itertools import combinations
import numpy as np
import astropy.units as u
import csv
import pandas as pd
from string import Template
import random
import config

from modules.stats_v3 import get_D_sph, get_D_rms, get_D_sph_flipped, get_R_med, random_choice_noreplace

elvis_name_template = Template('${suite_name}_isolated_elvis_iso_zrei8_etan_18_etap_045_run.csv')
caterpillar_name_template = Template('${suite_name}_LX14_zrei8_5_fix_run.csv')

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

def to_spherical(x, y, z):
  """convert vectors to spherical coordinate for aitoff plots

  Parameters
  ----------
  x : float
      x position
  y : float
      y position
  z : float
      z position

  Returns
  -------
  tuple
      theta, phi angles
  """
  theta = np.arctan(z/(x**2+y**2)**(1/2))
  phi = np.arctan(y/x)
  if x > 0:
    if phi > 0:
      phi -= np.pi
    else:
      phi += np.pi
  return theta, phi

def to_degree(angle):
  """convert rad to degree

  Parameters
  ----------
  angle : float
      angle in rad

  Returns
  -------
  float
      angle in degrees
  """
  return angle/np.pi*180

def normalize(vectors):
  """normalize vectors

  Parameters
  ----------
  vectors : array
      vectors shape (a_1, ..., a_l, 3)

  Returns
  -------
  array
      normalized vectors of shape (a_1, ..., a_l, 3)
  """
  # shape (a_1, ..., a_l)
  r = np.sum(vectors**2, axis=-1)**(1/2)

  # shape (a_1, ..., a_l, 3)
  normal_vectors = np.moveaxis(np.moveaxis(vectors, -1, 0)/r, 0, -1)

  return normal_vectors

def read_MW(MW_path=config.MW_path):
  """_summary_

  Parameters
  ----------
  MW_path : str, optional
      path to MW data, by default '../../Data/pawlowski_tab2.csv'

  Returns
  -------
  dictionary
      dic['pos']: pos
      dic['vec']: vec
      dic['poles']: poles
  """
  MW = pd.read_csv(MW_path)
  
  pos = np.array([MW['x'], MW['y'], MW['z']]).T
  vec = np.array([MW['vx'], MW['vy'], MW['vz']]).T
  poles = np.cross(pos, vec)
  temp = (np.sum(poles**2, axis = 1))**(1/2)
  poles = (poles.T/temp).T

  varnames = ['pos', 'vec', 'poles']
  varinfo = [pos, vec, poles]

  dic = dict(zip(varnames, varinfo))

  return dic

def get_MW(MW_path=config.MW_path, is_D_rms=True, is_R_med=True, num_D_sph=11, num_D_sph_flipped=11):
  """get relevant quantities for MW

  if num_D_sph is int, dic['D_sph'] is D_sph for num_D_sph
  if num_D_sph is a tuple (k_1, k_2, ...), dic['D_sph'] is a dictionary where dic['D_sph'][k_i] is the smallest D_sph for a subset of k_i satellites

  Parameters
  ----------
  MW_path : str, optional
      path to MW data, by default '../../Data/pawlowski_tab2.csv'
  is_D_rms : bool, optional
      return D_rms if true, by default True
  is_R_med : bool, optional
      return R_med if true, by default True
  num_D_sph : int or list, optional
      return D_sph for specific num_D_sph, by default 11
  num_D_sph_flipped : int or list, optional
      return D_sph_flipped for specific num_D_sph_flipped, by default 11

  Returns
  -------
  dictionary
      dic['D_rms']: float, D_rms of MW satellites
      dic['R_med']: float, R_med of MW satellites
      dic['D_sph']: float or dic, D_sph of MW satellites
      dic['D_sph_flipped']: float or dic, D_sph_flipped of MW satellites
  """
  dic_MW = read_MW(MW_path)
  dic = {}

  pos = dic_MW['pos']
  poles = dic_MW['poles']

  num_MW = len(pos)

  if is_D_rms:
    dic['D_rms'] = get_D_rms(pos)['D_rms']

  if is_R_med:
    dic['R_med'] = np.median(np.sum(pos**2, axis=-1)**(1/2))

  if num_D_sph is not None:
    if type(num_D_sph) is int:
      chosen_poles = poles[np.array(list(combinations(np.arange(num_MW),num_D_sph)))]
      dic['D_sph'] = np.min(get_D_sph(chosen_poles)['D_sph'])
    if type(num_D_sph) is list:
      dic['D_sph'] = {}
      for num in num_D_sph:
        chosen_poles = poles[np.array(list(combinations(np.arange(num_MW),num)))]
        dic['D_sph'][num] = np.min(get_D_sph(chosen_poles)['D_sph'])
        dic[f'D_sph_{num}'] = dic['D_sph'][num]

  if num_D_sph_flipped is not None:
    if type(num_D_sph_flipped) is int:
      chosen_poles = poles[np.array(list(combinations(np.arange(num_MW),num_D_sph_flipped)))]
      dic['D_sph_flipped'] = np.min(get_D_sph_flipped(chosen_poles)['D_sph_flipped'])
    if type(num_D_sph_flipped) is list:
      dic['D_sph_flipped'] = {}
      for num in num_D_sph_flipped:
        chosen_poles = poles[np.array(list(combinations(np.arange(num_MW),num)))]
        dic['D_sph_flipped'][num] = np.min(get_D_sph_flipped(chosen_poles)['D_sph_flipped'])
        dic[f'D_sph_flipped_{num}'] = dic['D_sph_flipped'][num]

  return dic

def read_halo(suite_name_decorated, suite_dir, varnames=None):
  """read data for specific host halo

  Parameters
  ----------
  suite_name_decorated : string
      file name for suite_name
  suite_dir : string
      directory to suite_name data
  varnames : list, optional
      names of variables to read, by default None

  Returns
  -------
  dataframe
      dataframe with varnames attributes
  """
  if varnames is None:
    varnames = ['file_name','Mvir','Mpeak','rvir','xpos','ypos','zpos','vx','vy','vz','upID','ID','mV','M200c','Ms','surv_probs']
  filepath = f'{suite_dir}/{suite_name_decorated}'
  data = pd.read_csv(filepath, usecols=varnames)

  def rescale_vec(v):
    return np.array((v*u.km/u.s).to(u.kpc/u.Gyr))
  
  # velocity in kpc/Gyr
  data['vx'] = data['vx'].apply(rescale_vec)
  data['vy'] = data['vy'].apply(rescale_vec)
  data['vz'] = data['vz'].apply(rescale_vec)

  def new_rescale_vec(v, v0):
    return v - v0

  # velocity relative to host halo
  data['vx'] = data['vx'].apply(lambda v: new_rescale_vec(v, data['vx'][0]))
  data['vy'] = data['vy'].apply(lambda v: new_rescale_vec(v, data['vy'][0]))
  data['vz'] = data['vz'].apply(lambda v: new_rescale_vec(v, data['vz'][0]))

  def rescale_pos(x):
    return x * 1000

  # position in kpc
  data['xpos'] = data['xpos'].apply(rescale_pos)
  data['ypos'] = data['ypos'].apply(rescale_pos)
  data['zpos'] = data['zpos'].apply(rescale_pos)

  return data.rename(columns={'xpos': 'X', 'ypos': 'Y', 'zpos': 'Z', 'vx': 'Vx', 'vy': 'Vy', 'vz': 'Vz', 'rvir': 'Rvir'})

def extract_inside(data, mass_cutoff = config.MASS_CUTOFF, select_by_Rvir=True, is_comp_parent = False):
  """extract subhalos inside Rvir or 300kpc 

  Parameters
  ----------
  data : dataframe
      dataframe containing relevant attributes
  mass_cutoff : float, optional
      mass cutoff for halo selection, by default 5e8
  select_by_Rvir : bool, optional
      True if choosing subhalos inside Rvir, by default True
  is_comp_parent : bool, optional
      True if comparing host ID with subhalo upID, by default False

  Returns
  -------
  dictionary
      dictionary of pos, vec, poles, mV, Mpeak, Ms, surv_probs
  """
  rad_selected = data['Rvir'][0] if select_by_Rvir else 300

  X = data['X'][1:]
  Y = data['Y'][1:]
  Z = data['Z'][1:]

  # indices of subhalos inside shifted below by 1
  inside_index = np.logical_and((X**2 + Y**2 + Z**2 < rad_selected**2), data['Mpeak'][1:]>mass_cutoff)

  if is_comp_parent:
    inside_index = np.logical_and(inside_index, data['upID'][1:]==data['ID'][0])

  new_X = X[inside_index]
  new_Y = Y[inside_index]
  new_Z = Z[inside_index]

  new_Vx = data['Vx'][1:][inside_index]
  new_Vy = data['Vy'][1:][inside_index]
  new_Vz = data['Vz'][1:][inside_index]

  pos = np.array([new_X, new_Y, new_Z]).T
  vec = np.array([new_Vx, new_Vy, new_Vz]).T

  poles = np.cross(pos, vec)
  temp = np.sum(poles**2, axis=1)**(1/2)
  poles = (poles.T/temp).T

  varnames = ['pos', 'vec', 'poles', 'mV', 'Mpeak', 'Ms', 'surv_probs']
  varinfo = [pos, vec, poles, np.array(data['mV'][1:][inside_index]), np.array(data['Mpeak'][1:][inside_index]), np.array(data['Ms'][1:][inside_index]), np.array(data['surv_probs'][1:][inside_index])]

  dic_extracted = dict(zip(varnames, varinfo))

  return dic_extracted

def read_specific(data, type='brightest', is_surv_probs=True, num_chosen=11, select_by_Rvir=True, seed=None, is_comp_parent = False):
  """get num_chosen brightest/heaviest subhalos

  if the is_surv_probs is True, the sammple is generated as followed
    init: chosen_indices = [], i=0\\
    sort the subhalos in decreasing in relevance (more relevant first)\\
    while len(chosen_indices) < num_chosen:
      randomly decide if the i-th subhalo is chosen based on surv_probs\\
      if it is, add to i-th index to chosen_indices, else continue\\
      i = i + 1

  Parameters
  ----------
  data : dataframe
      dataframe containing relevant attributes
  type : str, optional
      brightest or heaviest, by default 'brightest'
  is_surv_probs : bool, optional
      True if the sample of size num_chosen uses surv_probs, by default True
  num_chosen : int, optional
      number of chosen subhalos, by default 11
  select_by_Rvir : bool, optional
      True if choosing subhalos inside Rvir, by default True
  seed : int, optional
      set random seed, by default None
  is_comp_parent : bool, optional
      True if comparing host ID with subhalo upID, by default False

  Returns
  -------
  dictionary
      dictionary of relevant quantities

  Raises
  ------
  ValueError
      raise error if type is not supported
  """
  dic_extracted = extract_inside(data, select_by_Rvir=select_by_Rvir, is_comp_parent = is_comp_parent)
  if type == 'brightest':
    indices = np.argsort(dic_extracted['mV'])
  elif type == 'heaviest':
    indices = np.argsort(dic_extracted['Mpeak'])[::-1]
  else:
    raise ValueError('type not supported')

  if seed is not None:
    random.seed(seed)

  chosen_indices = []

  i = 0
  if is_surv_probs:
    while len(chosen_indices) < num_chosen:
      index = indices[i]
      random_state = random.random()

      if random_state < dic_extracted['surv_probs'][index]:
        chosen_indices.append(index)

      i += 1
  else:
    chosen_indices = indices[:num_chosen]
  
  new_dic_extracted = {'names': type}

  for attr in dic_extracted.keys():
    new_dic_extracted[attr] = dic_extracted[attr][chosen_indices]

  return new_dic_extracted

def generate_distribution(suite_name_decorated, filename, suite_dir, is_surv_probs=True, iterations=config.ITERATIONS, chunk_size=config.CHUNK_SIZE, select_by_Rvir = False):
  """generate distribution with surv_probs

  Parameters
  ----------
  suite_name_decorated : str
      filename of the suite corresponding to the host halo
  filename : str
      filename of the generated data
  suite_dir : str
      directory to the raw data
  is_surv_probs: bool, optional
      True if the sample of size num_chosen uses surv_probs, by default True
  iterations : int, optional
      number of iterations, by default 250000
  chunk_size : int, optional
      number of samples for each run, by default 200
  select_by_Rvir : bool, optional
      True if choosing subhalos inside Rvir, by default False
  """
  data = read_halo(suite_name_decorated, suite_dir)
  dic = extract_inside(data, select_by_Rvir=select_by_Rvir)

  with open(filename, "w", newline='') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(["D_rms", "R_med", "D_sph_11","D_sph_10","D_sph_9","D_sph_8","D_sph_7","D_sph_6","D_sph_5","D_sph_4","D_sph_3","ID0","ID1","ID2","ID3","ID4","ID5","ID6","ID7","ID8","ID9","ID10"])

  num_time = iterations // chunk_size

  pos = dic['pos']
  poles = dic['poles']
  surv_probs = dic['surv_probs']

  size = len(pos)

  for _ in range(num_time):
    # (size,)
    random_probs = np.random.rand(size)

    if is_surv_probs:
      temp = random_probs < surv_probs

      new_indices = np.arange(size)[temp]
    else:
      new_indices = np.arange(size)

    if len(new_indices) < 11:
      continue

    # (chunk_size, 11)
    indices = new_indices[random_choice_noreplace(chunk_size, len(new_indices), 11)]

    # (chunk_size, 11, 3)
    chosen_pos = pos[indices]

    # (chunk_size, 11, 3)
    chosen_poles = poles[indices]

    d_angles = []
    
    for k in range(11,2,-1):
      # shape (chunk_size, 11, 3)
      indices_ = np.array(list(combinations(np.arange(11),k)))

      chosen_poles_ = chosen_poles[:, indices_,:]

      d_angles.append(np.min(np.around(get_D_sph(chosen_poles_)['D_sph'], decimals=3), axis=-1))

    arr = np.array([np.around(get_D_rms(chosen_pos, num_random=5000)['D_rms'], decimals=3), np.around(get_R_med(chosen_pos)['R_med'], decimals=3)]+d_angles).T
    arr = np.concatenate((arr, indices), axis=1)

    with open(filename, "a+", newline='') as file:
      writer = csv.writer(file, delimiter=',')
      writer.writerows(arr)

def generate_brightest_distribution_with_surv_probs(suite_name_decorated, filename, suite_dir, iterations=config.ITERATIONS_BRIGHTEST, select_by_Rvir = False):
  """generate brightest distribution with surv_probs

  Parameters
  ----------
  suite_name_decorated : str
      filename of the suite corresponding to the host halo
  filename : str
      filename of the generated data
  suite_dir : str
      directory to the raw data
  iterations : int, optional
      number of iterations, by default 250000
  chunk_size : int, optional
      number of samples for each run, by default 200
  select_by_Rvir : bool, optional
      True if choosing subhalos inside Rvir, by default False
  """
  data = read_halo(suite_name_decorated, suite_dir)

  with open(filename, "w", newline='') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(["D_rms", "R_med", "D_sph_11","D_sph_10","D_sph_9","D_sph_8","D_sph_7","D_sph_6","D_sph_5","D_sph_4","D_sph_3"])

  for _ in range(iterations):
    data_brightest = read_specific(data, select_by_Rvir=select_by_Rvir)

    D_rms = get_D_rms(data_brightest['pos'])['D_rms']
    R_med = get_R_med(data_brightest['pos'])['R_med']

    D_sphs = []

    for k in range(11, 2, -1):
      indices = np.array(list(combinations(np.arange(11),k)))

      poles = data_brightest['poles'][indices]

      D_sphs.append(np.around(np.min(get_D_sph(poles)['D_sph']),3))

    with open(filename, "a", newline='') as file:
      writer = csv.writer(file, delimiter=',')
      writer.writerow([np.around(D_rms,3), np.around(R_med,3)]+D_sphs)