from itertools import combinations
import numpy as np
import astropy.units as u
import csv
import pandas as pd
from string import Template
import random

from modules.stats_v3 import get_D_sph, get_smallest_D_rms, get_smallest_D_sph_flipped

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

def to_spherical(x: float, y: float, z: float)->tuple[float, float]:
  """find [theta, phi] in [-pi/2,pi/2]x[-pi,pi] where
    x = r*cos(theta)*cos(phi)\\
    y = r*cos(theta)*sin(phi)\\
    z = r*sin(theta)\\

  Args:
      x (float): x position
      y (float): y position
      z (float): z position

  Returns:
      tuple[float, float]: _description_
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
  return angle/np.pi*180

def normalize(vectors: np.ndarray)->np.ndarray:
  """normalize vectors

  Args:
      vectors (np.ndarray): ndarray of shape (n, 3)

  Returns:
      np.ndarray: ndarray of shape (n, 3)
  """
  # shape (n,)
  r = np.sum(vectors**2, axis=-1)**(1/2)

  # shape (n, 3)
  normal_vectors = ((vectors.T)/r).T

  return normal_vectors

def read_MW():
  """read MW data

  Returns:
      list[np.ndarray]: pos, vec of MW subhalos
  """
  MW = pd.read_csv('../../Data/pawlowski_tab2.csv')

  X_MW = MW['x']
  Y_MW = MW['y']
  Z_MW = MW['z']
  Vx_MW = MW['vx']
  Vy_MW = MW['vy']
  Vz_MW = MW['vz']
  
  pos = np.array([X_MW, Y_MW, Z_MW]).T
  vec = np.array([Vx_MW, Vy_MW, Vz_MW]).T
  poles = np.cross(pos, vec)
  temp = (np.sum(poles**2, axis = 1))**(1/2)
  poles = (poles.T/temp).T

  varnames = ['pos', 'vec', 'poles']
  varinfo = [pos, vec, poles]

  dic = dict(zip(varnames, varinfo))

  return dic

def get_MW(is_D_rms=True, is_R_med=True, num_D_sph=11, num_D_sph_flipped=11):
  dic_MW = read_MW()
  dic = {}

  pos = dic_MW['pos']
  poles = dic_MW['poles']

  num_MW = len(pos)

  if is_D_rms:
    dic['D_rms'] = get_smallest_D_rms(pos)['D_rms']

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

  if num_D_sph_flipped is not None:
    if type(num_D_sph_flipped) is int:
      chosen_poles = poles[np.array(list(combinations(np.arange(num_MW),num_D_sph_flipped)))]
      dic['D_sph_flipped'] = np.min(get_smallest_D_sph_flipped(chosen_poles)['D_sph_flipped'])
    if type(num_D_sph_flipped) is list:
      dic['D_sph_flipped'] = {}
      for num in num_D_sph_flipped:
        chosen_poles = poles[np.array(list(combinations(np.arange(num_MW),num)))]
        dic['D_sph_flipped'][num] = np.min(get_smallest_D_sph_flipped(chosen_poles)['D_sph_flipped'])

  return dic

def read_halo(suite_name_decorated, suite_dir, varnames=None):
  if varnames is None:
    varnames = ['file_name','Mvir','Mpeak','rvir','xpos','ypos','zpos','vx','vy','vz','upID','ID','mV','M200c','Ms','surv_probs']
  filepath = suite_dir+'/'+suite_name_decorated
  data = pd.read_csv(filepath, usecols=varnames)

  def rescale_vec(v):
    return np.array((v*u.km/u.s).to(u.kpc/u.Gyr))
  
  data['vx'] = data['vx'].apply(rescale_vec)
  data['vy'] = data['vy'].apply(rescale_vec)
  data['vz'] = data['vz'].apply(rescale_vec)

  def new_rescale_vec(v, v0):
    return v - v0

  data['vx'] = data['vx'].apply(lambda v: new_rescale_vec(v, data['vx'][0]))
  data['vy'] = data['vy'].apply(lambda v: new_rescale_vec(v, data['vy'][0]))
  data['vz'] = data['vz'].apply(lambda v: new_rescale_vec(v, data['vz'][0]))

  def rescale_pos(x):
    return x * 1000

  data['xpos'] = data['xpos'].apply(rescale_pos)
  data['ypos'] = data['ypos'].apply(rescale_pos)
  data['zpos'] = data['zpos'].apply(rescale_pos)

  return data.rename(columns={'xpos': 'X', 'ypos': 'Y', 'zpos': 'Z', 'vx': 'Vx', 'vy': 'Vy', 'vz': 'Vz', 'rvir': 'Rvir'})

def extract_inside(data, mass_cutoff = 5e8, select_by_Rvir=True, is_comp_parent = False):
  rad_selected = data['Rvir'][0] if select_by_Rvir else 300

  X = data['X'][1:]
  Y = data['Y'][1:]
  Z = data['Z'][1:]

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


