import numpy as np
from tqdm.notebook import tqdm
import astropy.units as u
from astropy.cosmology import WMAP7
from .spline import spline, new_spline
import csv
import matplotlib.pyplot as plt
from modules.spline import *
from modules.stats import get_smallest_rms, get_rms_poles
import pandas as pd

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

def solve(diff_time, diff_X, diff_Y, diff_Z, coef_Rvir_0, stol = 10e-9):
  """
  solve for time t at which the subhalo enters the virial radius by Newton's method
  """
  ctime = diff_time/2
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

def newSolve(diff_time, diff_X, diff_Y, diff_Z, coef_Rvir_0, stol = 10e-10):
  """
  solve for time t at which the subhalo enters the virial radius by binary search
  """
  start_time = 0
  end_time = diff_time

  start_val = eval_diff_sq_dist(start_time, diff_X, diff_Y, diff_Z, coef_Rvir_0)
  end_val = eval_diff_sq_dist(end_time, diff_X, diff_Y, diff_Z, coef_Rvir_0)

  while abs(end_time - start_time) >= stol:
    mid_time = (start_time + end_time) / 2
    mid_val = eval_diff_sq_dist(mid_time, diff_X, diff_Y, diff_Z, coef_Rvir_0)
    if mid_val <= 0:
      start_time = mid_time
      start_val = mid_val
    else:
      end_time = mid_time
      end_val = mid_val
  return start_time

def is_parent(ID0, pID):
  n0 = len(ID0)
  n = len(pID)
  m = min(n0, n)
  result = []
  for i in range(m):
    result.append(ID0[i] == pID[i])
  return np.array(result)

def dist(j, X, Y, Z, X0, Y0, Z0):
  return (X[j]-X0[j])**2 + (Y[j]-Y0[j])**2 + (Z[j]-Z0[j]) **2

def extract_data(data, row, isVel = True, isID = False, ispID = False, isMvir = False, isRvir = False, isCoefsPos = False, isCoefsMvir = False, isCoefsRvir = False):
  non_zero = data['Mvir'][row] > 0
  result = []
  
  scale = data['scale'][row][non_zero]
  X = data['X'][row][non_zero]
  Y = data['Y'][row][non_zero]
  Z = data['Z'][row][non_zero]
  Vx = -np.array((data['Vx'][row][non_zero] * u.km/u.s).to(u.Mpc/u.Gyr))/scale
  Vy = -np.array((data['Vy'][row][non_zero] * u.km/u.s).to(u.Mpc/u.Gyr))/scale
  Vz = -np.array((data['Vz'][row][non_zero] * u.km/u.s).to(u.Mpc/u.Gyr))/scale
  
  zreds = 1/scale-1
  lookback_time = np.array(WMAP7.lookback_time(zreds))
  
  result = [scale, lookback_time, X, Y, Z]
  if isVel:
    result += [Vx, Vy, Vz]
  
  if isID:
      ID = data['ID'][row][non_zero]
      result.append(ID)
  
  if ispID:
      pID = data['pID'][row][non_zero]
      result.append(pID)
  
  if isMvir or isCoefsMvir:
      Mvir = data['Mvir'][row][non_zero]
      result.append(Mvir)
      
  if isRvir or isCoefsRvir:
      Rvir = data['Rvir'][row][non_zero] / 1000
      result.append(Rvir)
  
  if isCoefsPos:
      coefs_X = np.array(list(new_spline(lookback_time, X, Vx)))
      coefs_Y = np.array(list(new_spline(lookback_time, Y, Vy)))
      coefs_Z = np.array(list(new_spline(lookback_time, Z, Vz)))
      result.append(coefs_X)
      result.append(coefs_Y)
      result.append(coefs_Z)
      
  if isCoefsMvir:
      coefs_Mvir = np.array(list(spline(lookback_time, Mvir)))
      result.append(coefs_Mvir)

  if isCoefsRvir:
      coefs_Rvir = np.array(list(spline(lookback_time, Rvir)))
      result.append(coefs_Rvir)
      
  return result

def find_time_of_accretion(data, suite_name):
  _, lookback_time0, X0, Y0, Z0, _, _, _, ID0, Rvir0, coefs_X0, coefs_Y0, coefs_Z0, coefs_Rvir0 = extract_data(data, 0, isID = True, isCoefsPos = True, isCoefsRvir = True)
  num_halos = get_num_halos(data)
  mass_cutoff = 5e8

  with open(f"timedata/isolated/{suite_name}.csv", "a+", newline='') as file:
    writer = csv.writer(file, delimiter=",")
    writer.writerow(["row", "accretion_time"])

  for i in range(1, num_halos):
    _, _, X, Y, Z, _, _, _, pID, Mvir, coefs_X, coefs_Y, coefs_Z = extract_data(data, i, ispID = True, isCoefsPos = True, isMvir = True)
    temp = is_parent(ID0, pID)
    if temp.any() and Mvir.max() > mass_cutoff:
      j = len(temp)
  
      while j > 0 and dist(j-1, X, Y, Z, X0, Y0, Z0) >= Rvir0[j-1] ** 2:
        j = j-1
      if j == 0 or j == len(temp):
        break
      
      coef_X0 = coefs_X0[:,j-1]
      coef_Y0 = coefs_Y0[:,j-1]
      coef_Z0 = coefs_Z0[:,j-1]
      
      coef_X = coefs_X[:,j-1]
      coef_Y = coefs_Y[:,j-1]
      coef_Z = coefs_Z[:,j-1]
  
      coef_Rvir0 = coefs_Rvir0[:,j-1]
      
      diff_X = coef_X0 - coef_X
      diff_Y = coef_Y0 - coef_Y
      diff_Z = coef_Z0 - coef_Z
      
      diff_time = lookback_time0[j]-lookback_time0[j-1]
      
      t = solve(diff_time, diff_X, diff_Y, diff_Z, coef_Rvir0, stol = 10e-10)

      if t < 0 or t > diff_time:
        break

      t += lookback_time0[j-1]

      with open(f"timedata/isolated/{suite_name}.csv", "a+", newline='') as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow([i, t])

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
  for iv, var in enumerate(varnames):
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

def prep_data(data, i, arr_row):
  ## x,y,z in physical Mpc\
  arr_row = list(arr_row)
  mass = data['Mvir'].T[i][[0]+arr_row]
  non_zero_index = mass > 0
  scale = data['scale'].T[i][[0]+arr_row][non_zero_index]
  x = data['X'].T[i][[0]+arr_row][non_zero_index]*scale
  y = data['Y'].T[i][[0]+arr_row][non_zero_index]*scale
  z = data['Z'].T[i][[0]+arr_row][non_zero_index]*scale
  rvir = data['Rvir'].T[i][[0]+arr_row][non_zero_index]*scale
  mass = mass[non_zero_index]
  return mass, x, y, z, rvir, scale

def get_arrays(suite_name, type_suite, data, halo_row):
  df = pd.read_csv(f'timedata/{type_suite}/{suite_name}.csv')
  # array row of halos that 
  arr_row = np.array(df['row'])

  # array of accretion time
  arr_time = np.array(df['accretion_time'])

  # array of pos and vec in angular coordinates
  arr_ang_pos_acc = []
  arr_ang_vec_acc = []

  arr_ang_pos_cur = []
  arr_ang_vec_cur = []

  # array of pos and vec in cartesian coordinates
  arr_pos_acc = []
  arr_vec_acc = []

  arr_pos_cur = []
  arr_vec_cur = []

  arr_pos_cur_dis = []
  arr_vec_cur_dis = []

  # array of accretion mass
  arr_mass_acc = []
  arr_mass_cur = []

  _, lookback_time0, X0, Y0, Z0, Vx0, Vy0, Vz0, coefs_X0, coefs_Y0, coefs_Z0, = extract_data(data, halo_row, isCoefsPos = True)
  for i, t in zip(arr_row, arr_time):
    _, lookback_time, X, Y, Z, Vx, Vy, Vz, Mvir, coefs_X, coefs_Y, coefs_Z, coefs_Mvir= extract_data(data, i, isCoefsPos = True, isCoefsMvir = True)
    # relative position of subhalo in the host frame at accretion
    x = eval_new_spline(t, lookback_time, *coefs_X) - eval_new_spline(t, lookback_time0, *coefs_X0)
    y = eval_new_spline(t, lookback_time, *coefs_Y) - eval_new_spline(t, lookback_time0, *coefs_Y0)
    z = eval_new_spline(t, lookback_time, *coefs_Z) - eval_new_spline(t, lookback_time0, *coefs_Z0)
    # relative velocity of subhalo in the host frame at accretion
    vx = eval_der_new_spline(t, lookback_time, *coefs_X[1:]) - eval_der_new_spline(t, lookback_time0, *coefs_X0[1:])
    vy = eval_der_new_spline(t, lookback_time, *coefs_Y[1:]) - eval_der_new_spline(t, lookback_time0, *coefs_Y0[1:])
    vz = eval_der_new_spline(t, lookback_time, *coefs_Z[1:]) - eval_der_new_spline(t, lookback_time0, *coefs_Z0[1:])

    m = eval_spline(t, lookback_time, *coefs_Mvir)
    
    arr_mass_acc.append(m)
    arr_mass_cur.append(Mvir[halo_row])
    
    arr_ang_pos_acc.append(to_spherical(x,y,z))
    arr_ang_vec_acc.append(to_spherical(-vx, -vy, -vz))
    arr_ang_pos_cur.append(to_spherical(X[halo_row]-X0[halo_row], Y[halo_row]-Y0[halo_row], Z[halo_row]-Z0[halo_row]))
    arr_ang_vec_cur.append(to_spherical(-Vx[halo_row]+Vx0[halo_row], -Vy[halo_row]+Vy0[halo_row], -Vz[halo_row]+Vz0[halo_row]))

    arr_pos_acc.append(to_direction(x,y,z))
    arr_vec_acc.append(to_direction(-vx,-vy,-vz))
    arr_pos_cur.append(to_direction(X[halo_row]-X0[halo_row], Y[halo_row]-Y0[halo_row], Z[halo_row]-Z0[halo_row]))
    arr_vec_cur.append(to_direction(-Vx[halo_row]+Vx0[halo_row], -Vy[halo_row]+Vy0[halo_row], -Vz[halo_row]+Vz0[halo_row]))

    arr_pos_cur_dis.append([X[halo_row]-X0[halo_row], Y[halo_row]-Y0[halo_row], Z[halo_row]-Z0[halo_row]])
    arr_vec_cur_dis.append([-Vx[halo_row]+Vx0[halo_row], -Vy[halo_row]+Vy0[halo_row], -Vz[halo_row]+Vz0[halo_row]])

  arr_pos_acc = np.array(arr_pos_acc)
  arr_vec_acc = np.array(arr_vec_acc)
  arr_ang_pos_acc = np.array(arr_ang_pos_acc)
  arr_ang_vec_acc = np.array(arr_ang_vec_acc)

  arr_pos_cur = np.array(arr_pos_cur)
  arr_vec_cur = np.array(arr_vec_cur)
  arr_ang_pos_cur = np.array(arr_ang_pos_cur)
  arr_ang_vec_cur = np.array(arr_ang_vec_cur)

  arr_mass_acc = np.array(arr_mass_acc)
  arr_mass_cur = np.array(arr_mass_cur)

  arr_pos_cur_dis = np.array(arr_pos_cur_dis)
  arr_vec_cur_dis = np.array(arr_vec_cur_dis)

  return arr_row, arr_time, arr_pos_acc, arr_vec_acc, arr_ang_pos_acc, arr_ang_vec_acc, arr_pos_cur, arr_vec_cur, arr_ang_pos_cur, arr_ang_vec_cur, arr_pos_cur_dis, arr_vec_cur_dis, arr_mass_acc, arr_mass_cur

def extract_inside_at_timestep(suite_name, halo_row, data, timestep = 0, isVel=False, timedata_dir="timedata/isolated"):
  df = pd.read_csv(timedata_dir+f'/{suite_name}.csv')
  arr_row = np.array(df['row'])
  
  _, _, X0, Y0, Z0, Rvir0 = extract_data(data, halo_row, isVel = False, isRvir = True)

  X = (data['X'][arr_row][:,timestep] - X0[timestep]) / Rvir0[timestep]
  Y = (data['Y'][arr_row][:,timestep] - Y0[timestep]) / Rvir0[timestep]
  Z = (data['Z'][arr_row][:,timestep] - Z0[timestep]) / Rvir0[timestep]

  inside_index = (X**2 + Y**2 + Z**2 < 1)

  new_X = X[inside_index]
  new_Y = Y[inside_index]
  new_Z = Z[inside_index]

  if isVel:
    Vx = data['Vx'][arr_row][:,timestep] - data['Vx'][halo_row][timestep]
    Vy = data['Vy'][arr_row][:,timestep] - data['Vx'][halo_row][timestep]
    Vz = data['Vz'][arr_row][:,timestep] - data['Vx'][halo_row][timestep]

    new_Vx = Vx[inside_index]
    new_Vy = Vy[inside_index]
    new_Vz = Vz[inside_index]
    return inside_index, np.array([new_X, new_Y, new_Z]).T, np.array([new_Vx, new_Vy, new_Vz]).T

  return inside_index, np.array([new_X, new_Y, new_Z]).T

def extract_poles_inside_at_timestep(suite_name, halo_row, data, timestep=0, timedata_dir="timedata/isolated"):
  _, pos, vec = extract_inside_at_timestep(suite_name, halo_row, data, timestep=timestep, isVel=True, timedata_dir=timedata_dir)
  poles = np.cross(pos, vec)
  temp = np.sum(poles**2, axis=1)**(1/2)
  poles = (poles.T/temp).T
  return poles

def read_MW():
  MW = pd.read_csv('../../Data/pawlowski_tab2.csv')

  X_MW = MW['x']
  Y_MW = MW['y']
  Z_MW = MW['z']
  Vx_MW = MW['vx']
  Vy_MW = MW['vy']
  Vz_MW = MW['vz']

  return X_MW, Y_MW, Z_MW, Vx_MW, Vy_MW, Vz_MW

def get_rms_MW():
  X_MW, Y_MW, Z_MW, _, _, _ = read_MW()

  pos_MW = np.array([X_MW, Y_MW, Z_MW]).T
  r_MW = np.sum(pos_MW**2, axis=1)**(1/2)

  return get_smallest_rms(pos_MW)/np.median(r_MW)

def get_rms_poles_MW():
  X_MW, Y_MW, Z_MW, Vx_MW, Vy_MW, Vz_MW = read_MW()
  pos = np.array([X_MW, Y_MW, Z_MW]).T
  vec = np.array([Vx_MW, Vy_MW, Vz_MW]).T
  poles = np.cross(pos, vec)
  temp = (np.sum(poles**2, axis = 1))**(1/2)
  poles = (poles.T/temp).T

  return get_rms_poles(poles)