import numpy as np
from scipy.special import kolmogorov 

# import time
# import tracemalloc

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

def conf_interval(x, pdf, conf_level):
  return np.sum(pdf[pdf > x])-conf_level

def sample_spherical_angle(size):
  """sample spherical angles phi, theta

  Parameters
  ----------
  size : int or tuple
      shape of the angles (a_1, ..., a_l)

  Returns
  -------
  tuples of arrays
      phi, theta arrays of shape (a_1, ..., a_l)
  """
  # shape of size
  phi = np.random.uniform(size=size) * 2 * np.pi
  theta = np.arccos(1-2*np.random.uniform(size=size))

  return phi, theta

def sample_spherical_pos(size):
  """_summary_

  Parameters
  ----------
  size : int or tuple
      shape of the angles (a_1, ..., a_l)

  Returns
  -------
  array 
      normalized vectors of shape (a_1, ..., a_l, 3)
  """
  # shape of size
  phi, theta = sample_spherical_angle(size=size)

  # shape of size
  X = np.cos(phi) * np.sin(theta)
  Y = np.sin(phi) * np.sin(theta)
  Z = np.cos(theta)

  return np.moveaxis(np.array([X,Y,Z]), 0, -1)

def _get_D_rms(pos, normal_vectors, isCoef = False):
  """calculate the D_rms from pos and normal_vectors

  Ax + By + Cz = D represents the plane of smallest D_rms in the direction (A, B, C)

  Parameters
  ----------
  pos : array
      array of shape (a_1, a_2, ..., a_l, k) where k is dimension
  normal_vectors : array
      array of shape (m, k) of normal vectors considering 
  isCoef : bool, optional
      true if offset coefficient D is returned, by default False

  Returns
  -------
  dictionary
      dic['D_rms']: array of D_rms of shape (a_1, a_2, ..., a_l, m)
      dic['coef']: array of offset coefficient D of shape (a_1, a_2, ..., a_l,m), is returned if isCoef is True
  """
  # shape (a_1, a_2, ..., a_l, n, m)
  prod = np.matmul(pos, normal_vectors.T)
  
  # shape (a_1, a_2, ..., a_l, m)
  coef = np.mean(prod, axis=-2)
  
  # shape (a_1, a_2, ..., a_l, m)
  D_rms = np.mean((np.moveaxis(prod, -2, 0)-coef)**2, axis = 0) ** (1/2)

  dic = {'D_rms': D_rms}
  if isCoef:
    dic['coef'] = coef

  return dic

def get_D_rms(pos, num_random=5000, normal_vectors=None, isCoef=False, isAvg=False):
  """calculate the D_rms from pos

  Ax + By + Cz = D represents the plane of smallest D_rms in the direction (A, B, C)

  Parameters
  ----------
  pos : array
      array of shape (a_1, a_2, ..., a_l, 3)
  num_random : int, optional
      number of random direction generated if normal_vectors is None, by default 5000
  normal_vectors : array, optional
      directions to calculate D_rms, by default None
  isCoef : bool, optional
      true if offset coefficient D is returned, by default False
  isAvg : bool, optional
      true if normal vector (A, B, C) is returned, by default False

  Returns
  -------
  dictionary
      dic['D_rms']: array of D_rms of shape (a_1, a_2, ..., a_l)
      dic['coef']: array of offset coefficient D of shape (a_1, a_2, ..., a_l), is returned if isCoef is True
      dic['avg]: array of normal vectors (A, B, C) of shape (a_1, a_2, ..., a_l, 3), is returned if isAvg is True
  """
  # shape (num_random, 3)
  if normal_vectors is None:
    normal_vectors = sample_spherical_pos(num_random)

  # result dic
  temp_dic = _get_D_rms(pos, normal_vectors, isCoef = isCoef)

  # shape (a_1, a_2, ..., a_l)
  min_d_rms = np.amin(temp_dic['D_rms'], axis=-1)

  dic = {'D_rms': min_d_rms}

  indices = None

  if isCoef:
    # shape (a_1, a_2, ..., a_l)
    shape = min_d_rms.shape
    
    # shape (a_1, a_2, ..., a_l)
    if indices is None:
      indices = np.argmin(temp_dic['D_rms'], axis=-1)

    tuple_shape = []

    for idx, x in enumerate(shape):
      temp_shape = np.ones(len(shape), dtype='int')
      temp_shape[idx] = x

      arange = np.reshape(np.arange(x),temp_shape)

      arange_broadcasted, _ = np.broadcast_arrays(arange, min_d_rms)

      tuple_shape.append(arange_broadcasted)

    # tuple of arrays of shape (a_1, a_2, ..., a_l)
    tuple_shape.append(indices)
    tuple_shape = tuple(tuple_shape)

    # extract offset coefficients
    dic['coef'] = temp_dic['coef'][tuple_shape]

  if isAvg:
    # shape (a_1, a_2, ..., a_l)
    if indices is None:
      indices = np.argmin(temp_dic['D_rms'], axis=-1)

    # shape (a_1, a_2, ..., a_l, k)
    dic['avg'] = normal_vectors[indices,:]

  return dic

def get_R_med(pos):
  """calculate R_med from pos

  Parameters
  ----------
  pos : array
      array of shape (a_1, a_2, ..., a_l, 3)

  Returns
  -------
  dictionary
      dic['R_med']: array of R_med of shape (a_1, a_2, ..., a_l)
  """
  # shape (a_1, a_2, ..., a_l)
  return {"R_med": np.median(np.sum(pos**2, axis=-1)**(1/2), axis=-1)}

def get_D_sph(poles, isAvg = False):
  """calculate D_sph from poles

  Parameters
  ----------
  poles : array
      array of shape (a_1, a_2, ..., a_l, n, k)
  isAvg : bool, optional
      true if average direction is returned, by default False

  Returns
  -------
  dictionary
      dic['D_sph']: array of D_sph of shape (a_1, a_2, ..., a_l)
      dic['avg']: array of average direction of shape (a_1, a_2, ..., a_l, k)
  """
  # shape (a_1, ..., a_l, 3)
  avg = np.sum(poles, axis=-2)

  # shape (a_1, ..., a_l)
  norm = np.sum(avg**2, axis=-1)**(1/2)

  # shape (a_1, ..., a_l, 3)
  avg = np.moveaxis(np.moveaxis(avg, -1, 0)/norm, 0, -1)

  # shape (a_1, ..., a_l)
  D_sph = np.mean(np.arccos(np.sum(np.moveaxis(poles, -2, 0)*avg, axis=-1))**2, axis=0)**(1/2)

  dic = {'D_sph': D_sph}

  if isAvg:
    dic['avg'] = avg

  return dic

def _get_D_sph_flipped(poles, normal_vectors):
  """calculate D_sph_flipped from poles in given direction of normal_vecors

  Parameters
  ----------
  poles : array
      array of shape (a_1, a_2, ..., a_l, n, k)
  normal_vectors : array
      array of shape (m, k) of normal vectors considering 

  Returns
  -------
  dictionary
      dic['D_sph_flipped']: array of D_sph_flipped of shape (a_1, a_2, ..., a_l, m)
  """
  # shape (a_1, ..., a_l, n, m)
  prod = np.abs(np.matmul(poles, normal_vectors.T))

  # shape (a_1, ..., a_l, n, m)
  angles = np.arccos(prod)

  # shape (a_1, ..., a_l, m)
  D_sph_flipped = np.mean(angles**2, axis=-2)**(1/2)

  dic = {'D_sph_flipped': D_sph_flipped}

  return dic

def get_D_sph_flipped(poles, num_random=5000, normal_vectors=None, isAvg=False):
  """calculate D_sph_flipped from poles

  Parameters
  ----------
  poles : array
      array of shape (a_1, a_2, ..., a_l, n, 3)
  num_random : int, optional
      number of random direction generated if normal_vectors is None, by default 5000
  normal_vectors : array, optional
      array of shape (m, 3) of normal vectors considering , by default None
  isAvg : bool, optional
      true if average direction is returned, by default False

  Returns
  -------
  dictionary
      dic['D_sph_flipped']: array of D_sph_flipped of shape (a_1, a_2, ..., a_l)
      dic['avg']: array of average direction of shape (a_1, a_2, ..., a_l, k)
  """
  if normal_vectors is None:
    normal_vectors = sample_spherical_pos(num_random)

  # shape (a_1, ..., a_l, m)
  d_sph_flipped_arr = _get_D_sph_flipped(poles, normal_vectors)['D_sph_flipped']

  # shape (a_1, ..., a_l)
  dic = {'D_sph_flipped': np.amin(d_sph_flipped_arr, axis=-1)}

  if isAvg:
    # shape (a_1, a_2, ..., a_l)
    indices = np.argmin(d_sph_flipped_arr, axis=-1)

    # shape (a_1, a_2, ..., a_l, 3)
    dic['avg'] = normal_vectors[indices,:]

  return dic

def get_D_sph_flipped_from_angles(angles):
  """calculate D_sph_flipped from angles

  Parameters
  ----------
  angles : array
      array of shape (a_1, a_2, ..., a_l, n, m)

  Returns
  -------
  dictionary
      dic['D_sph_flipped']: array of D_sph_flipped of shape (a_1, a_2, ..., a_l, m)
  """
  # shape (a_1, a_2, ..., a_l, m)
  D_sph_flipped = np.mean(angles**2, axis=-2)**(1/2)

  return {'D_sph_flipped': D_sph_flipped}

def random_choice_noreplace(m, n, k):
  """generate m sets of indices from 0 to n-1, each set has k elements

  Parameters
  ----------
  m : int
      number of iterations
  n : int
      indices range
  k : int
      number of indices in the indices range

  Returns
  -------
      array of shape (m,k) representing m sets of indices from 0 to n-1, each set has k elements
  """
  return np.random.rand(m,n).argsort(axis=-1)[:,:k]
  


