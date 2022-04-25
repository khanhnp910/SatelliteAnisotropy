import numpy as np
from typing import Union
from scipy.special import kolmogorov 
from deprecated import deprecated

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

@deprecated(reason="This functions is no longer supported")
def to_normal_vector(loc):
  theta, phi = loc
  x = np.sin(theta)*np.cos(phi)
  y = np.sin(theta)*np.sin(phi)
  z = np.cos(theta)
  return np.array([x,y,z])

def get_rms(pos: np.ndarray, normal_vectors: np.ndarray)->np.ndarray:
  """get the rms array of for the normal vectors

  Args:
      pos (np.ndarray): ndarray of shape (n, 3) representing n points in space
      normal_vectors (np.ndarray): ndarray of shape (m, 3) representing m normal vectors

  Returns:
      np.ndarray: ndarray of shape (m,) representing the rms of the plane corresponding the normal vectors
  """
  # shape (n, m)
  temp = np.matmul(pos, normal_vectors.T)
  
  # shape (m,)
  coef = np.mean(temp, axis = 0)
  
  # shape (m,)
  rms = np.average((temp-coef)**2, axis=0)**(1/2)
  
  return rms

def get_rms_and_coef(pos: np.ndarray, normal_vectors: np.ndarray)->tuple[np.ndarray, np.ndarray]:
  """get the rms array of pos corresponding to the normal vectors

  Args:
      pos (np.ndarray): ndarray of shape (n, 3) representing n points in space
      normal_vectors (np.ndarray): ndarray of shape (m, 3) representing m normal vectors

  Returns:
      tuple[np.ndarray, np.ndarray]: 
        np.ndarray: ndarray of shape (m,) representing the rms of the plane corresponding the normal vectors
        np.ndarray: ndarray of shape (m,) representing the displacements of the rms plane

  """
  # shape (n, m)
  temp = np.matmul(pos, normal_vectors.T)

  # shape (m,)
  coef = np.mean(temp, axis = 0)

  # shape (m,)
  rms = np.average((temp-coef)**2, axis=0)**(1/2)

  return rms, coef

def get_smallest_rms(pos: np.ndarray, num_random_points: int=5000)->float:
  """get the smallest rms of pos

  Args:
      pos (np.ndarray): ndarray of shape (n, 3) representing n points in space
      num_random_points (int, optional): number of random normal vectors generated. Defaults to 5000.

  Returns:
      float: the smallest rms of pos
  """
  # shape (num_random_points, 3)
  normal_vectors = sample_spherical_pos(size=num_random_points)

  # shape (num_random_points,)
  rms = get_rms(pos, normal_vectors)
  
  return np.min(rms)

def get_min_vec(pos: np.ndarray, num_random_points: int=10000)->tuple[float, np.ndarray, float]:
  """get the smallest rms of pos with the normal vector and displacement coefficient

  Args:
      pos (np.ndarray): ndarray of shape (n, 3) representing n points in space
      num_random_points (int, optional): number of random normal vectors generated. Defaults to 10000.

  Returns:
      tuple[int, np.ndarray, int]: 
        float: represents the smallest rms
        np.ndarray: ndarray of shape (3,) representing the normal vector of the rms plane
        float: represents the displacement of the rms plane
  """
   # shape (num_random_points, 3)
  normal_vectors = sample_spherical_pos(size=num_random_points)

  # int, ndarray of shape (3,)
  rms, coef = get_rms_and_coef(pos, normal_vectors)

  # index
  i = np.argmin(rms)

  return rms[i], normal_vectors[i], coef[i] 

def get_arr_rms(arr_pos: np.ndarray, normal_vectors: np.ndarray)->np.ndarray:
  """get rms of l groups of pos, each group has n points, 3 dimensions

  Args:
      arr_pos (np.ndarray): ndarray of shape (l, n, 3)
      normal_vectors (np.ndarray): ndarray of shape (m, 3)

  Returns:
      np.ndarray: ndarray of shape (l, m)
  """
  # arr_pos has shape (l, n, k): l iterations, n vectors, k dimensions
  # normal_vectors has shape (m, k): m vectors, k dimensions

  # shape (l, n, m)
  temp = np.matmul(arr_pos, normal_vectors.T)
  
  # shape (l, m)
  coef = np.mean(temp, axis=1)
  
  # shape (l, m)
  rms = np.average((np.transpose(temp, axes=(1,0,2))-coef)**2, axis=0)**(1/2)
  
  return rms

def get_smallest_arr_rms(arr_pos: np.ndarray, num_random_points: int=5000)->np.ndarray:
  """get smallest rms of l groups of pos, each group has n points, 3 dimensions

  Args:
      arr_pos (np.ndarray): ndarray of shape (l, n, 3)
      num_random_points (int, optional): number of random normal vectors generated. Defaults to 5000.

  Returns:
      np.ndarray: ndarray of shape (l,) the smallest rms of arr_pos
  """
  # shape (num_random_points, 3)
  normal_vectors = sample_spherical_pos(size=num_random_points)

  # shape (l, num_random_points)
  rms = get_arr_rms(arr_pos, normal_vectors)
  
  return np.min(rms, axis = 1)

def sample_rad_dis_uniform(size: Union[int, list] = 1)->Union[float, np.ndarray]:
  """generates a random sample of size=size from the radius distribution of uniform points in a sphere

  Args:
      size (Union[int, list], optional): size of points sampled. Defaults to 1.

  Returns:
      Union[float, np.ndarray]: sample of size=size from the radius distribution of uniform points in a sphere
  """
  return np.random.rand(size)**(1/3)

def get_rms_with_percentile(pos: np.ndarray, normal_vectors: np.ndarray)->np.ndarray:
  """get the rms array of for the normal vectors by percentile

  Args:
      pos (np.ndarray): ndarray of shape (n, 3) representing n points in space
      normal_vectors (np.ndarray): ndarray of shape (m, 3) representing m normal vectors

  Returns:
      np.ndarray: ndarray of shape (m,) representing the rms of the plane corresponding the normal vectors by percentile
  """
  # shape (n, m)
  temp = np.matmul(pos, normal_vectors.T)

  # shape (m,)
  coef = np.mean(temp, axis = 0)

  # shape (m,)
  rms = np.percentile(np.abs(temp-coef), 68, axis = 0)

  return rms

def get_smallest_rms_with_percentile(pos: np.ndarray, num_random_points: int=5000)->float:
  """get the smallest rms of pos by percentile

  Args:
      pos (np.ndarray): ndarray of shape (n, 3) representing n points in space
      num_random_points (int, optional): number of random normal vectors generated. Defaults to 5000.

  Returns:
      float: the smallest rms of pos by percentile
  """
  # shape (num_random_points, 3)
  normal_vectors = sample_spherical_pos(size=num_random_points)

  # shape (num_random_points,)
  rms = get_rms_with_percentile(pos, normal_vectors)
  
  return np.min(rms)

@deprecated(reason="This functions is no longer supported")
def generate_random_points_with_rms():
  rms = np.random.rand()
  a,b,c,d = np.random.rand(4)
  n1 = np.reshape(np.array([a,b,c]), (1,3))
  n2 = np.reshape(np.array([-b,a,0]),(1,3))
  n3 = np.reshape(np.array([a,b,-(a**2+b**2)/c]),(1,3))
  
  z = np.reshape(np.random.rand(10000)*10-1, (10000,1))
  y = np.reshape(np.random.rand(10000)*10-1, (10000,1))
  x = np.reshape(np.random.normal(size=10000)*rms, (10000,1))
  
  pos = np.matmul(x,n1) + np.matmul(y,n2) + np.matmul(z,n3)
  
  normal_vector = n1/(np.sum(n1**2))**(1/2)
  
  print(get_rms(pos, normal_vector)/(a**2+b**2+c**2)**(1/2))
  print(rms)
  print(normal_vector)
  guessed_rms = get_smallest_rms(pos)
  # guess_rms = guessed_rms/(a**2+b**2+c**2)**(1/2)
  print(guessed_rms)

def sample_spherical_angle(size: Union[int, list]=1)->tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
  """generates a random sample of size=size of phi, theta isotropically 

  Args:
      size (Union[int, list], optional): size of angles sampled. Defaults to 1.

  Returns:
      tuple[Union[float, np.ndarray], Union[float, np.ndarray]]: 
        Union[float, np.ndarray]: sample of phi
        Union[float, np.ndarray]: sample of theta

  """
  # shape of size
  phi = np.random.uniform(size=size) * 2 * np.pi
  theta = np.arccos(1-2*np.random.uniform(size=size))

  return phi, theta

def sample_spherical_pos(size: Union[int, list]=1)->np.ndarray:
  """generates a random sample of size=size of phi, theta isotropically 

  Args:
      size (Union[int, list], optional): size of angles sampled. Defaults to 1.

  Returns:
      np.ndarray: ndarray of shape (size, 3)
  """
  # shape of size
  phi, theta = sample_spherical_angle(size=size)

  # shape of size
  X = np.cos(phi) * np.sin(theta)
  Y = np.sin(phi) * np.sin(theta)
  Z = np.cos(theta)

  return np.moveaxis(np.array([X,Y,Z]), 0, -1)

def get_rms_poles(poles: np.ndarray)->float:
  """calculate d_sph for 1 group of poles

  Args:
      poles (np.ndarray): ndarray of shape (n,k) representing n poles in each group, k dimensions in each pole

  Returns:
      float: represents the d_sph
  """
  return get_rms_poles_with_avg(poles)[0]

def get_rms_poles_with_avg(poles: np.ndarray)->tuple[float, np.ndarray]:
  """calculate d_sph and avg_pole for 1 group of poles

  Args:
      poles (np.ndarray): ndarray of shape (n,k) representing n poles in each group, k dimensions in each pole

  Returns:
      tuple[float, np.ndarray]: 
        float: represents the d_sph
        np.ndarray: ndarray of shape (k,) representing avg_pole
  """
  # shape of (k,)
  avg_pole = np.mean(poles, axis = 0)
  avg_pole = avg_pole/np.sum(avg_pole**2) ** (1/2)

  # shape of (n,k)
  dot_prod = np.sum(avg_pole * poles, axis = 1)
  angles = np.arccos(dot_prod)

  # int
  d_sph = np.mean(angles**2)**(1/2)

  return d_sph, avg_pole

def get_rms_arr_poles(arr_poles: np.ndarray)->np.ndarray:
  """calculate d_sph for m groups of poles

  Args:
      arr_poles (np.ndarray): ndarray of shape (m,n,k) representing m groups of poles, n poles in each group, k dimensions in each pole

  Returns:
      np.ndarray: ndarray of shape (m,) representing m d_sph, one for each group
  """
  return get_rms_arr_poles_with_avg(arr_poles)[0]

def get_rms_arr_poles_with_avg(arr_poles: np.ndarray)->tuple[np.ndarray, np.ndarray]:
  """calculate d_sph and avg_pole for m groups of poles

  Args:
      arr_poles (np.ndarray): ndarray of shape (m,n,k) representing m groups of poles, n poles in each group, k dimensions in each pole

  Returns:
      tuple[np.ndarray, np.ndarray]: 
        np.ndarray: ndarray of shape (m,) representing m d_sph, one for each group
        np.ndarray: ndarray of shape (m,k) representing m avg_pole, one for each group
  """
  # shape (m,k)
  avg_poles = np.sum(arr_poles, axis=1)

  avg_poles = (avg_poles.T / np.sum(avg_poles**2, axis=1)**(1/2)).T

  # shape (n,m,k)
  trans_arr_poles = np.transpose(arr_poles, axes=(1,0,2))

  # shape (n,m)
  dotprods = np.sum(np.transpose(trans_arr_poles*avg_poles, axes=(1,0,2)), axis = 2)

  # shape (n,m)
  angles = np.arccos(dotprods)

  # shape (m,)
  d_sph = np.average(angles**2, axis=1)**(1/2)

  return d_sph, avg_poles

def random_choice_noreplace(m: int, n: int, k:int)->np.ndarray:
  """generate m sets of indices from 0 to n-1, each set has k elements

  Args:
      m (int): number of iterations
      n (int): indices range
      k (int): number of indices in the indices range

  Returns:
      np.ndarray: ndarray of shape (m,k) representing m sets of indices from 0 to n-1, each set has k elements
  """

  return np.random.rand(m,n).argsort(axis=-1)[:,:k]

def get_rms_arr_poles_for_k(poles: np.ndarray, iterations: int = 100000, num_chosen: int = 11)->np.ndarray:
  """for each iteration, calculate d_sph for a random set of num_chosen poles in poles 

  Args:
      poles (np.ndarray): ndarray of shape (n,3) representing n poles
      iterations (int, optional): number of iterations. Defaults to 100000.
      num_chosen (int, optional): number of chosen poles among the n poles. Defaults to 11.

  Returns:
      np.ndarray: ndarray of shape (iterations,) representing the d_sph
  """
  # shape (iterations, num_chosen)
  indices = random_choice_noreplace(iterations, len(poles), num_chosen)

  # shape (iterations, num_chosen, 3)
  chosen_poles = poles[indices]

  return get_rms_arr_poles(chosen_poles)

def get_rms_arr_poles_with_avg_for_k(poles: np.ndarray, iterations: int = 100000, num_chosen: int = 11)->tuple[np.ndarray, np.ndarray]:
  """for each iteration, calculate d_sph and avg_pole for a random set of num_chosen poles in poles

  Args:
      poles (np.ndarray): ndarray of shape (n,3) representing n poles
      iterations (int, optional): number of iterations. Defaults to 100000.
      num_chosen (int, optional): number of chosen poles among the n poles. Defaults to 11.

  Returns:
      tuple[np.ndarray, np.ndarray]: 
        np.ndarray: ndarray of shape (iterations,) representing the d_sph
        np.ndarray: ndarray of shape (iterations,3) representing the avg_pole
  """
  # shape (iterations, num_chosen)
  indices = random_choice_noreplace(iterations, len(poles), num_chosen)

  # shape (iterations, num_chosen, 3)
  chosen_poles = poles[indices]

  return get_rms_arr_poles_with_avg(chosen_poles)

