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

def to_normal_vector(loc):
  theta, phi = loc
  x = np.sin(theta)*np.cos(phi)
  y = np.sin(theta)*np.sin(phi)
  z = np.cos(theta)
  return np.array([x,y,z])

def get_rms(pos, normal_vector, isRemovingOutliers = False):
  """
  pos: (n,3) array, n points, each with 3 pos
  isRemovingOutliers: True if only considering the 16-84 percentile (1 sigma)
  """
  n = len(pos)
  temp = np.sum(pos*normal_vector, axis=1)
  if isRemovingOutliers:
    lower = np.percentile(temp, 16)
    upper = np.percentile(temp, 84)
    indices_between = np.logical_and(temp > lower, temp < upper)
    temp = temp[indices_between]
  coef = np.mean(temp)
  rms = (1/n*np.sum((temp-coef)**2))**(1/2)
  return rms

def get_smallest_rms(pos, isRemovingOutliers = False, num_random_points=10000):
  """
  pos: (n,3) array, n points, each with 3 pos
  """
  arr = np.random.rand(num_random_points,2)*np.array([np.pi/2, 2*np.pi])
  normal_vectors = np.array(list(map(to_normal_vector, arr)))
  rms = np.array(list(map(lambda normal_vector: get_rms(pos, normal_vector, isRemovingOutliers), normal_vectors)))
  
  i = np.argmin(rms)
  
  return normal_vectors[i], rms[i]

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
  guessed_vector, guessed_rms = get_smallest_rms(pos)
  guess_rms = guessed_rms/(a**2+b**2+c**2)**(1/2)
  print(guessed_vector)
  print(guessed_rms)