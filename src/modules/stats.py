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

def get_rms(pos, normal_vectors):
  """
  pos: (n,3) array, n points, each with 3 pos
  normal_vectors: (m, 3) array, m normal vectors, each with 3 pos
  """
  n = len(pos)
  temp = np.matmul(pos, normal_vectors.T)
  coef = np.mean(temp, axis = 0)
  rms = (1/n*np.sum((temp-coef)**2, axis=0))**(1/2)
  return rms

def get_rms_and_coef(pos, normal_vectors):
  """
  pos: (n,3) array, n points, each with 3 pos
  normal_vectors: (m, 3) array, m normal vectors, each with 3 pos
  """
  n = len(pos)
  temp = np.matmul(pos, normal_vectors.T)
  coef = np.mean(temp, axis = 0)
  rms = (1/n*np.sum((temp-coef)**2, axis=0))**(1/2)
  return rms, coef

def get_smallest_rms(pos, num_random_points=5000):
  """
  pos: (n,3) array, n points, each with 3 pos
  """
  # arr = np.random.rand(num_random_points,2)*np.array([np.pi/2, 2*np.pi])
  theta = np.arccos(1-2*np.random.rand(num_random_points))
  phi = np.random.rand(num_random_points)*2*np.pi
  z = np.cos(theta)
  sintheta = np.sin(theta)
  y = sintheta * np.cos(phi)
  x = sintheta * np.sin(phi)
  normal_vectors = np.array([x,y,z]).T
  rms = get_rms(pos, normal_vectors)
  
  return rms.min()

def get_min_vec(pos, num_random_points=10000):
  theta = np.arccos(1-2*np.random.rand(num_random_points))
  phi = np.random.rand(num_random_points)*2*np.pi
  z = np.cos(theta)
  sintheta = np.sin(theta)
  y = sintheta * np.cos(phi)
  x = sintheta * np.sin(phi)
  normal_vectors = np.array([x,y,z]).T
  rms, coef = get_rms_and_coef(pos, normal_vectors)
  i = rms.argmin()
  return rms[i], normal_vectors[i], coef[i] 

def gen_dist(size = 1):
  return np.random.rand(size)**(1/3)

def get_rms_with_percentile(pos, normal_vectors):
  n = len(pos)
  temp = np.matmul(pos, normal_vectors.T)
  coef = np.mean(temp, axis = 0)
  rms = np.percentile(np.abs(temp-coef), 68, axis = 0)

  return rms

def get_smallest_rms_with_percentile(pos, num_random_points=5000):
  theta = np.arccos(1-2*np.random.rand(num_random_points))
  phi = np.random.rand(num_random_points)*2*np.pi
  z = np.cos(theta)
  sintheta = np.sin(theta)
  y = sintheta * np.cos(phi)
  x = sintheta * np.sin(phi)
  normal_vectors = np.array([x,y,z]).T
  rms = get_rms_with_percentile(pos, normal_vectors)
  
  return rms.min()

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

def sample_spherical_angle(size=1):
  phi = np.random.uniform(size=size) * 2 * np.pi
  theta = np.arccos(1-2*np.random.uniform(size=size))

  return phi, theta

def sample_spherical_pos(size=1):
  phi, theta = sample_spherical_angle(size=size)

  X = np.cos(phi) * np.sin(theta)
  Y = np.sin(phi) * np.sin(theta)
  Z = np.cos(theta)

  return np.array([X,Y,Z]).T

def get_rms_poles(poles):
  return get_rms_poles_with_avg(poles)[0]

def get_rms_poles_with_avg(poles):
  average_pole = np.mean(poles, axis = 0)
  average_pole = average_pole/np.sum(average_pole**2) ** (1/2)
  dot_prod = np.sum(average_pole * poles, axis = 1)
  angles = np.arccos(dot_prod)
  d_angle = np.mean(angles**2)**(1/2)

  return d_angle, average_pole
