import numpy as np
from modules.spline import eval_new_spline, eval_spline
from modules.helper_functions import extract_data, prep_data
from modules.stats import get_smallest_rms
import matplotlib.pyplot as plt
from modules.stats import ks_uniformity_test
from matplotlib.animation import FuncAnimation

def show_track(data, row, ax = None):
  _, lookback_time, X, Y, Z, _, _, _, coefs_X, coefs_Y, coefs_Z = extract_data(data, row, isCoefsPos = True)


  t = np.linspace(lookback_time[0], lookback_time[-1], 150)
  x = np.array(list(map(lambda u: eval_new_spline(u, lookback_time, *coefs_X), t)))
  y = np.array(list(map(lambda u: eval_new_spline(u, lookback_time, *coefs_Y), t)))
  z = np.array(list(map(lambda u: eval_new_spline(u, lookback_time, *coefs_Z), t)))
  
  if not ax:
    ax = plt.figure().add_subplot(projection='3d')
  ax.set_box_aspect(aspect = (1,1,1))
  ax.scatter(X,Y,Z, color='g')
  ax.scatter(x,y,z, color='r')
    
def show_1D_track(data, row, arg = 'X', ax = None):
  _, lookback_time, X, Y, Z, _, _, _, coefs_X, coefs_Y, coefs_Z = extract_data(data, row, isCoefsPos = True)

  
  t = np.linspace(lookback_time[0], lookback_time[-1], 150)
  if arg == 'X':
    v = np.array(list(map(lambda u: eval_new_spline(u, lookback_time, *coefs_X), t)))
    s = X
  if arg == 'Y':
    v = np.array(list(map(lambda u: eval_new_spline(u, lookback_time, *coefs_Y), t)))
    s = Y
  if arg == 'Z':
    v = np.array(list(map(lambda u: eval_new_spline(u, lookback_time, *coefs_Z), t)))
    s = Z
  
  if not ax:
    ax = plt.figure().add_subplot()
  ax.plot(lookback_time, s, color='g')
  ax.plot(t, v, color='r')


def plot_spherical_coordinates(suite_name, spherical_coors, arr_mass, title, img_name, arr_time = None, isColorBar = False, saveimage = False):
  plt.figure(figsize=(8, 6)).add_subplot(111, projection="aitoff")
  if isColorBar:
    plt.scatter(spherical_coors[:,1], spherical_coors[:,0], marker = '.', c = arr_time, cmap="viridis", s = (arr_mass/max(arr_mass))**(2/5)*200)
  else:
    plt.scatter(spherical_coors[:,1], spherical_coors[:,0], marker = '.', s = (arr_mass/max(arr_mass))**(2/5)*200)
  plt.rcParams['axes.titley'] = 1.1
  plt.title(title.format(suite_name))
  plt.grid(True)

  if isColorBar:
    clb = plt.colorbar(orientation="horizontal", ticks=[-14, -12, -10,-8,-6,-4,-2,0])
    _ = clb.ax.set_title('Lookback time')
    _ = clb.ax.set_xlabel('Gyrs', loc='right')
    _ = clb.ax.set_xticklabels([14,12,10,8,6,4,2,0])

  if saveimage:
    plt.savefig(img_name.format(suite_name, suite_name))

def plot_kolmogorov(suite_name, arr, title, img_name, saveimage = False):
  random_phi = np.random.rand()*2*np.pi-np.pi
  random_theta = np.pi/2-np.random.rand()*np.pi

  direction = np.array([-np.cos(random_theta)*np.cos(random_phi),-np.cos(random_theta)*np.sin(random_phi),np.sin(random_theta)])
  dist_pos = np.sum(arr*direction, axis = 1)
  n = len(dist_pos)

  plt.figure(figsize=(8, 6))

  _, _, _ = plt.hist(dist_pos, bins = n, range=(-1,1), density = True, cumulative=True, histtype='step', label = 'cumulative distribution of cos(theta)')
  uniform = (2*np.arange(1, n+1)-n-1)/(n-1)
  _, _, _ = plt.hist(uniform, bins=len(uniform), density = True, cumulative=True, histtype='step', label = 'expected cumulative distribution')
  plt.title(title.format(suite_name))
  plt.rcParams['axes.titley'] = 1
  plt.legend(loc='upper left')
  plt.text(-1, 0.8, "kolmogorov probability is {:.2g}".format(ks_uniformity_test(dist_pos)))

  if saveimage:
    plt.savefig(img_name.format(suite_name,suite_name))

def plot_quiver(suite_name, arr_pos, arr_vec, title, arr_time = None, isColor = False):
  ax = plt.figure().add_subplot(projection='3d')
  ax.set_box_aspect(aspect = (1,1,1))
  reshape_arr_pos = arr_pos.T
  reshape_arr_vec = arr_vec.T

  x = reshape_arr_pos[0]
  y = reshape_arr_pos[1]
  z = reshape_arr_pos[2]
  u = reshape_arr_vec[0]
  v = reshape_arr_vec[1]
  w = reshape_arr_vec[2]

  if isColor:
    c = arr_time
    c = (c.ravel() - c.min()) / c.ptp()
    # Repeat for each body line and two head lines
    c = np.concatenate((c, np.repeat(c, 2)))
    # Colormap
    c = plt.cm.viridis(c)

    ax.quiver(x, y, z, u, v, w, length=0.2, colors=c)
  else:
    ax.quiver(x, y, z, u, v, w, length=0.2)
  ax.set_title(title.format(suite_name))
  

  plt.show()

def plot3D(suite_name, X, Y, Z):
  ax = plt.figure().add_subplot(projection='3d')
  ax.set_box_aspect(aspect = (1,1,1))
  ax.scatter(X,Y,Z)

def plot_evolution(suite_name, lookback_time, coefs, prop, saveimage = False):
  t_0 = min(lookback_time)
  t_1 = max(lookback_time)

  time_range = np.linspace(t_0, t_1, 500)
  mass_range = np.array(list(map(lambda t: eval_spline(t, lookback_time, *coefs), time_range)))

  _, ax = plt.subplots(figsize=(8, 6))

  plt.plot(time_range, mass_range, label="{} of {} halo".format(prop, suite_name))
  plt.legend(loc='upper left')
  ax.set_xlabel('Lookback time (Gyrs)')
  ax.set_ylabel('{}'.format(prop)+" ($M_{\\textrm{Sun}}$)")
  ax.set_title('{} evolution of {} halo'.format(prop, suite_name))
  ax.invert_xaxis()

  if saveimage:
    plt.savefig("../../result/data/{}/{}_evolution_of_{}.pdf".format(suite_name, prop, suite_name))

def make_animations(suite_name, data, arr_row, num_time, saveimage):
  max_mass = max(data['Mvir'][0])

  fig, axs = plt.subplots(2,3)
  fig.set_figheight(6)
  fig.set_figwidth(9)

  scats = []
  scats2 = []
  patches = []
  planes = ['XY','YZ','ZX']

  def init():
    return scats, scats2


    
  def animate(i):
    i = num_time-i-1
    mass, x, y, z, rvir, scale = prep_data(data, i, arr_row)
    pos = [[x,y],[y,z],[z,x]]
    if len(scale) > 0:
      redshift = 1/scale-1        
      axs[0][1].set_title("z={:.3f}".format(redshift[0]))
      
      for j in range(3):
          scats[j].set_offsets(np.array([pos[j][0][1:]-pos[j][0][0],pos[j][1][1:]-pos[j][1][0]]).T)
          scats[j].set_sizes(200*(mass[1:]/max_mass)**(1/4))
          
          scats2[j].set_offsets(np.array([pos[j][0][:1]-pos[j][0][0],pos[j][1][:1]-pos[j][1][0]]).T)
          scats2[j].set_sizes(200*(mass[:1]/max_mass)**(1/4))
      
      for j in range(3):
        patches[j].set(radius=rvir[0]/1000, fill=False)
    else:
      for j in range(3):
        scats[j].set_offsets(np.array([[],[]]).T)
        scats2[j].set_offsets(np.array([[],[]]).T)
    return scats, scats2


  mass, x, y, z, rvir, scale = prep_data(data, 0, arr_row)
  pos = [[x,y],[y,z],[z,x]]

  for i in range(3):
    axs[0][i].set_xlim((-0.5,0.5))
    axs[0][i].set_ylim((-0.5,0.5))
    scats.append(axs[0][i].scatter([],[]))
    scats2.append(axs[0][i].scatter([],[], c='red'))
    patches.append(plt.Circle((0,0),20, fill=False))
    axs[0][i].add_patch(patches[i])
    
    axs[0][i].set_xlabel(planes[i][0])
    axs[0][i].set_ylabel(planes[i][1])

    axs[1][i].scatter(pos[i][0][1:]-pos[i][0][0],pos[i][1][1:]-pos[i][1][0], s=200*(mass[1:]/max_mass)**(1/4))
    axs[1][i].scatter(pos[i][0][:1]-pos[i][0][0],pos[i][1][:1]-pos[i][1][0], s=200*(mass[:1]/max_mass)**(1/4), c='red')
    
    
    axs[1][i].set_xlim((-0.5,0.5))
    axs[1][i].set_ylim((-0.5,0.5))
    axs[1][i].set_xlabel(planes[i][0])
    axs[1][i].set_ylabel(planes[i][1])
    axs[1][i].add_patch(plt.Circle((0,0),rvir[0]/1000, fill=False))

  anim = FuncAnimation(fig, animate, init_func=init, frames=num_time, interval=200, blit=False)

  
  if saveimage:
    anim.save(f'../../result/data/{suite_name}/accretion_animation_of_{suite_name}.gif', writer='ffmpeg')

  return anim

def plot_rms(suite_name, data, arr_row, lookback_time0, X0, Y0, Z0, Rvir0, img_name, saveimage = False):
  Xs = []
  Ys = []
  Zs = []

  arr_rms = []

  for row in arr_row:
    Xs.append(data['X'][row])
    Ys.append(data['Y'][row])
    Zs.append(data['Z'][row])

  Xs = np.array(Xs)
  Ys = np.array(Ys)
  Zs = np.array(Zs)
    
  for j in range(len(lookback_time0)):
    X = Xs[:,j]
    Y = Ys[:,j]
    Z = Zs[:,j]
    
    non_zero = X != 0
    
    new_X = (X[non_zero] - X0[j]) / Rvir0[j]
    new_Y = (Y[non_zero] - Y0[j]) / Rvir0[j]
    new_Z = (Z[non_zero] - Z0[j]) / Rvir0[j]
    
             
             
             
    inside_radius = ((new_X**2 + new_Y**2 + new_Z**2) < 1)
    
    new_new_X = new_X[inside_radius]
    new_new_Y = new_Y[inside_radius]
    new_new_Z = new_Z[inside_radius]
    
    pos = np.array([new_new_X, new_new_Y, new_new_Z]).T
    if len(pos) < 4:
      arr_rms.append(0)
    else:
      rms = get_smallest_rms(pos)
      arr_rms.append(rms)

  _, ax = plt.subplots(figsize=(8, 6))

  plt.scatter(lookback_time0, arr_rms, label="rms height".format(suite_name=suite_name))
  plt.legend(loc='upper left')
  # plt.yscale('log')
  ax.set_xlabel('Lookback time (Gyrs)')
  ax.set_ylabel('rms (rvir)')
  ax.set_title('rms height evolution of subhalos of {suite_name}'.format(suite_name=suite_name))
  ax.invert_xaxis()

  if saveimage:
    plt.savefig(img_name.format(suite_name, suite_name))