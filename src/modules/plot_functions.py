import numpy as np
from modules.spline import eval_new_spline, eval_spline
from modules.helper_functions import extract_data, prep_data, extract_inside_at_timestep, to_spherical
import matplotlib.pyplot as plt
from modules.stats import ks_uniformity_test, sample_spherical_pos, gen_dist, get_smallest_rms
from matplotlib.animation import FuncAnimation
from scipy.stats import gaussian_kde
import pandas as pd

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


def plot_spherical_coordinates(spherical_coors, arr_mass, title, img_name, arr_time = None, isColorBar = False, saveimage = False):
  plt.figure(figsize=(8, 6)).add_subplot(111, projection="aitoff")
  if isColorBar:
    plt.scatter(spherical_coors[:,1], spherical_coors[:,0], marker = '.', c = arr_time, cmap="viridis", s = (arr_mass/max(arr_mass))**(2/5)*200)
  else:
    plt.scatter(spherical_coors[:,1], spherical_coors[:,0], marker = '.', s = (arr_mass/max(arr_mass))**(2/5)*200)
  plt.rcParams['axes.titley'] = 1.1
  plt.title(title)
  plt.grid(True)

  if isColorBar:
    clb = plt.colorbar(orientation="horizontal", ticks=[-14, -12, -10,-8,-6,-4,-2,0])
    _ = clb.ax.set_title('Lookback time')
    _ = clb.ax.set_xlabel('Gyrs', loc='right')
    _ = clb.ax.set_xticklabels([14,12,10,8,6,4,2,0])

  if saveimage:
    plt.savefig(img_name)

def plot_kolmogorov(arr, img_name, title = "", ax = None, saveimage = False):
  random_phi = np.random.rand()*2*np.pi-np.pi
  random_theta = np.pi/2-np.arccos(1-2*np.random.rand())

  direction = np.array([-np.cos(random_theta)*np.cos(random_phi),-np.cos(random_theta)*np.sin(random_phi),np.sin(random_theta)])
  dist_pos = np.sum(arr*direction, axis = 1)
  n = len(dist_pos)

  if ax is None:
    ax = plt.figure(figsize=(8, 6)).add_subplot()

  _, _, _ = ax.hist(dist_pos, bins = n, range=(-1,1), density = True, cumulative=True, histtype='step', label = 'cumulative distribution of cos(theta)')
  uniform = (2*np.arange(1, n+1)-n-1)/(n-1)
  _, _, _ = ax.hist(uniform, bins=len(uniform), density = True, cumulative=True, histtype='step', label = 'expected cumulative distribution')
  ax.set_title(title)
  plt.rcParams['axes.titley'] = 1
  ax.legend(loc='upper left')
  ax.text(-1, 0.8, "kolmogorov probability is {:.2g}".format(ks_uniformity_test(dist_pos)))

  if saveimage:
    plt.savefig(img_name)

def plot_quiver(arr_pos, arr_vec, title = "", arr_time = None, isColor = False, ax = None):
  if ax is None:
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
  ax.set_title(title)
  

  plt.show()

def plot3D(X, Y, Z):
  ax = plt.figure().add_subplot(projection='3d')
  ax.set_box_aspect(aspect = (1,1,1))
  ax.scatter(X,Y,Z)

def plot_evolution(lookback_time, coefs, prop, label, title, imgname, ax = None, saveimage = False):
  t_0 = min(lookback_time)
  t_1 = max(lookback_time)

  time_range = np.linspace(t_0, t_1, 500)
  mass_range = np.array(list(map(lambda t: eval_spline(t, lookback_time, *coefs), time_range)))

  if ax is None:
    ax = plt.figure(figsize=(8, 6)).add_subplot()

  ax.plot(time_range, mass_range, label=label)
  ax.legend(loc='upper left')
  ax.set_xlabel('Lookback time (Gyrs)')
  ax.set_ylabel(prop)
  ax.set_title(title)
  ax.invert_xaxis()

  if saveimage:
    plt.savefig(imgname)

def make_animations(data, arr_row, num_time, halo_row, img_name, saveimage=False):
  max_mass = max(data['Mvir'][halo_row])

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
    anim.save(img_name, writer='ffmpeg')

  return anim

def plot_rms(data, arr_row, lookback_time0, X0, Y0, Z0, Rvir0, title, img_name, ax = None, saveimage = False):
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

  if ax is None:
    ax = plt.figure(figsize=(8, 6)).add_subplot()

  ax.scatter(lookback_time0, arr_rms, label="rms height")
  ax.legend(loc='upper left')
  # plt.yscale('log')
  ax.set_xlabel('Lookback time (Gyrs)')
  ax.set_ylabel('rms (rvir)')
  ax.set_title(title)
  ax.invert_xaxis()

  if saveimage:
    plt.savefig(img_name)

def plot_distribution_rms_dispersion(suite_name, halo_row, imgname, set_title = True, show_halo = True, show_isotropized = True, show_isotropy = False, show_uniform = False, iterations = 10000, num_chosen_subhalos = 11,  ax=None, data=None, timestep=0, timedata_dir="timedata/isolated", elvis_iso_dir="../../Elvis/IsolatedTrees", saveimage=False):
  df = pd.read_csv(timedata_dir+f'/{suite_name}.csv')
  arr_row = np.array(df['row'])
  
  
  inside_index, pos = extract_inside_at_timestep(suite_name, halo_row, data, timestep=timestep, timedata_dir=timedata_dir, elvis_iso_dir=elvis_iso_dir)
  size = len(pos)

  if size < num_chosen_subhalos:
    print("fewer than 11 subhalos...")
    return
  
  if ax is None:
    ax = plt.figure(figsize=(8, 6)).add_subplot()

  rmss_halo = []
  rmss_isotropized = []
  rmss_isotropy = []
  rmss_uniform = []

  for _ in range(iterations):
    temp_range = np.random.choice(np.arange(size), size=num_chosen_subhalos, replace=False)

    pos_halo = pos[temp_range,:]
    r_halo = np.sum(pos_halo**2, axis=1)**(1/2)
    r_med_halo = np.median(r_halo)
    
    iso_pos = sample_spherical_pos(size=num_chosen_subhalos)

    if show_halo:
      rmss_halo.append(get_smallest_rms(pos_halo)/r_med_halo)
      
    if show_isotropized:
      pos_isotropized = np.reshape(r_halo, (num_chosen_subhalos,1)) * iso_pos
      rmss_isotropized.append(get_smallest_rms(pos_isotropized)/r_med_halo)

    if show_isotropy:
      r_isotropy = np.random.uniform(size=num_chosen_subhalos)
      pos_isotropy = np.reshape(r_isotropy, (num_chosen_subhalos,1)) * iso_pos
      rmss_isotropy.append(get_smallest_rms(pos_isotropy)/np.median(r_isotropy))

    if show_uniform:
      r_uniform = gen_dist(size=num_chosen_subhalos)
      pos_uniform = np.reshape(r_uniform, (num_chosen_subhalos,1)) * iso_pos
      rmss_uniform.append(get_smallest_rms(pos_uniform)/np.median(r_uniform))
  
  points = np.linspace(0, 1, num = 1000)

  if set_title:
    ax.set_title('The distribution of the rms dispersions')
    ax.set_xlabel('$D_{\\textrm{rms}}/R_{\\textrm{med}}$')
    ax.set_ylabel('P($D_{\\textrm{rms}}/R_{\\textrm{med}}$)')

  max_arr = np.max(data['Mvir'][arr_row[inside_index]], axis=1)
  argmax = np.argsort(max_arr)[-num_chosen_subhalos:]
  pos_max = pos[argmax,:]
  r_max = np.sum(pos_max**2, axis=1)**(1/2)
  r_med_max = np.median(r_max)

  if show_halo:
    kernel_halo = gaussian_kde(rmss_halo)
    ax.plot(points, kernel_halo(points), label=f"{suite_name}", color='b')
    ax.arrow(get_smallest_rms(pos_max)/r_med_max,0,0,0.5,color='b',head_width=0.01,head_length=0.1)

  if show_isotropized:
    kernel_isotropized = gaussian_kde(rmss_isotropized)
    ax.plot(points, kernel_isotropized(points), label=f"{suite_name} isotropized", color='r')
    pos_max_isotropized = np.reshape(r_max, (num_chosen_subhalos,1))*sample_spherical_pos(size=num_chosen_subhalos)
    ax.arrow(get_smallest_rms(pos_max_isotropized)/r_med_max,0,0,0.5,color='r',head_width=0.01,head_length=0.1)

  if show_isotropy:
    kernel_isotropy = gaussian_kde(rmss_isotropy)
    ax.plot(points, kernel_isotropy(points), label=f"isotropy", color='g')

  if show_uniform:
    kernel_uniform = gaussian_kde(rmss_uniform)
    ax.plot(points, kernel_uniform(points), label=f"uniform", color='y')

  ax.legend()
  if saveimage:
    plt.savefig(imgname)
  
  

def plot_orbital_poles(suite_name, arr_pos_cur, arr_vec_cur, imgname, iterations = 500000, num_chosen_subhalos=11, ax = None, saveimage=False):
  poles = np.cross(arr_pos_cur, arr_vec_cur).T
  temp = (np.sum(poles**2, axis = 0))**(1/2)
  poles = np.array(poles/temp).T

  min_average_poles = []
  min_d_angles = []

  if len(poles) >= num_chosen_subhalos:
    if ax is None:
      ax = plt.figure(figsize=(8, 6)).add_subplot(projection='aitoff')
    for k in range(3,num_chosen_subhalos+1):
      average_poles = []
      d_angles = []
      
      for _ in range(iterations):
        indices = np.random.choice(len(poles), k, replace=False)
        chosen_poles = poles[indices]
        
        average_pole = np.mean(chosen_poles, axis = 0)
        
        average_pole = average_pole/np.sum(average_pole**2) ** (1/2)
        
        dot_prod = np.sum(average_pole * chosen_poles, axis = 1)
        
        angles = np.arccos(dot_prod)
        
        d_angle = np.mean(angles**2)**(1/2)
        
        average_poles.append(average_pole)
        d_angles.append(d_angle)
          
      average_poles = np.array(average_poles)
      d_angles = np.array(d_angles)
      
      min_index = np.argmin(d_angles)
      
      min_average_poles.append(average_poles[min_index])
      min_d_angles.append(d_angles[min_index])
    
    count = 3
    for average_pole, d_angle in zip(min_average_poles, min_d_angles):
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
      
      ax.scatter(aitoff_phis, aitoff_thetas, marker = '.', label=f"k={count}")
      ax.grid(True)
      count += 1

    ax.legend(loc='upper right', bbox_to_anchor=(0.6, 0., 0.5, 0.3))

    if saveimage:
      plt.savefig(imgname)