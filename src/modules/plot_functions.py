import numpy as np
from modules.spline import eval_new_spline, eval_spline
from modules.helper_functions import extract_data, prep_data, extract_inside_at_timestep, to_spherical, get_rms_MW, get_rms_poles_MW, extract_poles_inside_at_timestep, to_degree
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from modules.stats import *
from matplotlib.animation import FuncAnimation
from scipy.stats import gaussian_kde
import scipy.optimize as opt
import pandas as pd

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
      sig = opt.brentq( conf_interval, 0., 1., args=(H,cld) )   
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

def plot_spherical_coordinates(spherical_coors, arr_mass, title, img_name, 
                        arr_time = None, isColorBar = False, saveimage = False):
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
  ax.set_ylabel('\\Delta_{\\textrm{rms}} (rvir)')
  ax.set_title(title)
  ax.invert_xaxis()

  if saveimage:
    plt.savefig(img_name)

def plot_distribution_rms_dispersion(suite_name, halo_row, imgname, set_title = True, show_halo = True, show_isotropized = True, 
                        show_isotropy = False, show_uniform = False, iterations = 100000, num_chosen_subhalos = 11,  
                        ax=None, data=None, timestep=0, timedata_dir="timedata/isolated", saveimage=False):
  df = pd.read_csv(timedata_dir+f'/{suite_name}.csv')
  arr_row = np.array(df['row'])
  
  
  inside_index, pos = extract_inside_at_timestep(suite_name, halo_row, data, timestep=timestep, timedata_dir=timedata_dir)
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
      r_uniform = sample_rad_dis_uniform(size=num_chosen_subhalos)
      pos_uniform = np.reshape(r_uniform, (num_chosen_subhalos,1)) * iso_pos
      rmss_uniform.append(get_smallest_rms(pos_uniform)/np.median(r_uniform))
  
  points = np.linspace(0, 1, num = 1000)

  if set_title:
    ax.set_title('The distribution of the rms dispersions')
    ax.set_xlabel('$\\Delta_{\\textrm{rms}}/R_{\\textrm{med}}$')
    ax.set_ylabel('P($\\Delta_{\\textrm{rms}}/R_{\\textrm{med}}$)')

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
  
def plot_circle_around_vector(average_pole, d_angle,ax=None,label=""):
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

def plot_orbital_poles(suite_name, halo_row, data, imgname, timestep=0, iterations = 500000, 
                num_chosen_subhalos=11, timedata_dir="timedata/isolated", ax = None, saveimage=False):
  poles = extract_poles_inside_at_timestep(suite_name, halo_row, data, timestep=timestep, timedata_dir=timedata_dir)

  min_average_poles = []
  min_d_angles = []

  if len(poles) >= num_chosen_subhalos:
    if ax is None:
      ax = plt.figure(figsize=(8, 6)).add_subplot(projection='aitoff')
    for k in range(3,num_chosen_subhalos+1):
      d_angles, average_poles = get_rms_arr_poles_with_avg_for_k(poles, iterations=iterations, num_chosen=k)
      
      min_index = np.argmin(d_angles)
      
      min_average_poles.append(average_poles[min_index])
      min_d_angles.append(d_angles[min_index])
    
    count = 3
    for average_pole, d_angle in zip(min_average_poles, min_d_angles):
      plot_circle_around_vector(average_pole, d_angle, ax=ax,label=f"{count}")
      count += 1

    plt.rcParams['axes.titley'] = 1.1
    ax.set_title(f"Distribution of orbital poles and uncertainties for {suite_name}")
    ax.legend(loc='upper right', bbox_to_anchor=(0.6, 0., 0.5, 0.3))
    


    if saveimage:
      plt.savefig(imgname)

def plot_orbital_poles_with_best_config(suite_name, halo_row, data, imgname, timestep=0, iterations = 500000, 
                        num_chosen_subhalos=11, timedata_dir="timedata/isolated", ax = None, saveimage=False):
  
  poles = extract_poles_inside_at_timestep(suite_name, halo_row, data, timestep=timestep, timedata_dir=timedata_dir)

  if len(poles) >= num_chosen_subhalos:
    if ax is None:
      ax = plt.figure(figsize=(8, 6)).add_subplot(projection='aitoff')

    all_chosen_indices = random_choice_noreplace(iterations, len(poles), k=num_chosen_subhalos)
    chosen_poles = poles[all_chosen_indices]
    d_angles, average_poles = get_rms_arr_poles_with_avg(chosen_poles)

    min_index = np.argmin(d_angles)

    average_pole = average_poles[min_index]
    d_angle = d_angles[min_index]
    chosen_indice = all_chosen_indices[min_index]

    plot_circle_around_vector(average_pole, d_angle,ax=ax,label=f"{num_chosen_subhalos}")

    chosen_poles_ = poles[chosen_indice]

    pos_phis = []
    pos_thetas = []

    for i in range(len(chosen_poles_)):
      cur = chosen_poles_[i]
      aitoff_theta, aitoff_phi = to_spherical(cur[0],cur[1],cur[2])
      pos_thetas.append(aitoff_theta)
      pos_phis.append(aitoff_phi)
    
    ax.scatter(pos_phis, pos_thetas, marker = '.')

    _, pos = extract_inside_at_timestep(suite_name, halo_row, data, timedata_dir=timedata_dir)

    chosen_pos_dis = pos[chosen_indice]
    chosen_r = np.sum(chosen_pos_dis ** 2, axis=1) ** (1/2)
    ax.grid(True)
    plt.rcParams['axes.titley'] = 1.1
    ax.set_title(f"Distribution of orbital poles and uncertainties with best config for {suite_name}")
    ax.set_xlabel("$\\Delta_{\\textrm{rms}}$: "+"{:.2f}".format(get_smallest_rms(chosen_pos_dis)/np.median(chosen_r))+"-$\\Delta_{\\textrm{sph}}: $"+"{:.2f}$^\circ$".format(d_angle/(np.pi)*180))
    ax.legend(loc='upper right', bbox_to_anchor=(0.6, 0., 0.5, 0.3))

    if saveimage:
      plt.savefig(imgname)

def plot_rms_orbital_poles_vs_k(suite_name, halo_row, data, imgname, isDeg=True, timestep=0, iterations = 200000, num_chosen_subhalos=11, 
                        timedata_dir="timedata/isolated", ax = None, saveimage=False):
  poles = extract_poles_inside_at_timestep(suite_name, halo_row, data, timestep=timestep, timedata_dir=timedata_dir)
  
  if len(poles) <= num_chosen_subhalos:
    return

  arr = []
  
  if ax is None:
    ax = plt.figure(figsize=(8, 6)).add_subplot()
  for k in range(3,num_chosen_subhalos+1):
    d_angles = get_rms_arr_poles_for_k(poles, iterations=iterations, num_chosen=k)
    if isDeg:
      arr.append(to_degree(np.min(d_angles)))
    else:
      arr.append(np.min(d_angles))
  
  ax.scatter(np.arange(3,num_chosen_subhalos+1), arr)
  ax.set_title(f"rms orbital poles vs number of chosen halos for {suite_name}")
  ax.set_xlabel("k")
  ax.set_ylabel("$\\Delta_{\\textrm{sph}}$ "+"($^\\circ$)" if isDeg else "$\\Delta_{\\textrm{sph}}$ (rad)")

  if saveimage:
    plt.savefig(imgname)

def plot_distribution_rms_orbital_poles_dispersion(suite_name, halo_row, data, imgname, timestep=0, iterations = 20000, 
                            num_chosen_subhalos=11, timedata_dir="timedata/isolated", ax = None, saveimage=False):
  poles = extract_poles_inside_at_timestep(suite_name, halo_row, data, timestep=timestep, timedata_dir=timedata_dir)
  
  if len(poles) <= num_chosen_subhalos:
    return

  if ax is None:
    ax = plt.figure(figsize=(8, 6)).add_subplot()

  d_angles = get_rms_arr_poles_for_k(poles, iterations=iterations, num_chosen=num_chosen_subhalos)
  
  points = np.linspace(0.4, 2, num = 1000)
  kernel = gaussian_kde(d_angles)
  ax.plot(points, kernel(points), label=f"{suite_name}", color='b')
  ax.set_title("$\\Delta_{\\textrm{sph}}$ "+f"dispersion for {suite_name}")
  ax.set_xlabel("$\\Delta_{\\textrm{sph}} (rad)$")
  ax.set_ylabel("$P(\\Delta_{\\textrm{sph}}$)")
  ax.legend()
  if saveimage:
    plt.savefig(imgname)

def plot_distribution_rms_orbital_poles_dispersion_with_selection(suite_name, halo_row, data, imgname, timestep=0, iterations = 800, 
                        timedata_dir="timedata/isolated", num_chosen_subhalos=11, ax = None, saveimage=False):
  poles = extract_poles_inside_at_timestep(suite_name, halo_row, data, timestep=timestep, timedata_dir=timedata_dir)
  
  if len(poles) <= num_chosen_subhalos:
    return

  if ax is None:
    ax = plt.figure(figsize=(8, 6)).add_subplot()
  
  _, pos = extract_inside_at_timestep(suite_name, halo_row, data, timestep=timestep, timedata_dir=timedata_dir)
  size = len(pos)

  d_angles = []
  rms_MW = get_rms_MW()
  count = 0
  count2 = 0
  while count < iterations:
    count2 += 1
    indices = np.random.choice(size, num_chosen_subhalos, replace=False)
    
    pos_halo = pos[indices,:]
    r_halo = np.sum(pos_halo**2, axis=1)**(1/2)
    r_med_halo = np.median(r_halo)

    if get_smallest_rms(pos_halo)/r_med_halo >= rms_MW:
      continue
    
    count += 1
    chosen_poles = poles[indices]

    d_angle = get_rms_poles(chosen_poles)

    d_angles.append(d_angle)

  d_angles = np.array(d_angles)
  
  points = np.linspace(0.4, 2, num = 1000)
  kernel = gaussian_kde(d_angles)
  ax.plot(points, kernel(points), label=f"{suite_name}", color='b')
  ax.set_title("$\\Delta_{\\textrm{sph}}$"+f" dispersion for {suite_name} with selection")
  ax.set_xlabel("$\\Delta_{\\textrm{sph}} (rad)$")
  ax.set_ylabel("$P(\\Delta_{\\textrm{sph}})$")
  ax.text(0.5,1.7,"Probability of selection: {:.2f}".format(count/count2))
  
  ax.arrow(get_rms_poles_MW(), 0,0,0.5, color='r',head_width=0.01,head_length=0.1)
  
  
  ax.legend()
  if saveimage:
    plt.savefig(imgname)

def plot_hist_rms_vs_rms_poles(suite_name, halo_row, data, imgname, isDeg=True, newData=False,timestep=0, iterations = 20000, 
                              timedata_dir="timedata/isolated", num_chosen_subhalos=11, saveimage=False):
  
  d_angles = []
  rmss = []

  coef = 180/np.pi if isDeg else 1

  if newData:
    poles = extract_poles_inside_at_timestep(suite_name, halo_row, data, timestep=timestep, timedata_dir=timedata_dir)

    if len(poles) <= num_chosen_subhalos:
      return

    _, pos = extract_inside_at_timestep(suite_name, halo_row, data, timestep=timestep, timedata_dir=timedata_dir)
    size = len(pos)

    for _ in range(iterations):
      indices = np.random.choice(size, num_chosen_subhalos, replace=False)

      pos_halo = pos[indices,:]
      r_halo = np.sum(pos_halo**2, axis=1)**(1/2)
      r_med_halo = np.median(r_halo)

      rmss.append(get_smallest_rms(pos_halo)/r_med_halo) 

      chosen_poles = poles[indices]

      d_angle = get_rms_poles(chosen_poles)

      d_angles.append(d_angle)

    d_angles = np.array(d_angles)
    rmss = np.array(rmss)
  else:
    type_suite = 'isolated' if suite_name[0] == 'i' else 'paired'
    df = pd.read_csv(f"../../Data/subhalo_log/{type_suite}/{suite_name}.csv")

    d_angles = df['Delta_sph_11']

    rmss = df['Delta_rms']/df['Rmedian']

  # all_chosen_indices = random_choice_noreplace(iterations, len(poles), k=num_chosen_subhalos)

  # chosen_poles = poles[all_chosen_indices]
  # chosen_pos = pos[all_chosen_indices]
  
  # d_angles = get_rms_arr_poles(chosen_poles)
  # rmss = get_smallest_arr_rms(chosen_pos)

  fig, ax = plt.subplots()

  # x_bins = np.linspace(np.min(rmss), np.max(rmss), 100)
  # y_bins = np.linspace(np.min(d_angles), np.max(d_angles), 100)
  # hist = ax.hist2d(rmss, d_angles, bins =[x_bins, y_bins], cmap='hot_r')
  
  ax = plot_2d_dist(rmss,d_angles*coef, [np.min(rmss),np.max(rmss)], [np.min(d_angles*coef), np.max(d_angles*coef)], 50, 50, figsize=(5,5),fig_setup=ax,clevs=[0.6827, 0.9545, 0.9973])
  ax.scatter(get_rms_MW(), get_rms_poles_MW()*coef, label='MW', marker='x', c='orangered', s=20)

  i = np.argmin(d_angles)
  ax.scatter(rmss[i], d_angles[i]*coef, label='subhalo with min d_sph', marker='x', c='b', s=30)

  ax.set_title(f"histogram of rms_plane and rms_pole of {suite_name}")
  ax.set_xlabel("$\\Delta_{\\textrm{rms}}/R_{\\textrm{med}}$")
  ax.set_ylabel("$\\Delta_{\\textrm{sph}} $"+("$(^{\\circ})$" if isDeg else "$(\\textrm{rad})$"))
  # fig.colorbar(hist[3], orientation='vertical')
  ax.legend()

  if saveimage:
    plt.savefig(imgname)

