#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from os import listdir, system
from sys import argv

import numpy as np


from modules.spline import *
from modules.stats import *
from modules.helper_functions import *
from modules.plot_functions import *

import matplotlib.pyplot as plt
import matplotlib

from scipy.stats import gaussian_kde

import __main__


# In[ ]:


executed_as_python = hasattr(__main__, '__file__')
saveimage = False
if executed_as_python:
    matplotlib.use('Agg')
    saveimage = True


# In[ ]:


# Insert Elvis directory here
elvis_iso_dir = "../../../Project/Elvis/IsolatedTrees"

# Insert suite name here
suite_name = "iHall"

if executed_as_python:
    suite_name = argv[1]


# In[ ]:


# Do not modify
data = read_elvis_tracks(elvis_iso_dir, suite_name,varnames = ['Mvir','scale','X','Y','Z','ID','pID','Rvir','Vx','Vy','Vz'])


# In[ ]:


def gen_dist(size = 1):
    return np.random.rand(size)**(1/3)


# In[ ]:


plot_pretty()


# In[ ]:


num_halos = get_num_halos(data)
num_time = get_num_time(data)


# In[ ]:


df = pd.read_csv(f'timedata/{suite_name}.csv')
arr_row = np.array(df['row'])


# In[ ]:


_, _, X0, Y0, Z0, Rvir0 = extract_data(data, 0, isVel = False, isRvir = True)


# In[ ]:


X = (data['X'][arr_row][:,0] - X0[0]) / Rvir0[0]
Y = (data['Y'][arr_row][:,0] - Y0[0]) / Rvir0[0]
Z = (data['Z'][arr_row][:,0] - Z0[0]) / Rvir0[0]

inside_index = (X**2 + Y**2 + Z**2 < 1)
new_X = X[inside_index]
new_Y = Y[inside_index]
new_Z = Z[inside_index]


# In[ ]:


if len(new_X) < 11:
    print("fewer than 11 subhalos...")
else:
    rmss = []
    rmss_uniform_dense = []
    rmss_subhalo = []
    rmss_subhalo_isotropized = []
    rs_subhalo = []
    rmss_over_r_med_subhalo = []
    iterations = 20000
    num_chosen_subhalos = 11

    for _ in range(iterations):
        phi = np.random.uniform(size=num_chosen_subhalos)*2*np.pi
        theta = np.random.uniform(size=num_chosen_subhalos)*np.pi
        r = np.random.uniform(size=num_chosen_subhalos)
        r_uniform_dense = gen_dist(size=num_chosen_subhalos)


        x = r * np.cos(phi) * np.sin(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(theta)
        pos = np.array([x,y,z]).T
        r_med = np.median(r)
        rms = get_smallest_rms(pos, num_random_points = 10000)
        rmss.append(rms/r_med)

        x_uniform_dense = r_uniform_dense * np.cos(phi) * np.sin(theta)
        y_uniform_dense = r_uniform_dense * np.sin(phi) * np.sin(theta)
        z_uniform_dense = r_uniform_dense * np.cos(theta)
        pos_uniform_dense = np.array([x_uniform_dense,y_uniform_dense,z_uniform_dense]).T
        r_med_uniform_dense = np.median(r_uniform_dense)
        rms_uniform_dense = get_smallest_rms(pos_uniform_dense, num_random_points = 10000)
        rmss_uniform_dense.append(rms_uniform_dense/r_med_uniform_dense)
        
        temp_range = np.random.choice(np.arange(len(new_X)), size=num_chosen_subhalos, replace=False)
        x_subhalo = new_X[temp_range]
        y_subhalo = new_Y[temp_range]
        z_subhalo = new_Z[temp_range]
        pos_subhalo = np.array([x_subhalo,y_subhalo,z_subhalo]).T
        r_subhalo = (x_subhalo**2+y_subhalo**2+z_subhalo**2)**(1/2)
        r_med_subhalo = np.median(r_subhalo)
        rms_subhalo = get_smallest_rms(pos_subhalo, num_random_points = 10000)
        rmss_subhalo.append(rms_subhalo/r_med_subhalo)
        
        x_subhalo_isotropized = r_subhalo * np.cos(phi) * np.sin(theta)
        y_subhalo_isotropized = r_subhalo * np.sin(phi) * np.sin(theta)
        z_subhalo_isotropized = r_subhalo * np.cos(theta)
        pos_subhalo_isotropized = np.array([x_subhalo_isotropized,y_subhalo_isotropized,z_subhalo_isotropized]).T
        rms_subhalo_isotropized = get_smallest_rms(pos_subhalo_isotropized, num_random_points = 10000)
        rmss_subhalo_isotropized.append(rms_subhalo_isotropized/r_med_subhalo)
        
        
        rs_subhalo.append(r_med_subhalo)
        rmss_over_r_med_subhalo.append(rms_subhalo/r_med_subhalo)


# In[ ]:


if len(new_X) >= 11:
    kernel = gaussian_kde(rmss)
    kernel_uniform_dense = gaussian_kde(rmss_uniform_dense)
    kernel_subhalo = gaussian_kde(rmss_subhalo)
    kernel_subhalo_isotropized = gaussian_kde(rmss_subhalo_isotropized)
    
    
    x = np.linspace(0, 1, num = 1000)
    fig, ax = plt.subplots()
    ax.plot(x, kernel(x), label="Isotropic and uniform in radius")
    ax.plot(x, kernel_uniform_dense(x), label="Uniformly dense")
    ax.plot(x, kernel_subhalo(x), label=f"{suite_name}")
    ax.plot(x, kernel_subhalo_isotropized(x), label=f"{suite_name} isotropized")
    ax.set_title('The distribution of the rms dispersions of satellite distribution around their best fit planes')
    ax.set_xlabel('$D_{\\textrm{rms}}/R_{\\textrm{med}}$')
    ax.set_ylabel('P($D_{\\textrm{rms}}/R_{\\textrm{med}}$)')
    ax.legend()
    plt.savefig(f"../../result/data/{suite_name}/distribution_rms_dispersion_for_{suite_name}.pdf")


# In[ ]:


if len(new_X) >= 11:
    fig, ax = plt.subplots()
    ax.scatter(rs_subhalo, rmss_over_r_med_subhalo)
    ax.set_title('rms dispersions of satellite vs median of radius')
    ax.set_xlabel('$R_{\\textrm{med}}/R_{\\textrm{vir}}$')
    ax.set_ylabel('$D_{\\textrm{rms}}/R_{\\textrm{med}}$')
    plt.savefig(f"../../result/data/{suite_name}/rms_dispersion_vs_median_radius_for_{suite_name}.pdf")

