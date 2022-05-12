from os import listdir, system
from sys import argv

import time 

t0 = time.time()

from modules.helper_functions_v3 import *
from modules.stats_v3 import *

import csv
import __main__


# Insert suite name here
if len(argv) > 1:
  suite_name = argv[1]
  select_by_Rvir = True if argv[2] == '0' else False
  _300kpc = "" if select_by_Rvir else "_300kpc"
else:
  suite_name = 'iBurr'
  select_by_Rvir = False
  _300kpc = "" if select_by_Rvir else "_300kpc"

if suite_name[0] == 'i':
  suite_dir = f'../../elvis_isolated'
  filename = f'../../Data/log_elvis_isolated{_300kpc}_surv_probs_v3/{suite_name}.csv'
  suite_name_decorated = elvis_name_template.substitute(suite_name=suite_name)
else:
  suite_dir = f'../../caterpillar_zrei8_5_fix'
  filename = f'../../Data/log_caterpillar{_300kpc}_surv_probs_v3/{suite_name}.csv'
  suite_name_decorated = caterpillar_name_template.substitute(suite_name=suite_name)

data = read_halo(suite_name_decorated, suite_dir)
dic = extract_inside(data, select_by_Rvir=select_by_Rvir)

with open(filename, "w", newline='') as file:
  writer = csv.writer(file, delimiter=',')
  writer.writerow(["D_rms", "R_med", "D_sph_11","D_sph_10","D_sph_9","D_sph_8","D_sph_7","D_sph_6","D_sph_5","D_sph_4","D_sph_3","D_sph_11_f","D_sph_10_f","D_sph_9_f","D_sph_8_f","D_sph_7_f","D_sph_6_f","D_sph_5_f","D_sph_4_f","D_sph_3_f","ID0","ID1","ID2","ID3","ID4","ID5","ID6","ID7","ID8","ID9","ID10"])

iterations = 250000
chunk_size = 200
num_time = iterations // chunk_size

pos = dic['pos']
poles = dic['poles']
surv_probs = dic['surv_probs']

size = len(pos)

# shape (num_random, 3)
# normal_vectors = sample_spherical_pos(num_random)

# shape (size, num_random)
# angles = np.matmul(poles, normal_vectors.T)

t0 = time.time()
for _ in range(num_time):
  # (size,)
  random_probs = np.random.rand(size)

  temp = random_probs < surv_probs

  new_indices = np.arange(size)[temp]

  # print(f'{suite_name}: {len(new_indices)}')

  if len(new_indices) < 11:
    continue

  # (chunk_size, 11)
  indices = new_indices[random_choice_noreplace(chunk_size, len(new_indices), 11)]

  # (chunk_size, 11, 3)
  chosen_pos = pos[indices]

  # (chunk_size, 11, 3)
  chosen_poles = poles[indices]

  # (chunk_size, 11, num_random)
  # chosen_angles = angles[indices]

  d_angles = []
  d_angles_f = []
  
  for k in range(11,2,-1):
    # shape (chunk_size, 11, 3)
    indices_ = np.array(list(combinations(np.arange(11),k)))

    chosen_poles_ = chosen_poles[:, indices_,:]
    # chosen_angles_ = chosen_angles[:, indices_,:]

    d_angles.append(np.min(np.around(get_D_sph(chosen_poles_)['D_sph'], decimals=3), axis=-1))

  arr = np.array([np.around(get_smallest_D_rms(chosen_pos, num_random=5000)['D_rms'], decimals=3), np.around(get_R_med(chosen_pos)['R_med'], decimals=3)]+d_angles+list(np.zeros_like(d_angles, dtype='float'))).T
  arr = np.concatenate((arr, indices), axis=1)

  with open(filename, "a+", newline='') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerows(arr)

print(time.time()-t0)