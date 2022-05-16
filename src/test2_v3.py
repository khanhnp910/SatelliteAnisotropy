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
  filename = f'../../Data/log_brightest_elvis_isolated{_300kpc}_v3.1/{suite_name}.csv'
  suite_name_decorated = elvis_name_template.substitute(suite_name=suite_name)
else:
  suite_dir = f'../../caterpillar_zrei8_5_fix'
  filename = f'../../Data/log_brightest_caterpillar{_300kpc}_v3.1/{suite_name}.csv'
  suite_name_decorated = caterpillar_name_template.substitute(suite_name=suite_name)

data = read_halo(suite_name_decorated, suite_dir)
dic = extract_inside(data, select_by_Rvir=select_by_Rvir)

with open(filename, "w", newline='') as file:
  writer = csv.writer(file, delimiter=',')
  writer.writerow(["D_rms", "R_med", "D_sph_11","D_sph_10","D_sph_9","D_sph_8","D_sph_7","D_sph_6","D_sph_5","D_sph_4","D_sph_3"])

iterations = 50000
# chunk_size = 200
# num_time = iterations // chunk_size

t0 = time.time()

for _ in range(iterations):
  data_brightest = read_specific(data, select_by_Rvir=select_by_Rvir)

  D_rms = get_smallest_D_rms(data_brightest['pos'])['D_rms']
  R_med = get_R_med(data_brightest['pos'])['R_med']

  D_sphs = []

  for k in range(11, 2, -1):
    indices = np.array(list(combinations(np.arange(11),k)))

    poles = data_brightest['poles'][indices]

    D_sphs.append(np.around(np.min(get_D_sph(poles)['D_sph']),3))

  with open(filename, "a", newline='') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow([np.around(D_rms,3), np.around(R_med,3)]+D_sphs)

print(time.time()-t0)