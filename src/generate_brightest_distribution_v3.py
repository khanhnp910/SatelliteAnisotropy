from os import listdir, system
from sys import argv

import time 

t0 = time.time()

from modules.helper_functions_v3 import generate_brightest_distribution_with_surv_probs, caterpillar_name_template, elvis_name_template

import __main__

prev_suite_dir = '../..'
data_dir = '../../Data'

if len(argv) > 1:
  suite_name = argv[1]
  select_by_Rvir = True if argv[2] == '0' else False
  _300kpc = "" if select_by_Rvir else "_300kpc"

else:
  suite_name = 'iBurr'
  select_by_Rvir = False
  _300kpc = "" if select_by_Rvir else "_300kpc"

if suite_name[0] == 'i':
  suite_dir = f'{prev_suite_dir}/elvis_isolated'
  filename = f'{data_dir}/log_brightest_elvis_isolated{_300kpc}_v3.2/{suite_name}.csv'
  suite_name_decorated = elvis_name_template.substitute(suite_name=suite_name)
else:
  suite_dir = f'{prev_suite_dir}/caterpillar_zrei8_5_fix'
  filename = f'{data_dir}/log_brightest_caterpillar{_300kpc}_v3.2/{suite_name}.csv'
  suite_name_decorated = caterpillar_name_template.substitute(suite_name=suite_name)

generate_brightest_distribution_with_surv_probs(suite_name_decorated, filename, suite_dir, iterations=50000, select_by_Rvir = False)