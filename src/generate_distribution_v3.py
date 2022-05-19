import os
from sys import argv

from modules.helper_functions_v3 import generate_distribution_with_surv_probs, elvis_name_template, caterpillar_name_template

import __main__

prev_suite_dir = '../data/'
data_dir = '../data/gendata/'

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
  temp_dir = f'{data_dir}/log_elvis_isolated{_300kpc}_surv_probs_v3.1'
  suite_name_decorated = elvis_name_template.substitute(suite_name=suite_name)
else:
  suite_dir = f'{prev_suite_dir}/caterpillar_zrei8_5_fix'
  temp_dir = f'{data_dir}/log_caterpillar{_300kpc}_surv_probs_v3.1'
  suite_name_decorated = caterpillar_name_template.substitute(suite_name=suite_name)

if not os.path.isdir(temp_dir):
  os.makedirs(temp_dir)

filename = f'{temp_dir}/{suite_name}.csv'

generate_distribution_with_surv_probs(suite_name_decorated, filename, suite_dir, iterations=250000, chunk_size=200, select_by_Rvir = False)
