from sys import argv
from os.path import join, isdir
from os import makedirs
import config

from modules.helper_functions_v3 import generate_distribution, elvis_name_template, caterpillar_name_template


raw_dir = config.raw_dir
gendata_dir = config.gendata_dir

if len(argv) > 1:
  suite_name = argv[1]
  select_by_Rvir = True if argv[2] == '0' else False
  _300kpc = "" if select_by_Rvir else "_300kpc"
else:
  suite_name = 'iBurr'
  select_by_Rvir = False
  _300kpc = "" if select_by_Rvir else "_300kpc"

if suite_name[0] == 'i':
  suite_dir = join(raw_dir, config.elvis_isolated_raw_name)
  catalog = 'elvis_isolated'
  suite_name_decorated = elvis_name_template.substitute(suite_name=suite_name)
else:
  suite_dir = join(raw_dir, config.caterpillar_raw_name)
  catalog = 'caterpillar'
  suite_name_decorated = caterpillar_name_template.substitute(suite_name=suite_name)

if config.generate_with_surv_probs:
  temp_dir = join(gendata_dir, config.gendata_name_template.substitute(catalog=catalog, _300kpc=_300kpc, _surv_probs='_surv_probs'))

  if not isdir(temp_dir):
    makedirs(temp_dir)

  filename = join(temp_dir, f'{suite_name}.csv')

  generate_distribution(suite_name_decorated, filename, suite_dir, is_surv_probs=True, select_by_Rvir = select_by_Rvir)

if config.generate_without_surv_probs:
  temp_dir = join(gendata_dir, config.gendata_name_template.substitute(catalog=catalog, _300kpc=_300kpc, _surv_probs=''))

  if not isdir(temp_dir):
    makedirs(temp_dir)

  filename = join(temp_dir, f'{suite_name}.csv')

  generate_distribution(suite_name_decorated, filename, suite_dir, is_surv_probs=False, select_by_Rvir = select_by_Rvir)