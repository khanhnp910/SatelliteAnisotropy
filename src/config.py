from os.path import dirname, join, isdir
from os import makedirs, listdir
from string import Template
import configparser

parser = configparser.ConfigParser()
parser.read(join(dirname(__file__), 'config.ini'))

elvis_name_template = Template('${suite_name}_isolated_elvis_iso_zrei8_etan_18_etap_045_run.csv')
caterpillar_name_template = Template('${suite_name}_LX14_zrei8_5_fix_run.csv')

# Directory to the folder containing elvis_isolated and caterpillar raw data
raw_dir = parser['directories']['raw_dir']

# Directory to the folder containing generated data
gendata_dir = parser['directories']['gendata_dir']

if not isdir(gendata_dir):
  makedirs(gendata_dir)

# Directory to results
result_dir = parser['directories']['result_dir']

if not isdir(result_dir):
  makedirs(result_dir)

# Path to the MW data
MW_path = parser['directories']['MW_path']

# Name of the folder containing elvis_isolated raw data
elvis_isolated_raw_name = parser['raw_name']['elvis_isolated_raw_name']

# Name of the folder containing caterpillar raw data
caterpillar_raw_name = parser['raw_name']['caterpillar_raw_name']

# Name of folder containing generated data of random 11 subhalos with/without surv_probs with/without selected_by_Rvir
gendata_name_template = Template('log_${catalog}${_300kpc}${_surv_probs}_v3')

# Name of folder containing generated data of brightest 11 subhalos with surv_probs with/without selected_by_Rvir
gendata_brightest_name_template = Template('log_brightest_${catalog}${_300kpc}_v3.1')

# parse suite names
def get_caterpillar_names():
  caterpillar_dir = join(raw_dir, caterpillar_raw_name)
  caterpillar_names = []

  for filename in listdir(caterpillar_dir):
    if filename[-4:] == '.csv':
      caterpillar_names.append(filename.split('_zrei8')[0][:-5])
  
  return caterpillar_names

def get_elvis_isolated_names():
  elvis_isolated_dir = join(raw_dir, elvis_isolated_raw_name)
  elvis_isolated_names = []

  for filename in listdir(elvis_isolated_dir):
    if filename[-4:] == '.csv':
      elvis_isolated_names.append(filename.split('_isolated')[0])

  return elvis_isolated_names

# Mass Cutoff in M_sun
temp = parser['cutoff']['MASS_CUTOFF'].split('e')
MASS_CUTOFF = int(temp[0])*10**int(temp[1])

# Number of iterations for each generated sample for random 11 subhalos
ITERATIONS = int(parser['gen_paras']['ITERATIONS'])

# Number of iterations for each generated sample for 11 brightest subhalos
ITERATIONS_BRIGHTEST = int(parser['gen_paras']['ITERATIONS_BRIGHTEST'])

# Chunk size for parallelization
CHUNK_SIZE = int(parser['gen_paras']['CHUNK_SIZE'])

# generate sample of random 11 subhalos without surv_probs
generate_without_surv_probs = True if parser['gen_configs']['generate_without_surv_probs'] == 'True' else False

# generate sample of random 11 subhalos with surv_probs
generate_with_surv_probs = True if parser['gen_configs']['generate_with_surv_probs'] == 'True' else False

# generate sample of brightest 11 subhalos with surv_probs
generate_brightest = True if parser['gen_configs']['generate_brightest'] == 'True' else False

# get all suite_names()
def get_suite_names():
  return get_elvis_isolated_names() + get_caterpillar_names()

if __name__ == '__main__':
  print(generate_brightest)