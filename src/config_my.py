from os.path import dirname, join, isdir
from os import makedirs, listdir
from string import Template

# Directory to the folder containing elvis_isolated and caterpillar raw data
#raw_dir = dirname(dirname(dirname(__file__)))
raw_dir = 'C:/Users/akrav/Documents/prj/2205_sat_anisotropy/data/'
#print(raw_dir)
# Directory to the folder containing generated data
#gendata_dir = join(dirname(dirname(dirname(__file__))), '2205_sat_anisotropy/gendata')
gendata_dir = 'C:/Users/akrav/Documents/prj/2205_sat_anisotropy/gendata/'

if not isdir(gendata_dir):
  makedirs(gendata_dir)

# Directory to results
result_dir = join(dirname(dirname(dirname(__file__))), '2205_sat_anisotropy/gendata')

if not isdir(result_dir):
  makedirs(result_dir)

# Path to the MW data
MW_path = join(dirname(dirname(dirname(__file__))), '2205_sat_anisotropy/data', 'pawlowski_tab2.csv')

# Name of the folder containing elvis_isolated raw data
elvis_isolated_raw_name = 'elvis_isolated'

# Name of the folder containing caterpillar raw data
caterpillar_raw_name = 'caterpillar_zrei8_5_fix'

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
MASS_CUTOFF = 5e8

# Number of iterations for each generated sample for random 11 subhalos
ITERATIONS = 250000

# Number of iterations for each generated sample for 11 brightest subhalos
ITERATIONS_BRIGHTEST = 50000

# Chunk size for parallelization
CHUNK_SIZE = 200

# generate sample of random 11 subhalos without surv_probs
generate_without_surv_probs = True

# generate sample of random 11 subhalos with surv_probs
generate_with_surv_probs = True

# generate sample of brightest 11 subhalos with surv_probs
generate_brightest = True

# get all suite_names()
def get_suite_names():
  return get_elvis_isolated_names() + get_caterpillar_names()