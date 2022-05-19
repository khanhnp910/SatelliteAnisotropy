import os
from os.path import join, dirname
import config

state = True

if state:
  filenames = []
  
  if config.generate_with_surv_probs or config.generate_without_surv_probs:
    filenames.append('generate_distribution_v3.py')

  if config.generate_brightest:
    filenames.append('generate_brightest_distribution_v3.py')

  for filename in filenames:
    filename_expanded = join(dirname(__file__), filename)

    for suite_name in config.get_suite_names():
      print(f'Running for {suite_name}')
      os.system(f'python {filename_expanded} {suite_name} 1')

