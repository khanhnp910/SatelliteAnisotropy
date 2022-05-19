import os
from os.path import join, dirname
import config 

state = True

if state:
  filename = join(dirname(__file__), 'test_notebook_v3')
  os.system(f'jupyter nbconvert {filename} --to python')

  for suite_name in config.get_suite_names():
    print(f'Running for {suite_name}')
    os.system(f'python {filename}.py {suite_name} 1')

  os.system(f'rm {filename}.py')