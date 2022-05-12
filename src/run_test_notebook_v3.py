import os

state = True

if state:
  filename = 'test_notebook_v3'
  os.system(f'jupyter nbconvert {filename}.ipynb --to python')
  caterpillar_dir = '../../caterpillar_zrei8_5_fix'
  caterpillar_names = [name for name in os.listdir(caterpillar_dir) if os.path.isdir(caterpillar_dir+'/'+name)]

  elvis_dir = '../../elvis_isolated'
  elvis_names = [name for name in os.listdir(elvis_dir) if os.path.isdir(elvis_dir+'/'+name)]

  suite_names = elvis_names + caterpillar_names

  for suite_name in suite_names:
    print(f'Running for {suite_name}')
    os.system(f'python {filename}.py {suite_name} 1')

  os.system(f'rm {filename}.py')