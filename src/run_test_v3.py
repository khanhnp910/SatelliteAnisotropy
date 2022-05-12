import os

state = True

if state:
  caterpillar_dir = '../../caterpillar_zrei8_5_fix'
  caterpillar_names = [name for name in os.listdir(caterpillar_dir) if os.path.isdir(caterpillar_dir+'/'+name)]

  elvis_dir = '../../elvis_isolated'
  elvis_names = [name for name in os.listdir(elvis_dir) if os.path.isdir(elvis_dir+'/'+name)]

  suite_names = elvis_names + caterpillar_names

  for suite_name in suite_names:
    print(f'Running for {suite_name}')
    os.system(f'python test_v3.py {suite_name} 1')
    # os.makedirs(f'../../result_v3/caterpillar/{suite_name}')

