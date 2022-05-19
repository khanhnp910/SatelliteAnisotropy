import os

state = True

if state:
  filename = 'generate_distribution_v3'

  elvis_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'elvis_isolated')
  elvis_names = [] 
  
  
  for name in os.listdir(elvis_dir):
    split = name.split('_')
    elvis_names.append(split[0])
  print(elvis_names)
  
  # caterpillar_dir = '../data/caterpillar_zrei8_5_fix'
  caterpillar_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'caterpillar_zrei8_5_fix')
  caterpillar_names = [] 
    
  for name in os.listdir(caterpillar_dir):
    split = name.split('_zrei')
    caterpillar_names.append(split[0][:-5])
  print(caterpillar_names)
    
  
  suite_names = caterpillar_names + elvis_names

  for suite_name in suite_names:
    print(f'Running for {suite_name}')
    os.system(f'python {filename}.py {suite_name} 1')
    # os.makedirs(f'../../result_v3/caterpillar/{suite_name}')

