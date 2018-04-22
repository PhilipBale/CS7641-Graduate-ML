import matplotlib.pyplot as plt
import pandas

colors = ['r', 'b', 'g', 'k', 'm', 'c', 'y', '#eeefff', '.75', '.25']


def get_df(path, headerVal=0):
  return pandas.read_csv('../../results/' + path, header=headerVal)



def find_best_q_learning(difficulty):
  bestResults = list()

  for lr in [0.1,0.9]:
    for qInit in [-100,0,100]:
      for epsilon in [0.1,0.3,0.5]:
        params = 'L' + str(lr) + ' q' + str(qInit) + '.0 E' + str(epsilon)
        path = difficulty + '/' + difficulty.title() + ' Q-Learning ' + params + '.csv'
        # print(path)
        df = get_df(path)

        df = df.sort_values('iter', ascending=False)
        max_reward = df['reward'].idxmax()
        max_row = df.ix[max_reward]
        max_row['params'] = params
        bestResults.append(max_row)

  df = pandas.DataFrame(bestResults)
  df = df.sort_values('iter', ascending=False)
  df = df.sort_values('reward', ascending=False)
  print(df)

  count = 0
  for index, row in df.iterrows():
    count = count + 1
    if count > 10:
      continue

    sig = 4
    sigString = '{:0.' + str(sig) + 'f}'
    vals = map(lambda x: sigString.format(x).replace(".0000", ''), [ row['iter'], row['time'], row['reward'], row['steps'], row['convergence'] ])
    print('\\textbf{' + row['params'] + '} & ' + ' & '.join(vals) + ' \\\\ \\hline')


def print_value_results(difficulty, policy):
  path = difficulty + '/' + difficulty.title() + ' ' + policy + '.csv' 
  print(path)
  df = get_df(path)

  count = 0
  for index, row in df.iterrows():
    if count == 0 or (count + 1) % 5 == 0 or count + 1 == len(df):
      sig = 4
      sigString = '{:0.' + str(sig) + 'f}'
      vals = map(lambda x: sigString.format(x).replace(".0000", ''), [ row['iter'], row['time'], row['reward'], row['steps'], row['convergence'] ])
      print(' & '.join(vals) + ' \\\\ \\hline')

    count = count + 1
  print('')





find_best_q_learning('easy')
print(' ')
find_best_q_learning('hard')

print_value_results('easy', 'Value')
print_value_results('easy', 'Policy')

print_value_results('hard', 'Value')
print_value_results('hard', 'Policy')