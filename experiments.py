import os
import shutil
from sklearn.model_selection import ParameterGrid

params = {
    'dataset': ['femnist'],
    'model': ['cnn'],
    'num_rounds': [100],
    'eval_every': [10],
    'clients_per_round':  [15],
    'num_tips':  [2,3,5],
    'sample_size':  [2,5,10],
    'reference_avg_top':  [1,2,10,50,100],
    'target_accuracy':  [0.7],
    'learning_rate':  [0.06]
}

file_name = 'results.txt'
    
for p in ParameterGrid(params):
    os.system('rm -rf tangle_data')
    command = "python3 main.py -dataset %s -model %s --num-rounds %s --eval-every %s --clients-per-round %s --num-tips %s --sample-size %s --reference-avg-top %s --target-accuracy %s -lr %s"
    parameters = (p['dataset'], p['model'], p['num_rounds'], p['eval_every'], p['clients_per_round'], p['num_tips'], p['sample_size'], p['reference_avg_top'], p['target_accuracy'], p['learning_rate'])
    command = command % parameters
    with open(file_name, 'a+') as file:
        file.write('\n\n' + command + '\n')
    os.system(command)
