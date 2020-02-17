import os
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

for p in ParameterGrid(params):
    print(p)
    shutil.rmtree('/tangle_data', ignore_errors=True)
    os.system("python3 main.py -dataset %s -model %s --num-rounds %s --eval-every %s --clients-per-round %s --num-tips %s --sample-size %s --reference-avg-top %s --target-accuracy %s -lr %s" %
        (p['dataset'], p['model'], p['num_rounds'], p['eval_every'], p['clients_per_round'], p['num_tips'], p['sample_size'], p['reference_avg_top'], p['target_accuracy'], p['learning_rate']))