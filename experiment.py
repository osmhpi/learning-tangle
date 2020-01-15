#! /usr/bin/env python

import os
import itertools
import subprocess

import numpy as np
import tensorflow as tf
from tensorflow_federated import python as tff

from tangle import Tangle, Node, Transaction, Model

NUM_CLIENTS_PER_ROUND = 5

def evaluate(tangle, dataset_iter):
    # To compute the 'current performance' of the collaboratively trained model, create a node and let it pick a model
    evaluation_node = Node(None, tangle)
    evaluation_data = next(dataset_iter)
    return evaluation_node.compute_current_loss(tf.data.Dataset.from_tensor_slices(
      {
        'pixels': [p for p in evaluation_data['pixels']],
        'label': [l for l in evaluation_data['label']]
      }
    ))[0]

def run():
    os.makedirs('tangle_data/transactions', exist_ok=True)
    genesis = Transaction(Model().get_weights(), [], tag=0)
    tangle = Tangle({genesis.name(): genesis}, genesis.name())
    tangle.save(0, 100)

    # For visualization purposes
    global_loss = []
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
    testdataset_iter = iter(tff.simulation.datasets.build_synthethic_iid_datasets(emnist_test, 100))

    nodes = os.listdir('data')

    rounds = int(len(nodes) / NUM_CLIENTS_PER_ROUND)

    # Organize transactions in artificial 'rounds'
    for rnd in range(rounds):

        print(f"Starting round {rnd + 1} / {rounds}")

        # In each round, a set of nodes performs a training step and potentially publishes the result as a transaction.
        # Why would nodes continuously publish updates (in the real world)?
        # The stability of the tangle results from a continuous stream of well-behaved updates
        # even if they only provide a negligible improvement of the model.

        selected_nodes = np.random.choice(nodes, NUM_CLIENTS_PER_ROUND, replace=False)

        processes = []
        for n in selected_nodes:
            processes.append(subprocess.Popen(["./step.py", n, str(rnd)], stdout=subprocess.PIPE))

        # Interleave training and evaluation of the previous round
        global_loss.append(evaluate(tangle, testdataset_iter))

        for p in processes:
            out, err = p.communicate()
            result = out.decode('utf-8').strip()
            if len(result) > 0:
                parts = result.split()
                tangle.add_transaction(Transaction(None, parts[1:], parts[0], rnd+1))


        # tangle.show()
        tangle.save(rnd+1, global_loss)

if __name__ == '__main__':
    run()
