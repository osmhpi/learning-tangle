#! /usr/bin/env python

import os
import itertools
import subprocess

import numpy as np
import tensorflow as tf

from tangle import Tangle, Node, Transaction, Model

NUM_CLIENTS = 5
NUM_ROUNDS = 10

def perform_training(node_id, tangle):
    node = Node(node_id, tangle)
    return node.process_next_batch()

def run():
    os.makedirs('tangle_data/transactions')
    genesis = Transaction(Model().get_weights(), [], tag=0)
    tangle = Tangle({genesis.name(): genesis}, genesis.name())
    tangle.save(0, 100)

    # For visualization purposes
    global_loss = []

    nodes = os.listdir('data')

    # Organize transactions in artificial 'rounds'
    for rnd in range(NUM_ROUNDS):

        print(f"Starting round {rnd + 1} / {NUM_ROUNDS}")

        # In each round, a set of nodes performs a training step and potentially publishes the result as a transaction.
        # Why would nodes continuously publish updates (in the real world)?
        # The stability of the tangle results from a continuous stream of well-behaved updates
        # even if they only provide a negligible improvement of the model.

        selected_nodes = np.random.choice(nodes, NUM_CLIENTS, replace=False)

        processes = []
        for n in selected_nodes:
            processes.append(subprocess.Popen(["./step.py", n, str(rnd)], stdout=subprocess.PIPE))

        for p in processes:
            out, err = p.communicate()
            result = out.decode('utf-8').strip()
            if len(result) > 0:
                parts = result.split()
                tangle.add_transaction(Transaction(None, parts[1:], parts[0], rnd+1))

        # global_loss.append(sum([loss for _, loss in new_transactions]) / len(selected_nodes))

        # tangle.show()
        tangle.save(rnd+1, global_loss)

if __name__ == '__main__':
    run()
