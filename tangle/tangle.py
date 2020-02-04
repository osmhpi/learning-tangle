import json
import os
from multiprocessing import Pool, Process

import numpy as np

from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY

from .transaction import Transaction
from .node import Node

class Tangle:
    def __init__(self, transactions, genesis):
        self.transactions = transactions
        self.genesis = genesis

    def add_transaction(self, tip):
        self.transactions[tip.name()] = tip

    def run_nodes(self, train_fn, clients, rnd, num_epochs=1, batch_size=10):
        norm_this_round = []
        new_transactions = []

        sys_metrics = {
            c.id: {BYTES_WRITTEN_KEY: 0,
                   BYTES_READ_KEY: 0,
                   LOCAL_COMPUTATIONS_KEY: 0} for c in clients}

        train_params = [[client.id, client.group, client.model.flops, client.train_data, client.eval_data, rnd-1] for client in clients]

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        with Pool(50) as p:
            results = p.starmap(train_fn, train_params)

        for tx, metrics, comp, client_id, client_sys_metrics in results:
            if tx is None:
                continue

            sys_metrics[client_id][BYTES_READ_KEY] += client_sys_metrics[BYTES_READ_KEY]
            sys_metrics[client_id][BYTES_WRITTEN_KEY] += client_sys_metrics[BYTES_WRITTEN_KEY]
            sys_metrics[client_id][LOCAL_COMPUTATIONS_KEY] = client_sys_metrics[LOCAL_COMPUTATIONS_KEY]

            tx.tag = rnd
            new_transactions.append(tx)

        for tx in new_transactions:
            self.add_transaction(tx)

        return sys_metrics

    def test_model(self, clients_to_test, set_to_use='test'):
        metrics = {}

        for client in clients_to_test:
            node = Node(client, self)
            reference = node.obtain_reference_params()
            node.client.model.set_params(reference)
            c_metrics = node.client.test(set_to_use)
            metrics[client.id] = c_metrics

        return metrics

    def save(self, sequence_no, global_loss, global_accuracy, norm):
        n = [{'name': t.name(), 'time': t.tag, 'parents': list(t.parents)} for _, t in self.transactions.items()]

        with open(f'tangle_data/tangle_{sequence_no}.json', 'w') as outfile:
            json.dump({'nodes': n, 'genesis': self.genesis, 'global_loss': global_loss, 'global_accuracy': global_accuracy, 'norm': norm}, outfile)

    @classmethod
    def fromfile(cls, sequence_no):
      with open(f'tangle_data/tangle_{sequence_no}.json', 'r') as tanglefile:
          t = json.load(tanglefile)

      transactions = {n['name']: Transaction(None, set(n['parents']), n['name'], n['time']) for n in t['nodes']}
      return cls(transactions, t['genesis'])
