import json
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

    def run_nodes(self, clients, rnd, num_epochs=1, batch_size=10):
        norm_this_round = []
        new_transactions = []

        sys_metrics = {
            c.id: {BYTES_WRITTEN_KEY: 0,
                   BYTES_READ_KEY: 0,
                   LOCAL_COMPUTATIONS_KEY: 0} for c in clients}

        for client in clients:
            node = Node(client, self)
            tx, metrics, comp = node.process_next_batch(num_epochs, batch_size)

            if tx is None:
                continue

            sys_metrics[node.client.id][BYTES_READ_KEY] += node.client.model.size
            sys_metrics[node.client.id][BYTES_WRITTEN_KEY] += node.client.model.size
            sys_metrics[node.client.id][LOCAL_COMPUTATIONS_KEY] = comp

            tx.tag = rnd
            new_transactions.append(tx)

            # Compute norm
            if (len(tx.parents) == 2):
                parents = list(tx.parents)
                pw1 = self.transactions[parents[0]].load_weights()
                pw2 = self.transactions[parents[1]].load_weights()
                norm_this_round.append([np.linalg.norm(np.array(weights)[0]-np.array(weights)[1]) for weights in zip(pw1, pw2)])

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
