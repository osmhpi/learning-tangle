import numpy as np
import tensorflow as tf
import sys

from .tip_selector import TipSelector
from .transaction import Transaction
from .malicious_type import MaliciousType

class Node:
  def __init__(self, client, tangle, malicious=MaliciousType.NONE):
    self.client = client
    self.tangle = tangle
    self.malicious = malicious

  def choose_tips(self, selector=None):
      if len(self.tangle.transactions) < 2:
          return self.tangle.transactions[self.tangle.genesis], self.tangle.transactions[self.tangle.genesis]

      if selector is None:
          selector = TipSelector(self.tangle)
      tip1, tip2 = selector.tip_selection()

      return self.tangle.transactions[tip1], self.tangle.transactions[tip2]

  def compute_confidence(self, selector=None, approved_transactions_cache={}):
      num_sampling_rounds = 10

      transaction_confidence = {x: 0 for x in self.tangle.transactions}

      def approved_transactions(transaction):
          if transaction not in approved_transactions_cache:
              result = set([transaction]).union(*[approved_transactions(parent) for parent in self.tangle.transactions[transaction].parents])
              approved_transactions_cache[transaction] = result

          return approved_transactions_cache[transaction]

      # Use a cached tip selector
      if selector is None:
          selector = TipSelector(self.tangle)

      for i in range(num_sampling_rounds):
          branch, trunk = self.choose_tips(selector=selector)
          for tx in approved_transactions(branch.name()):
              transaction_confidence[tx] += 1
          for tx in approved_transactions(trunk.name()):
              transaction_confidence[tx] += 1

      return {tx: float(transaction_confidence[tx]) / (num_sampling_rounds * 2) for tx in self.tangle.transactions}

  def compute_cumulative_score(self, transactions, approved_transactions_cache={}):
      def compute_approved_transactions(transaction):
          if transaction not in approved_transactions_cache:
              result = set([transaction]).union(*[compute_approved_transactions(parent) for parent in self.tangle.transactions[transaction].parents])
              approved_transactions_cache[transaction] = result

          return approved_transactions_cache[transaction]

      return {tx: len(compute_approved_transactions(tx)) for tx in transactions}

  def obtain_reference_params(self, selector=None):
      # Establish the 'current best'/'reference' weights from the tangle

      approved_transactions_cache = {}

      # 1. Perform tip selection n times, establish confidence for each transaction
      # (i.e. which transactions were already approved by most of the current tips?)
      transaction_confidence = self.compute_confidence(selector=selector, approved_transactions_cache=approved_transactions_cache)

      # 2. Compute cumulative score for transactions
      # (i.e. how many other transactions does a given transaction indirectly approve?)
      keys = [x for x in self.tangle.transactions]
      scores = self.compute_cumulative_score(keys, approved_transactions_cache=approved_transactions_cache)  # Todo: Reuse approved_transactions_cache from compute_confidence

      # 3. For the top 100 transactions, compute the average
      best = sorted(
          {tx: scores[tx] * transaction_confidence[tx] for tx in keys}.items(),
          key=lambda kv: kv[1], reverse=True
      )[0]
      return self.tangle.transactions[best[0]].load_weights()

  def average_model_params(self, *params):
    return [np.array(p).mean(axis=0) for p in zip(*params)]

  def process_next_batch(self, num_epochs, batch_size):
    selector = TipSelector(self.tangle)

    # Compute reference metrics
    reference = self.obtain_reference_params(selector=selector)
    self.client.model.set_params(reference)
    c_metrics = self.client.test('test')

    # Obtain two tips from the tangle
    tip1, tip2 = self.choose_tips(selector=selector)

    if self.malicious == MaliciousType.RANDOM:
        weights = self.client.model.get_params()
        malicious_weights = [np.random.normal(size=w.shape) for w in weights]
        # Todo Set identifiable ID
        return Transaction(malicious_weights, set([tip1.name(), tip2.name()])), None, None
    elif self.malicious == MaliciousType.LABELFLIP:
        self.client.model.set_params(reference)
        comp, num_samples, update = self.client.train(num_epochs, batch_size)
        return Transaction(self.client.model.get_params(), set([tip1.name(), tip2.name()])), None, None
    else:
        # Perform averaging

        # How averaging is done exactly (e.g. weighted, using which weights) is left to the
        # network participants. It is not reproducible or verifiable by other nodes because
        # only the resulting weights are published.
        # Once a node has published its training results, it thus can't be sure if
        # and by what weight its delta is being incorporated into approving transactions.
        # However, assuming most nodes are well-behaved, they will make sure that eventually
        # those weights will prevail that incorporate as many partial results as possible
        # in order to prevent over-fitting.

        # Here: simple unweighted average
        averaged_weights = self.average_model_params(tip1.load_weights(), tip2.load_weights())
        self.client.model.set_params(averaged_weights)
        comp, num_samples, update = self.client.train(num_epochs, batch_size)

        c_averaged_model_metrics = self.client.test('test')
        if c_averaged_model_metrics['loss'] < c_metrics['loss']:
            return Transaction(self.client.model.get_params(), set([tip1.name(), tip2.name()])), c_averaged_model_metrics, comp

    return None, None, None
