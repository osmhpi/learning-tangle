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
      if selector is None:
          selector = TipSelector(self.tangle)

      if len(self.tangle.transactions) < selector.num_tips:
          return [self.tangle.transactions[self.tangle.genesis] for i in range(selector.num_tips)]
      tips = selector.tip_selection()

      return [self.tangle.transactions[tip] for tip in tips]

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
          tips = self.choose_tips(selector=selector)
          for tip in tips:
              for tx in approved_transactions(tip.name()):
                  transaction_confidence[tx] += 1

      # todo vorher hardcoded 2, warum genau?
      return {tx: float(transaction_confidence[tx]) / (num_sampling_rounds * selector.num_tips) for tx in self.tangle.transactions}

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
    s = sum(params)
    return sum(params) / len(params)
    #return [np.array(p).mean(axis=0) for p in zip(params)]

  def process_next_batch(self, num_epochs, batch_size, num_tips=2):
    selector = TipSelector(self.tangle, num_tips)

    # Compute reference metrics
    reference = self.obtain_reference_params(selector=selector)
    self.client.model.set_params(reference)
    c_metrics = self.client.test('test')

    # Obtain two tips from the tangle
    tips = self.choose_tips(selector=selector)

    if self.malicious == MaliciousType.RANDOM:
        weights = self.client.model.get_params()
        malicious_weights = [np.random.RandomState().normal(size=w.shape) for w in weights]
        print('generated malicious weights')
        return Transaction(malicious_weights, set([tip.name() for tip in tips]), malicious=True), None, None
    elif self.malicious == MaliciousType.LABELFLIP:
        self.client.model.set_params(reference)
        self.client.train(num_epochs, batch_size)
        return Transaction(self.client.model.get_params(), set([tip.name() for tip in tips]), malicious=True), None, None
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
        averaged_weights = self.average_model_params(*[tip.load_weights() for tip in tips])
        self.client.model.set_params(averaged_weights)
        comp, num_samples, update = self.client.train(num_epochs, batch_size)

        c_averaged_model_metrics = self.client.test('test')
        if c_averaged_model_metrics['loss'] < c_metrics['loss']:
            return Transaction(self.client.model.get_params(), set([tip.name() for tip in tips])), c_averaged_model_metrics, comp

    return None, None, None
