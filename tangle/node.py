import numpy as np
import tensorflow as tf

from .tip_selector import TipSelector
from .emnist_model import Model
from .transaction import Transaction

class Node:
  def __init__(self, id, tangle):
    self.id = id
    self.tangle = tangle

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

  def obtain_reference_model(self, selector=None):
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
      top_onehundred = sorted(
          {tx: scores[tx] * transaction_confidence[tx] for tx in keys}.items(),
          key=lambda kv: kv[1], reverse=True
      )[:50]
      return Model(self.tangle.transactions[top_onehundred[0][0]].load_weights()).average(
          *[self.tangle.transactions[x[0]].load_weights() for x in top_onehundred[1:]])

  def process_next_batch(self):
    train_data = Model.load_dataset(self.id, 'train')
    test_data = Model.load_dataset(self.id, 'test')

    selector = TipSelector(self.tangle)

    reference = self.obtain_reference_model(selector=selector)
    loss, accuracy = reference.evaluate(test_data)
    # Obtain two tips from the tangle
    tip1, tip2 = self.choose_tips(selector=selector)

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
    averaged_model = Model(tip1.load_weights()).average(tip2.load_weights())
    averaged_model.train(train_data)

    if averaged_model.performs_better_than(loss, test_data):
        return Transaction(averaged_model.get_weights(), set([tip1.name(), tip2.name()])), loss, accuracy

    return None, loss, accuracy

