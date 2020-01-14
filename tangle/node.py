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

  def compute_confidence(self, selector=None):
      num_sampling_rounds = 10

      transaction_confidence = {x: 0 for x in self.tangle.transactions}

      approved_transactions_cache = {}

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

      return {tx: float(transaction_confidence[tx]) / num_sampling_rounds for tx in self.tangle.transactions}

  def compute_cumulative_score(self, transactions):
      cumulative_score = {}

      def compute_score(transaction):
          if transaction in cumulative_score:
              return cumulative_score[transaction]

          result = 1
          for unique_parent in self.tangle.transactions[transaction].parents:
              result += compute_score(unique_parent)

          cumulative_score[transaction] = result
          return result

      for t in transactions:
          compute_score(t)

      return {tx: cumulative_score[tx] for tx in transactions}

  def compute_current_loss(self, data, selector=None):
      # Establish the 'current best' weights from the tangle

      # 1. Perform tip selection n times, establish confidence for each transaction
      transaction_confidence = self.compute_confidence(selector=selector)

      # 2. Compute cumulative score for transactions with confidence greater than threshold
      approved_transactions = [tx for tx, confidence in transaction_confidence.items() if confidence >= 0.5]
      if len(approved_transactions) < 50:  # Below some threshold of transactions, we cannot 'trust' the network
          approved_transactions = [tx for tx, confidence in
                                    sorted(transaction_confidence.items(), key=lambda kv: kv[1], reverse=True)[:5]]
      scores = self.compute_cumulative_score(approved_transactions)

      # 3. For the top n percent of scored transactions run model evaluation and choose the best
      top_five = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:5]  # Sort by high score -> low score
      evaluated_tx = {tx[0]: Model(self.tangle.transactions[tx[0]].load_weights()).evaluate(data) for tx in top_five}
      return sorted(evaluated_tx.items(), key=lambda tx: tx[1])[0][1]

  def process_next_batch(self):
    train_data = Model.load_dataset(self.id, 'train')
    test_data = Model.load_dataset(self.id, 'test')

    selector = TipSelector(self.tangle)

    current_loss = self.compute_current_loss(test_data, selector)

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
    averaged_model = Model(tip1.load_weights()).average(Model(tip2.load_weights()))

    averaged_model.train(train_data)

    if averaged_model.performs_better_than(current_loss, test_data):
        return Transaction(averaged_model.get_weights(), set([tip1.name(), tip2.name()])), current_loss[0]

    return None, current_loss[0]

