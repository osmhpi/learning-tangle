import itertools
import random

import numpy as np

# https://docs.iota.org/docs/node-software/0.1/iri/references/iri-configuration-options
ALPHA = 0.001

class TipSelector:
    def __init__(self, tangle):
        self.tangle = tangle

        # Build a map of transactions that directly approve a given transaction
        self.approving_transactions = {x: [] for x in self.tangle.transactions}
        for x, tx in self.tangle.transactions.items():
            for unique_parent in tx.parents:
                self.approving_transactions[unique_parent].append(x)

        self.ratings = self.compute_ratings(self.approving_transactions)

    def tip_selection(self, num_tips):
        # https://docs.iota.org/docs/node-software/0.1/iri/concepts/tip-selection

        # The docs say entry_point = latestSolidMilestone - depth. Ignoring these concepts for now.
        entry_point = self.tangle.genesis

        tips = []
        for i in range(num_tips):
             tips.append(self.walk(entry_point, self.ratings, self.approving_transactions))

        return tips

    def compute_ratings(self, approving_transactions):
        rating = {}
        future_set_cache = {}
        for tx in self.tangle.transactions:
            rating[tx] = len(self.future_set(tx, approving_transactions, future_set_cache)) + 1

        return rating

    def future_set(self, tx, approving_transactions, future_set_cache):
        def recurse_future_set(t):
            if t not in future_set_cache:
                direct_approvals = set(approving_transactions[t])
                future_set_cache[t] = direct_approvals.union(*[recurse_future_set(x) for x in direct_approvals])

            return future_set_cache[t]

        return recurse_future_set(tx)

    def walk(self, tx, ratings, approving_transactions):
        step = tx
        prev_step = None

        while step:
            approvers = approving_transactions[step]
            prev_step = step
            step = self.next_step(ratings, approvers)

        # When there are no more steps, this transaction is a tip
        return prev_step

    def next_step(self, ratings, approvers):
        approvers_with_rating = approvers  # There is a rating for every possible approver

        # There is no valid approver, this transaction is a tip
        if len(approvers_with_rating) == 0:
            return None

        approvers_ratings = [ratings[a] for a in approvers_with_rating]
        weights = self.ratings_to_weight(approvers_ratings)
        approver = self.weighted_choice(approvers_with_rating, weights)

        # Skip validation.
        # At least a validation of some PoW is necessary in a real-world implementation.

        return approver

        # if approver is not None:
        #     tail = validator.findTail(approver)
        #
        #     # If the selected approver is invalid, step back and try again
        #     if validator.isInvalid(tail):
        #         approvers = approvers.remove(approver)
        #
        #         return self.next_step(ratings, approvers)
        #
        #     return tail
        #
        # return None

    @staticmethod
    def weighted_choice(approvers, weights):
        # Instead of a random choice, one could also think about a more 'intelligent'
        # variant for this use case. E.g. choose a transaction that was published by a
        # node with 'similar' characteristics

        rn = random.uniform(0, sum(weights))
        for i in range(len(approvers)):
            rn -= weights[i]
            if rn <= 0:
                return approvers[i]
        return approvers[-1]

    @staticmethod
    def ratings_to_weight(ratings):
        highest_rating = max(ratings)
        normalized_ratings = [r - highest_rating for r in ratings]
        return [np.exp(r * ALPHA) for r in normalized_ratings]

    @staticmethod
    def ratings_to_probability(ratings):
        # Calculating a probability according to the IOTA randomness blog
        # https://blog.iota.org/alpha-d176d7601f1c
        b = sum(map(lambda r: np.exp(ALPHA * r),ratings))
        return [np.exp(r * ALPHA) / b for r in ratings]
