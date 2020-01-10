import itertools
import random

import numpy as np

# https://docs.iota.org/docs/iri/0.1/references/iri-configuration-options#alpha
ALPHA = 0.001

class TipSelector:
    def __init__(self, tangle):
        self.tangle = tangle

        # Build a map of transactions that directly approve a given transaction
        self.approving_transactions = {x: [] for x in self.tangle.transactions}
        for x, tx in self.tangle.transactions.items():
            for unique_parent in tx.parents:
                self.approving_transactions[unique_parent].append(x)

        self.ratings = self.calculate_cumulative_weight(self.approving_transactions)

    def tip_selection(self):
        # https://docs.iota.org/docs/node-software/0.1/iri/concepts/tip-selection

        # The docs say entry_point = latestSolidMilestone - depth. Ignoring these concepts for now.
        entry_point = self.tangle.genesis

        entry_point_trunk = entry_point
        entry_point_branch = entry_point  # reference or entry_point, according to the docs

        # TODO: I don't understand the difference between trunk and branch
        trunk = self.walk(entry_point_trunk, self.ratings, self.approving_transactions)
        branch = self.walk(entry_point_branch, self.ratings, self.approving_transactions)

        return trunk, branch

    def calculate_cumulative_weight(self, approving_transactions):
        rating = {}
        for tx in self.tangle.transactions:
            rating[tx] = len(self.future_set(tx, approving_transactions)) + 1

        return rating

    def future_set(self, tx, approving_transactions):
        def recurse_future_set(t):
            direct_approvals = approving_transactions[t]
            indirect_approvals = [recurse_future_set(x) for x in approving_transactions[t]]
            return list(itertools.chain.from_iterable([direct_approvals] + indirect_approvals))

        return set(recurse_future_set(tx))

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

        rn = random.randint(0, int(sum(weights)))
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
