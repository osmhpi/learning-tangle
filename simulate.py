import collections
import json

import numpy as np
import tensorflow as tf

import tensorflow.keras.backend as K

import networkx as nx
import matplotlib.pyplot as plt

from multiprocessing.dummy import Pool

from tensorflow_federated import python as tff

NUM_EPOCHS = 10
BATCH_SIZE = 20
SHUFFLE_BUFFER = 500
BATCHES_PER_ROUND = 100

NUM_CLIENTS = 10

# https://docs.iota.org/docs/iri/0.1/references/iri-configuration-options#alpha
ALPHA = 0.001


class TipSelector:
    def __init__(self, tangle):
        self.tangle = tangle

    def tip_selection(self):
        # https://docs.iota.org/docs/the-tangle/0.1/concepts/tip-selection

        # The docs say entry_point = latestSolidMilestone - depth. Ignoring these concepts for now.
        entry_point = self.tangle.genesis

        entry_point_trunk = entry_point
        entry_point_branch = entry_point  # reference or entry_point, according to the docs

        # Build a map of transactions that directly approve a given transaction
        approving_transactions = {x: [] for x in self.tangle.transactions}
        for x in self.tangle.transactions:
            if x.p1 is not None:
                approving_transactions[x.p1].append(x)
            if x.p2 is not None and x.p1 != x.p2:
                approving_transactions[x.p2].append(x)

        ratings = self.calculate_cumulative_weight(approving_transactions)

        # TODO: I don't understand the difference between trunk and branch
        trunk = self.walk(entry_point_trunk, ratings, approving_transactions)
        branch = self.walk(entry_point_branch, ratings, approving_transactions)

        return trunk, branch

    def calculate_cumulative_weight(self, approving_transactions):
        rating = {}
        for tx in self.tangle.transactions:
            rating[tx] = len(self.future_set(tx, approving_transactions)) + 1

        return rating

    def future_set(self, tx, approving_transactions):
        direct_approvals = approving_transactions[tx]
        indirect_approvals = [self.future_set(x, approving_transactions) for x in direct_approvals]
        return {id(approver) for s in [direct_approvals, indirect_approvals] for approver in s}

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

        total_weight = sum(weights)
        return np.random.choice(approvers, p=[w / total_weight for w in weights])

    @staticmethod
    def ratings_to_weight(ratings):
        highest_rating = max(ratings)
        normalized_ratings = [r - highest_rating for r in ratings]
        return [np.exp(r * ALPHA) for r in normalized_ratings]


class Transaction:
    def __init__(self, weights, p1, p2):
        self.p1 = p1
        self.p2 = p2

        if p1 is not None and p2 is not None:
            self.height = max(p1.height, p2.height) + 1
        else:
            self.height = 1

        self.weights = weights
        self.tag = None

    def add_tag(self, tag):
        self.tag = tag


class Tangle:
    def __init__(self, genesis):
        self.genesis = genesis
        self.transactions = [genesis]

    def add_transaction(self, tip):
        self.transactions.append(tip)

    def show(self):
        graph = nx.DiGraph()

        graph.add_edges_from([(id(x), id(x.p1)) for x in self.transactions if x.p1 is not None])
        graph.add_edges_from([(id(x), id(x.p2)) for x in self.transactions if x.p2 is not None])

        val_map = {id(x): x.height for x in self.transactions}
        values = [val_map.get(node) for node in graph.nodes()]

        # Need to create a layout when doing
        # separate calls to draw nodes and edges
        pos = nx.spring_layout(graph)
        nx.draw_networkx_nodes(graph, pos, cmap=plt.get_cmap('jet'), node_color=values, node_size=100)
        nx.draw_networkx_labels(graph, pos)
        nx.draw_networkx_edges(graph, pos, edgelist=graph.edges(), arrows=False)
        plt.show()

    def save(self, sequence_no, global_loss):
        # Mark untagged transactions with the sequence number
        for t in self.transactions:
            if t.tag is None:
                t.add_tag(sequence_no + 1)  # Genesis block sequence number is 0

        node_ids = {id(self.transactions[i]): i for i in range(len(self.transactions))}
        n = [{'name': f'{i}', 'time': self.transactions[i].tag} for i in range(len(self.transactions))]
        edges = [
            *[{'source': f'{node_ids[id(x)]}', 'target': f'{node_ids[id(x.p1)]}'}
              for x in self.transactions if x.p1 is not None],
            *[{'source': f'{node_ids[id(x)]}', 'target': f'{node_ids[id(x.p2)]}'}
              for x in self.transactions if x.p2 is not None and x.p1 != x.p2]
        ]

        with open(f'viewer/tangle_{sequence_no}.json', 'w') as outfile:
            json.dump({'nodes': n, 'links': edges, 'global_loss': global_loss}, outfile)


class Model:
    def __init__(self, weights=None):
        self.model = self.create_compiled_keras_model()
        if weights is not None:
            self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def average(self, other):
        new_weights = [np.array(weights).mean(axis=0) for weights in zip(*[self.get_weights(), other.get_weights()])]
        return Model(new_weights)

    @staticmethod
    def preprocess(dataset):
        Example = collections.namedtuple('Example', ['x', 'y'])

        def element_fn(element):
            return Example(
                x=tf.reshape(element['pixels'], [-1]),
                y=tf.reshape(element['label'], [1]))

        return dataset.map(element_fn).apply(
            tf.data.experimental.shuffle_and_repeat(
                buffer_size=SHUFFLE_BUFFER, count=-1)).batch(BATCH_SIZE)
        # return dataset.repeat(NUM_EPOCHS).map(element_fn).shuffle(
        #     SHUFFLE_BUFFER).batch(BATCH_SIZE)

    @staticmethod
    def preprocess_test(dataset):
        Example = collections.namedtuple('Example', ['x', 'y'])

        def element_fn(element):
            return Example(
                x=tf.reshape(element['pixels'], [-1]),
                y=tf.reshape(element['label'], [1]))

        return dataset.map(element_fn).batch(100, drop_remainder=False)

    @staticmethod
    def create_compiled_keras_model():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(
                10, activation=tf.nn.softmax, kernel_initializer='zeros', input_shape=(784,))])

        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.02),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

        return model

    def train(self, data):
        self.model.fit(self.preprocess(data), steps_per_epoch=BATCHES_PER_ROUND, epochs=NUM_EPOCHS, verbose=0)

    def evaluate(self, data):
        return self.model.evaluate(self.preprocess_test(data))

    def performs_better_than(self, other_result, data):
        # print(self.model.metrics_names) tells us that:
        # self.evaluate() -> (loss, sparse_categorical_accuracy)
        # Not sure if can ignore the sparse_categorical_accuracy here
        return self.evaluate(data)[0] < other_result[0]


class Node:
    def __init__(self, tangle):
        self.tangle = tangle

    def choose_tips(self):
        if len(self.tangle.transactions) < 2:
            return tuple([self.tangle.genesis, self.tangle.genesis])

        selector = TipSelector(self.tangle)
        return selector.tip_selection()

    def compute_confidence(self):
        transaction_confidence = {id(x): 0 for x in self.tangle.transactions}

        def approved_transactions(transaction):
            txns = [id(transaction)]
            if transaction.p1 is not None:
                txns.extend(approved_transactions(transaction.p1))
            if transaction.p2 is not None and transaction.p1 != transaction.p2:
                txns.extend(approved_transactions(transaction.p2))

            return set(txns)

        for i in range(50):
            branch, trunk = self.choose_tips()
            for tx in approved_transactions(branch):
                transaction_confidence[tx] += 1
            for tx in approved_transactions(trunk):
                transaction_confidence[tx] += 1

        return {tx: transaction_confidence[id(tx)] for tx in self.tangle.transactions}

    @staticmethod
    def compute_cumulative_score(transactions):
        cumulative_score = {}

        def compute_score(transaction):
            if id(transaction) in cumulative_score:
                return cumulative_score[id(transaction)]

            result = 1
            if transaction.p1 is not None:
                result += compute_score(transaction.p1)
            if transaction.p2 is not None and transaction.p1 != transaction.p2:
                result += compute_score(transaction.p2)

            cumulative_score[id(transaction)] = result
            return result

        for t in transactions:
            compute_score(t)

        return {tx: cumulative_score[id(tx)] for tx in transactions}

    def compute_current_loss(self, data):
        # Establish the 'current best' weights from the tangle

        # 1. Perform tip selection n times, establish confidence for each transaction
        transaction_confidence = self.compute_confidence()

        # 2. Compute cumulative score for transactions with confidence greater than threshold
        approved_transactions = [tx for tx, confidence in transaction_confidence.items() if confidence > 50]
        if len(approved_transactions) < 50:  # Below some threshold of transactions, we cannot 'trust' the network
            approved_transactions = [tx for tx, confidence in
                                     sorted(transaction_confidence.items(), key=lambda kv: kv[1], reverse=True)[:5]]
        scores = self.compute_cumulative_score(approved_transactions)

        # 3. For the top n percent of scored transactions run model evaluation and choose the best
        top_five = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:5]  # Sort by high score -> low score
        best_tx = sorted(top_five, key=lambda tx: Model(tx[0].weights).evaluate(data))[0][0]

        return Model(best_tx.weights).evaluate(data)

    def process_next_batch(self, train_data, test_data):
        current_loss = self.compute_current_loss(test_data)

        # Obtain two tips from the tangle
        tip1, tip2 = self.choose_tips()

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
        averaged_model = Model(tip1.weights).average(Model(tip2.weights))

        averaged_model.train(train_data)

        if averaged_model.performs_better_than(current_loss, test_data):
            return Transaction(averaged_model.get_weights(), tip1, tip2), current_loss[0]

        return None, current_loss[0]


def run():
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

    tangle = Tangle(Transaction(Model().get_weights(), None, None))

    # For visualization purposes
    tangle.transactions[0].add_tag(0)
    global_loss = []

    nodes = [Node(tangle) for _ in range(len(emnist_train.client_ids))]

    # Organize transactions in artificial 'rounds'
    for rnd in range(15):
        def process_next_batch(node, i):
            with tf.Session(graph=tf.Graph()) as sess:
                K.set_session(sess)
                train_data = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[i])
                test_data = emnist_test.create_tf_dataset_for_client(emnist_test.client_ids[i])
                return node.process_next_batch(train_data, test_data)

        # In each round, a set of nodes performs a training step and potentially publishes the result as a transaction.
        # Why would nodes continuously publish updates (in the real world)?
        # The stability of the tangle results from a continuous stream of well-behaved updates
        # even if they only provide a negligible improvement of the model.

        selected_nodes = np.random.choice(range(len(emnist_train.client_ids)), NUM_CLIENTS, replace=False)

        with Pool(NUM_CLIENTS) as p:
            new_transactions = p.starmap(process_next_batch, [(nodes[i], i) for i in selected_nodes])

            for t, _ in new_transactions:
                if t is not None:
                    tangle.add_transaction(t)

            global_loss.append(sum([loss for _, loss in new_transactions]) / len(selected_nodes))

        tangle.show()
        tangle.save(rnd, global_loss)


if __name__ == '__main__':
    run()
