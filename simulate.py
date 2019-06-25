import collections

import numpy as np
import tensorflow as tf

import networkx as nx
import matplotlib.pyplot as plt

from multiprocessing.dummy import Pool
from itertools import starmap

from tensorflow_federated import python as tff

NUM_EPOCHS = 10
BATCH_SIZE = 20
SHUFFLE_BUFFER = 500
BATCHES_PER_ROUND = 100

NUM_CLIENTS = 5

ALPHA = 0.2


class TipSelector:
    def __init__(self, tangle):
        self.tangle = tangle

    def tip_selection(self):
        entry_point = self.tangle.genesis

        entry_point_trunk = entry_point
        entry_point_branch = entry_point

        # Build a map of transactions that directly approve a given transaction
        approving_transactions = {x: [] for x in self.tangle.transactions}
        for x in self.tangle.transactions:
            if x.p1 is not None:
                approving_transactions[x.p1].append(x)
            if x.p2 is not None and x.p1 != x.p2:
                approving_transactions[x.p2].append(x)

        ratings = self.calculate_cumulative_weight(approving_transactions)

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

        return approver

        # Skip validation
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


class Tangle:
    def __init__(self, tip):
        self.genesis = tip
        self.transactions = [self.genesis]

    def choose_tips(self):
        if len(self.transactions) < 2:
            return tuple([self.genesis, self.genesis])

        selector = TipSelector(self)
        return selector.tip_selection()

    def add_transaction(self, tip):
        print("Adding tip")
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


class Model:
    def __init__(self, weights=None):
        self.model = self.create_compiled_keras_model()
        if weights is not None:
            self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def average(self, other):
        new_weights = [np.array(weights_).mean(axis=0) for weights_ in zip(*[self.get_weights(), other.get_weights()])]
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

    def performs_better_than(self, other, data):
        # Return true if loss is less
        return self.evaluate(data) < other.evaluate(data)


class Node:
    def __init__(self, tangle):
        self.model = Model()
        self.tangle = tangle

    def process_next_batch(self, data):
        # Obtain two tips from the tangle
        tip1, tip2 = self.tangle.choose_tips()
        print(f"Got tips with heights {tip1.height} and {tip2.height}")

        # Perform averaging
        averaged_model = Model(tip1.weights).average(Model(tip2.weights))

        averaged_model.train(data)

        if averaged_model.performs_better_than(self.model, data):
            self.model = averaged_model

            return Transaction(self.model.get_weights(), tip1, tip2)

        return None


if __name__ == '__main__':
    # Load test data
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

    # Create tangle
    tangle = Tangle(Transaction(Model().get_weights(), None, None))

    # Create clients
    sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]
    sample_datasets = [emnist_test.create_tf_dataset_for_client(x) for x in sample_clients]
    nodes = [Node(tangle) for x in sample_clients]

    for _ in range(10):
        def process_next_batch(node, data):
            return node.process_next_batch(data)

        new_transactions = list(starmap(process_next_batch, [(nodes[i], sample_datasets[i]) for i in range(NUM_CLIENTS)]))
        for t in new_transactions:
            if t is not None:
                tangle.add_transaction(t)

        # with Pool(NUM_CLIENTS) as p:
        #     new_transactions = p.starmap(process_next_batch, [(nodes[i], sample_datasets[i]) for i in range(NUM_CLIENTS)])
        #
        #     for t in new_transactions:
        #         if t is not None:
        #             tangle.add_transaction(t)

        tangle.show()
