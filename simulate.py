import collections

import numpy as np
import tensorflow as tf

import networkx as nx
import matplotlib.pyplot as plt

from tensorflow_federated import python as tff

NUM_EPOCHS = 10
BATCH_SIZE = 20
SHUFFLE_BUFFER = 500

NUM_CLIENTS = 5


class Tip:
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
        self.tips = [tip]

    def choose_tips(self):
        if len(self.tips) < 2:
            return tuple([self.tips[0], self.tips[0]])

        total_height = sum([2**x.height for x in self.tips])
        return tuple(np.random.choice(self.tips, 2, p=[(2**x.height) / total_height for x in self.tips], replace=False))

    def add_tip(self, tip):
        print("Adding tip")
        self.tips.append(tip)


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
        BATCHES_PER_ROUND = 100
        self.model.fit(self.preprocess(data), steps_per_epoch=BATCHES_PER_ROUND, epochs=NUM_EPOCHS, verbose=0)

    def evaluate(self, data):
        return self.model.evaluate(self.preprocess_test(data))

    def performs_better_than(self, other, data):
        # Return true if loss is less
        return self.evaluate(data) < other.evaluate(data)


class Node:
    def __init__(self, data, tangle):
        self.model = Model()
        self.data = data
        self.tangle = tangle

    def process_next_batch(self):
        # Obtain two tips from the tangle
        tip1, tip2 = self.tangle.choose_tips()
        print(f"Got tips with heights {tip1.height} and {tip2.height}")

        # Perform averaging
        averaged_model = Model(tip1.weights).average(Model(tip2.weights))

        averaged_model.train(self.data)

        if averaged_model.performs_better_than(self.model, self.data):
            self.model = averaged_model

            self.tangle.add_tip(Tip(self.model.get_weights(), tip1, tip2))


def show(tangle):
    G = nx.DiGraph()

    G.add_edges_from([(id(x), id(x.p1)) for x in tangle.tips if x.p1 is not None])
    G.add_edges_from([(id(x), id(x.p2)) for x in tangle.tips if x.p2 is not None])

    val_map = {id(x): x.height for x in tangle.tips}
    values = [val_map.get(node) for node in G.nodes()]

    # Need to create a layout when doing
    # separate calls to draw nodes and edges
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_color=values, node_size=100)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrows=False)
    plt.show()


if __name__ == '__main__':
    # Load test data
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

    # Create tangle
    tangle = Tangle(Tip(Model().get_weights(), None, None))

    # Create clients
    sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]
    nodes = [Node(emnist_test.create_tf_dataset_for_client(x), tangle) for x in sample_clients]

    for _ in range(10):
        for i in range(NUM_CLIENTS):
            nodes[i].process_next_batch()
        show(tangle)
