import collections

import numpy as np
import tensorflow as tf

NUM_EPOCHS = 10
BATCH_SIZE = 100
SHUFFLE_BUFFER = 500
BATCHES_PER_ROUND = 100

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
        return self.model.evaluate(self.preprocess_test(data), verbose=0)

    def performs_better_than(self, other_result, data):
        # print(self.model.metrics_names) tells us that:
        # self.evaluate() -> (loss, sparse_categorical_accuracy)
        # Not sure if we can ignore the sparse_categorical_accuracy here
        return self.evaluate(data)[0] < other_result[0]
