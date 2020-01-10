#! /usr/bin/env python

import os as os
import numpy as np
import tensorflow as tf
from tensorflow_federated import python as tff
import tensorflow_datasets as tfds

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

def write_sample(basepath, s, i):
  label = s['label']

  sample_path = f'{basepath}/{label}'
  os.makedirs(sample_path, exist_ok=True)

  with open(f'{sample_path}/{i}', 'wb') as pixels:
    np.save(pixels, s['pixels'])

for client in emnist_train.client_ids:
  client_path = f'data/{client}'
  train_data = emnist_train.create_tf_dataset_for_client(client)
  test_data = emnist_test.create_tf_dataset_for_client(client)

  i = 0

  for sample in tfds.as_numpy(train_data):
      write_sample(f'{client_path}/train', sample, i)
      i = i + 1

  for sample in tfds.as_numpy(test_data):
      write_sample(f'{client_path}/test', sample, i)
      i = i + 1
