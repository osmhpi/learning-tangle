#! /usr/bin/env python

import sys
import os
import itertools

import numpy as np
import tensorflow as tf

from tangle import Tangle, Node, Transaction, Model

def run():
    node_id = sys.argv[1]
    tangle_name = sys.argv[2]

    tangle = Tangle.fromfile(int(tangle_name))

    node = Node(node_id, tangle)
    tx, loss = node.process_next_batch()

    if tx is not None:
      print(tx.name(), *tx.parents)

if __name__ == '__main__':
    run()
