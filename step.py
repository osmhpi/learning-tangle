#! /usr/bin/env python

import sys
import os
import itertools

sys.path.insert(1, './leaf/models')

import numpy as np
import tensorflow as tf

from tangle import Tangle, train_single
from utils.model_utils import read_data
from utils.args import parse_args

def main():

    args = parse_args()

    # client_id = sys.argv[1]
    # tangle_name = sys.argv[2]
    client_id = 'f0044_12'
    tangle_name = 120

    train_data_dir = os.path.join('leaf', 'data', args.dataset, 'data', 'train_sm')
    test_data_dir = os.path.join('leaf', 'data', args.dataset, 'data', 'test_sm')

    print("Loading data...")
    users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)
    print("Loading data... complete")

    print(train_single(client_id, None, 1, 0, train_data[client_id], test_data[client_id], tangle_name))

if __name__ == '__main__':
    main()
