"""Script to run the baselines."""
import argparse
import importlib
import numpy as np
import os
import sys
import random
import math
import tensorflow as tf
import multiprocessing as mp

sys.path.insert(1, './leaf/models')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import metrics.writer as metrics_writer

from baseline_constants import MAIN_PARAMS, MODEL_PARAMS
from client import Client
from server import Server
from model import ServerModel

from tangle import Tangle, Transaction, MaliciousType, train_single, test_single

from utils.args import parse_args
from utils.model_utils import read_data

STAT_METRICS_PATH = 'metrics/stat_metrics.csv'
SYS_METRICS_PATH = 'metrics/sys_metrics.csv'

FLIP_FROM_CLASS = 3
FLIP_TO_CLASS = 8

def main():
    mp.set_start_method('spawn')

    args = parse_args()

    start_from_round = 0

    # Set the random seed if provided (affects client sampling, and batching)
    random.seed(1 + args.seed)
    np.random.seed(12 + args.seed)
    tf.set_random_seed(123 + args.seed)

    model_path = 'leaf/models/%s/%s.py' % (args.dataset, args.model)
    if not os.path.exists(model_path):
        print('Please specify a valid dataset and a valid model.')
    model_path = '%s.%s' % (args.dataset, args.model)

    print('############################## %s ##############################' % model_path)
    mod = importlib.import_module(model_path)
    ClientModel = getattr(mod, 'ClientModel')

    tup = MAIN_PARAMS[args.dataset][args.t]
    num_rounds = (args.num_rounds if args.num_rounds != -1 else tup[0]) + start_from_round
    eval_every = args.eval_every if args.eval_every != -1 else tup[1]
    clients_per_round = args.clients_per_round if args.clients_per_round != -1 else tup[2]

    # Suppress tf warnings
    tf.logging.set_verbosity(tf.logging.WARN)

    # Create 2 models
    model_params = MODEL_PARAMS[model_path]
    if args.lr != -1:
        model_params_list = list(model_params)
        model_params_list[0] = args.lr
        model_params = tuple(model_params_list)

    # Create client model, and share params with server model
    tf.reset_default_graph()
    client_model = ClientModel(args.seed, *model_params)

    # Create tangle and tangle metrics
    os.makedirs('tangle_data/transactions', exist_ok=True)

    # Legacy metrics variables (not used)
    global_loss = [],
    global_accuracy = [],
    norm = []

    if start_from_round == 0:
        genesis = Transaction(client_model.get_params(), [], tag=0)
        tangle = Tangle({genesis.name(): genesis}, genesis.name())
        tangle.save(0, global_loss, global_accuracy, norm)
    else:
        tangle = Tangle.fromfile(start_from_round)

    # Create server
    server = Server(client_model)

    # Create clients
    clients, malicious_clients = setup_clients(args.dataset, client_model, args.use_val_set, args.malicious_fraction, MaliciousType[args.malicious_type])
    malicious_type = MaliciousType[args.malicious_type]
    client_ids, client_groups, client_num_samples = server.get_clients_info(clients)
    print('Clients in Total: %d' % len(clients))

    # Initial status
    print('--- Random Initialization ---')
    stat_writer_fn = get_stat_writer_function(client_ids, client_groups, client_num_samples, args)
    sys_writer_fn = get_sys_writer_function(args)
    print_stats(0, tangle, random_sample(clients, clients_per_round * 10), client_num_samples, args, stat_writer_fn, args.use_val_set)

    # Simulate training
    for i in range(start_from_round, num_rounds):
        print('--- Round %d of %d: Training %d Clients ---' % (i + 1, num_rounds, clients_per_round))

        # Select clients to train this round
        server.select_clients(i, online(clients), num_clients=clients_per_round)
        c_ids, c_groups, c_num_samples = server.get_clients_info(server.selected_clients)

        # Simulate server model training on selected clients' data
        sys_metrics = tangle.run_nodes(train_single, server.selected_clients, i+1,
                                       num_epochs=args.num_epochs, batch_size=args.batch_size,
                                       malicious_clients=malicious_clients, malicious_type=malicious_type)
        # norm.append(np.array(norm_this_round).mean(axis=0).tolist() if len(norm_this_round) else [])
        sys_writer_fn(i + 1, c_ids, sys_metrics, c_groups, c_num_samples)

        # Update global metrics
        # Todo: Add global accuracy & loss to json files
        #global_accuracy.append(sys_metrics)

        # Update tangle on disk
        tangle.save(i+1, global_loss, global_accuracy, norm)

        # Test model
        if (i + 1) % eval_every == 0 or (i + 1) == num_rounds:
            print_stats(i + 1, tangle, random_sample(clients, clients_per_round * 10), client_num_samples, args, stat_writer_fn, args.use_val_set)

    #compute_confusion_matrix()
    # Close models
    # server.close_model()

def online(clients):
    """We assume all users are always online."""
    return clients

def random_sample(clients, sample_size):
    """Choose a subset of clients to perform the model validation. Only to be used during development to speed up experiment run times"""
    return np.random.choice(clients, sample_size, replace=False)

def create_clients(users, groups, train_data, test_data, model):
    if len(groups) == 0:
        groups = [[] for _ in users]
    clients = [Client(u, g, train_data[u], test_data[u], model) for u, g in zip(users, groups)]
    return clients

def setup_clients(dataset, model=None, use_val_set=False, malicious_fraction=0, malicious_type=MaliciousType.NONE):
    """Instantiates clients based on given train and test data directories.

    Return:
        all_clients: list of Client objects.
    """
    eval_set = 'test' if not use_val_set else 'val'
    train_data_dir = os.path.join('leaf', 'data', dataset, 'data', 'train')
    test_data_dir = os.path.join('leaf', 'data', dataset, 'data', eval_set)

    users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)

    clients = create_clients(users, groups, train_data, test_data, model)

    num_malicious_clients = math.floor(len(clients) * malicious_fraction)
    malicious_clients = [client.id for client in clients[:num_malicious_clients]]
    if malicious_type == MaliciousType.LABELFLIP:
        for client in clients[:num_malicious_clients]:
            # flip labels
            client.train_data['y'] = [FLIP_TO_CLASS for y in client.train_data['y'] if y == FLIP_FROM_CLASS]

    return clients, malicious_clients


def get_stat_writer_function(ids, groups, num_samples, args):

    def writer_fn(num_round, metrics, partition):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, partition, 'leaf/models/metrics', '{}_{}'.format('stat', args.metrics_name))

    return writer_fn


def get_sys_writer_function(args):

    def writer_fn(num_round, ids, metrics, groups, num_samples):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, 'train', 'leaf/models/metrics', '{}_{}'.format('sys', args.metrics_name))

    return writer_fn


def print_stats(
    num_round, tangle, clients, num_samples, args, writer, use_val_set):

    train_stat_metrics = tangle.test_model(test_single, clients, set_to_use='train')
    print_metrics(train_stat_metrics, num_samples, prefix='train_')
    writer(num_round, train_stat_metrics, 'train')

    eval_set = 'test' if not use_val_set else 'val'
    test_stat_metrics = tangle.test_model(test_single, clients, set_to_use=eval_set)
    print_metrics(test_stat_metrics, num_samples, prefix='{}_'.format(eval_set), print_conf_matrix=True)
    writer(num_round, test_stat_metrics, eval_set)


def print_metrics(metrics, weights, prefix='', print_conf_matrix=False):
    """Prints weighted averages of the given metrics.

    Args:
        metrics: dict with client ids as keys. Each entry is a dict
            with the metrics of that client.
        weights: dict with client ids as keys. Each entry is the weight
            for that client.
    """
    ordered_weights = [weights[c] for c in sorted(weights) if c in metrics]
    metric_names = metrics_writer.get_metrics_names(metrics)
    to_ret = None
    for metric in metric_names:
        if metric == 'conf_matrix':
            continue
        ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
        print('%s: %g, 10th percentile: %g, 50th percentile: %g, 90th percentile %g' \
              % (prefix + metric,
                 np.average(ordered_metric, weights=ordered_weights),
                 np.percentile(ordered_metric, 10),
                 np.percentile(ordered_metric, 50),
                 np.percentile(ordered_metric, 90)))

    # print confusion matrix
    if print_conf_matrix:
        if 'conf_matrix' in metric_names:
            full_conf_matrix = sum([metrics[c]['conf_matrix'] for c in sorted(metrics)])
            #print(full_conf_matrix)
            np.savetxt('conf_matrix.txt', full_conf_matrix, fmt='%4u')



if __name__ == '__main__':
    main()
