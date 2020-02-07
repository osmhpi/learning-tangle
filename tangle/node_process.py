import importlib
import random
import tensorflow as tf
import numpy as np

from utils.args import parse_args
from baseline_constants import MAIN_PARAMS, MODEL_PARAMS, BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY
from client import Client

from .tangle import Tangle
from .node import Node

def build_client(u, g, flops, train_data, eval_data):
    args = parse_args()

    model_path = '%s.%s' % (args.dataset, args.model)
    mod = importlib.import_module(model_path)
    ClientModel = getattr(mod, 'ClientModel')

    tup = MAIN_PARAMS[args.dataset][args.t]
    num_rounds = args.num_rounds if args.num_rounds != -1 else tup[0]
    eval_every = args.eval_every if args.eval_every != -1 else tup[1]
    clients_per_round = args.clients_per_round if args.clients_per_round != -1 else tup[2]

    model_params = MODEL_PARAMS[model_path]
    if args.lr != -1:
        model_params_list = list(model_params)
        model_params_list[0] = args.lr
        model_params = tuple(model_params_list)


    # Create client model, and share params with server model
    tf.reset_default_graph()
    client_model = ClientModel(1234, *model_params)
    client_model.flops = flops
    return Client(u, g, train_data, eval_data, client_model)

def train_single(u, g, flops, seed, train_data, eval_data, tangle_name, malicious_node, malicious_type):
    # Suppress tf warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    random.seed(1 + seed)
    np.random.seed(12 + seed)
    tf.compat.v1.set_random_seed(123 + seed)

    client = build_client(u, g, flops, train_data, eval_data)

    tangle = Tangle.fromfile(tangle_name)
    if malicious_node:
        node = Node(client, tangle, malicious_type)
    else:
        node = Node(client, tangle)

    args = parse_args()
    tx, metrics, comp = node.process_next_batch(args.num_epochs, args.batch_size)

    sys_metrics = {BYTES_WRITTEN_KEY: 0,
                   BYTES_READ_KEY: 0,
                   LOCAL_COMPUTATIONS_KEY: 0}
    sys_metrics[BYTES_READ_KEY] += node.client.model.size
    sys_metrics[BYTES_WRITTEN_KEY] += node.client.model.size
    sys_metrics[LOCAL_COMPUTATIONS_KEY] = comp

    return tx, metrics, comp, u, sys_metrics

def test_single(u, g, flops, seed, train_data, eval_data, tangle_name, set_to_use):
    # Suppress tf warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    random.seed(1 + seed)
    np.random.seed(12 + seed)
    tf.compat.v1.set_random_seed(123 + seed)

    client = build_client(u, g, flops, train_data, eval_data)

    tangle = Tangle.fromfile(tangle_name)
    node = Node(client, tangle)
    reference = node.obtain_reference_params()
    node.client.model.set_params(reference)
    return u, node.client.test(set_to_use)
