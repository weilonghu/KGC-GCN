"""Evaluate the model"""

import argparse
import random
import logging
import os

import torch

import utils
from model import MGCN
from data_set import DataSet
from metric import calc_mrr


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='wn18',
                    help="Directory containing the dataset")
parser.add_argument('--seed', default=2019,
                    help="random seed for initialization")
parser.add_argument('--restore_dir', default=None,
                    help="Optional, name of the directory containing weights to reload before training, e.g., 'experiments'")
parser.add_argument('--multi_gpu', default=False, action='store_true',
                    help="Whether to use multiple GPUs if available")


def evaluate(model, eval_triplets, all_triplets, eval_graph, params, mark='Eval', verbose=False):
    """Evaluate the model on dataset 'data'"""
    # set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        entity_embedding = model(eval_graph.entity, eval_graph.edge_index, eval_graph.edge_type, eval_graph.edge_norm)

    mrr = calc_mrr(entity_embedding, model.relation_embedding, eval_triplets, all_triplets, hits=[1, 3, 10])

    # logging and report
    metrics = {}
    metrics['mrr'] = mrr
    metrics['measure'] = mrr
    metrics_str = "; ".join("{}: {:05.2f}".format(k, v)
                            for k, v in metrics.items())
    logging.info("- {} metrics: ".format(mark) + metrics_str)

    return metrics


if __name__ == '__main__':
    args = parser.parse_args()
    # directory containing saved model
    model_dir = os.path.join('experiments', args.dataset)
    # load the parameters from json file
    json_path = os.path.join(model_dir, 'params.json')
    assert os.path.isfile(
        json_path), 'No json configuration file found at {}'.format(json_path)
    params = utils.Params(json_path)

    # use GPUs if available
    params.device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    params.n_gpu = torch.cuda.device_count()
    params.multi_gpu = args.multi_gpu

    # set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if params.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)  # set random seed for all GPUs
    params.seed = args.seed

    # set the logger
    utils.set_logger(os.path.join(model_dir, 'evalate.log'))
    logging.info('device: {}, n_gpu: {}'.format(params.device, params.n_gpu))

    # create dataset and normalize
    logging.info('Loading the dataset...')

    dataset = DataSet(args.dataset, params)
    eval_graph = dataset.build_eval_graph()
    all_triplets = dataset.total_triplets()

    # prepare model
    model = MGCN(dataset.n_entity, dataset.num_relations,
                 params.n_bases, params.dropout)
    utils.load_checkpoint(os.path.join(model_dir, 'last.ckpt'), model)
    model.to(params.device)

    if params.n_gpu > 1 and args.multi_gpu:
        model = torch.nn.DataParallel(model)

    # prepare optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)

    # train and evaluate the model
    logging.info('Starting training for {} epoch(s)'.format(params.epoch_num))
    evaluate(model, dataset.test_triplets, all_triplets, eval_graph, params, mark='Test', verbose=True)
