"""Evaluate the model"""

import argparse
import random
import logging
import os

import torch

import utils
from model import MGCN_Net
from data_loader import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='wn18',
                    help="Directory containing the dataset")
parser.add_argument('--seed', default=2019,
                    help="random seed for initialization")
parser.add_argument('--restore_dir', default=None,
                    help="Optional, name of the directory containing weights to reload before training, e.g., 'experiments'")
parser.add_argument('--multi_gpu', default=False, action='store_true',
                    help="Whether to use multiple GPUs if available")


def evaluate(model, data_iterator, params, mark='Eval', verbose=False):
    """Evaluate the model on dataset 'data'"""
    # set the model to evaluation mode
    model.eval()

    # a running average object for loss
    loss_avg = utils.RunningAverage()

    for batch in data_iterator:
        with torch.no_grad():
            loss, acc = model(batch)
        if params.n_gpu > 1 and params.multi_gpu:
            loss = loss.mean()
        loss_avg.update(loss.item())

    # logging and report
    metrics = {}
    metrics['acc'] = acc
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

    dataloader = DataLoader(args.dataset, params)
    test_data = dataloader.load_data(data_type='test')
    test_data_iterator = dataloader.data_iterator(test_data, params.batch_size)

    # prepare model
    model = MGCN_Net(params)
    utils.load_checkpoint(os.path.join(model_dir, 'last.ckpt'), model)
    model.to(params.device)

    if params.n_gpu > 1 and args.multi_gpu:
        model = torch.nn.DataParallel(model)

    # prepare optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)

    # train and evaluate the model
    logging.info('Starting training for {} epoch(s)'.format(params.epoch_num))
    evaluate(model, test_data_iterator, params, mark='Test', verbose=True)
