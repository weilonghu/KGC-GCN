"""Train and evaluate the model"""

import argparse
import logging
import os
import random

import torch
import torch.nn as nn
from tqdm import trange

import utils
from model import MGCN
from evaluate import evaluate
from data_set import DataSet


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='wn18',
                    help="Directory containing the dataset")
parser.add_argument('--seed', default=2019,
                    help="random seed for initialization")
parser.add_argument('--restore_dir', default=None,
                    help="Optional, name of the directory containing weights to reload before training, e.g., 'experiments'")
parser.add_argument('--multi_gpu', default=False, action='store_true',
                    help="Whether to use multiple GPUs if available")


def train(model, dataset, optimizer, params):
    """Train the model for one epoch"""
    # set the model to training mode
    model.train()

    optimizer.zero_grad()

    # compute model output and loss
    train_data = dataset.build_train_graph()
    entity_embedding = model(
        train_data.entity, train_data.edge_index,
        train_data.edge_type, train_data.edge_norm)
    loss = model.score_loss(entity_embedding, train_data.samples, train_data.labels) + \
        params.regularization * model.reg_loss(entity_embedding)

    if params.n_gpu > 1 and params.multi_gpu:
        loss = loss.mean()  # mean() to average on multi-gpu
    loss.backward()

    # gradient clipping
    nn.utils.clip_grad_norm_(
        parameters=model.parameters(), max_norm=params.clip_grad)
    optimizer.step()
    model.zero_grad()


def train_and_evaluate(model, dataset, optimizer, params, model_dir, restore_dir):
    """Train the model and evaluate every epoch"""
    # reload weights from restore_dir if specified
    if restore_dir is not None:
        utils.load_checkpoint(os.path.join(restore_dir, 'last.ckpt'), model)

    # main evaluation criteria
    best_measure = 0
    # early stopping
    patience_counter = 0

    # evaluation triplet and graph
    eval_triplets = dataset.valid_triplets
    all_triplets = dataset.total_triplets()
    eval_graph = dataset.build_eval_graph()

    # use tqdm for progress bar
    for epoch in trange(1, (params.epoch_num + 1), desc='Epochs', position=0):
        # run one epoch
        # logging.info('Epoch {}/{}'.format(epoch + 1, params.epoch_num))

        # train for one epoch on training set
        train(model, dataset, optimizer, params)

        if epoch % params.eval_every == 0:
            val_metrics = evaluate(model, eval_triplets, all_triplets, eval_graph, params, mark='Val')
            val_measure = val_metrics['measure']
            improve_measure = val_measure - best_measure
            if improve_measure > 0:
                logging.info('- Found new best measure')
                best_measure = val_measure
                state = {'state_dict': model.state_dict(
                ), 'optim_dict': optimizer.state_dict()}
                utils.save_checkpoint(state, is_best=False,
                                      checkpoint_dir=model_dir)
                if improve_measure < params.patience:
                    patience_counter += 1
                else:
                    patience_counter = 0
            else:
                patience_counter += 1

            # early stopping and logging best measure
            if (patience_counter >= params.patience_num and epoch > params.min_epoch_num) or epoch == params.epoch_num:
                logging.info("Best val measure: {:05.2f}".format(best_measure))
                break


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
    utils.set_logger(os.path.join(model_dir, 'train.log'))
    logging.info('device: {}, n_gpu: {}'.format(params.device, params.n_gpu))

    # create dataset and normalize
    logging.info('Loading the dataset...')

    dataset = DataSet(args.dataset, params)

    # prepare model
    model = MGCN(dataset.n_entity, dataset.n_relation,
                 params.n_bases, params.dropout)
    model.to(params.device)

    if params.n_gpu > 1 and args.multi_gpu:
        model = torch.nn.DataParallel(model)

    # prepare optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)

    # train and evaluate the model
    logging.info('Starting training for {} epoch(s)'.format(params.epoch_num))
    train_and_evaluate(model, dataset, optimizer,
                       params, model_dir, args.restore_dir)
