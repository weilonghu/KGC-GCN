"""Train and evaluate the model"""

import argparse
import logging
import os
import random
import math

import torch
import torch.nn as nn
from tqdm import trange
from torch_geometric.data import NeighborSampler

import utils
from model import MGCN
from evaluate import evaluate
from data_set import DataSet
from data_set import negative_sampling


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='FB15k-237',
                    help="Directory containing the dataset")
parser.add_argument('--seed', default=2019,
                    help="random seed for initialization")
parser.add_argument('--restore_dir', default=None,
                    help="Optional, name of the directory containing weights to reload before training, e.g., 'experiments'")
parser.add_argument('--multi_gpu', default=False, action='store_true',
                    help="Whether to use multiple GPUs if available")


def train(model, loader, train_data, optimizer, params):
    """Train the model for one epoch"""
    # set the model to training mode
    model.train()

    optimizer.zero_grad()

    # a running average object for loss and acc
    loss_avg = utils.RunningAverage()
    acc_avg = utils.RunningAverage()
    
    for data_flow in loader(train_data.train_mask):
        embedding, n_id, e_id, edge_index = model(train_data.edge_attr.to(params.device),
                                                  data_flow.to(params.device))
        # construct batch triplets
        head, tail = edge_index[0], edge_index[1]
        rel = train_data.edge_attr[e_id][:, 1].to(params.device)
        # negative sampling for training
        pos_samples = torch.cat((head.view(-1, 1), rel.view(-1, 1), tail.view(-1, 1)), dim=1)
        samples, labels = negative_sampling(pos_samples, n_id.size(0), params.negative_rate, params.device)

        loss, acc = model.loss_func(embedding, samples.to(params.device), labels.to(params.device))

        if params.n_gpu > 1 and params.multi_gpu:
            loss = loss.mean()  # mean() to average on multi-gpu
        loss.backward()

        # gradient clipping
        nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=params.clip_grad)
        optimizer.step()
        model.zero_grad()

        # update the average loss
        loss_avg.update(loss.item())
        acc_avg.update(acc.item())

    return loss_avg(), acc_avg()


def train_and_evaluate(model, dataset, optimizer, params, model_dir, restore_dir):
    """Train the model and evaluate every epoch"""
    # reload weights from restore_dir if specified
    if restore_dir is not None:
        utils.load_checkpoint(os.path.join(restore_dir, 'last.ckpt'), model)

    # main evaluation criteria
    best_measure = 0
    # early stopping
    patience_counter = 0

    # evaluation triplet
    eval_triplets = dataset.valid_triplets
    all_triplets = dataset.total_triplets()

    # build graph using training set, create NeighborSampler
    train_data = dataset.build_train_graph()
    loader = NeighborSampler(train_data, size=5, num_hops=1, bipartite=False,
                             batch_size=params.batch_size, shuffle=True, add_self_loops=True)

    epoches = trange(params.epoch_num)
    for epoch in epoches:
        # train for one epoch on training set
        loss, acc = train(model, loader, train_data, optimizer, params)

        epoches.set_postfix(loss='{:05.3f}'.format(loss), acc='{:05.3f}'.format(acc))

        if (epoch + 1) % params.eval_every == 0:
            val_metrics = evaluate(model, loader, train_data, eval_triplets, all_triplets, params, mark='Val')
            val_measure = val_metrics['measure']
            improve_measure = val_measure - best_measure
            if improve_measure < 0:
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
    num_edges = dataset.train_triplets.shape[0]

    # prepare model
    model = MGCN(dataset.n_entity, dataset.n_relation, num_edges,
                 params.emb_dim, params.dropout)
    if params.load_pretrain:
        model.from_pretrained_emb(dataset.pretrained_entity,
                                  dataset.pretrained_relation)
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
