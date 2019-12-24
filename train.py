"""Train and evaluate the model"""

import argparse
import logging
import os
import random
import math

import torch
import torch.nn as nn
from tqdm import trange

import utils
from model import MGCN
from evaluate import evaluate
from data_manager import DataManager


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='WN18',
                    help="Directory containing the dataset")
parser.add_argument('--seed', default=2019,
                    help="random seed for initialization")
parser.add_argument('--restore_dir', default=None,
                    help="Optional, name of the directory containing weights to reload before training, e.g., 'experiments'")
parser.add_argument('--multi_gpu', default=False, action='store_true',
                    help="Whether to use multiple GPUs if available")
parser.add_argument('--loader_type', default=0, type=int,
                    help="0: one graph and one batch in each epoch;\
                          1: torch_geometric.data.NeighborSampler, can generate n-hops graph;\
                          2: torch.utils.data.DataLoader, split dataset to n batches and generate n graphs")


def train(model, loader, optimizer, params):
    """Train the model for one epoch"""
    # set the model to training mode
    model.train()

    optimizer.zero_grad()

    # a running average object for loss and acc
    loss_avg = utils.RunningAverage()
    acc_avg = utils.RunningAverage()
    
    for data in loader:
        entity_embedding = model(data.to(params.device))
        loss, acc = model.loss_func(entity_embedding, data.samples, data.labels)

        if params.n_gpu > 1 and params.multi_gpu:
            loss = loss.mean()  # mean() to average on multi-gpu
        loss.backward()

        # gradient clipping
        nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=params.clip_grad)
        optimizer.step()

        # update the average loss
        loss_avg.update(loss.item())
        acc_avg.update(acc.item())

    return loss_avg(), acc_avg()


def train_and_evaluate(model, data_manager, optimizer, params, model_dir, restore_dir):
    """Train the model and evaluate every epoch"""
    # reload weights from restore_dir if specified
    if restore_dir is not None:
        utils.load_checkpoint(os.path.join(restore_dir, 'last.ckpt'), model)

    # main evaluation criteria
    best_measure = 0
    # early stopping
    patience_counter = 0

    # evaluation triplet and graph
    eval_triplets = data_manager.fetch_triplets('val')
    all_triplets = data_manager.all_triplets()
    test_graph = data_manager.build_test_graph()

    # build graph using training set
    if args.loader_type == 2:
        loader = data_manager.data_iterator(batch_size=params.batch_size, shuffle=True)
    elif args.loader_type == 1:
        neighbor_sampler_size = [int(size) for size in params.sampler_size.split()]
        loader = data_manager.neighbor_sampler(batch_size=params.batch_size,
                                               shuffle=True,
                                               size=neighbor_sampler_size,
                                               num_hops=params.sampler_num_hops)

    epoches = trange(params.epoch_num)
    for epoch in epoches:
        # train for one epoch on training set
        if args.loader_type == 0:
            loader = [data_manager.build_train_graph()]
        loss, acc = train(model, loader, optimizer, params)

        epoches.set_postfix(loss='{:05.3f}'.format(loss), acc='{:05.3f}'.format(acc))

        if (epoch + 1) % params.eval_every == 0:
            val_metrics = evaluate(model, test_graph, eval_triplets, all_triplets, params, mark='Val')
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

    # set multiprocessing start method
    utils.multiprocess_setting()

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
    data_manager = DataManager(args.dataset, params)

    # prepare model
    model = MGCN(data_manager.num_entity, data_manager.num_relation, params)
    if params.load_pretrain:
        model.from_pretrained_emb(data_manager.pretrained_entity,
                                  data_manager.pretrained_relation)
    model.to(params.device)

    if params.n_gpu > 1 and args.multi_gpu:
        model = torch.nn.DataParallel(model)

    # prepare optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.3)

    # train and evaluate the model
    logging.info('Starting training for {} epoch(s)'.format(params.epoch_num))
    train_and_evaluate(model, data_manager, optimizer,
                       params, model_dir, args.restore_dir)
