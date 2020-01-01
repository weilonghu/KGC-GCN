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
from data_manager import DataManager


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='FB15k-237',
                    help="Directory containing the dataset")
parser.add_argument('--seed', default=2019,
                    help="random seed for initialization")
parser.add_argument('--restore_dir', default=None,
                    help="Optional, name of the directory containing weights to reload before training, e.g., 'experiments'")
parser.add_argument('--multi_gpu', default=False, action='store_true',
                    help="Whether to use multiple GPUs if available")
parser.add_argument('--sampler_method', default='uniform', type=str,
                    help="uniform sampling or neighborhood sampling")


def train(model, loader, optimizer, params):
    """Train the model for one epoch"""
    # set the model to training mode
    model.train()

    # a running average object for loss and acc
    loss_avg = utils.RunningAverage()
    acc_avg = utils.RunningAverage()

    for data in loader:
        optimizer.zero_grad()

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


def train_and_evaluate(model, data_manager, optimizer, scheduler, params, model_dir, restore_dir):
    """Train the model and evaluate every epoch"""

    # main evaluation criteria
    best_measure = 0
    # early stopping
    patience_counter = 0

    # reload weights from restore_dir if specified
    if restore_dir is not None:
        best_measure = utils.load_checkpoint(os.path.join(restore_dir, 'last.ckpt'), model)
        logging.info('Restore model from {} with best measure: {}'.format(os.path.join(restore_dir, 'last.ckpt'), best_measure))

    # evaluation triplet and graph
    eval_triplets = torch.from_numpy(data_manager.fetch_triplets('val', size=0.5))
    all_triplets = torch.from_numpy(data_manager.all_triplets())
    test_graph = data_manager.build_test_graph()
    logging.info('Sample {} triplets for evaluation'.format(eval_triplets.size(0)))

    logging.info('Starting training for {} epoch(s)'.format(params.epoch_num))

    with trange(params.epoch_num, desc='Main') as bar:
        for epoch in bar:
            # train for one epoch on training set
            loader = data_manager.get_data_loader(params.sampler_method)

            loss, acc = train(model, loader, optimizer, params)

            scheduler.step()

            bar.set_postfix(loss='{:05.3f}'.format(loss), acc='{:05.3f}'.format(acc))

            if (epoch + 1) % params.eval_every == 0:
                val_metrics = evaluate(model, test_graph, eval_triplets, all_triplets, params, mark='Val')
                val_measure = val_metrics['measure']
                improve_measure = val_measure - best_measure
                if improve_measure > 0:
                    best_measure = val_measure
                    state = {'state_dict': model.state_dict(),
                             'optim_dict': optimizer.state_dict(),
                             'measure': best_measure}
                    utils.save_checkpoint(state, is_best=False, checkpoint_dir=model_dir)
                    if improve_measure < params.patience:
                        patience_counter += 1
                    else:
                        patience_counter = 0
                else:
                    patience_counter += 1

                # early stopping and logging best measure
                if (patience_counter >= params.patience_num and epoch > params.min_epoch_num) or epoch == params.epoch_num:
                    logging.info("Early stopping with best val measure: {:05.2f}".format(best_measure))
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
    params.sampler_method = args.sampler_method

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
    model.to(params.device)

    if params.n_gpu > 1 and args.multi_gpu:
        model = torch.nn.DataParallel(model)

    # prepare optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9, last_epoch=-1)

    # train and evaluate the model
    train_and_evaluate(model, data_manager, optimizer, scheduler,
                       params, model_dir, args.restore_dir)
