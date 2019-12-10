"""Train and evaluate the model"""

import argparse
import logging
import os
import random
import pickle

import torch
import torch.nn.functional as F

import utils
from models import LGCN_Net3
from evaluate import evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='conll',
                    help="Directory containing the dataset")
parser.add_argument('--seed', default=2019,
                    help="random seed for initialization")
parser.add_argument('--restore_dir', default=None,
                    help="Optional, name of the directory containing weights to reload before training, e.g., 'experiments'")
parser.add_argument('--multi_gpu', default=False, action='store_true',
                    help="Whether to use multiple GPUs if available")


def train(model, data, train_class_weights, optimizer, params):
    """Train the model for one epoch"""
    # set the model to training mode
    model.train()

    optimizer.zero_grad()
    loss = F.nll_loss(model(data)[data.train_mask],
                      data.y[data.train_mask], weight=train_class_weights)
    loss.backward()
    optimizer.step()


def train_and_evaluate(model, train_data, val_data, optimizer, params, model_dir, restore_dir):
    """Train the model and evaluate every epoch"""
    # reload weights from restore_dir if specified
    if restore_dir is not None:
        model = utils.load_checkpoint(
            os.path.join(restore_dir, 'last.pth.tar'))

    # main evaluation criteria
    best_measure = 0
    # early stopping
    patience_counter = 0

    train_class_ratio = dataset.y[dataset.train_mask].sum().item() / dataset.y[dataset.train_mask].shape[0]
    train_class_weights = torch.Tensor([train_class_ratio, 1 - train_class_ratio]).to(params.device)

    for epoch in range(1, params.epoch_num + 1):
        # run one epoch
        logging.info('Epoch {}/{}'.format(epoch, params.epoch_num))

        # train for one epoch on training set
        train(model, train_data, train_class_weights, optimizer, params)

        # try evaluate this epoch if val_data exists
        if val_data is not None:
            val_metrics = evaluate(model, val_data, params, mark='Val')
            val_measure = val_metrics['measure']
            improve_measure = val_measure - best_measure
            if improve_measure > 0:
                logging.info('- Found new best measure')
                best_measure = val_measure
                utils.save_checkpoint(model['state_dict'], is_best=False, checkpoint=model_dir)
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
        else:
            # if no valid dataset exists, run all epochs
            utils.save_checkpoint(model['state_dict'], is_best=False, checkpoint=model_dir)


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

    dataset = pickle.load(open(os.path.join('data', args.dataset), 'rb'))
    val, pos = dataset.x.max(dim=0)
    dataset.x /= val.abs()
    train_data = dataset.to(params.device)

    # prepare model
    model = LGCN_Net3(dataset.to(params.device))
    model.to(params.device)

    if params.n_gpu > 1 and args.multi_gpu:
        model = torch.nn.DataParallel(model)

    # prepare optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)

    # train and evaluate the model
    logging.info('Starting training for {} epoch(s)'.format(params.epoch_num))
    train_and_evaluate(model, train_data, None, optimizer,
                       params, model_dir, args.restore_dir)
