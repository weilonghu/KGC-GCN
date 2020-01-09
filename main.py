"""Train and evaluate the model"""

import argparse
import logging
import os
import random

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm

import utils
from model import MGCN
from data_loader import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='WN18RR', help="Directory containing the dataset")
parser.add_argument('--seed', default=19960326, help="random seed for initialization")
parser.add_argument('--restore_dir', default=None, help='Optional, directory containing weights to reload before training')
parser.add_argument('--multi_gpu', default=False, action='store_true', help="Whether to use multiple GPUs if available")
parser.add_argument('--batch_size', default=128, type=int, help="Batch size")
parser.add_argument('--max_epoch', default=400, type=int, help='Number of maximum epochs')
parser.add_argument('--min_epoch', default=50, type=int, help='Number of minimum epochs')
parser.add_argument('--eval_every', default=2, type=int, help='Number of epochs to test the model')
parser.add_argument('--patience', default=0.001, type=float, help='Increasement between two epochs')
parser.add_argument('--patience_num', default=-1, type=int, help='Early stopping creteria')
parser.add_argument('--learning_rate', default=0.001, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=0, type=float, help='Weight decay for the optimizer')
parser.add_argument('--lbl_smooth', default=0.1, type=float, help="Label smoothing")
parser.add_argument('--num_workers', default=0, type=int, help='Number of processes to construct batches')
parser.add_argument('--bias', action='store_true', help='Whether to use bias in the model')
parser.add_argument('--embed_dim', default=200, type=int, help='Dimension size for entities and relations')
parser.add_argument('--hidden_drop', default=0.3, type=float, help='GCN: Dropout after GCN')
parser.add_argument('--hidden_drop2', default=0.3, type=float, help='ConvE: hidden dropout')
parser.add_argument('--feat_drop', default=0.3, type=float, help='ConvE: feature dropout')
parser.add_argument('--k_w', default=10, type=int, help='ConvE: k_w')
parser.add_argument('--k_h', default=20, type=int, help='ConvE: k_h')
parser.add_argument('--num_filter', default=200, type=int, help='ConvE: number of filters in convolution')
parser.add_argument('--kernel_size', default=7, type=int, help='ConvE: kernel size to use')
parser.add_argument('--clip_grad', default=1.0, type=float, help='Gradient clipping')
parser.add_argument('--do_train', action='store_false', help='If train the model')
parser.add_argument('--do_test', action='store_false', help='If test the model')
parser.add_argument('--bi_direction', action='store_true', help='If add reverse relation to the graph')


def train(model, data_iter, graph, optimizer, params):
    """Train the model for one epoch"""
    # set the model to training mode
    model.train()

    loss_avg = utils.RunningAverage()

    with tqdm(data_iter) as bar:
        for triplets, labels in bar:
            optimizer.zero_grad()

            triplets = triplets.to(params.device)
            pred = model(triplets[:, 0], triplets[:, 1], graph)
            loss = model.loss(pred, labels.to(params.device))

            if params.n_gpu > 1 and params.multi_gpu:
                loss = loss.mean()  # mean() to average on multi-gpu
            loss.backward()

            # gradient clipping
            nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=params.clip_grad)
            optimizer.step()

            # update the progress bar
            loss_avg.update(loss.item())
            bar.set_postfix(loss='{:07.5f}'.format(loss_avg()))

    return loss_avg()


def evaluate(model, data_iters, graph, data_type, mark='Val', hits=[1, 3, 10]):
    tail_results = predict(model, data_iters, graph, data_type, params.device, mode='tail_batch')
    head_results = predict(model, data_iters, graph, data_type, params.device, mode='head_batch')

    results = {}
    count = float(tail_results['count'])

    # results['left_mr'] = np.round(tail_results['mr'] / count, 5)
    # results['left_mrr'] = np.round(tail_results['mrr'] / count, 5)
    # results['right_mr'] = np.round(head_results['mr'] / count, 5)
    # results['right_mrr'] = np.round(head_results['mrr'] / count, 5)
    results['mr'] = np.round((tail_results['mr'] + head_results['mr']) / (2 * count), 5)
    results['mrr'] = np.round((tail_results['mrr'] + head_results['mrr']) / (2 * count), 5)

    for k in hits:
        # results['left_hits@{}'.format(k)] = np.round(tail_results['hits@{}'.format(k)] / count, 5)
        # results['right_hits@{}'.format(k)] = np.round(head_results['hits@{}'.format(k)] / count, 5)
        results['hits@{}'.format(k)] = np.round((tail_results['hits@{}'.format(k)] + head_results['hits@{}'.format(k)]) / (2 * count), 5)

    metrics_str = "; ".join("{}: {:05.3f}".format(k, v) for k, v in results.items())
    logging.info("- {} metrics: {}  ".format(mark, metrics_str))

    return results


def predict(model, data_iters, graph, data_type, device, mode='tail_batch'):
    """Function to run model evaluation for a given mode

    Return:
        result['mr], result['mrr], result['hits@k']
    """
    model.eval()

    with torch.no_grad():
        results = {}
        data_iter = iter(data_iters['{}_{}'.format(data_type, mode.split('_')[0])])

        for batch in data_iter:
            triplets, label = [_.to(device) for _ in batch]
            sub, rel, obj, label = triplets[:, 0], triplets[:, 1], triplets[:, 2], label

            pred = model(sub, rel, graph)
            b_range = torch.arange(pred.size(0), device=device)
            target_pred = pred[b_range, obj]
            pred = torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
            pred[b_range, obj] = target_pred
            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]

            ranks = ranks.float()
            results['count'] = torch.numel(ranks) + results.get('count', 0.0)
            results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
            results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
            for k in range(10):
                results['hits@{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get('hits@{}'.format(k + 1), 0.0)

    return results


def train_and_evaluate(model, data_iters, graph, optimizer, scheduler, params, model_dir, restore_dir):
    """Train the model and evaluate every epoch"""

    # main evaluation criteria
    best_measure = 0
    # early stopping
    patience_counter = 0

    # reload weights from restore_dir if specified
    if restore_dir is not None:
        best_measure = utils.load_checkpoint(os.path.join(restore_dir, 'last.ckpt'), model)
        logging.info('Restore model from {} with best measure: {}'.format(os.path.join(restore_dir, 'last.ckpt'), best_measure))

    logging.info('Starting training for {} epoch(s)'.format(params.max_epoch))

    for epoch in range(1, params.max_epoch + 1):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch, params.max_epoch))
        train(model, data_iters['train'], graph, optimizer, params)
        scheduler.step()

        if epoch % params.eval_every == 0:
            val_metrics = evaluate(model, data_iters, graph, 'valid', mark='Val')

            val_measure = val_metrics['mrr']
            improve_measure = val_measure - best_measure
            if improve_measure > 0:
                best_measure = val_measure
                state = {'state_dict': model.state_dict(), 'optim_dict': optimizer.state_dict(),
                         'measure': best_measure}
                utils.save_checkpoint(state, is_best=False, checkpoint_dir=model_dir)
                if improve_measure < params.patience:
                    patience_counter += 1
                else:
                    patience_counter = 0
            else:
                patience_counter += 1

            # early stopping and logging best measure
            if params.patience_num > 0 and patience_counter >= params.patience_num and epoch > params.min_epoch:
                logging.info("Early stopping with best val measure: {:05.3f}".format(best_measure))
                break


if __name__ == '__main__':
    args = parser.parse_args()
    # directory containing saved model
    model_dir = os.path.join('experiments', args.dataset)
    # save the parameters to json file
    json_path = os.path.join(model_dir, 'params.json')
    utils.save_json(vars(args), json_path)
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
    data_loader = DataLoader(args.dataset, params)
    data_loader.graph.to(params.device)
    data_iters = data_loader.get_data_loaders(params.batch_size, params.num_workers, params)

    # prepare model
    model = MGCN(data_loader.num_entity, data_loader.num_relation, params)
    model.to(params.device)

    if params.n_gpu > 1 and args.multi_gpu:
        model = torch.nn.DataParallel(model)

    # prepare optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.95, last_epoch=-1)

    # train and evaluate the model
    if params.do_train:
        train_and_evaluate(model, data_iters, data_loader.graph, optimizer, scheduler, params, model_dir, args.restore_dir)
    if params.do_test:
        evaluate(model, data_iters, data_loader.graph, 'test', mark='Test')
