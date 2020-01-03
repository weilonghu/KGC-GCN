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
from data_loader import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='FB15k-237', help="Directory containing the dataset")
parser.add_argument('--seed', default=2020, help="random seed for initialization")
parser.add_argument('--restore_dir', default=None, help='Optional, directory containing weights to reload before training')
parser.add_argument('--multi_gpu', default=False, action='store_true', help="Whether to use multiple GPUs if available")
parser.add_argument('--batch_size', default=128, type=int, help="Batch size")
parser.add_argument('--max_epoch', default=500, type=int, help='Number of maximum epochs')
parser.add_argument('--min_epoch', default=500, type=int, help='Number of minimum epochs')
parser.add_argument('--eval_every', default=5, type=int, help='Number of epochs to test the model')
parser.add_argument('--patience', default=0.01, type=float, help='Increasement between two epochs')
parser.add_argument('--patience_num', default=10, type=int, help='Early stopping creteria')
parser.add_argument('--learning_rate', default=0.001, type=float, help='Learning rate')
parser.add_argument('--lbl_smooth', default=0.1, type=float, help="Label smoothing")
parser.add_argument('--num_workers', default=0, type=int, help='Number of processes to construct batches')
parser.add_argument('--bias', action='store_true', help='Whether to use bias in the model')
parser.add_argument('--embed_dim', default=100, type=int, help='Dimension size for entities and relations')
parser.add_argument('--hidden_drop', default=0.3, type=float, help='Dropout after GCN')
parser.add_argument('--hidden_drop2', default=0.3, type=float, help='ConvE: hidden dropout')
parser.add_argument('--feat_drop', default=0.3, type=float, help='ConvE: feature dropout')
parser.add_argument('--k_w', default=10, type=int, help='ConvE: k_w')
parser.add_argument('--k_h', default=20, type=int, help='ConvE: k_h')
parser.add_argument('--num_filter', default=200, type=int, help='ConvE: number of filters in convolution')
parser.add_argument('--kernel_size', default=7, type=int, help='ConvE: kernel size to use')
parser.add_argument('--clip_grad', default=1.0, type=float, help='Gradient clipping')
parser.add_argument('--do_train', action='store_true', help='If train the model')
parser.add_argument('--do_test', action='store_true', help='If test the model')


def train(model, data_iter, optimizer, params):
    """Train the model for one epoch"""
    # set the model to training mode
    model.train()

    epoch = trange(len(data_iter) // params.batch_size)
    for step, batch in zip(epoch, data_iter):
        optimizer.zero_grad()

        pred = model(batch)
        loss = model.loss(pred, batch.labels)

        if params.n_gpu > 1 and params.multi_gpu:
            loss = loss.mean()  # mean() to average on multi-gpu
        loss.backward()

        # gradient clipping
        nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=params.clip_grad)
        optimizer.step()

        # update the progress bar
        epoch.set_postfix(loss='{:05.3f}'.format(loss))


def evaluate(model, test_graph, eval_triplets, all_triplets, params, mark='Eval', verbose=False):
    """Evaluate the model on dataset 'data'"""
    # set the model to evaluation mode
    model.eval()
    model.cpu()

    # compute embeddings for entities
    with torch.no_grad():
        entity_embedding = model(test_graph)
        relation_embeddings = model.relation_embedding.cpu()
        # calculate mrr
        metrics = calc_mrr(entity_embedding, relation_embeddings, eval_triplets, all_triplets, hits=[1, 3, 10])

    model.to(params.device)

    # logging and report
    metrics['measure'] = metrics['mrr']
    metrics_str = "; ".join("{}: {:05.3f}".format(k, v)
                            for k, v in metrics.items())
    tqdm.write("- {} metrics: {}  ".format(mark, metrics_str))

    return metrics


def train_and_evaluate(model, data_loader, optimizer, scheduler, params, model_dir, restore_dir):
    """Train the model and evaluate every epoch"""

    # main evaluation criteria
    best_measure = 0
    # early stopping
    patience_counter = 0

    # reload weights from restore_dir if specified
    if restore_dir is not None:
        best_measure = utils.load_checkpoint(os.path.join(restore_dir, 'last.ckpt'), model)
        logging.info('Restore model from {} with best measure: {}'.format(os.path.join(restore_dir, 'last.ckpt'), best_measure))

    # create dataset
    train_set = data_loader.get_dataset('train', params)
    eval_set = data_loader.get_dataset('valid', params)
    train_iter = data_loader.data_iterator(train_set, params.batch_size, params.num_worker, shuffle=True)
    eval_iter = data_loader.data_iterator(eval_set, params.batch_size, 0, False)

    logging.info('Starting training for {} epoch(s)'.format(params.max_epoch))

    for epoch in range(1, params.max_epochs + 1):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch, params.max_epoch + 1))
        train(model, train_iter, optimizer, params)
        scheduler.step()

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

    # prepare model
    model = MGCN(data_loader.num_entity, data_loader.num_relation, params)
    model.to(params.device)

    if params.n_gpu > 1 and args.multi_gpu:
        model = torch.nn.DataParallel(model)

    # prepare optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95, last_epoch=-1)

    # train and evaluate the model
    train_and_evaluate(model, data_loader, optimizer, scheduler,
                       params, model_dir, args.restore_dir)
