"""Calculate metrics for knowledge graph representation"""

import torch
import numpy as np
# import multiprocessing as mp

import os
from tqdm import tqdm
import utils
from data_set import DataSet


def score_func(entity, relation, triplets):
    scores = entity[triplets[:, 0].long()] + relation[triplets[:, 1].long()] - entity[triplets[:, 2].long()]
    return torch.mean(scores, dim=1)


def calc_mrr(entity, relation, model, test_triplets, all_triplets, hits=[]):
    """Calculate MRR (filtered) and Hits @ (1, 3, 10)

    For each test triplet, the head is removed and replaced by each of the entities of the dictionary in turn.
    Dissimi-larities (or energies) of those corrupted triplets are first computed by the models and then sorted by
    ascending order; the rank of the correct entity is finally stored. This whole procedure is repeated
    while removing the tail instead of the head.

    Some corrupted triplets maybe valid ones, from the training set for instance. TransE propose to remove from the
    list of corrupted triplets all the triplets that appear either in the training, validation or test set
    (except the test triplet of interest)

    Args:
        entity: entity embeddings output by our model
        relation: relation embeddings output by out model
        model: (MGCN) used for score function
        test_triplets: (2-d array) triplets in test set
        all_triplets: (2-d array) triplets in train, valid and test set
        hits: (list) [1, 3, 10]

    Return:
        metrics: (dict) including mrr and hits@n
    """
    test_triplets = torch.from_numpy(test_triplets)
    all_triplets = torch.from_numpy(all_triplets)

    # ranking results when replace head entity or tail entity
    ranks_s = []
    ranks_o = []

    # def _compute_rank(test_triplets):
    for test_triplet in tqdm(test_triplets):

        # Perturb object firstly
        sub, rel, obj = test_triplet[0], test_triplet[1], test_triplet[2]

        # generate filtered tail entities for replacement
        # find triplets which have the same subject_relation with current triplet
        subject_relation = test_triplet[:2]
        delete_indices = torch.sum(all_triplets[:, :2] == subject_relation, dim=1)
        delete_indices = torch.nonzero(delete_indices == 2).squeeze()

        # using delete_indices to get all valid tail entities in those triplets
        delete_entity_ids = all_triplets[delete_indices, 2].view(-1).numpy()
        # perturb_entity_index = np.array(list(set(np.arange(14541)) - set(delete_entity_index)))
        perturb_entity_ids = np.setdiff1d(np.arange(entity.size(0)), np.unique(delete_entity_ids))
        perturb_entity_ids = torch.from_numpy(perturb_entity_ids)
        # add the current test triplet
        perturb_entity_ids = torch.cat((perturb_entity_ids, obj.view(-1)))

        # generate new triplets for scoring
        corrupted_triplets = torch.cat(
            (
                sub * torch.ones(perturb_entity_ids.size(0), dtype=torch.int32).view(-1, 1),
                rel * torch.ones(perturb_entity_ids.size(0), dtype=torch.int32).view(-1, 1),
                perturb_entity_ids.view(-1, 1)
            ),
            dim=1
        )

        scores = score_func(entity, relation, corrupted_triplets)
        rank = torch.argmax(torch.argsort(scores))
        ranks_o.append(rank)

        # Then, perturb subjects
        relation_object = torch.tensor([rel, obj])
        delete_indices = torch.sum(all_triplets[:, 1:3] == relation_object, dim=1)
        delete_indices = torch.nonzero(delete_indices == 2).squeeze()

        delete_entity_ids = all_triplets[delete_indices, 0].view(-1).numpy()
        perturb_entity_ids = np.setdiff1d(np.arange(entity.size(0)), np.unique(delete_entity_ids))
        perturb_entity_ids = torch.from_numpy(perturb_entity_ids)
        # add the current test triplet
        perturb_entity_ids = torch.cat((perturb_entity_ids, sub.view(-1)))

        # generate new triplets for scoring
        corrupted_triplets = torch.cat(
            (
                perturb_entity_ids.view(-1, 1),
                torch.ones(perturb_entity_ids.size(0), dtype=torch.int32).view(-1, 1) * rel,
                torch.ones(perturb_entity_ids.size(0), dtype=torch.int32).view(-1, 1) * obj,
            ),
            dim=1
        )

        scores = score_func(entity, relation, corrupted_triplets)
        rank = torch.argmax(torch.argsort(scores))
        ranks_s.append(rank)

    # begin to compute mrr and hit@n using ranks
    metrics = {}
    ranks = torch.cat([torch.tensor(ranks_s), torch.tensor(ranks_o)])

    metrics['mrr'] = torch.mean(ranks.float()).item()

    for hit in hits:
        avg_count = torch.mean((ranks < hit).float())
        metrics['hits@{}'.format(hit)] = avg_count.item()

    return metrics


if __name__ == '__main__':
    # directory containing saved model
    model_dir = os.path.join('experiments', 'WN18')
    # load the parameters from json file
    json_path = os.path.join(model_dir, 'params.json')
    assert os.path.isfile(
        json_path), 'No json configuration file found at {}'.format(json_path)
    params = utils.Params(json_path)

    params.device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    dataset = DataSet('WN18', params)
    # evaluation triplet
    eval_triplets = dataset.test_triplets
    all_triplets = dataset.total_triplets()

    metrics = calc_mrr(dataset.pretrained_entity, dataset.pretrained_relation, None, eval_triplets, all_triplets, hits=[1, 3, 10])

    metrics_str = "; ".join("{}: {:05.2f}".format(k, v)
                            for k, v in metrics.items())
    print("- {} metrics: ".format('test') + metrics_str)
