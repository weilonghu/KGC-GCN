"""
Calculate metrics for knowledge graph representation and use multi-processing for accelerating.
Be care for the termination of the sub-processes. For example, the CTRL+C signal.
"""

import torch
import numpy as np
from tqdm import tqdm


def sort_and_rank(scores, target):
    """Find the rank of the 'target'-th element in 'scores'

    Example:
        scores: [0.2, 0.3, 0.1, 0.5]
        target: 3
        return 0
    Args:
        scores: scores of all corrupted triplets, shape=[1, N]
        target: index of the intersted triplet, shape=[]
    Return:
        rank of the interested triplet, shape=[]
    """
    _, indices = torch.sort(scores.view(1, -1), dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices


def calc_mrr(entity, relation, test_triplets, all_triplets, hits=[]):
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
        model: (MGCN) used as score function
        test_triplets: (2-d array) triplets in test set
        all_triplets: (2-d array) triplets in train, valid and test set
        hits: (list) [1, 3, 10]

    Return:
        metrics: (dict) including mrr and hits@n
    """

    # ranking results when replace head entity or tail entity
    ranks_s = []
    ranks_o = []

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
        perturb_entity_ids = np.setdiff1d(np.arange(entity.size(0), dtype=np.int64), np.unique(delete_entity_ids))
        perturb_entity_ids = torch.from_numpy(perturb_entity_ids)
        # add the current test triplet
        perturb_entity_ids = torch.cat((perturb_entity_ids, obj.view(-1)))

        # generate new triplets for scoring
        # corrupted_triplets = torch.cat(
        #     (
        #         sub * torch.ones_like(perturb_entity_ids).view(-1, 1),
        #         rel * torch.ones_like(perturb_entity_ids).view(-1, 1),
        #         perturb_entity_ids.view(-1, 1)
        #     ),
        #     dim=1
        # )

        # calculate scores for all corrupted triplets
        product = entity[sub] * relation[rel]
        product = product.view(-1, 1, 1)

        perurb_obj_emb = entity[perturb_entity_ids].transpose(0, 1).unsqueeze(1)
            
        out_prod = torch.bmm(product, perurb_obj_emb)

        scores = torch.sigmoid(torch.sum(out_prod, dim = 0))
        # scores = model.score_func(entity, corrupted_triplets)
            
        target = torch.tensor(len(perturb_entity_ids) - 1)
        ranks_o.append(sort_and_rank(scores, target))

        # Then, perturb subjects
        relation_object = torch.tensor([rel, obj])
        delete_indices = torch.sum(all_triplets[:, 1:3] == relation_object, dim=1)
        delete_indices = torch.nonzero(delete_indices == 2).squeeze()

        delete_entity_ids = all_triplets[delete_indices, 0].view(-1).numpy()
        perturb_entity_ids = np.setdiff1d(np.arange(entity.size(0), dtype=np.int64), np.unique(delete_entity_ids))
        perturb_entity_ids = torch.from_numpy(perturb_entity_ids)
        # add the current test triplet
        perturb_entity_ids = torch.cat((perturb_entity_ids, sub.view(-1)))

        # generate new triplets for scoring
        # corrupted_triplets = torch.cat(
        #     (
        #         perturb_entity_ids.view(-1, 1),
        #         rel * torch.ones_like(perturb_entity_ids).view(-1, 1),
        #         obj * torch.ones_like(perturb_entity_ids).view(-1, 1),
        #     ),
        #     dim=1
        # )

        # calculate scores for all corrupted triplets
        product = entity[obj] * relation[rel]
        product = product.view(-1, 1, 1)

        perurb_sub_emb = entity[perturb_entity_ids].transpose(0, 1).unsqueeze(1)
            
        out_prod = torch.bmm(product, perurb_sub_emb)

        scores = torch.sigmoid(torch.sum(out_prod, dim = 0))
        # scores = model.score_func(entity, corrupted_triplets)
            
        target = torch.tensor(len(perturb_entity_ids) - 1)
        ranks_s.append(sort_and_rank(scores, target))

    # begin to compute mrr and hit@n using ranks
    metrics = {}
    ranks = torch.cat([torch.tensor(ranks_s), torch.tensor(ranks_o)]) + 1  # change to 1-indexed

    metrics['mrr'] = torch.mean(1.0 / ranks.float()).item()

    for hit in hits:
        avg_count = torch.mean((ranks <= hit).float())
        metrics['hits@{}'.format(hit)] = avg_count.item()

    return metrics
