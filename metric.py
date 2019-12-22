"""
Calculate metrics for knowledge graph representation and use multi-processing for accelerating.
Be care for the termination of the sub-processes. For example, the CTRL+C signal.
"""

import torch
import numpy as np
from tqdm import tqdm


# def score_func(entity, relation, triplets):
#     scores = entity[triplets[:, 0].long()] + relation[triplets[:, 1].long()] - entity[triplets[:, 2].long()]
#     return torch.norm(scores, p=1, dim=1)


# def calc_mrr(entity, model, test_triplets, all_triplets, device, hits=[]):
#     """Calculate MRR (filtered) and Hits @ (1, 3, 10)

#     For each test triplet, the head is removed and replaced by each of the entities of the dictionary in turn.
#     Dissimi-larities (or energies) of those corrupted triplets are first computed by the models and then sorted by
#     ascending order; the rank of the correct entity is finally stored. This whole procedure is repeated
#     while removing the tail instead of the head.

#     Some corrupted triplets maybe valid ones, from the training set for instance. TransE propose to remove from the
#     list of corrupted triplets all the triplets that appear either in the training, validation or test set
#     (except the test triplet of interest)

#     Args:
#         entity: entity embeddings output by our model
#         model: (MGCN) used for score function
#         test_triplets: (2-d array) triplets in test set
#         all_triplets: (2-d array) triplets in train, valid and test set
#         hits: (list) [1, 3, 10]

#     Return:
#         metrics: (dict) including mrr and hits@n
#     """
#     entity = entity.to(device)
#     test_triplets = torch.from_numpy(test_triplets).long()
#     all_triplets = torch.from_numpy(all_triplets).long()

#     # ranking results when replace head entity or tail entity
#     ranks_s = []
#     ranks_o = []

#     for test_triplet in tqdm(test_triplets):

#         # Perturb object firstly
#         sub, rel, obj = test_triplet[0], test_triplet[1], test_triplet[2]

#         # generate filtered tail entities for replacement
#         # find triplets which have the same subject_relation with current triplet
#         subject_relation = test_triplet[:2]
#         delete_indices = torch.sum(all_triplets[:, :2] == subject_relation, dim=1)
#         delete_indices = torch.nonzero(delete_indices == 2).squeeze()

#         # using delete_indices to get all valid tail entities in those triplets
#         delete_entity_ids = all_triplets[delete_indices, 2].view(-1).numpy()
#         perturb_entity_ids = np.setdiff1d(np.arange(entity.size(0), dtype=np.int64), np.unique(delete_entity_ids))
#         perturb_entity_ids = torch.from_numpy(perturb_entity_ids)
#         # add the current test triplet
#         perturb_entity_ids = torch.cat((perturb_entity_ids, obj.view(-1)))

#         # generate new triplets for scoring
#         corrupted_triplets = torch.cat(
#             (
#                 sub * torch.ones_like(perturb_entity_ids).view(-1, 1),
#                 rel * torch.ones_like(perturb_entity_ids).view(-1, 1),
#                 perturb_entity_ids.view(-1, 1)
#             ),
#             dim=1
#         )

#         # scores = score_func(entity, relation, corrupted_triplets)
#         scores = model.score_func(entity, corrupted_triplets.to(device))
#         rank = torch.argmax(torch.argsort(scores.cpu()))
#         ranks_o.append(rank)

#         # Then, perturb subjects
#         relation_object = torch.tensor([rel, obj])
#         delete_indices = torch.sum(all_triplets[:, 1:3] == relation_object, dim=1)
#         delete_indices = torch.nonzero(delete_indices == 2).squeeze()

#         delete_entity_ids = all_triplets[delete_indices, 0].view(-1).numpy()
#         perturb_entity_ids = np.setdiff1d(np.arange(entity.size(0), dtype=np.int64), np.unique(delete_entity_ids))
#         perturb_entity_ids = torch.from_numpy(perturb_entity_ids)
#         # add the current test triplet
#         perturb_entity_ids = torch.cat((perturb_entity_ids, sub.view(-1)))

#         # generate new triplets for scoring
#         corrupted_triplets = torch.cat(
#             (
#                 perturb_entity_ids.view(-1, 1),
#                 torch.ones_like(perturb_entity_ids).view(-1, 1) * rel,
#                 torch.ones_like(perturb_entity_ids).view(-1, 1) * obj,
#             ),
#             dim=1
#         )

#         scores = model.score_func(entity, corrupted_triplets.to(device))
#         rank = torch.argmax(torch.argsort(scores.cpu()))
#         ranks_s.append(rank)

#     # begin to compute mrr and hit@n using ranks
#     metrics = {}
#     ranks = torch.cat([torch.tensor(ranks_s), torch.tensor(ranks_o)])

#     metrics['mrr'] = torch.mean(ranks.float()).item()

#     for hit in hits:
#         avg_count = torch.mean((ranks < hit).float())
#         metrics['hits@{}'.format(hit)] = avg_count.item()

#     return metrics


def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices

# return MRR (filtered), and Hits @ (1, 3, 10)
def calc_mrr(embedding, w, test_triplets, all_triplets, hits=[]):
    with torch.no_grad():
        
        ranks_s = []
        ranks_o = []

        head_relation_triplets = all_triplets[:, :2]
        tail_relation_triplets = torch.stack((all_triplets[:, 2], all_triplets[:, 1])).transpose(0, 1)

        for test_triplet in tqdm(test_triplets):

            # Perturb object
            subject = test_triplet[0]
            relation = test_triplet[1]
            object_ = test_triplet[2]

            subject_relation = test_triplet[:2]
            delete_index = torch.sum(head_relation_triplets == subject_relation, dim = 1)
            delete_index = torch.nonzero(delete_index == 2).squeeze()

            delete_entity_index = all_triplets[delete_index, 2].view(-1).numpy()
            perturb_entity_index = np.array(list(set(np.arange(40943)) - set(delete_entity_index)))
            perturb_entity_index = torch.from_numpy(perturb_entity_index)
            perturb_entity_index = torch.cat((perturb_entity_index, object_.view(-1)))
            
            emb_ar = embedding[subject] * w[relation]
            emb_ar = emb_ar.view(-1, 1, 1)

            emb_c = embedding[perturb_entity_index]
            emb_c = emb_c.transpose(0, 1).unsqueeze(1)
            
            out_prod = torch.bmm(emb_ar, emb_c)
            score = torch.sum(out_prod, dim = 0)
            score = torch.sigmoid(score)
            
            target = torch.tensor(len(perturb_entity_index) - 1)
            ranks_s.append(sort_and_rank(score, target))

            # Perturb subject
            object_ = test_triplet[2]
            relation = test_triplet[1]
            subject = test_triplet[0]

            object_relation = torch.tensor([object_, relation])
            delete_index = torch.sum(tail_relation_triplets == object_relation, dim = 1)
            delete_index = torch.nonzero(delete_index == 2).squeeze()

            delete_entity_index = all_triplets[delete_index, 2].view(-1).numpy()
            perturb_entity_index = np.array(list(set(np.arange(40943)) - set(delete_entity_index)))
            perturb_entity_index = torch.from_numpy(perturb_entity_index)
            perturb_entity_index = torch.cat((perturb_entity_index, subject.view(-1)))

            emb_ar = embedding[object_] * w[relation]
            emb_ar = emb_ar.view(-1, 1, 1)

            emb_c = embedding[perturb_entity_index]
            emb_c = emb_c.transpose(0, 1).unsqueeze(1)

            out_prod = torch.bmm(emb_ar, emb_c)
            score = torch.sum(out_prod, dim = 0)
            score = torch.sigmoid(score)

            target = torch.tensor(len(perturb_entity_index) - 1)
            ranks_o.append(sort_and_rank(score, target))

        ranks_s = torch.cat(ranks_s)
        ranks_o = torch.cat(ranks_o)

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1 # change to 1-indexed

        metrics = {}

        mrr = torch.mean(1.0 / ranks.float())
        metrics['mrr'] = mrr.item()

        for hit in hits:
            avg_count = torch.mean((ranks < hit).float())
            metrics['hits@{}'.format(hit)] = avg_count.item()
            
    return metrics
