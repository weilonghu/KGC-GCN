"""Calculate metrics for knowledge graph representation"""

import torch
import numpy as np


def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices


def calc_mrr(embedding, w, test_triplets, all_triplets, hits=[]):
    """Calculate MRR (filtered) and Hits @ (1, 3, 10)

    Args:
        embeddings: entity embeddings
        w: relation embeddings
        test_triplets: (2-d array) triplets in test set
        all_triplets: (2-d array) triplets in train, valid and test set
        hits: (list)

    Return:
        filtered MRR and Hit@n
    """
    test_triplets = torch.from_numpy(test_triplets)
    all_triplets = torch.from_numpy(all_triplets)

    ranks_s = []
    ranks_o = []

    head_relation_triplets = all_triplets[:, :2]
    tail_relation_triplets = torch.stack((all_triplets[:, 2], all_triplets[:, 1])).transpose(0, 1)

    for test_triplet in test_triplets:

        # Perturb object
        subject = test_triplet[0]
        relation = test_triplet[1]
        object_ = test_triplet[2]

        subject_relation = test_triplet[:2]
        delete_index = torch.sum(head_relation_triplets == subject_relation, dim=1)
        delete_index = torch.nonzero(delete_index == 2).squeeze()

        delete_entity_index = all_triplets[delete_index, 2].view(-1).numpy()
        perturb_entity_index = np.array(list(set(np.arange(14541)) - set(delete_entity_index)))
        perturb_entity_index = torch.from_numpy(perturb_entity_index)
        perturb_entity_index = torch.cat((perturb_entity_index, object_.view(-1)))

        emb_ar = embedding[subject] * w[relation]
        emb_ar = emb_ar.view(-1, 1, 1)

        emb_c = embedding[perturb_entity_index]
        emb_c = emb_c.transpose(0, 1).unsqueeze(1)

        out_prod = torch.bmm(emb_ar, emb_c)
        score = torch.sum(out_prod, dim=0)
        score = torch.sigmoid(score)

        target = torch.tensor(len(perturb_entity_index) - 1)
        ranks_s.append(sort_and_rank(score, target))

        # Perturb subject
        object_ = test_triplet[2]
        relation = test_triplet[1]
        subject = test_triplet[0]

        object_relation = torch.tensor([object_, relation])
        delete_index = torch.sum(tail_relation_triplets == object_relation, dim=1)
        delete_index = torch.nonzero(delete_index == 2).squeeze()

        delete_entity_index = all_triplets[delete_index, 2].view(-1).numpy()
        perturb_entity_index = np.array(list(set(np.arange(14541)) - set(delete_entity_index)))
        perturb_entity_index = torch.from_numpy(perturb_entity_index)
        perturb_entity_index = torch.cat((perturb_entity_index, subject.view(-1)))

        emb_ar = embedding[object_] * w[relation]
        emb_ar = emb_ar.view(-1, 1, 1)

        emb_c = embedding[perturb_entity_index]
        emb_c = emb_c.transpose(0, 1).unsqueeze(1)

        out_prod = torch.bmm(emb_ar, emb_c)
        score = torch.sum(out_prod, dim=0)
        score = torch.sigmoid(score)

        target = torch.tensor(len(perturb_entity_index) - 1)
        ranks_o.append(sort_and_rank(score, target))

    ranks_s = torch.cat(ranks_s)
    ranks_o = torch.cat(ranks_o)

    ranks = torch.cat([ranks_s, ranks_o])
    ranks += 1  # change to 1-indexed

    mrr = torch.mean(1.0 / ranks.float())
    print("MRR (filtered): {:.6f}".format(mrr.item()))

    for hit in hits:
        avg_count = torch.mean((ranks <= hit).float())
        print("Hits (filtered) @ {}: {:.6f}".format(hit, avg_count.item()))

    return mrr.item()
