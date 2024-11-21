import torch
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as F

import functools
import numpy as np


def random_uniform(shape, low, high, cuda):
    x = torch.rand(*shape)
    result_cpu = (high - low) * x + low
    if cuda:
        return result_cpu.cuda()
    else:
        return result_cpu


def distance(a, b):
    return torch.sqrt(((a - b) ** 2).sum()).unsqueeze(0)


def distance_batch(a, b):
    bs, _ = a.shape
    result = distance(a[0], b)
    for i in range(bs - 1):
        result = torch.cat((result, distance(a[i], b)), 0)
    return result


def multiply(x):  # to flatten matrix into a vector
    return functools.reduce(lambda x, y: x * y, x, 1)


def flatten(x):
    """ Flatten matrix into a vector """
    count = multiply(x.size())
    return x.resize_(count)


def index(batch_size, x):
    idx = torch.arange(0, batch_size).long()
    idx = torch.unsqueeze(idx, -1)
    return torch.cat((idx, x), dim=1)


def MemoryLoss(memory):
    m, d = memory.size()
    memory_t = torch.t(memory)
    similarity = (torch.matmul(memory, memory_t)) / 2 + 1 / 2  # 30X30
    identity_mask = torch.eye(m).cuda()
    sim = torch.abs(similarity - identity_mask)
    return torch.sum(sim) / (m * (m - 1))


class Memory_trans_update(nn.Module):
    def __init__(self):
        super(Memory_trans_update, self).__init__()

    def get_update_query(self, mem, max_indices, score, value):
        m, d = mem.size()
        query_updated_memory = torch.zeros((m, d)).cuda()
        for i in range(m):
            idx = torch.nonzero(max_indices.squeeze(1) == i)
            if idx.size()[0] == 0:
                query_updated_memory[i] = 0
            else:
                query_updated_memory[i] = torch.sum(((score[idx, i] / torch.max(score[:, i])) * value[idx].squeeze(1)),
                                                    dim=0)
        return query_updated_memory

    def get_score(self, mem, query):
        score = torch.matmul(query.float(), torch.t(mem).float())
        score_query = F.softmax(score, dim=0)
        score_memory = F.softmax(score, dim=1)
        return score_query, score_memory

    def forward(self, keys, query, value):
        softmax_score_query, softmax_score_memory = self.get_score(keys, query)
        _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)  # 10x1 #row
        query_updated_memory = self.get_update_query(keys, gathering_indices, softmax_score_query, value)
        updated_memory = F.normalize(query_updated_memory + keys, dim=1)
        return updated_memory.detach()


class Memory_trans_read(nn.Module):
    def __init__(self, ):
        super(Memory_trans_read, self).__init__()
        # Constants
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def get_score(self, mem, query):
        score = torch.matmul(query.float(), torch.t(mem).float())
        score_query = F.softmax(score, dim=0)
        score_memory = F.softmax(score, dim=1)
        return score_query, score_memory

    def forward(self, memory, query):
        _, softmax_score_memory = self.get_score(memory, query)
        concat_memory = torch.matmul(softmax_score_memory.detach().clone(), memory)
        return concat_memory