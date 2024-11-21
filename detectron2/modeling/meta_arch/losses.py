from __future__ import print_function

import torch
import torch.nn as nn

import torch.nn.functional as F


class MemConLoss_trans(nn.Module):
    def __init__(self, temperature=0.07):
        super(MemConLoss_trans, self).__init__()
        self.temperature = temperature

    def get_score(self, mem_bank, query, items=None):
        score = torch.matmul(query.float(), torch.t(mem_bank).float())
        score_memory = F.softmax(score, dim=1)
        _, top_neg_idx = torch.topk(score_memory, items, dim=1, largest=False)
        neg_logits = torch.gather(score, 1, top_neg_idx)
        return neg_logits

    def forward(self, s_query, s_box_feat, mem_s_query, mem_bank):
        batch_size, _ = s_query.shape
        mask = torch.eye(batch_size, dtype=torch.float32).cuda()

        anchor_feat = F.normalize(s_query, dim=1)
        contrast_feat = F.normalize(mem_s_query, dim=1)

        logits = torch.div(torch.matmul(anchor_feat, contrast_feat.T), self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        sm_logits = logits - logits_max.detach()

        # mem_query = s_box_feat.contiguous().detach()
        # mem_query = s_query.contiguous()
        # sm_neg_logits = self.get_score(mem_bank, mem_query, items=5)
        # s_all_logits = torch.exp(torch.cat((sm_logits, sm_neg_logits), dim=1))

        s_all_logits = torch.exp(sm_logits)

        log_prob = sm_logits - torch.log(s_all_logits.sum(1, keepdim=True))
        loss = -1 * ((mask * log_prob).sum(1) / mask.sum(1))

        if torch.isnan(loss.mean()):
            loss = loss * 0

        return loss.mean()


class GraphConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07 * 2):
        super(GraphConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

        dim_in = 1024
        feat_dim = 1024
        self.head_1 = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim)
        )
        self.head_2 = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim)
        )

    def forward(self, t_feat, s_feat, graph_cn, labels=None, mask=None):
        qx = graph_cn.graph.wq(s_feat)
        kx = graph_cn.graph.wk(s_feat)
        sim_mat = qx.matmul(kx.transpose(-1, -2))
        dot_mat = sim_mat.detach().clone()

        thresh = 0.5
        dot_mat -= dot_mat.min(1, keepdim=True)[0]
        dot_mat /= dot_mat.max(1, keepdim=True)[0]
        mask = ((dot_mat > thresh) * 1).detach().clone()
        mask.fill_diagonal_(1)

        anchor_feat = self.head_1(s_feat)
        contrast_feat = self.head_2(s_feat)

        anchor_feat = F.normalize(anchor_feat, dim=1)
        contrast_feat = F.normalize(contrast_feat, dim=1)

        ss_anchor_dot_contrast = torch.div(torch.matmul(anchor_feat, contrast_feat.T),
                                           self.temperature)  ##### torch.Size([6, 6])
        logits_max, _ = torch.max(ss_anchor_dot_contrast, dim=1,
                                  keepdim=True)  ##### torch.Size([6, 1]) - contains max value along dim=1
        ss_graph_logits = ss_anchor_dot_contrast - logits_max.detach()

        ss_graph_all_logits = torch.exp(ss_graph_logits)
        ss_log_prob = ss_graph_logits - torch.log(ss_graph_all_logits.sum(1, keepdim=True))
        ss_mean_log_prob_pos = (mask * ss_log_prob).sum(1) / mask.sum(1)

        # loss
        ss_loss = - (self.temperature / self.base_temperature) * ss_mean_log_prob_pos
        ss_loss = ss_loss.mean()

        return ss_loss


class CTAODConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(CTAODConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, s_box_features, t_box_features):
        batch_size, _ = s_box_features.shape
        mask = torch.eye(batch_size, dtype=torch.float32).cuda()

        anchor_feat = F.normalize(s_box_features, dim=1)
        contrast_feat = F.normalize(t_box_features, dim=1)

        logits = torch.div(torch.matmul(anchor_feat, contrast_feat.T), self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        sm_logits = logits - logits_max.detach()
        s_all_logits = torch.exp(sm_logits)
        log_prob = sm_logits - torch.log(s_all_logits.sum(1, keepdim=True))
        loss = -1 * ((mask * log_prob).sum(1) / mask.sum(1))

        if torch.isnan(loss.mean()):
            loss = loss * 0

        return loss.mean()
