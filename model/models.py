# -*- coding: utf-8 -*-
# @Time    : 2021/1/25 6:40 PM
# @Author  : liuxiyang
from helper import *
from model.ragat_conv import RagatConv
from model.SpecialSpmmFinal import SpecialSpmmFinal
import torch.nn as nn


class BaseModel(torch.nn.Module):
    def __init__(self, params):
        super(BaseModel, self).__init__()

        self.p = params
        self.act = torch.tanh
        self.bceloss = torch.nn.BCELoss()

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)


class RagatBase(BaseModel):
    def __init__(self, edge_index, edge_type, num_rel, params=None):
        super(RagatBase, self).__init__(params)

        self.edge_index = edge_index
        self.edge_type = edge_type
        self.p.gcn_dim = self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim
        self.init_embed = get_param((self.p.num_ent, self.p.init_dim))
        self.device = self.edge_index.device

        if self.p.score_func == 'transe':
            self.init_rel = get_param((num_rel, self.p.init_dim))
        else:
            self.init_rel = get_param((num_rel * 2, self.p.init_dim))

        self.conv1 = RagatConv(self.edge_index, self.edge_type, self.p.init_dim, self.p.gcn_dim, num_rel,
                               act=self.act, params=self.p, head_num=self.p.head_num)
        self.conv2 = RagatConv(self.edge_index, self.edge_type, self.p.gcn_dim, self.p.embed_dim, num_rel,
                               act=self.act, params=self.p, head_num=1) if self.p.gcn_layer == 2 else None

        self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))
        self.special_spmm = SpecialSpmmFinal()
        self.rel_drop = nn.Dropout(0.1)

    def forward_base(self, sub, rel, drop1, drop2):
        # r: (2 * num_relation) x init_dim
        init_rel = self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
        ent_embed1, rel_embed1 = self.conv1(x=self.init_embed, rel_embed=init_rel)
        ent_embed1 = drop1(ent_embed1)

        ent_embed2, rel_embed2 = self.conv2(x=ent_embed1, rel_embed=rel_embed1) if self.p.gcn_layer == 2 else (
            ent_embed1, rel_embed1)
        ent_embed2 = drop2(ent_embed2) if self.p.gcn_layer == 2 else ent_embed1

        final_ent = ent_embed2 if self.p.gcn_layer == 2 else ent_embed1
        final_rel = rel_embed2 if self.p.gcn_layer == 2 else rel_embed1
        sub_emb = torch.index_select(final_ent, 0, sub)
        rel_emb = torch.index_select(final_rel, 0, rel)
        return sub_emb, rel_emb, final_ent

    def gather_neighbours(self):
        edge_weight = torch.ones_like(self.edge_type).float().unsqueeze(1)
        deg = self.special_spmm(self.edge_index, edge_weight, self.p.num_ent, self.p.num_ent, 1,
                                dim=1)
        deg[deg == 0.0] = 1.0
        entity_neighbours = self.init_embed[self.edge_index[1, :], :]
        entity_gathered = self.special_spmm(
            self.edge_index, entity_neighbours, self.p.num_ent, self.p.num_ent, self.p.init_dim,
            dim=1).div(deg)
        relation_neighbours = torch.index_select(self.init_rel, 0, self.edge_type)
        relation_gathered = self.special_spmm(
            self.edge_index, relation_neighbours, self.p.num_ent, self.p.num_ent, self.p.init_dim, dim=1).div(deg)
        return entity_gathered, relation_gathered


class RagatTransE(RagatBase):
    def __init__(self, edge_index, edge_type, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
        self.drop = torch.nn.Dropout(self.p.hid_drop)

    def forward(self, sub, rel):
        sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.drop, self.drop)
        obj_emb = sub_emb + rel_emb

        x = self.p.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)
        score = torch.sigmoid(x)

        return score


class RagatDistMult(RagatBase):
    def __init__(self, edge_index, edge_type, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
        self.drop = torch.nn.Dropout(self.p.hid_drop)

    def forward(self, sub, rel):
        sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.drop, self.drop)
        obj_emb = sub_emb * rel_emb

        x = torch.mm(obj_emb, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)

        score = torch.sigmoid(x)
        return score


class RagatConvE(RagatBase):
    def __init__(self, edge_index, edge_type, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
        self.embed_dim = self.p.embed_dim

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.p.num_filt)
        self.bn2 = torch.nn.BatchNorm1d(self.embed_dim)

        self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.p.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.p.feat_drop)
        self.m_conv1 = torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz),
                                       stride=1, padding=0, bias=self.p.bias)

        flat_sz_h = int(2 * self.p.k_w) - self.p.ker_sz + 1
        flat_sz_w = self.p.k_h - self.p.ker_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.p.num_filt
        self.fc = torch.nn.Linear(self.flat_sz, self.embed_dim)

    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.embed_dim)
        stack_inp = torch.cat([e1_embed, rel_embed], 1)
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2 * self.p.k_w, self.p.k_h))
        return stack_inp

    def forward(self, sub, rel, neg_ents=None):
        sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.hidden_drop, self.feature_drop)
        stk_inp = self.concat(sub_emb, rel_emb)
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.mm(x, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)

        score = torch.sigmoid(x)
        return score


class RagatInteractE(RagatBase):
    def __init__(self, edge_index, edge_type, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
        # self.ent_embed = torch.nn.Embedding(self.p.num_ent, self.p.embed_dim, padding_idx=None)
        # xavier_normal_(self.ent_embed.weight)
        # self.rel_embed = torch.nn.Embedding(self.p.num_rel * 2, self.p.embed_dim, padding_idx=None)
        # xavier_normal_(self.rel_embed.weight)

        self.inp_drop = torch.nn.Dropout(self.p.iinp_drop)
        self.feature_map_drop = torch.nn.Dropout2d(self.p.ifeat_drop)
        self.hidden_drop = torch.nn.Dropout(self.p.ihid_drop)

        self.hidden_drop_gcn = torch.nn.Dropout(0)

        self.bn0 = torch.nn.BatchNorm2d(self.p.iperm)

        flat_sz_h = self.p.ik_h
        flat_sz_w = 2 * self.p.ik_w
        self.padding = 0

        self.bn1 = torch.nn.BatchNorm2d(self.p.inum_filt * self.p.iperm)
        self.flat_sz = flat_sz_h * flat_sz_w * self.p.inum_filt * self.p.iperm

        self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim)
        self.fc = torch.nn.Linear(self.flat_sz, self.p.embed_dim)
        self.chequer_perm = self.get_chequer_perm()

        self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))
        self.register_parameter('conv_filt',
                                Parameter(torch.zeros(self.p.inum_filt, 1, self.p.iker_sz, self.p.iker_sz)))
        xavier_normal_(self.conv_filt)

    def circular_padding_chw(self, batch, padding):
        upper_pad = batch[..., -padding:, :]
        lower_pad = batch[..., :padding, :]
        temp = torch.cat([upper_pad, batch, lower_pad], dim=2)

        left_pad = temp[..., -padding:]
        right_pad = temp[..., :padding]
        padded = torch.cat([left_pad, temp, right_pad], dim=3)
        return padded

    def forward(self, sub, rel, neg_ents=None):
        sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.inp_drop, self.hidden_drop_gcn)
        comb_emb = torch.cat([sub_emb, rel_emb], dim=1)
        chequer_perm = comb_emb[:, self.chequer_perm]
        stack_inp = chequer_perm.reshape((-1, self.p.iperm, 2 * self.p.ik_w, self.p.ik_h))
        stack_inp = self.bn0(stack_inp)
        x = stack_inp
        x = self.circular_padding_chw(x, self.p.iker_sz // 2)
        x = F.conv2d(x, self.conv_filt.repeat(self.p.iperm, 1, 1, 1), padding=self.padding, groups=self.p.iperm)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        if self.p.strategy == 'one_to_n' or neg_ents is None:
            x = torch.mm(x, all_ent.transpose(1, 0))
            x += self.bias.expand_as(x)
        else:
            x = torch.mul(x.unsqueeze(1), all_ent[neg_ents]).sum(dim=-1)
            x += self.bias[neg_ents]

        pred = torch.sigmoid(x)

        return pred

    def get_chequer_perm(self):
        """
        Function to generate the chequer permutation required for InteractE model

        Parameters
        ----------

        Returns
        -------

        """
        ent_perm = np.int32([np.random.permutation(self.p.embed_dim) for _ in range(self.p.iperm)])
        rel_perm = np.int32([np.random.permutation(self.p.embed_dim) for _ in range(self.p.iperm)])

        comb_idx = []
        for k in range(self.p.iperm):
            temp = []
            ent_idx, rel_idx = 0, 0

            for i in range(self.p.ik_h):
                for j in range(self.p.ik_w):
                    if k % 2 == 0:
                        if i % 2 == 0:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx] + self.p.embed_dim)
                            rel_idx += 1
                        else:
                            temp.append(rel_perm[k, rel_idx] + self.p.embed_dim)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                    else:
                        if i % 2 == 0:
                            temp.append(rel_perm[k, rel_idx] + self.p.embed_dim)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                        else:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx] + self.p.embed_dim)
                            rel_idx += 1

            comb_idx.append(temp)

        chequer_perm = torch.LongTensor(np.int32(comb_idx)).to(self.device)
        return chequer_perm
