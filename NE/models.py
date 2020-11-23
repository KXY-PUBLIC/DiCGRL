import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers import GraphAttentionLayer, SpGraphAttentionLayer, SpGraphAttentionLayer_2


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj, only_edge=False, ori_sep_edge=None):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1), 0


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj, only_edge=False, ori_sep_edge=None):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1), 0


class Param_W(nn.Module):
    def __init__(self, nfeat, nhid):
        super(Param_W, self).__init__()
        self.W_k = nn.Parameter(torch.zeros(size=(nfeat, nhid)))
        nn.init.xavier_normal_(self.W_k.data, gain=1.414)

    def forward(self, x):
        return torch.mm(x, self.W_k)

class Param_a(nn.Module):
    def __init__(self, K, nhid):
        super(Param_a, self).__init__()
        self.a_k = nn.Parameter(torch.zeros(size=(K, 2 * nhid * K)))
        nn.init.xavier_normal_(self.a_k.data, gain=1.414)

    def forward(self, x):
        return self.a_k.mm(x)


class SpGAT_2(nn.Module):
    def __init__(self, nfeat, nclass, config=None):
        """Sparse version of GAT."""
        super(SpGAT_2, self).__init__()
        self.nfeat = nfeat
        self.nhid = config.hidden
        self.nheads = config.nb_heads
        self.nclass = nclass
        self.alpha = config.alpha
        self.dropout = config.dropout
        self.K = config.k_factors
        self.top_n = config.top_n
        self.use_cuda = config.use_cuda
        print("nfeat: ", nfeat, "nclass: ", nclass)

        self.W_ks = nn.ModuleList()
        for k in range(self.K):
            W_k = Param_W(self.nfeat, self.nhid)
            self.W_ks += [W_k]
        self.a_k = Param_a(self.K, self.nhid)
        self.W_o = nn.Parameter(torch.zeros(size=(self.nhid * self.K, self.nclass)))

        nn.init.xavier_normal_(self.W_o.data, gain=1.414)

        self.softmax = torch.nn.Softmax(dim=-1)

        self.attentions = [SpGraphAttentionLayer_2(self.nfeat,
                                                   self.nhid,
                                                 dropout=self.dropout,
                                                 alpha=self.alpha,
                                                 concat=True) for _ in range(self.K)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer_2(self.nhid,
                                               self.nclass,
                                             dropout=self.dropout,
                                             alpha=self.alpha,
                                             concat=False)

    def forward(self, x, adj, only_edge=False, ori_sep_edge=None):
        x = F.dropout(x, self.dropout, training=self.training)

        # get k embedding
        h_k = [W_k(x) for W_k in self.W_ks]
        h = torch.cat(h_k, dim=1)
        edge = adj.nonzero().t()  # (2, edge')

        # get edge's k attention
        if ori_sep_edge is None:
            edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()  # (2 * k * nhid, edge')
            edge_att = self.softmax(self.a_k(edge_h).t())  # (edge', k)
            #print("edge_att", edge_att.size(), edge_att[0:5])

            # choose top n
            sorted_att, sorted_indices_in = torch.sort(edge_att, dim=-1, descending=True)
            top_indices_in = sorted_indices_in[:, :self.top_n]
            top_att = sorted_att[:, :self.top_n]   # (edge', top_n)
            #print("top_indices_in", top_indices_in.size(), top_indices_in[0:5], top_att[0:5])

            # choose separate edge
            sep_edge = []
            for j in range(self.K):
                for i in range(self.top_n):
                    split_edge_ind = top_indices_in[:, i]  # (edge')
                    #print("split_edge_ind", i, split_edge_ind.size(), split_edge_ind[:5])

                    tmp = torch.eq(split_edge_ind, j)
                    sep_idx = torch.nonzero(tmp).squeeze(-1)
                    tmp_edge = edge[:, sep_idx]
                    if i == 0:
                        k_edge = tmp_edge
                    else:
                        k_edge = torch.cat((k_edge, tmp_edge), dim=1)
                #print("k_edge", k_edge.size(), k_edge[:5])
                sep_edge.append(k_edge)

            if only_edge:
                return sep_edge
        else:
            sep_edge = ori_sep_edge

        # get k output
        x = []
        for k in range(self.K):
            k_out = self.attentions[k](h_k[k], sep_edge[k])  # (node, nhid)
            x.append(k_out)

        x = torch.cat(x, dim=1)   # (node, k * nhid)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.mm(x, self.W_o)  # (node, nclass)
        x = F.elu(self.out_att(x, edge))

        att_loss = 0.0
        if ori_sep_edge is None:
            top_num_att = torch.sum(top_att, 1)
            y2 = torch.ones(int(top_num_att.size(0)))
            if self.use_cuda:
                y2 = y2.cuda()
            att_loss = torch.mean(y2 - top_num_att)

        return F.log_softmax(x, dim=1), att_loss
