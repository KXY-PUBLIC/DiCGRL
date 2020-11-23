import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.autograd import Variable
import numpy as np

CUDA = torch.cuda.is_available()  # checking cuda availability


class TransE(nn.Module):
    def __init__(self, entity_emb, relation_emb, config=None):
        '''
        entity_in_dim -> Entity Input Embedding dimensions
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        '''

        super(TransE, self).__init__()
        self.do_normalize = config.do_normalize
        self.l1 = config.l1
        self.valid_invalid_ratio = config.valid_invalid_ratio
        self.margin = config.margin

        self.entity_embeddings = nn.Parameter(entity_emb)
        self.relation_embeddings = nn.Parameter(relation_emb)

        self.num_nodes = entity_emb.shape[0]
        self.entity_in_dim = entity_emb.shape[1]

        # Properties of Relations
        self.num_relation = relation_emb.shape[0]
        self.relation_dim = relation_emb.shape[1]

        # loss function
        self.loss = nn.MarginRankingLoss(margin=self.margin, reduction='none')

    def forward(self, batch_inputs, batch_labels=None, batch_loss_weight=None):
        if self.do_normalize:
            self.entity_embeddings.data = F.normalize(
                self.entity_embeddings.data, p=2, dim=1).detach()

        len_pos_triples = int(
            batch_inputs.size(0) / (int(self.valid_invalid_ratio) + 1))
        pos_triples = batch_inputs[:len_pos_triples]
        neg_triples = batch_inputs[len_pos_triples:]
        #print("batch", batch_inputs.size(), self.valid_invalid_ratio, len_pos_triples, neg_triples.size())

        pos_triples = pos_triples.repeat(int(self.valid_invalid_ratio), 1)

        pos_head = self.entity_embeddings[pos_triples[:, 0], :]
        pos_rel = self.relation_embeddings[pos_triples[:, 1], :]
        pos_tail = self.entity_embeddings[pos_triples[:, 2], :]

        neg_head = self.entity_embeddings[neg_triples[:, 0], :]
        neg_rel = self.relation_embeddings[neg_triples[:, 1], :]
        neg_tail = self.entity_embeddings[neg_triples[:, 2], :]

        pos_x = pos_head + pos_rel - pos_tail
        neg_x = neg_head + neg_rel - neg_tail
        if self.l1:
            pos_norm = torch.norm(pos_x, p=1, dim=1)
            neg_norm = torch.norm(neg_x, p=1, dim=1)
        else:
            pos_norm = torch.norm(pos_x, p=2, dim=1)
            neg_norm = torch.norm(neg_x, p=2, dim=1)

        y = -torch.ones(int(self.valid_invalid_ratio) * len_pos_triples).cuda()
        output = (pos_norm, neg_norm, y)

        if batch_labels is not None:
            sep_loss = self.loss(pos_norm, neg_norm, y)
            if batch_loss_weight is not None:
                loss = torch.mean(sep_loss * batch_loss_weight.view(-1))
            else:
                loss = torch.mean(sep_loss)
            return loss, 0

        return output, 0

    def test(self, batch_inputs):
        head = self.entity_embeddings[batch_inputs[:, 0], :]
        rel = self.relation_embeddings[batch_inputs[:, 1], :]
        tail = self.entity_embeddings[batch_inputs[:, 2], :]

        x = head + rel - tail
        if self.l1:
            score = torch.norm(x, p=1, dim=1)
        else:
            score = torch.norm(x, p=2, dim=1)

        y = -torch.ones(int(batch_inputs.size(0))).cuda()
        score = score * y

        return score, 0


class ConvKB(nn.Module):
    def __init__(self, entity_emb, relation_emb, config = None):
        '''
        entity_in_dim -> Entity Input Embedding dimensions
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        '''

        super(ConvKB, self).__init__()
        self.do_normalize = config.do_normalize

        self.entity_embeddings = nn.Parameter(entity_emb)
        self.relation_embeddings = nn.Parameter(relation_emb)

        self.num_nodes = entity_emb.shape[0]
        self.entity_in_dim = entity_emb.shape[1]

        # Properties of Relations
        self.num_relation = relation_emb.shape[0]
        self.relation_dim = relation_emb.shape[1]

        self.conv_layer = nn.Conv2d(1, config.out_channels, (1, 3))  # kernel size -> 1*input_seq_length(i.e. 2)
        self.dropout = nn.Dropout(config.dropout)
        self.non_linearity = nn.ReLU()
        self.fc_layer = nn.Linear((self.entity_in_dim) * config.out_channels, 1)

        nn.init.xavier_uniform_(self.fc_layer.weight, gain=1.414)
        nn.init.xavier_uniform_(self.conv_layer.weight, gain=1.414)

        # loss function
        self.loss = torch.nn.SoftMarginLoss(reduction='none')


    def forward(self, batch_inputs, batch_labels=None, batch_loss_weight=None):
        if self.do_normalize:
            self.entity_embeddings.data = F.normalize(
                self.entity_embeddings.data, p=2, dim=1).detach()

        conv_input = torch.cat((self.entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.relation_embeddings[
            batch_inputs[:, 1]].unsqueeze(1), self.entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)), dim=1)

        batch_size, length, dim = conv_input.size()
        # assuming inputs are of the form ->
        conv_input = conv_input.transpose(1, 2)
        # batch * length(which is 3 here -> entity,relation,entity) * dim
        # To make tensor of size 4, where second dim is for input channels
        conv_input = conv_input.unsqueeze(1)

        out_conv = self.dropout(
            self.non_linearity(self.conv_layer(conv_input)))

        input_fc = out_conv.squeeze(-1).view(batch_size, -1)
        output = self.fc_layer(input_fc)

        if batch_labels is not None:
            sep_loss = self.loss(output.view(-1), batch_labels.view(-1))
            if batch_loss_weight is not None:
                loss = torch.mean(sep_loss * batch_loss_weight.view(-1))
            else:
                loss = torch.mean(sep_loss)
            return loss, 0

        return output, 0

    def test(self, batch_inputs):
        if self.do_normalize:
            self.entity_embeddings.data = F.normalize(
                self.entity_embeddings.data, p=2, dim=1).detach()

        conv_input = torch.cat((self.entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.relation_embeddings[
            batch_inputs[:, 1]].unsqueeze(1), self.entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)), dim=1)

        batch_size, length, dim = conv_input.size()
        conv_input = conv_input.transpose(1, 2)
        conv_input = conv_input.unsqueeze(1)

        out_conv = self.dropout(
            self.non_linearity(self.conv_layer(conv_input)))

        input_fc = out_conv.squeeze(-1).view(batch_size, -1)
        output = self.fc_layer(input_fc)

        return output, 0

class Rel_attention(nn.Module):
    def __init__(self, num_rel, K):
        super(Rel_attention, self).__init__()
        self.rel_attention = nn.Parameter(torch.zeros((num_rel, K)))
        nn.init.xavier_normal_(self.rel_attention.data, gain=1.414)

    def forward(self, batch_relation):
        return self.rel_attention[batch_relation, :]

class Param_a(nn.Module):
    def __init__(self, K, emb_s, cnt):
        super(Param_a, self).__init__()
        self.a_k = nn.Parameter(torch.zeros(size=(K, cnt * emb_s * K)))
        nn.init.xavier_normal_(self.a_k.data, gain=1.414)

    def forward(self, x):
        return (self.a_k.mm(x.t())).t()

class TransE_2(nn.Module):
    def __init__(self, entity_emb, relation_emb, config=None):

        super(TransE_2, self).__init__()
        self.l1 = config.l1
        self.valid_invalid_ratio = config.valid_invalid_ratio
        self.K = config.k_factors
        self.top_n = config.top_n

        self.entity_embeddings = nn.Parameter(entity_emb)
        self.relation_embeddings = nn.Parameter(relation_emb)

        self.emb_s = int(entity_emb.shape[1] / self.K)
        self.num_rel = relation_emb.shape[0]
        self.rel_attention = Rel_attention(self.num_rel, self.K)
        self.softmax = torch.nn.Softmax(dim=1)

        # loss function
        self.margin = config.margin
        self.loss = nn.MarginRankingLoss(margin=self.margin, reduction='none')

    def forward(self, batch_inputs, batch_labels=None, batch_loss_weight=None, batch_top_att=None, batch_top_indices=None):
        len_pos_triples = int(
            batch_inputs.size(0) / (int(self.valid_invalid_ratio) + 1))

        # [b_s, k, emb_s]
        head_o = self.entity_embeddings[batch_inputs[:, 0], :]
        tail_o = self.entity_embeddings[batch_inputs[:, 2], :]
        rel = self.relation_embeddings[batch_inputs[:, 1], :]
        head_ori = head_o.view(-1, self.K, self.emb_s)
        tail_ori = tail_o.view(-1, self.K, self.emb_s)

        # calculate k attention
        if batch_top_att is not None:
            top_indices_in = batch_top_indices
            top_att = batch_top_att
        else:
            tmp = self.rel_attention(batch_inputs[:, 1])
            att = self.softmax(tmp)

            # choose top n
            sorted_att, sorted_indices_in = torch.sort(att, dim=-1, descending=True)
            top_indices_in = sorted_indices_in[:, :self.top_n]
            top_att = sorted_att[:, :self.top_n]
            #print("top_indices_in", top_indices_in.size())

        head_ori = head_ori.gather(1, top_indices_in.unsqueeze(-1).expand(-1, self.top_n, self.emb_s))
        tail_ori = tail_ori.gather(1, top_indices_in.unsqueeze(-1).expand(-1, self.top_n, self.emb_s))

        head = head_ori.view(-1, self.top_n * self.emb_s)
        tail = tail_ori.view(-1, self.top_n * self.emb_s)

        pos_head = head[:len_pos_triples].repeat(int(self.valid_invalid_ratio), 1)
        pos_tail = tail[:len_pos_triples].repeat(int(self.valid_invalid_ratio), 1)
        pos_rel = rel[:len_pos_triples].repeat(int(self.valid_invalid_ratio), 1)
        neg_head = head[len_pos_triples:]
        neg_tail = tail[len_pos_triples:]
        neg_rel = rel[len_pos_triples:]
        # print("pos_head", pos_head.size(), neg_head.size())

        pos_x = pos_head + pos_rel - pos_tail
        neg_x = neg_head + neg_rel - neg_tail
        if self.l1:
            pos_norm = torch.norm(pos_x, p=1, dim=1)
            neg_norm = torch.norm(neg_x, p=1, dim=1)
        else:
            pos_norm = torch.norm(pos_x, p=2, dim=1)
            neg_norm = torch.norm(neg_x, p=2, dim=1)

        y = -torch.ones(int(self.valid_invalid_ratio) * len_pos_triples).cuda()
        output = (pos_norm, neg_norm, y)

        att_loss = 0.0
        if batch_top_att is None:
            top_num_att = torch.sum(top_att, 1)
            y2 = torch.ones(int(batch_inputs.size(0))).cuda()
            att_loss = torch.mean(y2 - top_num_att)

        if batch_labels is not None:
            sep_loss = self.loss(pos_norm, neg_norm, y)
            if batch_loss_weight is not None:
                batch_loss_weight = batch_loss_weight[len_pos_triples:]
                loss = torch.mean(sep_loss * batch_loss_weight.view(-1))
            else:
                loss = torch.mean(sep_loss)
            return loss, att_loss
        return output, att_loss

    def test(self, batch_inputs, only_att=False):
        head_o = self.entity_embeddings[batch_inputs[:, 0], :]
        tail_o = self.entity_embeddings[batch_inputs[:, 2], :]
        rel = self.relation_embeddings[batch_inputs[:, 1], :]
        head_ori = head_o.view(-1, self.K, self.emb_s)
        tail_ori = tail_o.view(-1, self.K, self.emb_s)

        # calculate k attention
        tmp = self.rel_attention(batch_inputs[:, 1])  # [b_s, k]
        att = self.softmax(tmp)

        # choose top n
        sorted_att, sorted_indices_in = torch.sort(att, dim=-1, descending=True)
        top_indices_in = sorted_indices_in[:, :self.top_n]
        top_att = sorted_att[:, :self.top_n]
        if only_att:
            return top_att, top_indices_in

        head_ori = head_ori.gather(1, top_indices_in.unsqueeze(-1).expand(-1, self.top_n, self.emb_s))
        tail_ori = tail_ori.gather(1, top_indices_in.unsqueeze(-1).expand(-1, self.top_n, self.emb_s))

        head = head_ori.view(-1, self.top_n * self.emb_s)
        tail = tail_ori.view(-1, self.top_n * self.emb_s)


        x = head + rel - tail
        if self.l1:
            score = torch.norm(x, p=1, dim=1)
        else:
            score = torch.norm(x, p=2, dim=1)

        y = -torch.ones(int(batch_inputs.size(0))).cuda()
        score = score * y

        return score, att


class ConvKB_2(nn.Module):
    def __init__(self, entity_emb, relation_emb, config=None):
        super(ConvKB_2, self).__init__()
        self.K = config.k_factors
        self.top_n = config.top_n

        self.entity_embeddings = nn.Parameter(entity_emb)
        self.relation_embeddings = nn.Parameter(relation_emb)

        self.emb_s = int(entity_emb.shape[1] / self.K)
        self.num_rel = relation_emb.shape[0]
        self.rel_attention = Rel_attention(self.num_rel, self.K)
        self.softmax = torch.nn.Softmax(dim=1)

        self.conv_layer = nn.Conv2d(1, config.out_channels, (1, 3))  # kernel size -> 1*input_seq_length(i.e. 2)
        self.dropout = nn.Dropout(config.dropout)
        self.non_linearity = nn.ReLU()
        self.fc_layer = nn.Linear(self.emb_s * self.top_n * config.out_channels, 1)

        nn.init.xavier_uniform_(self.fc_layer.weight, gain=1.414)
        nn.init.xavier_uniform_(self.conv_layer.weight, gain=1.414)

        # loss function
        self.loss = torch.nn.SoftMarginLoss(reduction='none')

    def forward(self, batch_inputs, batch_labels=None, batch_loss_weight=None, batch_top_att=None, batch_top_indices=None):
        head_o = self.entity_embeddings[batch_inputs[:, 0], :]
        tail_o = self.entity_embeddings[batch_inputs[:, 2], :]
        rel_embedded = self.relation_embeddings[batch_inputs[:, 1], :]
        head_ori = head_o.view(-1, self.K, self.emb_s)
        tail_ori = tail_o.view(-1, self.K, self.emb_s)

        # calculate k attention
        if batch_top_att is not None:
            top_indices_in = batch_top_indices
            top_att = batch_top_att
        else:
            tmp = self.rel_attention(batch_inputs[:, 1])
            att = self.softmax(tmp)

            # choose top n
            sorted_att, sorted_indices_in = torch.sort(att, dim=-1, descending=True)
            top_indices_in = sorted_indices_in[:, :self.top_n]
            top_att = sorted_att[:, :self.top_n]

        head_ori = head_ori.gather(1, top_indices_in.unsqueeze(-1).expand(-1, self.top_n, self.emb_s))
        tail_ori = tail_ori.gather(1, top_indices_in.unsqueeze(-1).expand(-1, self.top_n, self.emb_s))

        head_embedded = head_ori.view(-1, self.top_n * self.emb_s)
        tail_embedded = tail_ori.view(-1, self.top_n * self.emb_s)

        conv_input = torch.cat((head_embedded.unsqueeze(1), rel_embedded.unsqueeze(1), tail_embedded.unsqueeze(1)), dim=1)

        batch_size, length, dim = conv_input.size()
        conv_input = conv_input.transpose(1, 2)
        conv_input = conv_input.unsqueeze(1)

        out_conv = self.dropout(self.non_linearity(self.conv_layer(conv_input)))

        input_fc = out_conv.squeeze(-1).view(batch_size, -1)
        output = self.fc_layer(input_fc)

        att_loss = 0.0
        if batch_top_att is None:
            top_num_att = torch.sum(top_att, 1)
            y2 = torch.ones(int(batch_inputs.size(0))).cuda()
            att_loss = torch.mean(y2 - top_num_att)

        if batch_labels is not None:
            sep_loss = self.loss(output.view(-1), batch_labels.view(-1))
            if batch_loss_weight is not None:
                loss = torch.mean(sep_loss * batch_loss_weight.view(-1))
            else:
                loss = torch.mean(sep_loss)
            return loss, att_loss

        return output, att_loss

    def test(self, batch_inputs, only_att=False):
        head_o = self.entity_embeddings[batch_inputs[:, 0], :]
        tail_o = self.entity_embeddings[batch_inputs[:, 2], :]
        rel_embedded = self.relation_embeddings[batch_inputs[:, 1], :]
        head_ori = head_o.view(-1, self.K, self.emb_s)
        tail_ori = tail_o.view(-1, self.K, self.emb_s)

        # calculate k attention
        tmp = self.rel_attention(batch_inputs[:, 1])  # [b_s, k]
        att = self.softmax(tmp)

        # choose top n
        sorted_att, sorted_indices_in = torch.sort(att, dim=-1, descending=True)
        top_indices_in = sorted_indices_in[:, :self.top_n]
        top_att = sorted_att[:, :self.top_n]

        if only_att:
            return top_att, top_indices_in

        head_ori = head_ori.gather(1, top_indices_in.unsqueeze(-1).expand(-1, self.top_n, self.emb_s))
        tail_ori = tail_ori.gather(1, top_indices_in.unsqueeze(-1).expand(-1, self.top_n, self.emb_s))

        head_embedded = head_ori.view(-1, self.top_n * self.emb_s)
        tail_embedded = tail_ori.view(-1, self.top_n * self.emb_s)

        conv_input = torch.cat((head_embedded.unsqueeze(1), rel_embedded.unsqueeze(1), tail_embedded.unsqueeze(1)),
                               dim=1)

        batch_size, length, dim = conv_input.size()
        conv_input = conv_input.transpose(1, 2)
        conv_input = conv_input.unsqueeze(1)

        out_conv = self.dropout(self.non_linearity(self.conv_layer(conv_input)))

        input_fc = out_conv.squeeze(-1).view(batch_size, -1)
        output = self.fc_layer(input_fc)

        return output, att
