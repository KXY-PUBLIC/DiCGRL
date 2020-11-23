import torch
import numpy as np
from collections import defaultdict
import time
import copy
from torch.autograd import Variable


class Corpus:
    def __init__(self, args, train_data, validation_data, test_data, entity2id, relation2id,
                 batch_size, valid_to_invalid_samples_ratio, valid_triples_list, valid_train_triples_list):
        self.train_triples = train_data[0]
        self.first_nei = train_data[1]
        self.second_nei = train_data[2]
        self.all_neighbor = []
        self.ratio = args.ratio
        self.loss_weight_list = []
        self.final_nei = []
        self.nei_top_att_list = []
        self.nei_top_idx_list = []
        self.top_n = args.top_n

        self.validation_triples = validation_data
        self.test_triples = test_data

        self.entity2id = entity2id
        self.sub_entity = list(self.entity2id.values())
        self.id2entity = {v: k for k, v in self.entity2id.items()}
        self.relation2id = relation2id
        self.id2relation = {v: k for k, v in self.relation2id.items()}
        self.batch_size = batch_size
        # ratio of valid to invalid samples per batch
        self.invalid_valid_ratio = int(valid_to_invalid_samples_ratio)

        self.train_indices = np.array(list(self.train_triples)).astype(np.int32)
        self.first_nei_indices = np.array(list(self.first_nei)).astype(np.int32)
        self.second_nei_indices = np.array(list(self.second_nei)).astype(np.int32)
        self.loss_weight = np.array(list(self.loss_weight_list)).astype(np.float32)
        self.final_nei_indices = np.array(list(self.final_nei)).astype(np.int32)
        self.nei_top_att = np.array(list(self.nei_top_att_list)).astype(np.float32)
        self.nei_top_idx = np.array(list(self.nei_top_idx_list)).astype(np.int32)
        self.final_nei_values = np.array([[1]] * len(self.final_nei)).astype(np.float32)
        # These are valid triples, hence all have value 1
        self.train_values = np.array([[1]] * len(self.train_triples)).astype(np.float32)

        self.validation_indices = np.array(list(self.validation_triples)).astype(np.int32)
        self.validation_values = np.array([[1]] * len(self.validation_triples)).astype(np.float32)

        self.test_indices = np.array(list(self.test_triples)).astype(np.int32)
        self.test_values = np.array([[1]] * len(self.test_triples)).astype(np.float32)

        self.valid_triples_set = set(valid_triples_list)
        self.valid_train_triples_set = set(valid_train_triples_list)
        print("training triples {}, validation_triples {}, test_triples {}".format(len(self.train_indices), len(self.validation_indices), len(self.test_indices)))
        # For training purpose
        self.batch_triples = np.empty(
            (self.batch_size * (self.invalid_valid_ratio + 1), 3)).astype(np.int32)
        self.batch_labels = np.empty(
            (self.batch_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)

    def get_iteration_batch(self, iter_num, is_nei=False, seed=42, is_emr=False):
        np.random.seed(seed)
        if is_nei or is_emr:
            train_data = self.final_nei_indices
            train_label = self.final_nei_values
        else:
            train_data = self.train_indices
            train_label = self.train_values

        self.tmp_size = self.batch_size
        if (iter_num + 1) * self.batch_size > len(train_data):
            last_iter_size = len(train_data) - self.batch_size * iter_num
            self.tmp_size = last_iter_size

        self.batch_triples = np.empty((self.tmp_size * (self.invalid_valid_ratio + 1), 3)).astype(np.int32)
        self.batch_labels = np.empty((self.tmp_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)
        if is_nei:
            self.batch_loss_weight = np.empty((self.tmp_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)
            self.batch_top_att = np.empty((self.tmp_size * (self.invalid_valid_ratio + 1), self.top_n)).astype(np.float32)
            self.batch_top_idx = np.empty((self.tmp_size * (self.invalid_valid_ratio + 1), self.top_n)).astype(np.int32)

        if (iter_num + 1) * self.batch_size <= len(train_data):
            indices = range(self.batch_size * iter_num, self.batch_size * (iter_num + 1))
        else:
            indices = range(self.batch_size * iter_num, len(train_data))

        self.batch_triples[:self.tmp_size, :] = train_data[indices, :]
        self.batch_labels[:self.tmp_size, :] = train_label[indices, :]
        if is_nei:
            self.batch_loss_weight[:self.tmp_size, :] = self.loss_weight[indices, :]
            self.batch_top_att[:self.tmp_size, :] = self.nei_top_att[indices, :]
            self.batch_top_idx[:self.tmp_size, :] = self.nei_top_idx[indices, :]

        last_index = self.tmp_size

        if self.invalid_valid_ratio > 0:
            #random_entities = np.random.randint(0, len(self.entity2id), last_index * self.invalid_valid_ratio)
            random_entities = np.random.choice(self.sub_entity, size=last_index * self.invalid_valid_ratio)

            # Precopying the same valid indices from 0 to batch_size to rest
            # of the indices
            self.batch_triples[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                self.batch_triples[:last_index, :], (self.invalid_valid_ratio, 1))
            self.batch_labels[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                self.batch_labels[:last_index, :], (self.invalid_valid_ratio, 1))
            if is_nei:
                self.batch_loss_weight[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_loss_weight[:last_index, :], (self.invalid_valid_ratio, 1))
                self.batch_top_att[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_top_att[:last_index, :], (self.invalid_valid_ratio, 1))
                self.batch_top_idx[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_top_idx[:last_index, :], (self.invalid_valid_ratio, 1))

            for i in range(last_index):
                for j in range(self.invalid_valid_ratio // 2):
                    current_index = i * (self.invalid_valid_ratio // 2) + j

                    while (random_entities[current_index], self.batch_triples[last_index + current_index, 1],
                           self.batch_triples[last_index + current_index, 2]) in self.valid_train_triples_set:
                        #random_entities[current_index] = np.random.randint(0, len(self.entity2id))
                        random_entities[current_index] = np.random.choice(self.sub_entity, size=1)
                    self.batch_triples[last_index + current_index,
                                       0] = random_entities[current_index]
                    self.batch_labels[last_index + current_index, :] = [-1]

                for j in range(self.invalid_valid_ratio // 2):
                    current_index = last_index * (self.invalid_valid_ratio // 2) + (i * (self.invalid_valid_ratio // 2) + j)

                    while (self.batch_triples[last_index + current_index, 0], self.batch_triples[last_index + current_index, 1],
                           random_entities[current_index]) in self.valid_train_triples_set:
                        #random_entities[current_index] = np.random.randint(0, len(self.entity2id))
                        random_entities[current_index] = np.random.choice(self.sub_entity, size=1)
                    self.batch_triples[last_index + current_index,
                                       2] = random_entities[current_index]
                    self.batch_labels[last_index + current_index, :] = [-1]

            if is_nei:
                return self.batch_triples, self.batch_labels, self.batch_loss_weight, self.batch_top_att, self.batch_top_idx
            else:
                return self.batch_triples, self.batch_labels

        if is_nei:
            return self.batch_triples, self.batch_labels, self.batch_loss_weight, self.batch_top_att, self.batch_top_idx
        else:
            return self.batch_triples, self.batch_labels

    def get_top_att(self, args, model, data):
        final_top_att = []
        final_top_indices = []
        batch_size = args.batch_size * (self.invalid_valid_ratio + 1)

        if len(data) % batch_size == 0:
            num_iters = len(data) // batch_size
        else:
            num_iters = (len(data) // batch_size) + 1

        for iter_num in range(num_iters):
            if (iter_num + 1) * batch_size <= len(data):
                indices = range(batch_size * iter_num, batch_size * (iter_num + 1))
            else:
                indices = range(batch_size * iter_num, len(data))

            batch_triples_ori = data[indices, :]
            batch_triples = Variable(torch.LongTensor(batch_triples_ori)).cuda()
            top_att, top_indices_in = model(batch_triples, only_att=True)
            top_att = top_att.detach().cpu().numpy().tolist()
            top_indices_in = top_indices_in.detach().cpu().numpy().tolist()
            final_top_att.extend(top_att)
            final_top_indices.extend(top_indices_in)

        return final_top_att, final_top_indices

    def get_ent_att_idx(self, args, model):
        dic = defaultdict(set)
        start_time = time.time()
        top_att, top_indices = self.get_top_att(args, model, self.train_indices)

        for idx, (h, r, t) in enumerate(self.train_indices):
            dic[h].update([x for x in top_indices[idx]])
            dic[t].update([x for x in top_indices[idx]])

        print("dic's length", len(dic))
        print("get new triple's attention time {}s".format(time.time() - start_time))

        return dic

    def get_final_nei(self, args, model, ent_indices_dic, first_top_att=None, first_top_indices=None, do_ratio_same=True):
        self.final_nei = []
        self.nei_top_att_list = []
        self.nei_top_idx_list = []
        self.loss_weight_list = []
        ent_indices_dic_new = copy.deepcopy(ent_indices_dic)
        start_time = time.time()
        if first_top_att is None:
            first_top_att, first_top_indices = self.get_top_att(args, model, self.first_nei_indices)

        for idx, (h, r, t) in enumerate(self.first_nei_indices):
            top_indices = [x for x in first_top_indices[idx]]
            ent_indices_dic_new[h].update(top_indices)
            ent_indices_dic_new[t].update(top_indices)
            ent_indices = set()
            if h in ent_indices_dic:
                ent_indices.update(list(ent_indices_dic[h]))
            if t in ent_indices_dic:
                ent_indices.update(list(ent_indices_dic[t]))

            flag_1, flag_2 = 0, 0
            for x in top_indices:
                if x in ent_indices:
                    flag_1 = 1
                else:
                    flag_2 = 1
            if flag_1 == 1:
                self.final_nei.append((h, r, t))
                self.nei_top_att_list.append(first_top_att[idx])
                self.nei_top_idx_list.append((first_top_indices[idx]))
                if flag_2 == 0 or do_ratio_same:
                    self.loss_weight_list.append([1.0 * self.ratio])
                else:
                    self.loss_weight_list.append([1.0 * self.ratio * self.ratio])

        print("first_nei, final_first_nei", len(self.first_nei), len(self.final_nei))
        len_first = len(self.final_nei)

        if args.use_second_nei:
            second_top_att, second_top_indices = self.get_top_att(args, model, self.second_nei_indices)
            for idx, (h, r, t) in enumerate(self.second_nei_indices):
                top_indices = [x for x in second_top_indices[idx]]
                ent_indices = set()
                if h in ent_indices_dic_new:
                    ent_indices.update(list(ent_indices_dic_new[h]))
                if t in ent_indices_dic_new:
                    ent_indices.update(list(ent_indices_dic_new[t]))
                flag_1, flag_2 = 0, 0
                for x in top_indices:
                    if x in ent_indices:
                        flag_1 = 1
                    else:
                        flag_2 = 1
                if flag_1 == 1 and (flag_2 == 0 or do_ratio_same):
                    self.final_nei.append((h, r, t))
                    self.nei_top_att_list.append(first_top_att[idx])
                    self.nei_top_idx_list.append((first_top_indices[idx]))
                    self.loss_weight_list.append([1.0 * self.ratio * self.ratio])

            print("second_nei, final_second_nei", len(self.second_nei), len(self.final_nei) - len_first)

        print("get neighbors' attention time {}s".format(time.time() - start_time))

        return first_top_att, first_top_indices


    def get_validation_pred(self, args, model):
        average_hits_at_100_head, average_hits_at_100_tail = [], []
        average_hits_at_ten_head, average_hits_at_ten_tail = [], []
        average_hits_at_three_head, average_hits_at_three_tail = [], []
        average_hits_at_one_head, average_hits_at_one_tail = [], []
        average_mean_rank_head, average_mean_rank_tail = [], []
        average_mean_recip_rank_head, average_mean_recip_rank_tail = [], []

        for iters in range(1):
            start_time = time.time()

            indices = [i for i in range(len(self.test_indices))]
            batch_triples = self.test_indices[indices, :]
            print("test set length ", len(self.test_indices))
            print("candidate entity length ", len(self.sub_entity))
            entity_list = self.sub_entity

            ranks_head, ranks_tail = [], []
            reciprocal_ranks_head, reciprocal_ranks_tail = [], []
            hits_at_100_head, hits_at_100_tail = 0, 0
            hits_at_ten_head, hits_at_ten_tail = 0, 0
            hits_at_three_head, hits_at_three_tail = 0, 0
            hits_at_one_head, hits_at_one_tail = 0, 0

            for i in range(batch_triples.shape[0]):
                #print(len(ranks_head))
                start_time_it = time.time()
                new_x_batch_head = np.tile(
                    batch_triples[i, :], (len(entity_list), 1))
                new_x_batch_tail = np.tile(
                    batch_triples[i, :], (len(entity_list), 1))

                new_x_batch_head[:, 0] = entity_list
                new_x_batch_tail[:, 2] = entity_list

                last_index_head = []  # array of already existing triples
                last_index_tail = []
                for tmp_index in range(len(new_x_batch_head)):
                    temp_triple_head = (new_x_batch_head[tmp_index][0], new_x_batch_head[tmp_index][1],
                                        new_x_batch_head[tmp_index][2])
                    if temp_triple_head in self.valid_triples_set:
                        last_index_head.append(tmp_index)

                    temp_triple_tail = (new_x_batch_tail[tmp_index][0], new_x_batch_tail[tmp_index][1],
                                        new_x_batch_tail[tmp_index][2])
                    if temp_triple_tail in self.valid_triples_set:
                        last_index_tail.append(tmp_index)

                # Deleting already existing triples, leftover triples are invalid, according
                # to train, validation and test data
                # Note, all of them maynot be actually invalid
                new_x_batch_head = np.delete(
                    new_x_batch_head, last_index_head, axis=0)
                new_x_batch_tail = np.delete(
                    new_x_batch_tail, last_index_tail, axis=0)

                # adding the current valid triples to the top, i.e, index 0
                new_x_batch_head = np.insert(
                    new_x_batch_head, 0, batch_triples[i], axis=0)
                new_x_batch_tail = np.insert(
                    new_x_batch_tail, 0, batch_triples[i], axis=0)

                import math
                # Have to do this, because it doesn't fit in memory

                if 'WN' in args.dataset:
                    num_triples_each_shot = int(
                        math.ceil(new_x_batch_head.shape[0] / 4))

                    scores1_head, _ = model(torch.LongTensor(
                        new_x_batch_head[:num_triples_each_shot, :]).cuda())
                    scores2_head, _ = model(torch.LongTensor(
                        new_x_batch_head[num_triples_each_shot: 2 * num_triples_each_shot, :]).cuda())
                    scores3_head, _ = model(torch.LongTensor(
                        new_x_batch_head[2 * num_triples_each_shot: 3 * num_triples_each_shot, :]).cuda())
                    scores4_head, _ = model(torch.LongTensor(
                        new_x_batch_head[3 * num_triples_each_shot: 4 * num_triples_each_shot, :]).cuda())

                    scores_head = torch.cat(
                        [scores1_head, scores2_head, scores3_head, scores4_head], dim=0)
                else:
                    new_x_batch_head = Variable(torch.LongTensor(new_x_batch_head)).cuda()
                    scores_head, _ = model(new_x_batch_head)

                sorted_scores_head, sorted_indices_head = torch.sort(
                    scores_head.view(-1), dim=-1, descending=True)
                # Just search for zeroth index in the sorted scores, we appended valid triple at top
                ranks_head.append(
                    np.where(sorted_indices_head.cpu().numpy() == 0)[0][0] + 1)
                reciprocal_ranks_head.append(1.0 / ranks_head[-1])

                # Tail part here

                if 'WN' in args.dataset:
                    num_triples_each_shot = int(
                        math.ceil(new_x_batch_tail.shape[0] / 4))

                    scores1_tail, _ = model(torch.LongTensor(
                        new_x_batch_tail[:num_triples_each_shot, :]).cuda())
                    scores2_tail, _ = model(torch.LongTensor(
                        new_x_batch_tail[num_triples_each_shot: 2 * num_triples_each_shot, :]).cuda())
                    scores3_tail, _ = model(torch.LongTensor(
                        new_x_batch_tail[2 * num_triples_each_shot: 3 * num_triples_each_shot, :]).cuda())
                    scores4_tail, _ = model(torch.LongTensor(
                        new_x_batch_tail[3 * num_triples_each_shot: 4 * num_triples_each_shot, :]).cuda())

                    scores_tail = torch.cat(
                        [scores1_tail, scores2_tail, scores3_tail, scores4_tail], dim=0)

                else:
                    scores_tail, _ = model(torch.LongTensor(new_x_batch_tail).cuda())

                sorted_scores_tail, sorted_indices_tail = torch.sort(
                    scores_tail.view(-1), dim=-1, descending=True)

                # Just search for zeroth index in the sorted scores, we appended valid triple at top
                ranks_tail.append(
                    np.where(sorted_indices_tail.cpu().numpy() == 0)[0][0] + 1)
                reciprocal_ranks_tail.append(1.0 / ranks_tail[-1])
                #print("ranks_head - ", ranks_head[-1], "    ranks_tail - ", ranks_tail[-1])

            for i in range(len(ranks_head)):
                if ranks_head[i] <= 100:
                    hits_at_100_head = hits_at_100_head + 1
                if ranks_head[i] <= 10:
                    hits_at_ten_head = hits_at_ten_head + 1
                if ranks_head[i] <= 3:
                    hits_at_three_head = hits_at_three_head + 1
                if ranks_head[i] == 1:
                    hits_at_one_head = hits_at_one_head + 1

            for i in range(len(ranks_tail)):
                if ranks_tail[i] <= 100:
                    hits_at_100_tail = hits_at_100_tail + 1
                if ranks_tail[i] <= 10:
                    hits_at_ten_tail = hits_at_ten_tail + 1
                if ranks_tail[i] <= 3:
                    hits_at_three_tail = hits_at_three_tail + 1
                if ranks_tail[i] == 1:
                    hits_at_one_tail = hits_at_one_tail + 1

            assert len(ranks_head) == len(reciprocal_ranks_head)
            assert len(ranks_tail) == len(reciprocal_ranks_tail)
            print("\nCurrent iteration time {}".format(time.time() - start_time))

            average_hits_at_100_head.append(
                hits_at_100_head / len(ranks_head))
            average_hits_at_ten_head.append(
                hits_at_ten_head / len(ranks_head))
            average_hits_at_three_head.append(
                hits_at_three_head / len(ranks_head))
            average_hits_at_one_head.append(
                hits_at_one_head / len(ranks_head))
            average_mean_rank_head.append(sum(ranks_head) / len(ranks_head))
            average_mean_recip_rank_head.append(
                sum(reciprocal_ranks_head) / len(reciprocal_ranks_head))

            average_hits_at_100_tail.append(
                hits_at_100_tail / len(ranks_head))
            average_hits_at_ten_tail.append(
                hits_at_ten_tail / len(ranks_head))
            average_hits_at_three_tail.append(
                hits_at_three_tail / len(ranks_head))
            average_hits_at_one_tail.append(
                hits_at_one_tail / len(ranks_head))
            average_mean_rank_tail.append(sum(ranks_tail) / len(ranks_tail))
            average_mean_recip_rank_tail.append(
                sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail))


        cumulative_hits_100 = (sum(average_hits_at_100_head) / len(average_hits_at_100_head)
                               + sum(average_hits_at_100_tail) / len(average_hits_at_100_tail)) / 2
        cumulative_hits_ten = (sum(average_hits_at_ten_head) / len(average_hits_at_ten_head)
                               + sum(average_hits_at_ten_tail) / len(average_hits_at_ten_tail)) / 2
        cumulative_hits_three = (sum(average_hits_at_three_head) / len(average_hits_at_three_head)
                                 + sum(average_hits_at_three_tail) / len(average_hits_at_three_tail)) / 2
        cumulative_hits_one = (sum(average_hits_at_one_head) / len(average_hits_at_one_head)
                               + sum(average_hits_at_one_tail) / len(average_hits_at_one_tail)) / 2
        cumulative_mean_rank = (sum(average_mean_rank_head) / len(average_mean_rank_head)
                                + sum(average_mean_rank_tail) / len(average_mean_rank_tail)) / 2
        cumulative_mean_recip_rank = (sum(average_mean_recip_rank_head) / len(average_mean_recip_rank_head) + sum(
            average_mean_recip_rank_tail) / len(average_mean_recip_rank_tail)) / 2

        print("\nCumulative stats are -> ")
        print("Hits@100 are {}".format(cumulative_hits_100))
        print("Hits@10 are {}".format(cumulative_hits_ten))
        print("Hits@3 are {}".format(cumulative_hits_three))
        print("Hits@1 are {}".format(cumulative_hits_one))
        print("Mean rank {}".format(cumulative_mean_rank))
        print("Mean Reciprocal Rank {}".format(cumulative_mean_recip_rank))

        return cumulative_mean_recip_rank, cumulative_mean_rank, cumulative_hits_one, cumulative_hits_three, cumulative_hits_ten
