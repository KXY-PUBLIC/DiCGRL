import torch
import os
import numpy as np
import pickle
import copy

def init_embeddings(entity_file, relation_file, emb_size, k=1):
    entity_emb, relation_emb = [], []

    with open(entity_file) as f:
        for line in f:
            ent_vec = []
            tmp = [float(val) for val in line.strip().split()]
            emb_k = int(emb_size / len(tmp))
            tot_k = int(emb_k * k)
            for i in range(tot_k):
                ent_vec += tmp
            entity_emb.append(ent_vec)

    with open(relation_file) as f:
        for line in f:
            rel_vec = []
            tmp = [float(val) for val in line.strip().split()]
            emb_k = int(emb_size / len(tmp))
            for i in range(emb_k):
                rel_vec += tmp
            relation_emb.append(rel_vec)

    return np.array(entity_emb, dtype=np.float32), np.array(relation_emb, dtype=np.float32)


def load_entity(filename):
    entity2id = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip().split()) > 1:
                line_split = line.strip().split()
                entity, entity_id = line_split[0].strip(), line_split[1].strip()
                entity2id[entity] = int(entity_id)
    return entity2id


def load_relation(filename):
    relation2id = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip().split()) > 1:
                line_split = line.strip().split()
                relation, relation_id = line_split[0].strip(), line_split[1].strip()
                relation2id[relation] = int(relation_id)
    return relation2id

def get_final_nei(nei_ent, nei_rel, max_neighbor, fi_nei_ent, fi_nei_rel):
    for ent in nei_ent:
        tmp_nei_ent = nei_ent[ent]
        tmp_nei_rel = nei_rel[ent]
        if len(tmp_nei_ent) >= max_neighbor:
            rand_idx = np.random.choice(tmp_nei_ent.shape[0], max_neighbor, replace=False)
            nei_ent_tmp = tmp_nei_ent[rand_idx]
            nei_rel_tmp = tmp_nei_rel[rand_idx]
            fi_nei_ent[ent] = nei_ent_tmp
            fi_nei_rel[ent] = nei_rel_tmp
        else:
            left = 0
            len_nei = tmp_nei_ent.shape[0]
            #print("len_nei", len_nei, tmp_nei_ent)
            while max_neighbor - left >= len_nei:
                #print("left", left, fi_nei_ent[ent], fi_nei_rel[ent])
                fi_nei_ent[ent][left:left + len_nei] = tmp_nei_ent
                fi_nei_rel[ent][left:left + len_nei] = tmp_nei_rel
                left += len_nei
            rand_idx = np.random.choice(tmp_nei_ent.shape[0], max_neighbor - left, replace=False)
            nei_ent_tmp = tmp_nei_ent[rand_idx]
            nei_rel_tmp = tmp_nei_rel[rand_idx]
            fi_nei_ent[ent][left:] = nei_ent_tmp
            fi_nei_rel[ent][left:] = nei_rel_tmp

    return fi_nei_ent, fi_nei_rel

def load_train_data(filename, entity2id, relation2id, nei_ent, ent_cnt, low_th=20):
    with open(filename) as f:
        lines = f.readlines()

    print("nei_ent", len(nei_ent))
    triples_data = []
    first_nei, second_nei = [], []
    nei_ent_new = copy.deepcopy(nei_ent)
    for line in lines:
        line = line.strip().split()
        e1, relation, e2 = line[0].strip(), line[1].strip(), line[2].strip()
        head, rel, tail = entity2id[e1], relation2id[relation], entity2id[e2]
        triples_data.append((head, rel, tail))

        if head not in nei_ent_new:
            nei_ent_new[head] = set()
            ent_cnt[head] = 0
        if tail not in nei_ent_new:
            nei_ent_new[tail] = set()
            ent_cnt[tail] = 0
        nei_ent_new[head].add((head, rel, tail))
        nei_ent_new[tail].add((head, rel, tail))
        ent_cnt[head] += 1
        ent_cnt[tail] += 1

        # first-order neighbors
        if head in nei_ent:
            first_nei.extend(list(nei_ent[head]))
        if tail in nei_ent:
            first_nei.extend(list(nei_ent[tail]))


    low_ent = [ent for ent, cnt in ent_cnt.items() if cnt < low_th]
    print("before first_nei", len(first_nei))
    first_nei = list(set(first_nei))
    first_nei = [(h, r, t) for (h, r, t) in first_nei if h in low_ent or t in low_ent]
    for (h, r, t) in first_nei:
        if (h, r, t) in triples_data:
            print("wrong", h, r, t)
        second_nei.extend(list(nei_ent[h]))
        second_nei.extend(list(nei_ent[t]))
    print("before second_nei", len(second_nei))
    second_nei = list(set(second_nei) - set(first_nei))
    second_nei = [(h, r, t) for (h, r, t) in second_nei if h in low_ent or t in low_ent]

    print("nei_ent_new", len(nei_ent_new))

    return triples_data, (nei_ent_new, ent_cnt), first_nei, second_nei

def load_data(filename, entity2id, relation2id):
    with open(filename) as f:
        lines = f.readlines()

    triples_data = []
    for line in lines:
        line = line.strip().split()
        e1, relation, e2 = line[0].strip(), line[1].strip(), line[2].strip()
        head, rel, tail = entity2id[e1], relation2id[relation], entity2id[e2]
        triples_data.append((head, rel, tail))

    return triples_data

def build_data(path='./data/WN18RR/', seed=42, data_idx=0, test_idx="", process=0, low_th=20):
    np.random.seed(seed)

    entity2id = load_entity(os.path.join(path, 'entity2id.txt'))
    relation2id = load_relation(os.path.join(path, 'relation2id.txt'))
    sub_entity2id = load_entity(os.path.join(path, str(data_idx), 'entity2id.txt'))
    print("sub_entity", len(sub_entity2id))

    train_file = os.path.join(path, str(data_idx), "train.txt")
    print("dealing ", train_file)
    nei_file_name = os.path.join(path, str(data_idx), "train_nei.pkl")
    if os.path.exists(nei_file_name) and not process:
        train_triples = load_data(train_file, entity2id, relation2id)
        mat_file = open(nei_file_name, 'rb')
        first_nei, second_nei = pickle.load(mat_file)
    else:
        if data_idx > 0:
            dict_name = os.path.join(path, str(data_idx - 1), "train_nei_dict.pkl")
            print("loading dict file ", dict_name)
            dict_file = open(dict_name, 'rb')
            nei_ent, ent_cnt = pickle.load(dict_file)
        else:
            print("init neighbor dict")
            nei_ent, ent_cnt = {}, {}

        train_triples, train_adjacency_dict, first_nei, second_nei = load_train_data(train_file,
            entity2id, relation2id, nei_ent, ent_cnt, low_th=low_th)
        dict_name = os.path.join(path, str(data_idx), "train_nei_dict.pkl")
        print("saving dict to ", dict_name)
        dict_file = open(dict_name, 'wb')
        pickle.dump(train_adjacency_dict, dict_file)
        dict_file.close()

        print("saving first and second neighbor to ", nei_file_name)
        nei_file = open(nei_file_name, 'wb')
        pickle.dump((first_nei, second_nei), nei_file)
        nei_file.close()

    print("\n train_triples_data", len(train_triples), "first_nei:", len(first_nei), "second_nei", len(second_nei))

    validation_file = os.path.join(path, str(data_idx), "valid.txt")
    print("dealing ", validation_file)
    validation_triples = load_data(validation_file, entity2id, relation2id)

    test_sub_triples = {}
    if test_idx == "":
        test_file = os.path.join(path, str(data_idx), "test.txt")
        for idx in range(data_idx + 1):
            test_sub_file = os.path.join(path, str(idx), "test.txt")
            test_sub_triples[idx] = load_data(test_sub_file, entity2id, relation2id)
    else:
        test_file = os.path.join(path, str(test_idx), "test.txt")
    print("dealing ", test_file)
    test_triples = load_data(test_file, entity2id, relation2id)

    valid_train_triples_list, valid_triples_list = [], []
    valid_triples_name = os.path.join(path, str(data_idx), "valid_triples.pkl")
    if os.path.exists(valid_triples_name) and not process:
        valid_train_triples_list, valid_triples_list = pickle.load(open(valid_triples_name, 'rb'))
    else:
        if data_idx > 0:
            ori_valid_triples_name = os.path.join(path, str(data_idx - 1), "valid_triples.pkl")
            valid_train_triples_list, valid_triples_list = pickle.load(open(ori_valid_triples_name, 'rb'))
        valid_train_triples_list += train_triples
        valid_triples_list += train_triples + validation_triples + test_triples
        print("saving valid_triples_list to ", valid_triples_name)
        valid_triples_file = open(valid_triples_name, 'wb')
        pickle.dump([valid_train_triples_list, valid_triples_list], valid_triples_file)
        valid_triples_file.close()

    print("valid_triples_list", len(valid_triples_list))

    return (train_triples, first_nei, second_nei), validation_triples, \
           test_triples, entity2id, relation2id, sub_entity2id, test_sub_triples, valid_triples_list, valid_train_triples_list

def build_all_data(path='./data/WN18RR/', seed=42, up_bound=0, data_idx=0):
    np.random.seed(seed)

    entity2id = load_entity(os.path.join(path, 'entity2id.txt'))
    relation2id = load_relation(os.path.join(path, 'relation2id.txt'))

    first_nei, second_nei = [], []

    if up_bound:
        train_triples = load_data(os.path.join(path, str(data_idx), "train.txt"), entity2id, relation2id)
        for idx in range(data_idx):
            train_sub_file = os.path.join(path, str(idx), "train.txt")
            train_triples += load_data(train_sub_file, entity2id, relation2id)
    else:
        train_triples = load_data(os.path.join(path, "train.txt"), entity2id, relation2id)

    validation_triples = load_data(os.path.join(path, "valid.txt"), entity2id, relation2id)

    test_sub_triples = {}
    if up_bound:
        test_file = os.path.join(path, str(data_idx), "test.txt")
        for idx in range(data_idx + 1):
            test_sub_file = os.path.join(path, str(idx), "test.txt")
            test_sub_triples[idx] = load_data(test_sub_file, entity2id, relation2id)
        test_triples = load_data(test_file, entity2id, relation2id)
    else:
        test_triples = load_data(os.path.join(path, "test.txt"), entity2id, relation2id)

    if up_bound:
        valid_triples_name = os.path.join(path, str(data_idx), "valid_triples.pkl")
        valid_train_triples_list, valid_triples_list = pickle.load(open(valid_triples_name, 'rb'))
    else:
        valid_train_triples_list = train_triples
        valid_triples_list = train_triples + validation_triples + test_triples

    return (train_triples, first_nei, second_nei), validation_triples, test_triples, \
           entity2id, relation2id, entity2id, test_sub_triples, valid_triples_list, valid_train_triples_list
