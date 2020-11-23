import numpy as np
import scipy.sparse as sp
import torch
import os

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(args, data_idx, base_path="./data/cora", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    print('Loading {} dataset...'.format(dataset))

    idx_labels = np.genfromtxt(os.path.join(base_path, "labels.txt"), dtype=np.dtype(str))
    labels = encode_onehot(idx_labels[:, -1])

    idx_features = np.genfromtxt(os.path.join(base_path, "features.txt"), dtype=np.dtype(str))
    features = np.array(sp.csr_matrix(idx_features[:, 1:], dtype=np.float32).todense())
    #features = np.tile(features, (1, args.k_factors))

    if args.all_data == 0:
        path = os.path.join(base_path, str(data_idx))
    else:
        path = base_path

    # build graph
    edges = np.genfromtxt(os.path.join(path, "edgelist.txt"), dtype=np.int32)
    ori_adj = None
    if args.all_data == 0 and data_idx > 0:
        tmp_edges = None
        for idx in range(data_idx):
            tmp = np.genfromtxt(os.path.join(base_path, str(idx), "edgelist.txt"), dtype=np.int32)
            if tmp_edges is None:
                tmp_edges = tmp
            else:
                tmp_edges = np.row_stack((tmp, tmp_edges))
        if args.up_bound:
            edges = np.row_stack((edges, tmp_edges))
        else:
            ori_adj = sp.coo_matrix((np.ones(tmp_edges.shape[0]), (tmp_edges[:, 0], tmp_edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
            ori_adj = ori_adj + ori_adj.T.multiply(ori_adj.T > ori_adj) - ori_adj.multiply(ori_adj.T > ori_adj)
            ori_adj = normalize_adj(ori_adj + sp.eye(ori_adj.shape[0]))
            ori_adj = torch.FloatTensor(np.array(ori_adj.todense()))
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    print("idx_labels, features, edges", idx_labels.shape, features.shape, edges.shape)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    if dataset == "cora":
        features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = np.genfromtxt(os.path.join(path, "train.txt"), dtype=np.int32)[:, 0]
    idx_val = np.genfromtxt(os.path.join(path, "valid.txt"), dtype=np.int32)[:, 0]
    idx_test = np.genfromtxt(os.path.join(path, "test.txt"), dtype=np.int32)[:, 0]
    ori_idx_train, ori_idx_valid = None, None
    if args.all_data == 0 and data_idx > 0:
        tmp_train, tmp_val = None, None
        for idx in range(data_idx):
            tmp_t = np.genfromtxt(os.path.join(base_path, str(idx), "train.txt"), dtype=np.int32)[:, 0]
            tmp_v = np.genfromtxt(os.path.join(base_path, str(idx), "valid.txt"), dtype=np.int32)[:, 0]
            if tmp_train is None:
                tmp_train = tmp_t
                tmp_val = tmp_v
            else:
                tmp_train = np.concatenate((tmp_t, tmp_train))
                tmp_val = np.concatenate((tmp_v, tmp_train))

        if args.up_bound:
            idx_train = np.concatenate((idx_train, tmp_train))
            idx_val = np.concatenate((idx_val, tmp_val))
        else:
            ori_idx_train = torch.LongTensor(tmp_train)
            ori_idx_valid = torch.LongTensor(tmp_val)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    print("adj, features, labels", adj.size(), features.size(), labels.size())
    print("train, val, test", idx_train.size(), idx_val.size(), idx_test.size())

    test_sub_idx = {}
    if args.all_data == 0:
        for idx in range(data_idx + 1):
            test_sub_idx[idx] = np.genfromtxt(os.path.join(base_path, str(idx), "test.txt"), dtype=np.int32)[:, 0]
            test_sub_idx[idx] = torch.LongTensor(test_sub_idx[idx])

    return adj, features, labels, idx_train, idx_val, idx_test, test_sub_idx, ori_adj, ori_idx_train, ori_idx_valid


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


class Corpus:
    def __init__(self, features, adj, labels, idx_train, idx_val, idx_test, ori_adj, ori_idx_train, ori_idx_valid):
        self.features = features
        self.adj = adj
        self.labels = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.ori_adj = ori_adj
        self.ori_idx_train = ori_idx_train
        self.ori_idx_valid = ori_idx_valid