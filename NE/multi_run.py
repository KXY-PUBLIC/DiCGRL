from __future__ import division
from __future__ import print_function

import os
import copy
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from dataloader import load_data, accuracy, Corpus
from models import GAT, SpGAT, SpGAT_2
import sys
sys.path.append("..")
from utils import save_model, load_model

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="./data/", help="data directory")
parser.add_argument("--output_dir", default="./results/", help="Folder name to save the models.")
parser.add_argument("--model_name", default="SpGAT", help="")
parser.add_argument("--dataset", default="cora", help="dataset")
parser.add_argument("--evaluate", type=int, default=0, help="only evaluate")
parser.add_argument("--ckpt", default="None", help="")
parser.add_argument("--load", default="None", help="")
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument("--all_data", type=int, default=0, help="use whole dataset")
parser.add_argument("--process", type=int, default=0, help="process from scratch")
parser.add_argument("--up_bound", type=int, default=0, help="train up_bound")
parser.add_argument("--N", type=int, default=4, help="data numbers")
parser.add_argument("--use_cuda", type=int, default=0, help="use_cuda")

parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument("--att_lr", type=int, default=0, help="do try_att_lr")

parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=200, help='Patience')

parser.add_argument("--k_factors", type=int, default=1, help="Number of k")
parser.add_argument("--top_n", type=int, default=2, help="choose top n")
parser.add_argument("--w1", type=float, default=0.0, help="loss_2 weight: top2 constrain")

args = parser.parse_args()
CUDA = torch.cuda.is_available()
if CUDA:
    print("using CUDA")

def main():
    args.data_dir = os.path.join(args.data_dir, args.dataset)
    args.output_dir = os.path.join(args.output_dir, args.dataset)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if CUDA:
        args.use_cuda = CUDA
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    print("args = ", args)

    ori_model = 'None'
    ori_load = True

    for idx in range(args.N):
        data_idx = idx
        # Load data
        adj, features, labels, idx_train, idx_val, idx_test, test_sub_idx, ori_adj, ori_idx_train, ori_idx_valid = \
            load_data(args, data_idx, base_path=args.data_dir, dataset=args.dataset)

        file_name = "model_name_" + str(args.model_name) + "_lr_" + str(args.lr) + "_epochs_" + str(
            args.epochs) + "_k_factors_" + str(args.k_factors) + "_up_bound_" + str(
            args.up_bound) + "_top_n_" + str(args.top_n) + "_att_lr_" + str(args.att_lr) + "_hidden_" + str(
            args.hidden) + "_w1_" + str(args.w1)

        if args.all_data:
            model_path = os.path.join(args.output_dir, file_name)
        else:
            model_path = os.path.join(args.output_dir, str(data_idx), file_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # Model and optimizer
        if args.model_name == "SpGAT":
            model = SpGAT(nfeat=features.shape[1],
                        nhid=args.hidden,
                        nclass=int(labels.max()) + 1,
                        dropout=args.dropout,
                        nheads=args.nb_heads,
                        alpha=args.alpha)
        elif args.model_name == "SpGAT_2":
            model = SpGAT_2(nfeat=features.shape[1], nclass=int(labels.max()) + 1, config=args)
        elif args.model_name == "SpGAT2":
            model = SpGAT_2(nfeat=features.shape[1], nclass=int(labels.max()) + 1, config=args)
        else:
            model = GAT(nfeat=features.shape[1],
                        nhid=args.hidden,
                        nclass=int(labels.max()) + 1,
                        dropout=args.dropout,
                        nheads=args.nb_heads,
                        alpha=args.alpha)

        print("load path", args.load)
        if args.load != 'None' and ori_load:
            model = load_model(model, args.load)
            print("model loaded")
            ori_load = False

        if ori_model != 'None':
            model = copy.deepcopy(ori_model)
            print("load model from", idx - 1)

        print(model.state_dict().keys())

        if CUDA:
            model.cuda()
            features = Variable(features.cuda())
            adj = Variable(adj.cuda())
            labels = Variable(labels.cuda())
            idx_train = idx_train.cuda()
            idx_val = idx_val.cuda()
            idx_test = idx_test.cuda()
            if "_" in args.model_name and not args.all_data and data_idx > 0 and ori_adj is not None:
                ori_adj = Variable(ori_adj.cuda())
                ori_idx_train = ori_idx_train.cuda()
                ori_idx_valid = ori_idx_valid.cuda()

        loader = Corpus(features, adj, labels, idx_train, idx_val, idx_test, ori_adj, ori_idx_train, ori_idx_valid)

        for name, param in model.named_parameters():
            if param.requires_grad == False:
                print("False", name)
                param.requires_grad = True

        best_epoch = 0
        if args.evaluate == 0:
            best_epoch = train(model, model_path, loader, data_idx)
            ori_model = copy.deepcopy(model)
        evaluate(model, model_path, loader, data_idx, best_epoch=best_epoch, test_sub_idx=test_sub_idx)
        evaluate(model, model_path, loader, data_idx, best_epoch=best_epoch, test_sub_idx=test_sub_idx, best_or_final='final')

        args.load = os.path.join(model_path, 'trained_final.pth')


def train(model, model_path, loader, data_idx):
    print("model training")
    ori_model = copy.deepcopy(model)

    if args.att_lr and data_idx > 0:
        att_params = list(map(id, model.a_k.parameters()))
        base_params = filter(lambda p: id(p) not in att_params, model.parameters())
        optimizer = torch.optim.Adam([
            {'params': base_params},
            {'params': model.rel_attention.parameters(), 'lr': args.lr * 0.1}], lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Train model
    t_total = time.time()
    bad_counter = 0
    best = args.epochs + 1
    best_epoch = 0

    for epoch in range(args.epochs):
        cur_lr = optimizer.param_groups[0]['lr']
        epoch_loss = []

        if "_" in args.model_name and not args.all_data and data_idx > 0 and loader.ori_adj is not None:
            t = time.time()
            if epoch % 10 == 0:
                model.eval()
                sep_edge = model(loader.features, loader.ori_adj, only_edge=True)

            model.train()
            optimizer.zero_grad()

            output, _ = model(loader.features, loader.ori_adj, ori_sep_edge=sep_edge)
            loss_train = F.nll_loss(output[loader.ori_idx_train], loader.labels[loader.ori_idx_train])
            acc_train = accuracy(output[loader.ori_idx_train], loader.labels[loader.ori_idx_train])

            loss_train.backward()
            optimizer.step()

            model.eval()
            loss_val = F.nll_loss(output[loader.ori_idx_valid], loader.labels[loader.ori_idx_valid])
            acc_val = accuracy(output[loader.ori_idx_valid], loader.labels[loader.ori_idx_valid])

            if epoch % 10 == 0:
                print('\ntrain neighbors-> '
                      'Epoch: {:04d}'.format(epoch + 1),
                      'loss_train: {:.4f}'.format(loss_train.data.item()),
                      'acc_train: {:.4f}'.format(acc_train.data.item()),
                      'loss_val: {:.4f}'.format(loss_val.data.item()),
                      'acc_val: {:.4f}'.format(acc_val.data.item()),
                      'time: {:.4f}s'.format(time.time() - t),
                      'lr: {:04f}'.format(cur_lr))

            epoch_loss.append(loss_train.data.item())

        t = time.time()
        model.train()
        optimizer.zero_grad()

        output, att_loss = model(loader.features, loader.adj)
        loss_train = F.nll_loss(output[loader.idx_train], loader.labels[loader.idx_train])
        acc_train = accuracy(output[loader.idx_train], loader.labels[loader.idx_train])
        att_loss_data = 0.0
        if args.w1 != 0:
            loss_train = loss_train + args.w1 * att_loss
            att_loss_data = att_loss.data.item()

        loss_train.backward()
        optimizer.step()

        model.eval()
        #with torch.no_grad():
        #    output = model(features, adj)
        loss_val = F.nll_loss(output[loader.idx_val], loader.labels[loader.idx_val])
        acc_val = accuracy(output[loader.idx_val], loader.labels[loader.idx_val])

        if epoch % 10 == 0:
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.data.item()),
                  'acc_train: {:.4f}'.format(acc_train.data.item()),
                  'loss_val: {:.4f}'.format(loss_val.data.item()),
                  'acc_val: {:.4f}'.format(acc_val.data.item()),
                  'loss_att: {:.4f}'.format(att_loss_data),
                  'time: {:.4f}s'.format(time.time() - t))

        loss_value = loss_val.data.item()
        epoch_loss.append(loss_value)

        avg_loss = sum(epoch_loss) / len(epoch_loss)
        if avg_loss < best:
            save_model(model, "best", model_path)
            best = avg_loss
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1


    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    save_model(model, "final", model_path)

    return best_epoch


def evaluate(model, model_path, loader, data_idx, best_epoch=0, test_sub_idx=None, best_or_final='best'):
    # Restore best model
    # model = SpGAT_2(nfeat=loader.features.shape[1], nclass=int(loader.labels.max()) + 1, config=args)
    # model.cuda()
    print("\n\nmodel evaluating: ", best_or_final, "k = ", args.k_factors, "top_n = ", args.top_n)
    if best_epoch != 0:
        print("best_epoch", best_epoch)
    if args.ckpt != 'None':
        model_path = args.ckpt
    ckpt_path = os.path.join(model_path, 'trained_' + best_or_final + '.pth')
    model = load_model(model, ckpt_path)
    print("model loaded")

    model.eval()
    # model.train()
    output, _ = model(loader.features, loader.adj)

    if test_sub_idx and not args.all_data and data_idx > 0:
        for idx, sub_idx in test_sub_idx.items():
            if sub_idx is not None:
                if CUDA:
                    sub_idx = sub_idx.cuda()
                loss_train = F.nll_loss(output[loader.idx_train], loader.labels[loader.idx_train])
                acc_train = accuracy(output[loader.idx_train], loader.labels[loader.idx_train])
                loss_val = F.nll_loss(output[loader.idx_val], loader.labels[loader.idx_val])
                acc_val = accuracy(output[loader.idx_val], loader.labels[loader.idx_val])

                loss_test = F.nll_loss(output[sub_idx], loader.labels[sub_idx])
                acc_test = accuracy(output[sub_idx], loader.labels[sub_idx])
                print("\ndealing test:", idx,
                      'loss_train: {:.4f}'.format(loss_train.data.item()),
                      'acc_train: {:.4f}'.format(acc_train.data.item()),
                      'loss_val: {:.4f}'.format(loss_val.data.item()),
                      'acc_val: {:.4f}'.format(acc_val.data.item()),
                      "loss= {:.4f}".format(loss_test.data.item()),
                      "accuracy= {:.4f}".format(acc_test.data.item()))
    else:
        loss_train = F.nll_loss(output[loader.idx_train], loader.labels[loader.idx_train])
        acc_train = accuracy(output[loader.idx_train], loader.labels[loader.idx_train])
        loss_val = F.nll_loss(output[loader.idx_val], loader.labels[loader.idx_val])
        acc_val = accuracy(output[loader.idx_val], loader.labels[loader.idx_val])

        loss_test = F.nll_loss(output[loader.idx_test], loader.labels[loader.idx_test])
        acc_test = accuracy(output[loader.idx_test], loader.labels[loader.idx_test])
        print("Test set results:",
              'loss_train: {:.4f}'.format(loss_train.data.item()),
              'acc_train: {:.4f}'.format(acc_train.data.item()),
              'loss_val: {:.4f}'.format(loss_val.data.item()),
              'acc_val: {:.4f}'.format(acc_val.data.item()),
              "loss= {:.4f}".format(loss_test.data.item()),
              "accuracy= {:.4f}".format(acc_test.data.item()))


if __name__ == '__main__':

    main()