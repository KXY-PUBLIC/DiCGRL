import torch

from models import ConvKB, TransE, ConvKB_2, TransE_2
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from process_data import init_embeddings, build_data, build_all_data, load_entity
from dataloader import Corpus
from utils import save_model, load_model

import random
import argparse
import os
import copy
import logging
import time


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="./data/", help="data directory")
parser.add_argument("--output_dir", default="./results/", help="Folder name to save the models.")
parser.add_argument("--model_name", default="ConvKB", help="")
parser.add_argument("--dataset", default="FB15k-237", help="dataset")
parser.add_argument("--evaluate", type=int, default=0, help="only evaluate")
parser.add_argument("--ckpt", default="None", help="")
parser.add_argument("--load", default="None", help="")
parser.add_argument("--test_idx", default="", help="test index")
parser.add_argument("--all_data", type=int, default=0, help="use whole dataset")
parser.add_argument("--process", type=int, default=0, help="process from scratch")
parser.add_argument("--up_bound", type=int, default=0, help="train up_bound")
parser.add_argument("--s_N", type=int, default=0, help="data start index")
parser.add_argument("--N", type=int, default=5, help="data numbers")

parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument("--pretrained_emb", type=int, default=0, help="Use pretrained embeddings")
parser.add_argument("--embedding_size", type=int, default=200, help="Size of embeddings (if pretrained not used)")
parser.add_argument("--valid_invalid_ratio", type=int, default=40, help="Ratio of valid to invalid triples for training")
parser.add_argument("--seed", type=int, default=42, help="seed")

parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--att_lr", type=int, default=0, help="make the rel_att change slower")
parser.add_argument("--weight_decay", type=float, default=1e-5, help="L2 reglarization")
parser.add_argument("--step_size", type=int, default=50, help="step_size")
parser.add_argument("--gamma", type=float, default=0.5, help="gamma")

parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability")
parser.add_argument("--out_channels", type=int, default=50, help="Number of output channels in conv layer")
parser.add_argument("--do_normalize", type=int, default=0, help="normalize for init embedding")
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument("--use_second_nei", type=int, default=1, help="use_second_nei")
parser.add_argument("--l1", type=int, default=1, help="use l1")
parser.add_argument("--ratio", type=float, default=0.5, help="neighbor's influence ratio")
parser.add_argument("--margin", type=float, default=5, help="Margin used in hinge loss")
parser.add_argument("--w1", type=float, default=0.0, help="loss_2 weight: top2 constrain")
parser.add_argument("--low_th", type=float, default=10000, help="low threshold in process")
parser.add_argument("--k_factors", type=int, default=1, help="Number of k")
parser.add_argument("--top_n", type=int, default=2, help="choose top n")
parser.add_argument("--disen_first", type=int, default=0, help="do disen_first")


args = parser.parse_args()


def main():
    args.data_dir = os.path.join(args.data_dir, args.dataset)
    args.output_dir = os.path.join(args.output_dir, args.dataset)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    CUDA = torch.cuda.is_available()
    if CUDA:
        print("using CUDA")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print("args = ", args)

    ori_model = 'None'
    ori_load = True

    for idx in range(args.s_N, args.N):
        data_idx = idx

        if args.all_data or args.up_bound:
            train_data, validation_data, test_data, entity2id, relation2id, sub_entity2id, test_sub_triples, valid_triples_list, valid_train_triples_list = \
                build_all_data(args.data_dir, seed=args.seed, up_bound=args.up_bound, data_idx=data_idx)
        else:
            train_data, validation_data, test_data, entity2id, relation2id, sub_entity2id, test_sub_triples, valid_triples_list, valid_train_triples_list = \
                build_data(args.data_dir, seed=args.seed, data_idx=data_idx,
                           test_idx=args.test_idx, process=args.process, low_th=args.low_th)

        entity_embeddings = np.random.randn(len(entity2id), args.embedding_size * args.k_factors)
        if "_" in args.model_name:
            relation_embeddings = np.random.randn(len(relation2id), args.embedding_size * args.top_n)
        else:
            relation_embeddings = np.random.randn(len(relation2id), args.embedding_size)
        print("Initialised relations and entities randomly")

        entity_embeddings = torch.FloatTensor(entity_embeddings)
        relation_embeddings = torch.FloatTensor(relation_embeddings)
        print("Initial entity dimensions {} , relation dimensions {}".format(entity_embeddings.size(),
                                                                             relation_embeddings.size()))

        train_loader = Corpus(args, train_data, validation_data, test_data, sub_entity2id, relation2id,
                        args.batch_size, args.valid_invalid_ratio, valid_triples_list, valid_train_triples_list)

        file_name = "model_name_" + str(args.model_name) + "_embedding_size_" + str(args.embedding_size) + "_lr_" + str(
            args.lr) + "_epochs_" + str(args.epochs) + "_k_factors_" + str(args.k_factors) + "_batch_size_" + str(
            args.batch_size) + "_step_size_" + str(args.step_size) + "_l1_" + str(args.l1) + "_use_second_nei_" + str(
            args.use_second_nei) + "_w1_" + str(args.w1) + "_up_bound_" + str(args.up_bound) + "_top_n_" + str(
            args.top_n) + "_att_lr_" + str(args.att_lr)

        if args.all_data:
            model_path = os.path.join(args.output_dir, file_name)
        else:
            model_path = os.path.join(args.output_dir, str(data_idx), file_name)

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        if args.model_name == 'ConvKB':
            model = ConvKB(entity_embeddings, relation_embeddings, config=args)
        elif args.model_name == 'TransE':
            model = TransE(entity_embeddings, relation_embeddings, config=args)
        elif args.model_name == 'ConvKB_2':
            model = ConvKB_2(entity_embeddings, relation_embeddings, config=args)
        elif args.model_name == 'TransE_2':
            model = TransE_2(entity_embeddings, relation_embeddings, config=args)
        else:
            print("no such model name")

        print("load path", args.load)
        if args.load != 'None' and ori_load:
            model = load_model(model, args.load)
            print("model loaded")
            ori_load = False

        if ori_model != 'None':
            model = copy.deepcopy(ori_model)
            print("load model from", idx - 1)


        model.cuda()

        for name, param in model.named_parameters():
            #print("name", name)
            if param.requires_grad == False:
                print("False", name)
                param.requires_grad = True

        best_epoch = 0
        if args.evaluate == 0:
            best_epoch = train(args, train_loader, model, model_path, data_idx)
            ori_model = copy.deepcopy(model)
        evaluate(args, model, model_path, train_loader, file_name, data_idx, best_epoch=best_epoch, test_sub_triples=test_sub_triples)
        evaluate(args, model, model_path, train_loader, file_name, data_idx, best_epoch=best_epoch, test_sub_triples=test_sub_triples, best_or_final='final')

        args.load = os.path.join(model_path, 'trained_final.pth')

def train(args, train_loader, model, model_path, data_idx):
    print("model training")
    ori_param_dic = {}
    ori_model = copy.deepcopy(model)
    for n, p in ori_model.named_parameters():
        ori_param_dic[n] = Variable(p.data).cuda()

    if args.att_lr and data_idx > 0:
        att_params = list(map(id, model.rel_attention.parameters()))
        base_params = filter(lambda p: id(p) not in att_params, model.parameters())
        optimizer = torch.optim.Adam([
            {'params': base_params},
            {'params': model.rel_attention.parameters(), 'lr': args.lr * 0.01}], lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma, last_epoch=-1)

    epoch_losses = []   # losses of all epochs
    print("Number of epochs {}".format(args.epochs))

    min_loss = 10000.0
    best_epoch = 0
    start_time = time.time()
    first_top_att, first_top_indices = None, None
    num_iters_per_epoch = 0

    for epoch in range(args.epochs):
        tmp_seed = np.random.randint(0, 1000)
        print("\nepoch-> ", epoch)
        epoch_loss = []

        if "_" in args.model_name and not args.all_data and data_idx > 0:
            if epoch % 10 == 0:
                print("\nget neighbors-> ")
                model.eval()
                ent_indices_dic = train_loader.get_ent_att_idx(args, model.test)
                if epoch < 50:
                    # using original model to get neighbor
                    first_top_att, first_top_indices = train_loader.get_final_nei(args, ori_model.test, ent_indices_dic,
                                                                                  first_top_att, first_top_indices)
                else:
                    train_loader.get_final_nei(args, model.test, ent_indices_dic)

            print("\ntrain_neighbors-> ", len(train_loader.final_nei))
            tmp = list(zip(train_loader.final_nei, train_loader.loss_weight_list, train_loader.nei_top_att_list, train_loader.nei_top_idx_list))
            random.shuffle(tmp)
            train_loader.final_nei[:], train_loader.loss_weight_list[:], train_loader.nei_top_att_list[:], train_loader.nei_top_idx_list[:] = zip(*tmp)
            train_loader.final_nei_indices = np.array(list(train_loader.final_nei)).astype(np.int32)
            train_loader.final_nei_values = np.array([[1]] * len(train_loader.final_nei)).astype(np.float32)
            train_loader.loss_weight = np.array(list(train_loader.loss_weight_list)).astype(np.float32)
            train_loader.nei_top_att = np.array(list(train_loader.nei_top_att_list)).astype(np.float32)
            train_loader.nei_top_idx = np.array(list(train_loader.nei_top_idx_list)).astype(np.int32)

            model.train()  # getting in training mode

            if len(train_loader.final_nei_indices) % args.batch_size == 0:
                num_iters_per_epoch = len(train_loader.final_nei_indices) // args.batch_size
            else:
                num_iters_per_epoch = (len(train_loader.final_nei_indices) // args.batch_size) + 1

            for iters in range(num_iters_per_epoch):
                start_time_iter = time.time()
                batch_triples, batch_labels, batch_loss_weight, batch_top_att, batch_top_idx = \
                    train_loader.get_iteration_batch(iters, is_nei=True, seed=tmp_seed)

                batch_triples = Variable(torch.LongTensor(batch_triples)).cuda()
                batch_labels = Variable(torch.FloatTensor(batch_labels)).cuda()
                batch_loss_weight = Variable(torch.FloatTensor(batch_loss_weight)).cuda()
                batch_top_att = Variable(torch.FloatTensor(batch_top_att)).cuda()
                batch_top_idx = Variable(torch.LongTensor(batch_top_idx)).cuda()

                loss_1, _ = model(batch_triples, batch_labels=batch_labels, batch_loss_weight=batch_loss_weight,
                                 batch_top_att=batch_top_att, batch_top_indices=batch_top_idx)
                loss = loss_1
                param_loss_data = 0.0

                optimizer.zero_grad()
                end_time_iter = time.time()

                loss.backward()
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                epoch_loss.append(loss.data.item())

                if iters % 50 == 0:
                    print("Iteration-> {0}  , Iteration_time-> {1:.4f} , Triple loss {2:.4f}, Param loss {3:.4f}, "
                          "Total loss {4:.4f}, total_norm {5:.4f}".
                          format(iters, end_time_iter - start_time_iter, loss_1.data.item(), param_loss_data,
                                 loss.data.item(), total_norm))


        print("train_triples", len(train_loader.train_triples))
        random.shuffle(train_loader.train_triples)
        train_loader.train_indices = np.array(list(train_loader.train_triples)).astype(np.int32)

        model.train()  # getting in training mode
        # epoch_loss = []

        if len(train_loader.train_indices) % args.batch_size == 0:
            num_iters_per_epoch = len(train_loader.train_indices) // args.batch_size
        else:
            num_iters_per_epoch = (len(train_loader.train_indices) // args.batch_size) + 1

        for iters in range(num_iters_per_epoch):
            start_time_iter = time.time()
            batch_triples, batch_labels = train_loader.get_iteration_batch(iters, seed=tmp_seed)

            batch_triples = Variable(torch.LongTensor(batch_triples)).cuda()
            batch_labels = Variable(torch.FloatTensor(batch_labels)).cuda()

            loss_1, att_loss = model(batch_triples, batch_labels=batch_labels)
            loss = loss_1
            att_or_baseline = "Attention"
            other_loss_data = 0.0
            if args.w1 != 0:
                loss = loss + args.w1 * att_loss
                other_loss_data = att_loss.data.item()

            optimizer.zero_grad()
            end_time_iter = time.time()

            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss.append(loss.data.item())

            if iters % 50 == 0:
                print("Iteration-> {0} , Iteration_time-> {1:.4f} , Triple loss {2:.4f}, {3} loss {4:.4f}, "
                      "Total loss {5:.4f}, total_norm {6:.4f}".
                      format(iters, end_time_iter - start_time_iter, loss_1.data.item(), att_or_baseline, other_loss_data,
                             loss.data.item(), total_norm))

        cur_lr = optimizer.param_groups[0]['lr']
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        print("Epoch {} , average loss {} , tot_time {}, learning rate {}".format(
            epoch, avg_loss, (time.time() - start_time)/60/60, cur_lr))
        epoch_losses.append(avg_loss)

        if avg_loss < min_loss:
            min_loss = avg_loss
            best_epoch = epoch
            save_model(model, "best", model_path)
            print("best_epoch-> ", epoch)

        scheduler.step()

    save_model(model, "final", model_path)

    return best_epoch


def evaluate(args, model, model_path, train_loader, file_name, data_idx, best_epoch=0, test_sub_triples=None, best_or_final='best'):
    print("\n\nmodel evaluating: ", best_or_final)
    if best_epoch != 0:
        print("best_epoch", best_epoch)
    if args.ckpt != 'None':
        model_path = args.ckpt
    ckpt_path = os.path.join(model_path, 'trained_' + best_or_final + '.pth')
    model = load_model(model, ckpt_path)
    model.eval()
    print("model loaded")

    if args.all_data:
        output_file = os.path.join(args.output_dir, "results_" + file_name + ".txt")
    else:
        output_file = os.path.join(args.output_dir, str(data_idx), "results_" + file_name + ".txt")

    sub_res = {}
    mean_mrr, mean_h10 = [], []
    if test_sub_triples and not args.all_data and data_idx > 0:
        for idx, sub_triples in test_sub_triples.items():
            if sub_triples:
                print("\ndealing test ", idx)
                train_loader.test_indices = np.array(list(sub_triples)).astype(np.int32)
                sub_MRR, sub_MR, sub_H1, sub_H3, sub_H10 = train_loader.get_validation_pred(args, model.test)
                sub_res[str(idx) + "_MRR"] = sub_MRR
                sub_res[str(idx) + "_MR"] = sub_MR
                sub_res[str(idx) + "_H1"] = sub_H1
                sub_res[str(idx) + "_H3"] = sub_H3
                sub_res[str(idx) + "_H10"] = sub_H10

                mean_mrr.append(sub_MRR)
                mean_h10.append(sub_H10)

        mean_mrr, mean_h10 = np.mean(mean_mrr), np.mean(mean_h10)
        print("mean_mrr, mean_h10", mean_mrr, mean_h10)
        with open(output_file, "w") as writer:
            logging.info("***** results *****")
            writer.write('Best epoch: %s\n' % str(best_epoch))
            writer.write("%s = %s\n" % ('args', str(args)))
            for key, val in sub_res.items():
                writer.write("sub_%s = %s\n" % (key, str(val)))
            writer.write("mean_MRR = %s\n" % (str(mean_mrr)))
            writer.write("mean_H10 = %s\n" % (str(mean_h10)))
    else:
        with torch.no_grad():
            MRR, MR, H1, H3, H10 = train_loader.get_validation_pred(args, model.test)

        with open(output_file, "w") as writer:
            logging.info("***** results *****")
            writer.write('Hits @1: %s\n' % (H1))
            writer.write('Hits @3: %s\n' % (H3))
            writer.write('Hits @10: %s\n' % (H10))
            writer.write('Mean rank: %s\n' % MR)
            writer.write('Mean reciprocal rank: %s\n' % MRR)
            writer.write('Best epoch: %s\n' % str(best_epoch))
            writer.write("%s = %s\n" % ('args', str(args)))


if __name__ == '__main__':

    main()