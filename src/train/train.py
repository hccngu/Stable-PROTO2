import os
import time
import datetime

import torch
import numpy as np
import copy

from train.utils import grad_param, get_norm
from dataset.sampler import ParallelSampler, ParallelSampler_Test, task_sampler
from tqdm import tqdm
from termcolor import colored
from train.test import test
import torch.nn.functional as F
from dataset import utils
from tools.tool import neg_dist, reidx_y


def train(train_data, val_data, model, class_names, args):
    '''
        Train the model
        Use val_data to do early stopping
    '''
    # creating a tmp directory to save the models
    out_dir = os.path.abspath(os.path.join(
                                  os.path.curdir,
                                  "tmp-runs",
                                  str(int(time.time() * 1e7))))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    best_acc = 0
    sub_cycle = 0
    best_path = None

    optG = torch.optim.Adam(grad_param(model, ['G']), lr=args.meta_lr)
    optCLF = torch.optim.Adam(grad_param(model, ['clf']), lr=args.task_lr)

    if args.lr_scheduler == 'ReduceLROnPlateau':
        schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optG, 'max', patience=args.patience//2, factor=0.1, verbose=True)
        schedulerCLF = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optCLF, 'max', patience=args.patience // 2, factor=0.1, verbose=True)

    elif args.lr_scheduler == 'ExponentialLR':
        schedulerG = torch.optim.lr_scheduler.ExponentialLR(optG, gamma=args.ExponentialLR_gamma)
        schedulerCLF = torch.optim.lr_scheduler.ExponentialLR(optCLF, gamma=args.ExponentialLR_gamma)



    print("{}, Start training".format(
        datetime.datetime.now()), flush=True)

    # train_gen = ParallelSampler(train_data, args, args.train_episodes)
    # train_gen_val = ParallelSampler_Test(train_data, args, args.val_episodes)
    # val_gen = ParallelSampler_Test(val_data, args, args.val_episodes)

    # sampled_classes, source_classes = task_sampler(train_data, args)
    acc = 0
    loss = 0
    for ep in range(args.train_epochs):

        sampled_classes, source_classes = task_sampler(train_data, args)
        class_names_dict = {}
        class_names_dict['label'] = class_names['label'][sampled_classes]
        class_names_dict['text'] = class_names['text'][sampled_classes]
        class_names_dict['text_len'] = class_names['text_len'][sampled_classes]
        class_names_dict['is_support'] = False


        train_gen = ParallelSampler(train_data, args, sampled_classes, source_classes, args.train_episodes)

        sampled_tasks = train_gen.get_epoch()
        class_names_dict = utils.to_tensor(class_names_dict, args.cuda, exclude_keys=['is_support'])

        grad = {'clf': [], 'G': []}

        if not args.notqdm:
            sampled_tasks = tqdm(sampled_tasks, total=train_gen.num_episodes,
                    ncols=80, leave=False, desc=colored('Training on train',
                        'yellow'))

        for task in sampled_tasks:
            if task is None:
                break
            q_loss, q_acc = train_one(task, class_names_dict, model, optG, optCLF, args, grad)
            acc += q_acc
            loss += q_loss

        if ep % 100 == 0:
            print("--------[TRAIN] ep:" + str(ep) + ", loss:" + str(q_loss.item()) + ", acc:" + str(q_acc.item()) + "-----------")

        if (ep % 200 == 0) and (ep != 0):
            acc = acc / args.train_episodes / 200
            loss = loss / args.train_episodes / 200
            print("--------[TRAIN] ep:" + str(ep) + ", mean_loss:" + str(loss.item()) + ", mean_acc:" + str(acc.item()) + "-----------")

            net = copy.deepcopy(model)
            acc, std = test(train_data, class_names, optCLF, net, args, args.test_epochs, False)
            print("[TRAIN] {}, {:s} {:2d}, {:s} {:s}{:>7.4f} ± {:>6.4f} ".format(
                datetime.datetime.now(),
                "ep", ep,
                colored("train", "red"),
                colored("acc:", "blue"), acc, std,
                ), flush=True)
            acc = 0
            loss = 0

            # Evaluate validation accuracy
            cur_acc, cur_std = test(val_data, class_names, optCLF, net, args, args.test_epochs, False)
            print(("[EVAL] {}, {:s} {:2d}, {:s} {:s}{:>7.4f} ± {:>6.4f}, "
                   "{:s} {:s}{:>7.4f}, {:s}{:>7.4f}").format(
                   datetime.datetime.now(),
                   "ep", ep,
                   colored("val  ", "cyan"),
                   colored("acc:", "blue"), cur_acc, cur_std,
                   colored("train stats", "cyan"),
                   colored("G_grad:", "blue"), np.mean(np.array(grad['G'])),
                   colored("clf_grad:", "blue"), np.mean(np.array(grad['clf'])),
                   ), flush=True)

            # Update the current best model if val acc is better
            if cur_acc > best_acc:
                best_acc = cur_acc
                best_path = os.path.join(out_dir, str(ep))

                # save current model
                print("{}, Save cur best model to {}".format(
                    datetime.datetime.now(),
                    best_path))

                torch.save(model['G'].state_dict(), best_path + '.G')
                torch.save(model['clf'].state_dict(), best_path + '.clf')

                sub_cycle = 0
            else:
                sub_cycle += 1

            # Break if the val acc hasn't improved in the past patience epochs
            if sub_cycle == args.patience:
                break

            if args.lr_scheduler == 'ReduceLROnPlateau':
                schedulerG.step(cur_acc)
                schedulerCLF.step(cur_acc)

            elif args.lr_scheduler == 'ExponentialLR':
                schedulerG.step()
                schedulerCLF.step()

    print("{}, End of training. Restore the best weights".format(
            datetime.datetime.now()),
            flush=True)

    # restore the best saved model
    model['G'].load_state_dict(torch.load(best_path + '.G'))
    model['clf'].load_state_dict(torch.load(best_path + '.clf'))

    if args.save:
        # save the current model
        out_dir = os.path.abspath(os.path.join(
                                      os.path.curdir,
                                      "saved-runs",
                                      str(int(time.time() * 1e7))))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        best_path = os.path.join(out_dir, 'best')

        print("{}, Save best model to {}".format(
            datetime.datetime.now(),
            best_path), flush=True)

        torch.save(model['G'].state_dict(), best_path + '.G')
        torch.save(model['clf'].state_dict(), best_path + '.clf')

        with open(best_path + '_args.txt', 'w') as f:
            for attr, value in sorted(args.__dict__.items()):
                f.write("{}={}\n".format(attr, value))

    return optCLF


def train_one(task, class_names_dict, model, optG, optCLF, args, grad):
    '''
        Train the model on one sampled task.
    '''
    model['G'].train()
    model['clf'].train()

    support, query = task
    # print("support, query:", support, query)
    # print("class_names_dict:", class_names_dict)

    # Embedding the document
    XS = model['G'](support)  # XS:[N*K, 256(hidden_size*2)]
    # print("XS:", XS.shape)
    YS = support['label']
    # print('YS:', YS)

    CN = model['G'](class_names_dict)  # CN:[N, 256(hidden_size*2)]]
    # print("CN:", CN.shape)

    XQ = model['G'](query)
    YQ = query['label']
    # print('YQ:', YQ)

    YS, YQ = reidx_y(args, YS, YQ)

    for _ in range(args.train_iter):

        # Embedding the document
        XS_mlp = model['clf'](XS)  # [N*K, 256(hidden_size*2)] -> [N*K, 128]

        CN_mlp = model['clf'](CN)  # [N, 256(hidden_size*2)]] -> [N, 128]

        neg_d = neg_dist(XS_mlp, CN_mlp)  # [N*K, N]
        # print("neg_d:", neg_d.shape)

        mlp_loss = model['clf'].loss(neg_d, YS)
        # print("mlp_loss:", mlp_loss)

        optCLF.zero_grad()
        mlp_loss.backward(retain_graph=True)
        optCLF.step()

    XQ_mlp = model['clf'](XQ)
    CN_mlp = model['clf'](CN)
    neg_d = neg_dist(XQ_mlp, CN_mlp)
    g_loss = model['clf'].loss(neg_d, YQ)

    optG.zero_grad()
    g_loss.backward()
    optG.step()

    _, pred = torch.max(neg_d, 1)
    acc_q = model['clf'].accuracy(pred, YQ)

        # YQ_d = torch.ones(query['label'].shape, dtype=torch.long).to(query['label'].device)
        # print('YQ', set(YQ.numpy()))

        # XSource, XSource_inputD, _ = model['G'](source)
        # YSource_d = torch.zeros(source['label'].shape, dtype=torch.long).to(source['label'].device)

        # XQ_logitsD = model['D'](XQ_inputD)
        # XSource_logitsD = model['D'](XSource_inputD)
        #
        # d_loss = F.cross_entropy(XQ_logitsD, YQ_d) + F.cross_entropy(XSource_logitsD, YSource_d)
        # d_loss.backward(retain_graph=True)
        # grad['D'].append(get_norm(model['D']))
        # optD.step()
        #
        # # *****************update G****************
        # optG.zero_grad()
        # XQ_logitsD = model['D'](XQ_inputD)
        # XSource_logitsD = model['D'](XSource_inputD)
        # d_loss = F.cross_entropy(XQ_logitsD, YQ_d) + F.cross_entropy(XSource_logitsD, YSource_d)
        #
        # acc, d_acc, loss, _ = model['clf'](XS, YS, XQ, YQ, XQ_logitsD, XSource_logitsD, YQ_d, YSource_d)
        #
        # g_loss = loss - d_loss
        # if args.ablation == "-DAN":
        #     g_loss = loss
        #     print("%%%%%%%%%%%%%%%%%%%This is ablation mode: -DAN%%%%%%%%%%%%%%%%%%%%%%%%%%")
        # g_loss.backward(retain_graph=True)
        # grad['G'].append(get_norm(model['G']))
        # grad['clf'].append(get_norm(model['clf']))
        # optG.step()

    return g_loss, acc_q