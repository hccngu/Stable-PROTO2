import datetime

import numpy as np
import torch
from termcolor import colored
from tqdm import tqdm
import copy

from dataset.sampler import ParallelSampler_Test
from dataset.sampler import ParallelSampler, ParallelSampler_Test, task_sampler
from dataset import utils
from tools.tool import reidx_y, neg_dist


def test_one(task, class_names, model, optCLF, args, grad):
    '''
        Train the model on one sampled task.
    '''
    # model['G'].eval()
    # model['clf'].train()

    support, query = task
    # print("support, query:", support, query)
    # print("class_names_dict:", class_names_dict)

    sampled_classes = torch.unique(support['label']).cpu().numpy().tolist()
    # print("sampled_classes:", sampled_classes)

    class_names_dict = {}
    class_names_dict['label'] = class_names['label'][sampled_classes]
    # print("class_names_dict['label']:", class_names_dict['label'])
    class_names_dict['text'] = class_names['text'][sampled_classes]
    class_names_dict['text_len'] = class_names['text_len'][sampled_classes]
    class_names_dict['is_support'] = False
    class_names_dict = utils.to_tensor(class_names_dict, args.cuda, exclude_keys=['is_support'])

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

    for _ in range(args.test_iter):

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

    _, pred = torch.max(neg_d, 1)
    acc_q = model['clf'].accuracy(pred, YQ)

    return acc_q


def test(test_data, class_names, optCLF, model, args, num_episodes, verbose=True):
    '''
        Evaluate the model on a bag of sampled tasks. Return the mean accuracy
        and its std.
    '''
    model['G'].train()
    model['clf'].train()

    acc = []
    for ep in range(num_episodes):
        # if args.embedding == 'mlada':
        #     acc1, d_acc1, sentence_ebd, avg_sentence_ebd, sentence_label, word_weight, query_data, x_hat = test_one(task, model, args)
        #     if count < 20:
        #         if all_sentence_ebd is None:
        #             all_sentence_ebd = sentence_ebd
        #             all_avg_sentence_ebd = avg_sentence_ebd
        #             all_sentence_label = sentence_label
        #             all_word_weight = word_weight
        #             all_query_data = query_data
        #             all_x_hat = x_hat
        #         else:
        #             all_sentence_ebd = np.concatenate((all_sentence_ebd, sentence_ebd), 0)
        #             all_avg_sentence_ebd = np.concatenate((all_avg_sentence_ebd, avg_sentence_ebd), 0)
        #             all_sentence_label = np.concatenate((all_sentence_label, sentence_label))
        #             all_word_weight = np.concatenate((all_word_weight, word_weight), 0)
        #             all_query_data = np.concatenate((all_query_data, query_data), 0)
        #             all_x_hat = np.concatenate((all_x_hat, x_hat), 0)
        #     count = count + 1
        #     acc.append(acc1)
        #     d_acc.append(d_acc1)
        # else:
        #     acc.append(test_one(task, model, args))
        sampled_classes, source_classes = task_sampler(test_data, args)
        # class_names_dict = {}
        # class_names_dict['label'] = class_names['label'][sampled_classes]
        # class_names_dict['text'] = class_names['text'][sampled_classes]
        # class_names_dict['text_len'] = class_names['text_len'][sampled_classes]
        # class_names_dict['is_support'] = False

        train_gen = ParallelSampler(test_data, args, sampled_classes, source_classes, args.train_episodes)

        sampled_tasks = train_gen.get_epoch()
        # class_names_dict = utils.to_tensor(class_names_dict, args.cuda, exclude_keys=['is_support'])

        grad = {'clf': [], 'G': []}

        if not args.notqdm:
            sampled_tasks = tqdm(sampled_tasks, total=train_gen.num_episodes,
                                 ncols=80, leave=False, desc=colored('Training on train',
                                                                     'yellow'))

        for task in sampled_tasks:
            if task is None:
                break
            q_acc = test_one(task, class_names, model, optCLF, args, grad)
            acc.append(q_acc.cpu().item())

    acc = np.array(acc)

    if verbose:
        if args.embedding != 'mlada':
            print("{}, {:s} {:>7.4f}, {:s} {:>7.4f}".format(
                datetime.datetime.now(),
                colored("test acc mean", "blue"),
                np.mean(acc),
                colored("test std", "blue"),
                np.std(acc),
                ), flush=True)
        else:
            print("{}, {:s} {:>7.4f}, {:s} {:>7.4f}".format(
                datetime.datetime.now(),
                colored("test acc mean", "blue"),
                np.mean(acc),
                colored("test std", "blue"),
                np.std(acc),
            ), flush=True)

    return np.mean(acc), np.std(acc)