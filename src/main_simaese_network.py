import os
import pickle
import time
import copy
import numpy as np

from classifier.classifier_getter import get_classifier
from tools.tool import parse_args, print_args, set_seed
# from tools.visualization import Print_Attention
import dataset.loader as loader
import datetime
from embedding.wordebd import WORDEBD
import torch
import torch.nn as nn
from embedding.rnn import RNN
import torch.nn.functional as F
from train.utils import grad_param, get_norm
from dataset.sampler2 import SerialSampler, task_sampler
from dataset import utils
from tools.tool import neg_dist, reidx_y
from tqdm import tqdm
from termcolor import colored
from torch import autograd
from collections import OrderedDict


def zero_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()


def get_embedding(vocab, args):
    print("{}, Building embedding".format(
        datetime.datetime.now()), flush=True)

    ebd = WORDEBD(vocab, args.finetune_ebd)

    modelG = ModelG(ebd, args)
    # modelD = ModelD(ebd, args)

    print("{}, Building embedding".format(
        datetime.datetime.now()), flush=True)

    if args.cuda != -1:
        modelG = modelG.cuda(args.cuda)
        # modelD = modelD.cuda(args.cuda)
        return modelG  # , modelD
    else:
        return modelG  # , modelD


class ModelG(nn.Module):

    def __init__(self, ebd, args):
        super(ModelG, self).__init__()

        self.args = args

        self.ebd = ebd

        self.ebd_dim = self.ebd.embedding_dim
        self.hidden_size = 128

        # self.rnn = RNN(300, 128, 1, True, 0)
        # self.lstm = nn.LSTM(input_size=300, hidden_size=128, num_layers=1, batch_first=True, dropout=0)
        #
        # self.seq = nn.Sequential(
        #     nn.Linear(500, 1),
        # )
        # Text CNN
        ci = 1  # input chanel size
        kernel_num = args.kernel_num  # output chanel size
        kernel_size = args.kernel_size
        dropout = args.dropout
        self.conv11 = nn.Conv2d(ci, kernel_num, (kernel_size[0], self.ebd_dim))
        self.conv12 = nn.Conv2d(ci, kernel_num, (kernel_size[1], self.ebd_dim))
        self.conv13 = nn.Conv2d(ci, kernel_num, (kernel_size[2], self.ebd_dim))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_size) * kernel_num, 64)

        self.cost = nn.CrossEntropyLoss()

    def forward_once(self, data):

        ebd = self.ebd(data)  # [b, text_len, 300]
        ebd = ebd[:, :10, :]
        ebd = ebd.unsqueeze(1)  # [b, 1, text_len, 300]

        x1 = self.conv11(ebd)  # [b, kernel_num, H_out, 1]
        # print("conv11", x1.shape)
        x1 = F.relu(x1.squeeze(3))  # [b, kernel_num, H_out]
        x1 = F.max_pool1d(x1, x1.size(2)).squeeze(2)  # [batch, kernel_num]

        x2 = self.conv12(ebd)  # [b, kernel_num, H_out, 1]
        # print("conv11", x2.shape)
        x2 = F.relu(x2.squeeze(3))  # [b, kernel_num, H_out]
        x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2)  # [batch, kernel_num]

        x3 = self.conv13(ebd)  # [b, kernel_num, H_out, 1]
        # print("conv11", x3.shape)
        x3 = F.relu(x3.squeeze(3))  # [b, kernel_num, H_out]
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)  # [b, kernel_num]

        x = torch.cat((x1, x2, x3), 1)  # [b, 3 * kernel_num]
        # x = self.dropout(x)

        x = self.fc(x)  # [b, 128]
        x = self.dropout(x)

        return x

    def forward_once_with_param(self, data, param):

        ebd = self.ebd(data)  # [b, text_len, 300]
        ebd = ebd.unsqueeze(1)  # [b, 1, text_len, 300]

        w1, b1 = param['conv11']['weight'], param['conv11']['bias']
        x1 = F.conv2d(ebd, w1, b1)  # [b, kernel_num, H_out, 1]
        # print("conv11", x1.shape)
        x1 = F.relu(x1.squeeze(3))  # [b, kernel_num, H_out]
        x1 = F.max_pool1d(x1, x1.size(2)).squeeze(2)  # [batch, kernel_num]

        w2, b2 = param['conv12']['weight'], param['conv12']['bias']
        x2 = F.conv2d(ebd, w2, b2)  # [b, kernel_num, H_out, 1]
        # print("conv11", x2.shape)
        x2 = F.relu(x2.squeeze(3))  # [b, kernel_num, H_out]
        x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2)  # [batch, kernel_num]

        w3, b3 = param['conv13']['weight'], param['conv13']['bias']
        x3 = F.conv2d(ebd, w3, b3)  # [b, kernel_num, H_out, 1]
        # print("conv11", x3.shape)
        x3 = F.relu(x3.squeeze(3))  # [b, kernel_num, H_out]
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)  # [b, kernel_num]

        x = torch.cat((x1, x2, x3), 1)  # [b, 3 * kernel_num]
        # x = self.dropout(x)

        w_fc, b_fc = param['fc']['weight'], param['fc']['bias']
        x = F.linear(x, w_fc, b_fc)  # [b, 128]
        x = self.dropout(x)

        return x

    def forward(self, inputs_1, inputs_2, param=None):
        if param is None:
            out_1 = self.forward_once(inputs_1)
            out_2 = self.forward_once(inputs_2)
        else:
            out_1 = self.forward_once_with_param(inputs_1, param)
            out_2 = self.forward_once_with_param(inputs_2, param)
        return out_1, out_2

    def cloned_fc_dict(self):
        return {key: val.clone() for key, val in self.fc.state_dict().items()}

    def cloned_conv11_dict(self):
        return {key: val.clone() for key, val in self.conv11.state_dict().items()}

    def cloned_conv12_dict(self):
        return {key: val.clone() for key, val in self.conv12.state_dict().items()}

    def cloned_conv13_dict(self):
        return {key: val.clone() for key, val in self.conv13.state_dict().items()}

    def loss(self, logits, label):
        loss_ce = self.cost(-logits / torch.mean(logits, dim=1, keepdim=True), label)
        return loss_ce

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label).type(torch.FloatTensor))


# 自定义ContrastiveLoss
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        euclidean_distance = euclidean_distance / torch.mean(euclidean_distance)
        # print("**********************************************************************")
        # print("euclidean_distance:", torch.mean(euclidean_distance, dim=0))
        # euclidean_distance = euclidean_distance
        # print("euclidean_distance_after_mean:", euclidean_distance)
        tmp1 = (label) * torch.pow(euclidean_distance, 2).squeeze(-1)
        #mean_val = torch.mean(euclidean_distance)
        tmp2 = (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0),
                                       2).squeeze(-1)
        loss_contrastive = torch.mean(tmp1 + tmp2)

        # print("**********************************************************************")
        return loss_contrastive



def train_one(task, class_names, model, optG, criterion, args, grad):
    '''
        Train the model on one sampled task.
    '''
    model['G'].train()
    # model['G2'].train()
    # model['clf'].train()

    support, query = task
    # print("support, query:", support, query)
    # print("class_names_dict:", class_names_dict)

    '''分样本对'''
    YS = support['label']
    YQ = query['label']

    sampled_classes = torch.unique(support['label']).cpu().numpy().tolist()
    # print("sampled_classes:", sampled_classes)

    class_names_dict = {}
    class_names_dict['label'] = class_names['label'][sampled_classes]
    # print("class_names_dict['label']:", class_names_dict['label'])
    class_names_dict['text'] = class_names['text'][sampled_classes]
    class_names_dict['text_len'] = class_names['text_len'][sampled_classes]
    class_names_dict['is_support'] = False
    class_names_dict = utils.to_tensor(class_names_dict, args.cuda, exclude_keys=['is_support'])

    YS, YQ = reidx_y(args, YS, YQ)
    # print('YS:', support['label'])
    # print('YQ:', query['label'])
    # print("class_names_dict:", class_names_dict['label'])

    """维度填充"""
    if support['text'].shape[1] != class_names_dict['text'].shape[1]:
        zero = torch.zeros(
            (class_names_dict['text'].shape[0], support['text'].shape[1] - class_names_dict['text'].shape[1]),
            dtype=torch.long)
        class_names_dict['text'] = torch.cat((class_names_dict['text'], zero.cuda()), dim=-1)

    support['text'] = torch.cat((support['text'], class_names_dict['text']), dim=0)
    support['text_len'] = torch.cat((support['text_len'], class_names_dict['text_len']), dim=0)
    support['label'] = torch.cat((support['label'], class_names_dict['label']), dim=0)
    # print("support['text']:", support['text'].shape)
    # print("support['label']:", support['label'])

    text_sample_len = support['text'].shape[0]
    # print("support['text'].shape[0]:", support['text'].shape[0])
    support['text_1'] = support['text'][0].view((1, -1))
    support['text_len_1'] = support['text_len'][0].view(-1)
    support['label_1'] = support['label'][0].view(-1)
    for i in range(text_sample_len):
        if i == 0:
            for j in range(1, len(sampled_classes)):
                support['text_1'] = torch.cat((support['text_1'], support['text'][i].view((1, -1))), dim=0)
                support['text_len_1'] = torch.cat((support['text_len_1'], support['text_len'][i].view(-1)), dim=0)
                support['label_1'] = torch.cat((support['label_1'], support['label'][i].view(-1)), dim=0)
        else:
            for j in range(len(sampled_classes)):
                support['text_1'] = torch.cat((support['text_1'], support['text'][i].view((1, -1))), dim=0)
                support['text_len_1'] = torch.cat((support['text_len_1'], support['text_len'][i].view(-1)), dim=0)
                support['label_1'] = torch.cat((support['label_1'], support['label'][i].view(-1)), dim=0)

    support['text_2'] = class_names_dict['text'][0].view((1, -1))
    support['text_len_2'] = class_names_dict['text_len'][0].view(-1)
    support['label_2'] = class_names_dict['label'][0].view(-1)
    for i in range(text_sample_len):
        if i == 0:
            for j in range(1, len(sampled_classes)):
                support['text_2'] = torch.cat((support['text_2'], class_names_dict['text'][j].view((1, -1))), dim=0)
                support['text_len_2'] = torch.cat((support['text_len_2'], class_names_dict['text_len'][j].view(-1)),dim=0)
                support['label_2'] = torch.cat((support['label_2'], class_names_dict['label'][j].view(-1)), dim=0)
        else:
            for j in range(len(sampled_classes)):
                support['text_2'] = torch.cat((support['text_2'], class_names_dict['text'][j].view((1, -1))), dim=0)
                support['text_len_2'] = torch.cat((support['text_len_2'], class_names_dict['text_len'][j].view(-1)),dim=0)
                support['label_2'] = torch.cat((support['label_2'], class_names_dict['label'][j].view(-1)), dim=0)

    # print("support['text_1']:", support['text_1'].shape, support['text_len_1'].shape, support['label_1'].shape)
    # print("support['text_2']:", support['text_2'].shape, support['text_len_2'].shape, support['label_2'].shape)
    support['label_final'] = support['label_1'].eq(support['label_2']).int()

    support_1 = {}
    support_1['text'] = support['text_1']
    support_1['text_len'] = support['text_len_1']
    support_1['label'] = support['label_1']

    support_2 = {}
    support_2['text'] = support['text_2']
    support_2['text_len'] = support['text_len_2']
    support_2['label'] = support['label_2']
    # print("**************************************")
    # print("1111111", support['label_1'])
    # print("2222222", support['label_2'])
    # print(support['label_final'])

    '''first step'''
    S_out1, S_out2 = model['G'](support_1, support_2)
    # print("-------0S1_2:", S_out1.shape, S_out2.shape)
    loss = criterion(S_out1, S_out2, support['label_final'])
    # print("s_1_loss:", loss)
    zero_grad(model['G'].parameters())

    grads_fc = autograd.grad(loss, model['G'].fc.parameters(), allow_unused=True, retain_graph=True)
    fast_weights_fc, orderd_params_fc = model['G'].cloned_fc_dict(), OrderedDict()
    for (key, val), grad in zip(model['G'].fc.named_parameters(), grads_fc):
        fast_weights_fc[key] = orderd_params_fc[key] = val - args.task_lr * grad

    grads_conv11 = autograd.grad(loss, model['G'].conv11.parameters(), allow_unused=True, retain_graph=True)
    fast_weights_conv11, orderd_params_conv11 = model['G'].cloned_conv11_dict(), OrderedDict()
    for (key, val), grad in zip(model['G'].conv11.named_parameters(), grads_conv11):
        fast_weights_conv11[key] = orderd_params_conv11[key] = val - args.task_lr * grad

    grads_conv12 = autograd.grad(loss, model['G'].conv12.parameters(), allow_unused=True, retain_graph=True)
    fast_weights_conv12, orderd_params_conv12 = model['G'].cloned_conv12_dict(), OrderedDict()
    for (key, val), grad in zip(model['G'].conv12.named_parameters(), grads_conv12):
        fast_weights_conv12[key] = orderd_params_conv12[key] = val - args.task_lr * grad

    grads_conv13 = autograd.grad(loss, model['G'].conv13.parameters(), allow_unused=True)
    fast_weights_conv13, orderd_params_conv13 = model['G'].cloned_conv13_dict(), OrderedDict()
    for (key, val), grad in zip(model['G'].conv13.named_parameters(), grads_conv13):
        fast_weights_conv13[key] = orderd_params_conv13[key] = val - args.task_lr * grad

    fast_weights = {}
    fast_weights['fc'] = fast_weights_fc
    fast_weights['conv11'] = fast_weights_conv11
    fast_weights['conv12'] = fast_weights_conv12
    fast_weights['conv13'] = fast_weights_conv13

    '''steps remaining'''
    for k in range(args.train_iter - 1):
        S_out1, S_out2 = model['G'](support_1, support_2, fast_weights)
        # print("-------1S1_2:", S_out1, S_out2)
        loss = criterion(S_out1, S_out2, support['label_final'])
        # print("train_iter: {} s_loss:{}".format(k, loss))
        zero_grad(orderd_params_fc.values())
        zero_grad(orderd_params_conv11.values())
        zero_grad(orderd_params_conv12.values())
        zero_grad(orderd_params_conv13.values())
        grads_fc = torch.autograd.grad(loss, orderd_params_fc.values(), allow_unused=True, retain_graph=True)
        grads_conv11 = torch.autograd.grad(loss, orderd_params_conv11.values(), allow_unused=True, retain_graph=True)
        grads_conv12 = torch.autograd.grad(loss, orderd_params_conv12.values(), allow_unused=True, retain_graph=True)
        grads_conv13 = torch.autograd.grad(loss, orderd_params_conv13.values(), allow_unused=True)
        # print('grads:', grads)
        # print("orderd_params.items():", orderd_params.items())
        for (key, val), grad in zip(orderd_params_fc.items(), grads_fc):
            if grad is not None:
                fast_weights['fc'][key] = orderd_params_fc[key] = val - args.task_lr * grad

        for (key, val), grad in zip(orderd_params_conv11.items(), grads_conv11):
            if grad is not None:
                fast_weights['conv11'][key] = orderd_params_conv11[key] = val - args.task_lr * grad

        for (key, val), grad in zip(orderd_params_conv12.items(), grads_conv12):
            if grad is not None:
                fast_weights['conv12'][key] = orderd_params_conv12[key] = val - args.task_lr * grad

        for (key, val), grad in zip(orderd_params_conv13.items(), grads_conv13):
            if grad is not None:
                fast_weights['conv13'][key] = orderd_params_conv13[key] = val - args.task_lr * grad

    """计算Q上的损失"""
    CN = model['G'].forward_once_with_param(class_names_dict, fast_weights)
    XQ = model['G'].forward_once_with_param(query, fast_weights)
    logits_q = neg_dist(XQ, CN)
    # print("logits_q:", logits_q)
    q_loss = model['G'].loss(logits_q, YQ)
    # print("q_loss:", q_loss)
    _, pred = torch.max(logits_q, 1)
    acc_q = model['G'].accuracy(pred, YQ)

    # optG.zero_grad()
    # q_loss.backward()
    # optG.step()

    return q_loss, acc_q


def train(train_data, val_data, model, class_names, criterion, args):
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

    optG = torch.optim.Adam(grad_param(model, ['G']), lr=args.meta_lr, weight_decay=args.weight_decay)
    # optG2 = torch.optim.Adam(grad_param(model, ['G2']), lr=args.task_lr)
    # optCLF = torch.optim.Adam(grad_param(model, ['clf']), lr=args.task_lr)

    if args.lr_scheduler == 'ReduceLROnPlateau':
        schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optG, 'max', patience=args.patience // 2, factor=0.1, verbose=True)
        # schedulerCLF = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optCLF, 'max', patience=args.patience // 2, factor=0.1, verbose=True)

    elif args.lr_scheduler == 'ExponentialLR':
        schedulerG = torch.optim.lr_scheduler.ExponentialLR(optG, gamma=args.ExponentialLR_gamma)
        # schedulerCLF = torch.optim.lr_scheduler.ExponentialLR(optCLF, gamma=args.ExponentialLR_gamma)

    print("{}, Start training".format(
        datetime.datetime.now()), flush=True)


    # sampled_classes, source_classes = task_sampler(train_data, args)
    acc = 0
    loss = 0
    for ep in range(args.train_epochs):
        ep_loss = 0
        for _ in range(args.train_episodes):

            sampled_classes, source_classes = task_sampler(train_data, args)

            train_gen = SerialSampler(train_data, args, sampled_classes, source_classes, 1)

            sampled_tasks = train_gen.get_epoch()

            grad = {'clf': [], 'G': []}

            if not args.notqdm:
                sampled_tasks = tqdm(sampled_tasks, total=train_gen.num_episodes,
                                     ncols=80, leave=False, desc=colored('Training on train',
                                                                         'yellow'))

            for task in sampled_tasks:
                if task is None:
                    break
                q_loss, q_acc = train_one(task, class_names, model, optG, criterion, args, grad)
                acc += q_acc
                loss = loss + q_loss
                ep_loss = ep_loss + q_loss

        ep_loss = ep_loss / args.train_episodes

        optG.zero_grad()
        ep_loss.backward()
        optG.step()

        if ep % 100 == 0:
            print("{}:".format(colored('--------[TRAIN] ep', 'blue')) + str(ep) + ", loss:" + str(q_loss.item()) + ", acc:" + str(
                q_acc.item()) + "-----------")

        test_count = 500
        if (ep % test_count == 0) and (ep != 0):
            acc = acc / args.train_episodes / test_count
            loss = loss / args.train_episodes / test_count
            print("{}:".format(colored('--------[TRAIN] ep', 'blue')) + str(ep) + ", mean_loss:" + str(loss.item()) + ", mean_acc:" + str(
                acc.item()) + "-----------")

            net = copy.deepcopy(model)
            # acc, std = test(train_data, class_names, optG, net, criterion, args, args.test_epochs, False)
            # print("[TRAIN] {}, {:s} {:2d}, {:s} {:s}{:>7.4f} ± {:>6.4f} ".format(
            #     datetime.datetime.now(),
            #     "ep", ep,
            #     colored("train", "red"),
            #     colored("acc:", "blue"), acc, std,
            #     ), flush=True)
            acc = 0
            loss = 0

            # Evaluate validation accuracy
            cur_acc, cur_std = test(val_data, class_names, optG, net, criterion, args, args.test_epochs, False)
            print(("[EVAL] {}, {:s} {:2d}, {:s} {:s}{:>7.4f} ± {:>6.4f}, "
                   ).format(
                datetime.datetime.now(),
                "ep", ep,
                colored("val  ", "cyan"),
                colored("acc:", "blue"), cur_acc, cur_std,
                # colored("train stats", "cyan"),
                # colored("G_grad:", "blue"), np.mean(np.array(grad['G'])),
                # colored("clf_grad:", "blue"), np.mean(np.array(grad['clf'])),
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
                # torch.save(model['G2'].state_dict(), best_path + '.G2')
                # torch.save(model['clf'].state_dict(), best_path + '.clf')

                sub_cycle = 0
            else:
                sub_cycle += 1

            # Break if the val acc hasn't improved in the past patience epochs
            if sub_cycle == args.patience:
                break

            if args.lr_scheduler == 'ReduceLROnPlateau':
                schedulerG.step(cur_acc)
                # schedulerCLF.step(cur_acc)

            elif args.lr_scheduler == 'ExponentialLR':
                schedulerG.step()
                # schedulerCLF.step()

    print("{}, End of training. Restore the best weights".format(
        datetime.datetime.now()),
        flush=True)

    # restore the best saved model
    model['G'].load_state_dict(torch.load(best_path + '.G'))
    # model['G2'].load_state_dict(torch.load(best_path + '.G2'))
    # model['clf'].load_state_dict(torch.load(best_path + '.clf'))

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
        # torch.save(model['clf'].state_dict(), best_path + '.clf')

        with open(best_path + '_args.txt', 'w') as f:
            for attr, value in sorted(args.__dict__.items()):
                f.write("{}={}\n".format(attr, value))

    return optG


def test_one(task, class_names, model, optG, criterion, args, grad):
    '''
        Train the model on one sampled task.
    '''
    model['G'].eval()

    support, query = task
    # print("support, query:", support, query)
    # print("class_names_dict:", class_names_dict)

    '''分样本对'''
    YS = support['label']
    YQ = query['label']

    sampled_classes = torch.unique(support['label']).cpu().numpy().tolist()
    # print("sampled_classes:", sampled_classes)

    class_names_dict = {}
    class_names_dict['label'] = class_names['label'][sampled_classes]
    # print("class_names_dict['label']:", class_names_dict['label'])
    class_names_dict['text'] = class_names['text'][sampled_classes]
    class_names_dict['text_len'] = class_names['text_len'][sampled_classes]
    class_names_dict['is_support'] = False
    class_names_dict = utils.to_tensor(class_names_dict, args.cuda, exclude_keys=['is_support'])

    YS, YQ = reidx_y(args, YS, YQ)
    # print('YS:', support['label'])
    # print('YQ:', query['label'])
    # print("class_names_dict:", class_names_dict['label'])

    """维度填充"""
    if support['text'].shape[1] != class_names_dict['text'].shape[1]:
        zero = torch.zeros(
            (class_names_dict['text'].shape[0], support['text'].shape[1] - class_names_dict['text'].shape[1]),
            dtype=torch.long)
        class_names_dict['text'] = torch.cat((class_names_dict['text'], zero.cuda()), dim=-1)

    support['text'] = torch.cat((support['text'], class_names_dict['text']), dim=0)
    support['text_len'] = torch.cat((support['text_len'], class_names_dict['text_len']), dim=0)
    support['label'] = torch.cat((support['label'], class_names_dict['label']), dim=0)
    # print("support['text']:", support['text'].shape)
    # print("support['label']:", support['label'])

    text_sample_len = support['text'].shape[0]
    # print("support['text'].shape[0]:", support['text'].shape[0])
    support['text_1'] = support['text'][0].view((1, -1))
    support['text_len_1'] = support['text_len'][0].view(-1)
    support['label_1'] = support['label'][0].view(-1)
    for i in range(text_sample_len):
        if i == 0:
            for j in range(1, len(sampled_classes)):
                support['text_1'] = torch.cat((support['text_1'], support['text'][i].view((1, -1))), dim=0)
                support['text_len_1'] = torch.cat((support['text_len_1'], support['text_len'][i].view(-1)), dim=0)
                support['label_1'] = torch.cat((support['label_1'], support['label'][i].view(-1)), dim=0)
        else:
            for j in range(len(sampled_classes)):
                support['text_1'] = torch.cat((support['text_1'], support['text'][i].view((1, -1))), dim=0)
                support['text_len_1'] = torch.cat((support['text_len_1'], support['text_len'][i].view(-1)), dim=0)
                support['label_1'] = torch.cat((support['label_1'], support['label'][i].view(-1)), dim=0)

    support['text_2'] = class_names_dict['text'][0].view((1, -1))
    support['text_len_2'] = class_names_dict['text_len'][0].view(-1)
    support['label_2'] = class_names_dict['label'][0].view(-1)
    for i in range(text_sample_len):
        if i == 0:
            for j in range(1, len(sampled_classes)):
                support['text_2'] = torch.cat((support['text_2'], class_names_dict['text'][j].view((1, -1))), dim=0)
                support['text_len_2'] = torch.cat((support['text_len_2'], class_names_dict['text_len'][j].view(-1)),dim=0)
                support['label_2'] = torch.cat((support['label_2'], class_names_dict['label'][j].view(-1)), dim=0)
        else:
            for j in range(len(sampled_classes)):
                support['text_2'] = torch.cat((support['text_2'], class_names_dict['text'][j].view((1, -1))), dim=0)
                support['text_len_2'] = torch.cat((support['text_len_2'], class_names_dict['text_len'][j].view(-1)),dim=0)
                support['label_2'] = torch.cat((support['label_2'], class_names_dict['label'][j].view(-1)), dim=0)

    # print("support['text_1']:", support['text_1'].shape, support['text_len_1'].shape, support['label_1'].shape)
    # print("support['text_2']:", support['text_2'].shape, support['text_len_2'].shape, support['label_2'].shape)
    support['label_final'] = support['label_1'].eq(support['label_2']).int()

    support_1 = {}
    support_1['text'] = support['text_1']
    support_1['text_len'] = support['text_len_1']
    support_1['label'] = support['label_1']

    support_2 = {}
    support_2['text'] = support['text_2']
    support_2['text_len'] = support['text_len_2']
    support_2['label'] = support['label_2']
    # print("**************************************")
    # print("1111111", support['label_1'])
    # print("2222222", support['label_2'])
    # print(support['label_final'])

    '''first step'''
    S_out1, S_out2 = model['G'](support_1, support_2)
    loss = criterion(S_out1, S_out2, support['label_final'])
    # print("s_1_loss:", loss)
    zero_grad(model['G'].parameters())

    grads_fc = autograd.grad(loss, model['G'].fc.parameters(), allow_unused=True, retain_graph=True)
    fast_weights_fc, orderd_params_fc = model['G'].cloned_fc_dict(), OrderedDict()
    for (key, val), grad in zip(model['G'].fc.named_parameters(), grads_fc):
        fast_weights_fc[key] = orderd_params_fc[key] = val - args.task_lr * grad

    grads_conv11 = autograd.grad(loss, model['G'].conv11.parameters(), allow_unused=True, retain_graph=True)
    fast_weights_conv11, orderd_params_conv11 = model['G'].cloned_conv11_dict(), OrderedDict()
    for (key, val), grad in zip(model['G'].conv11.named_parameters(), grads_conv11):
        fast_weights_conv11[key] = orderd_params_conv11[key] = val - args.task_lr * grad

    grads_conv12 = autograd.grad(loss, model['G'].conv12.parameters(), allow_unused=True, retain_graph=True)
    fast_weights_conv12, orderd_params_conv12 = model['G'].cloned_conv12_dict(), OrderedDict()
    for (key, val), grad in zip(model['G'].conv12.named_parameters(), grads_conv12):
        fast_weights_conv12[key] = orderd_params_conv12[key] = val - args.task_lr * grad

    grads_conv13 = autograd.grad(loss, model['G'].conv13.parameters(), allow_unused=True)
    fast_weights_conv13, orderd_params_conv13 = model['G'].cloned_conv13_dict(), OrderedDict()
    for (key, val), grad in zip(model['G'].conv13.named_parameters(), grads_conv13):
        fast_weights_conv13[key] = orderd_params_conv13[key] = val - args.task_lr * grad


    fast_weights = {}
    fast_weights['fc'] = fast_weights_fc
    fast_weights['conv11'] = fast_weights_conv11
    fast_weights['conv12'] = fast_weights_conv12
    fast_weights['conv13'] = fast_weights_conv13

    '''steps remaining'''
    for k in range(args.test_iter - 1):
        S_out1, S_out2 = model['G'](support_1, support_2, fast_weights)
        loss = criterion(S_out1, S_out2, support['label_final'])
        # print("train_iter: {} s_loss:{}".format(k, loss))
        zero_grad(orderd_params_fc.values())
        zero_grad(orderd_params_conv11.values())
        zero_grad(orderd_params_conv12.values())
        zero_grad(orderd_params_conv13.values())
        grads_fc = torch.autograd.grad(loss, orderd_params_fc.values(), allow_unused=True, retain_graph=True)
        grads_conv11 = torch.autograd.grad(loss, orderd_params_conv11.values(), allow_unused=True, retain_graph=True)
        grads_conv12 = torch.autograd.grad(loss, orderd_params_conv12.values(), allow_unused=True, retain_graph=True)
        grads_conv13 = torch.autograd.grad(loss, orderd_params_conv13.values(), allow_unused=True)

        for (key, val), grad in zip(orderd_params_fc.items(), grads_fc):
            if grad is not None:
                fast_weights['fc'][key] = orderd_params_fc[key] = val - args.task_lr * grad

        for (key, val), grad in zip(orderd_params_conv11.items(), grads_conv11):
            if grad is not None:
                fast_weights['conv11'][key] = orderd_params_conv11[key] = val - args.task_lr * grad

        for (key, val), grad in zip(orderd_params_conv12.items(), grads_conv12):
            if grad is not None:
                fast_weights['conv12'][key] = orderd_params_conv12[key] = val - args.task_lr * grad

        for (key, val), grad in zip(orderd_params_conv13.items(), grads_conv13):
            if grad is not None:
                fast_weights['conv13'][key] = orderd_params_conv13[key] = val - args.task_lr * grad

    """计算Q上的损失"""
    CN = model['G'].forward_once_with_param(class_names_dict, fast_weights)
    XQ = model['G'].forward_once_with_param(query, fast_weights)
    logits_q = neg_dist(XQ, CN)
    _, pred = torch.max(logits_q, 1)
    acc_q = model['G'].accuracy(pred, YQ)

    return acc_q


def test(test_data, class_names, optG, model, criterion, args, test_epoch, verbose=True):
    '''
        Evaluate the model on a bag of sampled tasks. Return the mean accuracy
        and its std.
    '''
    # model['G'].train()

    acc = []
    for ep in range(test_epoch):

        sampled_classes, source_classes = task_sampler(test_data, args)

        train_gen = SerialSampler(test_data, args, sampled_classes, source_classes, 1)

        sampled_tasks = train_gen.get_epoch()

        for task in sampled_tasks:
            if task is None:
                break
            q_acc = test_one(task, class_names, model, optG, criterion, args, grad={})
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


def main():
    args = parse_args()

    print_args(args)

    set_seed(args.seed)

    # load data
    train_data, val_data, test_data, class_names, vocab = loader.load_dataset(args)

    args.id2word = vocab.itos

    # initialize model
    model = {}
    model["G"] = get_embedding(vocab, args)
    print("-------------------------------------param----------------------------------------------")
    sum = 0
    for name, param in model["G"].named_parameters():
        num = 1
        for size in param.shape:
            num *= size
        sum += num
        print("{:30s} : {}".format(name, param.shape))
    print("total param num {}".format(sum))
    print("-------------------------------------param----------------------------------------------")

    criterion = ContrastiveLoss()
    # model["G2"] = get_embedding_M2(vocab, args)
    # model["clf"] = get_classifier(model["G"].hidden_size * 2, args)

    if args.mode == "train":
        # train model on train_data, early stopping based on val_data
        optG = train(train_data, val_data, model, class_names, criterion, args)

    # val_acc, val_std, _ = test(val_data, model, args,
    #                                         args.val_episodes)

    test_acc, test_std = test(test_data, class_names, optG, model, criterion, args, args.test_epochs, False)
    print(("[TEST] {}, {:s} {:s}{:>7.4f} ± {:>6.4f}, "
           ).format(
        datetime.datetime.now(),
        colored("test  ", "cyan"),
        colored("acc:", "blue"), test_acc, test_std,
        # colored("train stats", "cyan"),
        # colored("G_grad:", "blue"), np.mean(np.array(grad['G'])),
        # colored("clf_grad:", "blue"), np.mean(np.array(grad['clf'])),
    ), flush=True)

    # path_drawn = args.path_drawn_data
    # with open(path_drawn, 'w') as f_w:
    #     json.dump(drawn_data, f_w)
    #     print("store drawn data finished.")

    # file_path = r'../data/attention_data.json'
    # Print_Attention(file_path, vocab, model, args)

    if args.result_path:
        directory = args.result_path[:args.result_path.rfind("/")]
        if not os.path.exists(directory):
            os.mkdirs(directory)

        result = {
            "test_acc": test_acc,
            "test_std": test_std,
            # "val_acc": val_acc,
            # "val_std": val_std
        }

        for attr, value in sorted(args.__dict__.items()):
            result[attr] = value

        with open(args.result_path, "wb") as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()