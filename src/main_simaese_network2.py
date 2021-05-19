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

        self.fc = nn.Linear(300, 256)
        self.cost = nn.CrossEntropyLoss()

    def forward_once(self, data):

        ebd = self.ebd(data)
        w2v = ebd

        avg_sentence_ebd = torch.mean(w2v, dim=1)
        # print("avg_sentence_ebd.shape:", avg_sentence_ebd.shape)
        avg_sentence_ebd = self.fc(avg_sentence_ebd)
        # print("avg_sentence_ebd_fc.shape:", avg_sentence_ebd.shape)
        #
        # # scale = self.compute_score(data, ebd)
        # # print("\ndata.shape:", ebd.shape)  # [b, text_len, 300]
        #
        # # Generator部分
        # ebd = self.rnn(ebd, data['text_len'])
        # # ebd, (hn, cn) = self.lstm(ebd)
        # # print("\ndata.shape:", ebd.shape)  # [b, text_len, 256]
        # # for i, b in enumerate(ebd):
        # ebd = ebd.transpose(1, 2).contiguous()  # # [b, text_len, 256] -> [b, 256, text_len]
        #
        # # [b, 256, text_len] -> [b, 256, 500]
        # if ebd.shape[2] < 500:
        #     zero = torch.zeros((ebd.shape[0], ebd.shape[1], 500-ebd.shape[2]))
        #     if self.args.cuda != -1:
        #        zero = zero.cuda(self.args.cuda)
        #     ebd = torch.cat((ebd, zero), dim=-1)
        #     # print('reverse_feature.shape[2]', ebd.shape[2])
        # else:
        #     ebd = ebd[:, :, :500]
        #     # print('reverse_feature.shape[2]', ebd.shape[2])
        #
        # ebd = self.seq(ebd).squeeze(-1)  # [b, 256, 500] -> [b, 256]
        # ebd = torch.max(ebd, dim=-1, keepdim=False)[0]
        # print("\ndata.shape:", ebd.shape)  # [b, text_len]
        # word_weight = F.softmax(ebd, dim=-1)
        # print("word_weight.shape:", word_weight.shape)  # [b, text_len]
        # sentence_ebd = torch.sum((torch.unsqueeze(word_weight, dim=-1)) * w2v, dim=-2)
        # print("sentence_ebd.shape:", sentence_ebd.shape)

        # reverse_feature = word_weight
        #
        # if reverse_feature.shape[1] < 500:
        #     zero = torch.zeros((reverse_feature.shape[0], 500-reverse_feature.shape[1]))
        #     if self.args.cuda != -1:
        #        zero = zero.cuda(self.args.cuda)
        #     reverse_feature = torch.cat((reverse_feature, zero), dim=-1)
        #     print('reverse_feature.shape[1]', reverse_feature.shape[1])
        # else:
        #     reverse_feature = reverse_feature[:, :500]
        #     print('reverse_feature.shape[1]', reverse_feature.shape[1])
        #
        # if self.args.ablation == '-IL':
        #     sentence_ebd = torch.cat((avg_sentence_ebd, sentence_ebd), 1)
        #     print("%%%%%%%%%%%%%%%%%%%%This is ablation mode: -IL%%%%%%%%%%%%%%%%%%")

        return avg_sentence_ebd

    def forward_once_with_param(self, data, param):

        ebd = self.ebd(data)
        w2v = ebd

        avg_sentence_ebd = torch.mean(w2v, dim=1)
        # print("avg_sentence_ebd.shape:", avg_sentence_ebd.shape)
        avg_sentence_ebd = F.linear(avg_sentence_ebd, param['weight'])
        # print("avg_sentence_ebd_fc.shape:", avg_sentence_ebd.shape)
        #
        # # scale = self.compute_score(data, ebd)
        # # print("\ndata.shape:", ebd.shape)  # [b, text_len, 300]
        #
        # # Generator部分
        # ebd = self.rnn(ebd, data['text_len'])
        # # ebd, (hn, cn) = self.lstm(ebd)
        # # print("\ndata.shape:", ebd.shape)  # [b, text_len, 256]
        # # for i, b in enumerate(ebd):
        # ebd = ebd.transpose(1, 2).contiguous()  # # [b, text_len, 256] -> [b, 256, text_len]
        #
        # # [b, 256, text_len] -> [b, 256, 500]
        # if ebd.shape[2] < 500:
        #     zero = torch.zeros((ebd.shape[0], ebd.shape[1], 500-ebd.shape[2]))
        #     if self.args.cuda != -1:
        #        zero = zero.cuda(self.args.cuda)
        #     ebd = torch.cat((ebd, zero), dim=-1)
        #     # print('reverse_feature.shape[2]', ebd.shape[2])
        # else:
        #     ebd = ebd[:, :, :500]
        #     # print('reverse_feature.shape[2]', ebd.shape[2])
        #
        # ebd = self.seq(ebd).squeeze(-1)  # [b, 256, 500] -> [b, 256]
        # ebd = torch.max(ebd, dim=-1, keepdim=False)[0]
        # print("\ndata.shape:", ebd.shape)  # [b, text_len]
        # word_weight = F.softmax(ebd, dim=-1)
        # print("word_weight.shape:", word_weight.shape)  # [b, text_len]
        # sentence_ebd = torch.sum((torch.unsqueeze(word_weight, dim=-1)) * w2v, dim=-2)
        # print("sentence_ebd.shape:", sentence_ebd.shape)

        # reverse_feature = word_weight
        #
        # if reverse_feature.shape[1] < 500:
        #     zero = torch.zeros((reverse_feature.shape[0], 500-reverse_feature.shape[1]))
        #     if self.args.cuda != -1:
        #        zero = zero.cuda(self.args.cuda)
        #     reverse_feature = torch.cat((reverse_feature, zero), dim=-1)
        #     print('reverse_feature.shape[1]', reverse_feature.shape[1])
        # else:
        #     reverse_feature = reverse_feature[:, :500]
        #     print('reverse_feature.shape[1]', reverse_feature.shape[1])
        #
        # if self.args.ablation == '-IL':
        #     sentence_ebd = torch.cat((avg_sentence_ebd, sentence_ebd), 1)
        #     print("%%%%%%%%%%%%%%%%%%%%This is ablation mode: -IL%%%%%%%%%%%%%%%%%%")

        return avg_sentence_ebd

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

    def loss(self, logits, label):
        loss_ce = self.cost(-logits/torch.mean(logits, dim=0), label)
        return loss_ce

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label).type(torch.FloatTensor))


class Model2(nn.Module):

    def __init__(self, ebd, args):
        super(Model2, self).__init__()

        self.args = args

        self.ebd = ebd

        self.ebd_dim = self.ebd.embedding_dim
        self.hidden_size = 128

        self.rnn = RNN(300, 128, 1, True, 0)
        self.lstm = nn.LSTM(input_size=300, hidden_size=128, num_layers=1, batch_first=True, dropout=0)

        self.seq = nn.Sequential(
            nn.Linear(500, 1),
        )

    def forward(self, data, flag=None, return_score=False):

        ebd = self.ebd(data)
        w2v = ebd

        avg_sentence_ebd = torch.mean(w2v, dim=1)
        # print("avg_sentence_ebd.shape:", avg_sentence_ebd.shape)

        # scale = self.compute_score(data, ebd)
        # print("\ndata.shape:", ebd.shape)  # [b, text_len, 300]

        # Generator部分
        ebd = self.rnn(ebd, data['text_len'])
        # ebd, (hn, cn) = self.lstm(ebd)
        # print("\ndata.shape:", ebd.shape)  # [b, text_len, 256]
        # for i, b in enumerate(ebd):
        ebd = ebd.transpose(1, 2).contiguous()  # # [b, text_len, 256] -> [b, 256, text_len]

        # [b, 256, text_len] -> [b, 256, 500]
        if ebd.shape[2] < 500:
            zero = torch.zeros((ebd.shape[0], ebd.shape[1], 500-ebd.shape[2]))
            if self.args.cuda != -1:
               zero = zero.cuda(self.args.cuda)
            ebd = torch.cat((ebd, zero), dim=-1)
            # print('reverse_feature.shape[2]', ebd.shape[2])
        else:
            ebd = ebd[:, :, :500]
            # print('reverse_feature.shape[2]', ebd.shape[2])

        ebd = self.seq(ebd).squeeze(-1)  # [b, 256, 500] -> [b, 256]
        # ebd = torch.max(ebd, dim=-1, keepdim=False)[0]
        # print("\ndata.shape:", ebd.shape)  # [b, text_len]
        # word_weight = F.softmax(ebd, dim=-1)
        # print("word_weight.shape:", word_weight.shape)  # [b, text_len]
        # sentence_ebd = torch.sum((torch.unsqueeze(word_weight, dim=-1)) * w2v, dim=-2)
        # print("sentence_ebd.shape:", sentence_ebd.shape)

        # reverse_feature = word_weight
        #
        # if reverse_feature.shape[1] < 500:
        #     zero = torch.zeros((reverse_feature.shape[0], 500-reverse_feature.shape[1]))
        #     if self.args.cuda != -1:
        #        zero = zero.cuda(self.args.cuda)
        #     reverse_feature = torch.cat((reverse_feature, zero), dim=-1)
        #     print('reverse_feature.shape[1]', reverse_feature.shape[1])
        # else:
        #     reverse_feature = reverse_feature[:, :500]
        #     print('reverse_feature.shape[1]', reverse_feature.shape[1])
        #
        # if self.args.ablation == '-IL':
        #     sentence_ebd = torch.cat((avg_sentence_ebd, sentence_ebd), 1)
        #     print("%%%%%%%%%%%%%%%%%%%%This is ablation mode: -IL%%%%%%%%%%%%%%%%%%")

        return ebd


#自定义ContrastiveLoss
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                      (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


def get_embedding_M2(vocab, args):
    print("{}, Building embedding".format(
        datetime.datetime.now()), flush=True)

    ebd = WORDEBD(vocab, args.finetune_ebd)

    model2 = Model2(ebd, args)
    # modelD = ModelD(ebd, args)

    print("{}, Building embedding".format(
        datetime.datetime.now()), flush=True)

    if args.cuda != -1:
        model2 = model2.cuda(args.cuda)
        # modelD = modelD.cuda(args.cuda)
        return model2  # , modelD
    else:
        return model2  # , modelD



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
        zero = torch.zeros((class_names_dict['text'].shape[0], support['text'].shape[1] - class_names_dict['text'].shape[1]), dtype=torch.long)
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
                support['text_len_2'] = torch.cat((support['text_len_2'], class_names_dict['text_len'][j].view(-1)), dim=0)
                support['label_2'] = torch.cat((support['label_2'], class_names_dict['label'][j].view(-1)), dim=0)
        else:
            for j in range(len(sampled_classes)):
                support['text_2'] = torch.cat((support['text_2'], class_names_dict['text'][j].view((1, -1))), dim=0)
                support['text_len_2'] = torch.cat((support['text_len_2'], class_names_dict['text_len'][j].view(-1)), dim=0)
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
    zero_grad(model['G'].parameters())
    grads = autograd.grad(loss, model['G'].fc.parameters(), allow_unused=True)
    fast_weights, orderd_params = model['G'].cloned_fc_dict(), OrderedDict()
    for (key, val), grad in zip(model['G'].fc.named_parameters(), grads):
        fast_weights[key] = orderd_params[key] = val-args.task_lr*grad
    '''steps remaining'''
    for k in range(args.train_iter - 1):
        S_out1, S_out2 = model['G'](support_1, support_2, fast_weights)
        loss = criterion(S_out1, S_out2, support['label_final'])
        zero_grad(orderd_params.values())
        grads = torch.autograd.grad(loss, orderd_params.values(), allow_unused=True)
        # print('grads:', grads)
        # print("orderd_params.items():", orderd_params.items())
        for (key, val), grad in zip(orderd_params.items(), grads):
            if grad is not None:
                fast_weights[key] = orderd_params[key] = val - args.task_lr * grad

    """计算Q上的损失"""
    CN = model['G'].forward_once_with_param(class_names_dict, fast_weights)
    XQ = model['G'].forward_once_with_param(query, fast_weights)
    logits_q = neg_dist(XQ, CN)
    q_loss = model['G'].loss(logits_q, YQ)
    _, pred = torch.max(logits_q, 1)
    acc_q = model['G'].accuracy(pred, YQ)

    optG.zero_grad()
    q_loss.backward()
    optG.step()

    # '把CN过微调过的G， S和Q过G2'
    # CN = model['G'](class_names_dict)  # CN:[N, 256(hidden_size*2)]
    # # Embedding the document
    # XS = model['G2'](support)  # XS:[N*K, 256(hidden_size*2)]
    # # print("XS:", XS.shape)
    # YS = support['label']
    # # print('YS:', YS)
    #
    # XQ = model['G2'](query)
    # YQ = query['label']
    # # print('YQ:', YQ)
    #
    # YS, YQ = reidx_y(args, YS, YQ)  # 映射标签为从0开始
    #
    # '第二步：用Support更新MLP'
    # for _ in range(args.train_iter):
    #
    #     # Embedding the document
    #     XS_mlp = model['clf'](XS)  # [N*K, 256(hidden_size*2)] -> [N*K, 256]
    #
    #     neg_d = neg_dist(XS_mlp, CN)  # [N*K, N]
    #     # print("neg_d:", neg_d.shape)
    #
    #     mlp_loss = model['clf'].loss(neg_d, YS)
    #     # print("mlp_loss:", mlp_loss)
    #
    #     optCLF.zero_grad()
    #     mlp_loss.backward(retain_graph=True)
    #     optCLF.step()
    #
    # '第三步：用Q更新G2'
    # XQ_mlp = model['clf'](XQ)
    # neg_d = neg_dist(XQ_mlp, CN)
    # q_loss = model['clf'].loss(neg_d, YQ)

    # optG2.zero_grad()
    # q_loss.backward()
    # optG2.step()
    #
    # _, pred = torch.max(neg_d, 1)
    # acc_q = model['clf'].accuracy(pred, YQ)

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

    optG = torch.optim.Adam(grad_param(model, ['G']), lr=args.meta_lr)
    # optG2 = torch.optim.Adam(grad_param(model, ['G2']), lr=args.task_lr)
    # optCLF = torch.optim.Adam(grad_param(model, ['clf']), lr=args.task_lr)

    if args.lr_scheduler == 'ReduceLROnPlateau':
        schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optG, 'max', patience=args.patience//2, factor=0.1, verbose=True)
        # schedulerCLF = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optCLF, 'max', patience=args.patience // 2, factor=0.1, verbose=True)

    elif args.lr_scheduler == 'ExponentialLR':
        schedulerG = torch.optim.lr_scheduler.ExponentialLR(optG, gamma=args.ExponentialLR_gamma)
        # schedulerCLF = torch.optim.lr_scheduler.ExponentialLR(optCLF, gamma=args.ExponentialLR_gamma)


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
        # print("sampled_classes:", sampled_classes)
        # class_names_dict = {}
        # class_names_dict['label'] = class_names['label'][sampled_classes]
        # print("class_names_dict['label']:", class_names_dict['label'])
        # class_names_dict['text'] = class_names['text'][sampled_classes]
        # class_names_dict['text_len'] = class_names['text_len'][sampled_classes]
        # class_names_dict['is_support'] = False

        # print("train_data[label]: ", train_data["label"])
        # print("sampled_classes: ", sampled_classes)
        # print("source_classes: ", source_classes)
        # print("args.train_episodes: ", args.train_episodes)
        train_gen = SerialSampler(train_data, args, sampled_classes, source_classes, args.train_episodes)
        # print("pass1")

        sampled_tasks = train_gen.get_epoch()  # 此处并非返回数据，而是生成函数迭代器！！！

        grad = {'clf': [], 'G': []}

        if not args.notqdm:
            sampled_tasks = tqdm(sampled_tasks, total=train_gen.num_episodes,
                                 ncols=80, leave=False, desc=colored('Training on train',
                                                                     'yellow'))

        for task in sampled_tasks:
            if task is None:
                break
            # print("every sampled_tasks label: ", task[0]["label"])
            # print("every sampled_tasks len: ", task[0]["text_len"])
            q_loss, q_acc = train_one(task, class_names, model, optG, criterion, args, grad)
            acc += q_acc
            loss += q_loss


        if ep % 100 == 0:
            print("--------[TRAIN] ep:" + str(ep) + ", loss:" + str(q_loss.item()) + ", acc:" + str(q_acc.item()) + "-----------")

        test_count = 500
        if (ep % test_count == 0) and (ep != 0):
            acc = acc / args.train_episodes / test_count
            loss = loss / args.train_episodes / test_count
            print("--------[TRAIN] ep:" + str(ep) + ", mean_loss:" + str(loss.item()) + ", mean_acc:" + str(acc.item()) + "-----------")

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
        zero = torch.zeros((class_names_dict['text'].shape[0], support['text'].shape[1] - class_names_dict['text'].shape[1]), dtype=torch.long)
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
                support['text_len_2'] = torch.cat((support['text_len_2'], class_names_dict['text_len'][j].view(-1)), dim=0)
                support['label_2'] = torch.cat((support['label_2'], class_names_dict['label'][j].view(-1)), dim=0)
        else:
            for j in range(len(sampled_classes)):
                support['text_2'] = torch.cat((support['text_2'], class_names_dict['text'][j].view((1, -1))), dim=0)
                support['text_len_2'] = torch.cat((support['text_len_2'], class_names_dict['text_len'][j].view(-1)), dim=0)
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
    zero_grad(model['G'].parameters())
    grads = autograd.grad(loss, model['G'].fc.parameters(), allow_unused=True)
    fast_weights, orderd_params = model['G'].cloned_fc_dict(), OrderedDict()
    for (key, val), grad in zip(model['G'].fc.named_parameters(), grads):
        fast_weights[key] = orderd_params[key] = val-args.task_lr*grad
    '''steps remaining'''
    for k in range(args.train_iter - 1):
        S_out1, S_out2 = model['G'](support_1, support_2, fast_weights)
        loss = criterion(S_out1, S_out2, support['label_final'])
        zero_grad(orderd_params.values())
        grads = torch.autograd.grad(loss, orderd_params.values(), allow_unused=True)
        # print('grads:', grads)
        # print("orderd_params.items():", orderd_params.items())
        for (key, val), grad in zip(orderd_params.items(), grads):
            if grad is not None:
                fast_weights[key] = orderd_params[key] = val - args.task_lr * grad

    """计算Q上的损失"""
    CN = model['G'].forward_once_with_param(class_names_dict, fast_weights)
    XQ = model['G'].forward_once_with_param(query, fast_weights)
    logits_q = neg_dist(XQ, CN)
    _, pred = torch.max(logits_q, 1)
    acc_q = model['G'].accuracy(pred, YQ)

    return acc_q


def test(test_data, class_names, optG, model, criterion, args, num_episodes, verbose=True):
    '''
        Evaluate the model on a bag of sampled tasks. Return the mean accuracy
        and its std.
    '''
    model['G'].train()

    acc = []
    for ep in range(num_episodes):
        if ep % 100 == 0:
            print(ep)
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

        train_gen = SerialSampler(test_data, args, sampled_classes, source_classes, args.train_episodes)

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
            q_acc = test_one(task, class_names, model, optG, criterion, args, grad)
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
    model["G"] = get_embedding(vocab, args)  # model["G"]里面 是 词向量平均 + FC

    criterion = ContrastiveLoss()
    # model["G2"] = get_embedding_M2(vocab, args)
    # model["clf"] = get_classifier(model["G"].hidden_size * 2, args)

    if args.mode == "train":
        # train model on train_data, early stopping based on val_data
        optG = train(train_data, val_data, model, class_names, criterion, args)  # 使用孪生网络，来进行maml的方法，只改变FC

    # val_acc, val_std, _ = test(val_data, model, args,
    #                                         args.val_episodes)

    test_acc, test_std = test(test_data, class_names, optG, model, criterion, args, args.test_epochs, True)

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