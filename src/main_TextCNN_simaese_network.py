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
from torch import optim
import torch.nn as nn
from embedding.rnn import RNN
import torch.nn.functional as F
from train.utils import grad_param, get_norm
from dataset.sampler import ParallelSampler, ParallelSampler_Test, task_sampler
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

    model_TextCNN = Model_TextCNN(ebd, args)
    # modelD = ModelD(ebd, args)

    print("{}, Building embedding".format(
        datetime.datetime.now()), flush=True)

    if args.cuda != -1:
        model_TextCNN = model_TextCNN.cuda(args.cuda)
        # modelD = modelD.cuda(args.cuda)
        return model_TextCNN  # , modelD
    else:
        return model_TextCNN  # , modelD


class Model_TextCNN(nn.Module):

    def __init__(self, ebd, args, config):
        super(Model_TextCNN, self).__init__()

        self.args = args

        self.ebd = ebd

        # self.ebd_dim = self.ebd.embedding_dim
        # self.hidden_size = 128

        # self.rnn = RNN(300, 128, 1, True, 0)
        # self.lstm = nn.LSTM(input_size=300, hidden_size=128, num_layers=1, batch_first=True, dropout=0)
        #
        # self.seq = nn.Sequential(
        #     nn.Linear(500, 1),
        # )

        # self.fc = nn.Linear(300, 256)
        # self.cost = nn.CrossEntropyLoss()

        self.config = config

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()

        for i, (name, param) in enumerate(self.config):
            if name is 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            # elif name is 'convt2d':
            #     # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
            #     w = nn.Parameter(torch.ones(*param[:4]))
            #     # gain=1 according to cbfin's implementation
            #     torch.nn.init.kaiming_normal_(w)
            #     self.vars.append(w)
            #     # [ch_in, ch_out]
            #     self.vars.append(nn.Parameter(torch.zeros(param[1])))

            # elif name is 'linear':
            #     # [ch_out, ch_in]
            #     w = nn.Parameter(torch.ones(*param))
            #     # gain=1 according to cbfinn's implementation
            #     torch.nn.init.kaiming_normal_(w)
            #     self.vars.append(w)
            #     # [ch_out]
            #     self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])


            elif name in ['tanh', 'relu', 'upsample', 'avg_pool1d', 'max_pool1d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError

    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'

            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'


            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info

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
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


def train_one(task, class_names, model, opt_TextCNN, criterion, args, grad):
    '''
        Train the model on one sampled task.
    '''
    model['TextCNN'].train()
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
            for j in range(1, text_sample_len):
                support['text_1'] = torch.cat((support['text_1'], support['text'][i].view((1, -1))), dim=0)
                support['text_len_1'] = torch.cat((support['text_len_1'], support['text_len'][i].view(-1)), dim=0)
                support['label_1'] = torch.cat((support['label_1'], support['label'][i].view(-1)), dim=0)
        else:
            for j in range(text_sample_len):
                support['text_1'] = torch.cat((support['text_1'], support['text'][i].view((1, -1))), dim=0)
                support['text_len_1'] = torch.cat((support['text_len_1'], support['text_len'][i].view(-1)), dim=0)
                support['label_1'] = torch.cat((support['label_1'], support['label'][i].view(-1)), dim=0)

    support['text_2'] = support['text'][0].view((1, -1))
    support['text_len_2'] = support['text_len'][0].view(-1)
    support['label_2'] = support['label'][0].view(-1)
    for i in range(text_sample_len):
        if i == 0:
            for j in range(1, text_sample_len):
                support['text_2'] = torch.cat((support['text_2'], support['text'][j].view((1, -1))), dim=0)
                support['text_len_2'] = torch.cat((support['text_len_2'], support['text_len'][j].view(-1)), dim=0)
                support['label_2'] = torch.cat((support['label_2'], support['label'][j].view(-1)), dim=0)
        else:
            for j in range(text_sample_len):
                support['text_2'] = torch.cat((support['text_2'], support['text'][j].view((1, -1))), dim=0)
                support['text_len_2'] = torch.cat((support['text_len_2'], support['text_len'][j].view(-1)), dim=0)
                support['label_2'] = torch.cat((support['label_2'], support['label'][j].view(-1)), dim=0)

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
    S_out1, S_out2 = model['TextCNN'](support_1, support_2)
    loss = criterion(S_out1, S_out2, support['label_final'])
    zero_grad(model['TextCNN'].parameters())
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

    opt_TextCNN = torch.optim.Adam(grad_param(model, ['TextCNN']), lr=args.meta_lr)
    # optG2 = torch.optim.Adam(grad_param(model, ['G2']), lr=args.task_lr)
    # optCLF = torch.optim.Adam(grad_param(model, ['clf']), lr=args.task_lr)

    if args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler_TextCNN = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt_TextCNN, 'max', patience=args.patience//2, factor=0.1, verbose=True)
        # schedulerCLF = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optCLF, 'max', patience=args.patience // 2, factor=0.1, verbose=True)

    elif args.lr_scheduler == 'ExponentialLR':
        scheduler_TextCNN = torch.optim.lr_scheduler.ExponentialLR(opt_TextCNN, gamma=args.ExponentialLR_gamma)
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


        train_gen = ParallelSampler(train_data, args, sampled_classes, source_classes, args.train_episodes)

        sampled_tasks = train_gen.get_epoch()

        grad = {'clf': [], 'G': []}

        if not args.notqdm:
            sampled_tasks = tqdm(sampled_tasks, total=train_gen.num_episodes,
                    ncols=80, leave=False, desc=colored('Training on train',
                        'yellow'))

        for task in sampled_tasks:
            if task is None:
                break
            q_loss, q_acc = train_one(task, class_names, model, opt_TextCNN, criterion, args, grad)
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
            for j in range(1, text_sample_len):
                support['text_1'] = torch.cat((support['text_1'], support['text'][i].view((1, -1))), dim=0)
                support['text_len_1'] = torch.cat((support['text_len_1'], support['text_len'][i].view(-1)), dim=0)
                support['label_1'] = torch.cat((support['label_1'], support['label'][i].view(-1)), dim=0)
        else:
            for j in range(text_sample_len):
                support['text_1'] = torch.cat((support['text_1'], support['text'][i].view((1, -1))), dim=0)
                support['text_len_1'] = torch.cat((support['text_len_1'], support['text_len'][i].view(-1)), dim=0)
                support['label_1'] = torch.cat((support['label_1'], support['label'][i].view(-1)), dim=0)

    support['text_2'] = support['text'][0].view((1, -1))
    support['text_len_2'] = support['text_len'][0].view(-1)
    support['label_2'] = support['label'][0].view(-1)
    for i in range(text_sample_len):
        if i == 0:
            for j in range(1, text_sample_len):
                support['text_2'] = torch.cat((support['text_2'], support['text'][j].view((1, -1))), dim=0)
                support['text_len_2'] = torch.cat((support['text_len_2'], support['text_len'][j].view(-1)), dim=0)
                support['label_2'] = torch.cat((support['label_2'], support['label'][j].view(-1)), dim=0)
        else:
            for j in range(text_sample_len):
                support['text_2'] = torch.cat((support['text_2'], support['text'][j].view((1, -1))), dim=0)
                support['text_len_2'] = torch.cat((support['text_len_2'], support['text_len'][j].view(-1)), dim=0)
                support['label_2'] = torch.cat((support['label_2'], support['label'][j].view(-1)), dim=0)

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

    config = [
        ('conv2d', [16, 1, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [16]),
        ('max_pool1d', [16]),  ########
        ('conv2d', [16, 1, 4, 4, 1, 0]),
        ('relu', [True]),
        ('bn', [16]),
        ('max_pool1d', [16]),  ########
        ('conv2d', [16, 1, 5, 5, 1, 0]),
        ('relu', [True]),
        ('bn', [16]),
        ('max_pool1d', [16]),
        # ('conv2d', [32, 32, 3, 3, 1, 0]),
        # ('relu', [True]),
        # ('bn', [32]),
        # ('max_pool2d', [2, 1, 0]),
        ('flatten', []),
        ('linear', [args.n_way, 32 * 5 * 5])
    ]

    set_seed(args.seed)

    # load data
    train_data, val_data, test_data, class_names, vocab = loader.load_dataset(args)

    args.id2word = vocab.itos

    # initialize model
    model = {}
    model["TextCNN"] = get_embedding(vocab, args)

    criterion = ContrastiveLoss()
    # model["G2"] = get_embedding_M2(vocab, args)
    # model["clf"] = get_classifier(model["G"].hidden_size * 2, args)

    if args.mode == "train":
        # train model on train_data, early stopping based on val_data
        opt_TextCNN = train(train_data, val_data, model, class_names, criterion, args)

    # val_acc, val_std, _ = test(val_data, model, args,
    #                                         args.val_episodes)

    test_acc, test_std = test(test_data, class_names, optG, model, criterion, args, args.test_epochs, False)

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


class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.task_lr = args.task_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.ways
        self.k_spt = args.shot
        self.k_qry = args.query
        self.task_num = args.task_num
        self.train_iter = args.train_iter
        self.test_iter = args.test_iter

        self.net = Learner(config, args.imgc, args.imgsz)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]


        for i in range(task_num):

            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters())
            # 就是把梯度减掉
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct



        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step()


        accs = np.array(corrects) / (querysz * task_num)

        return accs


    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct


        del net

        accs = np.array(corrects) / querysz

        return accs


class Learner(nn.Module):
    """

    """

    def __init__(self, config, imgc, imgsz):
        """

        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(Learner, self).__init__()

        self.config = config

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()

        for i, (name, param) in enumerate(self.config):
            if name is 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            # elif name is 'convt2d':
            #     # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
            #     w = nn.Parameter(torch.ones(*param[:4]))
            #     # gain=1 according to cbfin's implementation
            #     torch.nn.init.kaiming_normal_(w)
            #     self.vars.append(w)
            #     # [ch_in, ch_out]
            #     self.vars.append(nn.Parameter(torch.zeros(param[1])))

            # elif name is 'linear':
            #     # [ch_out, ch_in]
            #     w = nn.Parameter(torch.ones(*param))
            #     # gain=1 according to cbfinn's implementation
            #     torch.nn.init.kaiming_normal_(w)
            #     self.vars.append(w)
            #     # [ch_out]
            #     self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])


            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool1d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError

    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'

            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'


            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info

    def forward(self, x, vars=None, bn_training=True):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """

        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0

        for name, param in self.config:
            if name is 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            # elif name is 'convt2d':
            #     w, b = vars[idx], vars[idx + 1]
            #     # remember to keep synchrozied of forward_encoder and forward_decoder!
            #     x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
            #     idx += 2
            #     # print(name, param, '\tout:', x.shape)
            # elif name is 'linear':
            #     w, b = vars[idx], vars[idx + 1]
            #     x = F.linear(x, w, b)
            #     idx += 2
            #     # print('forward:', idx, x.norm().item())
            elif name is 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2

            elif name is 'flatten':
                # print(x.shape)
                x = x.view(x.size(0), -1)
            elif name is 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name is 'relu':
                x = F.relu(x, inplace=param[0])
            elif name is 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name is 'tanh':
                x = F.tanh(x)
            elif name is 'sigmoid':
                x = torch.sigmoid(x)
            elif name is 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name is 'max_pool1d':
                x = F.max_pool1d(x, param[0])
            elif name is 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])

            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)


        return x


    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars


if __name__ == '__main__':
    main()