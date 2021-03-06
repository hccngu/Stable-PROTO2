import torch
import torch.nn as nn

from embedding.rnn import RNN
import torch.nn.functional as F


class ModelG(nn.Module):

    def __init__(self, ebd, args):
        super(ModelG, self).__init__()

        self.args = args

        self.ebd = ebd

        self.ebd_dim = self.ebd.embedding_dim
        self.hidden_size = 128

        self.rnn = RNN(300, 128, 1, True, 0)
        self.lstm = nn.LSTM(input_size=300, hidden_size=128, num_layers=1, batch_first=True, dropout=0)

        self.seq = nn.Sequential(
            nn.Dropout(0.6),
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