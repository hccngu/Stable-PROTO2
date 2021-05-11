import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, ebd, args):
        super(MLP, self).__init__()

        self.args = args

        self.ebd = ebd

        self.d = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(ebd, 128),
                )
        self.cost = nn.CrossEntropyLoss()

    def forward(self, inputs):

        logits = self.d(inputs)  # [b, 256] -> [b, 128]

        return logits

    def loss(self, logits, label):
        loss_ce = self.cost(logits/torch.mean(logits, dim=0), label)

        return loss_ce

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label).type(torch.FloatTensor))