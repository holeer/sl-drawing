# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config


class DrawingModel(nn.Module):
    def __init__(self, num_classes, device):
        super(DrawingModel, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.threshold = 0.5
        self.fc = nn.Linear(config.bert_embedding * 2, num_classes)
        self.loss_func = nn.L1Loss()

    def forward(self, features):
        logit = []
        for i in range(len(features)):
            y = self.fc(features[i])
            logit.append(y.argmax().item())
            # y[y >= self.threshold] = 1
            # y[y < self.threshold] = 0
        logit = torch.FloatTensor(logit).to(self.device)
        return logit

    def loss(self, logit, label):
        # if len(logit) < len(label):
        #     logit.extend([0] * (len(label) - len(logit)))
        # elif len(logit) > len(label):
        #     logit.pop()
        loss_value = self.loss_func(logit, label).requires_grad_()
        return loss_value

