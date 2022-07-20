# -*- coding: UTF-8 -*-
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import utils


class DrawingDataset(Dataset):

    def __init__(self, dataset_path):
        self.data = pd.read_csv(dataset_path)
        self.drawings = np.asarray(self.data['drawing'])
        self.labels = np.asarray(self.data['label'])

    def __getitem__(self, item):
        feature = utils.split_bottom_bar(self.drawings[item])
        label = list(map(int, self.labels[item].split(',')))
        # label = torch.LongTensor(label)
        return feature, label

    def __len__(self):
        return len(self.data.index)


# if __name__ == '__main__':
#     dataset = DrawingDataset('dataset/train/train.csv')
#     import torch.utils.data as data
#     train_loader = data.DataLoader(
#         dataset,
#         batch_size=4,
#         shuffle=True,
#         drop_last=False,
#         num_workers=0
#     )
#     for batch in train_loader:
#         print(batch)
#         sub_pics, contents, labels = batch
