# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import model_understanding
import dataset_understanding
from config import config
import os
import torch.utils.data as data
import utils
from tqdm import tqdm
import time

num_classes = len(utils.get_vocabulary(config.label_file))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(model, train_iter, test_iter):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="max",
                                                           factor=0.5,
                                                           patience=1)
    best_acc = 0.0
    print('*' * 30 + 'Training' + '*' * 30)
    for epoch in range(config.epochs):
        model.train()
        avg_time, avg_loss = 0.0, 0.0
        tqdm_batch_iterator = tqdm(train_iter)
        for batch_index, batch in enumerate(tqdm_batch_iterator):
            batch_start = time.time()
            features, labels = batch
            optimizer.zero_grad()
            logit = model(features)
            labels = [i.item() for i in labels]
            labels = torch.FloatTensor(labels).to(device)
            loss = model.loss(logit, labels)
            loss.backward()
            optimizer.step()
            avg_time += time.time() - batch_start
            avg_loss += loss.item()
            description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}" \
                .format(avg_time / (batch_index + 1),
                        avg_loss / (batch_index + 1))
            tqdm_batch_iterator.set_description(description)
        avg_loss /= len(train_iter)
        print('\rEpoch[{}] - loss: {:.4f}\n'.format(epoch, avg_loss))

        test_acc = valid(model, test_iter)
        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            save(model, config.target_dir, 'best', epoch)


def save(model, save_dir, save_prefix, epoch):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_epoch_{}.pt'.format(save_prefix, epoch)
    torch.save(model.state_dict(), save_path)


def valid(model, data_iter):
    model.eval()
    device = model.device
    corrects, avg_loss = 0, 0
    for batch in tqdm(data_iter):
        features, labels = batch
        logit = model(features).to(device)
        labels = [i.item() for i in labels]
        labels = torch.FloatTensor(labels).to(device)
        loss = model.loss(logit, labels)
        avg_loss += loss.item()
        corrects += (logit == labels).sum()

    avg_loss /= len(data_iter)
    accuracy = corrects / (len(data_iter) * len(labels))
    print('\nEvaluation - loss: {:.6f} acc: {:.4f} \n'.format(avg_loss, accuracy))

    return accuracy


if __name__ == '__main__':
    model = model_understanding.DrawingModel(num_classes, device).to(device)
    train_data = dataset_understanding.DrawingDataset(config.train_file)
    test_data = dataset_understanding.DrawingDataset(config.test_file)
    train_loader = data.DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        drop_last=config.drop_last
    )
    test_loader = data.DataLoader(
        test_data,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        drop_last=config.drop_last
    )
    train(model, train_loader, test_loader)
