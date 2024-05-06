# -*- coding: UTF-8 -*-
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import utils
from sklearn.model_selection import train_test_split
from config import config
from logger import Logger
import time
import os
from model_label import CNN
import random
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, precision_score

torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
np.random.seed(config.seed)
random.seed(config.seed)
torch.backends.cudnn.deterministic = True

vocabulary = utils.get_vocabulary(config.label_file)
average = "macro"
LOG_FILENAME = config.log_dir + "Label_" + str(int(time.time())) + ".log"
print(30 * "=",
      "Training log in file: {}".format(LOG_FILENAME),
      30 * "=")
log = Logger(filename=LOG_FILENAME)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DrawingLabelDataset(Dataset):
    def __init__(self, drawing, content, label):
        self.drawing = np.asarray(drawing)
        self.content = np.asarray(content)
        self.label = np.asarray(label)

    def __getitem__(self, item):
        # feature = utils.get_features(self.drawing[item], self.content[item])
        drawing = self.drawing[item]
        if isinstance(self.content[item], float):
            content = ''
        else:
            content = self.content[item]
        label = utils.onehot_label(vocabulary, self.label[item])
        label = torch.FloatTensor(label)
        return drawing, content, label

    def __len__(self):
        return len(self.drawing)


def main():
    data = pd.read_csv('dataset/data_new.csv')
    print(30 * "=",
          "加载数据集",
          30 * "=")
    drawing_dataset = DrawingLabelDataset(data['drawing'], data['content'], data['label'])
    train_dataset, test_dataset = train_test_split(drawing_dataset, test_size=0.2, random_state=1)
    train_loader = DataLoader(train_dataset, shuffle=config.shuffle, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, shuffle=config.shuffle, batch_size=config.batch_size)

    model = CNN(len(vocabulary), device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="max",
                                                           factor=0.5,
                                                           patience=1)

    best_score = 0.0
    start_epoch = 1
    print(30 * "=",
          "Training model on device: {}".format(device),
          30 * "=")

    patience_counter = 0
    # total_counter = 0
    # average_time = 0
    # total_time = 0
    for epoch in range(start_epoch, config.epochs + 1):

        log.logger.info("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss = train(model,
                                       train_loader,
                                       optimizer,
                                       config.max_grad_norm)
        log.logger.info("-> Training time: {:.4f}s, loss = {:.4f}"
                        .format(epoch_time, epoch_loss))

        # total_time += epoch_time
        # total_counter += 1
        # average_time = total_time / total_counter
        # log.logger.info("-> Average time: {:.4f}s"
        #         .format(average_time))

        epoch_time, valid_loss, valid_estimator = valid(model,
                                                        test_loader)
        log.logger.info("-> Valid time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%, precision: {:.4f}%, recall: {:.4f}%, f1: {:.4f}%"
                        .format(epoch_time, valid_loss, valid_estimator[0] * 100, valid_estimator[1] * 100, valid_estimator[2] * 100, valid_estimator[3] * 100))

        scheduler.step(valid_estimator[0])

        # Early stopping on validation accuracy.
        if valid_estimator[0] >= best_score:
            best_score = valid_estimator[0]
            patience_counter = 0
            torch.save({"epoch": epoch,
                        "model": model.state_dict()},
                       os.path.join(config.target_dir, "best.pth.tar"))
            log.logger.info("saved the best model in epoch {}.".format(epoch))
        else:
            patience_counter += 1

        # if epoch % 1 == 0:
        #     # Save the model at each epoch.
        #     torch.save({"epoch": epoch,
        #                 "model": model.state_dict(),
        #                 "optimizer": optimizer.state_dict()},
        #                os.path.join(config.target_dir, "RoBERTa_NER_{}.pth.tar".format(epoch)))

        # if patience_counter >= config.patience:
        #     log.logger.info("-> Early stopping: patience limit reached, stopping...")
        #     break


def train(model, dataloader, optimizer, max_gradient_norm):
    # Switch the model to train mode.
    model.train()
    device = model.device
    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0

    tqdm_batch_iterator = tqdm(dataloader)
    for batch_index, batch in enumerate(tqdm_batch_iterator):
        batch_start = time.time()

        # Move input and output data to the GPU if it is used.
        drawing, content, label = batch

        label = label.to(device)

        optimizer.zero_grad()
        logit = model(drawing, content)
        loss = model.loss(logit, label)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)

        optimizer.step()

        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()

        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}" \
            .format(batch_time_avg / (batch_index + 1),
                    running_loss / (batch_index + 1))
        tqdm_batch_iterator.set_description(description)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)

    return epoch_time, epoch_loss


def valid(model, dataloader):
    model.eval()
    device = model.device
    epoch_start = time.time()
    running_loss = 0.0
    target = []
    pred = []

    with torch.no_grad():
        tqdm_batch_iterator = tqdm(dataloader)
        for _, batch in enumerate(tqdm_batch_iterator):
            drawing, content, label = batch

            label = label.to(device)

            logit = model(drawing, content)
            loss = model.loss(logit, label)

            running_loss += loss.item()

            out = torch.argmax(logit, dim=1)
            label = torch.argmax(label, dim=1)
            out = out.reshape(-1)
            label = label.reshape(-1)
            target += label.cpu().numpy().tolist()
            pred += out.cpu().numpy().tolist()

    # t = classification_report(target, pred, target_names=['公司', '工程', '图名', '姓名', '签名', '属性', '图号'])
    # log.logger.info(t)

    accuracy = accuracy_score(target, pred)
    precision = precision_score(target, pred, average=average)
    recall = recall_score(target, pred, average=average)
    f1 = f1_score(target, pred, average=average)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    estimator = (accuracy, precision, recall, f1)
    # estimator = (accuracy, )

    return epoch_time, epoch_loss, estimator


if __name__ == '__main__':
    main()
