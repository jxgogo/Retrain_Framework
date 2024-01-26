import os
import random
import argparse
import numpy as np
from typing import Optional
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn

def print_args(args: argparse.ArgumentParser):
    """
    print the hyperparameters
    :param args: hyperparameters
    :return: None
    """
    s = "=========================================================\n"
    for arg, concent in args.__dict__.items():
        s += "{}:{}\n".format(arg, concent)
    return s


def init_weights(model: nn.Module):
    """
    Network Parameters Initialization Function
    :param model: the model to initialize
    :return: None
    """
    classname = model.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight, 1.0, 0.02)
        nn.init.zeros_(model.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(model.weight)
        nn.init.zeros_(model.bias)
    elif classname.find('Conv2d') != -1:
        nn.init.kaiming_uniform_(model.weight)


def bca_score(y_true, y_pred):
    m = confusion_matrix(y_true, y_pred)
    numb = m.shape[0]
    acc_each_label = 0
    for i in range(numb):
        acc = m[i, i] / np.sum(m[i, :], keepdims=False).astype(np.float32)
        acc_each_label += acc
    return acc_each_label / numb


def evaluation(feat: nn.Module, cls: nn.Module, criterion: nn.Module, data_loader: DataLoader, args):
    loss, correct = 0., 0
    labels, preds = [], []
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(args.device), y.to(args.device)
            out = cls(feat((x)))
            pred = nn.Softmax(dim=1)(out).cpu().argmax(dim=1)
            loss += criterion(out, y).item()
            correct += pred.eq(y.cpu().view_as(pred)).sum().item()
            labels.extend(y.cpu().tolist())
            preds.extend(pred.tolist())
    loss /= len(labels)
    acc = correct / len(labels)
    bca = bca_score(labels, preds)

    return loss, acc, bca


def seed(seed: Optional[int] = 0):
    """
    fix all the random seed
    :param seed: random seed
    :return: None
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def adjust_learning_rate(optimizer: nn.Module, epoch: int):
    """decrease the learning rate"""
    if epoch == 120:
        r = 0.1
    elif epoch == 180:
        r = 0.1
    else:
        r = 1.0
    for param_group in optimizer.param_groups:
        param_group['lr'] *= r


def weight_for_balanced_classes(y: torch.Tensor):
    count = [0.0] * len(np.unique(y.numpy()))
    for label in y:
        count[label] += 1.0
    count = [len(y) / x for x in count]
    weight = [0.0] * len(y)
    for idx, label in enumerate(y):
        weight[idx] = count[label]

    return weight


def split_data(data, split=0.8, shuffle=True, downsample=False):
    x = data[0]
    y = data[1]

    data_size = len(x)
    split_index = int(data_size * split)
    indices = np.arange(data_size)
    if shuffle:
        indices = np.random.permutation(indices)

    train_idx = indices[:split_index]
    test_idx  = indices[split_index:]
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]
    if downsample:
        x_train1 = x_train[np.where(y_train == 0)]
        x_train2 = x_train[np.where(y_train == 1)]
        sample_num = min(len(x_train1), len(x_train2))
        idx1, idx2 = np.random.permutation(np.arange(len(x_train1))), np.random.permutation(np.arange(len(x_train2)))
        x_train = np.concatenate([x_train1[idx1[:sample_num]], x_train2[idx2[:sample_num]]], axis=0)
        y_train = np.concatenate([np.zeros(shape=[sample_num]), np.ones(shape=[sample_num])], axis=0)

    return x_train, y_train, x_test, y_test


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
