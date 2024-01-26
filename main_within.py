import os
import logging
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from lib.data_loader import BNCILoad
from lib.pytorch_utils import seed, split_data, print_args, adjust_learning_rate, evaluation
from base_trainer import csp_trainer

build_model = {
    'CSP': csp_trainer.Trainer
}


def train(x_train, y_train, x_test, y_test, args, uid):
    # initialize the model
    trainer = build_model[args.alg](args)
    ori_acc, ori_bca = trainer.ori_train(x_train, y_train, x_test, y_test)
    feat_layer, cls_layer, params = trainer.retrain_net()

    optimizer = optim.SGD(params, momentum=0.9, weight_decay=5e-4, nesterov=True)
    criterion = nn.CrossEntropyLoss().to(args.device)

    # data loader
    x_train = Variable(torch.from_numpy(x_train).type(torch.FloatTensor))
    y_train = Variable(torch.from_numpy(y_train).type(torch.LongTensor))
    x_test = Variable(torch.from_numpy(x_test).type(torch.FloatTensor))
    y_test = Variable(torch.from_numpy(y_test).type(torch.LongTensor))

    train_loader = DataLoader(dataset=TensorDataset(x_train, y_train),
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=False)
    test_loader = DataLoader(dataset=TensorDataset(x_test, y_test),
                             batch_size=args.batch_size,
                             shuffle=True,
                             drop_last=False)
    feat_layer.eval()
    _, test_acc, test_bca = evaluation(feat_layer, cls_layer, criterion, test_loader, args)
    logging.info(f'origin {args.alg} {uid}: acc-{ori_acc} bca-{ori_bca}')
    logging.info(f'retrain initial {uid}: acc-{test_acc} bca-{test_bca}')

    hists = []
    for epoch in range(args.epochs):
        # model training
        adjust_learning_rate(optimizer=optimizer, epoch=epoch + 1)
        feat_layer.train()
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
            optimizer.zero_grad()
            out = cls_layer(feat_layer(batch_x))
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 1 == 0:
            feat_layer.eval()
            train_loss, train_acc, train_bca = evaluation(feat_layer, cls_layer, criterion, train_loader, args)
            test_loss, test_acc, test_bca = evaluation(feat_layer, cls_layer, criterion, test_loader, args)

            logging.info(
                'Epoch {}/{}: train loss: {:.4f} train acc: {:.2f} train bca: {:.2f}| test acc: {:.2f} test bca: {:.2f}'
                .format(epoch + 1, args.epochs, train_loss, train_acc, train_bca, test_acc, test_bca))

            hists.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "train_bca": train_bca,
                    "loss": test_loss,
                    "acc": test_acc,
                    "bca": test_bca,
                    "ori_acc": ori_acc,
                    "ori_bca": ori_bca
                }
            )
    return hists


def run(args):
    # path build
    log_name = f'{args.alg}' if not len(args.log) else f'{args.alg}' + f'_{args.log}'
    results_path = os.path.join('results_within', args.dataset, str(args.ratio))
    if not os.path.exists(results_path):
        os.system('mkdir -p ' + results_path)

    # logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # print logging
    print_log = logging.StreamHandler()
    logger.addHandler(print_log)
    # save logging
    save_log = logging.FileHandler(os.path.join(results_path, log_name + '.log'), mode='w', encoding='utf8')
    logger.addHandler(save_log)
    logging.info(print_args(args) + '\n')

    acc_list, bca_list = [], []
    original_acc_list, original_bca_list = [], []
    dfs = pd.DataFrame()
    for t in range(args.repeat):
        seed(t * 10)
        acc, bca = [], []
        original_acc, original_bca = [], []
        for s in range(args.subjects):
            x_train, y_train = BNCILoad(args.path, id=s, lb_ratio=1.0, align='')
            x_train, y_train, x_test, y_test = split_data([x_train, y_train], split=0.8, shuffle=True)
            x_train, y_train, _, _ = split_data([x_train, y_train], split=args.ratio, shuffle=True, downsample=True if 'MI' not in args.path else False)

            hists = train(x_train, y_train, x_test, y_test, args, s)
            original_acc.append(hists[-1]['ori_acc'])
            original_bca.append(hists[-1]['ori_bca'])
            acc.append(hists[-1]['acc'])
            bca.append(hists[-1]['bca'])

            df = pd.DataFrame(hists)
            df["method"] = [log_name] * len(hists)
            df["rep"] = [t] * len(hists)
            df["s"] = [s] * len(hists)
            dfs = pd.concat([dfs, df], axis=0)
        logging.info('*' * 200)
        logging.info(f'repeat {t}, or: mean_acc:{np.mean(original_acc)} mean_bca:{np.mean(original_bca)}')
        logging.info(f'repeat {t}, re: mean_acc:{np.mean(acc)} mean_bca:{np.mean(bca)}')
        np.savez(os.path.join(results_path, f'z_{log_name}_r{str(t)}_results.npz'),
                 acc=np.array(acc), bca=np.array(bca),
                 ori_acc=np.array(original_acc), ori_bca=np.array(original_bca))
        acc_list.append(acc)
        bca_list.append(bca)
        original_acc_list.append(original_acc)
        original_bca_list.append(original_bca)

    # log
    logging.info(f'Retrain acc: {acc_list}')
    logging.info(f'Retrain bca: {bca_list}')
    logging.info(f'Mean -- Retrain acc on subjects: {np.mean(acc_list, axis=0)}, bca: {np.mean(bca_list, axis=0)}')
    logging.info(f'Std -- Retrain acc on subjects: {np.std(acc_list, axis=0)}, bca: {np.std(bca_list, axis=0)}')

    logging.info(f'Baseline Mean acc: {np.mean(original_acc_list)} Mean bca: {np.mean(original_bca_list)}')
    logging.info(f'Baseline Std acc: {np.std(np.mean(original_acc_list, axis=1))} Std bca: {np.std(np.mean(original_bca_list, axis=1))}')
    logging.info(f'Retrain Mean acc: {np.mean(acc_list)} Mean bca: {np.mean(bca_list)}')
    logging.info(f'Retrain Std acc: {np.std(np.mean(acc_list, axis=1))} Std bca: {np.std(np.mean(bca_list, axis=1))}')

    # csv
    dfs.to_csv(os.path.join(results_path, log_name + '_raw.csv'), index=False)
    dfs_temp = dfs.groupby(by=["method", "rep", "epoch"]).mean().reset_index()
    avg_dfs = dfs_temp.groupby(by=["method", "epoch"]).mean().reset_index()
    avg_dfs = avg_dfs.sort_values(by=["method", "epoch"])
    avg_dfs["s"] = ["avg"] * len(avg_dfs)
    std_dfs = dfs_temp.groupby(by=["method", "epoch"]).std().reset_index()
    std_dfs = std_dfs.sort_values(by=["method", "epoch"])
    std_dfs["s"] = ["std"] * len(std_dfs)
    dfs = pd.concat([avg_dfs, std_dfs], axis=0)
    dfs = dfs.drop("rep", axis=1)
    dfs.to_csv(os.path.join(results_path, log_name + '_avg.csv'), index=False)

    pd_acc = pd.DataFrame(np.mean(acc_list, axis=0).reshape(1, -1), index=[log_name + '_acc'])
    pd_acc.insert(pd_acc.shape[1], 'mean', value=np.mean(acc_list))
    pd_acc.insert(pd_acc.shape[1], 'std', value=np.std(np.mean(acc_list, axis=1)))
    pd_acc.to_csv(results_path.replace(log_name, '') + 'results' + str(args.epochs) + '.csv', index=True, header=False, mode='a')
    pd_bca = pd.DataFrame(np.mean(bca_list, axis=0).reshape(1, -1), index=[log_name + '_bca'])
    pd_bca.insert(pd_bca.shape[1], 'mean', value=np.mean(bca_list))
    pd_bca.insert(pd_bca.shape[1], 'std', value=np.std(np.mean(bca_list, axis=1)))
    pd_bca.to_csv(results_path.replace(log_name, '') + 'results' + str(args.epochs) + '.csv', index=True, header=False, mode='a')

    pd_acc = pd.DataFrame(np.mean(original_acc_list, axis=0).reshape(1, -1), index=[log_name + '_ori_acc'])
    pd_acc.insert(pd_acc.shape[1], 'mean', value=np.mean(original_acc_list))
    pd_acc.insert(pd_acc.shape[1], 'std', value=np.std(np.mean(original_acc_list, axis=1)))
    pd_acc.to_csv(results_path.replace(log_name, '') + 'results' + str(args.epochs) + '.csv', index=True, header=False, mode='a')
    pd_bca = pd.DataFrame(np.mean(original_bca_list, axis=0).reshape(1, -1), index=[log_name + '_ori_bca'])
    pd_bca.insert(pd_bca.shape[1], 'mean', value=np.mean(original_bca_list))
    pd_bca.insert(pd_bca.shape[1], 'std', value=np.std(np.mean(original_bca_list, axis=1)))
    pd_bca.to_csv(results_path.replace(log_name, '') + 'results' + str(args.epochs) + '.csv', index=True, header=False, mode='a')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model train')
    parser.add_argument('--dataset', default='BNCI_MI2C')
    parser.add_argument('--gpu_id', type=str, default='7')
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--ratio', type=float, default=0.1)

    parser.add_argument('--alg', type=str, default='CSP')
    parser.add_argument('--filters', type=int, default=8, help='CSP parameter')
    parser.add_argument('--alpha', type=float, default=1e-2, help='TRCSP parameter')

    parser.add_argument('--log', type=str, default='test')
    args = parser.parse_args()

    classes = {'BNCI_MI2C': 2, 'BNCI_MI14S': 2, 'BNCI_MI9S': 2}
    subjects = {'BNCI_MI2C': 9, 'BNCI_MI14S': 14, 'BNCI_MI9S': 9}
    channels = {'BNCI_MI2C': 22, 'BNCI_MI14S': 15, 'BNCI_MI9S': 13}
    data_path = {'BNCI_MI2C': '../dataset/MIData/BNCI2014-001-2/',
                 'BNCI_MI14S': '../dataset/MIData/BNCI2014-002-2/',
                 'BNCI_MI9S': '../dataset/MIData/BNCI2015-001-2/'}
    data_fs = {'BNCI_MI2C': 250, 'BNCI_MI14S': 512, 'BNCI_MI9S': 512}
    args.classes, args.subjects, args.channel = classes[args.dataset], subjects[args.dataset], channels[args.dataset]
    args.path, args.fs = data_path[args.dataset], data_fs[args.dataset]

    args.device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(args.device)

    run(args)
