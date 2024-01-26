import torch
import logging
from sklearn.metrics import *

from lib.pytorch_utils import adjust_learning_rate


class Trainer:
    def __init__(self, modelF, modelC, optimizer, args):
        self.modelF     = modelF
        self.modelC     = modelC
        self.optimizer  = optimizer
        self.device     = args.device
        self.criterion  = torch.nn.CrossEntropyLoss()

    def train(self, label_loader, test_loader):
        self.modelF.train()
        self.modelC.train()
        for step, (batch_xs, batch_ys) in enumerate(label_loader):
            batch_xs, batch_ys = batch_xs.to(self.device), batch_ys.to(self.device)
            self.optimizer.zero_grad()
            outs = self.modelC(self.modelF(batch_xs))
            loss = self.criterion(outs, batch_ys)
            # backward
            loss.backward()
            self.optimizer.step()

    def test(self, data_loader):
        self.modelF.eval()
        self.modelC.eval()
        with torch.no_grad():
            total_loss, total_num = 0.0, 0.0
            y_true, y_pred = [], []
            for batch_idx, (x, y) in enumerate(data_loader):
                x, y = x.to(self.device), y.to(self.device)
                num_batch = x.shape[0]
                total_num += num_batch
                logits = self.modelC(self.modelF(x))
                loss = self.criterion(logits, y)
                y_true.extend(y.cpu().tolist())
                y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
                total_loss += loss.cpu().item() * num_batch
            acc = accuracy_score(y_true, y_pred)
            bca = balanced_accuracy_score(y_true, y_pred)
        return total_loss / total_num, acc, bca

    def loop(self, epochs, source_data, target_data):
        hists = []
        for ep in range(epochs):
            self.epoch = ep
            adjust_learning_rate(optimizer=self.optimizer, epoch=ep + 1)
            self.train(source_data, target_data)
            test_loss, test_acc, test_bca = self.test(target_data)

            if (ep + 1) % 1 == 0:
                train_loss, train_acc, train_bca = self.test(source_data)
                logging.info(
                    'Epoch {}/{}: train loss: {:.4f} train acc: {:.2f} train bca: {:.2f}| test acc: {:.2f} test bca: {:.2f}'
                        .format(ep + 1, epochs, train_loss, train_acc, train_bca, test_acc, test_bca))

                hists.append(
                    {
                        "epoch": ep + 1,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "train_bca": train_bca,
                        "loss": test_loss,
                        "acc": test_acc,
                        "bca": test_bca
                    }
                )
        return hists
