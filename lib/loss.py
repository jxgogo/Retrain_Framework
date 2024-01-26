import torch
import torch.nn as nn
from typing import Optional


def Entropy(input: torch.Tensor) -> torch.Tensor:
    """
    Compute the entropy
    :param input: the softmax output
    :return: entropy
    """
    entropy = -input * torch.log(input + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy


class ClassConfusionLoss(nn.Module):
    """
    The class confusion loss

    Parameters:
        - **t** Optional(float): the temperature factor used in MCC
    """

    def __init__(self,
                 t: Optional[float] = 2.0):
        super(ClassConfusionLoss, self).__init__()
        self.t = t

    def forward(self,
                output: torch.Tensor) -> torch.Tensor:
        n_sample, n_class = output.shape
        softmax_out = nn.Softmax(dim=1)(output / self.t)
        entropy_weight = Entropy(softmax_out).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (n_sample * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)
        class_confusion_matrix = torch.mm((softmax_out * entropy_weight).transpose(1, 0), softmax_out)
        class_confusion_matrix = class_confusion_matrix / torch.sum(class_confusion_matrix, dim=1)
        mcc_loss = (torch.sum(class_confusion_matrix) - torch.trace(class_confusion_matrix)) / n_class
        return mcc_loss

