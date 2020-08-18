from typing import Optional, Tuple

import pandas as pd
import torch
import torch.distributions as td
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import trange

from ethicml.algorithms.inprocess import LR, LRCV, InAlgorithm, Majority
from ethicml.algorithms.inprocess.blind import Blind
from ethicml.data import adult
from ethicml.evaluators import metric_per_sensitive_attribute
from ethicml.implementations.pytorch_common import CustomDataset
from ethicml.metrics import Accuracy, Metric, ProbPos
from ethicml.preprocessing import scale_continuous, train_test_split
from ethicml.utility import DataTuple


def evaluate_z(
    train: DataTuple,
    test: DataTuple,
    model: InAlgorithm = LRCV,
    metric: Metric = Accuracy,
    per_sens: bool = False,
):
    model = model()
    preds = model.run(train, test)

    if per_sens:
        score = metric_per_sensitive_attribute(preds, test, metric())
    else:
        score = metric().score(preds, test)
    print(f"{metric().name}: {score:.3f}")


class GradReverse(torch.autograd.Function):
    """Gradient reversal layer"""

    @staticmethod
    def forward(ctx, x: Tensor, lambda_: float) -> Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        return grad_output.neg().mul(ctx.lambda_), None


def grad_reverse(features: Tensor, lambda_: float = 1.0) -> Tensor:
    return GradReverse.apply(features, lambda_)


class FeatureEncoder(nn.Module):
    def __init__(self, in_size: int, latent_dim: int):
        super().__init__()
        self.hid_1 = nn.Linear(in_size, 100)
        self.bn_1 = nn.BatchNorm1d(100)
        self.hid_2 = nn.Linear(100, 100)
        self.bn_2 = nn.BatchNorm1d(100)

        self.mu = nn.Linear(100, latent_dim)
        self.logvar = nn.Linear(100, latent_dim)

    def forward(self, z: torch.Tensor):
        x = self.bn_1(F.relu(self.hid_1(z)))
        x = F.relu(self.hid_2(x))
        return td.Normal(loc=self.mu(x), scale=F.softplus(self.logvar(x)))


class FeatureAdv(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.hid = nn.Linear(latent_dim, 100)
        self.hid_1 = nn.Linear(100, 100)
        self.bn_1 = nn.BatchNorm1d(100)
        self.out = nn.Linear(100, 1)

    def forward(self, z: td.Distribution):
        s = self.bn_1(F.relu(self.hid(grad_reverse(z))))
        return self.out(s)

