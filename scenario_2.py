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
from ethicml.implementations.pytorch_common import CustomDataset
from ethicml.metrics import Accuracy
from ethicml.preprocessing import scale_continuous, train_test_split
from ethicml.utility import DataTuple


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


class EmbeddingPredictor(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.hid = nn.Linear(latent_dim, 100)
        self.hid_1 = nn.Linear(100, 100)
        self.bn_1 = nn.BatchNorm1d(100)
        self.out = nn.Linear(100, 1)

    def forward(self, z: td.Distribution):
        y = self.bn_1(F.relu(self.hid(z)))
        return self.out(y)


class Model2(nn.Module):
    def __init__(self, in_size: int, latent_dim: int):
        super().__init__()
        self.enc = FeatureEncoder(in_size, latent_dim)
        self.adv = FeatureAdv(latent_dim)
        self.pred = EmbeddingPredictor(latent_dim)

    def forward(self, x):
        z = self.enc(x)
        s = self.adv(z.rsample())
        y = self.pred(z.rsample())
        return z, s, y


def evaluate_z(train: DataTuple, test: DataTuple, model: InAlgorithm = LRCV):
    model = model()
    preds = model.run(train, test)
    acc = Accuracy().score(preds, test)
    print(f"Accuracy: {acc:.3f}")


def run_model_2(latent_dims: int, epochs: int):
    dataset = adult()
    data = dataset.load()
    scaler = StandardScaler()
    data, scaler2 = scale_continuous(dataset, data, scaler)

    _train, _test = train_test_split(data, train_percentage=0.9)
    train_data = CustomDataset(_train)
    train_loader = DataLoader(train_data, batch_size=256)

    test_data = CustomDataset(_test)
    test_loader = DataLoader(test_data, batch_size=256)

    print(f"Performance on the original data...")
    evaluate_z(_train, _test)

    print(f"Majority classifier...")
    evaluate_z(_train, _test, model=Majority)

    print(f"Random classifier...")
    evaluate_z(_train, _test, model=Blind)

    model = Model2(len(_train.x.columns), latent_dims)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    with trange(epochs) as t:
        for epoch in t:
            for (x, s, y) in train_loader:
                z, s_pred, y_pred = model(x)

                feat_prior = td.Normal(loc=torch.zeros(latent_dims), scale=torch.ones(latent_dims))
                feat_kl_loss = td.kl.kl_divergence(z, feat_prior)

                feat_sens_loss = F.binary_cross_entropy_with_logits(s_pred, s, reduction="mean")
                pred_y_loss = F.binary_cross_entropy_with_logits(y_pred, y, reduction="mean")

                loss = feat_kl_loss.mean() + feat_sens_loss + pred_y_loss

                t.set_postfix(loss=loss.item())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            scheduler.step()

    post_train = DataTuple(x=encode(train_loader, model, latent_dims), s=_train.s, y=_train.y)
    post_test = DataTuple(x=encode(test_loader, model, latent_dims), s=_test.s, y=_test.y)

    print(f"Performance on the original data...")
    evaluate_z(post_train, post_test)


def encode(loader: DataLoader, model: nn.Module, latent_dims: int):
    feats_train_encs: pd.DataFrame = pd.DataFrame(columns=list(range(latent_dims)))
    model.eval()
    with torch.no_grad():
        for (x, s, y) in loader:
            z, _, _ = model(x)
            feats_train_encs = pd.concat(
                [
                    feats_train_encs,
                    pd.DataFrame(z.sample().cpu().numpy(), columns=list(range(latent_dims))),
                ],
                axis="rows",
                ignore_index=True,
            )

    return feats_train_encs


if __name__ == "__main__":
    run_model_2(latent_dims=50, epochs=100)
