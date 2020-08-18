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




class Model1(nn.Module):
    def __init__(self, in_size: int, latent_dim: int):
        super().__init__()
        self.enc = FeatureEncoder(in_size, latent_dim)
        self.adv = FeatureAdv(latent_dim)

    def forward(self, x):
        z = self.enc(x)
        s = self.adv(z.rsample())
        return z, s


def encode(loader: DataLoader, model: nn.Module, latent_dims: int):
    feats_train_encs: pd.DataFrame = pd.DataFrame(columns=list(range(latent_dims)))
    model.eval()
    with torch.no_grad():
        for (x, s, y) in loader:
            z, _ = model(x)
            feats_train_encs = pd.concat(
                [
                    feats_train_encs,
                    pd.DataFrame(z.sample().cpu().numpy(), columns=list(range(latent_dims))),
                ],
                axis="rows",
                ignore_index=True,
            )

    return feats_train_encs


def run_model_1(latent_dims: int, epochs: int):
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

    model = Model1(len(_train.x.columns), latent_dims)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    with trange(epochs) as t:
        for epoch in t:
            for (x, s, y) in train_loader:
                z, s_pred = model(x)

                feat_prior = td.Normal(loc=torch.zeros(latent_dims), scale=torch.ones(latent_dims))
                feat_kl_loss = td.kl.kl_divergence(z, feat_prior)

                feat_sens_loss = F.binary_cross_entropy_with_logits(s_pred, s, reduction="mean")

                loss = feat_kl_loss.mean() + feat_sens_loss

                t.set_postfix(loss=loss.item())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            scheduler.step()

    post_train = DataTuple(x=encode(train_loader, model, latent_dims), s=_train.s, y=_train.y)
    post_test = DataTuple(x=encode(test_loader, model, latent_dims), s=_test.s, y=_test.y)

    print(f"Performance on the embeddings...")
    evaluate_z(post_train, post_test)

    print(f"Fairness on the original data...")
    evaluate_z(_train, _test, metric=ProbPos, per_sens=True)

    print(f"Fairness on the embeddings...")
    evaluate_z(post_train, post_test, metric=ProbPos, per_sens=True)


if __name__ == "__main__":
    run_model_1(latent_dims=50, epochs=100)
