import math
import random

from PIL import Image
import blobfile as bf
import numpy as np
import torch as th
from torch.utils.data import DataLoader, Dataset

import scanpy as sc
import torch
import sys

sys.path.append("..")
from VAE.VAE_model import VAE
from sklearn.preprocessing import LabelEncoder


def stabilize(expression_matrix):
    """Use Anscombes approximation to variance stabilize Negative Binomial data
    See https://f1000research.com/posters/4-1041 for motivation.
    Assumes columns are samples, and rows are genes
    """
    from scipy import optimize

    phi_hat, _ = optimize.curve_fit(
        lambda mu, phi: mu + phi * mu**2, expression_matrix.mean(1), expression_matrix.var(1)
    )

    return np.log(expression_matrix + 1.0 / (2 * phi_hat[0]))


def load_VAE(vae_path, num_gene, hidden_dim, device="cuda"):
    autoencoder = VAE(
        num_genes=num_gene,
        device=device,
        seed=0,
        loss_ae="mse",
        hidden_dim=hidden_dim,
        decoder_activation="ReLU",
    )
    autoencoder.load_state_dict(torch.load(vae_path))
    return autoencoder


def load_data(
    *,
    data_dir,
    batch_size,
    vae_path=None,
    deterministic=False,
    train_vae=False,
    hidden_dim=128,
    train_split_only=False,
    num_workers=1,
    device=None,
):
    """
    For a dataset, create a generator over (cells, kwargs) pairs.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param vae_path: the path to save autoencoder / read autoencoder checkpoint.
    :param deterministic: if True, yield results in a deterministic order.
    :param train_vae: train the autoencoder or use the autoencoder.
    :param hidden_dim: the dimensions of latent space. If use pretrained weight, set 128
    """
    dataset = get_dataset(
        data_dir=data_dir,
        vae_path=vae_path,
        train_vae=train_vae,
        hidden_dim=hidden_dim,
        train_split_only=train_split_only,
        device=device,
    )
    return dataset_to_loader(
        dataset=dataset, batch_size=batch_size, deterministic=deterministic, num_workers=num_workers
    )


def dataset_to_loader(*, dataset, batch_size, deterministic=False, num_workers=1):
    """
    Create a generator over (cells, kwargs) pairs from a CellDataset.

    :param dataset: a CellDataset.
    :param batch_size: the batch size of each returned pair.
    :param deterministic: if True, yield results in a deterministic order.
    """

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=num_workers,
        drop_last=True,
    )
    while True:
        yield from loader


def get_dataset(
    *,
    data_dir,
    vae_path=None,
    train_vae=False,
    hidden_dim=128,
    train_split_only=False,
    device=None,
):
    """
    Get CellDataset.

    :param data_dir: a dataset directory.
    :param vae_path: the path to save autoencoder / read autoencoder checkpoint.
    :param train_vae: train the autoencoder or use the autoencoder.
    :param hidden_dim: the dimensions of latent space. If use pretrained weight, set 128
    """

    if not data_dir:
        raise ValueError("unspecified data directory")

    adata = sc.read_h5ad(data_dir)

    # preporcess the data. modify this part if use your own dataset. the gene expression must first norm1e4 then log1p
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.filter_cells(adata, min_genes=10)
    adata.var_names_make_unique()

    if train_split_only:
        adata = adata[adata.obs["split"] == "train"]

    # if generate ood data, left this as the ood data
    # selected_cells = (adata.obs['organ'] != 'mammary') | (adata.obs['celltype'] != 'B cell')
    # adata = adata[selected_cells, :]

    classes = adata.obs["celltype"].values
    label_encoder = LabelEncoder()
    labels = classes
    label_encoder.fit(labels)
    classes = label_encoder.transform(labels)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    cell_data = adata.X.toarray()

    # turn the gene expression into latent space. use this if training the diffusion backbone.
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if not train_vae:
        num_gene = cell_data.shape[1]
        with th.no_grad():
            autoencoder = load_VAE(vae_path, num_gene, hidden_dim, device=device)
            cell_data = autoencoder(
                torch.tensor(cell_data, dtype=torch.float32, device=device),
                return_latent=True,
            )
            cell_data = cell_data.cpu().numpy()

    dataset = CellDataset(cell_data, classes)

    return dataset


class CellDataset(Dataset):
    def __init__(self, cell_data, class_name):
        super().__init__()
        self.data = cell_data
        self.class_name = class_name

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        arr = self.data[idx]
        out_dict = {}
        if self.class_name is not None:
            out_dict["y"] = torch.as_tensor(self.class_name[idx], dtype=torch.int64)
        return arr, out_dict
