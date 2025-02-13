"""Generates samples with same classes as h5ad, replacing values."""

# %%
import argparse
import sys
from unittest.mock import patch

import numpy as np
import scanpy as sc
import torch
from sklearn.preprocessing import LabelEncoder

import classifier_sample
from VAE.VAE_model import VAE

# %%
parser = argparse.ArgumentParser(
    description="Generates samples with same classes as h5ad, replacing values."
)
# parser.add_argument("--model_name", type=str, help="Name of the dataset.")
parser.add_argument("--batch_size", type=int, default=1024, help="Batch size.")
parser.add_argument("--device", default=None, help="Device")
parser.add_argument("--data_dir", type=str, help="Path to h5ad file.")
# parser.add_argument("--umap_path", type=str, help="Path to UMAP pickle.")
parser.add_argument("--model_path", type=str, help="Path to model checkpoint.")
parser.add_argument("--classifier_path", type=str, help="Path to classifier checkpoint.")
parser.add_argument("--vae_path", type=str, help="Path to VAE checkpoint.")
parser.add_argument("--sample_dir", type=str, help="Path to save samples.")

args = parser.parse_args()

# NAME = args.model_name
BATCH_SIZE = args.batch_size
DEVICE_ID = args.device

DATA_DIR = args.data_dir
# UMAP_PATH = f"../diffusion-scratch/data/processed/zeng23/umap_hvg.pkl"

MODEL_PATH = args.model_path
CLASSIFIER_PATH = args.classifier_path
VAE_PATH = args.vae_path
SAMPLE_DIR = args.sample_dir


# %%
def get_h5ad_classes(data_dir):
    """
    Get CellDataset.

    :param data_dir: a dataset directory.

    Returns:
    adata: the dataset.
    classes: the classes of the dataset, encoded as ordinal integers.

    """

    if not data_dir:
        raise ValueError("unspecified data directory")

    adata = sc.read_h5ad(data_dir)

    # preporcess the data. modify this part if use your own dataset. the gene expression must first norm1e4 then log1p
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.filter_cells(adata, min_genes=10)
    adata.var_names_make_unique()

    classes = adata.obs["celltype"].values
    label_encoder = LabelEncoder()
    labels = classes
    label_encoder.fit(labels)
    classes = label_encoder.transform(labels)

    return adata, classes


# %%
def gen_samples(c, n_samples, n_classes):
    args = "classifier_sample.py"
    args += f" --model_path {MODEL_PATH}"
    args += f" --classifier_path {CLASSIFIER_PATH}"
    args += f" --sample_dir {SAMPLE_DIR}"
    args += f" --num_samples {n_samples}"
    args += f" --batch_size {BATCH_SIZE}"
    args += f" --device_ids {DEVICE_ID}"
    args += f" --num_class {n_classes}"
    args = args.split()

    with patch.object(sys, "argv", args):
        classifier_sample.main([c])

    npzfile = np.load(f"{SAMPLE_DIR}{c}.npz", allow_pickle=True)
    return npzfile["cell_gen"][:n_samples]


def load_VAE(num_genes, device="cuda"):
    autoencoder = VAE(
        num_genes=num_genes,
        device=device,
        seed=0,
        loss_ae="mse",
        hidden_dim=128,
        decoder_activation="ReLU",
    )
    autoencoder.load_state_dict(torch.load(VAE_PATH))
    return autoencoder


# %%
print("Loading h5ad...")
adata, classes = get_h5ad_classes(DATA_DIR)

del adata.layers["loglsn_counts"]
del adata.layers["lsn_counts"]
del adata.raw
del adata.obsm["X_umap_HVGs"]

# %%
print("Generating samples...")

# create ndarray to hold latent values; avoids setting on adata.X ArrayView
X_masked = np.empty((adata.shape[0], 128), dtype=np.float32)

class_types = np.unique(classes)
n_classes = len(class_types)
for c in class_types:
    bool_idx = classes == c
    n_samples = bool_idx.sum()
    print(f"Class {c}: {n_samples} samples")

    X_masked[bool_idx.nonzero()] = gen_samples(c, n_samples, n_classes)

# %%
print("Loading VAE...")

autoencoder = load_VAE(adata.shape[1], device=torch.device(DEVICE_ID))
autoencoder.eval()
# %%
print("Decoding samples...")

with torch.no_grad():
    dl = torch.utils.data.DataLoader(
        torch.as_tensor(X_masked),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    x_out = []
    for i, x in enumerate(dl):
        x_out.append(autoencoder(x.to(DEVICE_ID), return_decoded=True).cpu().numpy())

adata.X = X_masked = x_out = np.concatenate(x_out, axis=0)

# %%
# print("Computing UMAP...")

# with open(UMAP_PATH, "rb") as file:
#     umapper = pickle.load(file)

# adata.obsm["X_umap_HVGs"] = umapper.transform(adata[:, adata.var["highly_variable"]].X)

# %%
print("Saving results...")
adata.write(f"{SAMPLE_DIR}.h5ad")
