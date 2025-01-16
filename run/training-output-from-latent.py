import os
import sys
import pickle
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from glob import glob
from experimet_config import Config

sys.path.append("../")
from src.models.deepsdf import DeepSDF
from src.utils.mesh_utils import create_mesh


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


cfg = Config()

# loading carfolderlatent_idx, latent_idx2carfolder
with open("../outputs/data-split/carfolder2latent_idx.pickle", "rb") as f:
    carfolder2latent_idx = pickle.load(f)

with open("../outputs/data-split/latent_idx2carfolder.pickle", "rb") as f:
    latent_idx2carfolder = pickle.load(f)

# loading latent codes
latent_codes = torch.load("../outputs/latent-codes/latent_codes.pth")

# loading deepsdf model
model = DeepSDF(
    hidden_dim=cfg.hidden_dim,
    xyz_pos_enc_dim=cfg.xyz_pos_enc_dim,
    latent_code_dim=cfg.latent_code_dim,
    dropout_prob=cfg.dropout_prob,
).to(cfg.device)
model.load_state_dict(torch.load("../outputs/models/deepsdf.pth"))

# generate 3d data
for carfolder in tqdm(carfolder2latent_idx.keys()):
    carname = carfolder

    # loading latent code
    latent_idx = carfolder2latent_idx[carfolder]
    latent_code = latent_codes[latent_idx].unsqueeze(0).to(cfg.device)

    # generate 3d data
    create_mesh(
        model=model,
        latent=latent_code,
        output_file=f"../outputs/mesh/training-latent-codes/{carname}",
        N=cfg.train_latent_output_resolution,
    )
