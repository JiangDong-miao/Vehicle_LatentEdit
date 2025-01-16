import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

sys.path.append("../")
from experimet_config import Config
from src.utils.utils import set_seed
from src.datasets.latent_navigation_dataset import GeometryNavigationDataset
from src.models.regressor import Regressor
from src.models.latent_walker import WalkMlpMultiW
from src.models.latent_walker import WalkKANMulti
from src.models.latent_walker import WalkEffKANMulti,WalkEffKAN
from src.utils.metric import regressor_criterion
# from kan import *
from tqdm import tqdm
import wandb



set_seed(42)
cfg = Config()


################# setting visualizer #################
print("Setting visualizer...")
print('regressor loss lambda:', cfg.geometry_walker_reg_lambda_)
print('content loss lambda:', cfg.geometry_walker_content_lambda_)




#################################
wandb.init(project="GeoWalker_sdf_re", config=cfg)



# loading geometry columns
geometry_columns = cfg.geometry_attribute

# loading training data
merge_df = pd.read_csv("../data/table/merged.csv").dropna().reset_index(drop=True)
merge_df = merge_df[~merge_df["folder_name"].isin(cfg.noise_data)][
    ["file_name", "folder_name"]
]

geometry_df = (
    pd.read_excel("../data/table/geometry.xlsx", index_col=0)
    .T.reset_index()
    .rename(columns={"index": "file_name"})
)

geometry_df["基調ライン基点長さ"] = geometry_df["基調ライン基点長さ"] * geometry_df["ホイールベース"]
geometry_df["前輪-Lの長さ"] = geometry_df["前輪-Lの長さ"] * geometry_df["ホイールベース"]
geometry_df["ノーズ高さ"] = geometry_df["ノーズ高さ"] * geometry_df["全高"]
geometry_df["ノーズスラント量"] = geometry_df["ノーズスラント量"] * geometry_df["フード長さ"]
geometry_df["基調ライン基点高さ"] = geometry_df["基調ライン基点高さ"] * geometry_df["全高"]
geometry_df["ベルトライン高さ"] = geometry_df["ベルトライン高さ"] * geometry_df["全高"]
geometry_df["フロントバンパー下端高さ"] = geometry_df["フロントバンパー下端高さ"] * geometry_df["全高"]
geometry_df["リアバンパー下端高さ"] = geometry_df["リアバンパー下端高さ"] * geometry_df["全高"]
geometry_df["サイドシル下端高さ"] = geometry_df["サイドシル下端高さ"] * geometry_df["全高"]
geometry_df["キャビン幅"] = geometry_df["キャビン幅"] * geometry_df["全幅"] * geometry_df["全高"]
geometry_df["ルーフ幅"] = geometry_df["ルーフ幅"] * geometry_df["全幅"] * geometry_df["全高"]
geometry_df["トレッド幅"] = geometry_df["トレッド幅"] * geometry_df["全幅"] * geometry_df["全高"]
geometry_df["バンパー下端幅"] = geometry_df["バンパー下端幅"] * geometry_df["全幅"] * geometry_df["全高"]
geometry_df["バンパー上端幅"] = geometry_df["バンパー上端幅"] * geometry_df["全幅"] * geometry_df["全高"]
geometry_df["ルーフ厚み"] = geometry_df["ルーフ厚み"] * geometry_df["全高"]
geometry_df["キャビン厚み"] = geometry_df["キャビン厚み"] * geometry_df["全高"]
geometry_df["ショルダー厚み"] = geometry_df["ショルダー厚み"] * geometry_df["全高"]
geometry_df["フード厚み"] = geometry_df["フード厚み"] * geometry_df["全高"]
geometry_df["バンパー厚み"] = geometry_df["バンパー厚み"] * geometry_df["全高"]

merge_df = (
    pd.merge(merge_df, geometry_df, on="file_name", how="left")
    .dropna()
    .reset_index(drop=True)
)

# 0~1にmin-max scaling
for col in geometry_columns:
    merge_df[col] = (merge_df[col] - merge_df[col].min()) / (
        merge_df[col].max() - merge_df[col].min()
    )

# loading carfolderlatent_idx, latent_idx2carfolder
with open("../outputs/data-split/carfolder2latent_idx.pickle", "rb") as f:
    carfolder2latent_idx = pickle.load(f)

with open("../outputs/data-split/latent_idx2carfolder.pickle", "rb") as f:
    latent_idx2carfolder = pickle.load(f)

# loading latent codes
latent_codes = torch.load("../outputs/latent-codes/latent_codes.pth")

# loading regressor model
regressor = (
    Regressor(
        input_dim=cfg.latent_code_dim,
        output_dim=len(geometry_columns),
    )
    .to(cfg.device)
    .eval()
)
regressor.load_state_dict(torch.load("../outputs/models/geometry_regressor.pth"))

# create latent walker model
# latent_walker = WalkMlpMultiW(
#     attribute_dim=len(geometry_columns), latent_code_dim=cfg.latent_code_dim
# ).to(cfg.device)


latent_walker = WalkEffKAN(
    attribute_dim=len(geometry_columns), latent_code_dim=cfg.latent_code_dim
).to(cfg.device)




# create dataset
train_dataset = GeometryNavigationDataset(geometry_columns, cfg.latent_code_dim)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=cfg.geometry_walker_batch_size,
    shuffle=True,
    drop_last=True,
)

# create optimizer
reg_criterion = regressor_criterion
content_critertion = nn.MSELoss()
optimizer = torch.optim.Adam(
    [
        {
            "params": latent_walker.parameters(),
            "lr": cfg.geometry_walker_initial_lr,
        },
    ]
)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=cfg.geometry_walker_warmup_epoch,
    num_training_steps=cfg.geometry_walker_epoch * len(train_dataloader),
)

# training
best_loss = np.inf
best_model = None

for epoch in tqdm(range(cfg.geometry_walker_epoch)):
    total_loss = 0.0
    latent_walker.train()
    for idx, batch in enumerate(train_dataloader):
        random_latent_code = batch["random_latent_code"].to(cfg.device)
        epsilon = batch["epsilon"].to(cfg.device)

        optimizer.zero_grad()

        alpha = regressor(random_latent_code)

        delta = torch.clip(alpha + epsilon, 0, 1) - alpha

        z_prime = latent_walker(random_latent_code, delta)

        alpha_prime = alpha + delta

        alpha_hat_prime = regressor(z_prime)
        

        reg_loss = reg_criterion(alpha_hat_prime, alpha_prime)
        content_loss = content_critertion(random_latent_code, z_prime)

        wandb.log({"reg_loss": reg_loss.item(), "content_loss": content_loss.item()})

        loss = (
            cfg.geometry_walker_reg_lambda_ * reg_loss
            + cfg.geometry_walker_content_lambda_ * content_loss
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    print(f"epoch: {epoch}, loss: {total_loss / len(train_dataloader)}")

    wandb.log({"epoch": epoch, "train_loss": total_loss / len(train_dataloader)})

    if best_loss > total_loss / len(train_dataloader):
        best_loss = total_loss / len(train_dataloader)
        best_model = latent_walker.state_dict()

    torch.save(best_model, "../outputs_m/models/geometry_walker_128_kan.pth"),
