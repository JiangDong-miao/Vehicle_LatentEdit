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
from src.datasets.latent_navigation_dataset import GeometryRegressorDataset
from src.models.regressor import Regressor
import wandb
from sklearn.metrics import mean_squared_error, r2_score


set_seed(42)
cfg = Config()




wandb.init(project="Geo_Regressor_sdf", config=cfg)



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



geometry_df["Key Line Base Length"] = geometry_df["Key Line Base Length"] * geometry_df["Wheelbase"]
geometry_df["Front Wheel-L Length"] = geometry_df["Front Wheel-L Length"] * geometry_df["Wheelbase"]
geometry_df["Nose Height"] = geometry_df["Nose Height"] * geometry_df["Total Height"]
geometry_df["Nose Slant Amount"] = geometry_df["Nose Slant Amount"] * geometry_df["Hood Length"]
geometry_df["Key Line Base Height"] = geometry_df["Key Line Base Height"] * geometry_df["Total Height"]
geometry_df["Belt Line Height"] = geometry_df["Belt Line Height"] * geometry_df["Total Height"]
geometry_df["Front Bumper Lower Edge Height"] = geometry_df["Front Bumper Lower Edge Height"] * geometry_df["Total Height"]
geometry_df["Rear Bumper Lower Edge Height"] = geometry_df["Rear Bumper Lower Edge Height"] * geometry_df["Total Height"]
geometry_df["Side Sill Lower Edge Height"] = geometry_df["Side Sill Lower Edge Height"] * geometry_df["Total Height"]
geometry_df["Cabin Width"] = geometry_df["Cabin Width"] * geometry_df["Overall Width"]
geometry_df["Roof Width"] = geometry_df["Roof Width"] * geometry_df["Overall Width"]
geometry_df["Tread Width"] = geometry_df["Tread Width"] * geometry_df["Overall Width"]
geometry_df["Bumper Lower Edge Width"] = geometry_df["Bumper Lower Edge Width"] * geometry_df["Overall Width"]
geometry_df["Bumper Upper Edge Width"] = geometry_df["Bumper Upper Edge Width"] * geometry_df["Overall Width"]
geometry_df["Roof Thickness"] = geometry_df["Roof Thickness"] * geometry_df["Total Height"]
geometry_df["Cabin Thickness"] = geometry_df["Cabin Thickness"] * geometry_df["Total Height"]
geometry_df["Shoulder Thickness"] = geometry_df["Shoulder Thickness"] * geometry_df["Total Height"]
geometry_df["Hood Thickness"] = geometry_df["Hood Thickness"] * geometry_df["Total Height"]
geometry_df["Bumper Thickness"] = geometry_df["Bumper Thickness"] * geometry_df["Total Height"]
 

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
with open("../outputs_m/data-split/carfolder2latent_idx_512.pickle", "rb") as f:
    carfolder2latent_idx = pickle.load(f)

with open("../outputs_m/data-split/latent_idx2carfolder_512.pickle", "rb") as f:
    latent_idx2carfolder = pickle.load(f)

# loading latent codes
latent_codes = torch.load("../outputs_m/latent-codes/latent_codes_512.pth")

# data split

valid_df = merge_df.sample(n=20, random_state=42)
train_df = merge_df.drop(valid_df.index)


# creatge dataset
train_dataset = GeometryRegressorDataset(
    geometry_columns, train_df, latent_codes, carfolder2latent_idx
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    drop_last=True,
)

valid_dataset = GeometryRegressorDataset(
    geometry_columns, valid_df, latent_codes, carfolder2latent_idx
)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    drop_last=True,
)


# create model
regressor = Regressor(
    input_dim=512,
    output_dim=len(geometry_columns),
).to(cfg.device)

# create optimizer
optimizer = torch.optim.Adam(
    [
        {
            "params": regressor.parameters(),
            "lr": 1e-3,
        },
    ]
)

# create scheduler
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=cfg.geometry_reg_warmup_epoch,
    num_training_steps=cfg.geometry_reg_epoch * len(train_dataloader),
)

# create loss function
criterion = nn.MSELoss()

# training regressor
best_loss = np.inf
best_model = None

for epoch in range(cfg.geometry_reg_epoch):
    # train
    regressor.train()
    train_loss = 0
    for batch in train_dataloader:
        latent_code = batch["latent_code"].squeeze().to(cfg.device)
        geometry = batch["geometry"].squeeze().to(cfg.device)

        optimizer.zero_grad()
        pred_geometry = regressor(latent_code)
        loss = criterion(pred_geometry, geometry)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()

    # validation
    regressor.eval()
    valid_loss = 0
    valid_preds = []
    valid_targets = []
    for batch in valid_dataloader:
        latent_code = batch["latent_code"].squeeze().to(cfg.device)
        keyword = batch["geometry"].squeeze().to(cfg.device)

        with torch.no_grad():
            pred_keyword = regressor(latent_code)
            loss = criterion(pred_keyword, keyword)
            valid_preds.append(pred_keyword.cpu().numpy())
            valid_targets.append(keyword.cpu().numpy())

        valid_loss += loss.item()



    # 在循环结束后，将列表转换为 NumPy 数组
    valid_preds = np.concatenate(valid_preds, axis=0)
    valid_targets = np.concatenate(valid_targets, axis=0)
    # 计算每个标签的MSE和R²，然后取平均
    mse_scores = []
    r2_scores = []

    for i, label in enumerate(geometry_columns):  # 假设valid_targets是二维的，且第二维是标签维
        mse = mean_squared_error(valid_targets[:, i], valid_preds[:, i])
        r2 = r2_score(valid_targets[:, i], valid_preds[:, i])
        mse_scores.append(mse)
        r2_scores.append(r2)
        wandb.log({f"{label}_mse": mse, f"{label}_r2": r2})

    avg_mse = np.mean(mse_scores)
    avg_r2 = np.mean(r2_scores)
    rmse = np.sqrt(avg_mse)

    print(f"Epoch: {epoch}, Average Validation MSE: {avg_mse}, RMSE: {rmse}, Average R² Score: {avg_r2}")
    wandb.log({"epoch": epoch, "valid_mse": avg_mse, "valid_rmse": rmse, "valid_r2": avg_r2})


    print(
        f"epoch: {epoch}, train_loss: {train_loss / len(train_dataloader)}, valid_loss: {valid_loss / len(valid_dataloader)}"
    )

    wandb.log({"epoch": epoch, "train_loss": train_loss / len(train_dataloader), "valid_loss": valid_loss / len(valid_dataloader)})

    if best_loss > valid_loss:
        best_loss = valid_loss
        best_model = regressor.state_dict()
        #torch.save(best_model, "../outputs_m/models/geometry_regressor_128_test.pth")

print(f"best_loss: {best_loss/len(valid_dataloader)}")
