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
from src.datasets.latent_navigation_dataset import KeywordRegressorDataset
from src.models.regressor import Regressor
import wandb
from sklearn.metrics import mean_squared_error, r2_score




set_seed(42)
cfg = Config()

wandb.init(project="Style_Regressor", config=cfg)
# loading keyword columns
keyword_columns = cfg.keyword_attribute

# loading training data
merge_df = pd.read_csv("../data/table/merged.csv").dropna().reset_index(drop=True)
merge_df = merge_df[~merge_df["folder_name"].isin(cfg.noise_data)]
for col in keyword_columns:
    # 0~1にmin-max scaling
    merge_df[col] = (merge_df[col] - merge_df[col].min()) / (
        merge_df[col].max() - merge_df[col].min()
    )

# loading carfolderlatent_idx, latent_idx2carfolder
with open("../outputs_m/data-split/carfolder2latent_idx_256.pickle", "rb") as f:
    carfolder2latent_idx = pickle.load(f)

with open("../outputs_m/data-split/latent_idx2carfolder_256.pickle", "rb") as f:
    latent_idx2carfolder = pickle.load(f)

# loading latent codes
latent_codes = torch.load("../outputs_m/latent-codes/latent_codes_256.pth")

# data split
# train_df, valid_df = train_test_split(
#     merge_df, test_size=0.2, random_state=42, shuffle=True
# )

valid_df = merge_df.sample(n=20, random_state=42)
train_df = merge_df.drop(valid_df.index)


# creatge dataset
train_dataset = KeywordRegressorDataset(
    keyword_columns, train_df, latent_codes, carfolder2latent_idx
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    drop_last=True,
)

valid_dataset = KeywordRegressorDataset(
    keyword_columns, valid_df, latent_codes, carfolder2latent_idx
)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    drop_last=True,
)

# create model
regressor = Regressor(
    input_dim=256,
    output_dim=len(keyword_columns),
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
    num_warmup_steps=cfg.keyword_reg_warmup_epoch,
    num_training_steps=cfg.keyword_reg_epoch * len(train_dataloader),
)

# create loss function
criterion = nn.MSELoss()

# training regressor
best_loss = np.inf
best_model = None




for epoch in range(cfg.keyword_reg_epoch):
    # train
    regressor.train()
    train_loss = 0
    for batch in train_dataloader:
        latent_code = batch["latent_code"].squeeze().to(cfg.device)
        keyword = batch["keyword"].squeeze().to(cfg.device)

        optimizer.zero_grad()
        pred_keyword = regressor(latent_code)
        loss = criterion(pred_keyword, keyword)
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
        keyword = batch["keyword"].squeeze().to(cfg.device)

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

    for i, label in enumerate(keyword_columns):  # 假设valid_targets是二维的，且第二维是标签维
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
    wandb.log({"train_loss": train_loss / len(train_dataloader), "valid_loss": valid_loss / len(valid_dataloader)})



    if best_loss > valid_loss:
        best_loss = valid_loss
        best_model = regressor.state_dict()
        torch.save(best_model, "../outputs_m/models/keyword_regressor_256.pth")

print(f"best_loss: {best_loss/len(valid_dataloader)}")
