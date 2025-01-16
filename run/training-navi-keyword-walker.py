import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

sys.path.append("../")
from experimet_config import Config
from src.utils.utils import set_seed
from src.datasets.latent_navigation_dataset import KeywordNavigationDataset
from src.models.regressor import Regressor
from src.models.latent_walker import WalkMlpMultiW
from src.models.latent_walker import WalkEffKANMulti,WalkEffKAN
from src.utils.metric import regressor_criterion
import wandb
set_seed(42)
cfg = Config()





wandb.init(project="StyleWalker", config=cfg)






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

# loading regressor model
regressor = (
    Regressor(
        input_dim=256,
        output_dim=len(keyword_columns),
    )
    .to(cfg.device)
    .eval()
)
regressor.load_state_dict(torch.load("../outputs_m/models/keyword_regressor_256.pth"))

# create latent walker model
latent_walker = WalkMlpMultiW(
    attribute_dim=len(keyword_columns), latent_code_dim=256
).to(cfg.device)

# create dataset
train_dataset = KeywordNavigationDataset(keyword_columns, cfg.latent_code_dim)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=cfg.keyword_walker_batch_size,
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
            "lr": cfg.keyword_walker_initial_lr,
        },
    ]
)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=cfg.keyword_walker_warmup_epoch,
    num_training_steps=cfg.keyword_walker_epoch * len(train_dataloader),
)

# training
best_loss = np.inf
best_model = None

for epoch in tqdm(range(cfg.keyword_walker_epoch)):
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

        wandb.log({
            "reg_loss": reg_loss.item(),
            "content_loss": content_loss.item(),
        })

        loss = (
            cfg.keyword_walker_reg_lambda_ * reg_loss
            + cfg.keyword_walker_content_lambda_ * content_loss
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()


    wandb.log({'epoch': epoch, 'train_loss': total_loss / len(train_dataloader)})

    print(f"epoch: {epoch}, loss: {total_loss / len(train_dataloader)}")

    if best_loss > total_loss / len(train_dataloader):
        best_loss = total_loss / len(train_dataloader)
        best_model = latent_walker.state_dict()

    torch.save(best_model, "../outputs_m/models/keyword_walker_mlp_256.pth"),
