import os
import sys
import pickle
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from glob import glob
from experimet_config import Config
import wandb

sys.path.append("../")
from src.models.deepsdf import DeepSDF
from src.utils.utils import car_foldername_latent_code_connetction, set_seed
from src.datasets.deepsdf_dataset import DeepSDFDataset
from src.utils.metric import DeepSDFLoss, mse

cfg = Config()
set_seed(42)





# loading training data path
points_path = glob("../data/From_Ikeda/**/points.npy")

latent_codes = nn.Parameter(
    torch.normal(
        0, 1e-4, size=(len(points_path), cfg.latent_code_dim)
    )  # (car_model_num, latent_code_dim)
)

model = DeepSDF(
    hidden_dim=cfg.hidden_dim,
    xyz_pos_enc_dim=cfg.xyz_pos_enc_dim,
    latent_code_dim=512,
    dropout_prob=cfg.dropout_prob,
).to(cfg.device)

latent_idx2carfolder, carfolder2latent_idx = car_foldername_latent_code_connetction(
    latent_codes, points_path
)


dataset = DeepSDFDataset(
    points_path, carfolder2latent_idx, subsample=cfg.sample_per_scene
)
dataloader = DataLoader(
    dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=True, drop_last=True
)

criterion = DeepSDFLoss(delta=cfg.clamping_distance)
optimizer = torch.optim.Adam(
    [
        {
            "params": model.parameters(),
            "lr": cfg.deepsdf_initial_lr,
        },
        {
            "params": latent_codes,
            "lr": cfg.latent_code_inital_lr,
        },
    ]
)

warmup_steps = int(len(dataloader) * cfg.warmup_epoch)
num_training_steps = int(len(dataloader) * cfg.n_epochs)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=num_training_steps,
)


# let's train
best_val_mse = np.inf
best_model = None
minimum_mse = np.inf


wandb.init(project="DeepSDF-ori-256", config=cfg)




for epoch in range(cfg.n_epochs):
    model.train()
    total_loss = []
    total_mse = []
    total_cd = []
    total_emd = []
    tq = tqdm(dataloader)

    for data in dataloader:
        xyz = (
            data["points"].float().cuda().chunk(cfg.batch_size)
        )  # [[sample_per_car, 3], [sample_per_car, 3], ...]]
        sdf = (
            data["sdf"].float().cuda().chunk(cfg.batch_size)
        )  # [[sample_per_car, 1], [sample_per_car, 1], ...]]
        car_foldername = data["car_foldername"]  # [batch_car]
        num_sdf_samples = (
            cfg.sample_per_scene * cfg.batch_size
        )  # sample_per_car * batch_car

        optimizer.zero_grad()
        batch_loss = []
        batch_mse = []

        for i in range(cfg.batch_size):
            latent_code = (
                latent_codes[carfolder2latent_idx[car_foldername[i]], :]
                .to(cfg.device)
                .unsqueeze(0)
            )  # [1, latent_code_dim]
            onecar_xyz = xyz[i].squeeze(0).to(cfg.device)  # [sample_per_car, 3]
            onecar_sdf = sdf[i].squeeze(0).to(cfg.device)
            onecar_sdf_pred = model(latent_code, onecar_xyz)  # [sample_per_car, 1]

            loss = criterion(onecar_sdf_pred, onecar_sdf) / num_sdf_samples
            loss += cfg.latent_code_regularization * torch.sum(
                torch.norm(latent_code, dim=1)
            )

            loss.backward()

            monitor_loss = mse(onecar_sdf_pred, onecar_sdf)
            batch_loss.append(loss.item())
            batch_mse.append(monitor_loss)

        optimizer.step()
        scheduler.step()

        total_loss.append(np.mean(batch_loss))
        total_mse.append(np.mean(batch_mse))

        # Log the loss to Weights and Biases
        wandb.log({"epoch": epoch, "loss": np.mean(total_loss), "mse": np.mean(total_mse),"min-mse": minimum_mse})



        tq.update()
        tq.set_postfix(
            {
                "epoch": epoch,
                "loss": np.mean(total_loss),
                "mse": np.mean(total_mse),
                "minimum_mse": minimum_mse,
            }
        )
    tq.close()

    if minimum_mse > np.mean(total_mse):
        minimum_mse = np.mean(total_mse)
        torch.save(model.state_dict(), "../outputs_m/models/deepsdf_512.pth")
        torch.save(latent_codes, "../outputs_m/latent-codes/latent_codes_512.pth")

    # save
    with open("../outputs_m/data-split/latent_idx2carfolder_512.pickle", "wb") as f:
        pickle.dump(latent_idx2carfolder, f)

    with open("../outputs_m/data-split/carfolder2latent_idx_512.pickle", "wb") as f:
        pickle.dump(carfolder2latent_idx, f)
