import os
import torch
import numpy as np
from torch.utils.data import Dataset


class KeywordRegressorDataset(Dataset):
    def __init__(self, keyword_columns, merge_df, latent_codes, carfolder2latent_idx):
        super().__init__()
        self.merge_df = merge_df.dropna().reset_index(drop=True)
        self.latent_codes = latent_codes
        self.carfolder2latent_idx = carfolder2latent_idx
        self.carfolders = list(carfolder2latent_idx.keys())
        self.keyword_columns = keyword_columns

    def __len__(self):
        return len(self.merge_df)

    def __getitem__(self, idx):
        row = self.merge_df.loc[idx]
        carfolder = row["folder_name"]
        latent_idx = self.carfolder2latent_idx[carfolder]
        latent_code = self.latent_codes[latent_idx].unsqueeze(0).to("cuda")
        keyword = row[self.keyword_columns].values.astype(np.float32)

        return {
            "latent_code": latent_code,
            "keyword": keyword,
        }


class KeywordNavigationDataset(Dataset):
    def __init__(self, keyword_columns, latent_code_dim):
        super().__init__()
        self.keyword_columns = keyword_columns
        self.latent_code_dim = latent_code_dim

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        random_noise_latent_code = torch.normal(0, 1e-1, size=(1, self.latent_code_dim))
        random_noise_latent_code = random_noise_latent_code.squeeze(0).to("cuda")

        epsilon = 2 * torch.rand(1, len(self.keyword_columns)) - 1  # 一様分布からサンプリング
        epsilon = epsilon.squeeze(0).to("cuda")

        return {
            "random_latent_code": random_noise_latent_code,
            "epsilon": epsilon,
        }


class GeometryRegressorDataset(Dataset):
    def __init__(
        self, geometry_columns, geometry_df, latent_codes, carfolder2latent_idx
    ):
        super().__init__()
        self.geometry_df = geometry_df.dropna().reset_index(drop=True)
        self.latent_codes = latent_codes
        self.carfolder2latent_idx = carfolder2latent_idx
        self.carfolders = list(carfolder2latent_idx.keys())
        self.geometry_columns = geometry_columns

    def __len__(self):
        return len(self.geometry_df)

    def __getitem__(self, index):
        row = self.geometry_df.loc[index]
        carfolder = row["folder_name"]
        latent_idx = self.carfolder2latent_idx[carfolder]
        latent_code = self.latent_codes[latent_idx].unsqueeze(0).to("cuda")
        geometry = row[self.geometry_columns].values.astype(np.float32)

        return {
            "latent_code": latent_code,
            "geometry": geometry,
        }


class GeometryNavigationDataset(Dataset):
    def __init__(self, geometry_columns, latent_code_dim):
        super().__init__()
        self.geometry_columns = geometry_columns
        self.latent_code_dim = latent_code_dim

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        random_noise_latent_code = torch.normal(0, 1e-1, size=(1, self.latent_code_dim))
        random_noise_latent_code = random_noise_latent_code.squeeze(0).to("cuda")

        epsilon = 2 * torch.rand(1, len(self.geometry_columns)) - 1  # 一様分布からサンプリング
        epsilon = epsilon.squeeze(0).to("cuda")

        return {
            "random_latent_code": random_noise_latent_code,
            "epsilon": epsilon,
        }
