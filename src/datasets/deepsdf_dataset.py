import os
import torch
import numpy as np
from torch.utils.data import Dataset


# 負の距離の座標と正の距離の座標をそれぞれsubsample個ずつランダムに取得
def unpack_sdf_samples(points, sdf, subsample=16000):
    points = torch.tensor(points).reshape(-1, 3)
    sdf = torch.tensor(sdf).reshape(-1, 1)
    samples = torch.cat([points, sdf], -1).reshape(-1, 4)

    pos_tensor = samples[samples[:, 3] > 0, :]
    neg_tensor = samples[samples[:, 3] < 0, :]

    half = int(subsample / 2)

    pos_idx = np.arange(0, len(pos_tensor))
    neg_idx = np.arange(0, len(neg_tensor))
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)

    sample_pos = pos_tensor[pos_idx[:half], :]
    sample_neg = neg_tensor[neg_idx[:half], :]

    samples = torch.cat([sample_pos, sample_neg], 0)

    xyz = samples[:, :3]
    sdf = samples[:, 3].reshape(-1, 1)

    return xyz, sdf


class DeepSDFDataset(Dataset):
    def __init__(self, points_path, carfolder2latent_idx, subsample=16000):
        super().__init__()
        self.subsample = subsample
        self.carfolder2latent_idx = carfolder2latent_idx

        self.idx2training_data = {}

        # read points.npy and sdf.npy
        for i, path in enumerate(points_path):
            assert os.path.exists(path), f"{path} does not exist."

            car_foldername = path.split("/")[-2]
            points_path = path
            sdf_path = path.replace("points.npy", "sdf_renew_by2.npy")

            points = np.load(points_path)
            sdf = np.load(sdf_path)

            self.idx2training_data[i] = {
                "car_foldername": car_foldername,
                "points": points,
                "sdf": sdf,
            }

    def __len__(self):
        return len(self.idx2training_data)

    def __getitem__(self, idx):
        car_foldername = self.idx2training_data[idx]["car_foldername"]
        points = self.idx2training_data[idx]["points"]
        sdf = self.idx2training_data[idx]["sdf"]

        points, sdf = unpack_sdf_samples(points, sdf, subsample=self.subsample)

        return {
            "car_foldername": car_foldername,
            "points": points,
            "sdf": sdf,
        }
