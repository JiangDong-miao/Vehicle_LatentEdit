import os
import sys
import trimesh
import numpy as np
import pandas as pd
from glob import glob
from tqdm.auto import tqdm

from mesh_to_sdf import sample_sdf_near_surface

number_of_points = 4000000

car_file_dir = "../data/From_Ikeda/**/*.obj"

filepaths = glob(car_file_dir, recursive=True)

import random

random.shuffle(filepaths)

error_files = []

for name in tqdm(filepaths):
    try:
        print(name)
        # name = name.replace(".stl", "")
        # name = name + ".obj"

        file_path = name
        # 最後以外を結合してディレクトリパスを作成

        points_output_path = os.path.join(os.path.dirname(file_path), "points_4times.npy")
        sdf_output_path = os.path.join(os.path.dirname(file_path), "sdf_4times.npy")
        colors_output_path = os.path.join(os.path.dirname(file_path), "colors_4times.npy")
        if os.path.exists(points_output_path) and os.path.exists(sdf_output_path):
            continue

        mesh = trimesh.load(name)
        points, sdf = sample_sdf_near_surface(
            mesh,
            number_of_points=number_of_points,
            surface_point_method="scan",
            sign_method="normal",
            scan_count=100,
            scan_resolution=400,
            sample_point_count=10000000,
            normal_sample_count=200,
            min_size=0,
            return_gradients=False,
        )
        print(points, sdf)
        np.save(points_output_path, points)
        np.save(sdf_output_path, sdf)
    except:
        print("Error: ", name)
        error_files.append(name)

print("Error files: ", error_files)
