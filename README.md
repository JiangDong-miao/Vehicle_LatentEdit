# Vehicle_LatentEdit



## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).  
You are free to share and adapt the content for non-commercial purposes, provided that you give appropriate credit to the author.

## Purpose of Release

This repository is shared to accompany the submission of our manuscript titled **"Fine-Grained 3D Vehicle Shape Manipulation via Latent Space Editing"** to **[The visual computer]**. 


## Dataset Preparation

1. Please place all 3D models used for training into one folder, which can contain various subfolders.

## Training Process

1. **SDF Sampling**:
   - Run `run/obj2sdf.py` to sample SDF values for all models.

2. **Train DeepSDF Model**:
   - Use `run/training-deepsdf.py` to train a DeepSDF model based on your dataset. This script will also store the latent codes representing each model.

3. **Train Latent Code Regressor**:
   - With the latent codes obtained from the previous step, train a regressor for the latent codes targeting the attributes involved in the editing process.
   - Example: `run/training-navi-geometry-regressor.py`. This step will produce a regressor that predicts attributes from a given latent code.

4. **Train Fine-Grained 3D Editing Framework**:
   - Use `run/training-navi-geometry-walker.py`.

## Inference

- After training the editing model, you can load the previously fitted latent codes of existing 3D models from `notebook/walk-geometry(keyword)/walk-geometry-latent.ipynb`.
- In this notebook, specify the attributes you want to edit and the intensity of the edits. The notebook will generate a series of 3D models edited at the specified granularity.

## Visualization

- Finally, use `visualize.ipynb` in the same folder to visualize all generated models.
