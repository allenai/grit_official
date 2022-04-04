# **GRIT**: **G**eneral **R**obust **I**mage **T**ask Benchmark

This repository provides various tools and resources for evaluating vision and vision-language models on the GRIT benchmark. Specifically, we include:

- Scripts to download the `ablation` and `test` set samples (images, annoations, additive distortion maps)
- Script to create distorted images
- Description and implementation of evaluation metrics
- Instructions for submitting predictions and analyzing performance on the leaderboard
- Utility functions for reading and visualizing sample inputs and predictions 


## Install dependencies
```
conda create -n grit python=3.9 -y
conda activate grit
pip install -r requirements.txt
pip install torch==1.11.0+cpu torchvision==0.12.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
```

## Configuration
We use Hydra to specify configuration parameters through `configs/default.yaml` file. You may need to specify the following parameters in `default.yaml`:
- `data_dir`: path to the directory where you want to download GRIT data
- `output_dir`: path to the directory where you want to save output logs

## Setup data
This involves two steps. 

First, download samples, images, and additive distortion maps by running
```
bash download.sh   
```
You may specify which datasets to download images for through `datasets_to_download` parameter in `configs/default.yaml`. Note that downloading scannet may take quite some time, so only download if evaluating on surface normal prediction task.  

Second, create distorted images by running 
```
python -m generate_distortions
```
You may control which datasets to download images from through `datasets_to_download` paramters in `configs/default.yaml`.
You may specify which tasks to generate distorted images for through `tasks_to_distort` parameter in `configs/default.yaml`

## Troubleshooting
`RuntimeError: Could not find mongod>=4.4`
See https://voxel51.com/docs/fiftyone/getting_started/troubleshooting.html#alternative-linux-builds
Depending on your system, you may need to change the type of fiftyone installed on your system.
E.g. for `RHEL 7` systems use `pip install fiftyone-db-rhel7`
