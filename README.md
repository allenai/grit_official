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
```