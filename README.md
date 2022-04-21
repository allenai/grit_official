# **GRIT**: **G**eneral **R**obust **I**mage **T**ask Benchmark
[[Project Website](https://allenai.org/project/grit/home) | [Arxiv Paper ](url) | [Github](https://github.com/allenai/grit_official/)] By [Tanmay Gupta](http://tanmaygupta.info/), [Ryan Marten](https://www.ryanmarten.com/), [Aniruddha Kembhavi](https://anikem.github.io/), and [Derek Hoiem](https://dhoiem.cs.illinois.edu/) 

This repository provides various tools and resources for evaluating vision and vision-language models on the GRIT benchmark. Specifically, we include:

- Scripts to download the `ablation` and `test` set samples (images, annoations, additive distortion maps)
- Script to create distorted images
- Description and implementation of evaluation metrics
- Instructions for submitting predictions and analyzing performance on the leaderboard
- Utility functions for reading and visualizing sample inputs and predictions 



## GRIT Tasks
GRIT includes 7 tasks. The following shows example inputs and outputs for each task:

<p align="left">
    <img src="data/teaser.png" width="500">
</p>

## Install dependencies
```
conda create -n grit python=3.9 -y
conda activate grit
pip install -r requirements.txt
```

## Configuration
We use Hydra to specify configuration parameters. You will need to specify in [`configs/prjpaths/default.yaml`](configs/prjpaths/default.yaml): 
- `data_dir`: path to the directory where you want to download GRIT data
- `output_dir`: path to the directory where you want to save output logs

## Setup data
This involves two steps. 

First, download samples, images, and additive distortion maps by running the following command. You can specify which datasets to download images for with the `datasets_to_download` parameter list in [`configs/default.yaml`](configs/default.yaml).
```
bash download.sh   
```


Second, create distorted images by running 
```
python -m generate_distortions
```
You may control which datasets to download images from through `datasets_to_download` parameters in [`configs/default.yaml`](configs/default.yaml).
You may specify which tasks to generate distorted images for through `tasks_to_distort` parameter in [`configs/default.yaml`](configs/default.yaml). 
 
## Input data format
Once downloaded, the GRIT evaluation data should look as follows:
```
GRIT/
|--images/                                  # contains images used in all tasks from various sources
|--distortions/                             # contains distortion delta maps
|--output_options/                          # contains answer option candidates for categorization
|  |--coco_categories.json
|  |--nyuv2_categories.json
|  |--open_images_categories.json
|--samples/                             
|  |--ablation/                             # contains ablation samples for each task 
|  |  |--categorization.json
|  |  |--localization.json
|  |  |--vqa.json
|  |  |--refexp.json
|  |  |--segmentation.json
|  |  |--keypoint.json
|  |  |--normal.json
|  |--test/                                 # contains test samples for each task (similar to ablation/)
```
Each of the seven tasks in GRIT require slightly different inputs and outputs. However, the tasks samples are stored in a json format following a consistent schema across tasks. Specifically, each json is a list of dicts with the following keys and values.
```
[
    {
        "example_id"     : str              # unique sample identifier 
        "image_id"       : str              # relative path to input image
        "task_name"      : str              # specifies the task such as "vqa", "refexp"
        "task_query"     : str              # object category, question, referring expressions, or null 
        "task_bbox"      : [x1,y1,x2,y2]    # input box coordinates for categorization task or null
        "output_options" : str              # answer options to select from e.g "coco_categories" or null
    }
]
```

# GRIT Leaderboards

GRIT provides the following 4 leaderboards depending on the evaluation subset (`ablation` or `test`) and training data restrictions (`restricted` or `unrestricted`):
- https://leaderboard.allenai.org/grit-ablation-restricted
- https://leaderboard.allenai.org/grit-ablation-unrestricted
- https://leaderboard.allenai.org/grit-test-restricted
- https://leaderboard.allenai.org/grit-test-unrestricted


## Submission format
You will need to prepare a single `ablation.zip` or `test.zip` file depending on the subset of GRIT you want to evaluate on. The following is the expected directory structure inside the `ablation.zip` file (replace `ablation` by `test` everywhere for `test.zip`) where each task json file contains predictions for the respective task and `params.json` contains parameter count in millions. The `normals/` directory contains each normal prediction saved as an RGB image. 
```
ablation/
|--params.json
|--categorization.json
|--localization.json
|--refexp.json
|--vqa.json
|--segmentation.json
|--keypoint.json
|--normal.json
|--normals/
```

If your model does not make predictions for a particular task, simply omit the correponding json file from the directory. The format of each of these files is described in [`submission_format.md`](submission_format.md).

## Scoring
In GRIT, various measures are computed per sample and aggregated across a subset of data points depending on the concept group, partition, and task of interest.
* **Nomenclature.** Metrics in GRIT follow the following 4 part nomenclature - *measure.cgroup.partition.task*. Given the large number of metrics, we recommend using the regex filters to select a subset of metrics to view. For example, "acc.any.newSrc" shows the measure "accuracy" computed across samples belonging to "any" concept group on "newSrc" partition on all tasks. To keep the number of metrics manageable, for cgroups other than "any" (e.g. people, tools), only "agg" partition is supported. Also note that the measure "rmse" is not computed for partition "deldist" as it is undefined. 
* **Confidence Intervals.** The superscript and subscript denote the 95% confidence interval for the mean statistic. We use t-interval for all metrics except metrics with task "all" or partition "agg" where we use z-interval for simplicity. 
* **Ranking.** The overall ranking is based on "overall" which is the same as "acc.any.agg.all" for each access. However, GRIT may be used to study a wide variety of properties of vision models and hence the leaderboard allows sorting models by any of the available metric. 
* **Downloading.** Leaderboard allows users to download all computed metrics for a particular submission or all public submissions for more finegrained programmatic analysis. 

## Rules
All public submissions to the GRIT leadeboard must adhere to the following rules for our research community to benefit from fair assessment of generality and robustness of vision models on the GRIT benchmark:
* **Training Data**: Models may not train on any data sources used to create the GRIT ablation and test sets except those identified as primary training source for each task. E.g. for categorization COCO train set is the primary training source, while ablation and test sets are created from COCO test-reserve, NYU v2 test set, and Open Images v6 test set. In the restricted setting, models are further allowed to use 5 auxiliary training sources: ImageNet, Conceptual Captions, Visual Genome (except any VQA annotations which are used for DCE-VQA, one of the data sources used for creating ablation and test sets for GRIT VQA), and Web10k. In the unrestricted setting, in addition to the above sources, models are free to use any other data source that does not have a known significant overlap with the GRIT ablation and test sources. Please refer to [Table 3 and Table 4 in the paper](url). 
* **Data source metadata**: Models may not use data source metadata information of the sample during training or inference. Eg. identifying which data source a sample belongs to from metadata and using a separate model for each source or inputting data source to the model during inference is not allowed. 
* **Distortion metadata**: Models may not use distortion metadata information for the samples during training or inference. E.g inferring whether a sample is distorted or inferring the type of distortion from the metadata to aid the model during training or inference is not allowed.
* **Reporting Parameter Counts**: Leaderboard participants are required to submit accurate parameters counts for their models. Parameter counts must include all model parameters including parameters of any model used to precompute image or text features. If a model uses a pretrained detector with D parameters to cache image features followed by a model that shares S parameters across tasks while using 2 task-specialized output heads with H1 and H2 parameters, the reportable parameter count is D + S + H1 + H2.
* **Submission Frequency**: GRIT ablation leaderboards allow unlimited private and public submissions. These are meant to be used for model ablation studies. GRIT test leaderboards are meant to be used for final evaluation and comparison between different models and hence only allow hidden private submissions and 1 public submission per 7 day period. The hidden private submission may be used to test whether the submitted files are correctly processed but they need to be made public (by using the "publish" button) in order to view the results.
* **Anonymity**: Public anonymous submissions are discouraged. However, leaderboard participants may create anonymous public submissions while waiting for conference review decisions. If so, the authors may use anonymous placeholders for name, email, and contributors fields in the submission form during the review cycle while clearly indicating paper id and conference name in the description field for reviewers to cross-reference the results. The description must also include a date when the authors intend to de-anonymize the results. Anonymous submission that are past the de-anonymization due date or those that do not meet any of the above criterion may be removed by the leaderboard creators. 

## Metrics and Evaluation
To help users understand the metrics and evaluation better, we provide metrics for each task under [`metrics/`](metrics/) and the evaluation script running on the leaderboard as [`evaluate.py`](evaluate.py). You may not be able to run the evaluation script as is because GRIT doesn't release meta data and ground truth for our ablation and test sets. However, it may still be beneficial for leaderboard participants to help understand how the numerous metrics on the GRIT leaderboards are computed. Participants may use our metric implementations to evaluate predictions on their selected training and validation data. 

## Troubleshooting

See [`troubleshoot.md`](troubleshoot.md)
