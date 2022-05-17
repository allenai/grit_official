# Troubleshooting

Please check this document to see if we have already answered your question before raising an `Issue` in this repository. 

## Data Downloading

#### `RuntimeError: Could not find mongodb>=4.4`
- See https://voxel51.com/docs/fiftyone/getting_started/troubleshooting.html#alternative-linux-builds
- Depending on your operating system, you may need to change your fiftyone installation to match it
- E.g. for `RHEL 7` systems use `pip install fiftyone-db-rhel7`

## Submission

#### Why did my submission fail?
A common reason for submissions to fail is incorrect naming of submission files. Please make sure that you have:
- named the zip files and its contents correctly, 
- the zip files contains a single directory called `ablation` or `test` depending on the leaderboard you are submitting to,
- the submission includes a `params.json` file containing the number of parameters
- verfied the prediction format matches that of example submissions:
  - [GPV-1](https://ai2-prior-grit.s3.us-west-2.amazonaws.com/public/baselines/gpv/ablation.zip)/[GPV-2](https://ai2-prior-grit.s3.us-west-2.amazonaws.com/public/baselines/gpv2/ablation.zip): Categorization, Localization, VQA, RefExp
  - [Mask-RCNN](https://ai2-prior-grit.s3.us-west-2.amazonaws.com/public/baselines/maskrcnn/ablation.zip): Localization, Segmentation, Keypoints
  - [Normals Baseline](https://ai2-prior-grit.s3.us-west-2.amazonaws.com/public/baselines/uncertsn/ablation.zip): Surface Normal

For more submission details please refer to [`submission_format.md`](submission_format.md)
