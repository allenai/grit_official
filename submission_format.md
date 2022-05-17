# Submission Format

You will need to prepare a single `ablation.zip` or `test.zip` file depending on the subset of GRIT you want to evaluate on. The zip files should contain a single directory called `ablation/` or `test/` respectively. 

**Note** - If the zip file or the directory name is not as described above, your submission would fail. 

The following is the expected directory structure inside the `ablation.zip` file (replace `ablation` by `test` everywhere for `test.zip`) where each task json file contains predictions for the respective task and `params.json` contains parameter count in millions. The `normals/` directory contains each normal prediction saved as an RGB image. 
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

If your model does not make predictions for a particular task, simply omit the correponding json file from the directory. The format for each of the json files are described below.

## params.json
This contains a dictionary with a single key-value pair
```
{
    "params_in_millions": int
}
```

## task.json
Each task json contains a list of dicts, a dict per sample. Each dict contains the following key-value pairs.
```
[
    {
        "example_id" : str
        "confidence" : float in [0,1]
        "words"      : str
        "bboxes"     : 2d list of int [[x1,y1,x2,y2],...]               # box coordinates, per instance
        "masks"      : list of dict   [{counts:"",shape:[]},...]        # rle encoded binary masks, per instance
        "points"     : 2d list of int [[x1,y1,v1,...,x17,y17,v17],...]  # 17 keypoint locations and visibility, per instance
        "normal"     : str                                              # normal image path (e.g. "0.png" for file `normals/0.png`)
    }
]
```

Not all keys are required to be present for each task. Here are the tasks that require specific keys to be present:
* `example_id`: all tasks
* `confidence`: all tasks
* `words`: categorization, vqa
* `bboxes`: localization, refexp
* `masks`: segmentation (to convert to RLE format, use the functions provided in [utils/rle.py](https://github.com/allenai/grit_official/blob/main/utils/rle.py))
* `points`: keypoint
* `normal`: normal

**Note** - For the keypoints task, the predicted visibility (`v1,v2,...`) is currently not used in the evaluation, however the ground truth visibility is used. For each keypoint, `v` takes one of three values `{0,1,2}` corresponding to `{'not labeled', 'labeled but not visible', 'labeled and visible'}`. If your model doesn't predict visibility you may use any of these values in your prediction file without affecting your score. 
