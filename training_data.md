# Training Data
We provide a few suggestions on how to setup training data for GRIT. We focus here on how to get data and annotations from datasets used in GRIT that you are allowed to train on.

## Surface Normals
Many of the surface normals in GRIT have been created from some other type of 3D data (e.g. depth, point clouds). Therefore, to get surface normals for the images in the GRIT dataset, you either need to generate them yourself or download the normals from someone else who has already done this. 
### ScanNet
For ScanNet images, we use surface normals created by the [FrameNet](https://github.com/hjwdzh/FrameNet) folks. Use the provided [download script](https://github.com/hjwdzh/FrameNet/blob/master/src/data/download.sh): 
- Downloads a bunch of zips from FrameNet. These are listed in [scannet-frame-links.txt](http://download.cs.stanford.edu/orion/framenet/scannet-frame-links.txt)
  - http://download.cs.stanford.edu/orion/framenet/scannet-frame/scene0016_02.zip
  - http://download.cs.stanford.edu/orion/framenet/scannet-frame/scene0631_02.zip
  - http://download.cs.stanford.edu/orion/framenet/scannet-frame/scene0748_00.zip
  - … and so on
- The zips contains frames from the scene (at intervals of 10) 
  - `scene0016_02`  has frames range(0,930,10)
  - `scene0575_00` has frames range(0,2930,10)
- Each frame has 6 variants saved as .png images
  - `scannet-frames/scene0016_02/frame-000000-color.png`
  - `scannet-frames/scene0016_02/frame-000000-normal.png`
  - `scannet-frames/scene0016_02/frame-000000-orient-X.png`
  - `scannet-frames/scene0016_02/frame-000000-orient-Y.png`
  - `scannet-frames/scene0016_02/frame-000000-orient-mask.png`
  - `scannet-frames/scene0016_02/frame-000000-orient.png`
- You only need to keep the `color.png` and `normal.png` for this task
- Remember not to train on images in the GRIT `ablation` or `test`
