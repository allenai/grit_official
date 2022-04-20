# Instructions to download GRIT split for ScanNet

Reference: https://github.com/ScanNet/ScanNet#scannet-data

1) Fill out the ScanNet Terms of Use and send it to scannet@googlegroups.com
2) Once accepted you will recieve a link to a `download-scannet.py` script from the ScanNet dataset owners
3) If you are using python3, change `raw_input` to `input`, `urllib.urlopen` to `urllib.request.urlopen`,  `urllib.urlretrieve` to `urllib.request.urlretrieve` and add `import urllib.request`
4) Run `python download-scannet.py -o [output dir] --grit` to download `ScanNet-GRIT.zip` to `[output dir]/tasks`
5) Extract compressed files to the GRIT images directory specified in the [`default.yaml`](../configs/default.yaml) configuration (e.g. `unzip ScanNet-GRIT.zip -d /scratch/marten4/grit_official_data/GRIT/images` 
