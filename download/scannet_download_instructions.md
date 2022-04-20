# Instructions to download GRIT split for ScanNet

Reference: https://github.com/ScanNet/ScanNet#scannet-data

1) Fill out the ScanNet Terms of Use and send it to scannet@googlegroups.com
2) Once accepted you will recieve a link to a `download-scannet.py` script
3) Run `python download-scannet.py -o [output dir] --grit` to download `ScanNet-GRIT.zip` to `[output dir]`
4) Extract compressed files to the GRIT images directory specified in the [`default.yaml`](../configs/default.yaml) configuration 
