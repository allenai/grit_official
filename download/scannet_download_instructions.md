# Instructions to download GRIT split for ScanNet

Reference: https://github.com/ScanNet/ScanNet#scannet-data

1) Fill out the [ScanNet Terms of Use](http://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf) and send it to scannet@googlegroups.com
3) Once accepted you will recieve a link to a `download-scannet.py` script from the ScanNet dataset owners
4) Run `python download-scannet.py -o [output dir] --grit` to download `ScanNet-GRIT.zip` to `[output dir]/tasks`
5) Extract compressed files to the GRIT images directory 
    -  unzip ScanNet-GRIT.zip -d `{prjpaths.data_dir}/GRIT/images` 
    -  `{prjpaths.data_dir}` is the value specified in the [`/configs/prjpaths/default.yaml`](../configs/prjpaths/default.yaml) configuration

Note: Use python2 to run the `download-scannet.py` script. 

If you are using python3, this requires changing references of raw_input to input, urllib.urlopen to urllib.request.urlopen,  urllib.urlretrieve to urllib.request.urlretrievet and add import urllib.request
