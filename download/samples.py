import os
import boto3
import hydra
from omegaconf import OmegaConf

import utils.io as io


@hydra.main(config_path='../configs',config_name='default')
def main(cfg):
    io.mkdir_if_not_exists(cfg.prjpaths.data_dir,recursive=True)
    print("Downloading GRIT samples...")
    io.download_from_url(cfg.urls.samples, cfg.prjpaths.data_dir)
    io.extract_zip(
        os.path.join(cfg.prjpaths.data_dir,'grit_data.zip'),
        cfg.prjpaths.data_dir)
    

if __name__=='__main__':
    main()