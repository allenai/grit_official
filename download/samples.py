import os
import boto3
import hydra
from omegaconf import OmegaConf

import utils.io as io


@hydra.main(config_path='../configs',config_name='default')
def main(cfg):
    if cfg.prjpaths.data_dir is None or cfg.prjpaths.output_dir is None:
        print("Please provide data_dir and output_dir paths in `configs/prjpaths/default.yaml`")
        return

    io.mkdir_if_not_exists(cfg.prjpaths.data_dir,recursive=True)
    print("Downloading GRIT samples...")
    io.download_from_url(cfg.urls.samples, cfg.prjpaths.data_dir)
    io.extract_zip(
        os.path.join(cfg.prjpaths.data_dir,'grit_data.zip'),
        cfg.prjpaths.data_dir)
    

if __name__=='__main__':
    main()