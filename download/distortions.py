import os
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from utils.io import (mkdir_if_not_exists, extract_targz, download_from_url)

from grit_paths import GritPaths

log = logging.getLogger('__main__')


def download_distortions(cfg):
    mkdir_if_not_exists(cfg.grit.base,recursive=True)
    download_from_url(cfg.urls.images.distortions, cfg.grit.base)
    extract_targz(
        os.path.join(cfg.grit.base,'distortions.tar.gz'),
        cfg.grit.base)


@hydra.main(config_path='../configs',config_name='default')
def main(cfg: DictConfig):
    if cfg.prjpaths.data_dir is None or cfg.prjpaths.output_dir is None:
        print("Please provide data_dir and output_dir paths in `configs/prjpaths/default.yaml`")
        return
    log.debug('\n' + OmegaConf.to_yaml(cfg))
    download_distortions(cfg)
    

if __name__=='__main__':
    main()
