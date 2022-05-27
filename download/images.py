import os
import h5py
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import wget
from tqdm import tqdm
import subprocess
from zipfile import ZipFile
from PIL import Image
import fiftyone.zoo as foz
from utils.io import (list_dir, mkdir_if_not_exists, extract_zip, extract_targz,
    download_from_url, load_json_object)
from grit_paths import GritPaths
import shutil 

log = logging.getLogger('__main__')


def download_coco(cfg):
    img_dir = f'{cfg.grit.images}/coco'
    mkdir_if_not_exists(img_dir,recursive=True)

    # download test2015
    download_from_url(cfg.urls.images.coco.test2015,img_dir)
    extract_zip(f'{img_dir}/test2015.zip', img_dir)

    # download train2014 for refcoco
    download_from_url(cfg.urls.images.coco.train2014,img_dir)
    extract_zip(f'{img_dir}/train2014.zip', img_dir)


def download_refclef(cfg):
    img_dir = f'{cfg.grit.images}/refclef'
    mkdir_if_not_exists(img_dir,recursive=True)

    download_from_url(cfg.urls.images.refclef, img_dir)
    extract_zip(f'{img_dir}/saiapr_tc-12.zip', img_dir)


def download_construction(cfg):
    mkdir_if_not_exists(cfg.grit.images,recursive=True)
    download_from_url(cfg.urls.images.construction, cfg.grit.images)
    extract_targz(
        f'{cfg.grit.images}/construction_images.tar.gz',
        cfg.grit.images)
    

def download_open_images(cfg):
    img_dir = f'{cfg.grit.images}/open_images'
    mkdir_if_not_exists(img_dir,recursive=True)
    image_ids = set()
    for task in ['localization','categorization','segmentation']:
        grit_paths = GritPaths(cfg.grit.base)
        for subset in ['ablation','test']:
            samples = load_json_object(grit_paths.samples(task, subset))
            image_ids.update(
                [os.path.splitext(os.path.basename(s['image_id']))[0] \
                    for s in samples if 'open_images' in s['image_id']])
    
    foz.download_zoo_dataset(
        "open-images-v6",
        dataset_dir=img_dir,
        split='test',
        image_ids=image_ids)

    src_dir = f'{img_dir}/test/data'
    tgt_dir = f'{img_dir}/test'
    # subprocess.call(
    #     f'mv {src_dir}/* {tgt_dir}',
    #     shell=True)
    allfiles = os.listdir(src_dir)
    for f in allfiles:
        srcfile = os.path.join(src_dir,f)
        tgtfile = os.path.join(tgt_dir,f)
        shutil.move(srcfile, tgtfile)
    os.rmdir(src_dir)
    
    subprocess.call(
        f'rm -rf {img_dir}/test/data & rm -rf {img_dir}/test/metadata & rm -rf {img_dir}/test/labels',
        shell=True)


def download_visual_genome(cfg):
    img_dir = f'{cfg.grit.images}/visual_genome'
    mkdir_if_not_exists(img_dir,recursive=True)
    image_ids = set()
    grit_paths = GritPaths(cfg.grit.base)
    for subset in ['ablation','test']:
        samples = load_json_object(grit_paths.samples('vqa', subset))
        image_ids.update([
            '/'.join(s['image_id'].split('/')[1:]) for s in samples if 'visual_genome' in s['image_id']])
    
    for image_id in image_ids:
        subdir = os.path.join(img_dir,image_id.split('/')[0])
        mkdir_if_not_exists(subdir,recursive=True)
        download_from_url(
            f'https://cs.stanford.edu/people/rak248/{image_id}',
            subdir)


def save_nyuv2(cfg):
    img_dir = f'{cfg.grit.images}/nyuv2'
    mkdir_if_not_exists(img_dir,recursive=True)

    download_from_url(cfg.urls.images.nyuv2, img_dir)
    f = h5py.File(os.path.join(img_dir,'nyu_depth_v2_labeled.mat'),'r')
    images = f['images'][()].transpose(0,3,2,1)
    num_images = images.shape[0]
    for i in tqdm(range(num_images)):
        im = Image.fromarray(images[i])
        im.save(f'{img_dir}/{i+1}.jpg')


def download_blended_mvs(cfg):
    mkdir_if_not_exists(cfg.grit.images,recursive=True)

    download_from_url(cfg.urls.images.blended_mvs,cfg.grit.images)
    extract_targz(
        os.path.join(cfg.grit.images,'blended_mvs_images.tar.gz'),
        cfg.grit.images)


def download_dtu(cfg):
    mkdir_if_not_exists(cfg.grit.images,recursive=True)

    download_from_url(cfg.urls.images.dtu,cfg.grit.images)
    extract_targz(
        os.path.join(cfg.grit.images,'dtu_images.tar.gz'),
        cfg.grit.images)
    

def download_scannet(cfg):
    print("ATTENTION - scannet has not been downloaded")
    print("You must sign a Terms of Service agreement before downloading scannet manually")
    print("Instructions can be found at https://github.com/allenai/grit_official/blob/main/download/scannet_download_instructions.md")

@hydra.main(config_path='../configs',config_name='default')
def main(cfg: DictConfig):
    if cfg.prjpaths.data_dir is None or cfg.prjpaths.output_dir is None:
        print("Please provide data_dir and output_dir paths in `configs/prjpaths/default.yaml`")
        return
    log.debug('\n' + OmegaConf.to_yaml(cfg))
    
    for dataset in cfg.datasets_to_download:  
        print(f"\n\nDownloading {dataset}...")  
        if dataset=='coco':
            download_coco(cfg)
        elif dataset=='construction':
            download_construction(cfg)
        elif dataset=='refclef':
            download_refclef(cfg)
        elif dataset=='open_images':
            download_open_images(cfg)
        elif dataset=='nyuv2':
            save_nyuv2(cfg)
        elif dataset=='visual_genome':
            download_visual_genome(cfg)
        elif dataset=='blended_mvs':
            download_blended_mvs(cfg)
        elif dataset=='scannet':
            download_scannet(cfg)
        elif dataset=='dtu':
            download_dtu(cfg)
        else:
            raise NotImplementedError
    

if __name__=='__main__':
    main()
