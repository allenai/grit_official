import os
import h5py
import hydra
import copy
import numpy as np
from PIL import Image

import utils.io as io
from grit_paths import GritPaths

@hydra.main(config_path='configs',config_name='default')
def main(cfg):
    if cfg.subsets_to_distort is not None:
        subsets_to_distort = cfg.subsets_to_distort
    else: 
        subsets_to_distort = ['ablation','test']

    for subset in subsets_to_distort:
        print(subset)
        grit_paths = GritPaths(cfg.grit.base)
        for task in cfg.tasks_to_distort:
            print(f'- {task}')
            samples = io.load_json_object(grit_paths.samples(task,subset))
            samples = [s for s in samples if 'distorted' in s['image_id']]

            deltas = h5py.File(grit_paths.dist_deltas(task,subset),'r')
            for sample in samples:
                example_id = sample['example_id']
                delta = deltas[example_id[:-5]][()]
                dist_image_id = sample['image_id']
                undist_image_id = '/'.join(dist_image_id.split('/')[2:])

                img = Image.open(os.path.join(
                    cfg.grit.images,
                    undist_image_id)).convert('RGB')
                dist_img = np.asarray(img) + delta
                dist_img = Image.fromarray(dist_img.astype(np.uint8))
                dist_img_path = os.path.join(
                    cfg.grit.images,
                    dist_image_id)
                
                io.mkdir_if_not_exists(
                    os.path.dirname(dist_img_path),
                    recursive=True)
                dist_img.save(dist_img_path)


if __name__=='__main__':
    main()