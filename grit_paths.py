import os

class GritPaths:
    tasks = [
        'categorization',
        'localization',
        'vqa',
        'refexp',
        'segmentation',
        'keypoint',
        'normal'
    ]

    def __init__(self,base_dir):
        self.base_dir = base_dir

    def samples(self,task,subset):
        return os.path.join(
            self.base_dir,
            f'samples/{subset}/{task}.json')

    def dist_deltas(self,task,subset):
        return os.path.join(
            self.base_dir,
            f'distortions/dist_deltas_{task}_{subset}.hdf5')

    def output_options(self,name):
        return os.path.join(
            self.base_dir,
            f'output_options/{name}.json')