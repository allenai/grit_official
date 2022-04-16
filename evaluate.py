import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.stats as scstats
from skimage.io import imread
import zipfile
import warnings
warnings.filterwarnings('error')

from metrics.vqa import vqa_accuracy
from metrics.localization import loc_metric
from metrics.categorization import accuracy
from metrics.segmentation import seg_metric
from metrics.keypoint import kp_metric
from metrics.refexp import refexp_metric
from metrics.normal import sn_metric, get_mask_from_normals
from utils.io import load_json_object, dump_json_object, mkdir_if_not_exists, dumps_json_object
import utils.rle as rle


parser = argparse.ArgumentParser()
parser.add_argument('--pred_zip',type=str,default=None)
parser.add_argument('--pred_dir',type=str,default='/predictions')
parser.add_argument('--grit_samples_dir',type=str)
parser.add_argument('--grit_normals_dir',type=str)
parser.add_argument('--lemma2groups',type=str)
parser.add_argument('--subset',type=str)
parser.add_argument('--outdir',type=str)


TASKS_ABREV = dict(
    categorization='cat',
    localization='loc',
    vqa='vqa',
    refexp='ref',
    segmentation='seg',
    keypoint='kp',
    normal='sn',
    overall='all'
)
TASKS = [k for k in TASKS_ABREV if k!='overall']
NUM_TASKS = len(TASKS)

GROUPS = [
    'people',
    'body_parts',
    'birds',
    'animals',
    'plants',
    'furniture',
    'structure',
    'clothing',
    'clothing_accessories',
    'vehicles',
    'transport_infrastructure',
    'musical_instruments',
    'food',
    'beverages',
    'technology',
    'bathroom_objects',
    'kitchen_objects',
    'household_objects',
    'stationery',
    'sports_equipment',
    'places',
    'tools',
    'brands',
    'natural_landscape',
]

def compute_metric(pred,sample,task,args):
    if task=='categorization':
        return accuracy(pred['words'], sample['output']['words'][0])

    elif task=='localization':
        return loc_metric(pred['bboxes'], sample['output']['bboxes'])

    elif task=='refexp':
        return refexp_metric(pred['bboxes'], sample['output']['bboxes'])

    elif task=='vqa':
        return vqa_accuracy([pred['words']], sample['output']['words'], 1)

    elif task=='segmentation':
        stuff = "(Stuff)" in sample['input']['task_query']
        pred_masks = rle.decode(pred['masks'])
        gt_masks = rle.decode(sample['output']['masks'])
        return seg_metric(pred_masks, gt_masks, stuff)

    elif task=='keypoint':
        return kp_metric(pred['points'], sample['output']['points'])

    elif task=='normal':
        source = sample['meta']['data_source']
        gt_normals_path = os.path.join(
            args.grit_normals_dir,
            sample['output']['out_image_name'])
        gt_normals_rgb = imread(gt_normals_path)[:,:,:3]
        valid_mask = get_mask_from_normals(gt_normals_rgb)
        pred_normals_path = os.path.join(
            args.pred_dir,
            args.subset,
            'normals',
            pred['normal'])
        pred_normals_rgb = imread(pred_normals_path)[:,:,:3]
        return sn_metric(pred_normals_rgb, gt_normals_rgb, valid_mask)

    else:
        raise NotImplementedError


def get_metric(df,metric,task,cgroup,partition):
    df = df[(df.task==task) & (df.cgroup==cgroup)]

    if partition in ['sameSrc','newSrc']:
        df = df[(df.src==partition) & (df.dist=='undist')]
    elif partition in ['sameCpt','newCpt']:
        df = df[(df.cpt==partition) & (df.dist=='undist')]
    elif partition in ['dist']:
        df = df[(df.dist==partition)]
    elif partition in ['undist']:
        df = df[df.undistp==True]
    elif partition=='deldist':
        df_dist = df[df.dist=='dist']
        r_dist = {r['example_id'][:-5]:r[metric] for r in df_dist.to_records()}
        df_undistp = df[df.undistp==True]
        r_undistp = {r['example_id']:r[metric] for r in df_undistp.to_records()}
        
        records = []
        for idx,undistp_metric in r_undistp.items():
            record  = {'example_id': idx}
            record[metric] = undistp_metric - r_dist[idx]
            records.append(record)

        df = pd.DataFrame.from_records(records,columns=['example_id',metric])
        
    else:
        raise NotImplementedError()

    samples = df[metric]

    N = len(samples)
    if N <= 1:
        if N==1:
            mean = samples.mean()
        else:
            mean = None

        return mean,None,None,None,N

    mean = samples.mean()
    var = samples.var()
    lower,upper = scstats.t.interval(0.95,N-1,mean,samples.sem()+1e-6)
    
    return mean,lower,upper,var,N


def get_parallel_undist_example_ids(samples):
    N = len('_dist')
    return {s['input']['example_id'][:-N] \
        for s in samples if s['meta']['is_distorted']}


def compute_avg_bounds(mus,vs,ns):
    d = np.sqrt(np.sum([v/n for v,n in zip(vs,ns)])) / len(ns)
    mean = np.mean(mus)
    upper = mean + 1.96*d
    lower = mean - 1.96*d
    return mean, upper, lower


def compute_sample_metrics(args):
    if args.pred_zip is not None:
        with zipfile.ZipFile(args.pred_zip,'r') as f:
            f.extractall(args.pred_dir)

    params = load_json_object(
        os.path.join(
            args.pred_dir,
            args.subset,
            'params.json'))['params_in_millions']
    
    records = []
    missing_tasks = set()
    for task in TASKS:
        samples_json = os.path.join(
            args.grit_samples_dir,
            args.subset,
            f'{task}.json')
        if not os.path.exists(samples_json):
            raise FileNotFoundError(f'{samples_json} does not exist')

        pred_json = os.path.join(
            args.pred_dir,
            args.subset,
            f'{task}.json')
        if not os.path.exists(pred_json):
            print(f'{pred_json} does not exist')
            missing_tasks.add(task)
            continue
        
        lemma2groups = load_json_object(args.lemma2groups)
        samples = load_json_object(samples_json)
        undistp_example_ids = get_parallel_undist_example_ids(samples)
        preds = load_json_object(pred_json)
        preds = {pred['example_id']:pred for pred in preds}

        task_metric = []
        same_src_metric = []
        for sample in tqdm(samples):
            pred = preds[sample['output']['example_id']]
            metric = compute_metric(pred,sample,task,args)
            task_metric.append(metric)
            if sample['meta']['is_new_source'] is False:
                same_src_metric.append(metric)

            concepts = set()
            groups = set()
            for c in sample['meta']['concepts']:
                concepts.update(c['lemma'])
                groups.update(lemma2groups.get(c['lemma'],['_ungrouped_']))

            conf = pred['confidence']
            
            src = 'sameSrc'
            if sample['meta']['is_new_source']:
                src = 'newSrc'

            cpt = 'sameCpt'
            if sample['meta']['has_new_concept']:
                cpt = 'newCpt'

            dist = 'undist'
            if sample['meta']['is_distorted']:
                dist = 'dist'
            
            undistp = False
            if sample['input']['example_id'] in undistp_example_ids:
                undistp = True

            records.append(dict(
                example_id=sample['input']['example_id'],
                task=task,
                src=src,
                cpt=cpt,
                dist=dist,
                undistp=undistp,
                cgroup='any',
                acc=100*metric,
                inf=100*conf*metric,
                misinf=100*conf*(1-metric),
                conf=100*(conf),
                sa=100*(conf*metric + (1-conf)*(1-metric)),
                rmse=(100*(conf-metric))**2
            ))

            for cg in groups:
                if cg=='_ungrouped_':
                    continue

                records.append(dict(
                    example_id=sample['input']['example_id'],
                    task=task,
                    src=src,
                    cpt=cpt,
                    dist=dist,
                    undistp=undistp,
                    cgroup=cg,
                    acc=100*metric,
                    inf=100*conf*metric,
                    misinf=100*conf*(1-metric),
                    conf=100*(conf),
                    sa=100*(conf*metric + (1-conf)*(1-metric)),
                    rmse=(100*(conf-metric))**2
                ))

    df = pd.DataFrame.from_records(records)

    metrics = dict()
    metric_vars = dict()
    for metric in ['acc','inf','misinf','conf','sa','rmse']:
        for cgroup in ['any',*GROUPS]:
            for partition in ['sameSrc','newSrc','agg','sameCpt','newCpt','dist','undist','deldist']:
                if cgroup!='any':
                    # sameSrc and newSrc are needed for computing agg
                    if metric=='acc' and partition in ['sameSrc','newSrc','agg']:    
                        pass
                    else:
                        continue

                if metric=='rmse' and partition=='deldist':
                    continue

                for task in TASKS:
                    task_abrev = TASKS_ABREV[task]
                    metric_name = f'{metric}.{cgroup}.{partition}.{task_abrev}'

                    if partition=='agg':
                        m1 = f'{metric}.{cgroup}.sameSrc.{task_abrev}'
                        m2 = f'{metric}.{cgroup}.newSrc.{task_abrev}'
                        if f'{m1}.mean' not in metrics or \
                            f'{m2}.mean' not in metrics:
                            continue

                        mus = [metrics[f'{m1}.mean'],metrics[f'{m2}.mean']]
                        vs = [metric_vars[m1],metric_vars[m2]]
                        ns = [metrics[f'{m1}.cnt'],metrics[f'{m2}.cnt']]
                        mean, upper, lower = compute_avg_bounds(mus, vs, ns)
                    
                    else:
                        mean,lower,upper,var,cnt = get_metric(df, metric, task, cgroup, partition)
                        if None in [mean,lower,upper,var]:
                            continue
                            
                        metrics[f'{metric_name}.cnt'] = cnt
                        metric_vars[metric_name] = var


                    metrics[f'{metric_name}.mean'] = mean
                    metrics[f'{metric_name}.upper'] = upper
                    metrics[f'{metric_name}.lower'] = lower


    for metric in ['acc','inf','misinf','conf','sa','rmse']:
        for partition in ['agg','sameSrc','newSrc','dist','undist','deldist']:
            if metric=='rmse' and partition=='deldist':
                continue

            mus = []
            vs = []
            ns = []
            cnts = dict()
            for task in TASKS:
                task_abrev = TASKS_ABREV[task]
                if task in missing_tasks and partition=='agg':
                    mus.extend([0,0])
                    vs.extend([0,0])
                    ns.extend([100,100]) # doesn't matter what number you put here since variance is 0

                elif task in missing_tasks:
                    mus.append(0)
                    vs.append(0)
                    ns.append(100) # doesn't matter what number you put here since variance is 0

                elif partition=='agg':
                    for p in ['sameSrc','newSrc']:
                        mus.append(metrics[f'{metric}.any.{p}.{task_abrev}.mean'])
                        vs.append(metric_vars[f'{metric}.any.{p}.{task_abrev}'])
                        ns.append(metrics[f'{metric}.any.{p}.{task_abrev}.cnt'])
                    
                else:
                    p = partition
                    mus.append(metrics[f'{metric}.any.{p}.{task_abrev}.mean'])
                    vs.append(metric_vars[f'{metric}.any.{p}.{task_abrev}'])
                    ns.append(metrics[f'{metric}.any.{p}.{task_abrev}.cnt'])

            mean, upper, lower = compute_avg_bounds(mus, vs, ns)

            metrics[f'{metric}.any.{partition}.all.mean'] = mean
            metrics[f'{metric}.any.{partition}.all.upper'] = upper
            metrics[f'{metric}.any.{partition}.all.lower'] = lower
                    
    overall_metric = 'acc.any.agg.all'
    metrics['overall.mean'] = metrics[f'{overall_metric}.mean']
    metrics['overall.upper'] = metrics[f'{overall_metric}.upper']
    metrics['overall.lower'] = metrics[f'{overall_metric}.lower']
    metrics['params'] = params

    for cgroup in GROUPS:
        for partition in ['sameSrc','newSrc']:
            for task in TASKS:
                task_abrev = TASKS_ABREV[task]
                for stat in ['mean','upper','lower','cnt','var']:
                    metrics.pop(f'acc.{cgroup}.{partition}.{task_abrev}.{stat}',None)

    for k,v in list(metrics.items()):
        if k=='params':
            metrics[k] = int(v)
            print(k,metrics[k])
            continue

        if 'overall' in k:
            _,stat = k.split('.')
            partition = 'overall'
        else:
            metric,cgroup,partition,task,stat = k.split('.')
            if metric=='rmse' and stat in ['mean','upper','lower']:
                if stat=='lower':
                    if partition!='deldist':
                        v = max(0,v)

                v = v ** 0.5
        
        if partition=='deldist':
            metrics[k] = round(v,4)
            continue
        
        if stat == 'upper':
            metrics[k] = round(min(100,v),4)
        
        elif stat == 'lower':
            metrics[k] = round(max(0,v),4)
        
        elif stat == 'mean':
            metrics[k] = round(v,4)

        elif stat == 'var':
            metrics.pop(k,None)
        
    print(dumps_json_object(metrics,indent=4))

    mkdir_if_not_exists(args.outdir)
    dump_json_object(
        metrics,
        os.path.join(args.outdir,f'{args.subset}_metrics.json'))


if __name__=='__main__':
    args = parser.parse_args()
    compute_sample_metrics(args)