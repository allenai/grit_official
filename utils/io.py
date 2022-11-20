import os
import pickle
import json
import yaml
import numpy as np
import gzip
from zipfile import ZipFile
import wget
import tarfile


def download_from_url(url,out_dir):
    wget.download(url,out=out_dir)


def extract_zip(zip_file_path,out_dir):
    with ZipFile(zip_file_path,'r') as f:
        f.extractall(out_dir)

def extract_targz(targz_file_path,out_dir):
    with tarfile.open(targz_file_path) as f:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(f, out_dir)

def load_pickle_object(file_name, compress=True):
    data = read(file_name)
    if compress:
        load_object = pickle.loads(gzip.decompress(data))
    else:
        load_object = pickle.loads(data)
    return load_object


def dump_pickle_object(dump_object, file_name, compress=True, compress_level=9):
    data = pickle.dumps(dump_object)
    if compress:
        write(file_name, gzip.compress(data, compresslevel=compress_level))
    else:
        write(file_name, data)


def load_json_object(file_name, compress=False):
    if compress:
        return json.loads(gzip.decompress(read(file_name)).decode('utf8'))
    else:
        return json.loads(read(file_name, 'r'))


def dump_json_object(dump_object, file_name, compress=False, indent=4, sort_keys=True):
    data = json.dumps(
        dump_object, cls=NumpyAwareJSONEncoder, sort_keys=sort_keys, indent=indent)
    if compress:
        write(file_name, gzip.compress(data.encode('utf8')))
    else:
        write(file_name, data, 'w')


def dumps_json_object(dump_object, indent=4, sort_keys=True):
    data = json.dumps(
        dump_object, cls=NumpyAwareJSONEncoder, sort_keys=sort_keys, indent=indent)
    return data


def load_yaml_object(file_name):
    return yaml.load(read(file_name, 'r'))


def read(file_name, mode='rb'):
    with open(file_name, mode) as f:
        return f.read()


def write(file_name, data, mode='wb'):
    with open(file_name, mode) as f:
        f.write(data)


def serialize_object(in_obj, method='json'):
    if method == 'json':
        return json.dumps(in_obj)
    else:
        return pickle.dumps(in_obj)


def deserialize_object(obj_str, method='json'):
    if method == 'json':
        return json.loads(obj_str)
    else:
        return pickle.loads(obj_str)

def list_dir(dir_name):
    if os.path.exists(dir_name) and not os.path.isfile(dir_name):
        return os.listdir(dir_name)
    return []

def mkdir_if_not_exists(dir_name, recursive=False):
    if os.path.exists(dir_name):
        return

    if recursive:
        os.makedirs(dir_name)
    else:
        os.mkdir(dir_name)
        
    
class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.ndim == 1:
                return obj.tolist()
            else:
                return [self.default(obj[i]) for i in range(obj.shape[0])]
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.int16):
            return int(obj)
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.float16):
            return float(obj)
        elif isinstance(obj, np.uint64):
            return int(obj)
        elif isinstance(obj, np.uint32):
            return int(obj)
        elif isinstance(obj, np.uint16):
            return int(obj)
        return json.JSONEncoder.default(self, obj)