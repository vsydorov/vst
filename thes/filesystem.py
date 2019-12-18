"""
All path management delegated to this module
Temporary solution which makes code immovable. Will be fixed later
"""
from pathlib import Path

__ROOT = Path('/home/vsydorov/projects/deployed/2019_09_CVPR_Video')


def get_dataset_path(data_id):
    return __ROOT/'links/datasets'/data_id


def get_cache_path():
    return __ROOT/'links/horus/sink/cache'


def get_testcache_path():
    return __ROOT/'links/gpuhost7/sink/test_cache'
