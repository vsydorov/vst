"""
All path management delegated to this module
Temporary solution which makes code immovable. Will be fixed later
"""
from pathlib import Path

__ROOT = Path('/home/vlad/projects/dervo_deployed/2020_10_12_Chanel')


def get_dataset_path(data_id):
    return __ROOT/'links/datasets'/data_id
