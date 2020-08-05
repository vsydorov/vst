import numpy as np
from typing import (  # NOQA
    Dict, Any, List, Optional, Tuple, TypedDict, Set)

from thes.data.dataset.external import (
    Vid_daly, Dataset_daly_ocv,
    get_daly_split_vids, split_off_validation_set)

from thes.data.tubes.types import (
    I_dwein, T_dwein, T_dwein_scored, I_dgt, T_dgt,
    get_daly_gt_tubes, remove_hard_dgt_tubes,
    loadconvert_tubes_dwein, dtindex_filter_split)


class Ncfg_daly:
    @staticmethod
    def set_defcfg(cfg):
        cfg.set_deftype("""
        dataset:
            name: ['daly', ['daly']]
            cache_folder: [~, str]
            mirror: ['uname', ~]
            val_split:
                fraction: [0.1, float]
                nsamplings: [20, int]
                seed: [42, int]
        """)

    @staticmethod
    def get_dataset(cf):
        dataset = Dataset_daly_ocv(cf['dataset.mirror'])
        dataset.populate_from_folder(cf['dataset.cache_folder'])
        return dataset

    @staticmethod
    def get_vids(cf, dataset) -> Dict[str, List[Vid_daly]]:
        v_fraction = cf['dataset.val_split.fraction']
        v_nsamplings = cf['dataset.val_split.nsamplings']
        v_seed = cf['dataset.val_split.seed']

        val, train = split_off_validation_set(
                dataset, v_fraction, v_nsamplings, v_seed)
        vgroup = {
            'train': train,
            'val': val,
            'trainval': get_daly_split_vids(dataset, 'train'),
            'test': get_daly_split_vids(dataset, 'test'),
        }
        return vgroup


def get_daly_keyframes_to_cover(
        dataset, vids, add_keyframes: bool, every_n: int,
        ) -> Dict[Vid_daly, np.ndarray]:
    frames_to_cover: Dict[Vid_daly, np.ndarray] = {}
    for vid in vids:
        v = dataset.videos_ocv[vid]
        # general keyframe ranges of all instances
        instance_ranges = []
        for action_name, instances in v['instances'].items():
            for ins_ind, instance in enumerate(instances):
                s, e = instance['start_frame'], instance['end_frame']
                keyframes = [int(kf['frame'])
                        for kf in instance['keyframes']]
                instance_ranges.append((s, e, keyframes))
        good = set()
        for s, e, keyframes in instance_ranges:
            if add_keyframes:
                good |= set(keyframes)
            if every_n > 0:
                good |= set(range(s, e+1, every_n))
        frames_to_cover[vid] = np.array(sorted(good))
    return frames_to_cover


def load_gt_and_wein_tubes(tubes_dwein_fold, dataset, vgroup):
    # / Load tubes
    tubes_dwein_all: Dict[I_dwein, T_dwein] = \
            loadconvert_tubes_dwein(tubes_dwein_fold)
    tubes_dgt_all: Dict[I_dgt, T_dgt] = get_daly_gt_tubes(dataset)
    tubes_dgt_all = remove_hard_dgt_tubes(tubes_dgt_all)
    # // Per subset
    tubes_dwein_d = {}
    tubes_dgt_d = {}
    for sset, vids in vgroup.items():
        tubes_dwein_d[sset] = dtindex_filter_split(tubes_dwein_all, vids)
        tubes_dgt_d[sset] = dtindex_filter_split(tubes_dgt_all, vids)
    return tubes_dwein_d, tubes_dgt_d


def create_keyframelist(dataset):
    # Record keyframes
    keyframes = []
    for vid, ovideo in dataset.videos_ocv.items():
        nframes = ovideo['nframes']
        for action_name, instances in ovideo['instances'].items():
            for ins_ind, instance in enumerate(instances):
                fl = instance['flags']
                diff = fl['isReflection'] or fl['isAmbiguous']
                if diff:
                    continue
                for kf_ind, keyframe in enumerate(instance['keyframes']):
                    frame0 = keyframe['frame']
                    action_id = dataset.action_names.index(action_name)
                    kf_dict = {
                            'vid': vid,
                            'action_id': action_id,
                            'action_name': action_name,
                            'ins_ind': ins_ind,
                            'kf_ind': kf_ind,
                            'bbox': keyframe['bbox_abs'],
                            'video_path': ovideo['path'],
                            'frame0': int(frame0),
                            'nframes': nframes,
                            'height': ovideo['height'],
                            'width': ovideo['width'],
                            }
                    keyframes.append(kf_dict)
    return keyframes


def to_keyframedict(keyframes):
    result = {}
    for kf in keyframes:
        kf_ind = (kf['vid'], kf['action_name'],
                kf['ins_ind'], kf['kf_ind'])
        result[kf_ind] = kf
    return result
