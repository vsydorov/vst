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
from thes.data.tubes.routines import (
    match_dwein_to_dgt)


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

    def set_defcfg_v2(cfg):
        cfg.set_defaults_yaml("""
        dataset:
            name: 'daly'
            cache_folder: ~
            mirror: 'uname'
            val_split:
                fraction: 0.1
                nsamplings: 20
                seed: 42
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


def sample_daly_frames_from_instances(
        dataset: Dataset_daly_ocv,
        stride: int,
        add_keyframes: bool = True,
        include_diff: bool = True,
        vids: List[Vid_daly] = None,
        ) -> Dict[Vid_daly, np.ndarray]:
    """
    Universal helper function to get a subset of daly frame numbers,
      corresponding to GT instance annotations
    Args:
        - stride: if > 0, will sample instance frames at this stride
        - keyframes: include annotated keyframes
        - include_diff: include difficult instances
    Returns:
        - frames numbers per each vid
    """
    if vids is None:
        vids = list(dataset.videos_ocv.keys())
    frames_to_cover: Dict[Vid_daly, np.ndarray] = {}
    for vid in vids:
        v = dataset.videos_ocv[vid]
        # general keyframe ranges of all instances
        instance_ranges = []
        for action_name, instances in v['instances'].items():
            for ins_ind, instance in enumerate(instances):
                fl = instance['flags']
                diff = fl['isReflection'] or fl['isAmbiguous']
                if not include_diff and diff:
                    continue
                s, e = instance['start_frame'], instance['end_frame']
                keyframes = [int(kf['frame'])
                        for kf in instance['keyframes']]
                instance_ranges.append((s, e, keyframes))
        good = set()
        for s, e, keyframes in instance_ranges:
            if add_keyframes:
                good |= set(keyframes)
            if stride > 0:
                good |= set(range(s, e+1, stride))
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

def group_dwein_frames_wrt_kf_distance(
        dataset, stride, tubes_dwein_train, tubes_dgt_train):
    # Keyframes by which we divide
    vids_kf_nums: Dict[Vid_daly, np.ndarray] = \
            sample_daly_frames_from_instances(dataset, stride=0)
    # Frames which we consider
    vids_good_nums: Dict[Vid_daly, np.ndarray] = \
            sample_daly_frames_from_instances(dataset, stride=stride)
    # keyframes per dwt
    dwtis_kf_nums = {}
    dwtis_good_nums = {}
    for dwt_index, dwt in tubes_dwein_train.items():
        (vid, bunch_id, tube_id) = dwt_index
        s, e = dwt['start_frame'], dwt['end_frame']
        # kf_nums
        vid_kf_nums = vids_kf_nums[vid]
        vid_kf_nums = vid_kf_nums[
                (vid_kf_nums >= s) & (vid_kf_nums <= e)]
        dwtis_kf_nums[dwt_index] = vid_kf_nums
        # good_nums
        vid_good_nums = vids_good_nums[vid]
        vid_good_nums = vid_good_nums[
                (vid_good_nums >= s) & (vid_good_nums <= e)]
        dwtis_good_nums[dwt_index] = vid_good_nums

    # Associate tubes
    best_hits: Dict[I_dwein, Tuple[I_dgt, float]] = match_dwein_to_dgt(
        tubes_dgt_train, tubes_dwein_train)

    # Compute distance, disperse
    dist_boxes = {}
    for dwti, dwt in tubes_dwein_train.items():
        dwt = tubes_dwein_train[dwti]
        kf_nums = dwtis_kf_nums[dwti]
        good_nums = dwtis_good_nums[dwti]
        # Distance matrix
        M = np.zeros((len(kf_nums), len(good_nums)), dtype=np.int)
        for i, kf in enumerate(kf_nums):
            M[i] = np.abs(good_nums-kf)
        best_dist = M.min(axis=0)
        # Associate kf_nums and boxes
        assert np.in1d(good_nums, dwt['frame_inds'].tolist()).all()
        rel_inds = np.searchsorted(dwt['frame_inds'], good_nums)
        boxes = [dwt['boxes'][j] for j in rel_inds]

        # Determine foreground vs background
        foreground = False
        if dwti in best_hits:
            dgti, iou = best_hits[dwti]
            if iou >= 0.5:
                foreground = True

        # Disperse boxes per dist
        for f0, box, dist in zip(good_nums, boxes, best_dist):
            metabox = {
                'frame_ind': f0, 'box': box, 'dwti': dwti}
            if foreground:
                metabox.update({'kind': 'fg', 'dgti': dgti, 'iou': iou})
            else:
                metabox.update({'kind': 'bg'})
            dist_boxes.setdefault(dist, []).append(metabox)
    dist_boxes = dict(sorted(list(dist_boxes.items())))
    return dist_boxes
