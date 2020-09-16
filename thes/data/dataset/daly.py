import numpy as np
import logging
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
    get_dwein_overlaps_per_dgt,
    select_fg_bg_tubes,
)

log = logging.getLogger(__name__)


class Frame_labeled(TypedDict):
    vid: Vid_daly
    frame_ind: int
    label: int


class Box_labeled(TypedDict):
    vid: Vid_daly
    frame_ind: int
    box: np.ndarray  # LTRD, absolute scale
    label: int
    dwti: Optional[I_dwein]
    kfi: Optional[int]


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
        dataset, stride, tubes_dwein,
        tube_metas: Dict[I_dwein, Dict]
        ) -> Dict[int, List[Dict]]:
    """
    Will only work with tubes in tube_metas
    """
    tubes_dwein_proper = {k: tubes_dwein[k] for k in tube_metas}

    # Keyframes by which we divide
    vids_kf_nums: Dict[Vid_daly, np.ndarray] = \
            sample_daly_frames_from_instances(dataset, stride=0)
    # Frames which we consider
    vids_good_nums: Dict[Vid_daly, np.ndarray] = \
            sample_daly_frames_from_instances(dataset, stride=stride)
    # keyframes per dwt
    dwtis_kf_nums = {}
    dwtis_good_nums = {}
    for dwt_index, dwt in tubes_dwein_proper.items():
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

    # Compute distance, disperse
    dist_boxes: Dict[int, List[Dict]] = {}
    for dwti, dwt in tubes_dwein_proper.items():
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

        # Possible FG/BG computations
        tube_meta = tube_metas[dwti]

        # Disperse boxes per dist
        for f0, box, dist in zip(good_nums, boxes, best_dist):
            metabox = {
                'frame_ind': f0, 'box': box, 'dwti': dwti}
            metabox.update(tube_meta)
            dist_boxes.setdefault(dist, []).append(metabox)
    dist_boxes = dict(sorted(list(dist_boxes.items())))
    return dist_boxes


def prepare_label_fullframes_for_training(
        tubes_dgt_train, dataset,
        stride: int, max_distance: int
        ) -> List[Frame_labeled]:
    # Frames which we consider for this experiment
    # (universal, present in every experiment)
    vids_good_nums: Dict[Vid_daly, np.ndarray] = \
            sample_daly_frames_from_instances(dataset, stride=stride)
    # Retrieve frames from GT tubes
    dist_frames: Dict[int, List[Dict]] = {}
    for dgti, tube_dgt in tubes_dgt_train.items():
        vid, action_name, _ = dgti
        label = dataset.action_names.index(action_name)
        good_nums = vids_good_nums[vid]
        kf_nums = tube_dgt['frame_inds']
        # Distance matrix
        M = np.zeros((len(kf_nums), len(good_nums)), dtype=np.int)
        for i, kf in enumerate(kf_nums):
            M[i] = np.abs(good_nums-kf)
        best_dist = M.min(axis=0)
        # Disperse boxes per dist
        for f0, dist in zip(good_nums, best_dist):
            metaframe = {
                'frame_ind': f0, 'dwti': dgti, 'label': label}
            dist_frames.setdefault(dist, []).append(metaframe)
    dist_frames = dict(sorted(list(dist_frames.items())))
    # Take only proper distance, rearrange into (vid,frame_ind)
    labeled_frames: List[Frame_labeled] = []
    for i, frames_ in dist_frames.items():
        if i > max_distance:
            break
        for frame_ in frames_:
            lframe: Frame_labeled = {
                    'vid': frame_['dwti'][0],
                    'frame_ind': frame_['frame_ind'],
                    'label': frame_['label']}
            labeled_frames.append(lframe)
    # Group into frame_groups
    frame_groups: Dict[Tuple[Vid_daly, int], List[Frame_labeled]] = {}
    for lframe in labeled_frames:
        vid = lframe['vid']
        frame_ind = lframe['frame_ind']
        frame_groups.setdefault((vid, frame_ind), []).append(lframe)
    # Exclude frames with >1 box from training
    singular_labeled_frames: List[Frame_labeled] = []
    for vf, lframes in frame_groups.items():
        if len(lframes) == 1:
            singular_labeled_frames.append(lframes[0])
    return singular_labeled_frames


def prepare_label_roiboxes_for_training(
        tubes_dgt_train, dataset,
        stride: int, max_distance: int,
        tubes_dwein_train, keyframes_train, top_n_matches: int,
        add_keyframes=True
        ) -> List[Box_labeled]:
    # ROI box extract
    matched_dwts: Dict[I_dgt, Dict[I_dwein, float]] = \
        get_dwein_overlaps_per_dgt(tubes_dgt_train, tubes_dwein_train)
    fg_meta, bg_meta = select_fg_bg_tubes(matched_dwts, top_n_matches)
    log.info('Selected {} FG and {} BG tubes from a total of {}'.format(
        len(fg_meta), len(bg_meta), len(tubes_dwein_train)))
    # Merge fg/bg
    tube_metas = {}
    tube_metas.update(fg_meta)
    tube_metas.update(bg_meta)
    # Break into frames, sort by distance
    dist_boxes_train: Dict[int, List[Dict]] = \
        group_dwein_frames_wrt_kf_distance(
            dataset, stride, tubes_dwein_train, tube_metas)
    # Special
    lbox: Box_labeled
    labeled_boxes: List[Box_labeled] = []
    for i, boxes in dist_boxes_train.items():
        if i > max_distance:
            break
        for box in boxes:
            (vid, bunch_id, tube_id) = box['dwti']
            if box['kind'] == 'fg':
                (vid, action_name, ins_id) = box['dgti']
                label = dataset.action_names.index(action_name)
            else:
                label = len(dataset.action_names)
            lbox = {
                'vid': vid,
                'frame_ind': box['frame_ind'],
                'box': box['box'],
                'label': label,
                'dwti': box['dwti'],
                'kfi': None
            }
            labeled_boxes.append(lbox)
    if add_keyframes:
        # Merge keyframes too
        for kfi, kf in enumerate(keyframes_train):
            action_name = kf['action_name']
            label = dataset.action_names.index(action_name)
            lbox = {
                'vid': kf['vid'],
                'frame_ind': kf['frame0'],
                'box': kf['bbox'],
                'label': label,
                'dwti': None,
                'kfi': kfi,
            }
            labeled_boxes.append(lbox)
    return labeled_boxes
