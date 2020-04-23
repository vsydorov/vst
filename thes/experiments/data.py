import re
import numpy as np
from typing import (List, Tuple, Dict, cast, TypedDict, Set)  # NOQA

from vsydorov_tools import small

from thes.tools import snippets
from thes.data.dataset.external import (Dataset_daly_ocv)
from thes.data.tubes.types import (
    DALY_wein_tube_index, DALY_wein_tube, DALY_gt_tube_index, DALY_gt_tube)
from thes.data.tubes.types import (
    AV_dict, Frametube, get_daly_gt_tubes)
from thes.data.tubes.routines import wein_coverage_stats


def precompute_cache(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    dataset: [~, ['daly', 'voc2007']]
    """)
    cf = cfg.parse()

    if cf['dataset'] == 'daly':
        dataset = Dataset_daly_ocv()
    elif cf['dataset'] == 'voc2007':
        raise NotImplementedError()
    else:
        raise RuntimeError('Wrong dataset')
    dataset.precompute_to_folder(out)


def load_wein_tubes(workfolder, cfg_dict, add_args):
    """
    Philippe tubes:
        tube:
             (one row per frame):
                index of the frame (starting at 1)
                x1 y1 x2 y2
                score of the generic human detector
                score of the instance-specific detector
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    wein_tubes: [~, str]
    dataset:
        cache_folder: [~, str]
    """)
    cf = cfg.parse()
    # Dataset
    dataset = Dataset_daly_ocv()
    dataset.populate_from_folder(cf['dataset.cache_folder'])
    # Tubes
    # 0 has 510 elements, 1 has 200 elements
    wein_package = small.load_py2_pkl(cf['wein_tubes'])[0]
    # We got a dictionary of filenames (w .mp4 suffix)
    extracted_tubes: Dict[DALY_wein_tube_index, DALY_wein_tube] = {}
    for vid_mp4, wein_bunches in wein_package.items():
        vid = re.search(r'(.*)\.mp4', vid_mp4).group(1)
        rs = dataset.rstats[vid]
        for bunch_id, wein_tubes in enumerate(wein_bunches):
            for tube_id, wein_tube in enumerate(wein_tubes):
                frame_inds = wein_tube[:, 0].astype(np.int) - 1
                assert max(frame_inds) < rs['max_pos_frames']
                boxes_ltrd = wein_tube[:, 1:5]  # ltrd
                human_scores = wein_tube[:, 5]
                instance_scores = wein_tube[:, 6]
                tube = {
                        'frame_inds': frame_inds,
                        'boxes': boxes_ltrd,
                        'hscores': human_scores,
                        'iscores': instance_scores}
                extracted_tubes[(vid, bunch_id, tube_id)] = tube
    small.save_pkl(out/'extracted_tubes.pkl', extracted_tubes)


def stats_weintubes(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    imported_wein_tubes: [~, str]
    dataset:
        name: [~, ['daly']]
        cache_folder: [~, str]
    """)
    cf = cfg.parse()
    # Dataset
    dataset = Dataset_daly_ocv()
    dataset.populate_from_folder(cf['dataset.cache_folder'])
    # for split_label in ['train', 'test']:
    #     split_vids = get_daly_split_vids(dataset, split_label)
    gt_tubes: Dict[DALY_gt_tube_index, DALY_gt_tube] = \
            get_daly_gt_tubes(dataset)
    # av_gt_tubes: AV_dict[Frametube] = convert_dgt_tubes(gt_tubes)

    dwein_tubes: Dict[DALY_wein_tube_index, DALY_wein_tube] = \
            small.load_pkl(cf['imported_wein_tubes'])
    wein_coverage_stats(gt_tubes, dwein_tubes)
