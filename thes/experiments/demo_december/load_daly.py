import logging
from tqdm import tqdm
from abc import abstractmethod, ABC
import re
import collections
import pprint
import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from mypy_extensions import TypedDict

import torch
import torch.utils.data
import torch.backends.cudnn as cudnn

from thes.tools import snippets
from thes.data.external_dataset import (
        DatasetDALY, DALY_vid)
from thes.daly_d2 import (get_daly_gt_tubes, ex_tubes_to_df, gt_tubes_to_df)

from vsydorov_tools import small

log = logging.getLogger(__name__)


"""
Philippe tubes:
    tube:
         (one row per frame):
            index of the frame (starting at 1)
            x1 y1 x2 y2
            score of the generic human detector
            score of the instance-specific detector
"""

DALY_wein_tube = TypedDict('DALY_wein_tube', {
    'frame_inds': np.ndarray,
    'boxes': np.ndarray,  # LTRD
    'hscores': np.ndarray,  # human
    'iscores': np.ndarray  # instance
    })


DALY_tube_index = Tuple[DALY_vid, int, int]


def temporal_coverage_stats(ex_df, gt_df):
    # Let's compute temporal coverage stats
    coverage_dict = {}
    for key, line in tqdm(gt_df.iterrows(), total=len(gt_df),
            desc='total_coverage_stats'):
        vid_tubes = ex_df.query('vid=="{}"'.format(line['vid']))
        s, e = line['start_frame'], line['end_frame']
        frange = vid_tubes[['min_frame', 'max_frame']].to_numpy()
        # Interesting tubes: one of the limits lies inside
        limits_inside = (s <= frange) & (frange <= e)
        either_limit_inside = limits_inside.any(1)
        interesting_frange = frange[either_limit_inside]
        if len(interesting_frange):
            # Clip within range, compute "intersect" part
            clipped_frange = np.clip(interesting_frange, s, e)
            total_gt = e - s
            total_intersect = clipped_frange[:, 1] - clipped_frange[:, 0]
            # Compute union
            union_frange = np.empty_like(interesting_frange)
            union_frange[:, 0] = np.minimum(interesting_frange[:, 0], s)
            union_frange[:, 1] = np.maximum(interesting_frange[:, 1], e)
            total_union = union_frange[:, 1] - union_frange[:, 0]
            # Compute fraction
            fraction_intersect = total_intersect/total_gt
            fraction_iou = total_intersect/total_union
            max_intersect = np.max(fraction_intersect)
            max_iou = np.max(fraction_iou)
        else:
            max_intersect = 0.0
            max_iou = 0.0
        coverage_dict[key] = [max_intersect, max_iou]

    coverage_df = pd.DataFrame(coverage_dict).T
    coverage_df.columns = ['minter', 'miou']

    # Compute stats
    cdf = gt_df.copy()
    cdf[coverage_df.columns] = coverage_df[coverage_df.columns]
    stats = {}
    stats['mean_iou'] = cdf.miou.mean() * 100
    stats['mean_inter'] = cdf.minter.mean() * 100
    N = len(cdf)
    N_tubes_above_iou05 = (cdf.miou >= 0.5).sum()
    stats['tubes_above_iou_0.5'] = '{}/{} = {}'.format(
            N_tubes_above_iou05, N, N_tubes_above_iou05/N * 100)
    N_tubes_above_minter05 = (cdf.minter >= 0.5).sum()
    stats['tubes_above_inter_0.5'] = '{}/{} = {}'.format(
            N_tubes_above_minter05, N, N_tubes_above_minter05/N * 100)
    return coverage_df, stats


def spatial_coverage_stats(ex_df, gt_df, dataset, extracted_tubes):
    # Let's compute spatial coverage stats
    coverage_dict = {}
    for key, line in tqdm(gt_df.iterrows(), total=len(gt_df),
            desc='total_coverage_stats'):
        vid_tubes = ex_df.query('vid=="{}"'.format(line['vid']))
        s, e = line['start_frame'], line['end_frame']
        frange = vid_tubes[['min_frame', 'max_frame']].to_numpy()
        # Interesting tubes: one of the limits lies inside
        limits_inside = (s <= frange) & (frange <= e)
        either_limit_inside = limits_inside.any(1)
        interesting_frange = frange[either_limit_inside]
        tubes_inside = vid_tubes.iloc[either_limit_inside]
        if len(interesting_frange):
            # // Compute keyframe intersections
            # Retrieve GT keyframes
            gt_instance = (dataset.video_odict[line.vid]
                    ['instances'][line.action][line.ins_id])
            vmp4 = dataset.source_videos[line.vid]
            gt_frames = []
            gt_boxes_unscaled = []
            for kf in gt_instance['keyframes']:
                gt_frames.append(kf['frameNumber'])
                gt_boxes_unscaled.append(kf['boundingBox'])
            gt_frames = np.array(gt_frames)
            gt_boxes_unscaled = np.vstack(gt_boxes_unscaled)
            gt_boxes = gt_boxes_unscaled * np.tile(
                    [vmp4['width'], vmp4['height']], 2)
            # Retrieve those keyframes from proposals that match gt_frames
            retrieved = []
            for i, tube_row in tubes_inside.iterrows():
                ext_tube = extracted_tubes[
                        tube_row.vid, tube_row.bunch_id, tube_row.tube_id]
                found = np.isin(gt_frames, ext_tube['frame_inds'])
                found_ind = np.searchsorted(ext_tube['frame_inds'], gt_frames)
                found_ind[~found] = 0
                found_boxes = ext_tube['boxes'][found_ind]
                retrieved.append({'boxes': found_boxes, 'found': found})
            # Compute pairwise box IOUs
            pairwise_box_ious = []
            for i, x in enumerate(retrieved):
                ious = []
                for gt_box, p_box, found in zip(
                        gt_boxes, x['boxes'], x['found']):
                    if not found:
                        iou = 0.0
                    else:
                        # Computing IOU
                        inter = np.r_[
                            np.maximum(gt_box[:2], p_box[:2]),
                            np.minimum(gt_box[2:], p_box[2:])]
                        if np.any(inter[:2] > inter[2:]):
                            iou = 0.0
                        else:
                            inter_area = np.prod(inter[2:] - inter[:2])
                            union_area = (
                                np.prod(gt_box[2:] - gt_box[:2]) +
                                np.prod(p_box[2:] - p_box[:2]) - inter_area)
                            iou = inter_area/union_area
                    ious.append(iou)
                pairwise_box_ious.append(ious)
            pairwise_box_ious = np.array(pairwise_box_ious)
            # Mean per GT frame
            mean_box_ious = np.mean(pairwise_box_ious, axis=1)
            # Maximum iou
            max_iou = np.max(mean_box_ious)
        else:
            max_iou = 0.0
        coverage_dict[key] = max_iou

    coverage_df = pd.Series(coverage_dict).to_frame()
    coverage_df.columns = ['max_iou']

    cdf = gt_df.copy()
    cdf[coverage_df.columns] = coverage_df[coverage_df.columns]
    stats = {}
    stats['mean_iou'] = cdf.max_iou.mean()*100
    N = len(cdf)
    N_tubes_above_iou05 = (cdf.max_iou > 0.5).sum()
    stats['N_tubes_above_iou05'] = '{}/{} = {}'.format(
            N_tubes_above_iou05, N, N_tubes_above_iou05/N * 100)
    N_tubes_above_iou03 = (cdf.max_iou > 0.3).sum()
    stats['N_tubes_above_iou03'] = '{}/{} = {}'.format(
            N_tubes_above_iou03, N, N_tubes_above_iou03/N * 100)

    # Stats per action
    acdf = cdf[['action', 'max_iou']].copy()
    acdf['iou_above_05'] = acdf['max_iou'] > 0.5
    acdf['iou_above_03'] = acdf['max_iou'] > 0.3
    sum_per_action = acdf.groupby('action').sum()
    count_per_action = acdf.groupby('action').count()
    stats_df_peraction = sum_per_action/count_per_action*100
    return coverage_df, stats, stats_df_peraction


def stats_of_wein_tubes(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    imported_wein_tubes: [~, str]
    dataset:
        cache_folder: [~, str]
    """)
    cf = cfg.parse()
    # Dataset
    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])

    extracted_tubes = small.load_pkl(cf['imported_wein_tubes'])
    gt_tubes = get_daly_gt_tubes(dataset)

    # We reference video only in terms of ocv frames from now on
    # We ASSUME extracted philippe frames here are the OCV frames
    # Probably this is not the case
    ex_df = ex_tubes_to_df(extracted_tubes)
    gt_df = gt_tubes_to_df(dataset, gt_tubes)

    # coverage_df, temp_stats = temporal_coverage_stats(ex_df, gt_df)
    coverage_df, spat_stats = spatial_coverage_stats(
            ex_df, gt_df, dataset, extracted_tubes)
    pass


def load_wein_tubes(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    wein_tubes: [~, str]
    dataset:
        cache_folder: [~, str]
    """)
    cf = cfg.parse()
    # Dataset
    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])
    # Tubes
    # For reason I forgot we only care about [0] element
    wein_package = small.load_py2_pkl(cf['wein_tubes'])[0]
    # We got a dictionary of filenames (w .mp4 suffix)
    extracted_tubes: Dict[DALY_tube_index, DALY_wein_tube] = {}
    for vid_mp4, wein_bunches in wein_package.items():
        vid = re.search(r'(.*)\.mp4', vid_mp4).group(1)
        for bunch_id, wein_tubes in enumerate(wein_bunches):
            for tube_id, wein_tube in enumerate(wein_tubes):
                frame_inds = wein_tube[:, 0].astype(np.int)
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
