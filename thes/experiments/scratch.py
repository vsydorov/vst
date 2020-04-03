import logging
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from thes.data.dataset.external import (Dataset_daly_ocv)
from thes.tools import snippets


log = logging.getLogger(__name__)


def _examine_stats(self, rstats):
    vids = list(self.videos.keys())
    d_rstats = pd.DataFrame(rstats).T.loc[vids]
    d_pmeta = pd.DataFrame(self.provided_metas).T.loc[vids]

    # frame count mismatch
    bad_nframes = d_rstats.loc[
            (d_rstats['frame_count'] != d_pmeta['nbframes_ffmpeg'])]
    d_pmeta.loc[bad_nframes.index]
    d_rstats.loc[bad_nframes.index]
    d_rstats['meta_duration'] = d_pmeta['duration']
    d_rstats['meta_fps'] = d_pmeta['fps']
    d_rstats['meta_frame_count'] = d_pmeta['nbframes_ffmpeg']
    d_rstats['max_time'] = d_rstats['max_pos_msec']/1000
    d_rstats['max_frame'] = d_rstats['max_pos_frames'] - 1
    d_rstats['est_fps'] = d_rstats['max_frame']/d_rstats['max_time']
    d_rstats['est_length'] = d_rstats['max_time'] + 1/d_rstats['est_fps']
    # Count mismatches
    d_rstats['diff_fps'] = (d_rstats['est_fps']-d_rstats['meta_fps']).abs()
    d_rstats['diff_nframes'] = (
            d_rstats['max_pos_frames']-d_rstats['meta_frame_count']).abs()
    d_rstats['diff_length'] = (
            d_rstats['est_length']-d_rstats['meta_duration']).abs()
    diff_columns = ['diff_fps', 'diff_nframes', 'diff_length']
    diff_describe = d_rstats[diff_columns].describe()
    d_bad = pd.concat([d_rstats[d] > 1e-8 for d in diff_columns], axis=1)
    d_bad.sum()
    # Let's aggregate the keyframes
    kflist = []
    for vid, video in self.videos.items():
        for action_name, ains in video['instances'].items():
            for ins_ind, instance in enumerate(ains):
                for kf_ind, kf in enumerate(instance['keyframes']):
                    kfitem = {}
                    kfitem.update(kf)
                    kfitem.update({
                        'vid': vid,
                        'action_name': action_name,
                        'ins_ind': ins_ind,
                        'kf_ind': kf_ind})
                    kflist.append(kfitem)
    d_kf = pd.DataFrame(kflist)
    d_kf['meta_fps'] = d_rstats.loc[d_kf['vid']].reset_index()['meta_fps']
    d_kf['est_fps'] = d_rstats.loc[d_kf['vid']].reset_index()['est_fps']
    d_kf['x'] = (d_kf['time']*d_kf['meta_fps']).apply(np.ceil)
    d_kf[d_kf['x'] != d_kf['frameNumber']]


def compare_data(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_defaults_handling(raise_without_defaults=False)
    cfg.set_defaults("""
    cache_folders: ~
    """)
    cf = cfg.parse()

    dataset = Dataset_daly_ocv()
    dataset.precompute_to_folder(out)
    # # Dataset per each cache folder
    # ddict = {}
    # for k, v in cfg_dict['cache_folders'].items():
    #     dataset = DatasetDALY()
    #     dataset.populate_from_folder(v)
    #     ddict[k] = dataset
    #
    # odict_pds = {}
    # sv_pds = {}
    # for k, v in ddict.items():
    #     odict_pds[k] = pd.DataFrame(v.video_odict).T.sort_index()
    #     sv_pds[k] = pd.DataFrame(v.source_videos).T.sort_index()
    #
    # sv_pds['d18'] == sv_pds['m317']
    # (sv_pds['d18']['frames_reached'] == sv_pds['m317']['frames_reached']).all()

    # vid10 = list(ddict['d18'].video_odict.keys())[:10]
    # newstats = {}
    # for vid in tqdm(vid10):
    #     vmp4_d18 = ddict['d18'].source_videos[vid]
    #     vcap = cv2.VideoCapture(str(vmp4_d18['video_path']))
    #     vcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    #     while True:
    #         frames_reached = int(vcap.get(cv2.CAP_PROP_POS_FRAMES))
    #         ms_reached = int(vcap.get(cv2.CAP_PROP_POS_MSEC))
    #         ret = vcap.grab()
    #         if ret is False:
    #             break
    #     vcap.close()
    #     newstats[vid] = (frames_reached, ms_reached/1000)
    # new4 = pd.DataFrame(newstats).T
    # new4.columns = ['frames', 'length']
    # new4['fps'] = new4['frames']/new4['length']
    #
    # old4 = sv_pds['d18'].loc[vid10][['frames_reached', 'length_reached']]
    # old4.columns = ['frames', 'length']
    # old4['fps'] = old4['frames']/old4['length']
    #
    # odict = odict_pds['d18'].loc[vid10][['nbframes_ffmpeg', 'duration', 'fps']]
