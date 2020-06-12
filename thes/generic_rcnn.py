import logging
import pprint
import numpy as np
import cv2
from typing import (
        List, Tuple, Dict, cast, TypedDict, Set,
        Sequence, Optional, Literal)
from tqdm import tqdm

from vsydorov_tools import small
from vsydorov_tools import cv as vt_cv

from thes.data.tubes.types import (
    I_dwein, T_dwein, T_dwein_scored, I_dgt, T_dgt,
    loadconvert_tubes_dwein, get_daly_gt_tubes,
    push_into_avdict, dtindex_filter_split,
    Objaction_dets, Frametube,
    av_filter_split, av_stubes_above_score,
    AV_dict,)
from thes.data.dataset.external import (
    Dataset_daly_ocv, Vid_daly)
from thes.tools import snippets


log = logging.getLogger(__name__)


def sample_dict(dct: Dict, N=10, NP_SEED=0) -> Dict:
    np_rstate = np.random.RandomState(NP_SEED)
    prm_key_indices = np_rstate.permutation(np.arange(len(dct)))
    key_list = list(dct.keys())
    some_keys = [key_list[i] for i in prm_key_indices[:N]]
    some_tubes = {k: dct[k] for k in some_keys}
    return some_tubes


class Box_connections_dwti(TypedDict):
    vid: Vid_daly
    frame_ind: int
    dwti_sources: List[I_dwein]  # N
    boxes: List[np.ndarray]  # N, 4


class Ncfg_generic_rcnn_eval:
    @staticmethod
    def set_defcfg(cfg):
        cfg.set_deftype("""
        scoring:
            keyframes_only: [True, bool]
        score_agg_kind: ['mean', ['mean', 'max', 'sum']]
        demo_run:
            enabled: [False, bool]
            N: [50, int]
            seed: [0, int]
        compute:
            save_period: ['::10', str]
            split:
                enabled: [False, bool]
                chunk: [0, "VALUE >= 0"]
                total: [1, int]
        """)

    @staticmethod
    def _prepare_ftube_box_computations(
            tubes_dwein: Dict[I_dwein, T_dwein],
            frames_to_cover: Optional[Dict[Vid_daly, np.ndarray]]
            ) -> Dict[Vid_daly, Dict[int, Box_connections_dwti]]:
        """
        Assign boxes (and keep connections to the original ftubes)
        If frames_to_cover passed - compute only in those frames
        """
        vf_connections_dwti_list: Dict[Vid_daly, Dict[int,
            List[Tuple[I_dwein, np.ndarray]]]] = {}
        for dwt_index, tube in tubes_dwein.items():
            (vid, bunch_id, tube_id) = dwt_index
            tube_finds = tube['frame_inds']
            if frames_to_cover is None:
                common_finds = tube_finds
                good_tube_boxes = tube['boxes']
            else:
                good_finds = frames_to_cover[vid]
                common_finds, comm1, comm2 = np.intersect1d(
                    tube_finds, good_finds,
                    assume_unique=True, return_indices=True)
                if len(common_finds) == 0:
                    continue
                good_tube_boxes = tube['boxes'][comm1]
            for find, box in zip(common_finds, good_tube_boxes):
                (vf_connections_dwti_list
                    .setdefault(vid, {})
                    .setdefault(find, []).append((dwt_index, box)))
        # Prettify
        vf_connections_dwti: Dict[Vid_daly, Dict[int, Box_connections_dwti]] = {}
        for vid, f_connections_dwti_list in vf_connections_dwti_list.items():
            for find, connections_dwti_list in f_connections_dwti_list.items():
                lsources, lboxes = zip(*connections_dwti_list)
                boxes = np.vstack(lboxes)
                bcs: Box_connections_dwti = {
                    'vid': vid,
                    'frame_ind': find,
                    'dwti_sources': lsources,
                    'boxes': boxes
                }
                vf_connections_dwti.setdefault(vid, {})[find] = bcs
        return vf_connections_dwti

    @classmethod
    def _get_daly_keyframes(
            cls, dataset: Dataset_daly_ocv, split_vids
            ) -> Dict[Vid_daly, np.ndarray]:
        to_cover_: Dict[Vid_daly, Set] = {}
        for vid in split_vids:
            v = dataset.videos_ocv[vid]
            for action_name, instances in v['instances'].items():
                for ins_ind, instance in enumerate(instances):
                    frames = [kf['frame'] for kf in instance['keyframes']]
                    to_cover_[vid] = \
                            to_cover_.get(vid, set()) | set(list(frames))
        frames_to_cover = \
                {k: np.array(sorted(v)) for k, v in to_cover_.items()}
        return frames_to_cover

    @classmethod
    def _define_boxes_to_evaluate(cls, cf, dataset, split_vids, tubes_dwein):
        if cf['scoring.keyframes_only']:
            # Cover only keyframes when evaluating dwti tubes
            frames_to_cover = cls._get_daly_keyframes(dataset, split_vids)
        else:
            frames_to_cover = None
        vf_connections_dwti: Dict[Vid_daly, Dict[int, Box_connections_dwti]]
        vf_connections_dwti = cls._prepare_ftube_box_computations(
                tubes_dwein, frames_to_cover)
        return vf_connections_dwti

    @classmethod
    def _demovis_apply(cls,
            vfold, neth, dataset: Dataset_daly_ocv, vf_connections_dwti):
        nicolas_labels = ['background', ] + cast(List[str], dataset.action_names)
        for vid, f_connections_dwti in tqdm(
                vf_connections_dwti.items(), 'nicphil_demovis'):
            video_path = dataset.videos[vid]['path']
            finds = list(f_connections_dwti)
            with vt_cv.video_capture_open(video_path) as vcap:
                frames_u8 = vt_cv.video_sample(
                        vcap, finds, debug_filename=video_path)

            video_fold = small.mkdir(vfold/f'vid{vid}')

            for find, frame_BGR in zip(finds, frames_u8):
                connections_dwti = f_connections_dwti[find]
                boxes = connections_dwti['boxes']
                box_cls_probs = neth.score_boxes(frame_BGR, boxes)  # N, (bcg+10)
                # Draw and print
                txt_output = []
                image = frame_BGR.copy()
                for i, cls_probs in enumerate(box_cls_probs):
                    box = boxes[i]
                    best_score_id = np.argmax(cls_probs)
                    best_score = cls_probs[best_score_id]
                    best_nicolas_label = nicolas_labels[best_score_id]
                    snippets.cv_put_box_with_text(image, box,
                        text='{} {} {:.2f}'.format(
                            i, best_nicolas_label, best_score))
                    line = (' '.join([f'{y}: {x:.3f}'
                        for x, y in zip(cls_probs, nicolas_labels)])
                        + str(box))
                    txt_output.append(line)
                cv2.imwrite(str(
                    video_fold/'Fr{:05d}.png'.format(find)), image)
                with (video_fold/f'Fr{find:05d}_scores.txt').open('w') as f:
                    f.write('\n'.join(txt_output))

    @classmethod
    def _perform_split(cls, cf, vf_connections_dwti):
        # Reduce keys according to split
        vids_to_eval = list(vf_connections_dwti.keys())
        weights_dict = {k: len(v) for k, v in vf_connections_dwti.items()}
        weights = np.array(list(weights_dict.values()))
        cc, ct = (cf['compute.split.chunk'], cf['compute.split.total'])
        vids_split = snippets.weighted_array_split(
                vids_to_eval, weights, ct)
        ktw = dict(zip(vids_to_eval, weights))
        weights_split = [np.sum([ktw[vid] for vid in vids])
                for vids in vids_split]
        chunk_vids = vids_split[cc]
        log.info(f'Quick split stats [{cc,ct=}]: ''Vids(frames): {}({}) -> {}({})'.format(
            len(vids_to_eval), np.sum(weights),
            len(chunk_vids), weights_split[cc]))
        log.debug(f'Full stats [{cc,ct=}]:\n'
                f'vids_split={pprint.pformat(vids_split)}\n'
                f'{weights_split=}\n'
                f'{chunk_vids=}\n'
                f'{weights_split[cc]=}')
        chunk_vf_connections_dwti = {vid: vf_connections_dwti[vid]
                for vid in chunk_vids}
        return chunk_vf_connections_dwti

    @classmethod
    def _simple_gpu_compute(
            cls, out, dataset, neth, vf_connections_dwti
            ) -> Dict[Vid_daly, Dict[int, np.ndarray]]:
        """Progress saved on video-level scale"""
        def isaver_eval_func(vid):
            f_connections_dwti = vf_connections_dwti[vid]
            video_path = dataset.videos[vid]['path']
            finds = list(f_connections_dwti)
            with vt_cv.video_capture_open(video_path) as vcap:
                frames_u8 = vt_cv.video_sample(
                        vcap, finds, debug_filename=video_path)
            f_cls_probs = {}
            for find, frame_BGR in zip(finds, frames_u8):
                connections_dwti = f_connections_dwti[find]
                boxes = connections_dwti['boxes']
                cls_probs = neth.score_boxes(frame_BGR, boxes)  # N, (bcg+10)
                f_cls_probs[find] = cls_probs
            return f_cls_probs
        vids_to_eval = list(vf_connections_dwti.keys())
        isaver = snippets.Isaver_simple(
                small.mkdir(out/'isave_rcnn_vid_eval'),
                vids_to_eval, isaver_eval_func,
                '::10', 120)
        isaver_items = isaver.run()
        vf_cls_probs: Dict[Vid_daly, Dict[int, np.ndarray]]
        vf_cls_probs = dict(zip(vids_to_eval, isaver_items))
        return vf_cls_probs

    @classmethod
    def _involved_gpu_compute(
            cls, out, dataset, neth,
            vf_connections_dwti,
            size_video_chunk,
            ) -> Dict[Vid_daly, Dict[int, np.ndarray]]:
        frame_chunks = []
        vids_to_eval = list(vf_connections_dwti.keys())
        for vid in vids_to_eval:
            f_connections_dwti = vf_connections_dwti[vid]
            finds = np.array(list(f_connections_dwti))
            finds_split = snippets.leqn_split(
                    finds, size_video_chunk)
            for subset_finds in finds_split:
                frame_chunks.append((vid, subset_finds))

        def isaver_eval_func(frame_chunk):
            vid, finds = frame_chunk
            f_connections_dwti = vf_connections_dwti[vid]
            video_path = dataset.videos[vid]['path']
            with vt_cv.video_capture_open(video_path) as vcap:
                frames_u8 = vt_cv.video_sample(
                        vcap, finds, debug_filename=video_path)
            f_cls_probs = {}
            for find, frame_BGR in zip(finds, frames_u8):
                connections_dwti = f_connections_dwti[find]
                boxes = connections_dwti['boxes']
                cls_probs = neth.score_boxes(
                        frame_BGR, boxes)  # N, (bcg+10)
                f_cls_probs[find] = cls_probs
            return f_cls_probs
        isaver = snippets.Isaver_simple(
                small.mkdir(out/'isave_rcnn_vid_eval'),
                frame_chunks, isaver_eval_func,
                save_interval=60,
                log_interval=300)
        isaver_items = isaver.run()
        vf_cls_probs: Dict[Vid_daly, Dict[int, np.ndarray]]
        vf_cls_probs = {}
        for (vid, subset_finds), f_cls_probs in zip(
                frame_chunks, isaver_items):
            vf_cls_probs.setdefault(vid, {}).update(f_cls_probs)
        return vf_cls_probs

    @classmethod
    def evaluate_rcnn_boxes(
            cls, cf, out,
            dataset: Dataset_daly_ocv,
            split_vids,
            tubes_dwein: Dict[I_dwein, T_dwein],
            neth):
        """
        Logic behind simple "evaluate boxes" experiment
        """
        vf_connections_dwti: Dict[Vid_daly, Dict[int, Box_connections_dwti]] = \
            cls._define_boxes_to_evaluate(cf, dataset, split_vids, tubes_dwein)

        if cf['compute.split.enabled']:
            vf_connections_dwti = cls._perform_split(
                    cf, vf_connections_dwti)

        if cf['scoring.keyframes_only']:
            vf_cls_probs = cls._simple_gpu_compute(
                out, dataset, neth, vf_connections_dwti)
        else:
            size_video_chunk = 300
            vf_cls_probs = cls._involved_gpu_compute(
                out, dataset, neth, vf_connections_dwti, size_video_chunk)
        return vf_connections_dwti, vf_cls_probs

    @classmethod
    def demo_run(cls, cf, out, dataset, split_vids, tubes_dwein, neth):
        vf_connections_dwti = cls._define_boxes_to_evaluate(
                cf, dataset, split_vids, tubes_dwein)
        vf_connections_dwti = sample_dict(
            vf_connections_dwti, N=5, NP_SEED=0)
        vfold = small.mkdir(out/'demovis')
        cls._demovis_apply(vfold, neth, dataset, vf_connections_dwti)

    @staticmethod
    def aggregate_rcnn_scores(
            dataset, tubes_dwein,
            vf_connections_dwti,
            vf_cls_probs,
            agg_kind: Literal['mean', 'max', 'sum']
            ) -> AV_dict[T_dwein_scored]:
        assert vf_connections_dwti.keys() == vf_cls_probs.keys()

        if agg_kind in ['mean', 'sum']:
            ftube_sum: Dict[I_dwein, np.ndarray] = {}
            ftube_counts: Dict[I_dwein, int] = {}
            for vid, f_cls_probs in vf_cls_probs.items():
                for f, cls_probs in f_cls_probs.items():
                    dwtis = vf_connections_dwti[vid][f]['dwti_sources']
                    for dwti, prob in zip(dwtis, cls_probs):
                        ftube_sum[dwti] = \
                                ftube_sum.get(dwti, np.zeros(11)) + prob
                        ftube_counts[dwti] = \
                                ftube_counts.get(dwti, 0) + 1
            if agg_kind == 'mean':
                ftube_scores = {k: v/ftube_counts[k]
                        for k, v in ftube_sum.items()}
            else:
                ftube_scores = ftube_sum
        elif agg_kind == 'max':
            ftube_scores = {}
            for vid, f_cls_probs in vf_cls_probs.items():
                for f, cls_probs in f_cls_probs.items():
                    dwtis = vf_connections_dwti[vid][f]['dwti_sources']
                    for dwti, prob in zip(dwtis, cls_probs):
                        ftube_scores[dwti] = \
                            np.maximum(ftube_scores.get(
                                dwti, np.zeros(11)), prob)
        else:
            raise NotImplementedError()

        # Create av_stubes
        av_stubes: AV_dict[T_dwein_scored] = {}
        for dwt_index, scores in ftube_scores.items():
            (vid, bunch_id, tube_id) = dwt_index
            for action_name, score in zip(
                    dataset.action_names, scores[1:]):
                stube = tubes_dwein[dwt_index].copy()
                stube = cast(T_dwein_scored, stube)
                stube['score'] = score
                (av_stubes
                        .setdefault(action_name, {})
                        .setdefault(vid, []).append(stube))
        return av_stubes
