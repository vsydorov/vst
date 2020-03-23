"""
Classes here manage interfacing with SpecificAccess in a task specific way
"""
import numpy as np
import logging
from pathlib import Path
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Any, Optional, Tuple
from mypy_extensions import TypedDict

import torch

from tavid.tools import snippets
from tavid.data.specific_access import (SpecificAccess, VideoData)
from tavid.data.video_utils import (tfm_video_resize_threaded,
        tfm_video_random_crop, tfm_video_center_crop, tfm_maybe_flip,)

log = logging.getLogger(__name__)


class DataAccess(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_item(self, vid: str):
        raise NotImplementedError()


Train_Transform_Params = TypedDict('Train_Transform_Params', {
    'resize': Any,
    'rcrop': Any,
    'flip': Any,
    })


Eval_Transform_Params = TypedDict('Eval_Transform_Params', {
    'resize': Any,
    'ccrop': Any,
    })


@dataclass
class Train_Meta:
    vid: str
    video_path: Path
    real_sampled_inds: np.ndarray
    do_not_collate: bool = True
    centerbox: Optional[np.ndarray] = None
    params: Optional[Train_Transform_Params] = None


@dataclass
class Eval_Meta:
    vid: str
    video_path: Path
    shifts: np.ndarray
    unique_real_sampled_inds: np.ndarray
    params: Optional[Eval_Transform_Params] = None
    centerbox: Optional[np.ndarray] = None
    rel_frame_inds: Optional[np.ndarray] = None
    tw: Optional[snippets.TicToc] = None


Train_Item = Tuple[torch.Tensor, torch.Tensor, Train_Meta]


Eval_Item = TypedDict('Eval_Item', {
        'X': torch.Tensor,
        'stacked_targets': torch.Tensor,
        'meta': Eval_Meta})


class DataAccess_Train(DataAccess):
    sa: SpecificAccess
    initial_resize: int
    input_size: int
    train_gap: int
    fps: int
    params_to_meta: bool
    new_target: str

    def __init__(
            self, sa: SpecificAccess,
            initial_resize, input_size,
            train_gap, fps, params_to_meta, new_target):
        super().__init__()
        self.sa = sa
        self.initial_resize = initial_resize
        self.input_size = input_size
        self.train_gap = train_gap
        self.fps = fps
        self.params_to_meta = params_to_meta
        self.new_target = new_target

    @staticmethod
    def _train_prepare(X, initial_resize, input_size
                       ) -> Tuple[np.ndarray, Train_Transform_Params]:
        X, resize_params = tfm_video_resize_threaded(X, initial_resize)
        X, rcrop_params = tfm_video_random_crop(
                X, input_size, input_size)
        X, flip_params = tfm_maybe_flip(X)
        assert X.dtype is np.dtype('uint8'), 'must be uin8'
        params: Train_Transform_Params = {
                'resize': resize_params,
                'rcrop': rcrop_params,
                'flip': flip_params}
        return X, params

    def _adjust_train_target(self, X, target):
        # Target adjustment
        assert len(X) == len(target)
        if self.new_target == 'single':
            pass
        elif self.new_target == 'append':
            target = target.repeat(2, 1)
        else:
            raise NotImplementedError()
        return target

    def _get_item_prepare_train(self, vid: str, shift: Optional[float]):
        if shift is None:
            shift = np.random.rand()

        videodata: VideoData = self.sa.get_videodata(vid)

        real_sampled_inds, sampled_times = \
                self.sa.sample_frameids_and_times(
                        videodata, shift, self.train_gap, self.fps)
        frames_u8 = self.sa.sample_frames(videodata, real_sampled_inds)
        frames_u8 = np.flip(frames_u8, -1)  # Make RGB
        target = self.sa.sample_targets(videodata, sampled_times)

        meta = Train_Meta(
            vid=vid, video_path=videodata.video_path,
            real_sampled_inds=real_sampled_inds)
        return target, meta, frames_u8

    def get_item(self, vid: str) -> Train_Item:
        if not self.sa.labels_present:
            raise RuntimeError('SA does not allow sampling targets. '
                    'Training not supported')
        meta: Train_Meta
        target, meta, frames_rgb_u8 = \
                self._get_item_prepare_train(vid, None)
        X, params = self._train_prepare(frames_rgb_u8,
                self.initial_resize, self.input_size)
        if self.params_to_meta:
            meta.params = params
        X_ = torch.from_numpy(X)
        target = self._adjust_train_target(X, target)
        return X_, target, meta


class DataAccess_Eval(DataAccess):
    sa: SpecificAccess
    initial_resize: int
    input_size: int
    train_gap: int
    fps: int
    params_to_meta: bool
    new_target: str
    eval_gap: int

    def __init__(self, sa: SpecificAccess,
            initial_resize, input_size,
            train_gap, fps,
            params_to_meta, new_target, eval_gap):
        super().__init__()
        self.sa = sa
        self.initial_resize = initial_resize
        self.input_size = input_size
        self.train_gap = train_gap
        self.fps = fps
        self.params_to_meta = params_to_meta
        self.new_target = new_target
        self.eval_gap = eval_gap

    @staticmethod
    def _eval_prepare(X, initial_resize, input_size
                      ) -> Tuple[np.ndarray, Eval_Transform_Params]:
        X, resize_params = tfm_video_resize_threaded(X, initial_resize)
        X, ccrop_params = tfm_video_center_crop(
                X, input_size, input_size)
        assert X.dtype is np.dtype('uint8'), 'must be uin8'
        params: Eval_Transform_Params = {
                'resize': resize_params,
                'ccrop': ccrop_params}
        return X, params

    def _adjust_eval_target(self, input_, stacked_targets):
        assert input_.shape[1] == stacked_targets.shape[1]
        if self.new_target == 'single':
            pass
        elif self.new_target == 'append':
            stacked_targets = stacked_targets.repeat(1, 2, 1)
        else:
            raise NotImplementedError()
        return stacked_targets

    def _get_item_prepare_eval(self, vid: str):
        shifts = np.linspace(0, 1.0, self.eval_gap)

        videodata: VideoData = self.sa.get_videodata(vid)

        # Sample frames_inds and times
        sampled2 = [self.sa.sample_frameids_and_times(
                    videodata, shift, self.train_gap, self.fps)
                    for shift in shifts]
        all_real_sampled_inds, all_sampled_times = zip(*sampled2)

        unique_real_sampled_inds = \
                np.unique(np.hstack(all_real_sampled_inds))
        all_relative_sampled_inds = []
        for x in all_real_sampled_inds:
            y = np.searchsorted(unique_real_sampled_inds, x)
            all_relative_sampled_inds.append(y)
        stacked_relative_sampled_inds = \
                np.vstack(all_relative_sampled_inds)

        unique_frames = self.sa.sample_frames(
                videodata, unique_real_sampled_inds)
        unique_frames_rgb_u8 = np.flip(unique_frames, -1)  # Make RGB

        if self.sa.labels_present:
            all_targets = [self.sa.sample_targets(videodata, sampled_times)
                for sampled_times in all_sampled_times]
            stacked_targets = torch.stack(all_targets)  # N_batch, T, N_class
        else:
            stacked_targets = None

        # Resize, centercrop
        unique_frames_prepared_u8, params = self._eval_prepare(
                unique_frames_rgb_u8, self.initial_resize,
                self.input_size)

        meta = Eval_Meta(
            vid=vid, video_path=videodata.video_path,
            shifts=shifts,
            unique_real_sampled_inds=unique_real_sampled_inds)

        return (unique_frames_rgb_u8, unique_frames_prepared_u8,
                stacked_relative_sampled_inds,
                stacked_targets, params, meta)

    def get_item(self, vid: str) -> Eval_Item:
        tw = snippets.TicToc([
            'get_unique_frames', 'prepare_inputs'])
        tw.tic('get_unique_frames')
        meta: Eval_Meta
        params: Eval_Transform_Params
        (unique_frames_rgb_u8, unique_frames_prepared_u8,
                stacked_relative_sampled_inds,
                stacked_targets, params, meta) = \
                self._get_item_prepare_eval(vid)
        tw.toc('get_unique_frames')
        tw.tic('prepare_inputs')
        normal = unique_frames_prepared_u8[
                stacked_relative_sampled_inds]
        input_ = torch.from_numpy(normal)
        tw.toc('prepare_inputs')
        meta.tw = tw
        if self.params_to_meta:
            meta.params = params
            meta.rel_frame_inds = stacked_relative_sampled_inds
        if stacked_targets is not None:
            stacked_targets = self._adjust_eval_target(
                    input_, stacked_targets)
        eval_item = Eval_Item(
                X=input_,
                stacked_targets=stacked_targets,
                meta=meta)
        return eval_item
