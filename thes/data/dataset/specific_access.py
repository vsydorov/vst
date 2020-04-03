"""
Interface over classes defined in external.py
"""
import numpy as np
import logging
from pathlib import Path
from abc import abstractmethod, ABC
from dataclasses import dataclass
#
# import torch
#
# from vsydorov_tools import cv as vt_cv
#
# from thes.data.dataset.external import (DatasetCharades, Charades_vid, Vid,
#         Charades_video, VideoMP4, DatasetHMDB51, HMDB51_vid, HMDB51_video,
#         VideoMP4_reached)
#
log = logging.getLogger(__name__)


@dataclass
class VideoData:
    vid: str
    video_path: Path


class SpecificAccess(ABC):
    labels_present: bool

    def __init__(self, ):
        super().__init__()

    @abstractmethod
    def get_videodata(self, vid):
        raise NotImplementedError()

    @abstractmethod
    def sample_frameids_and_times(
            self, videodata,
            shift: float, train_gap: int, fps: int):
        raise NotImplementedError()

    @abstractmethod
    def sample_frames(self, videodata, real_sampled_inds) -> np.array:
        raise NotImplementedError()

    @abstractmethod
    def sample_targets(self, videodata, sampled_times):
        raise NotImplementedError()
#
#
# @dataclass
# class VideoData_Charades(VideoData):
#     vid: Charades_vid
#     video: Charades_video
#     video_mp4: VideoMP4
#
#
# def _sample_frame_indices(
#         train_gap: int, total_frames: int, shift: float):
#     """
#     Sample a window of "train_gap" frames from "total_frames", starting
#     from "shift" relative position
#     If "train_gap" is bigger than "total_frames" - do edge padding
#     Returns:
#         sampled_inds - indices of sampled frames
#     """
#     n_padded = max(0, train_gap - total_frames)
#     start_segment = total_frames - train_gap + n_padded
#     shift_segment = int(shift * start_segment)
#     sampled_inds = shift_segment + np.arange(train_gap-n_padded)
#     sampled_inds = np.pad(sampled_inds, [0, n_padded], mode='edge')
#     return sampled_inds
#
#
# class SA_Charades(SpecificAccess):
#     data: DatasetCharades
#
#     def __init__(self, data: DatasetCharades, labels_present=True):
#         super().__init__()
#         self.data = data
#         self.labels_present = labels_present
#
#     def get_videodata(self, vid) -> VideoData_Charades:
#         vid = Charades_vid(Vid(vid))
#         video = self.data.video_odict[vid]
#         video_mp4 = self.data.get_source_video_for_vid(vid)
#         video_path = video_mp4['video_path']
#         videodata = VideoData_Charades(
#                 vid=vid, video_path=video_path,
#                 video=video, video_mp4=video_mp4)
#         return videodata
#
#     def sample_frameids_and_times(
#             self, videodata: VideoData_Charades,
#             shift: float, train_gap: int, fps: int):
#         nframes = videodata.video_mp4['nframes']
#         length_in_seconds = videodata.video['length']
#
#         # Video frames to query
#         real_fps = nframes/length_in_seconds
#         if fps is None:
#             # Use video fps, everything is straightforwards
#             good_inds = np.arange(nframes)
#         else:
#             n_fake_frames = int(length_in_seconds*fps)
#             fake_inds = np.interp(
#                     np.linspace(0, 1, n_fake_frames),
#                     np.linspace(0, 1, nframes),
#                     np.arange(nframes))
#             good_inds = fake_inds.round().astype(int)
#
#         total_frames = len(good_inds)
#         sampled_inds = _sample_frame_indices(
#                 train_gap, total_frames, shift)
#         real_sampled_inds = good_inds[sampled_inds]
#         sampled_times = real_sampled_inds/real_fps
#         return real_sampled_inds, sampled_times
#
#     def sample_frames(
#             self, videodata: VideoData_Charades,
#             real_sampled_inds: np.ndarray) -> np.array:
#         video_path = videodata.video_path
#         with vt_cv.video_capture_open(video_path, np.inf) as vcap:
#             frames_u8 = np.array(vt_cv.video_sample(vcap, real_sampled_inds))
#         return frames_u8
#
#     def sample_targets(
#             self, videodata: VideoData_Charades,
#             sampled_times: np.ndarray):
#         action_names = self.data.action_names
#         num_classes = len(action_names)
#         video_actions = videodata.video['actions']
#         tars = []
#         for sampled_time in sampled_times:
#             target = torch.IntTensor(num_classes).zero_()  # type: ignore
#             for action_name, start, end in video_actions:
#                 if start < sampled_time < end:
#                     action_id = action_names.index(action_name)
#                     target[action_id] = 1
#             tars.append(target)
#         target = torch.stack(tars)
#         return target
#
#
# @dataclass
# class VideoData_HMDB51(VideoData):
#     vid: HMDB51_vid
#     video: HMDB51_video
#     video_mp4: VideoMP4_reached
#
#
# class SA_HMDB51(SpecificAccess):
#     data: DatasetHMDB51
#
#     def __init__(self, data: DatasetHMDB51):
#         super().__init__()
#         self.data = data
#         self.labels_present = True
#
#     def get_videodata(self, vid) -> VideoData_HMDB51:
#         vid = HMDB51_vid(Vid(vid))
#         video = self.data.video_odict[vid]
#         video_mp4 = self.data.get_source_video_for_vid(vid)
#         video_path = video_mp4['video_path']
#         videodata = VideoData_HMDB51(
#                 vid=vid, video_path=video_path,
#                 video=video, video_mp4=video_mp4)
#         return videodata
#
#     def sample_frameids_and_times(
#             self, videodata: VideoData_HMDB51,
#             shift: float, train_gap: int, fps: int):
#         nframes = videodata.video_mp4['frames_reached']
#         length_in_seconds = videodata.video_mp4['length_reached']
#
#         # Video frames to query
#         real_fps = nframes/length_in_seconds
#         if fps is None:
#             # Use video fps, everything is straightforwards
#             good_inds = np.arange(nframes)
#         else:
#             n_fake_frames = int(length_in_seconds*fps)
#             fake_inds = np.interp(
#                     np.linspace(0, 1, n_fake_frames),
#                     np.linspace(0, 1, nframes),
#                     np.arange(nframes))
#             good_inds = fake_inds.round().astype(int)
#
#         total_frames = len(good_inds)
#         sampled_inds = _sample_frame_indices(
#                 train_gap, total_frames, shift)
#         real_sampled_inds = good_inds[sampled_inds]
#         sampled_times = real_sampled_inds/real_fps
#         return real_sampled_inds, sampled_times
#
#     def sample_frames(
#             self, videodata: VideoData_HMDB51,
#             real_sampled_inds: np.ndarray) -> np.array:
#         video_path = videodata.video_path
#         with vt_cv.video_capture_open(video_path, np.inf) as vcap:
#             frames_u8 = np.array(vt_cv.video_sample(vcap, real_sampled_inds))
#         return frames_u8
#
#     def sample_targets(
#             self, videodata: VideoData_HMDB51,
#             sampled_times: np.ndarray):
#
#         action_names = self.data.action_names
#         action_name = videodata.video['action_name']
#         num_classes = len(action_names)
#
#         target = torch.IntTensor(num_classes).zero_()  # type: ignore
#         action_id = action_names.index(action_name)
#         target[action_id] = 1
#         # Blow up to sample_times dimension
#         target = target[None, :].expand(len(sampled_times), 51)
#         return target
