from tqdm import tqdm
import numpy as np

import torch
import torch.utils.data

import vst

from thes.data.dataset.external import (
    Dataset_daly_ocv, Vid_daly)
from thes.data.dataset.daly import (
    Ncfg_daly)
from thes.feat_extract import (
    Ncfg_extractor)
from thes.pytorch import (
    sequence_batch_collate_v2, Sampler_grid, Frameloader_video_slowfast,
    TDataset_over_keyframes, TDataset_over_connections,
    Dataloader_isaver, to_gpu_normalize_permute, )


class TDataset_over_frames(torch.utils.data.Dataset):
    def __init__(self, video_path, nframes, sampled_frames,
            sampler_grid, frameloader_vsf):

        self.video_path = video_path
        self.nframes = nframes
        self.sampled_frames = sampled_frames
        self.sampler_grid = sampler_grid
        self.frameloader_vsf = frameloader_vsf

    def __getitem__(self, index):
        # Extract frames
        video_path = self.video_path
        nframes = self.nframes
        i0 = self.sampled_frames[index]

        finds_to_sample = self.sampler_grid.apply(i0, nframes)
        frame_list, resize_params, ccrop_params = \
            self.frameloader_vsf.prepare_frame_list(
                    video_path, finds_to_sample)
        meta = {
                'index': index,
                'i0': i0,
                'do_not_collate': True}
        return frame_list, meta

    def __len__(self):
        return len(self.sampled_frames)


def extract_dataset_fullframe_features(workfolder, cfg_dict, add_args):
    out, = vst.exp.get_subfolders(workfolder, ['out'])
    cfg = vst.exp.YConfig(cfg_dict)
    Ncfg_daly.set_defcfg_v2(cfg)
    Ncfg_extractor.set_defcfg_v2(cfg)
    cf = cfg.parse()

    # prepare extractor
    norm_mean_t, norm_std_t, sampler_grid, frameloader_vsf, fextractor = \
            Ncfg_extractor.prepare(cf)
    BATCH_SIZE = cf['extraction.batch_size']
    NUM_WORKERS = cf['extraction.num_workers']

    # prepare data
    dataset: Dataset_daly_ocv = Ncfg_daly.get_dataset(cf)

    # / extract
    def prepare_func(start_i):
        remaining_keyframes_dict = dict(list(
            keyframes_dict.items())[start_i+1:])
        tdataset_kf = TDataset_over_keyframes(
                remaining_keyframes_dict, sampler_grid, frameloader_vsf)
        loader = torch.utils.data.DataLoader(tdataset_kf,
                batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=True,
                collate_fn=sequence_batch_collate_v2)
        return loader

    bboxes_batch_index = torch.arange(
        BATCH_SIZE).type(torch.DoubleTensor)[:, None]

    def func(data_input):
        metas, Xts, bboxes = data_input
        kkeys = [tuple(m['kkey']) for m in metas]
        Xts_f32c = [to_gpu_normalize_permute(
            x, norm_mean_t, norm_std_t) for x in Xts]

        bsize = bboxes.shape[0]
        bboxes0 = torch.cat(
                (bboxes_batch_index[:bsize], bboxes), axis=1)
        bboxes0_c = bboxes0.type(torch.cuda.FloatTensor)
        with torch.no_grad():
            result = fextractor.forward(Xts_f32c, bboxes0_c)
        result_dict = {k: v.cpu().numpy()
                for k, v in result.items()}
        last_i = list(keyframes_dict.keys()).index(kkeys[-1])
        return result_dict, last_i

    def extract_func(key):
        pass

    stride = 4
    features_temp = vst.mkdir(out/'features')
    for vid, video in dataset.videos_ocv.items():
        output_file = features_temp/f'{vid}.pkl'
        if output_file.exists():
            continue
        # Extract keyframes specifically
        all_keyframes = []
        for action_name, instances in video['instances'].items():
            for ins_ind, instance in enumerate(instances):
                keyframes = [int(kf['frame'])
                        for kf in instance['keyframes']]
                all_keyframes.extend(keyframes)
        # Sample at stride
        strided_frames = set(range(0, video['nframes'], stride))
        frames_to_sample = np.array(
                sorted(set(all_keyframes) | strided_frames))
        # Dataset
        tdataset_kf = TDataset_over_frames(
                video['path'], video['nframes'], frames_to_sample,
                sampler_grid, frameloader_vsf)
        loader = torch.utils.data.DataLoader(tdataset_kf,
                batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=True,
                collate_fn=sequence_batch_collate_v2)
        pbar = tqdm(loader, total=len(tdataset_kf))
        features = []
        for data_input in pbar:
            frame_list, metas = data_input
            Xts_f32c = [to_gpu_normalize_permute(
                x, norm_mean_t, norm_std_t) for x in frame_list]
            with torch.no_grad():
                result = fextractor.forward(Xts_f32c, None)
            features.append(result)

        import pudb; pudb.set_trace()  # XXX BREAKPOINT
        pass


    # disaver_fold = vst.mkdir(out/'disaver')
    # total = len(dataset.videos_ocv)
    # disaver = Dataloader_isaver(disaver_fold, total, func, prepare_func,
    #     save_interval_seconds=cf['extraction.save_interval'])
    # outputs = disaver.run()
    #
    # keys = next(iter(outputs)).keys()
    # dict_outputs = {}
    # for k in keys:
    #     stacked = np.vstack([o[k] for o in outputs])
    #     dict_outputs[k] = stacked
    # vst.save_pkl(out/'dict_outputs.pkl', dict_outputs)
    # vst.save_pkl(out/'keyframes.pkl', keyframes)
