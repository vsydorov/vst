import logging

log = logging.getLogger(__name__)


def filter_tube_keyframes_only_gt(dataset, tubes_per_video):
    gt_tubes = get_daly_gt_tubes(dataset)
    gt_df = gt_tubes_to_df(dataset, gt_tubes)
    # Query good inds per vid
    good_inds_per_vid = {}
    for vid, gindices in gt_df.groupby('vid').groups.items():
        qdf = gt_df.loc[gindices]
        sorted_inds = sorted(
                itertools.chain.from_iterable(qdf.frame_inds.tolist()))
        good_inds_per_vid[vid] = sorted_inds
    # Filter tubes to only gt keyframes
    filtered_tubes = {}
    for k, v in tqdm(tubes_per_video.items(), 'filter_tubes'):
        (vid, bunch_id, tube_id) = k
        good_inds = good_inds_per_vid[vid]
        intersecting_inds, comm1, comm2 = \
            np.intersect1d(v['frame_inds'], good_inds, return_indices=True)
        if len(intersecting_inds):
            v_intersect = {}
            for k0, v0 in v.items():
                v_intersect[k0] = v0[comm1]
            filtered_tubes[k] = v_intersect
    return filtered_tubes


def get_subset_tubes_(split_vids, tubes_per_video):
    subset_tubes_per_video: Dict[DALY_tube_index, DALY_wein_tube] = {}
    for k, v in tubes_per_video.items():
        (vid, bunch_id, tube_id) = k
        if vid in split_vids:
            subset_tubes_per_video[k] = v
    return subset_tubes_per_video


def get_subset_tubes(dataset, split_label, tubes_per_video):
    split_vids = get_daly_split_vids(dataset, split_label)
    return get_subset_tubes_(split_vids, tubes_per_video)


def sample_some_tubes(tubes_per_video, N=10, NP_SEED=0):
    np_rstate = np.random.RandomState(NP_SEED)
    prm_key_indices = np_rstate.permutation(
            np.arange(len(tubes_per_video)))
    key_list = list(tubes_per_video.keys())
    some_keys = [key_list[i] for i in prm_key_indices[:N]]
    some_tubes = {k: tubes_per_video[k] for k in some_keys}
    return some_tubes


def _set_tubes(cf, dataset):
    tubes_per_video: Dict[DALY_tube_index, DALY_wein_tube] = \
        small.load_pkl(cf['tubes.imported_wein_tubes'])
    if cf['tubes.filter_gt']:
        tubes_per_video = filter_tube_keyframes_only_gt(
                dataset, tubes_per_video)
    split_label = cf['dataset.subset']
    tubes_per_video = \
            get_subset_tubes(dataset, split_label, tubes_per_video)
    return tubes_per_video


def eval_daly_tubes_RGB_demovis(workfolder, cfg_dict, add_args):
    raise NotImplementedError()


def eval_daly_tubes_RGB(workfolder, cfg_dict, add_args):
    raise NotImplementedError()


def hacky_gather_evaluated_tubes(workfolder, cfg_dict, add_args):
    raise NotImplementedError()


def actual_eval_of_nicphil_etubes(workfolder, cfg_dict, add_args):
    raise NotImplementedError()


def actual_eval_of_action_object_predictions(workfolder, cfg_dict, add_args):
    raise NotImplementedError()


def assign_objactions_to_tubes(workfolder, cfg_dict, add_args):
    raise NotImplementedError()
