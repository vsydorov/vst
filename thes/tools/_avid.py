import os
import re
import logging
import random
import numpy as np
from dataclasses import dataclass, asdict

import torch
import torch.backends.cudnn as cudnn

from vsydorov_tools import small

log = logging.getLogger(__name__)


def set_env():
    os.environ['TORCH_HOME'] = \
            '/home/vsydorov/scratch/gpuhost7/bulk/torch_home/'
    cudnn.benchmark = False
    cudnn.enabled = False


def set_manual_seed(manualseed):
    random.seed(manualseed)
    np.random.seed(manualseed)
    torch.manual_seed(manualseed)
    torch.cuda.manual_seed(manualseed)


def prepare_charades_data(cache_folder, mirror, source, split_id):
    data = DatasetCharades()
    data.populate_from_folder(cache_folder)
    data.set_video_source(mirror, source)
    data.set_split(split_id)
    return data


def cull_vids_fraction(vids, fraction):
    if fraction is None:
        return vids
    shuffle = fraction < 0.0
    fraction = abs(fraction)
    assert 0.0 <= fraction < 1.0
    N_total = int(len(vids) * fraction)
    if shuffle:
        culled_vids = np.random.permutation(vids)[:N_total]
    else:
        culled_vids = vids[:N_total]
    return culled_vids


def prepare_charades_vids(data, inputs_vids_qeval, cull_specs):
    # Prepare vids
    train_vids = [v for v, s in data.split.items() if s == 'train']
    val_vids = [v for v, s in data.split.items() if s == 'val']
    if inputs_vids_qeval:
        qeval_vids = small.load_pkl(inputs_vids_qeval)
    else:
        qeval_vids = None
    # Cull if necessary
    if cull_specs is not None:
        train_vids = cull_vids_fraction(train_vids, cull_specs['train'])
        val_vids = cull_vids_fraction(val_vids, cull_specs['val'])
        if qeval_vids:
            qeval_vids = cull_vids_fraction(qeval_vids, cull_specs['qeval'])
    # Eval dict
    eval_vids_dict = {'eval': val_vids}
    if qeval_vids:
        eval_vids_dict['qeval'] = qeval_vids
    return train_vids, eval_vids_dict


def _follow_longest_symlink(path):
    """hack inherent in the dervo system"""
    # Find symlink with longest name :DDD
    symlinks = [f for f in path.iterdir() if f.is_symlink()]
    longest_symlink = max(symlinks, key=lambda x: len(x.name))
    # Resolve
    longest_symlink = longest_symlink.resolve()
    return longest_symlink


def np_multilabel_batch_accuracy_topk(outputs, targets, topk):
    """Computes the precision@k for the specified values of k"""
    batch_size = targets.shape[0]
    maxk = max(topk)
    topk_ids = outputs.argsort(axis=1)[:, ::-1][:, :maxk]

    correct = np.zeros_like(topk_ids)
    for i in range(correct.shape[0]):
        for j in range(correct.shape[1]):
            correct[i, j] = targets[i, topk_ids[i, j]] > 0.5

    res = []
    for k in topk:
        correct_k = correct[:, :k].sum()
        res.append(correct_k/batch_size)
    return res


def additional_logging(rundir, start_gstep):
    # Also log to rundir
    id_string = snippets.get_experiment_id_string()
    logfilename = small.mkdir(rundir)/'{}_{}.log'.format(
            start_gstep, id_string)
    out_filehandler = logging.FileHandler(str(logfilename))
    LOG_FORMATTER = logging.Formatter(
            "%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    out_filehandler.setFormatter(LOG_FORMATTER)
    out_filehandler.setLevel(logging.INFO)
    logging.getLogger().addHandler(out_filehandler)
    # Also print platform info
    log.info(snippets.platform_info())


def _map(submission_array, gt_array):
    """ Returns mAP, weighted mAP, and AP array """
    m_aps = []
    n_classes = submission_array.shape[1]
    for oc_i in range(n_classes):
        sorted_idxs = np.argsort(-submission_array[:, oc_i])
        tp = gt_array[:, oc_i][sorted_idxs] == 1
        fp = np.invert(tp)
        n_pos = tp.sum()
        if n_pos < 0.1:
            m_aps.append(float('nan'))
            continue
        fp.sum()
        f_pcs = np.cumsum(fp)
        t_pcs = np.cumsum(tp)
        prec = t_pcs / (f_pcs+t_pcs).astype(float)
        avg_prec = 0
        for i in range(submission_array.shape[0]):
            if tp[i]:
                avg_prec += prec[i]
        m_aps.append(avg_prec / n_pos.astype(float))
    m_aps = np.array(m_aps)
    m_ap = np.nanmean(m_aps)
    w_ap = (m_aps * gt_array.sum(axis=0) / gt_array.sum().sum().astype(float))
    return m_ap, w_ap, m_aps


def charades_map(submission_array, gt_array):
    """
    Approximate version of the charades evaluation function
    For precise numbers, use the submission file with the official matlab script
    """
    fix = submission_array.copy()
    empty = np.sum(gt_array, axis=1) == 0
    fix[empty, :] = np.NINF
    return _map(fix, gt_array)


def _print_df(df):
    _values = df.reset_index().values
    _columns = ['', ] + list(df.columns)
    _stbl = snippets.string_table(_values, _columns,
            col_formats=['{}']+['{:.2f}']*len(_columns))
    return _stbl


# == Evaluation ==

@dataclass
class Metrics_Charades:
    # map, acc in [0,1]
    mAP: float=np.nan
    acc1: float=np.nan
    acc5: float=np.nan
    loss: float=np.nan
    tloss: float=np.nan


def char_metrics_to_string(m: Metrics_Charades) -> str:
    m_dict = asdict(m)
    m_dict['mAP'] = m_dict['mAP']*100
    m_dict['acc1'] = m_dict['acc1']*100
    m_dict['acc5'] = m_dict['acc5']*100
    metrics_str = ' '.join((
            'mAP: {mAP:.5f}%',
            'acc1: {acc1:.5f}%',
            'acc5: {acc5:.5f}%',
            'loss: {loss:.5f}',
            'tloss: {tloss:.5f}'
            )).format(**m_dict)
    return metrics_str


def compute_charades_metrics(inter_outputs) -> Metrics_Charades:
    if len(inter_outputs) == 0:
        log.warn('Trying to compute scores on empty video outputs')
        return Metrics_Charades()

    scores_, score_targets_ = zip(*[
            (x['scores'], x['score_target'])
            for x in inter_outputs.values()])
    scores_ = np.array([x for x in scores_])
    score_targets_ = np.array([x for x in score_targets_])
    vscores_ = scores_.max(1)
    vscore_targets_ = score_targets_.max(1)
    mAP, wAP, ap = charades_map(vscores_, vscore_targets_)
    acc1, acc5 = np_multilabel_batch_accuracy_topk(
            vscores_,
            vscore_targets_, topk=(1, 5))
    loss = np.mean([x['loss']
        for x in inter_outputs.values()])
    train_loss = np.mean([x['train_loss']
        for x in inter_outputs.values()])
    return Metrics_Charades(mAP, acc1, acc5, loss, train_loss)
