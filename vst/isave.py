# Incremental savers
import time
import re
import logging
import numpy as np
from tqdm import tqdm
from abc import ABC
from pathlib import Path
from typing import (  # NOQA
            Optional, Iterable, List, Dict,
            Any, Union, Callable, TypeVar)

from vst import small


log = logging.getLogger(__name__)


class Isaver_base(ABC):
    def __init__(self, folder, total):
        self._re_finished = (
            r'item_(?P<i>\d+)_of_(?P<N>\d+).finished')
        self._fmt_finished = 'item_{:04d}_of_{:04d}.finished'
        self._history_size = 3

        self._folder = folder
        self._total = total

    def _get_filenames(self, i) -> Dict[str, Path]:
        base_filenames = {
            'finished': self._fmt_finished.format(i, self._total)}
        base_filenames['pkl'] = Path(base_filenames['finished']).with_suffix('.pkl')
        filenames = {k: self._folder/v for k, v in base_filenames.items()}
        return filenames

    def _get_intermediate_files(self) -> Dict[int, Dict[str, Path]]:
        """Check re_finished, query existing filenames"""
        intermediate_files = {}
        for ffilename in self._folder.iterdir():
            matched = re.match(self._re_finished, ffilename.name)
            if matched:
                i = int(matched.groupdict()['i'])
                # Check if filenames exist
                filenames = self._get_filenames(i)
                all_exist = all([v.exists() for v in filenames.values()])
                assert ffilename == filenames['finished']
                if all_exist:
                    intermediate_files[i] = filenames
        return intermediate_files

    def _purge_intermediate_files(self):
        """Remove old saved states"""
        intermediate_files: Dict[int, Dict[str, Path]] = \
                self._get_intermediate_files()
        inds_to_purge = np.sort(np.fromiter(
            intermediate_files.keys(), np.int))[:-self._history_size]
        files_purged = 0
        for ind in inds_to_purge:
            filenames = intermediate_files[ind]
            for filename in filenames.values():
                filename.unlink()
                files_purged += 1
        log.debug('Purged {} states, {} files'.format(
            len(inds_to_purge), files_purged))


class Isaver_mixin_restore_save(object):
    def _restore(self):
        intermediate_files: Dict[int, Dict[str, Path]] = \
                self._get_intermediate_files()
        start_i, ifiles = max(intermediate_files.items(),
                default=(-1, None))
        if ifiles is not None:
            restore_from = ifiles['pkl']
            self.result = small.load_pkl(restore_from)
            log.info('Restore from {}'.format(restore_from))
        return start_i

    def _save(self, i):
        ifiles = self._get_filenames(i)
        savepath = ifiles['pkl']
        small.save_pkl(savepath, self.result)
        ifiles['finished'].touch()


class Isaver_simple(Isaver_mixin_restore_save, Isaver_base):
    """
    Will process a list with a func

    - save_perid: SSLICE spec
    - log_interval, save_inverval: in seconds
    """
    def __init__(self, folder, arg_list, func,
            save_period='::',
            save_interval=120,  # every 2 minutes by default
            log_interval=None,):
        super().__init__(folder, len(arg_list))
        self.arg_list = arg_list
        self.result = []
        self.func = func
        self._save_period = save_period
        self._save_interval = save_interval
        self._log_interval = log_interval

    def run(self):
        start_i = self._restore()
        run_range = np.arange(start_i+1, self._total)
        self._time_last_save = time.perf_counter()
        self._time_last_log = time.perf_counter()
        pbar = tqdm(run_range)
        for i in pbar:
            self.result.append(self.func(self.arg_list[i]))
            # Save check
            SAVE = small.check_step(i, self._save_period)
            if self._save_interval:
                since_last_save = time.perf_counter() - self._time_last_save
                SAVE |= since_last_save > self._save_interval
            SAVE |= (i+1 == self._total)
            if SAVE:
                self._save(i)
                self._purge_intermediate_files()
                self._time_last_save = time.perf_counter()
            # Log check
            if self._log_interval:
                since_last_log = time.perf_counter() - self._time_last_log
                if since_last_log > self._log_interval:
                    log.info(small.tqdm_str(pbar))
                    self._time_last_log = time.perf_counter()
        return self.result
