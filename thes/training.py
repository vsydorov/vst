import logging
import re
import string
import shutil
import random
from datetime import datetime
from pathlib import Path

from vsydorov_tools import small

log = logging.getLogger(__name__)

class Manager_checkpoint_name(object):
    ckpt_re = r'model_at_epoch_(?P<i_epoch>\d*).pth.tar'
    ckpt_format = 'model_at_epoch_{:03d}.pth.tar'

    @classmethod
    def get_checkpoint_path(self, rundir, i_epoch) -> Path:
        save_filepath = rundir/self.ckpt_format.format(i_epoch)
        return save_filepath

    @classmethod
    def find_checkpoints(self, rundir):
        checkpoints = {}
        for subfolder_item in rundir.iterdir():
            search = re.search(self.ckpt_re, subfolder_item.name)
            if search:
                i_epoch = int(search.groupdict()['i_epoch'])
                checkpoints[i_epoch] = subfolder_item
        return checkpoints

    @classmethod
    def find_last_checkpoint(self, rundir):
        checkpoints = self.find_checkpoints(rundir)
        if len(checkpoints):
            checkpoint_path = max(checkpoints.items())[1]
        else:
            checkpoint_path = None
        return checkpoint_path

    @staticmethod
    def rename_old_rundir(rundir):
        if len(list(rundir.iterdir())) > 0:
            timestamp = datetime.fromtimestamp(
                    rundir.stat().st_mtime).strftime('%Y-%m-%d_%H:%M:%S')
            str_rnd = ''.join(random.choices(string.ascii_uppercase, k=3))
            new_foldname = f'old_{timestamp}_{str_rnd}'
            log.info(f'Existing experiment moved to {new_foldname}')
            shutil.move(rundir, rundir.parent/new_foldname)
            small.mkdir(rundir)
