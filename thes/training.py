import re
from pathlib import Path


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
