from pathlib import Path

import slowfast
from slowfast.config.defaults import get_cfg

def basic_sf_cfg(rel_yml_path):
    sf_cfg = get_cfg()
    yml_path = (Path(slowfast.__file__)
            .parents[1]/f'configs/{rel_yml_path}')
    sf_cfg.merge_from_file(yml_path)
    return sf_cfg
