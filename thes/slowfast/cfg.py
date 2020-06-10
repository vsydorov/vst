import yacs
from pathlib import Path

import slowfast
from slowfast.config.defaults import get_cfg

def base_sf_i3d_config():
    sf_cfg = get_cfg()
    # I3D keys
    YML_RPATH = 'Kinetics/c2/I3D_8x8_R50.yaml'
    yml_path = Path(slowfast.__file__).parents[1]/f'configs/{YML_RPATH}'
    # Can be done via merge_from_file
    with yml_path.open('r') as f:
        yml_string = f.read()
    loaded_cfg = yacs.config.CfgNode.load_cfg(yml_string)
    sf_cfg.merge_from_other_cfg(loaded_cfg)
    return sf_cfg
