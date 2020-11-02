import logging

log = logging.getLogger(__name__)

# Experiments


def demo(workfolder, cfg_dict, add_arg):
    out, = vst.exp.get_subfolders(workfolder, ['out'])
    cfg = vst.exp.YConfig(cfg_dict)
    cfg.set_defaults_yaml("""
    seed: 42
    """)
    cf = cfg.parse()
    log.info('Demo success')
