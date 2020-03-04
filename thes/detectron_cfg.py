import numpy as np


D2DICT_GPU_SCALING_DEFAULTS = """
gpu_scaling:
    enabled: True
    base:
        # more per gpu
        SOLVER.BASE_LR: 0.0025
        SOLVER.IMS_PER_BATCH: 2
        # less per gpu
        SOLVER.MAX_ITER: 18000
        SOLVER.STEPS: [12000, 16000]
"""


def d2dict_gpu_scaling(cf, cf_add_d2, num_gpus):
    imult = 8 / num_gpus
    prepared = {}
    prepared['SOLVER.BASE_LR'] = \
        cf['gpu_scaling.base.SOLVER.BASE_LR'] * num_gpus
    prepared['SOLVER.IMS_PER_BATCH'] = \
        cf['gpu_scaling.base.SOLVER.IMS_PER_BATCH'] * num_gpus
    prepared['SOLVER.MAX_ITER'] = \
        int(cf['gpu_scaling.base.SOLVER.MAX_ITER'] * imult)
    prepared['SOLVER.STEPS'] = tuple((np.array(
            cf['gpu_scaling.base.SOLVER.STEPS']
            ) * imult).astype(int).tolist())
    return cf_add_d2.update(prepared)
