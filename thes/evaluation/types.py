from typing import (
    Dict, List, Tuple, TypeVar, Literal,
    Callable, TypedDict, NewType, NamedTuple)
from thes.data.tubes.types import (Frametube, Sframetube, V_dict, AV_dict)
import numpy as np

from thes.data.dataset.external import (
        DALY_vid, DALY_action_name)


class Recall_coverage(TypedDict):
    max_spatial: float
    max_spatiotemp: float


class Options_tube_ap(TypedDict):
    iou_thresh: float
    spatiotemporal: bool
    use_07_metric: bool
    use_diff: bool
