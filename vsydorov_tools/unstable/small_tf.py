"""
Helpers for tensorflow function
"""
import subprocess
import scipy
import numpy as np
import cv2  # type: ignore
import re
import logging
from contextlib import contextmanager
from pathlib import Path
from collections import OrderedDict, defaultdict
from typing import Union, Any, NamedTuple, List, Tuple, Callable, TypeVar, Iterator, Iterable, Sequence  # NOQA

import tensorflow as tf


log = logging.getLogger(__name__)

"""
Neat, I didn't know you could do that
# Collect outputs
with slim.arg_scope(
        [slim.convolution, slim.fully_connected, slim.pool],
        outputs_collections=end_points_collection):
    end_points_collection = sc.name + '_end_points'
    ...
    end_points = slim.utils.convert_collection_to_dict(end_points_collection)

"""


def get_variables_in_checkpoint_file(file_name):
    reader = tf.train.NewCheckpointReader(file_name)
    var_to_shape_map = reader.get_variable_to_shape_map()
    return var_to_shape_map


def load_graph_from_pb(pb_filepath):
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(str(pb_filepath), 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph
