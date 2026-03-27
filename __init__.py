"""
VST - Vladyslav Sydorov Tools

Lib contains useful snippets and other boilerplate code
"""

from . import isave, yml, log, small
from .path import mkdir, mkpar, npath
from .log import (
    logging_disabled,
    LogCaptorToRecords,
    add_filehandler,
    reasonable_logging_setup,
    quick_log_setup,
    additional_logging,
    loglevel_str_to_int,
    loglevel_int_to_str,
    docopt_loglevel,
)
from .small import *

# __all__ = ["yml", "isave", "small"]
# 'plot' not exported automatically, I don't want vst to have opencv as requirement
