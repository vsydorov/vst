"""
VST - Vladyslav Sydorov Tools

Lib contains useful snippets and other boilerplate code
"""

from . import exp, isave, small
from .small import *

__all__ = ["exp", "isave", "small"]
# 'plot' not exported automatically, I don't want vst to have opencv as requirement
