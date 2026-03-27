import io
import logging
from contextlib import contextmanager

from vst.path import mkdir
from vst.small import get_experiment_id_string

log = logging.getLogger(__name__)

"""
Logging
"""

reasonable_formatters = {
    "extended": logging.Formatter(
        "%(asctime)s %(name)s %(funcName)s %(levelname)s: %(message)s",
        "%Y-%m-%d %H:%M:%S",
    ),
    "short": logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    ),
    "shorter": logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    ),
    "shortest": logging.Formatter(
        "%(asctime)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    ),
}


@contextmanager
def logging_disabled(disable_level=logging.CRITICAL):
    """Temporarily disable logging inside context
    http://stackoverflow.com/questions/2266646/how-to-i-disable-and-re-enable-console-logging-in-python
    """
    logging.disable(disable_level)
    yield
    logging.disable(logging.NOTSET)


class CaptureLogRecordsHandler(logging.Handler):
    def __init__(self):
        logging.Handler.__init__(self)
        self.captured_records = []

    def emit(self, record):
        self.captured_records.append(record)

    def close(self):
        logging.Handler.close(self)


class LogCaptorToRecords(object):
    """Capture log records while optionally pausing handlers.

    pause='none' — capture alongside all active handlers; handle_captured() is a no-op
    pause='all'  — pause all handlers; handle_captured() replays to all current handlers
    pause='file' — pause only FileHandlers, keep stream (stdout) active;
                   handle_captured() replays only to current FileHandlers
    """

    def __init__(self, pause="none"):
        if pause not in ("none", "all", "file"):
            raise ValueError(
                f"pause must be 'none', 'all', or 'file', got {pause!r}"
            )
        self.pause = pause
        self._logger = logging.getLogger()
        self._captor_handler = CaptureLogRecordsHandler()
        self._paused_handlers = []
        self.captured = []

    def __enter__(self):
        if self.pause == "all":
            self._paused_handlers = self._logger.handlers.copy()
        elif self.pause == "file":
            self._paused_handlers = [
                h
                for h in self._logger.handlers
                if isinstance(h, logging.FileHandler)
            ]
        for h in self._paused_handlers:
            self._logger.removeHandler(h)
        self._logger.addHandler(self._captor_handler)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._logger.removeHandler(self._captor_handler)
        for h in self._paused_handlers:
            if h not in self._logger.handlers:
                self._logger.addHandler(h)
        self.captured = self._captor_handler.captured_records[:]
        if exc_type is not None:
            log.error(
                "<<(CAPTURED BEGIN)>> Capturer encountered an "
                "exception and released captured records"
            )
            self.handle_captured()
            log.error("<<(CAPTURED END)>> End of captured records")

    def handle_captured(self):
        if self.pause == "none":
            return  # All handlers were active during capture, no replay needed
        elif self.pause == "all":
            targets = self._logger.handlers.copy()
        else:  # 'file'
            targets = [
                h
                for h in self._logger.handlers
                if isinstance(h, logging.FileHandler)
            ]
        for record in self.captured:
            for h in targets:
                h.handle(record)


class LogCaptorToString(object):
    def __init__(self, loglevel=logging.DEBUG, pause_other_handlers=False):

        self.loglevel = loglevel
        self.pause_other_handlers = pause_other_handlers
        self._logger = logging.getLogger()

    def __enter__(self):
        self._log_capture_string = io.StringIO()
        if self.pause_other_handlers:
            self._other_handlers = self._logger.handlers.copy()
            for handle in self._logger.handlers:
                self._logger.removeHandler(handle)
        self._temporary_stream_handler = logging.StreamHandler(
            self._log_capture_string
        )
        self._temporary_stream_handler.setLevel(self.loglevel)
        self._logger.addHandler(self._temporary_stream_handler)
        return self

    def __exit__(self, *args):
        self.captured = self._log_capture_string.getvalue()
        self._logger.removeHandler(self._temporary_stream_handler)
        if self.pause_other_handlers:
            for handle in self._other_handlers:
                self._logger.addHandler(handle)


def add_filehandler(logfilename, level=logging.DEBUG, formatter="extended"):
    if isinstance(formatter, str):
        formatter = reasonable_formatters[formatter]
    out_filehandler = logging.FileHandler(str(logfilename))
    out_filehandler.setFormatter(formatter)
    out_filehandler.setLevel(level)
    logging.getLogger().addHandler(out_filehandler)
    return logfilename


def reasonable_logging_setup(stream_loglevel: int, formatter="extended"):
    """Create STDOUT stream handler, curtail spam"""
    if isinstance(formatter, str):
        formatter = reasonable_formatters[formatter]
    # Get root logger (with NOTSET level)
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)
    # Stream handler takes 'loglevel'
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(stream_loglevel)
    logger.addHandler(handler)
    # Prevent some spammy packages from exceeding INFO verbosity
    spammy_packages = [
        "PIL",
        "git",
        "tensorflow",
        "matplotlib",
        "selenium",
        "urllib3",
        "fiona",
        "rasterio",
    ]
    for packagename in spammy_packages:
        logging.getLogger(packagename).setLevel(
            max(logging.INFO, stream_loglevel)
        )
    return logger


def quick_log_setup(level):
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def additional_logging(rundir):
    # Also log to rundir
    id_string = get_experiment_id_string()
    logfilename = mkdir(rundir) / "{}.log".format(id_string)
    out_filehandler = logging.FileHandler(str(logfilename))
    LOG_FORMATTER = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    out_filehandler.setFormatter(LOG_FORMATTER)
    out_filehandler.setLevel(logging.INFO)
    logging.getLogger().addHandler(out_filehandler)


def loglevel_str_to_int(loglevel: str) -> int:
    assert isinstance(loglevel, str)
    return logging._checkLevel(loglevel)  # type: ignore


def loglevel_int_to_str(loglevel: int) -> str:
    assert isinstance(loglevel, int)
    return logging.getLevelName(loglevel)


def docopt_loglevel(loglevel) -> int:
    """Tries to get int value softly.
    For parsing docopt argument
    """
    try:
        loglevel_int = int(loglevel)
    except ValueError:
        loglevel_int = loglevel_str_to_int(loglevel)
    return loglevel_int
