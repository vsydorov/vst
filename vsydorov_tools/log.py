"""
Snippets to help manage python logging
"""
import io
import logging
from contextlib import contextmanager

log = logging.getLogger(__name__)


reasonable_formatters = {
    'extended': logging.Formatter(
        "%(asctime)s %(name)s %(funcName)s %(levelname)s: %(message)s",
        "%Y-%m-%d %H:%M:%S"),
    'short': logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s: %(message)s",
        "%Y-%m-%d %H:%M:%S"),
    'shortest': logging.Formatter(
        "%(asctime)s: %(message)s",
        "%Y-%m-%d %H:%M:%S")}


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
    def __init__(self, pause_others=False):
        self.pause_others = pause_others
        self._logger = logging.getLogger()
        self._captor_handler = CaptureLogRecordsHandler()
        self.captured = []

    def _pause_other_handlers(self):
        self._other_handlers = self._logger.handlers.copy()
        for handle in self._logger.handlers:
            self._logger.removeHandler(handle)

    def _unpause_other_handlers(self):
        for handle in self._other_handlers:
            self._logger.addHandler(handle)

    def __enter__(self):
        if self.pause_others:
            self._pause_other_handlers()
        self._logger.addHandler(self._captor_handler)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.pause_others:
            self._unpause_other_handlers()
        self._logger.removeHandler(self._captor_handler)
        self.captured = \
                self._captor_handler.captured_records[:]
        # If exception was raise - handle captured right now
        if exc_type is not None:
            log.error('<<(CAPTURED BEGIN)>> Capturer encountered an '
                    'exception and released captured records')
            self.handle_captured()
            log.error('<<(CAPTURED END)>> End of captured records')

    def handle_captured(self):
        for record in self.captured:
            self._logger.handle(record)


class LogCaptorToString(object):
    def __init__(self,
            loglevel=logging.DEBUG,
            pause_other_handlers=False):

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
                self._log_capture_string)
        self._temporary_stream_handler.setLevel(self.loglevel)
        self._logger.addHandler(self._temporary_stream_handler)
        return self

    def __exit__(self, *args):
        self.captured = self._log_capture_string.getvalue()
        self._logger.removeHandler(self._temporary_stream_handler)
        if self.pause_other_handlers:
            for handle in self._other_handlers:
                self._logger.addHandler(handle)


def add_filehandler(logfilename,
        level=logging.DEBUG,
        formatter='extended'):
    if isinstance(formatter, str):
        formatter = reasonable_formatters[formatter]
    out_filehandler = logging.FileHandler(str(logfilename))
    out_filehandler.setFormatter(formatter)
    out_filehandler.setLevel(level)
    logging.getLogger().addHandler(out_filehandler)
    return logfilename


def reasonable_logging_setup(
        stream_loglevel: int,
        formatter='extended'):
    """ Create STDOUT stream handler, curtail spam """
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
    spammy_packages = ['PIL', 'git']
    spammy_packages.append('tensorflow')  # Prevent tensorflow spam
    spammy_packages.append('matplotlib')  # Prevent tensorflow spam
    for packagename in spammy_packages:
        logging.getLogger(packagename).setLevel(
                max(logging.INFO, stream_loglevel))
    return logger
