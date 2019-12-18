import logging
from matplotlib import pyplot as plt
from contextlib import contextmanager

log = logging.getLogger(__name__)


@contextmanager
def plt_backend(backend_name):
    original = plt.get_backend()
    plt.switch_backend(backend_name)
    yield
    plt.switch_backend(original)
