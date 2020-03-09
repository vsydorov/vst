import os
import logging
import logging.handlers
import functools
import multiprocessing
import threading

import torch
import torch.multiprocessing as torch_mp
import torch.distributed as dist

from fvcore.common.file_io import PathManager
from detectron2.utils.logger import (
        _cached_log_stream)
import detectron2.utils.comm as comm
from detectron2.engine.launch import _find_free_port


def launch_w_logging(
        main_func, num_gpus_per_machine, num_machines=1,
        machine_rank=0, dist_url=None, args=()):
    """
    Variation of detectron2.engine.launch that logs to a queue

    Args:
        main_func: a function that will be called by `main_func(*args)`
        num_machines (int): the total number of machines
        machine_rank (int): the rank of this machine (one per machine)
        dist_url (str): url to connect to for distributed training, including
        protocol
                       e.g. "tcp://127.0.0.1:8686".
                       Can be set to auto to automatically select a free port
                       on localhost
        args (tuple): arguments passed to main_func
    """
    world_size = num_machines * num_gpus_per_machine
    if world_size > 1:
        # https://github.com/pytorch/pytorch/pull/14391
        # TODO prctl in spawned processes

        if dist_url == "auto":
            assert num_machines == 1, \
                    "dist_url=auto cannot work with distributed training."
            port = _find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"

        mp = multiprocessing.get_context('spawn')
        logmsg_queue = mp.Queue()

        lp = threading.Thread(target=my_logger_thread, args=(logmsg_queue,))
        lp.start()

        try:
            torch_mp.spawn(
                _distributed_worker_w_logging,
                nprocs=num_gpus_per_machine,
                args=(main_func, world_size, num_gpus_per_machine, machine_rank,
                    dist_url, logmsg_queue, args),
                daemon=False,
            )
        finally:
            logmsg_queue.put(None)
            lp.join(1)
            print('Kill logging thread')

    else:
        main_func(*args)


def launch_without_logging(
        main_func, num_gpus_per_machine, num_machines=1,
        machine_rank=0, dist_url=None, args=()):
    """
    Variation of detectron2.engine.launch that logs to a queue

    Args:
        main_func: a function that will be called by `main_func(*args)`
        num_machines (int): the total number of machines
        machine_rank (int): the rank of this machine (one per machine)
        dist_url (str): url to connect to for distributed training, including
        protocol
                       e.g. "tcp://127.0.0.1:8686".
                       Can be set to auto to automatically select a free port
                       on localhost
        args (tuple): arguments passed to main_func
    """
    world_size = num_machines * num_gpus_per_machine
    if world_size > 1:
        # https://github.com/pytorch/pytorch/pull/14391
        # TODO prctl in spawned processes

        if dist_url == "auto":
            assert num_machines == 1, \
                    "dist_url=auto cannot work with distributed training."
            port = _find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"

        torch_mp.spawn(
            _distributed_worker_without_logging,
            nprocs=num_gpus_per_machine,
            args=(main_func, world_size, num_gpus_per_machine, machine_rank,
                dist_url, args),
            daemon=False,
        )
    else:
        main_func(*args)


def _distributed_worker_without_logging(
        local_rank, main_func, world_size,
        num_gpus_per_machine, machine_rank,
        dist_url, args):
    """
    Based on "detectron2.engine._distributed_worker'
    """
    assert torch.cuda.is_available(), \
            "cuda is not available. Please check your installation."
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    try:
        dist.init_process_group(
            backend="NCCL", init_method=dist_url,
            world_size=world_size, rank=global_rank
        )
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("Process group URL: {}".format(dist_url))
        raise e
    # synchronize is needed here to prevent a possible timeout after calling
    # init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    comm.synchronize()

    assert num_gpus_per_machine <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    # Setup the local process group (which contains ranks within the same
    # machine)
    assert comm._LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * num_gpus_per_machine,
            (i + 1) * num_gpus_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comm._LOCAL_PROCESS_GROUP = pg

    main_func(*args)


def _distributed_worker_w_logging(
        local_rank, main_func, world_size,
        num_gpus_per_machine, machine_rank,
        dist_url, logmsg_queue, args):
    """
    Based on "detectron2.engine._distributed_worker'

    """
    assert torch.cuda.is_available(), \
            "cuda is not available. Please check your installation."
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    try:
        dist.init_process_group(
            backend="NCCL", init_method=dist_url,
            world_size=world_size, rank=global_rank
        )
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("Process group URL: {}".format(dist_url))
        raise e
    # synchronize is needed here to prevent a possible timeout after calling
    # init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    comm.synchronize()

    assert num_gpus_per_machine <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    # Setup the local process group (which contains ranks within the same
    # machine)
    assert comm._LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * num_gpus_per_machine,
            (i + 1) * num_gpus_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comm._LOCAL_PROCESS_GROUP = pg

    _distributed_queue_logging(global_rank, logmsg_queue)
    main_func(*args)


"""
My multiprocess logging setup
"""


def _distributed_queue_logging(global_rank, logmsg_queue):
    """
    Send logs to queue (only first process)
    """
    print('log_queue: setup loggin')
    root_log = logging.getLogger()
    root_log.setLevel(logging.NOTSET)
    assert len(root_log.handlers) == 0, \
            ('root handlers should be empty '
              'for multiprocessing, instead got {}'.format(root_log.handlers))
    # Add filter to ROOT LOGGER that would record global tank
    filter_record_rank = LoggingFilter_record_rank(global_rank)
    # Add filter qhandler that would only keep rank=0
    # == Add q_handler ==
    q_handler = logging.handlers.QueueHandler(logmsg_queue)
    filter_keep_rank0 = LoggingFilter_keep_rank0()
    q_handler.addFilter(filter_record_rank)
    q_handler.addFilter(filter_keep_rank0)
    root_log.addHandler(q_handler)
    print('root handlers {}'.format(root_log))


@functools.lru_cache()
def setup_logger_d2(
        output=None, distributed_rank=0, *, color=True,
        name="detectron2", abbrev_name=None):
    """
    Stub of "detectron2.utils.logger.setup_logger"
      - Remove stdout part
      - Only file logging

    Args:
        output (str): a file name or a directory to save log. If None, will not
        save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger
        abbrev_name (str): an abbreviation of the module, to avoid long names
        in logs.
            Set to "" to not log the root module in logs.
            By default, will abbreviate "detectron2" to "d2" and leave other
            modules unchanged.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = True   # We handle the rest
    if abbrev_name is None:
        abbrev_name = "d2" if name == "detectron2" else name
    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%m/%d %H:%M:%S"
    )
    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        if distributed_rank > 0:
            filename = filename + ".rank{}".format(distributed_rank)
        PathManager.mkdirs(os.path.dirname(filename))

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger


def my_logger_thread(q):
    while True:
        record = q.get()
        if record is None:
            break
        logger = logging.getLogger(record.name)
        logger.handle(record)


class LoggingFilter_record_rank(logging.Filter):
    def __init__(self, rank):
        self.rank = rank

    def filter(self, record):
        mp_name = multiprocessing.current_process().name
        record.rank = self.rank
        record.mp_name = mp_name
        return True


class LoggingFilter_keep_rank0(logging.Filter):
    def filter(self, record):
        if record.rank == 0:
            return True
        else:
            return False
