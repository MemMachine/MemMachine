import logging
import os
import socket
import time


# from atf_helper.py
# ruff: noqa: PTH103, G004, C901
def get_logger(log_file=None, log_name="atf", log_rotate=False, log_console=False):
    """
    logging with multiple files and console handlers
    Args:
        log_file: None or str (full path of log file)
        log_name: str (logger name)
        log_rotate: bool
        log_console: bool
    Return: logger instance
    """
    logging.Formatter.converter = time.localtime
    logger = logging.getLogger(log_name)
    logger.propagate = False
    host_ip = socket.gethostbyname(socket.gethostname())
    log_format = logging.Formatter(
        f"%(asctime)s [{host_ip}] [%(levelname)s %(filename)s::"
        f"%(funcName)s::%(lineno)s::%(threadName)s] %(message)s"
    )
    if log_file:
        dir_path = os.path.dirname(log_file)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
    handler_exist = False
    for handler in logger.handlers:
        if hasattr(handler, "stream") and handler.stream:
            # print(f'stream handler exists {handler.stream}')
            handler.setFormatter(log_format)
            log_console = False
        if hasattr(handler, "baseFilename") and handler.baseFilename == log_file:
            handler.setFormatter(log_format)
            handler_exist = True
    if not handler_exist:
        if log_file:
            # print(f'create file handler {log_file}')
            file_handler = logging.FileHandler(filename=log_file)
            file_handler.setFormatter(log_format)
            file_handler.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)
        if log_console:
            # print('create stream handler')
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            f = "%(asctime)s [%(levelname)s %(filename)s::%(funcName)s] %(message)s"
            console.setFormatter(logging.Formatter(f))
            logger.addHandler(console)
    else:
        for handler in logger.handlers:
            if hasattr(handler, "baseFilename") and handler.baseFilename == log_file:
                logger.info(f"{handler.baseFilename} existed")
                break
    logger.level = logging.DEBUG
    return logger
