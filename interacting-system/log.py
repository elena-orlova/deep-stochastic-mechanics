import logging
import sys
import os
from logging import Logger, Formatter
from copy import deepcopy
from typing import Optional, Dict, Any, Union, MutableMapping, Any, Tuple
import importlib
import warnings

FORMAT = "[%(asctime)s][%(levelname)s][%(name)s]%(message)s (%(filename)s:%(lineno)d)"
FORMAT_LOGURU = "[<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>][<level>{level: <8}</level>][<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>][<magenta>rk:{extra[rank]}</magenta>] - <level>{message}</level>"


def get_logging_level() -> int:
    return os.environ.get("LOG_LEVEL", logging.INFO)


def is_loguru_installed() -> bool:
    return importlib.util.find_spec("loguru") is not None


def setup_warning_redirect(logger: Logger):
    def showwarning(message, *args, **kwargs):
        logger.warning(message)

    warnings.showwarning = showwarning


def setup_exceptions_redirect(logger: Logger):
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        if is_loguru_installed():
            logger.opt(exception=(exc_type, exc_value, exc_traceback)).error(
                "Uncaught exception:"
            )
        else:
            logger.error(
                "Uncaught exception:", exc_info=(exc_type, exc_value, exc_traceback)
            )

    sys.excepthook = handle_exception


class ColorFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = (
        "[%(asctime)s][%(levelname)s][%(name)s]%(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def zero_rank_filter(record: Union[logging.LogRecord, Dict]) -> bool:
    if isinstance(record, dict):
        # loguru
        exc_info = record["extra"].get("exc_info", None)
        if "rank" not in record["extra"]:
            record["extra"]["rank"] = 0
    else:
        exc_info = record.exc_info
    return True


class ZeroRankFilter(logging.Filter):
    def filter(self, record: logging.LogRecord):
        return zero_rank_filter(record)


if is_loguru_installed():
    import loguru

    # for compatibility
    ZeroRankLogger = lambda _: loguru.logger.bind(rank=0)
else:

    class ZeroRankLogger(logging.LoggerAdapter):
        def __init__(self, name: Optional[str] = None):
            self.logger: Logger = logging.getLogger(name)
            self.logger.addFilter(ZeroRankFilter())
            super().__init__(self.logger, extra={"rank": 0})

        def process(
            self, msg: Any, kwargs: MutableMapping[str, Any]
        ) -> Tuple[Any, MutableMapping[str, Any]]:
            return "[rk:%s] %s" % (self.extra["rank"], msg), kwargs



def _setup_std_logger(level: Union[int, str] = logging.INFO):
    fmt = FORMAT
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.formatter = ColorFormatter(fmt=fmt)
    logging.basicConfig(
        handlers=[handler],
        force=True,
    )
    logger = ZeroRankLogger(name=None)
    setup_exceptions_redirect(logger)
    setup_warning_redirect(logger)

    logger.setLevel(level)


def _setup_loguru_logger(level: Union[int, str] = logging.INFO):
    from loguru import logger

    try:
        # by default loguru writes to stderr
        # here we remove standard logger to stderr
        # (see https://github.com/Delgan/loguru/blob/master/loguru/__init__.py#L31)
        logger.remove()
    except ValueError:
        pass

    logger.add(
        sys.stdout,
        level=level,
        colorize=True,
        format=FORMAT_LOGURU,
        filter=zero_rank_filter,
        backtrace=True,
        diagnose=True,
    )
    setup_exceptions_redirect(logger)
    setup_warning_redirect(logger)


def setup_logger():
    level = get_logging_level()
    if is_loguru_installed():
        _setup_loguru_logger(level=level)
    else:
        _setup_std_logger(level=level)
        logging.getLogger().warning("Please install `loguru` for better logging.")
        