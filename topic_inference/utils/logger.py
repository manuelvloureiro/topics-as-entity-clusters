import os
import sys
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime

LOGSPATH = Path('logs')
LOGSPATH.mkdir(parents=True, exist_ok=True)
FILENAME = str(LOGSPATH / f"log_{datetime.now().strftime('%Y%m%d')}.txt")

FORMAT = '%(asctime)s : %(levelname)-4.4s : ' \
         '%(processName)-12.12s : ' \
         '%(process)-8.8s : ' \
         '%(name)-25.25s > %(message)s'


def get_logger(
        file_level=logging.DEBUG,
        stderr_level=logging.INFO,
        file_format=FORMAT,
        console_format='%(message)s',
        filename=FILENAME,
        verbose=True,
        message=None,
        program=None,
        **kwargs
):
    program = program or Path(os.path.basename(sys.argv[0])).name
    message = message or "Running " + ' '.join(sys.argv)

    logger = logging.getLogger(program)

    logging.basicConfig(
        level=file_level,
        format=file_format,
        filename=filename,
        filemode='a',
        **kwargs
    )

    if verbose:
        console = logging.StreamHandler()
        console.setLevel(stderr_level)
        console.setFormatter(logging.Formatter(console_format))
        logger.addHandler(console)

    if message:
        logger.info(message)

    return logger


def console_log(message, program):
    get_logger(message=message.strip(), program=program)


default_logger = get_logger()
