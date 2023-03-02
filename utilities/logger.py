import logging
import datetime
import os.path


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Logger(metaclass=Singleton):
    def __init__(self):
        timestamp = datetime.datetime.now().strftime("logfile_%Y-%m-%d_%H-%M-%S.log")
        log_filename = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", timestamp))
        logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', filename=log_filename, level=logging.DEBUG)
        self.logger = logging.getLogger()

    def get_logger(self):
        return self.logger
