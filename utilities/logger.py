import logging
import datetime
import os.path
from pathlib import Path

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class FlushingFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

class Logger(metaclass=Singleton):
    def __init__(self):
        # Ensure logs directory exists
        logs_dir = Path(__file__).parent.parent / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("logfile_%Y-%m-%d_%H-%M-%S.log")
        log_filename = logs_dir / timestamp
        try:
            # Setup custom logger
            self.logger = logging.getLogger()
            self.logger.setLevel(logging.INFO)  # Allowing DEBUG level messages

            # Create formatter
            formatter = logging.Formatter('%(asctime)s %(levelname)s %(funcName)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

            # File handler
            fh = FlushingFileHandler(log_filename)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

            # Stream handler (to console)
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            ch.setLevel(logging.DEBUG)  # Adjusting the console handler to DEBUG level
            self.logger.addHandler(ch)
        except Exception as e:
            # You can print or handle this exception accordingly
            print(f"Error initializing logger: {e}")

    def get_logger(self):
        return self.logger
