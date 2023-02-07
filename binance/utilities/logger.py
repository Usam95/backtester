import logging

file_name = "main_logger.log"
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

handler = logging.FileHandler(file_name)
handler.setFormatter(formatter)

logger.addHandler(handler)