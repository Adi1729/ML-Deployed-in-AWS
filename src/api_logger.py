import logging
from datetime import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter =  logging.Formatter('%(levelname)s:%(asctime)s:%(name)s:%(message)s')

time_ = datetime.now().strftime("%d_%m_%H_%M")
file_handler = logging.FileHandler(r'../logs/api_logs_'+ time_ +'.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)



