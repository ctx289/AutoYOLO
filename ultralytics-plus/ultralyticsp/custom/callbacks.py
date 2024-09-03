import time
import logging
import os
from ultralytics.yolo.utils import RANK

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def logger_train_start(trainer):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    st = trainer.train_time_start
    readable_st = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 创建文件处理器
    file_handler = logging.FileHandler(trainer.save_dir/'train_log_file.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f'Train start time: {readable_st}')
    
    if RANK in (-1, 0):
        trainer.validator.add_callback('on_val_end', logger_val_end)

def logger_train_epoch_start(trainer):
    if RANK in (-1, 0):
        logger.info(f'Epoch {trainer.epoch}')

def logger_val_end(validator):
    logger.info(f'{validator.metrics.results_dict}')

def logger_train_end(trainer):
    et = time.time()
    readable_et = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(et))
    duration = (et - trainer.train_time_start) / 3600
    logger.info(f'Train end time: {readable_et}')
    logger.info(f'Train duration: {duration:.3f} hours')

    # NOTE. modified by ryanwfu 2023/10/08, 关闭FileHandler
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            if os.path.basename(handler.baseFilename) == 'train_log_file.log':
                logger.info(f'Close FileHandler')
                handler.close()
                logger.removeHandler(handler)
