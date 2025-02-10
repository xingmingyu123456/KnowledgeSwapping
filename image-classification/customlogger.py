import os
import logging
import datetime
current = datetime.datetime.now()
time_str = current.strftime("%Y-%m-%d-%H-%M-%S")
class CustomLogger:
    def __init__(self):
        self.log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log',time_str)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # 创建不同级别的handler和formatter
        self.create_handler(logging.INFO, 'info')
        self.create_handler(logging.WARNING, 'warning')
        self.create_handler(logging.ERROR, 'error')

    def create_handler(self, level, filename):
        log_file = os.path.join(self.log_dir, f'{filename}.log')
        handler = logging.FileHandler(log_file)
        handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def info(self, message):
        self.logger.handlers[0].emit(logging.LogRecord('name', logging.INFO, '', 0, message, None, None))

    def warning(self, message):
        self.logger.handlers[1].emit(logging.LogRecord('name', logging.WARNING, '', 0, message, None, None))

    def error(self, message):
        self.logger.handlers[2].emit(logging.LogRecord('name', logging.ERROR, '', 0, message, None, None))

# 使用示例
logger = CustomLogger()
logger.info('This is an info message')

# logger.warning('This is a warning message')
# logger.error('This is an error message')

