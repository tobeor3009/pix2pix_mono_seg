import logging
from datetime import datetime


class TrainLogger:
    def __init__(self):
        self.logger = None
        self.__define_logger()

        self.log_info_dict = {
            "epoch": "",
            "discriminator_acces": "",
            "Mean generator loss": "",
            "Max generator loss": "",
            "Min generator loss": "",
            "Generator loss decrease": "",
            "Current lowest generator loss": "",
            "Current Learning_rate": "",
            "Elapsed_time": "",
        }

    def get_log(self, *args):
        # setting args to log_info_dict
        for key, value in zip(self.log_info_dict, args):
            self.log_info_dict[key] = value

        message = []
        for key, value in self.log_info_dict.items():
            message.append(f"{key} : {value}")
        self.logger.info("\n".join(message))

    def __define_logger(self):

        logger = logging.getLogger("train")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "{\n%(asctime)s - %(name)s - %(levelname)s - \n%(message)s\n}"
        )
        stream_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(f"./log/train_{datetime.now()}.log")
        stream_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
        self.logger = logger

