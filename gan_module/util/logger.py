import logging
from datetime import datetime

DATE_FORMAT = "%Y-%m-%d %H.%M"


class TrainLogger:
    def __init__(self):
        self.logger = None
        self.__define_logger()

        self.log_info_dict = {
            "Epoch": "",
            "Discriminator_acces": "",
            "Mean generator loss": "",
            "Max generator loss": "",
            "Min generator loss": "",
            "Generator loss decrease": "",
            "Current lowest generator loss": "",
            "Current Learning_rate": "",
            "Elapsed_time": "",
        }

        self.history_info_dict = {
            "Epoch": [],
            "Discriminator_acces": [],
            "Generator loss": [],
            "Train_f1_loss": [],
            "Valid_f1_loss": [],
        }

    def write_log(self, *args):
        # setting args to log_info_dict
        for key, value in zip(self.log_info_dict, args):
            self.log_info_dict[key] = value

        message = []
        for key, value in self.log_info_dict.items():
            message.append(f"{key} : {value}")
        self.logger.info("\n".join(message))

    def add_histroy(self, *args):
        # setting args to history_info_dict
        for key, value in zip(self.history_info_dict, args):
            self.history_info_dict[key] = value

        message = []
        for key, value in self.history_info_dict.items():
            message.append(f"{key} : {value}")
        self.logger.info("\n".join(message))

    def __define_logger(self):

        logger = logging.getLogger("train")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "{\n%(asctime)s - %(name)s - %(levelname)s - \n%(message)s\n}"
        )
        stream_handler = logging.StreamHandler()
        current_time = datetime.now().strftime(DATE_FORMAT)
        file_handler = logging.FileHandler(f"./log/train_{current_time}.log")
        stream_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
        self.logger = logger
