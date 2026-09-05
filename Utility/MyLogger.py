import logging
import os
import sys
from datetime import datetime


class MyLogger:
    def __init__(self, log_dir, log_file_name):
        self.log_dir = log_dir
        self.log_file_name = log_file_name
        self.logger = None
        self.output_path = None

    def setup(self):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.output_path = os.path.join(self.log_dir, timestamp)

        os.makedirs(self.output_path, exist_ok=True)

        log_file = os.path.join(self.output_path, self.log_file_name)

        logger = logging.getLogger("MyLogger")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        # Remove previous handlers
        for handler in logger.handlers[:]:
            handler.flush()
            handler.close()
            logger.removeHandler(handler)

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )

        file_handler = logging.FileHandler(
            log_file,
            mode="w",
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)

        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        self.logger = logger

        return logger, self.output_path

    def close(self):
        if self.logger:
            for handler in self.logger.handlers[:]:
                handler.flush()
                handler.close()
                self.logger.removeHandler(handler)

            self.logger = None

        logging.shutdown()