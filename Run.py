from datetime import datetime
from pathlib import Path
from Configs import SyntheticDatasetConfig
from SyntheticDatasetGenerator import PersianConversationGenerator
from Utility.MyLogger import MyLogger
import time
from Utility.TimeFormatHelper import TimeFormatHelper


def main():
    try:
        start_time = time.time()
        start_datetime = datetime.now()

        logger_obj = MyLogger(log_dir="alllogs", log_file_name="logs.txt")
        logger, path = logger_obj.setup()
        logger.info(f"Started: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

        configs = SyntheticDatasetConfig(logger)
        configs.log()
        path = Path(path)

        configs.output_file = path / configs.output_file
        configs.output_temp_file = Path(str(configs.output_file) + ".tmp")

        generator = PersianConversationGenerator(logger, configs)
        generator.generate_dataset()

        end_time = time.time()
        end_datetime = datetime.now()
        elapsed = end_time - start_time
        logger.info(f"Finished: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')} | "
                    f"Execution Time: {TimeFormatHelper.format_elapsed(elapsed)}\n")

    except ValueError as error:
        print(f"ERROR: {error}")

if __name__ == "__main__":
    main()