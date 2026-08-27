from datetime import datetime
from Utility.MyLogger import MyLogger
import time
from Configs import SyntheticDatasetConfig
from SyntheticDatasetGenerator import SyntheticDatasetGenerator
from Utility.TimeFormatHelper import TimeFormatHelper

def main():
    try:
        start_time = time.time()
        start_datetime = datetime.now()

        logger_obj = MyLogger(log_dir=r"SyntheticDatasetGenerator_log", log_file_name="logs.txt")
        logger, path = logger_obj.setup()
        logger.info(f"Started: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

        syntheticDatasetConfig = SyntheticDatasetConfig(logger)
        syntheticDatasetConfig.log()
        generator = SyntheticDatasetGenerator(logger,syntheticDatasetConfig)

        generator.run()
        end_time = time.time()
        end_datetime = datetime.now()
        elapsed = end_time - start_time
        logger.info(f"Finished: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"Execution Time: {TimeFormatHelper.format_elapsed(elapsed)}\n")

    except ValueError as error:
        print(f"ERROR: {error}")

if __name__ == "__main__":
    main()
