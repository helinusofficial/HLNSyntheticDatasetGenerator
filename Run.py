from datetime import datetime
from Utility.MyLogger import MyLogger
import subprocess
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

        generator = SyntheticDatasetGenerator(
            model_path=syntheticDatasetConfig.model_path,
            output_path=syntheticDatasetConfig.output_path,
            total_samples=syntheticDatasetConfig.total_samples,
            n_ctx=syntheticDatasetConfig.n_ctx,
            n_threads=syntheticDatasetConfig.n_threads,
            n_batch=syntheticDatasetConfig.n_batch,
            n_gpu_layers=syntheticDatasetConfig.n_gpu_layers,
            seed=syntheticDatasetConfig.seed,
            language=syntheticDatasetConfig.language,
            max_tokens=syntheticDatasetConfig.max_tokens,
            shard_size=syntheticDatasetConfig.shard_size,
            checkpoint_interval=syntheticDatasetConfig.checkpoint_interval,
            max_attempts_multiplier=syntheticDatasetConfig.max_attempts_multiplier,
            min_user_words=syntheticDatasetConfig.min_user_words,
            max_user_words=syntheticDatasetConfig.max_user_words,
            min_assistant_words=syntheticDatasetConfig.min_assistant_words,
            max_assistant_words=syntheticDatasetConfig.max_assistant_words,
            min_quality_score=syntheticDatasetConfig.min_quality_score,
            temperature=syntheticDatasetConfig.temperature,
            top_p=syntheticDatasetConfig.top_p,
            min_p=syntheticDatasetConfig.min_p,
            repeat_penalty=syntheticDatasetConfig.repeat_penalty,
            retry_count=syntheticDatasetConfig.retry_count,
            enable_quality_judge=syntheticDatasetConfig.enable_quality_judge,
            judge_model_path=syntheticDatasetConfig.judge_model_path,
            export_final=syntheticDatasetConfig.export_final,
            cleanup_shards=syntheticDatasetConfig.cleanup_shards,

            multi_turn=syntheticDatasetConfig.multi_turn,
            min_turns=syntheticDatasetConfig.min_turns,
            max_turns=syntheticDatasetConfig.max_turns,

            topics=syntheticDatasetConfig.topics[syntheticDatasetConfig.language],
        )

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
