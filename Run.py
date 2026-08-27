from datetime import datetime
from Utility.MyLogger import MyLogger
import subprocess
import time
from Configs import SyntheticDatasetConfig
from SyntheticDatasetGenerator import SyntheticDatasetGenerator
from Utility.TimeFormatHelper import TimeFormatHelper

def main():
    try:
        answer = input("Hibernate the computer after the program finishes? (y/n): ").strip().lower()
        shutdown_after = answer in ("y", "yes")

        start_time = time.time()
        start_datetime = datetime.now()

        logger_obj = MyLogger(log_dir=r"dh_log", log_file_name="deephit_log.txt")
        logger, path = logger_obj.setup()
        logger.info(
            f"Started: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"Auto hibernate: {'Disabled' if shutdown_after else 'Enabled'}"
        )

        generator = SyntheticDatasetGenerator(
            model_path=SyntheticDatasetConfig.model_path,
            output_path=SyntheticDatasetConfig.output_path,
            total_samples=SyntheticDatasetConfig.total_samples,
            n_ctx=SyntheticDatasetConfig.n_ctx,
            n_threads=SyntheticDatasetConfig.n_threads,
            n_batch=SyntheticDatasetConfig.n_batch,
            n_gpu_layers=SyntheticDatasetConfig.n_gpu_layers,
            seed=SyntheticDatasetConfig.seed,
            language=SyntheticDatasetConfig.language,
            max_tokens=SyntheticDatasetConfig.max_tokens,
            shard_size=SyntheticDatasetConfig.shard_size,
            checkpoint_interval=SyntheticDatasetConfig.checkpoint_interval,
            max_attempts_multiplier=SyntheticDatasetConfig.max_attempts_multiplier,
            min_user_words=SyntheticDatasetConfig.min_user_words,
            max_user_words=SyntheticDatasetConfig.max_user_words,
            min_assistant_words=SyntheticDatasetConfig.min_assistant_words,
            max_assistant_words=SyntheticDatasetConfig.max_assistant_words,
            min_quality_score=SyntheticDatasetConfig.min_quality_score,
            temperature=SyntheticDatasetConfig.temperature,
            top_p=SyntheticDatasetConfig.top_p,
            min_p=SyntheticDatasetConfig.min_p,
            repeat_penalty=SyntheticDatasetConfig.repeat_penalty,
            retry_count=SyntheticDatasetConfig.retry_count,
            enable_quality_judge=SyntheticDatasetConfig.enable_quality_judge,
            judge_model_path=SyntheticDatasetConfig.judge_model_path,
            export_final=SyntheticDatasetConfig.export_final,
            cleanup_shards=SyntheticDatasetConfig.cleanup_shards,

            multi_turn=SyntheticDatasetConfig.multi_turn,
            min_turns=SyntheticDatasetConfig.min_turns,
            max_turns=SyntheticDatasetConfig.max_turns,

            topics=SyntheticDatasetConfig.topics[SyntheticDatasetConfig.language],
        )

        generator.run()
        end_time = time.time()
        end_datetime = datetime.now()
        elapsed = end_time - start_time
        logger.info(f"Finished: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"Execution Time: {TimeFormatHelper.format_elapsed(elapsed)}\n")


        if shutdown_after:
           time.sleep(30)
           subprocess.run("shutdown /h", shell=True, check=True)

    except ValueError as error:
        print(f"ERROR: {error}")

if __name__ == "__main__":
    main()
