from Configs import SyntheticDatasetConfig
from SyntheticDatasetGenerator import SyntheticDatasetGenerator

if __name__ == "__main__":
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
        keep_shards=SyntheticDatasetConfig.keep_shards,
        export_final=SyntheticDatasetConfig.export_final,
        cleanup_shards=SyntheticDatasetConfig.cleanup_shards,
    )
    generator.run()
