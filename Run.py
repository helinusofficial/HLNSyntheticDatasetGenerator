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
        seed=SyntheticDatasetConfig.seed,
        language=SyntheticDatasetConfig.language
    )

    generator.run()