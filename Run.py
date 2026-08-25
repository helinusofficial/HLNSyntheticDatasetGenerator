from Configs import SyntheticDatasetConfig
from SyntheticDatasetGenerator import SyntheticDatasetGenerator

if __name__ == "__main__":
    generator = SyntheticDatasetGenerator(
        model_path=SyntheticDatasetConfig.MODEL_PATH,
        output_path=SyntheticDatasetConfig.OUTPUT_PATH,
        total_samples=SyntheticDatasetConfig.TOTAL_SAMPLES,
        n_ctx=SyntheticDatasetConfig.N_CTX,
        n_threads=SyntheticDatasetConfig.N_THREADS,
        n_batch=SyntheticDatasetConfig.N_BATCH,
        seed=SyntheticDatasetConfig.SEED
    )

    generator.run()