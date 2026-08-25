class SyntheticDatasetConfig:

    MODEL_PATH = r"C:\models\Qwen3-8B-Q6_K.gguf"
    OUTPUT_PATH = r"./dataset/synthetic.parquet"
    TOTAL_SAMPLES = 10000
    N_CTX = 4096
    N_THREADS = 8
    N_BATCH = 512
    SEED = 42