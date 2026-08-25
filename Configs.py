class SyntheticDatasetConfig:
    model_path = r"C:\models\Qwen3-8B-Q6_K.gguf"
    output_path = r"./dataset/synthetic.parquet"
    total_samples = 10000
    n_ctx = 4096
    n_threads = 8
    n_batch = 512
    seed = 42
    language = "fa"